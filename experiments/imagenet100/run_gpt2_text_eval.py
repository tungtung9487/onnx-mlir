#!/usr/bin/env python3
import argparse
import ctypes
import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


I64 = ctypes.c_longlong
F32 = ctypes.c_float

ONNX_TYPE_FLOAT = 1
ONNX_TYPE_INT64 = 7

# Columns recorded per score run so f32/int8/posit formats can be compared
# side by side (analogous to the CNN dataset_parallel format_summary log).
RESULT_COLUMNS = [
    "ts_iso", "tag", "model", "format", "mode",
    "num_tokens", "num_predicted", "ppl", "avg_nll",
    "next_token_top1_acc", "elapsed_sec", "tokens_per_sec",
    "omp_threads", "max_tokens", "text_file",
]


def _guess_format_tag(model_path: str) -> str:
    """Short format label from the model filename, e.g. nqdq-p8e1 / qdq-f32 /
    onnx-int8. Falls back to the stem."""
    stem = Path(model_path).stem
    m = re.search(r"((?:nqdq|qdq)-(?:p\d+e\d+|f32))", stem)
    if m:
        return m.group(1)
    if stem.endswith("_int8") or "int8" in stem:
        return "onnx-int8"
    if model_path.endswith(".onnx"):
        return "onnx-f32"
    return stem


def append_result_row(results_log: str, row: dict) -> None:
    """Append one TSV row to results_log (write header if the file is new).
    Multiple runs (different formats/models) accumulate in one comparable file."""
    path = Path(results_log)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists() or path.stat().st_size == 0
    with open(path, "a", encoding="utf-8") as f:
        if new_file:
            f.write("\t".join(RESULT_COLUMNS) + "\n")
        f.write("\t".join(str(row.get(c, "")) for c in RESULT_COLUMNS) + "\n")


def bytes_to_unicode() -> dict[int, str]:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def get_pairs(word: tuple[str, ...]) -> set[tuple[str, str]]:
    pairs = set()
    prev = word[0]
    for ch in word[1:]:
        pairs.add((prev, ch))
        prev = ch
    return pairs


class GPT2BPETokenizer:
    # ASCII-friendly fallback pattern. For English prompts it matches GPT-2 well.
    _PAT = re.compile(
        r"'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?\d+| ?[^A-Za-z0-9\s]+|\s+(?!\S)|\s+"
    )

    def __init__(self, tokenizer_dir: str):
        tokenizer_path = Path(tokenizer_dir)
        with open(tokenizer_path / "vocab.json", "r", encoding="utf-8") as f:
            self.encoder = json.load(f)
        self.decoder = {v: k for k, v in self.encoder.items()}
        merges_path = tokenizer_path / "merges.txt"
        merges = []
        with open(merges_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                a, b = line.split()
                merges.append((a, b))
        self.bpe_ranks = {pair: i for i, pair in enumerate(merges)}
        self.cache: dict[str, str] = {}
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        with open(
            tokenizer_path / "tokenizer_config.json", "r", encoding="utf-8"
        ) as f:
            cfg = json.load(f)
        self.eos_token = cfg.get("eos_token", "<|endoftext|>")
        self.bos_token = cfg.get("bos_token", self.eos_token)
        self.bos_token_id = self.encoder[self.bos_token]
        self.eos_token_id = self.encoder[self.eos_token]

    def bpe(self, token: str) -> str:
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)
        if not pairs:
            return token
        while True:
            bigram = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break
                if (
                    word[i] == first
                    and i < len(word) - 1
                    and word[i + 1] == second
                ):
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = get_pairs(word)
        out = " ".join(word)
        self.cache[token] = out
        return out

    def encode(self, text: str) -> list[int]:
        bpe_ids = []
        for token in self._PAT.findall(text):
            encoded = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_ids.extend(self.encoder[p] for p in self.bpe(encoded).split(" "))
        return bpe_ids

    def decode(self, token_ids: list[int]) -> str:
        text = "".join(self.decoder[int(i)] for i in token_ids)
        data = bytearray(self.byte_decoder[c] for c in text)
        return data.decode("utf-8", errors="replace")


def log_softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64, copy=False)
    m = np.max(x)
    shifted = x - m
    return shifted - math.log(np.sum(np.exp(shifted)))


def make_shape(shape: tuple[int, ...]) -> ctypes.Array:
    return (I64 * len(shape))(*map(int, shape))


@dataclass
class StepOutput:
    logits: np.ndarray
    past: list[np.ndarray]


class ORTGPT2Runner:
    def __init__(self, model_path: str):
        import onnxruntime as ort

        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self.num_layers = 12

    def empty_past(self) -> list[np.ndarray]:
        return [
            np.zeros((1, 12, 0, 64), dtype=np.float32) for _ in range(self.num_layers * 2)
        ]

    def step(
        self,
        input_ids: np.ndarray,
        past: list[np.ndarray],
        attention_mask: np.ndarray,
        position_ids: np.ndarray,
    ) -> StepOutput:
        feeds = {
            "input_ids": np.ascontiguousarray(input_ids, dtype=np.int64),
            "attention_mask": np.ascontiguousarray(attention_mask, dtype=np.int64),
            "position_ids": np.ascontiguousarray(position_ids, dtype=np.int64),
        }
        for i, arr in enumerate(past):
            layer = i // 2
            kind = "key" if i % 2 == 0 else "value"
            feeds[f"past_key_values.{layer}.{kind}"] = np.ascontiguousarray(
                arr, dtype=np.float32
            )
        outputs = self.session.run(None, feeds)
        logits = outputs[0]
        present = [np.ascontiguousarray(arr) for arr in outputs[1:25]]
        return StepOutput(logits=logits, past=present)


class SharedLibGPT2Runner:
    def __init__(self, so_path: str):
        self.lib = ctypes.CDLL(so_path)
        if not hasattr(self.lib, "run_main_graph"):
            raise RuntimeError(
                f"{so_path} does not export run_main_graph; this text runner currently supports OMTensor-style GPT-2 .so only"
            )
        self._init_api()
        self.num_layers = 12

    def _init_api(self) -> None:
        lib = self.lib
        lib.omTensorCreate.restype = ctypes.c_void_p
        lib.omTensorCreate.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(I64),
            I64,
            ctypes.c_int,
        ]
        lib.omTensorDestroy.restype = None
        lib.omTensorDestroy.argtypes = [ctypes.c_void_p]
        lib.omTensorGetDataPtr.restype = ctypes.c_void_p
        lib.omTensorGetDataPtr.argtypes = [ctypes.c_void_p]
        lib.omTensorGetShape.restype = ctypes.POINTER(I64)
        lib.omTensorGetShape.argtypes = [ctypes.c_void_p]
        lib.omTensorGetRank.restype = I64
        lib.omTensorGetRank.argtypes = [ctypes.c_void_p]
        lib.omTensorGetNumElems.restype = I64
        lib.omTensorGetNumElems.argtypes = [ctypes.c_void_p]
        lib.omTensorListCreate.restype = ctypes.c_void_p
        lib.omTensorListCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p), I64]
        lib.omTensorListDestroy.restype = None
        lib.omTensorListDestroy.argtypes = [ctypes.c_void_p]
        lib.omTensorListGetOmtByIndex.restype = ctypes.c_void_p
        lib.omTensorListGetOmtByIndex.argtypes = [ctypes.c_void_p, I64]
        lib.run_main_graph.restype = ctypes.c_void_p
        lib.run_main_graph.argtypes = [ctypes.c_void_p]

    def empty_past(self) -> list[np.ndarray]:
        return [
            np.zeros((1, 12, 0, 64), dtype=np.float32) for _ in range(self.num_layers * 2)
        ]

    def _make_omtensor(
        self, arr: np.ndarray, onnx_type: int
    ) -> tuple[ctypes.c_void_p, tuple[np.ndarray, ctypes.Array]]:
        arr = np.ascontiguousarray(arr)
        shape = make_shape(tuple(arr.shape))
        omt = self.lib.omTensorCreate(
            ctypes.c_void_p(arr.ctypes.data), shape, arr.ndim, onnx_type
        )
        if not omt:
            raise RuntimeError("omTensorCreate returned NULL")
        return ctypes.c_void_p(omt), (arr, shape)

    def _omtensor_to_np_f32(self, omt: ctypes.c_void_p) -> np.ndarray:
        rank = int(self.lib.omTensorGetRank(omt))
        shape_ptr = self.lib.omTensorGetShape(omt)
        shape = tuple(int(shape_ptr[i]) for i in range(rank))
        num_elems = int(self.lib.omTensorGetNumElems(omt))
        data_ptr = self.lib.omTensorGetDataPtr(omt)
        arr = np.ctypeslib.as_array(
            ctypes.cast(data_ptr, ctypes.POINTER(F32)), shape=(num_elems,)
        )
        return np.array(arr, copy=True).reshape(shape)

    def step(
        self,
        input_ids: np.ndarray,
        past: list[np.ndarray],
        attention_mask: np.ndarray,
        position_ids: np.ndarray,
    ) -> StepOutput:
        input_tensors = []
        keepalive = []
        input_list = None
        output_list = None
        try:
            omt, keep = self._make_omtensor(input_ids, ONNX_TYPE_INT64)
            input_tensors.append(omt)
            keepalive.append(keep)
            for p in past:
                omt, keep = self._make_omtensor(p, ONNX_TYPE_FLOAT)
                input_tensors.append(omt)
                keepalive.append(keep)
            omt, keep = self._make_omtensor(attention_mask, ONNX_TYPE_INT64)
            input_tensors.append(omt)
            keepalive.append(keep)
            omt, keep = self._make_omtensor(position_ids, ONNX_TYPE_INT64)
            input_tensors.append(omt)
            keepalive.append(keep)

            omt_array = (ctypes.c_void_p * len(input_tensors))(*input_tensors)
            input_list = self.lib.omTensorListCreate(omt_array, len(input_tensors))
            if not input_list:
                raise RuntimeError("omTensorListCreate returned NULL")
            output_list = self.lib.run_main_graph(input_list)
            if not output_list:
                raise RuntimeError("run_main_graph returned NULL")

            logits = self._omtensor_to_np_f32(
                self.lib.omTensorListGetOmtByIndex(output_list, 0)
            )
            present = [
                self._omtensor_to_np_f32(
                    self.lib.omTensorListGetOmtByIndex(output_list, i)
                )
                for i in range(1, 25)
            ]
            return StepOutput(logits=logits, past=present)
        finally:
            if output_list:
                self.lib.omTensorListDestroy(output_list)
            if input_list:
                self.lib.omTensorListDestroy(input_list)


def build_runner(model_path: str):
    if model_path.endswith(".onnx"):
        return ORTGPT2Runner(model_path)
    if model_path.endswith(".so"):
        return SharedLibGPT2Runner(model_path)
    raise RuntimeError(f"Unsupported model type for {model_path}")


def run_generate(
    runner, tokenizer: GPT2BPETokenizer, prompt: str, max_new_tokens: int
) -> dict[str, object]:
    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        prompt_ids = [tokenizer.bos_token_id if hasattr(tokenizer, "bos_token_id") else tokenizer.eos_token_id]
    past = runner.empty_past()
    input_ids = np.array([prompt_ids], dtype=np.int64)
    total_len = len(prompt_ids)
    attention_mask = np.ones((1, total_len), dtype=np.int64)
    position_ids = np.arange(total_len, dtype=np.int64)[None, :]

    t0 = time.time()
    out = runner.step(input_ids, past, attention_mask, position_ids)
    past = out.past
    generated = []
    step_top1 = []
    nll = 0.0  # model's self-NLL over the tokens it generates (greedy)

    for _ in range(max_new_tokens):
        last = out.logits[0, -1]
        next_id = int(np.argmax(last))
        step_top1.append(next_id)
        if next_id == tokenizer.eos_token_id:
            break
        nll -= float(log_softmax(last)[next_id])
        generated.append(next_id)
        input_ids = np.array([[next_id]], dtype=np.int64)
        total_len += 1
        attention_mask = np.ones((1, total_len), dtype=np.int64)
        position_ids = np.array([[total_len - 1]], dtype=np.int64)
        out = runner.step(input_ids, past, attention_mask, position_ids)
        past = out.past

    elapsed = time.time() - t0
    n_gen = len(generated)
    avg_nll = (nll / n_gen) if n_gen else 0.0
    return {
        "prompt_ids": prompt_ids,
        "generated_ids": generated,
        "prompt_text": tokenizer.decode(prompt_ids),
        "generated_text": tokenizer.decode(generated),
        "full_text": tokenizer.decode(prompt_ids + generated),
        "first_top1": step_top1[0] if step_top1 else None,
        "num_generated": n_gen,
        "gen_avg_nll": avg_nll,
        "gen_ppl": math.exp(avg_nll) if n_gen else float("nan"),
        "elapsed_sec": elapsed,
        "tokens_per_sec": (n_gen / elapsed) if elapsed > 0 else 0.0,
    }


def score_token_ids(runner, token_ids: list[int],
                     progress: int = 0) -> dict[str, object]:
    """Teacher-forcing score of a single token sequence (len 2..1024).

    Returns RAW sums (nll_sum/exact/compared) so callers can aggregate across
    several independent windows (e.g. the parallel runner). Tokens stay
    sequential here because GPT-2 is autoregressive (each step needs the
    previous KV cache); only whole windows can be parallelized.
    """
    past = runner.empty_past()
    total_len = 1
    input_ids = np.array([[token_ids[0]]], dtype=np.int64)
    attention_mask = np.ones((1, total_len), dtype=np.int64)
    position_ids = np.array([[0]], dtype=np.int64)

    nll = 0.0
    exact = 0
    compared = 0
    first_pred = None
    pred_ids: list[int] = []  # model's greedy next-token prediction at each step
    to_predict = len(token_ids) - 1
    t0 = time.time()

    for target in token_ids[1:]:
        out = runner.step(input_ids, past, attention_mask, position_ids)
        past = out.past
        last = out.logits[0, -1]
        pred = int(np.argmax(last))
        pred_ids.append(pred)
        if first_pred is None:
            first_pred = pred
        exact += int(pred == target)
        compared += 1
        log_probs = log_softmax(last)
        nll -= float(log_probs[target])
        input_ids = np.array([[target]], dtype=np.int64)
        total_len += 1
        attention_mask = np.ones((1, total_len), dtype=np.int64)
        position_ids = np.array([[total_len - 1]], dtype=np.int64)

        # Progress tracking (like the CNN runner's --progress): print a running
        # ppl / top1 / throughput / ETA every `progress` predicted tokens.
        if progress and compared % progress == 0:
            el = time.time() - t0
            rate = compared / el if el > 0 else 0.0
            cur_ppl = math.exp(nll / compared)
            eta = (to_predict - compared) / rate if rate > 0 else 0.0
            print(
                f"[progress] {compared}/{to_predict} tok  "
                f"ppl={cur_ppl:.3f}  top1={exact / compared:.3f}  "
                f"{rate:.2f} tok/s  elapsed={el:.0f}s  ETA={eta:.0f}s",
                flush=True,
            )

    elapsed = time.time() - t0
    return {
        "nll_sum": nll,
        "exact": exact,
        "compared": compared,
        "first_pred": first_pred,
        "pred_ids": pred_ids,
        "elapsed_sec": elapsed,
        "tokens_per_sec": (compared / elapsed) if elapsed > 0 else 0.0,
    }


def run_score(runner, tokenizer: GPT2BPETokenizer, text: str,
              max_tokens: int = 0, progress: int = 0) -> dict[str, object]:
    token_ids = tokenizer.encode(text)
    # GPT-2 max position is 1024; run_score grows position_ids to total_len-1,
    # so cap tokens at <=1024. Also caps cost for slow posit .so runs.
    cap = 1024 if max_tokens <= 0 else min(max_tokens, 1024)
    if len(token_ids) > cap:
        token_ids = token_ids[:cap]
    if len(token_ids) < 2:
        raise RuntimeError("Need at least 2 tokens to score perplexity")

    s = score_token_ids(runner, token_ids, progress=progress)
    avg_nll = s["nll_sum"] / s["compared"]
    return {
        "num_tokens": len(token_ids),
        "num_predicted": s["compared"],
        "avg_nll": avg_nll,
        "ppl": math.exp(avg_nll),
        "next_token_top1_acc": s["exact"] / s["compared"],
        "first_pred_token_id": s["first_pred"],
        "text_preview": tokenizer.decode(token_ids[: min(48, len(token_ids))]),
        # what GPT-2 predicts as the continuation (greedy next-token at each
        # position, decoded back to text).
        "predicted_text": tokenizer.decode(s["pred_ids"]),
        "elapsed_sec": s["elapsed_sec"],
        "tokens_per_sec": s["tokens_per_sec"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to .onnx or .so model")
    ap.add_argument(
        "--tokenizer-dir",
        default="/home/lai/onnx_mlir/ImageNet100/model/gpt2_onnx_community",
        help="Directory containing GPT-2 tokenizer files",
    )
    ap.add_argument(
        "--mode",
        choices=["generate", "score"],
        default="generate",
        help="generate: prompt -> text, score: text -> ppl/top1 agreement",
    )
    ap.add_argument("--prompt", default="Hello, my name is")
    ap.add_argument("--text", default="")
    ap.add_argument("--text-file", default="",
                    help="score mode: read corpus from this file (e.g. WikiText-2 test)")
    ap.add_argument("--max-tokens", type=int, default=1000,
                    help="score mode: cap #tokens (<=1024; GPT-2 context limit)")
    ap.add_argument("--progress", type=int, default=100,
                    help="score mode: print running ppl/top1/throughput every N "
                         "predicted tokens (0 = off)")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--results-log", default="",
                    help="score mode: append a TSV result row here. Default: "
                         "<model dir>/gpt2_text_eval_results.tsv (set to 'off' "
                         "to disable).")
    ap.add_argument("--tag", default="",
                    help="free-form label stored in the results row (e.g. run "
                         "name); defaults to the format tag.")
    args = ap.parse_args()

    tokenizer = GPT2BPETokenizer(args.tokenizer_dir)
    runner = build_runner(args.model)

    if args.mode == "generate":
        # Seed: use --text-file's first 48 tokens if given, else --prompt.
        if args.text_file:
            with open(args.text_file, "r", encoding="utf-8") as _f:
                seed_ids = tokenizer.encode(_f.read())[:48]
            prompt = tokenizer.decode(seed_ids) if seed_ids else args.prompt
        else:
            prompt = args.prompt
        result = run_generate(runner, tokenizer, prompt, args.max_new_tokens)
        gsummary = [
            ("mode", "generate"),
            ("model", os.path.basename(args.model)),
            ("format", _guess_format_tag(args.model)),
            ("max_new_tokens", args.max_new_tokens),
            ("num_generated", result["num_generated"]),
            # self-perplexity: model's confidence in its own greedy output
            # (NOT teacher-forced ppl on ground-truth text).
            ("gen_ppl", f"{result['gen_ppl']:.6f}"),
            ("gen_avg_nll", f"{result['gen_avg_nll']:.6f}"),
            ("elapsed_sec", f"{result['elapsed_sec']:.2f}"),
            ("tokens_per_sec", f"{result['tokens_per_sec']:.3f}"),
            ("first_top1_token_id", result["first_top1"]),
            ("omp_threads", os.environ.get("POSIT_OMP_THREADS", "")),
            ("seed_from", os.path.basename(args.text_file) if args.text_file
                          else "--prompt"),
        ]
        for k, v in gsummary:
            print(f"{k}={v}")
        print("\n# prompt_text (seed)")
        print(result["prompt_text"])
        print("\n# generated_text (GPT-2 free continuation)")
        print(result["generated_text"])
        if args.results_log.lower() != "off":
            model_dir = os.path.dirname(os.path.abspath(args.model))
            summary_path = os.path.join(
                model_dir, f"{Path(args.model).stem}.generate_summary.log")
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write("# gpt2 generate summary\n")
                for k, v in gsummary:
                    f.write(f"{k}={v}\n")
                f.write("\n# prompt_text (seed)\n")
                f.write(result["prompt_text"] + "\n")
                f.write("\n# generated_text (GPT-2 free continuation)\n")
                f.write(result["generated_text"] + "\n")
            print(f"summary_file={summary_path}")
        return 0

    if args.text_file:
        with open(args.text_file, "r", encoding="utf-8") as _f:
            text = _f.read()
    else:
        text = args.text if args.text else args.prompt
    result = run_score(runner, tokenizer, text, max_tokens=args.max_tokens,
                       progress=args.progress)
    fmt = _guess_format_tag(args.model)
    # One metric per line, key=value (build_posit_config.log style).
    summary = [
        ("mode", "score"),
        ("model", os.path.basename(args.model)),
        ("format", fmt),
        ("num_tokens", result["num_tokens"]),
        ("num_predicted", result["num_predicted"]),
        ("avg_nll", f"{result['avg_nll']:.6f}"),
        ("ppl", f"{result['ppl']:.6f}"),
        ("next_token_top1_acc", f"{result['next_token_top1_acc']:.6f}"),
        ("first_pred_token_id", result["first_pred_token_id"]),
        ("elapsed_sec", f"{result['elapsed_sec']:.2f}"),
        ("tokens_per_sec", f"{result['tokens_per_sec']:.3f}"),
        ("omp_threads", os.environ.get("POSIT_OMP_THREADS", "")),
        ("max_tokens", args.max_tokens),
        ("text_file", os.path.basename(args.text_file) if args.text_file else ""),
    ]
    for k, v in summary:
        print(f"{k}={v}")
    # The passage GPT-2 predicts as the continuation (greedy next-token, decoded).
    print("\n# input_text_preview (first 48 tokens)")
    print(result["text_preview"])
    print("\n# predicted_text (GPT-2 greedy next-token continuation)")
    print(result["predicted_text"])

    # Record the run (ppl / top1 / timing) like the CNN runner does.
    if args.results_log.lower() != "off":
        model_dir = os.path.dirname(os.path.abspath(args.model))
        # (a) per-run vertical key=value summary file, one per format
        #     (exactly build_posit_config.log style; overwritten each run).
        summary_path = os.path.join(
            model_dir, f"{Path(args.model).stem}.score_summary.log")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("# gpt2 score summary\n")
            for k, v in summary:
                f.write(f"{k}={v}\n")
            f.write("\n# input_text_preview (first 48 tokens)\n")
            f.write(result["text_preview"] + "\n")
            f.write("\n# predicted_text (GPT-2 greedy next-token continuation)\n")
            f.write(result["predicted_text"] + "\n")
        print(f"summary_file={summary_path}")
        # (b) shared TSV so multiple formats (f32/int8/p8/p16/p32) stay comparable.
        results_log = args.results_log or os.path.join(
            model_dir, "gpt2_text_eval_results.tsv")
        append_result_row(results_log, {
            "ts_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "tag": args.tag or fmt,
            "model": os.path.basename(args.model),
            "format": fmt,
            "mode": "score",
            "num_tokens": result["num_tokens"],
            "num_predicted": result["num_predicted"],
            "ppl": f"{result['ppl']:.6f}",
            "avg_nll": f"{result['avg_nll']:.6f}",
            "next_token_top1_acc": f"{result['next_token_top1_acc']:.6f}",
            "elapsed_sec": f"{result['elapsed_sec']:.2f}",
            "tokens_per_sec": f"{result['tokens_per_sec']:.3f}",
            "omp_threads": os.environ.get("POSIT_OMP_THREADS", ""),
            "max_tokens": args.max_tokens,
            "text_file": os.path.basename(args.text_file) if args.text_file else "",
        })
        print(f"results_log={results_log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
