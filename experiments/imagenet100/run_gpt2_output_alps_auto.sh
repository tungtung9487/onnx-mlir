#!/usr/bin/env bash
# One-shot runtime output/activation-ALPS for GPT-2, analogous to the CNN
# time_model11_dataset_parallel.sh --output-alps-auto on flow:
#   (1) collect  : run the OFFLINE-mode .so over some tokens, dumping per-op
#                  activation samples to a collect.csv (POSIT_RUNTIME_OUTPUT_ALPS_
#                  COLLECT_FILE). Unlike CNN (1 process per image), GPT-2 fills the
#                  sample buckets across token forwards inside ONE process.
#   (2) calibrate: output_alps_calibrate collect.csv -> params.csv (per-op theta/
#                  gamma via the same SQNR search the CNN path uses).
#   (3) apply    : re-run BOTH score and generate with the params applied
#                  (POSIT_RUNTIME_OUTPUT_ALPS_FILE=params.csv), plus a no-ALPS
#                  baseline for before/after comparison.
#
# REQUIREMENT: the .so MUST be built with `--runtime-output-alps offline` (the
# collect hook only fires in offline/full mode). All 0716 .so are `off`; build an
# offline one first, e.g. (p8e1, with GP so the meta path is active):
#
#   ONNX_MLIR_POSIT_CONST_ALPS=1 POSIT_CONST_ALPS_JOBS=25 \
#     ONNX_MLIR_POSIT_CONST_ALPS_THETA_MIN=0.006 ONNX_MLIR_POSIT_CONST_ALPS_THETA_MAX=120 \
#     ONNX_MLIR_POSIT_CONST_ALPS_THETA_STEPS=150 ONNX_MLIR_POSIT_CONST_ALPS_GAMMA_TARGET=1.0 \
#     ONNX_MLIR_POSIT_CONST_ALPS_GAMMA_PERCENTILE=0.99 ONNX_MLIR_POSIT_CONST_ALPS_MIN_GAIN=0.001 \
#     ONNX_MLIR_POSIT_CONST_ALPS_MAX_SAMPLES=0 \
#     POSIT_GP_EXPERIMENTAL_FORMATS=p8e1 POSIT_GP_RS_VALUES_P8=7,6,5,4,3 POSIT_GP_SC_VALUES_P8=3,2,1,0,-1,-2,-3 \
#     POSIT_FORMATS=p8e1 \
#     bash /home/lai/onnx_mlir/onnx-mlir/src/bash/build_gpt2_hf_11_sos.sh \
#       build_gpt2_nqdq_p8e1_offline --posit-source nqdq --runtime-format-scope single \
#       --runtime-qalign-mode alps-only --runtime-mixed-accum off --runtime-output-alps offline
#
# Usage:
#   bash run_gpt2_output_alps_auto.sh \
#     --so build_gpt2_nqdq_p8e1_offline/gpt2-hf-debug-nqdq-p8e1.so \
#     --text-file eval_text/wikitext2_test.txt \
#     --collect-tokens 256 --eval-tokens 512 --gen-tokens 40 --omp 24
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
py="${root_dir}/gpt2/bin/python"
runner="${root_dir}/run_gpt2_text_eval.py"

so=""
text_file="${root_dir}/eval_text/wikitext2_test.txt"
collect_tokens=256      # phase-1: tokens used to collect activation samples
eval_tokens=512         # phase-3: score-mode ppl eval length
gen_tokens=40           # phase-3: generate-mode new tokens
max_per_key=4096        # POSIT_RUNTIME_OUTPUT_ALPS_COLLECT_MAX_PER_KEY
per_call=512            # POSIT_RUNTIME_OUTPUT_ALPS_COLLECT_PER_CALL
omp=24
baseline="on"
prompt=""               # generate seed; empty => seed from --text-file
mode="both"             # which eval(s) to run in phase 3: score | generate | both

while [[ $# -gt 0 ]]; do
  case "$1" in
    --so) so="$2"; shift 2;;
    --text-file) text_file="$2"; shift 2;;
    --collect-tokens) collect_tokens="$2"; shift 2;;
    --eval-tokens) eval_tokens="$2"; shift 2;;
    --gen-tokens) gen_tokens="$2"; shift 2;;
    --max-per-key) max_per_key="$2"; shift 2;;
    --per-call) per_call="$2"; shift 2;;
    --omp) omp="$2"; shift 2;;
    --baseline) baseline="$2"; shift 2;;
    --prompt) prompt="$2"; shift 2;;
    --mode) mode="$2"; shift 2;;
    *) echo "unknown arg: $1"; exit 2;;
  esac
done
case "${mode}" in score|generate|both) ;; *) echo "ERROR: --mode must be score|generate|both"; exit 2;; esac

[[ -n "${so}" && -f "${so}" ]] || { echo "ERROR: --so <offline .so> required and must exist"; exit 2; }
[[ -f "${text_file}" ]] || { echo "ERROR: text-file not found: ${text_file}"; exit 2; }

so_dir="$(cd "$(dirname "${so}")" && pwd)"
so_stem="$(basename "${so}" .so)"
calib="${so_dir}/output_alps_calibrate"
[[ -x "${calib}" ]] || { echo "ERROR: calibrate tool missing: ${calib}"; exit 2; }

# sanity: warn if the .so was not built in offline mode (collect hook won't fire)
cfg="${so_dir}/build_posit_config.log"
if [[ -f "${cfg}" ]] && ! grep -q "runtime_output_alps=offline\|runtime_output_alps=full" "${cfg}"; then
  echo "WARN: ${cfg} says runtime_output_alps is NOT offline/full — collection may be empty."
fi

work="${so_dir}/runtime_output_alps_auto/${so_stem}"
mkdir -p "${work}"
collect_csv="${work}/collect_tokens_${collect_tokens}.csv"
params_csv="${work}/params_tokens_${collect_tokens}.csv"

echo "=================================================================="
echo " GPT-2 output-ALPS auto:  so=${so_stem}"
echo " collect_tokens=${collect_tokens}  eval_tokens=${eval_tokens}  gen_tokens=${gen_tokens}"
echo " max_per_key=${max_per_key}  per_call=${per_call}  omp=${omp}"
echo "=================================================================="

# ---- Phase 1: collect ------------------------------------------------------
echo "[1/3] collect activation samples -> ${collect_csv}"
env POSIT_OMP_THREADS="${omp}" \
    POSIT_RUNTIME_OUTPUT_ALPS_COLLECT_FILE="${collect_csv}" \
    POSIT_RUNTIME_OUTPUT_ALPS_COLLECT_MAX_PER_KEY="${max_per_key}" \
    POSIT_RUNTIME_OUTPUT_ALPS_COLLECT_PER_CALL="${per_call}" \
  "${py}" "${runner}" --model "${so}" --mode score \
    --text-file "${text_file}" --max-tokens "${collect_tokens}" --progress 128 \
    --results-log off > "${work}/collect_run.log" 2>&1 || true
if [[ ! -s "${collect_csv}" ]] || [[ "$(wc -l < "${collect_csv}")" -le 1 ]]; then
  echo "ERROR: collect.csv empty — is the .so built with --runtime-output-alps offline?"
  echo "       see ${work}/collect_run.log"
  exit 4
fi
echo "      collected rows: $(( $(wc -l < "${collect_csv}") - 1 ))"

# ---- Phase 2: calibrate ----------------------------------------------------
echo "[2/3] calibrate -> ${params_csv}"
"${calib}" "${collect_csv}" "${params_csv}" > "${work}/calibrate.log" 2>&1
[[ -s "${params_csv}" ]] || { echo "ERROR: params.csv not produced (see ${work}/calibrate.log)"; exit 4; }
echo "      params rows: $(( $(wc -l < "${params_csv}") - 1 ))"

# ---- Phase 3: apply + eval (score & generate), plus baseline ---------------
run_score() {   # $1=tag  $2=extra-env (POSIT_RUNTIME_OUTPUT_ALPS_FILE=... or empty)
  local tag="$1"; shift
  echo "----- score [${tag}] -----"
  env POSIT_OMP_THREADS="${omp}" "$@" \
    "${py}" "${runner}" --model "${so}" --mode score \
      --text-file "${text_file}" --max-tokens "${eval_tokens}" --progress 128 \
      --tag "${tag}" 2>&1 | grep -E "^(ppl|next_token_top1_acc|avg_nll|elapsed_sec|tokens_per_sec)="
}
run_generate() {
  local tag="$1"; shift
  echo "----- generate [${tag}] -----"
  local seedargs=(--text-file "${text_file}")
  [[ -n "${prompt}" ]] && seedargs=(--prompt "${prompt}")
  env POSIT_OMP_THREADS="${omp}" "$@" \
    "${py}" "${runner}" --model "${so}" --mode generate \
      "${seedargs[@]}" --max-new-tokens "${gen_tokens}" --tag "${tag}" 2>&1 \
    | grep -E "^(gen_ppl|tokens_per_sec|elapsed_sec)=|^# generated_text" -A1 | grep -vE "^--$"
}

if [[ "${baseline}" == "on" ]]; then
  if [[ "${mode}" != "generate" ]]; then run_score    "no-alps"; fi
  if [[ "${mode}" != "score" ]];    then run_generate "no-alps"; fi
fi
if [[ "${mode}" != "generate" ]]; then
  run_score    "alps-offline" POSIT_RUNTIME_OUTPUT_ALPS_FILE="${params_csv}"
fi
if [[ "${mode}" != "score" ]]; then
  run_generate "alps-offline" POSIT_RUNTIME_OUTPUT_ALPS_FILE="${params_csv}"
fi

echo "=================================================================="
echo " done. params: ${params_csv}"
echo " results TSV : ${so_dir}/gpt2_text_eval_results.tsv (tags no-alps vs alps-offline)"
echo "=================================================================="
