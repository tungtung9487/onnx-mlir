#!/usr/bin/env bash
# End-to-end reproduce: fresh machine -> MobileNetV2 posit .so with weight-ALPS
# -> eval with offline output/activation-ALPS (small quick-run settings).
#
# RUN:   bash reproduce_alps.sh          (NOT `bash *.md` — a .md is not a script)
# Re-runnable: each phase is skipped if its output already exists.
#
# Tunables via env (defaults reproduce the original run: 30 epochs, int8 too):
#   WS=<workspace root>   EPOCHS=30   EVAL_LIMIT=500   COLLECT_LIMIT=50
#   VAL_LIMIT=500 (int8 calibration imgs)   SOURCE=both (nqdq+int8 | nqdq | qdq)
#   THETA_MIN=0.01  THETA_MAX=10  THETA_STEPS=60  FORMAT=p8e2 (CSV ok: p8e0,p8e1,p8e2)
set -euo pipefail

WS="${WS:-$HOME/test_rebuild_project}"
BRANCH="tungtung9487/posit-work-20260329"
FORK_URL="https://github.com/tungtung9487/onnx-mlir.git"
LLVM_COMMIT="113f01aa82d055410f22a9d03b3468fa68600589"
EPOCHS="${EPOCHS:-30}"          # original MobileNetV2 training used 30 epochs
EVAL_LIMIT="${EVAL_LIMIT:-500}"
COLLECT_LIMIT="${COLLECT_LIMIT:-50}"
VAL_LIMIT="${VAL_LIMIT:-500}"   # int8 QDQ calibration images (original used 500)
SOURCE="${SOURCE:-both}"        # both = build nqdq(ALPS) + qdq(int8); or nqdq / qdq
THETA_MIN="${THETA_MIN:-0.01}"; THETA_MAX="${THETA_MAX:-10}"; THETA_STEPS="${THETA_STEPS:-60}"
FORMAT="${FORMAT:-p8e2}"
NP="$(nproc)"
OUT="build_mbv2_alps_quick"

om="$WS/onnx-mlir"; llvm="$WS/llvm-project"; in100="$WS/ImageNet100"
py="$in100/venv/bin/python"; pip="$in100/venv/bin/pip"
phase(){ echo; echo "==================== $* ===================="; }

mkdir -p "$WS"

# ---- 1. get code (download from GitHub) -----------------------------------
phase "1. clone onnx-mlir fork + LLVM"
[ -d "$om/.git" ]   || git clone -b "$BRANCH" "$FORK_URL" "$om"
# third_party/* (onnx, pybind11, rapidcheck, stablehlo, benchmark) are submodules
# that onnx-mlir needs to configure/build; init them (idempotent).
git -C "$om" submodule update --init --recursive
[ -d "$llvm/.git" ] || git clone https://github.com/llvm/llvm-project.git "$llvm"
git -C "$llvm" rev-parse --verify -q "$LLVM_COMMIT^{commit}" >/dev/null 2>&1 || git -C "$llvm" fetch --all
git -C "$llvm" checkout -q "$LLVM_COMMIT"

# ---- 2. build LLVM/MLIR (slow; skipped if already built) ------------------
# Memory-safe: LLVM linking uses ~2-4GB per job; a full -j(nproc) build on a
# RAM-limited box swap-thrashes and FREEZES the machine. Cap concurrent compiles
# by RAM (~1 per 2GB) and concurrent LINKS to 2, and use lld if available.
phase "2. build LLVM/MLIR"
if [ ! -x "$llvm/build/bin/mlir-opt" ]; then
  RAMGB="$(free -g | awk 'NR==2{print $2}')"
  CJ=$(( RAMGB / 2 )); [ "$CJ" -lt 1 ] && CJ=1; [ "$CJ" -gt "$NP" ] && CJ="$NP"
  llvm_extra=(-DLLVM_PARALLEL_COMPILE_JOBS="$CJ" -DLLVM_PARALLEL_LINK_JOBS=2)
  command -v ld.lld >/dev/null 2>&1 && llvm_extra+=(-DLLVM_ENABLE_LLD=ON)
  echo "  RAM=${RAMGB}GB nproc=${NP} -> compile_jobs=${CJ}, link_jobs=2$(command -v ld.lld >/dev/null 2>&1 && echo ' (+lld)')"
  mkdir -p "$llvm/build"
  cmake -G Ninja -S "$llvm/llvm" -B "$llvm/build" \
    -DLLVM_ENABLE_PROJECTS=mlir -DLLVM_TARGETS_TO_BUILD=host \
    -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON -DLLVM_ENABLE_RTTI=ON \
    "${llvm_extra[@]}"
  cmake --build "$llvm/build" -- -j"$NP"
fi

# ---- 3. posit deps + build onnx-mlir --------------------------------------
phase "3. posit deps + onnx-mlir"
# Check the actual softposit LIBRARY (not just the universal dir): a partial
# earlier run can leave universal/ present but libsoftposit.a missing, which
# makes onnx-mlir's cmake fail with "softposit library not found".
[ -f "$om/src/.deps/softposit-px1/libsoftposit.a" ] || bash "$om/src/bash/install_posit_deps.sh"
if [ ! -x "$om/build/Release/bin/onnx-mlir-opt" ]; then
  # A configure that ran before the posit deps were built caches SOFTPOSIT_LIBRARY
  # empty (find_library result is sticky); clear the stale cache so it re-detects,
  # and point SOFTPOSIT_LIBRARY at the lib explicitly to bypass find_library.
  rm -f "$om/build/CMakeCache.txt"
  mkdir -p "$om/build"
  cmake -G Ninja -S "$om" -B "$om/build" \
    -DMLIR_DIR="$llvm/build/lib/cmake/mlir" \
    -DLLVM_DIR="$llvm/build/lib/cmake/llvm" \
    -DCMAKE_BUILD_TYPE=Release \
    -DSOFTPOSIT_LIBRARY="$om/src/.deps/softposit-px1/libsoftposit.a"
  cmake --build "$om/build" --target onnx-mlir-opt onnx-mlir -- -j4
fi

# ---- 4. python env + scripts ----------------------------------------------
phase "4. python venv + scripts"
mkdir -p "$in100"
cp "$om/experiments/imagenet100/"*.py "$in100/" 2>/dev/null || true
cp "$om/experiments/imagenet100/build_imagenet100_"*.sh "$in100/" 2>/dev/null || true
if [ ! -x "$py" ]; then
  python3 -m venv "$in100/venv"; "$pip" install -U pip
  "$pip" install torch torchvision timm datasets onnx onnxruntime numpy pillow
fi

# ---- 5. dataset + f32 model -----------------------------------------------
phase "5. dataset + train f32 + export ONNX"
cd "$in100"
[ -d "$in100/imagenet100_hf/validation" ] || "$py" download_dataset.py
if [ ! -f "$in100/model/imagenet100_mobilenetv2.onnx" ]; then
  [ -f "$in100/imagenet100_mobilenetv2_out/best.pth" ] || \
    "$py" train_imagenet100_mobilenetv2.py \
      --train-dir imagenet100_hf/train --val-dir imagenet100_hf/validation \
      --epochs "$EPOCHS" --output-dir imagenet100_mobilenetv2_out
  "$py" export_imagenet100_mobilenetv2_onnx.py \
    --ckpt imagenet100_mobilenetv2_out/best.pth \
    --class-map imagenet100_mobilenetv2_out/class_map.json \
    --onnx-out model/imagenet100_mobilenetv2.onnx
fi

# ---- 5b. int8 QDQ calibration set (needed when SOURCE builds qdq) ----------
if [[ "$SOURCE" != "nqdq" ]]; then
  phase "5b. int8 calibration set val_224_txt ($VAL_LIMIT imgs)"
  if [ ! -d "$in100/val_224_txt" ] || [ -z "$(ls -A "$in100/val_224_txt" 2>/dev/null)" ]; then
    "$py" gen_val_224_txt.py --data-root imagenet100_hf/validation \
      --out-dir val_224_txt --limit "$VAL_LIMIT"
  fi
fi

# ---- 6. build posit .so via the wrapper -----------------------------------
# SOURCE=both -> builds nqdq (weight-ALPS) AND qdq (int8->posit); the wrapper
# auto-generates the int8 QDQ onnx from val_224_txt when it is missing.
phase "6. build MobileNetV2 posit .so ($FORMAT, source=$SOURCE, theta $THETA_MIN~$THETA_MAX)"
if [ ! -f "$in100/$OUT/build_posit_config.log" ]; then
  env ONNX_MLIR_POSIT_CONST_ALPS=1 POSIT_CONST_ALPS_JOBS="$NP" \
    ONNX_MLIR_POSIT_CONST_ALPS_THETA_MIN="$THETA_MIN" ONNX_MLIR_POSIT_CONST_ALPS_THETA_MAX="$THETA_MAX" \
    ONNX_MLIR_POSIT_CONST_ALPS_THETA_STEPS="$THETA_STEPS" ONNX_MLIR_POSIT_CONST_ALPS_GAMMA_TARGET=1.0 \
    ONNX_MLIR_POSIT_CONST_ALPS_GAMMA_PERCENTILE=0.99 ONNX_MLIR_POSIT_CONST_ALPS_MIN_GAIN=0.001 \
    ONNX_MLIR_POSIT_CONST_ALPS_MAX_SAMPLES=1024 \
    POSIT_GP_EXPERIMENTAL_FORMATS="$FORMAT" POSIT_GP_RS_VALUES_P8=7,5,3 POSIT_GP_SC_VALUES_P8=3,0,-3 \
    POSIT_FORMATS="$FORMAT" INCLUDE_F32_BASELINES=1 ONNX_MLIR_ROOT="$om" \
    bash "$in100/build_imagenet100_mobilenetv2_11_sos.sh" "$in100/$OUT" \
      --posit-source "$SOURCE" --posit-formats "$FORMAT" --runtime-format-scope single \
      --runtime-qalign-mode alps-only --runtime-mixed-accum off --runtime-output-alps offline
fi

# ---- 7. eval + offline output/activation-ALPS -----------------------------
phase "7. eval ($EVAL_LIMIT imgs) + offline output-ALPS (collect $COLLECT_LIMIT)"
# suffixes cover every built source x format (e.g. nqdq-p8e2,qdq-p8e2)
suffixes=""
for f in ${FORMAT//,/ }; do
  case "$SOURCE" in
    nqdq) suffixes+="nqdq-$f,";;
    qdq)  suffixes+="qdq-$f,";;
    both) suffixes+="nqdq-$f,qdq-$f,";;
  esac
done
suffixes="${suffixes%,}"
bash "$om/src/bash/time_model11_dataset_parallel.sh" \
  --model-name imagenet100_mobilenetv2 --out-dir "$in100/$OUT" \
  --image-dir "$in100/imagenet100_hf/validation" \
  --image-preprocess-script "$in100/preprocess_imagenet100_tensor.py" \
  --shape 1x3x224x224 --suffixes "$suffixes" \
  --baseline none --qalign-auto off --qalign-mode off \
  --jobs "$NP" --limit "$EVAL_LIMIT" --warmup 0 --iters 1 --no-benchmark --quire off \
  --output-alps-auto on --output-alps-formats "$FORMAT" \
  --output-alps-collect-limit "$COLLECT_LIMIT" --output-alps-collect-jobs "$NP" \
  --record-preds on --progress 100 --task-progress on

phase "DONE"
echo "results dir : $in100/$OUT"
echo "ALPS params : $in100/$OUT/runtime_output_alps_auto/imagenet100_mobilenetv2/nqdq-$FORMAT/params_upto_${COLLECT_LIMIT}.csv"
echo "build config: $in100/$OUT/build_posit_config.log"
