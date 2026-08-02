#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  build_imagenet100_mobilenetv2_11_sos.sh [OUT_DIR] [extra args forwarded to build_model11_sos.sh]

Default:
  OUT_DIR = ./build_posit11_mobilenetv2

What this script does:
  1) Ensure ImageNet100 MobileNetV2 QDQ ONNX exists (generate from f32 ONNX if missing).
  2) Build 11-style outputs via onnx-mlir/src/bash/build_model11_sos.sh.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd "${script_dir}" && pwd)"
# Portable: default assumes ImageNet100 and onnx-mlir are siblings; override with
# ONNX_MLIR_ROOT=/path/to/onnx-mlir if your layout differs.
onnx_mlir_root="${ONNX_MLIR_ROOT:-$(cd "${script_dir}/../onnx-mlir" && pwd)}"
build11="${onnx_mlir_root}/src/bash/build_model11_sos.sh"
quant_py="${root_dir}/quantize_imagenet100_qdq.py"

out_dir="${1:-${root_dir}/build_posit11_mobilenetv2}"
if [[ $# -gt 0 ]]; then
  shift
fi
forward_args=("$@")

nqdq_onnx="${root_dir}/model/imagenet100_mobilenetv2.onnx"
qdq_onnx="${root_dir}/model/imagenet100_mobilenetv2-int8-qdq.onnx"
txt_dir="${root_dir}/val_224_txt"
calib_limit="${CALIB_LIMIT:-256}"
calib_method="${CALIB_METHOD:-minmax}"
calib_act_type="${CALIB_ACT_TYPE:-int8}"
posit_formats="${POSIT_FORMATS:-p8e0,p8e1,p8e2,p16e0,p16e1,p16e2,p32e0,p32e1,p32e2}"
force_regen_qdq="${FORCE_REGEN_QDQ:-0}"
include_f32_baselines="${INCLUDE_F32_BASELINES:-1}"

if [[ ! -x "${build11}" ]]; then
  echo "ERROR: cannot find build script: ${build11}"
  exit 2
fi
if [[ ! -f "${quant_py}" ]]; then
  echo "ERROR: cannot find quant script: ${quant_py}"
  exit 2
fi
if [[ ! -f "${nqdq_onnx}" ]]; then
  echo "ERROR: missing f32 ONNX: ${nqdq_onnx}"
  exit 2
fi
if [[ ! -d "${txt_dir}" ]]; then
  echo "ERROR: missing calibration txt dir: ${txt_dir}"
  exit 2
fi

mkdir -p "${out_dir}"

if [[ "${force_regen_qdq}" == "1" || ! -f "${qdq_onnx}" ]]; then
  echo "[1/2] generate qdq onnx: ${qdq_onnx}"
  python3 "${quant_py}" \
    --model-in "${nqdq_onnx}" \
    --model-out "${qdq_onnx}" \
    --txt-dir "${txt_dir}" \
    --input-name input \
    --shape 1x3x224x224 \
    --limit "${calib_limit}" \
    --method "${calib_method}" \
    --activation-type "${calib_act_type}"
else
  echo "[1/2] qdq onnx exists, skip: ${qdq_onnx}"
fi

echo "[2/2] build 11 posit variants + f32 baselines"

has_skip_flag=0
for a in "${forward_args[@]}"; do
  if [[ "${a}" == "--skip-f32-baselines" ]]; then
    has_skip_flag=1
    break
  fi
done
if [[ "${include_f32_baselines}" != "1" && "${has_skip_flag}" != "1" ]]; then
  forward_args+=(--skip-f32-baselines)
fi

"${build11}" \
  --model-name imagenet100_mobilenetv2 \
  --qdq-onnx "${qdq_onnx}" \
  --nqdq-onnx "${nqdq_onnx}" \
  --out-dir "${out_dir}" \
  --posit-formats "${posit_formats}" \
  "${forward_args[@]}"

echo
echo "[done] outputs at: ${out_dir}"
