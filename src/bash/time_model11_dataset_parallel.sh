#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'USAGE_EOF'
Usage:
  time_model11_dataset_parallel.sh --model-name NAME --out-dir DIR \
    [--txt-dir DIR | --image-dir DIR] \
    [--shape NxCxHxW|N,C,H,W] [--limit N] [--jobs N] [--progress N] \
    [--warmup N] [--iters N] [--timeout-sec N] [--label-map FILE] \
    [--suffixes CSV] [--baseline none|nqdq-f32|qdq-f32] \
    [--qalign-mode off|alpha|alps] [--qalign-force-compand on|off] \
    [--qalign-auto on|off] [--qalign-formats CSV] [--qalign-csv-dir DIR] \
    [--qalign-txt-dir DIR] [--qalign-limit N] [--qalign-batch-size N] \
    [--qalign-format-jobs N] \
    [--output-alps-auto on|off] [--output-alps-formats CSV] \
    [--output-alps-collect-limit N] [--output-alps-collect-jobs N] \
    [--output-alps-params-dir DIR] \
    [--qalign-new-bucket-ratio F] [--qalign-mae-delta F] \
    [--qalign-min-rounds N] [--qalign-stable-rounds N] \
    [--qalign-collect-per-call N] [--qalign-collect-max-per-bucket N] \
    [--no-benchmark|--with-benchmark] [--quire on|off] \
    [--task-progress on|off] [--log-file FILE] \
    [--task-log-a FILE] [--format-log-b FILE] \
    [--prediction-log-c FILE] [--record-preds on|off] \
    [--record-logits on|off] [--logits-dir DIR] \
    [--image-exts CSV] [--image-preprocess-script FILE]

Purpose:
  Run available .so variants with dynamic multi-process scheduling.
  When a worker becomes idle, it immediately picks the next pending task.
  Posit formats are prioritized from higher precision to lower precision.

Output:
  - Per-so average latency (us)
  - Per-so metrics vs baseline (if baseline != none)
  - Optional GT Top1/Top5 (if label-map provided)
  - Total wall time

Notes:
  - Default is --no-benchmark (single infer per so)
  - Default suffix order is automatically reordered by priority:
      qdq-p32e2 > qdq-p32e1 > qdq-p32e0 > qdq-p16e2 > ... > qdq-f32 > nqdq-f32
  - Default is --task-progress on:
      {task 37/1100} format=p32e1 sample=4/100
  - With --task-progress off:
      {done p32e2}
  - qalign auto is ON by default for p8e0,p8e1,p8e2:
      it runs pre-collection/calibration until convergence, then applies
      per-format qalign CSV automatically during dataset run.
  - qalign mode default is ALPS (--qalign-mode alps).
USAGE_EOF
}

model_name=""
out_dir=""
txt_dir=""
image_dir=""
image_exts="jpg,jpeg,png,bmp"
image_preprocess_script=""
shape="1x1x28x28"
limit=0
jobs=1
progress=0
warmup=0
iters=1
timeout_sec=0
label_map=""
suffixes_csv=""
baseline_mode="nqdq-f32"
qalign_mode="alps"
qalign_force_compand_override=""
if [[ -n "${QALIGN_FORCE_COMPAND:-}" ]]; then
  # Respect an explicitly exported environment override unless the user passes
  # --qalign-force-compand on the command line.
  qalign_force_compand_override="$(printf '%s' "${QALIGN_FORCE_COMPAND}" | tr '[:upper:]' '[:lower:]')"
fi
no_benchmark=1
quire_mode="on"
task_progress_mode="on"
log_file=""
task_log_a=""
format_log_b=""
prediction_log_c=""
qalign_auto="on"
qalign_formats_csv="p8e0,p8e1,p8e2"
qalign_csv_dir=""
qalign_txt_dir=""
qalign_limit=1000
qalign_batch_size=25
qalign_format_jobs=1
output_alps_auto="off"
output_alps_formats_csv="p8e0,p8e1,p8e2"
output_alps_collect_limit=64
output_alps_collect_jobs=0
output_alps_params_dir=""
qalign_new_bucket_ratio=0.01
qalign_mae_delta=0.0001
qalign_min_rounds=2
qalign_stable_rounds=2
qalign_collect_per_call=512
qalign_collect_max_per_bucket=1024
record_preds="on"
record_logits="off"
logits_dir=""

while [[ $# -gt 0 ]]; do
  case "$1" in
  --model-name)
    model_name="$2"
    shift 2
    ;;
  --out-dir)
    out_dir="$2"
    shift 2
    ;;
  --txt-dir)
    txt_dir="$2"
    shift 2
    ;;
  --image-dir)
    image_dir="$2"
    shift 2
    ;;
  --image-exts)
    image_exts="$2"
    shift 2
    ;;
  --image-preprocess-script)
    image_preprocess_script="$2"
    shift 2
    ;;
  --shape)
    shape="$2"
    shift 2
    ;;
  --limit)
    limit="$2"
    shift 2
    ;;
  --jobs)
    jobs="$2"
    shift 2
    ;;
  --progress)
    progress="$2"
    shift 2
    ;;
  --warmup)
    warmup="$2"
    shift 2
    ;;
  --iters)
    iters="$2"
    shift 2
    ;;
  --timeout-sec)
    timeout_sec="$2"
    shift 2
    ;;
  --label-map)
    label_map="$2"
    shift 2
    ;;
  --suffixes)
    suffixes_csv="$2"
    shift 2
    ;;
  --baseline)
    baseline_mode="$2"
    shift 2
    ;;
  --qalign-mode)
    qalign_mode="$2"
    shift 2
    ;;
  --qalign-force-compand)
    qalign_force_compand_override="$2"
    shift 2
    ;;
  --qalign-auto)
    qalign_auto="$2"
    shift 2
    ;;
  --qalign-formats)
    qalign_formats_csv="$2"
    shift 2
    ;;
  --qalign-csv-dir)
    qalign_csv_dir="$2"
    shift 2
    ;;
  --qalign-txt-dir)
    qalign_txt_dir="$2"
    shift 2
    ;;
  --qalign-limit)
    qalign_limit="$2"
    shift 2
    ;;
  --qalign-batch-size)
    qalign_batch_size="$2"
    shift 2
    ;;
  --qalign-format-jobs)
    qalign_format_jobs="$2"
    shift 2
    ;;
  --output-alps-auto)
    output_alps_auto="$2"
    shift 2
    ;;
  --output-alps-formats)
    output_alps_formats_csv="$2"
    shift 2
    ;;
  --output-alps-collect-limit)
    output_alps_collect_limit="$2"
    shift 2
    ;;
  --output-alps-collect-jobs)
    output_alps_collect_jobs="$2"
    shift 2
    ;;
  --output-alps-params-dir)
    output_alps_params_dir="$2"
    shift 2
    ;;
  --qalign-new-bucket-ratio)
    qalign_new_bucket_ratio="$2"
    shift 2
    ;;
  --qalign-mae-delta)
    qalign_mae_delta="$2"
    shift 2
    ;;
  --qalign-min-rounds)
    qalign_min_rounds="$2"
    shift 2
    ;;
  --qalign-stable-rounds)
    qalign_stable_rounds="$2"
    shift 2
    ;;
  --qalign-collect-per-call)
    qalign_collect_per_call="$2"
    shift 2
    ;;
  --qalign-collect-max-per-bucket)
    qalign_collect_max_per_bucket="$2"
    shift 2
    ;;
  --log-file)
    log_file="$2"
    shift 2
    ;;
  --task-log-a)
    task_log_a="$2"
    shift 2
    ;;
  --format-log-b)
    format_log_b="$2"
    shift 2
    ;;
  --prediction-log-c)
    prediction_log_c="$2"
    shift 2
    ;;
  --record-preds)
    record_preds="$2"
    shift 2
    ;;
  --record-logits)
    record_logits="$2"
    shift 2
    ;;
  --logits-dir)
    logits_dir="$2"
    shift 2
    ;;
  --no-benchmark)
    no_benchmark=1
    shift
    ;;
  --with-benchmark)
    no_benchmark=0
    shift
    ;;
  --quire)
    quire_mode="$2"
    shift 2
    ;;
  --task-progress)
    task_progress_mode="$2"
    shift 2
    ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    echo "ERROR: unknown arg: $1"
    usage
    exit 2
    ;;
  esac
done

if [[ -z "${model_name}" || -z "${out_dir}" ]]; then
  usage
  exit 2
fi
if [[ -n "${txt_dir}" && -n "${image_dir}" ]]; then
  echo "ERROR: use either --txt-dir or --image-dir, not both"
  exit 2
fi
if [[ -z "${txt_dir}" && -z "${image_dir}" ]]; then
  usage
  exit 2
fi
image_mode=0
if [[ -n "${image_dir}" ]]; then
  image_mode=1
  if [[ -z "${image_preprocess_script}" ]]; then
    image_preprocess_script="${script_dir}/../../../ImageNet100/preprocess_imagenet100_tensor.py"
  fi
  if [[ ! -f "${image_preprocess_script}" ]]; then
    echo "ERROR: image preprocess script not found: ${image_preprocess_script}"
    exit 2
  fi
fi

is_int() {
  [[ "$1" =~ ^-?[0-9]+$ ]]
}

is_number() {
  local v="$1"
  local low="${v,,}"
  case "${low}" in
  nan|+nan|-nan|inf|+inf|-inf)
    return 1
    ;;
  esac
  [[ "${v}" =~ ^[-+]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][-+]?[0-9]+)?$ ]]
}

add_sum() {
  local -n sum_ref="$1"
  local -n cnt_ref="$2"
  local key="$3"
  local val="$4"
  if ! is_number "${val}"; then
    return
  fi
  local prev="${sum_ref[$key]:-0}"
  sum_ref["${key}"]="$(awk -v a="${prev}" -v b="${val}" 'BEGIN{print a+b}')"
  cnt_ref["${key}"]=$(( ${cnt_ref["${key}"]:-0} + 1 ))
}

avg_value() {
  local sum="$1"
  local cnt="$2"
  if [[ "${cnt}" -le 0 ]]; then
    echo "nan"
  else
    awk -v s="${sum}" -v n="${cnt}" 'BEGIN{print s/n}'
  fi
}

normalize_shape() {
  local s="$1"
  s="${s//,/x}"
  if [[ ! "${s}" =~ ^[0-9]+x[0-9]+x[0-9]+x[0-9]+$ ]]; then
    echo ""
  else
    echo "${s}"
  fi
}

detect_entry() {
  local so="$1"
  local sym
  sym="$(nm -D "${so}" 2>/dev/null | awk '$3 ~ /^_mlir_ciface_main_graph/ { print $3; exit }')"
  if [[ -z "${sym}" ]]; then
    echo "_mlir_ciface_main_graph"
  else
    echo "${sym}"
  fi
}

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "ERROR: file not found: $1"
    exit 2
  fi
}

extract_field() {
  local line="$1"
  local key="$2"
  echo "${line}" | sed -n "s/.* ${key}=\\([^ ]*\\).*/\\1/p"
}

extract_top1_match() {
  local line="$1"
  echo "${line}" | sed -n 's/.* Top1([^)]*)=\(match\|diff\).*/\1/p'
}

extract_top5_overlap() {
  local line="$1"
  echo "${line}" | sed -n 's/.* Top5Overlap=\([0-9]\+\).*/\1/p'
}

extract_pred_top1() {
  local line="$1"
  echo "${line}" | sed -n 's/^  top1=\(-\{0,1\}[0-9]\+\) top5=\[.*$/\1/p'
}

extract_pred_top5() {
  local line="$1"
  echo "${line}" | sed -n 's/^  top1=-\{0,1\}[0-9]\+ top5=\[\(.*\)\]$/\1/p'
}

suffix_priority() {
  local suffix="$1"
  if [[ "${suffix}" =~ ^qdq-p([0-9]+)e([0-9]+)$ ]]; then
    echo $((100000 + ${BASH_REMATCH[1]} * 100 + ${BASH_REMATCH[2]}))
    return
  fi
  case "${suffix}" in
  qdq-f32)
    echo 1000
    ;;
  nqdq-f32)
    echo 900
    ;;
  *)
    echo 0
    ;;
  esac
}

sort_model_arrays_by_priority() {
  local n="${#keys[@]}"
  local i j
  for ((i = 0; i < n; ++i)); do
    for ((j = i + 1; j < n; ++j)); do
      local si="${keys[$i]}"
      local sj="${keys[$j]}"
      local suffix_i suffix_j pri_i pri_j
      suffix_i="${si#${model_name}-}"
      suffix_j="${sj#${model_name}-}"
      suffix_i="${suffix_i%.so}"
      suffix_j="${suffix_j%.so}"
      pri_i="$(suffix_priority "${suffix_i}")"
      pri_j="$(suffix_priority "${suffix_j}")"
      if [[ "${pri_j}" -gt "${pri_i}" ]]; then
        local tmp
        tmp="${keys[$i]}"; keys[$i]="${keys[$j]}"; keys[$j]="${tmp}"
        tmp="${sos[$i]}"; sos[$i]="${sos[$j]}"; sos[$j]="${tmp}"
        tmp="${types[$i]}"; types[$i]="${types[$j]}"; types[$j]="${tmp}"
        tmp="${entries[$i]}"; entries[$i]="${entries[$j]}"; entries[$j]="${tmp}"
      fi
    done
  done
}

shape_x="$(normalize_shape "${shape}")"
if [[ -z "${shape_x}" ]]; then
  echo "ERROR: invalid --shape: ${shape}"
  exit 2
fi

for n in "${limit}" "${jobs}" "${progress}" "${warmup}" "${iters}" "${timeout_sec}"; do
  if ! is_int "${n}"; then
    echo "ERROR: numeric args must be integers"
    exit 2
  fi
done
for n in "${qalign_limit}" "${qalign_batch_size}" "${qalign_format_jobs}" "${qalign_min_rounds}" "${qalign_stable_rounds}" "${qalign_collect_per_call}" "${qalign_collect_max_per_bucket}"; do
  if ! is_int "${n}"; then
    echo "ERROR: qalign numeric args must be integers"
    exit 2
  fi
done
for n in "${output_alps_collect_limit}" "${output_alps_collect_jobs}"; do
  if ! is_int "${n}"; then
    echo "ERROR: output-alps numeric args must be integers"
    exit 2
  fi
done
if [[ "${jobs}" -lt 1 ]]; then
  echo "ERROR: --jobs must be >= 1"
  exit 2
fi
if [[ "${iters}" -lt 1 ]]; then
  echo "ERROR: --iters must be >= 1"
  exit 2
fi
if [[ "${warmup}" -lt 0 || "${limit}" -lt 0 || "${progress}" -lt 0 || "${timeout_sec}" -lt 0 ]]; then
  echo "ERROR: limit/progress/warmup/timeout-sec must be >= 0"
  exit 2
fi
if [[ "${qalign_batch_size}" -lt 1 || "${qalign_format_jobs}" -lt 1 || "${qalign_min_rounds}" -lt 1 || "${qalign_stable_rounds}" -lt 1 ]]; then
  echo "ERROR: qalign batch/format-jobs/min-rounds/stable-rounds must be >= 1"
  exit 2
fi
if [[ "${qalign_collect_per_call}" -lt 1 || "${qalign_collect_max_per_bucket}" -lt 1 ]]; then
  echo "ERROR: qalign collect-per-call/max-per-bucket must be >= 1"
  exit 2
fi
if [[ "${output_alps_collect_limit}" -lt 0 ]]; then
  echo "ERROR: output-alps collect-limit must be >= 0"
  exit 2
fi
if [[ "${output_alps_collect_jobs}" -lt 0 ]]; then
  echo "ERROR: output-alps collect-jobs must be >= 0"
  exit 2
fi
case "${baseline_mode}" in
none|nqdq-f32|qdq-f32) ;;
*)
  echo "ERROR: --baseline must be none|nqdq-f32|qdq-f32"
  exit 2
  ;;
esac
case "${qalign_auto}" in
on|off) ;;
*)
  echo "ERROR: --qalign-auto must be on|off"
  exit 2
  ;;
esac
case "${output_alps_auto}" in
on|off) ;;
*)
  echo "ERROR: --output-alps-auto must be on|off"
  exit 2
  ;;
esac
case "${record_preds}" in
on|off) ;;
*)
  echo "ERROR: --record-preds must be on|off"
  exit 2
  ;;
esac
case "${record_logits}" in
on|off) ;;
*)
  echo "ERROR: --record-logits must be on|off"
  exit 2
  ;;
esac
case "${qalign_mode}" in
off|alpha|alps) ;;
*)
  echo "ERROR: --qalign-mode must be off|alpha|alps"
  exit 2
  ;;
esac
if [[ -n "${qalign_force_compand_override}" ]]; then
  case "${qalign_force_compand_override}" in
  on|off) ;;
  *)
    echo "ERROR: --qalign-force-compand must be on|off"
    exit 2
    ;;
  esac
fi
if ! is_number "${qalign_new_bucket_ratio}" || ! is_number "${qalign_mae_delta}"; then
  echo "ERROR: --qalign-new-bucket-ratio and --qalign-mae-delta must be numbers"
  exit 2
fi
if ! awk -v a="${qalign_new_bucket_ratio}" -v b="${qalign_mae_delta}" 'BEGIN{exit !(a>=0 && b>=0)}'; then
  echo "ERROR: qalign ratio/delta thresholds must be >= 0"
  exit 2
fi
case "${task_progress_mode}" in
on|off) ;;
*)
  echo "ERROR: --task-progress must be on|off"
  exit 2
  ;;
esac

runner="${out_dir}/run_time_sp"
require_file "${runner}"

if [[ "${no_benchmark}" -eq 1 ]]; then
  if ! "${runner}" --help 2>&1 | grep -q -- "--no-benchmark"; then
    echo "WARN: ${runner} does not support --no-benchmark, fallback to --with-benchmark"
    no_benchmark=0
  fi
fi

runner_supports_quire=1
if ! "${runner}" --help 2>&1 | grep -q -- "--quire"; then
  runner_supports_quire=0
  echo "WARN: ${runner} does not support --quire, ignore quire mode (${quire_mode})"
fi

if [[ -z "${suffixes_csv}" ]]; then
  suffixes_csv="qdq-p8e0,qdq-p8e1,qdq-p8e2,qdq-p16e0,qdq-p16e1,qdq-p16e2,qdq-p32e0,qdq-p32e1,qdq-p32e2,qdq-f32,nqdq-f32"
fi

declare -a keys sos types entries
IFS=',' read -r -a _suffix_raw <<< "${suffixes_csv}"
for suffix in "${_suffix_raw[@]}"; do
  suffix="${suffix//[[:space:]]/}"
  [[ -n "${suffix}" ]] || continue
  key="${model_name}-${suffix}.so"
  so="${out_dir}/${key}"
  if [[ ! -f "${so}" ]]; then
    echo "WARN: skip missing ${so}"
    continue
  fi
  keys+=("${key}")
  sos+=("${so}")
  types+=("f32")
  entries+=("$(detect_entry "${so}")")
done

if [[ ${#keys[@]} -eq 0 ]]; then
  echo "ERROR: no runnable .so found under ${out_dir}"
  exit 2
fi

sort_model_arrays_by_priority
model_count="${#keys[@]}"
declare -A active_suffix_by_name
for key in "${keys[@]}"; do
  sfx="${key#${model_name}-}"
  sfx="${sfx%.so}"
  active_suffix_by_name["${sfx}"]=1
done

baseline_key=""
baseline_so=""
baseline_entry=""
if [[ "${baseline_mode}" != "none" ]]; then
  baseline_key="${model_name}-${baseline_mode}.so"
  baseline_so="${out_dir}/${baseline_key}"
  require_file "${baseline_so}"
  baseline_entry="$(detect_entry "${baseline_so}")"
fi

declare -a txts
if [[ "${image_mode}" -eq 1 ]]; then
  if [[ ! -d "${image_dir}" ]]; then
    echo "ERROR: image dir not found: ${image_dir}"
    exit 2
  fi
  mapfile -t txts < <(find "${image_dir}" -type f | awk -v csv="${image_exts}" '
    BEGIN {
      n = split(csv, a, ",");
      for (i = 1; i <= n; ++i) {
        gsub(/^[ \t.]+|[ \t]+$/, "", a[i]);
        exts[tolower(a[i])] = 1;
      }
    }
    {
      p = $0;
      n = split(p, parts, ".");
      ext = (n > 1 ? tolower(parts[n]) : "");
      if (ext in exts) print p;
    }' | sort)
  if [[ ${#txts[@]} -eq 0 ]]; then
    echo "ERROR: no image samples in ${image_dir}"
    exit 2
  fi
else
  if [[ ! -d "${txt_dir}" ]]; then
    echo "ERROR: txt dir not found: ${txt_dir}"
    exit 2
  fi
  mapfile -t txts < <(find "${txt_dir}" -maxdepth 1 -type f -name '*.txt' | sort)
  if [[ ${#txts[@]} -eq 0 ]]; then
    echo "ERROR: no txt samples in ${txt_dir}"
    exit 2
  fi
fi
if [[ "${limit}" -le 0 || "${limit}" -gt ${#txts[@]} ]]; then
  limit="${#txts[@]}"
fi

declare -A qalign_csv_by_suffix
declare -A qalign_round_info_by_suffix
declare -A output_alps_params_by_suffix
declare -A output_alps_info_by_suffix
qalign_source_dir="${txt_dir}"
if [[ -n "${qalign_txt_dir}" ]]; then
  qalign_source_dir="${qalign_txt_dir}"
fi

qalign_calib_compand_mode="off"
qalign_calib_force_compand="off"
qalign_runtime_compand_mode="off"
case "${qalign_mode}" in
alps)
  qalign_calib_compand_mode="alps"
  qalign_calib_force_compand="on"
  qalign_runtime_compand_mode="alps"
  ;;
alpha|off)
  qalign_calib_compand_mode="off"
  qalign_calib_force_compand="off"
  qalign_runtime_compand_mode="off"
  ;;
esac
if [[ -n "${qalign_force_compand_override}" ]]; then
  qalign_calib_force_compand="${qalign_force_compand_override}"
fi

run_qalign_format_auto() {
  local fmt="$1"
  local suffix="qdq-${fmt}"
  local qalign_so="${out_dir}/${model_name}-${suffix}.so"
  local fmt_dir="${qalign_work_root}/${fmt}"
  local result_dir="${qalign_work_root}/results"
  local status_file="${result_dir}/result_${fmt}.status"
  local csv_file="${result_dir}/result_${fmt}.csv"
  local info_file="${result_dir}/result_${fmt}.info"
  local err_file="${result_dir}/result_${fmt}.err"
  local prev_bucket_count=0
  local prev_avg_improve=""
  local stable_count=0
  local round=0
  local last_limit=0
  local last_calib_dir=""

  mkdir -p "${fmt_dir}" "${result_dir}"
  : > "${err_file}"

  while [[ "${last_limit}" -lt "${qalign_max_samples}" ]]; do
    round=$((round + 1))
    local round_limit=$((round * qalign_batch_size))
    if [[ "${round_limit}" -gt "${qalign_max_samples}" ]]; then
      round_limit="${qalign_max_samples}"
    fi
    last_limit="${round_limit}"

    local collect_csv_round="${fmt_dir}/collect_upto_${round_limit}.csv"
    local calib_dir_round="${fmt_dir}/calib_upto_${round_limit}"
    local round_log="${fmt_dir}/round_${round}.log"
    last_calib_dir="${calib_dir_round}"

    local qcmd=(
      bash "${script_dir}/qalign_pipeline_single.sh"
      --runner "${runner}"
      --so "${qalign_so}"
      --txt-dir "${qalign_source_dir}"
      --limit "${round_limit}"
      --jobs "${jobs}"
      --shape "${shape_x}"
      --out-type "${fmt}"
      --collect-csv "${collect_csv_round}"
      --calib-dir "${calib_dir_round}"
      --warmup 0
      --iters 1
      --apply-check off
    )
    if [[ "${no_benchmark}" -eq 1 ]]; then
      qcmd+=(--no-benchmark)
    else
      qcmd+=(--with-benchmark)
    fi

    set +e
    QALIGN_COMPAND_MODE="${qalign_calib_compand_mode}" \
    QALIGN_FORCE_COMPAND="${qalign_calib_force_compand}" \
      "${qcmd[@]}" > "${round_log}" 2>&1
    local qrc=$?
    set -e
    if [[ "${qrc}" -ne 0 ]]; then
      echo "fail" > "${status_file}"
      echo "qalign run failed format=${fmt} round=${round} rc=${qrc} log=${round_log}" > "${err_file}"
      return 0
    fi

    local detail_csv="${calib_dir_round}/qalign_${fmt}_detail.csv"
    if [[ ! -f "${detail_csv}" ]]; then
      echo "fail" > "${status_file}"
      echo "qalign detail file missing: ${detail_csv}" > "${err_file}"
      return 0
    fi

    local metrics
    metrics="$(python3 - "${detail_csv}" <<'PY'
import csv, sys
p = sys.argv[1]
n = 0
s = 0.0
with open(p) as f:
    r = csv.DictReader(f)
    for row in r:
        n += 1
        try:
            s += float(row.get("mae_improve", "0") or 0.0)
        except Exception:
            pass
avg = (s / n) if n else 0.0
print(f"{n}\t{avg}")
PY
)"

    local bucket_count avg_improve
    IFS=$'\t' read -r bucket_count avg_improve <<< "${metrics}"
    if [[ -z "${bucket_count}" ]]; then
      bucket_count=0
    fi
    if [[ -z "${avg_improve}" ]]; then
      avg_improve=0
    fi
    if [[ "${bucket_count}" -lt 1 ]]; then
      echo "WARN: qalign no buckets for format=${fmt} at round=${round}"
    fi

    local new_buckets=$((bucket_count - prev_bucket_count))
    if [[ "${new_buckets}" -lt 0 ]]; then
      new_buckets=0
    fi
    local new_bucket_ratio
    new_bucket_ratio="$(awk -v nb="${new_buckets}" -v bc="${bucket_count}" 'BEGIN{ if (bc<=0) print 0; else print nb/bc }')"

    local improve_delta
    if [[ -z "${prev_avg_improve}" ]]; then
      improve_delta="1e30"
    else
      improve_delta="$(awk -v a="${avg_improve}" -v b="${prev_avg_improve}" 'BEGIN{d=a-b; if (d<0) d=-d; print d}')"
    fi

    local round_stable=0
    if [[ "${round}" -ge "${qalign_min_rounds}" ]]; then
      round_stable="$(awk -v r="${new_bucket_ratio}" -v t1="${qalign_new_bucket_ratio}" -v d="${improve_delta}" -v t2="${qalign_mae_delta}" 'BEGIN{ print (r<=t1 && d<=t2) ? 1 : 0 }')"
    fi
    if [[ "${round_stable}" -eq 1 ]]; then
      stable_count=$((stable_count + 1))
    else
      stable_count=0
    fi

    echo "[qalign-auto][${fmt}] round=${round} limit=${round_limit} buckets=${bucket_count} new_ratio=${new_bucket_ratio} avg_improve=${avg_improve} delta=${improve_delta} stable=${stable_count}/${qalign_stable_rounds}"

    prev_bucket_count="${bucket_count}"
    prev_avg_improve="${avg_improve}"
    if [[ "${stable_count}" -ge "${qalign_stable_rounds}" ]]; then
      break
    fi
  done

  local final_csv="${last_calib_dir}/qalign_${fmt}.csv"
  if [[ ! -f "${final_csv}" ]]; then
    echo "fail" > "${status_file}"
    echo "qalign final csv missing for ${fmt}: ${final_csv}" > "${err_file}"
    return 0
  fi

  echo "ok" > "${status_file}"
  echo "${final_csv}" > "${csv_file}"
  echo "source=auto rounds=${round} samples=${last_limit}" > "${info_file}"
  return 0
}

if [[ "${qalign_auto}" == "on" || -n "${qalign_csv_dir}" ]]; then
  if [[ "${image_mode}" -eq 1 && "${qalign_auto}" == "on" && -z "${qalign_txt_dir}" ]]; then
    echo "ERROR: qalign-auto currently needs txt tensors. Use --qalign-auto off, or pass --qalign-txt-dir."
    exit 2
  fi
  if [[ ! -d "${qalign_source_dir}" ]]; then
    echo "ERROR: qalign txt dir not found: ${qalign_source_dir}"
    exit 2
  fi
  mapfile -t qalign_txts < <(find "${qalign_source_dir}" -maxdepth 1 -type f -name '*.txt' | sort)
  if [[ ${#qalign_txts[@]} -eq 0 && "${qalign_auto}" == "on" ]]; then
    echo "ERROR: qalign source has no txt files: ${qalign_source_dir}"
    exit 2
  fi
  qalign_max_samples="${limit}"
  if [[ "${qalign_limit}" -gt 0 ]]; then
    qalign_max_samples="${qalign_limit}"
  fi
  if [[ "${qalign_max_samples}" -gt ${#qalign_txts[@]} ]]; then
    qalign_max_samples="${#qalign_txts[@]}"
  fi
  if [[ "${qalign_auto}" == "on" && "${qalign_max_samples}" -le 0 ]]; then
    echo "ERROR: qalign max sample count is 0"
    exit 2
  fi

  qalign_work_root="${out_dir}/qalign_auto/${model_name}"
  mkdir -p "${qalign_work_root}"
  declare -a qalign_auto_formats=()
  IFS=',' read -r -a _qalign_fmt_raw <<< "${qalign_formats_csv}"
  for fmt in "${_qalign_fmt_raw[@]}"; do
    fmt="${fmt//[[:space:]]/}"
    [[ -n "${fmt}" ]] || continue
    if [[ ! "${fmt}" =~ ^p8e[0-2]$ ]]; then
      echo "WARN: qalign currently supports p8e0/p8e1/p8e2 only, skip format=${fmt}"
      continue
    fi
    suffix="qdq-${fmt}"
    if [[ -z "${active_suffix_by_name[${suffix}]:-}" ]]; then
      continue
    fi
    qalign_so="${out_dir}/${model_name}-${suffix}.so"
    if [[ ! -f "${qalign_so}" ]]; then
      echo "WARN: qalign skip missing so: ${qalign_so}"
      continue
    fi

    if [[ -n "${qalign_csv_dir}" && -f "${qalign_csv_dir}/qalign_${fmt}.csv" ]]; then
      qalign_csv_by_suffix["${suffix}"]="${qalign_csv_dir}/qalign_${fmt}.csv"
      qalign_round_info_by_suffix["${suffix}"]="source=external_csv_dir"
      continue
    fi

    if [[ "${qalign_auto}" != "on" ]]; then
      continue
    fi
    qalign_auto_formats+=("${fmt}")
  done

  if [[ "${qalign_auto}" == "on" && ${#qalign_auto_formats[@]} -gt 0 ]]; then
    echo "[qalign-auto] start format-level parallel collect jobs=${qalign_format_jobs} formats=${qalign_auto_formats[*]}"
    for fmt in "${qalign_auto_formats[@]}"; do
      run_qalign_format_auto "${fmt}" &
      while :; do
        running_jobs="$(jobs -pr | wc -l | tr -d ' ')"
        if [[ "${running_jobs}" -lt "${qalign_format_jobs}" ]]; then
          break
        fi
        sleep 0.2
      done
    done
    set +e
    wait
    set -e

    for fmt in "${qalign_auto_formats[@]}"; do
      suffix="qdq-${fmt}"
      result_status="${qalign_work_root}/results/result_${fmt}.status"
      result_csv="${qalign_work_root}/results/result_${fmt}.csv"
      result_info="${qalign_work_root}/results/result_${fmt}.info"
      result_err="${qalign_work_root}/results/result_${fmt}.err"
      if [[ ! -f "${result_status}" ]]; then
        echo "ERROR: qalign result status missing for ${fmt}: ${result_status}"
        exit 4
      fi
      status="$(<"${result_status}")"
      if [[ "${status}" != "ok" ]]; then
        echo "ERROR: qalign auto failed for ${fmt}"
        if [[ -f "${result_err}" ]]; then
          sed -n '1,20p' "${result_err}"
        fi
        exit 4
      fi
      if [[ ! -f "${result_csv}" ]]; then
        echo "ERROR: qalign result csv pointer missing for ${fmt}: ${result_csv}"
        exit 4
      fi
      final_csv="$(<"${result_csv}")"
      if [[ ! -f "${final_csv}" ]]; then
        echo "ERROR: qalign final csv missing for ${fmt}: ${final_csv}"
        exit 4
      fi
      info_line="source=auto"
      if [[ -f "${result_info}" ]]; then
        info_line="$(<"${result_info}")"
      fi
      qalign_csv_by_suffix["${suffix}"]="${final_csv}"
      qalign_round_info_by_suffix["${suffix}"]="${info_line}"
    done
  fi
fi

merge_runtime_output_alps_collect_csvs() {
  local out_csv="$1"
  shift
  awk -F',' '
    BEGIN {
      OFS=",";
      print "format,key,channel,seen,sample_count,samples";
    }
    FNR == 1 { next }
    NF < 6 { next }
    {
      k = $1 OFS $2 OFS $3;
      seen[k] += $4 + 0;
      sample_count[k] += $5 + 0;
      if ($6 != "") {
        if (samples[k] == "") samples[k] = $6;
        else samples[k] = samples[k] ";" $6;
      }
    }
    END {
      for (k in seen)
        print k, seen[k], sample_count[k], samples[k];
    }
  ' "$@" > "${out_csv}"
}

run_output_alps_auto_for_suffix() {
  local suffix="$1"
  local so="${out_dir}/${model_name}-${suffix}.so"
  local ent
  ent="$(detect_entry "${so}")"
  local fmt="${suffix##*-}"
  local params_root="${output_alps_params_dir:-${out_dir}/runtime_output_alps_auto/${model_name}}"
  local fmt_dir="${params_root}/${suffix}"
  local collect_count="${output_alps_collect_limit}"
  local collect_jobs="${output_alps_collect_jobs}"
  if [[ "${collect_count}" -le 0 || "${collect_count}" -gt "${limit}" ]]; then
    collect_count="${limit}"
  fi
  if [[ "${collect_jobs}" -le 0 ]]; then
    collect_jobs="${jobs}"
  fi
  if [[ "${collect_jobs}" -gt "${collect_count}" ]]; then
    collect_jobs="${collect_count}"
  fi
  mkdir -p "${fmt_dir}"
  local collect_csv="${fmt_dir}/collect_upto_${collect_count}.csv"
  local params_csv="${fmt_dir}/params_upto_${collect_count}.csv"
  local info_file="${fmt_dir}/info.txt"
  local collect_parts_dir="${fmt_dir}/collect_parts_upto_${collect_count}"
  local output_alps_calib_tool="${out_dir}/output_alps_calibrate"

  if [[ ! -x "${output_alps_calib_tool}" ]]; then
    echo "ERROR: output-alps calibrate tool missing: ${output_alps_calib_tool}"
    exit 4
  fi

  echo "[output-alps-auto] collect suffix=${suffix} format=${fmt} samples=${collect_count} jobs=${collect_jobs}"
  rm -f "${collect_csv}" "${params_csv}"
  rm -rf "${collect_parts_dir}"
  mkdir -p "${collect_parts_dir}"

  collect_one_output_alps_sample() {
    local sample_idx="$1"
    local sample="${txts[$sample_idx]}"
    local part_csv="${collect_parts_dir}/collect_${sample_idx}.csv"
    local cmd=(
      "${runner}" "${so}"
      --shape "${shape_x}"
      --out-type "f32"
      --entry "${ent}"
      --warmup "${warmup}"
      --iters "${iters}"
      --quiet
    )
    if [[ "${image_mode}" -eq 1 ]]; then
      cmd+=(--image "${sample}" --image-preprocess-script "${image_preprocess_script}")
    else
      cmd+=("${sample}")
    fi
    if [[ "${runner_supports_quire}" -eq 1 ]]; then
      cmd+=(--quire "${quire_mode}")
    fi
    if [[ "${no_benchmark}" -eq 1 ]]; then
      cmd+=(--no-benchmark)
    fi
    env "POSIT_RUNTIME_OUTPUT_ALPS_COLLECT_FILE=${part_csv}" \
      "${cmd[@]}" > /dev/null 2>&1
  }

  if [[ "${collect_jobs}" -le 1 ]]; then
    for ((i = 0; i < collect_count; ++i)); do
      collect_one_output_alps_sample "${i}"
    done
  else
    for ((i = 0; i < collect_count; ++i)); do
      collect_one_output_alps_sample "${i}" &
      while [[ "$(jobs -pr | wc -l | tr -d ' ')" -ge "${collect_jobs}" ]]; do
        wait -n
      done
    done
    wait
  fi

  shopt -s nullglob
  local collect_parts=("${collect_parts_dir}"/collect_*.csv)
  shopt -u nullglob
  if [[ ${#collect_parts[@]} -eq 0 ]]; then
    echo "ERROR: output-alps collect produced no partial csv for ${suffix}"
    exit 4
  fi
  merge_runtime_output_alps_collect_csvs "${collect_csv}" "${collect_parts[@]}"
  "${output_alps_calib_tool}" "${collect_csv}" "${params_csv}" > /dev/null 2>&1

  if [[ ! -f "${params_csv}" ]]; then
    echo "ERROR: output-alps params missing for ${suffix}: ${params_csv}"
    exit 4
  fi
  printf 'source=auto samples=%s collect=%s\n' "${collect_count}" "${collect_csv}" > "${info_file}"
  output_alps_params_by_suffix["${suffix}"]="${params_csv}"
  output_alps_info_by_suffix["${suffix}"]="source=auto samples=${collect_count}"
}

if [[ "${output_alps_auto}" == "on" || -n "${output_alps_params_dir}" ]]; then
  declare -a output_alps_auto_suffixes=()
  IFS=',' read -r -a _output_alps_fmt_raw <<< "${output_alps_formats_csv}"
  for fmt in "${_output_alps_fmt_raw[@]}"; do
    fmt="${fmt//[[:space:]]/}"
    [[ -n "${fmt}" ]] || continue
    if [[ ! "${fmt}" =~ ^p([4-9]|1[0-5])e[0-2]$ ]]; then
      echo "WARN: output-alps-auto supports p4..p15 e0/e1/e2 only, skip format=${fmt}"
      continue
    fi
    for prefix in qdq nqdq; do
      suffix="${prefix}-${fmt}"
      if [[ -z "${active_suffix_by_name[${suffix}]:-}" ]]; then
        continue
      fi
      # qdq/int8 posit path registers NO output-ALPS collect points (int8
      # activations are governed by dequant scales, not posit output metadata),
      # so collect would produce an empty CSV and hard-fail. Skip qdq for
      # output-alps-auto with a warning; it is still evaluated, just without
      # runtime output-ALPS. Only nqdq suffixes get output-ALPS.
      if [[ "${prefix}" == "qdq" ]]; then
        echo "WARN: output-alps-auto skips ${suffix} (qdq/int8 posit has no output-ALPS collect points); it runs without output-ALPS."
        continue
      fi
      params_root="${output_alps_params_dir:-${out_dir}/runtime_output_alps_auto/${model_name}}"
      fmt_dir="${params_root}/${suffix}"
      collect_count="${output_alps_collect_limit}"
      if [[ "${collect_count}" -le 0 || "${collect_count}" -gt "${limit}" ]]; then
        collect_count="${limit}"
      fi
      existing_params="${fmt_dir}/params_upto_${collect_count}.csv"
      if [[ -n "${output_alps_params_dir}" && -f "${existing_params}" ]]; then
        output_alps_params_by_suffix["${suffix}"]="${existing_params}"
        output_alps_info_by_suffix["${suffix}"]="source=existing"
        continue
      fi
      if [[ "${output_alps_auto}" == "on" ]]; then
        output_alps_auto_suffixes+=("${suffix}")
      fi
    done
  done
  if [[ "${output_alps_auto}" == "on" && ${#output_alps_auto_suffixes[@]} -gt 0 ]]; then
    echo "[output-alps-auto] start suffixes=${output_alps_auto_suffixes[*]}"
    for suffix in "${output_alps_auto_suffixes[@]}"; do
      run_output_alps_auto_for_suffix "${suffix}"
    done
  fi
fi

if [[ -z "${log_file}" ]]; then
  log_file="${out_dir}/${model_name}-11.dataset_parallel.log"
fi
if [[ -z "${task_log_a}" ]]; then
  task_log_a="${out_dir}/${model_name}-11.task_progress.A.log"
fi
if [[ -z "${format_log_b}" ]]; then
  format_log_b="${out_dir}/${model_name}-11.format_summary.B.log"
fi
if [[ "${record_preds}" == "on" && -z "${prediction_log_c}" ]]; then
  prediction_log_c="${out_dir}/${model_name}-11.predictions.C.log"
fi
if [[ "${record_logits}" == "on" && -z "${logits_dir}" ]]; then
  logits_dir="${out_dir}/${model_name}-11.logits"
fi
mkdir -p "$(dirname "${log_file}")"
mkdir -p "$(dirname "${task_log_a}")"
mkdir -p "$(dirname "${format_log_b}")"
if [[ "${record_preds}" == "on" ]]; then
  mkdir -p "$(dirname "${prediction_log_c}")"
fi
if [[ "${record_logits}" == "on" ]]; then
  mkdir -p "${logits_dir}"
fi
build_posit_config_file="${out_dir}/build_posit_config.log"
: > "${log_file}"
: > "${task_log_a}"
: > "${format_log_b}"
echo -e "#ts_ms\ttask_id\tsample_idx\tsample_name\tkey\tformat\trc\tlat_us\tmae\trmse\tmaxabs\tcosine\trmae\tjs\ttop1_match_bin\ttop5_overlap\tgt_top1_bin\tgt_top5_bin" >> "${task_log_a}"
echo -e "#ts_ms\tkey\tformat\ttasks_total\ttasks_ok\ttasks_fail\tavg_lat_us\tavg_mae\tavg_rmse\tavg_maxabs\tavg_cosine\tavg_rmae\tavg_js\ttop1_match_pct\ttop5_overlap\tgt_top1_pct\tgt_top5_pct" >> "${format_log_b}"
if [[ "${record_preds}" == "on" ]]; then
  : > "${prediction_log_c}"
  echo -e "#ts_ms\ttask_id\tsample_idx\tsample_name\tkey\tformat\trc\tgt_label\tpred_top1_class_id\tpred_top5_class_ids\tlogits_csv" >> "${prediction_log_c}"
fi

declare -A label_by_name
declare -A class_index_by_dir
if [[ "${image_mode}" -eq 1 ]]; then
  class_idx=0
  while IFS= read -r cls_dir; do
    class_index_by_dir["$(basename "${cls_dir}")"]="${class_idx}"
    class_idx=$((class_idx + 1))
  done < <(find "${image_dir}" -mindepth 1 -maxdepth 1 -type d | sort)
fi
label_enabled=0
if [[ -n "${label_map}" ]]; then
  require_file "${label_map}"
  while IFS= read -r raw; do
    [[ -z "${raw}" ]] && continue
    [[ "${raw}" =~ ^[[:space:]]*# ]] && continue
    line="${raw//,/ }"
    read -r f l _rest <<<"${line}"
    [[ -z "${f}" || -z "${l}" ]] && continue
    is_int "${l}" || continue
    label_by_name["${f}"]="${l}"
    label_by_name["$(basename "${f}")"]="${l}"
  done < "${label_map}"
  if [[ ${#label_by_name[@]} -gt 0 ]]; then
    label_enabled=1
  else
    echo "WARN: no valid labels parsed from ${label_map}, continue without GT top1/top5"
  fi
fi

declare -a sample_labels
missing_label=0
for ((i = 0; i < limit; ++i)); do
  sample="${txts[$i]}"
  lbl=""
  if [[ "${label_enabled}" -eq 1 ]]; then
    base="$(basename "${sample}")"
    rel="${sample}"
    if [[ "${image_mode}" -eq 1 ]]; then
      rel="${sample#${image_dir}/}"
    fi
    lbl="${label_by_name["${sample}"]:-${label_by_name["${rel}"]:-${label_by_name["${base}"]:-}}}"
    if [[ -z "${lbl}" ]]; then
      missing_label=$((missing_label + 1))
    fi
  elif [[ "${image_mode}" -eq 1 ]]; then
    parent="$(basename "$(dirname "${sample}")")"
    lbl="${class_index_by_dir["${parent}"]:-}"
    if [[ -z "${lbl}" ]]; then
      missing_label=$((missing_label + 1))
    fi
  fi
  sample_labels+=("${lbl}")
done

{
  echo "time_model11_dataset_parallel config"
  echo "  model_name=${model_name}"
  echo "  out_dir=${out_dir}"
  if [[ "${image_mode}" -eq 1 ]]; then
    echo "  input_mode=image"
    echo "  image_dir=${image_dir}"
    echo "  image_exts=${image_exts}"
    echo "  image_preprocess_script=${image_preprocess_script}"
  else
    echo "  input_mode=txt"
    echo "  txt_dir=${txt_dir}"
  fi
  echo "  shape=${shape_x}"
  echo "  jobs=${jobs}"
  echo "  limit=${limit}"
  echo "  warmup=${warmup}"
  echo "  iters=${iters}"
  echo "  timeout_sec=${timeout_sec}"
  if [[ "${no_benchmark}" -eq 1 ]]; then
    echo "  mode=no-benchmark"
  else
    echo "  mode=with-benchmark"
  fi
  echo "  task_progress=${task_progress_mode}"
  echo "  suffixes=${keys[*]}"
  echo "  qalign_auto=${qalign_auto}"
  echo "  qalign_mode=${qalign_mode}"
  echo "  output_alps_auto=${output_alps_auto}"
  echo "  output_alps_formats=${output_alps_formats_csv}"
  echo "  output_alps_collect_limit=${output_alps_collect_limit}"
  echo "  output_alps_collect_jobs=${output_alps_collect_jobs}"
  echo "  env.POSIT_RUNTIME_OUTPUT_ALPS_THETA_MIN=${POSIT_RUNTIME_OUTPUT_ALPS_THETA_MIN:-<unset>}"
  echo "  env.POSIT_RUNTIME_OUTPUT_ALPS_THETA_MAX=${POSIT_RUNTIME_OUTPUT_ALPS_THETA_MAX:-<unset>}"
  echo "  env.POSIT_RUNTIME_OUTPUT_ALPS_THETA_STEPS=${POSIT_RUNTIME_OUTPUT_ALPS_THETA_STEPS:-<unset>}"
  echo "  env.POSIT_RUNTIME_OUTPUT_ALPS_GAMMA_TARGET=${POSIT_RUNTIME_OUTPUT_ALPS_GAMMA_TARGET:-<unset>}"
  echo "  env.POSIT_RUNTIME_OUTPUT_ALPS_GAMMA_PERCENTILE=${POSIT_RUNTIME_OUTPUT_ALPS_GAMMA_PERCENTILE:-<unset>}"
  echo "  env.POSIT_RUNTIME_OUTPUT_ALPS_MIN_GAIN=${POSIT_RUNTIME_OUTPUT_ALPS_MIN_GAIN:-<unset>}"
  echo "  env.POSIT_RUNTIME_OUTPUT_ALPS_MAX_SAMPLES=${POSIT_RUNTIME_OUTPUT_ALPS_MAX_SAMPLES:-<unset>}"
  echo "  env.POSIT_RUNTIME_OUTPUT_ALPS_RS_VALUES_P8=${POSIT_RUNTIME_OUTPUT_ALPS_RS_VALUES_P8:-<unset>}"
  echo "  env.POSIT_RUNTIME_OUTPUT_ALPS_SC_VALUES_P8=${POSIT_RUNTIME_OUTPUT_ALPS_SC_VALUES_P8:-<unset>}"
  echo "  env.POSIT_RUNTIME_OUTPUT_ALPS_RS_VALUES_P8E0=${POSIT_RUNTIME_OUTPUT_ALPS_RS_VALUES_P8E0:-<unset>}"
  echo "  env.POSIT_RUNTIME_OUTPUT_ALPS_SC_VALUES_P8E0=${POSIT_RUNTIME_OUTPUT_ALPS_SC_VALUES_P8E0:-<unset>}"
  echo "  env.POSIT_RUNTIME_OUTPUT_ALPS_RS_VALUES_P8E1=${POSIT_RUNTIME_OUTPUT_ALPS_RS_VALUES_P8E1:-<unset>}"
  echo "  env.POSIT_RUNTIME_OUTPUT_ALPS_SC_VALUES_P8E1=${POSIT_RUNTIME_OUTPUT_ALPS_SC_VALUES_P8E1:-<unset>}"
  echo "  env.POSIT_RUNTIME_OUTPUT_ALPS_RS_VALUES_P8E2=${POSIT_RUNTIME_OUTPUT_ALPS_RS_VALUES_P8E2:-<unset>}"
  echo "  env.POSIT_RUNTIME_OUTPUT_ALPS_SC_VALUES_P8E2=${POSIT_RUNTIME_OUTPUT_ALPS_SC_VALUES_P8E2:-<unset>}"
  if [[ -n "${output_alps_params_dir}" ]]; then
    echo "  output_alps_params_dir=${output_alps_params_dir}"
  fi
  if [[ -f "${build_posit_config_file}" ]]; then
    echo "  build_posit_config_file=${build_posit_config_file}"
    while IFS= read -r line; do
      [[ -n "${line}" ]] || continue
      [[ "${line}" =~ ^# ]] && continue
      echo "  build_${line}"
    done < "${build_posit_config_file}"
  fi
  if [[ -n "${qalign_force_compand_override}" ]]; then
    echo "  qalign_force_compand_override=${qalign_force_compand_override}"
  fi
  echo "  qalign_calib_compand_mode=${qalign_calib_compand_mode}"
  echo "  qalign_calib_force_compand=${qalign_calib_force_compand}"
  echo "  qalign_runtime_compand_mode=${qalign_runtime_compand_mode}"
  echo "  qalign_formats=${qalign_formats_csv}"
  if [[ -n "${qalign_csv_dir}" ]]; then
    echo "  qalign_csv_dir=${qalign_csv_dir}"
  fi
  echo "  qalign_txt_dir=${qalign_source_dir}"
  if [[ "${qalign_auto}" == "on" ]]; then
    echo "  qalign_limit=${qalign_limit}"
    echo "  qalign_batch_size=${qalign_batch_size}"
    echo "  qalign_format_jobs=${qalign_format_jobs}"
    echo "  qalign_new_bucket_ratio=${qalign_new_bucket_ratio}"
    echo "  qalign_mae_delta=${qalign_mae_delta}"
    echo "  qalign_min_rounds=${qalign_min_rounds}"
    echo "  qalign_stable_rounds=${qalign_stable_rounds}"
    echo "  env.QALIGN_COMPAND_MODE=${QALIGN_COMPAND_MODE:-<unset>}"
    echo "  env.QALIGN_FORCE_COMPAND=${QALIGN_FORCE_COMPAND:-<unset>}"
    echo "  env.QALIGN_COMPAND_THETA_MIN=${QALIGN_COMPAND_THETA_MIN:-<unset>}"
    echo "  env.QALIGN_COMPAND_THETA_MAX=${QALIGN_COMPAND_THETA_MAX:-<unset>}"
    echo "  env.QALIGN_COMPAND_THETA_STEPS=${QALIGN_COMPAND_THETA_STEPS:-<unset>}"
    echo "  env.QALIGN_COMPAND_GAMMA_TARGET=${QALIGN_COMPAND_GAMMA_TARGET:-<unset>}"
    echo "  env.QALIGN_COMPAND_GAMMA_PERCENTILE=${QALIGN_COMPAND_GAMMA_PERCENTILE:-<unset>}"
    echo "  env.QALIGN_COMPAND_MIN_GAIN=${QALIGN_COMPAND_MIN_GAIN:-<unset>}"
    echo "  env.QALIGN_PAPER_SIGMA=${QALIGN_PAPER_SIGMA:-<unset>}"
    echo "  env.QALIGN_AUTO_SIGMA=${QALIGN_AUTO_SIGMA:-<unset>}"
    echo "  env.QALIGN_SIGMA_MIN=${QALIGN_SIGMA_MIN:-<unset>}"
    echo "  env.QALIGN_SIGMA_MAX=${QALIGN_SIGMA_MAX:-<unset>}"
    echo "  env.QALIGN_SCORE_MAE_WEIGHT=${QALIGN_SCORE_MAE_WEIGHT:-<unset>}"
    echo "  env.QALIGN_SCORE_ALPHA_REG=${QALIGN_SCORE_ALPHA_REG:-<unset>}"
  fi
  for key in "${keys[@]}"; do
    suffix="${key#${model_name}-}"
    suffix="${suffix%.so}"
    if [[ -n "${qalign_csv_by_suffix[${suffix}]:-}" ]]; then
      qalign_csv_info="${qalign_csv_by_suffix[${suffix}]}"
      qalign_round_info="${qalign_round_info_by_suffix[${suffix}]:-}"
      echo "  qalign_csv[${suffix}]=${qalign_csv_info} (${qalign_round_info})"
    fi
    if [[ -n "${output_alps_params_by_suffix[${suffix}]:-}" ]]; then
      output_alps_info="${output_alps_info_by_suffix[${suffix}]:-}"
      output_alps_params_path="${output_alps_params_by_suffix[${suffix}]}"
      echo "  output_alps_params[${suffix}]=${output_alps_params_path} (${output_alps_info})"
      if [[ -f "${output_alps_params_path}" ]]; then
        output_alps_rows="$(awk -F, 'NR > 1 && NF > 0 {n++} END {print n + 0}' "${output_alps_params_path}")"
        output_alps_keys="$(awk -F, 'NR > 1 && NF > 0 {k[$2] = 1} END {print length(k) + 0}' "${output_alps_params_path}")"
        output_alps_preview="$(awk -F, 'NR == 2 {printf "key=%s channel=%s theta=%s gamma=%s gp_enabled=%s gp_rs=%s gp_sc=%s", $2, $3, $5, $6, $7, $8, $9}' "${output_alps_params_path}")"
        echo "  output_alps_params_rows[${suffix}]=${output_alps_rows}"
        echo "  output_alps_params_keys[${suffix}]=${output_alps_keys}"
        if [[ -n "${output_alps_preview}" ]]; then
          echo "  output_alps_params_preview[${suffix}]=${output_alps_preview}"
        fi
      fi
    fi
  done
  if [[ "${label_enabled}" -eq 1 ]]; then
    echo "  label_map=${label_map}"
    echo "  samples_missing_label=${missing_label}"
  elif [[ "${image_mode}" -eq 1 ]]; then
    echo "  label_source=image_folder_order"
    echo "  samples_missing_label=${missing_label}"
  fi
  if [[ "${baseline_mode}" != "none" ]]; then
    echo "  baseline_so=${baseline_key}"
  else
    echo "  baseline_so=none"
  fi
  echo "  log_file=${log_file}"
  echo "  task_log_a=${task_log_a}"
  echo "  format_log_b=${format_log_b}"
  echo "  record_preds=${record_preds}"
  if [[ "${record_preds}" == "on" ]]; then
    echo "  prediction_log_c=${prediction_log_c}"
  fi
  echo "  record_logits=${record_logits}"
  if [[ "${record_logits}" == "on" ]]; then
    echo "  logits_dir=${logits_dir}"
  fi
  echo
} >> "${log_file}"

tmp_root="$(mktemp -d /tmp/model11_tasks.XXXXXX)"
task_file="${tmp_root}/tasks.tsv"
next_task_file="${tmp_root}/next_task.txt"
task_lock="${tmp_root}/task.lock"
progress_lock="${tmp_root}/progress.lock"
task_log_a_lock="${tmp_root}/task_a.lock"
format_log_b_lock="${tmp_root}/format_b.lock"
prediction_log_c_lock="${tmp_root}/pred_c.lock"
first_fail_lock="${tmp_root}/first_fail.lock"
meta_dir="${tmp_root}/meta"
task_log_dir="${tmp_root}/logs"
mkdir -p "${meta_dir}" "${task_log_dir}"
echo "1" > "${next_task_file}"
echo "0" > "${tmp_root}/completed_tasks.txt"
first_fail_printed_file="${tmp_root}/first_fail_printed"

cleanup_tmp() {
  if [[ -n "${tmp_root:-}" && -d "${tmp_root}" ]]; then
    rm -rf "${tmp_root}"
  fi
}
trap cleanup_tmp EXIT

task_id=0
for ((m = 0; m < model_count; ++m)); do
  key="${keys[$m]}"
  so="${sos[$m]}"
  ty="${types[$m]}"
  ent="${entries[$m]}"
  for ((i = 0; i < limit; ++i)); do
    sample="${txts[$i]}"
    lbl="${sample_labels[$i]}"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${task_id}" "${i}" "${key}" "${so}" "${ty}" "${ent}" "${sample}" "${lbl}" >> "${task_file}"
    task_id=$((task_id + 1))
  done
done
task_total="${task_id}"

claim_next_task() {
  local idx line
  exec {lfd}> "${task_lock}"
  flock "${lfd}"
  idx="$(cat "${next_task_file}")"
  line="$(sed -n "${idx}p" "${task_file}")"
  if [[ -n "${line}" ]]; then
    echo "$((idx + 1))" > "${next_task_file}"
  fi
  flock -u "${lfd}"
  exec {lfd}>&-
  [[ -n "${line}" ]] || return 1
  printf '%s\n' "${line}"
}

progress_label_for_key() {
  local key="$1"
  local suffix="${key#${model_name}-}"
  suffix="${suffix%.so}"
  if [[ "${suffix}" == qdq-p* ]]; then
    echo "${suffix#qdq-}"
  else
    echo "${suffix}"
  fi
}

record_first_failure() {
  local sample="$1"
  local key="$2"
  local rc="$3"
  local logf="$4"
  exec {lfd}> "${first_fail_lock}"
  flock "${lfd}"
  if [[ ! -f "${first_fail_printed_file}" ]]; then
    : > "${first_fail_printed_file}"
    echo "[first-fail] sample=${sample} so=${key} rc=${rc}" >&2
    sed -n '1,60p' "${logf}" >&2 || true
  fi
  flock -u "${lfd}"
  exec {lfd}>&-
}

append_task_log_a() {
  local task_id="$1"
  local sample_idx="$2"
  local key="$3"
  local sample="$4"
  local rc="$5"
  local lat="$6"
  local mae="$7"
  local rmse="$8"
  local maxabs="$9"
  local cosine="${10}"
  local rmae="${11}"
  local js="${12}"
  local t1m_bin="${13}"
  local t5ov="${14}"
  local gt1_bin="${15}"
  local gt5_bin="${16}"

  local ts_ms sample_name format_label
  ts_ms="$(date +%s%3N)"
  sample_name="$(basename "${sample}")"
  format_label="$(progress_label_for_key "${key}")"

  exec {lfd}> "${task_log_a_lock}"
  flock "${lfd}"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${ts_ms}" "${task_id}" "${sample_idx}" "${sample_name}" "${key}" "${format_label}" \
    "${rc}" "${lat}" "${mae}" "${rmse}" "${maxabs}" "${cosine}" "${rmae}" "${js}" \
    "${t1m_bin}" "${t5ov}" "${gt1_bin}" "${gt5_bin}" >> "${task_log_a}"
  flock -u "${lfd}"
  exec {lfd}>&-
}

append_prediction_log_c() {
  local task_id="$1"
  local sample_idx="$2"
  local key="$3"
  local sample="$4"
  local rc="$5"
  local gt_label="$6"
  local pred_top1="$7"
  local pred_top5="$8"
  local logits_csv="$9"

  local ts_ms sample_name format_label
  ts_ms="$(date +%s%3N)"
  sample_name="$(basename "${sample}")"
  format_label="$(progress_label_for_key "${key}")"

  exec {lfd}> "${prediction_log_c_lock}"
  flock "${lfd}"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${ts_ms}" "${task_id}" "${sample_idx}" "${sample_name}" "${key}" "${format_label}" \
    "${rc}" "${gt_label}" "${pred_top1}" "${pred_top5}" "${logits_csv}" >> "${prediction_log_c}"
  flock -u "${lfd}"
  exec {lfd}>&-
}

write_format_log_b() {
  local key="$1"
  local format_label="$2"
  local ts_ms summary_line

  ts_ms="$(date +%s%3N)"
  summary_line="$(awk -F'\t' -v key="${key}" '
    function isn(x) { return x ~ /^[-+]?([0-9]*[.])?[0-9]+([eE][-+]?[0-9]+)?$/ }
    $3 == key {
      total++
      if ($4 == 0) {
        ok++
        if (isn($5))  { lat_sum += $5; lat_cnt++ }
        if (isn($6))  { mae_sum += $6; mae_cnt++ }
        if (isn($7))  { rmse_sum += $7; rmse_cnt++ }
        if (isn($8))  { maxabs_sum += $8; maxabs_cnt++ }
        if (isn($9))  { cos_sum += $9; cos_cnt++ }
        if (isn($10)) { rmae_sum += $10; rmae_cnt++ }
        if (isn($11)) { js_sum += $11; js_cnt++ }
        if ($12 == "1" || $12 == "0") { t1_cnt++; if ($12 == "1") t1_yes++ }
        if (isn($13)) { t5_sum += $13; t5_cnt++ }
        if ($14 == "1" || $14 == "0") { gt1_cnt++; if ($14 == "1") gt1_yes++ }
        if ($15 == "1" || $15 == "0") { gt5_cnt++; if ($15 == "1") gt5_yes++ }
      } else {
        fail++
      }
    }
    END {
      avg_lat = (lat_cnt ? lat_sum / lat_cnt : "nan")
      avg_mae = (mae_cnt ? mae_sum / mae_cnt : "nan")
      avg_rmse = (rmse_cnt ? rmse_sum / rmse_cnt : "nan")
      avg_maxabs = (maxabs_cnt ? maxabs_sum / maxabs_cnt : "nan")
      avg_cos = (cos_cnt ? cos_sum / cos_cnt : "nan")
      avg_rmae = (rmae_cnt ? rmae_sum / rmae_cnt : "nan")
      avg_js = (js_cnt ? js_sum / js_cnt : "nan")
      top1_pct = (t1_cnt ? (100.0 * t1_yes / t1_cnt) : "nan")
      top5_avg = (t5_cnt ? t5_sum / t5_cnt : "nan")
      gt1_pct = (gt1_cnt ? (100.0 * gt1_yes / gt1_cnt) : "nan")
      gt5_pct = (gt5_cnt ? (100.0 * gt5_yes / gt5_cnt) : "nan")
      printf "%d\t%d\t%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s",
        total, ok, fail, avg_lat, avg_mae, avg_rmse, avg_maxabs, avg_cos,
        avg_rmae, avg_js, top1_pct, top5_avg, gt1_pct, gt5_pct
    }' "${meta_dir}"/*.tsv 2>/dev/null)"
  [[ -n "${summary_line}" ]] || summary_line=$'0\t0\t0\tnan\tnan\tnan\tnan\tnan\tnan\tnan\tnan\tnan\tnan\tnan'

  exec {lfd}> "${format_log_b_lock}"
  flock "${lfd}"
  printf '%s\t%s\t%s\t%s\n' "${ts_ms}" "${key}" "${format_label}" "${summary_line}" >> "${format_log_b}"
  flock -u "${lfd}"
  exec {lfd}>&-
}

mark_progress() {
  local sample_idx="$1"
  local key="$2"
  local rc="$3"
  local should_write_format_log=0
  local format_label=""
  exec {lfd}> "${progress_lock}"
  flock "${lfd}"

  local completed_tasks_file="${tmp_root}/completed_tasks.txt"
  local completed_tasks
  completed_tasks="$(cat "${completed_tasks_file}")"
  completed_tasks=$((completed_tasks + 1))
  echo "${completed_tasks}" > "${completed_tasks_file}"

  format_label="$(progress_label_for_key "${key}")"
  if [[ "${task_progress_mode}" == "on" && "${progress}" -gt 0 && $((completed_tasks % progress)) -eq 0 ]]; then
    if [[ "${rc}" -ne 0 ]]; then
      echo "{task ${completed_tasks}/${task_total}} format=${format_label} sample=$((sample_idx + 1))/${limit} rc=${rc}"
    else
      echo "{task ${completed_tasks}/${task_total}} format=${format_label} sample=$((sample_idx + 1))/${limit}"
    fi
  fi

  local key_cnt_file="${tmp_root}/key_${key}.count"
  local key_failed_file="${tmp_root}/key_${key}.failed"
  local key_cnt=0
  if [[ -f "${key_cnt_file}" ]]; then
    key_cnt="$(cat "${key_cnt_file}")"
  fi
  key_cnt=$((key_cnt + 1))
  echo "${key_cnt}" > "${key_cnt_file}"
  if [[ "${rc}" -ne 0 ]]; then
    : > "${key_failed_file}"
  fi
  if [[ "${task_progress_mode}" == "off" && "${progress}" -gt 0 && "${key_cnt}" -eq "${limit}" ]]; then
    if [[ -f "${key_failed_file}" ]]; then
      echo "{done ${format_label}} rc=1"
    else
      echo "{done ${format_label}}"
    fi
  fi
  if [[ "${key_cnt}" -eq "${limit}" ]]; then
    should_write_format_log=1
  fi

  flock -u "${lfd}"
  exec {lfd}>&-
  if [[ "${should_write_format_log}" -eq 1 ]]; then
    write_format_log_b "${key}" "${format_label}"
  fi
}

worker_loop() {
  local task_line task_id sample_idx key so ty ent sample lbl
  local logf metaf rc lat c1 mae rmse maxabs cosine rmae js t1m t1m_bin t5ov
  local gline t1 t5 gt1_bin gt5_bin suffix qalign_csv output_alps_params
  local ptop_line pred_top1 pred_top5 dump_logits_path gt_label_out

  while task_line="$(claim_next_task)"; do
    IFS=$'\t' read -r task_id sample_idx key so ty ent sample lbl <<< "${task_line}"
    logf="${task_log_dir}/${task_id}.log"
    metaf="${meta_dir}/${task_id}.tsv"

    cmd=(
      "${runner}" "${so}"
      --shape "${shape_x}"
      --out-type "${ty}"
      --entry "${ent}"
      --warmup "${warmup}"
      --iters "${iters}"
      --quiet
    )
    if [[ "${image_mode}" -eq 1 ]]; then
      cmd+=(--image "${sample}" --image-preprocess-script "${image_preprocess_script}")
    else
      cmd+=("${sample}")
    fi
    if [[ "${baseline_mode}" != "none" ]]; then
      cmd+=(--cmp "${baseline_so}:f32:${baseline_entry}" --baseline "cmp:1")
    fi
    if [[ "${runner_supports_quire}" -eq 1 ]]; then
      cmd+=(--quire "${quire_mode}")
    fi
    if [[ "${no_benchmark}" -eq 1 ]]; then
      cmd+=(--no-benchmark)
    fi
    if [[ -n "${lbl}" ]]; then
      cmd+=(--label "${lbl}")
    fi

    suffix="${key#${model_name}-}"
    suffix="${suffix%.so}"
    qalign_csv="${qalign_csv_by_suffix[${suffix}]:-}"
    output_alps_params="${output_alps_params_by_suffix[${suffix}]:-}"

    probe_file=""
    if [[ "${QALIGN_RUNTIME_PROBE:-off}" == "on" ]]; then
      probe_file="${out_dir}/${model_name}-${suffix}.qalign_runtime_probe.csv"
    fi

    run_env=(env)
    if [[ -n "${qalign_csv}" ]]; then
      run_env+=("POSIT_QALIGN_FILE=${qalign_csv}")
    fi
    if [[ -n "${output_alps_params}" ]]; then
      run_env+=("POSIT_RUNTIME_OUTPUT_ALPS_FILE=${output_alps_params}")
    fi
    if [[ -n "${probe_file}" ]]; then
      run_env+=("POSIT_QALIGN_PROBE_FILE=${probe_file}")
      if [[ -n "${QALIGN_RUNTIME_PROBE_KEYS:-}" ]]; then
        run_env+=("POSIT_QALIGN_PROBE_KEYS=${QALIGN_RUNTIME_PROBE_KEYS}")
      fi
      if [[ -n "${QALIGN_RUNTIME_PROBE_LIMIT:-}" ]]; then
        run_env+=("POSIT_QALIGN_PROBE_LIMIT=${QALIGN_RUNTIME_PROBE_LIMIT}")
      fi
      if [[ -n "${QALIGN_RUNTIME_PROBE_SOURCE:-}" ]]; then
        run_env+=("POSIT_QALIGN_PROBE_SOURCE=${QALIGN_RUNTIME_PROBE_SOURCE}")
      fi
    fi
    dump_logits_path=""
    if [[ "${record_logits}" == "on" ]]; then
      dump_logits_path="${logits_dir}/${suffix}/sample_${sample_idx}.csv"
      mkdir -p "$(dirname "${dump_logits_path}")"
      cmd+=(--dump-logits "${dump_logits_path}")
    fi

    set +e
    if [[ "${timeout_sec}" -gt 0 ]]; then
      "${run_env[@]}" timeout "${timeout_sec}" "${cmd[@]}" > "${logf}" 2>&1
    else
      "${run_env[@]}" "${cmd[@]}" > "${logf}" 2>&1
    fi
    rc=$?
    set -e

    lat=""
    mae=""
    rmse=""
    maxabs=""
    cosine=""
    rmae=""
    js=""
    t1m_bin=""
    t5ov=""
    gt1_bin=""
    gt5_bin=""
    pred_top1="-1"
    pred_top5=""
    gt_label_out="${lbl:--}"

    if [[ "${rc}" -eq 0 ]]; then
      lat="$(sed -n 's/^MAIN type=[^ ]* avg=\([^ ]*\) us.*/\1/p' "${logf}" | head -n1)"
      c1="$(sed -n '/^  C1 target=/p' "${logf}" | head -n1)"
      if [[ -n "${c1}" ]]; then
        mae="$(extract_field "${c1}" "MAE")"
        rmse="$(extract_field "${c1}" "RMSE")"
        maxabs="$(extract_field "${c1}" "MaxAbs")"
        cosine="$(extract_field "${c1}" "Cosine")"
        rmae="$(extract_field "${c1}" "RMAE")"
        js="$(extract_field "${c1}" "JS")"
        t1m="$(extract_top1_match "${c1}")"
        if [[ "${t1m}" == "match" ]]; then
          t1m_bin="1"
        elif [[ "${t1m}" == "diff" ]]; then
          t1m_bin="0"
        fi
        t5ov="$(extract_top5_overlap "${c1}")"
      fi
      ptop_line="$(sed -n '/^  top1=-\{0,1\}[0-9]\+ top5=\[.*\]$/p' "${logf}" | head -n1)"
      if [[ -n "${ptop_line}" ]]; then
        pred_top1="$(extract_pred_top1 "${ptop_line}")"
        pred_top5="$(extract_pred_top5 "${ptop_line}")"
      fi
      if [[ -n "${lbl}" ]]; then
        gline="$(sed -n '/^  label=[0-9-]\+ top1_hit=\(yes\|no\) top5_hit=\(yes\|no\)$/p' "${logf}" | head -n1)"
        if [[ -n "${gline}" ]]; then
          t1="$(echo "${gline}" | sed -n 's/.* top1_hit=\(yes\|no\).*/\1/p')"
          t5="$(echo "${gline}" | sed -n 's/.* top5_hit=\(yes\|no\).*/\1/p')"
          if [[ "${t1}" == "yes" ]]; then
            gt1_bin="1"
          elif [[ "${t1}" == "no" ]]; then
            gt1_bin="0"
          fi
          if [[ "${t5}" == "yes" ]]; then
            gt5_bin="1"
          elif [[ "${t5}" == "no" ]]; then
            gt5_bin="0"
          fi
        fi
      fi
    else
      record_first_failure "${sample}" "${key}" "${rc}" "${logf}"
    fi

    # Keep all TSV fields non-empty so later shell parsing does not collapse
    # consecutive tabs and shift columns (which can break GT Top1/Top5 stats).
    lat="${lat:-nan}"
    mae="${mae:-nan}"
    rmse="${rmse:-nan}"
    maxabs="${maxabs:-nan}"
    cosine="${cosine:-nan}"
    rmae="${rmae:-nan}"
    js="${js:-nan}"
    t1m_bin="${t1m_bin:--}"
    t5ov="${t5ov:-nan}"
    gt1_bin="${gt1_bin:--}"
    gt5_bin="${gt5_bin:--}"

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${task_id}" "${sample_idx}" "${key}" "${rc}" "${lat}" "${mae}" "${rmse}" "${maxabs}" \
      "${cosine}" "${rmae}" "${js}" "${t1m_bin}" "${t5ov}" "${gt1_bin}" "${gt5_bin}" > "${metaf}"
    append_task_log_a "${task_id}" "${sample_idx}" "${key}" "${sample}" "${rc}" "${lat}" \
      "${mae}" "${rmse}" "${maxabs}" "${cosine}" "${rmae}" "${js}" "${t1m_bin}" "${t5ov}" \
      "${gt1_bin}" "${gt5_bin}"
    if [[ "${record_preds}" == "on" ]]; then
      append_prediction_log_c "${task_id}" "${sample_idx}" "${key}" "${sample}" "${rc}" \
        "${gt_label_out}" "${pred_top1}" "${pred_top5}" "${dump_logits_path:--}"
    fi

    mark_progress "${sample_idx}" "${key}" "${rc}"
  done
}

script_start_ns="$(date +%s%N)"
worker_count="${jobs}"
if [[ "${worker_count}" -gt "${task_total}" ]]; then
  worker_count="${task_total}"
fi
for ((w = 0; w < worker_count; ++w)); do
  worker_loop &
done
wait
script_end_ns="$(date +%s%N)"

declare -A lat_sum lat_cnt
declare -A mae_sum mae_cnt
declare -A rmse_sum rmse_cnt
declare -A maxabs_sum maxabs_cnt
declare -A cos_sum cos_cnt
declare -A rmae_sum rmae_cnt
declare -A js_sum js_cnt
declare -A top1_match_yes top1_match_cnt
declare -A top5ov_sum top5ov_cnt
declare -A gt_top1_yes gt_top1_cnt
declare -A gt_top5_yes gt_top5_cnt
declare -A sample_success sample_fail

for metaf in "${meta_dir}"/*.tsv; do
  [[ -f "${metaf}" ]] || continue
  IFS=$'\t' read -r task_id sample_idx key rc lat mae rmse maxabs cosine rmae js t1m_bin t5ov gt1_bin gt5_bin < "${metaf}"
  if [[ "${rc}" -eq 0 ]]; then
    sample_success["${sample_idx}"]=$(( ${sample_success["${sample_idx}"]:-0} + 1 ))
    add_sum lat_sum lat_cnt "${key}" "${lat}"
    if [[ "${baseline_mode}" != "none" ]]; then
      add_sum mae_sum mae_cnt "${key}" "${mae}"
      add_sum rmse_sum rmse_cnt "${key}" "${rmse}"
      add_sum maxabs_sum maxabs_cnt "${key}" "${maxabs}"
      add_sum cos_sum cos_cnt "${key}" "${cosine}"
      add_sum rmae_sum rmae_cnt "${key}" "${rmae}"
      add_sum js_sum js_cnt "${key}" "${js}"
      if [[ "${t1m_bin}" == "1" || "${t1m_bin}" == "0" ]]; then
        top1_match_cnt["${key}"]=$(( ${top1_match_cnt["${key}"]:-0} + 1 ))
        if [[ "${t1m_bin}" == "1" ]]; then
          top1_match_yes["${key}"]=$(( ${top1_match_yes["${key}"]:-0} + 1 ))
        fi
      fi
      add_sum top5ov_sum top5ov_cnt "${key}" "${t5ov}"
    fi
    if [[ "${gt1_bin}" == "1" || "${gt1_bin}" == "0" ]]; then
      gt_top1_cnt["${key}"]=$(( ${gt_top1_cnt["${key}"]:-0} + 1 ))
      if [[ "${gt1_bin}" == "1" ]]; then
        gt_top1_yes["${key}"]=$(( ${gt_top1_yes["${key}"]:-0} + 1 ))
      fi
    fi
    if [[ "${gt5_bin}" == "1" || "${gt5_bin}" == "0" ]]; then
      gt_top5_cnt["${key}"]=$(( ${gt_top5_cnt["${key}"]:-0} + 1 ))
      if [[ "${gt5_bin}" == "1" ]]; then
        gt_top5_yes["${key}"]=$(( ${gt_top5_yes["${key}"]:-0} + 1 ))
      fi
    fi
  else
    sample_fail["${sample_idx}"]=$(( ${sample_fail["${sample_idx}"]:-0} + 1 ))
  fi
done

used=0
failed=0
partial_fail=0
for ((i = 0; i < limit; ++i)); do
  success_n="${sample_success["${i}"]:-0}"
  fail_n="${sample_fail["${i}"]:-0}"
  if [[ "${success_n}" -gt 0 ]]; then
    used=$((used + 1))
  else
    failed=$((failed + 1))
  fi
  if [[ "${fail_n}" -gt 0 ]]; then
    partial_fail=$((partial_fail + 1))
  fi
done

total_sec="$(awk -v a="${script_start_ns}" -v b="${script_end_ns}" 'BEGIN{print (b-a)/1e9}')"
avg_sample_sec="$(avg_value "${total_sec}" "${used}")"

if [[ "${used}" -eq 0 ]]; then
  echo "ERROR: no valid sample parsed (failed=${failed})"
  exit 3
fi

{
  echo
  echo "Dataset summary"
  echo "  directory=${txt_dir}"
  echo "  samples_used=${used}"
  echo "  samples_failed=${failed}"
  echo "  samples_with_any_model_fail=${partial_fail}"
  echo "  shape=${shape_x}"
  echo "  jobs=${jobs}"
  echo "  warmup=${warmup}"
  echo "  iters=${iters}"
  if [[ "${timeout_sec}" -gt 0 ]]; then
    echo "  timeout_sec=${timeout_sec}"
  fi
  if [[ "${no_benchmark}" -eq 1 ]]; then
    echo "  mode=no-benchmark (single infer per task)"
  else
    echo "  mode=with-benchmark"
  fi
  if [[ "${label_enabled}" -eq 1 ]]; then
    echo "  label_map=${label_map}"
    echo "  samples_missing_label=${missing_label}"
  elif [[ "${image_mode}" -eq 1 ]]; then
    echo "  label_source=image_folder_order"
    echo "  samples_missing_label=${missing_label}"
  fi
  if [[ "${baseline_mode}" != "none" ]]; then
    echo "  baseline_so=${baseline_key}"
  else
    echo "  baseline_so=none"
  fi
  echo "  total_wall_time_sec=${total_sec}"
  echo "  avg_wall_time_sec_per_sample=${avg_sample_sec}"
  echo
  echo "Per-SO latency summary (avg us)"
  for key in "${keys[@]}"; do
    avg_lat="$(avg_value "${lat_sum[$key]:-0}" "${lat_cnt[$key]:-0}")"
    echo "  ${key}: ${avg_lat}"
  done
  if [[ "${baseline_mode}" != "none" ]]; then
    echo
    echo "Per-SO metric summary vs baseline (${baseline_key})"
    for key in "${keys[@]}"; do
      avg_mae="$(avg_value "${mae_sum[$key]:-0}" "${mae_cnt[$key]:-0}")"
      avg_rmse="$(avg_value "${rmse_sum[$key]:-0}" "${rmse_cnt[$key]:-0}")"
      avg_maxabs="$(avg_value "${maxabs_sum[$key]:-0}" "${maxabs_cnt[$key]:-0}")"
      avg_cos="$(avg_value "${cos_sum[$key]:-0}" "${cos_cnt[$key]:-0}")"
      avg_rmae="$(avg_value "${rmae_sum[$key]:-0}" "${rmae_cnt[$key]:-0}")"
      avg_js="$(avg_value "${js_sum[$key]:-0}" "${js_cnt[$key]:-0}")"
      t1_rate="$(avg_value "${top1_match_yes[$key]:-0}" "${top1_match_cnt[$key]:-0}")"
      t1_pct="$(awk -v r="${t1_rate}" 'BEGIN{print r*100}')"
      t5ov_avg="$(avg_value "${top5ov_sum[$key]:-0}" "${top5ov_cnt[$key]:-0}")"
      echo "  ${key}: MAE=${avg_mae} RMSE=${avg_rmse} MaxAbs=${avg_maxabs} Cosine=${avg_cos} RMAE=${avg_rmae} JS=${avg_js} Top1Match(%)=${t1_pct} Top5Overlap=${t5ov_avg}"
    done
  fi
  if [[ "${label_enabled}" -eq 1 || "${image_mode}" -eq 1 ]]; then
    echo
    echo "GT accuracy summary"
    for key in "${keys[@]}"; do
      gt1_rate="$(avg_value "${gt_top1_yes[$key]:-0}" "${gt_top1_cnt[$key]:-0}")"
      gt5_rate="$(avg_value "${gt_top5_yes[$key]:-0}" "${gt_top5_cnt[$key]:-0}")"
      gt1_pct="$(awk -v r="${gt1_rate}" 'BEGIN{print r*100}')"
      gt5_pct="$(awk -v r="${gt5_rate}" 'BEGIN{print r*100}')"
      echo "  ${key}: Top1(%)=${gt1_pct} Top5(%)=${gt5_pct}"
    done
  fi
} | tee -a "${log_file}"
