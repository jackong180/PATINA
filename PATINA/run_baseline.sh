#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
OUTPUTS_ROOT="${REPO_ROOT}/outputs"
TMP_CONFIG_ROOT="${OUTPUTS_ROOT}/.run_baseline_configs"
DEFAULT_PRETRAIN="${PROJECT_ROOT}/checkpoints/InpaintingModel_gen.pth"
DEFAULT_EXP_PREFIX="PATINA"
DEFAULT_TEST_OUTPUT="${OUTPUTS_ROOT}/PATINA_final_test"

usage() {
  cat <<'EOF'
Usage:
  ./run_baseline.sh train [iters] [exp_name]
  ./run_baseline.sh test [run_dir|latest] [checkpoint_name] [output_dir]
  ./run_baseline.sh latest [exp_name]
  ./run_baseline.sh show-best [run_dir|latest]

Examples:
  ./run_baseline.sh train
  ./run_baseline.sh train 30000
  ./run_baseline.sh train 60000 PATINA-60000iter
  ./run_baseline.sh latest
  ./run_baseline.sh show-best latest
  ./run_baseline.sh test latest
  ./run_baseline.sh test outputs/PATINA-30000iter/<run_id> best.pth
EOF
}

activate_env() {
  # shellcheck disable=SC1091
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  conda activate "${PATINA_CONDA_ENV:-py310}"
}

default_exp_name() {
  local iters="$1"
  echo "${DEFAULT_EXP_PREFIX}-${iters}iter"
}

latest_run_dir() {
  local exp_name="${1:-$(default_exp_name 30000)}"
  local base_dir="${OUTPUTS_ROOT}/${exp_name}"
  if [[ ! -d "${base_dir}" ]]; then
    echo "No experiment directory found: ${base_dir}" >&2
    return 1
  fi
  ls -dt "${base_dir}"/* 2>/dev/null | head -n 1
}

write_temp_config() {
  local iters="$1"
  local exp_name="$2"
  local timestamp
  timestamp="$(date +%Y%m%d-%H%M%S)"
  local cfg_dir="${TMP_CONFIG_ROOT}/${exp_name}_${timestamp}_iter${iters}"
  mkdir -p "${cfg_dir}"
  cp "${PROJECT_ROOT}/checkpoints/config.yml" "${cfg_dir}/config.yml"
  CFG_DIR="${cfg_dir}" MAX_ITERS_OVERRIDE="${iters}" python - <<'PY'
import os
from pathlib import Path
import yaml

cfg_dir = Path(os.environ["CFG_DIR"])
config_path = cfg_dir / "config.yml"
data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
data["MAX_ITERS"] = int(os.environ["MAX_ITERS_OVERRIDE"])
config_path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
print(cfg_dir)
PY
}

ensure_checkpoint_exists() {
  local checkpoint_path="$1"
  if [[ ! -f "${checkpoint_path}" ]]; then
    echo "Checkpoint not found: ${checkpoint_path}" >&2
    return 1
  fi
}

cmd_train() {
  local iters="${1:-30000}"
  local exp_name="${2:-$(default_exp_name "${iters}")}"
  activate_env
  mkdir -p "${TMP_CONFIG_ROOT}"
  local cfg_dir
  cfg_dir="$(write_temp_config "${iters}" "${exp_name}" | tail -n 1)"
  local log_path="${OUTPUTS_ROOT}/${exp_name}_train_$(date +%Y%m%d-%H%M%S).log"

  echo "Using config dir: ${cfg_dir}"
  echo "Training log: ${log_path}"
  cd "${PROJECT_ROOT}"
  python main.py \
    --mode 1 \
    --path "${cfg_dir}" \
    --exp_name "${exp_name}" \
    --outputs_dir "${OUTPUTS_ROOT}" \
    --pretrain_from "${DEFAULT_PRETRAIN}" \
    2>&1 | tee "${log_path}"
}

resolve_run_dir_arg() {
  local arg="${1:-latest}"
  if [[ "${arg}" == "latest" ]]; then
    latest_run_dir "${2:-$(default_exp_name 30000)}"
  else
    echo "${arg}"
  fi
}

cmd_test() {
  local run_arg="${1:-latest}"
  local checkpoint_name="${2:-best.pth}"
  local output_dir="${3:-${DEFAULT_TEST_OUTPUT}}"
  activate_env
  local run_dir
  run_dir="$(resolve_run_dir_arg "${run_arg}" "${4:-$(default_exp_name 30000)}")"
  local checkpoint_path="${run_dir}/checkpoints/${checkpoint_name}"
  ensure_checkpoint_exists "${checkpoint_path}"

  local exp_name
  exp_name="$(basename "$(dirname "${run_dir}")")"
  local log_path="${OUTPUTS_ROOT}/$(basename "${run_dir}")_test_$(date +%Y%m%d-%H%M%S).log"

  echo "Run dir: ${run_dir}"
  echo "Checkpoint: ${checkpoint_path}"
  echo "Test log: ${log_path}"
  cd "${PROJECT_ROOT}"
  python main.py \
    --mode 2 \
    --path "${PROJECT_ROOT}/checkpoints" \
    --exp_name "${exp_name}" \
    --outputs_dir "${OUTPUTS_ROOT}" \
    --run_dir "${run_dir}" \
    --resume_from "${checkpoint_path}" \
    --output "${output_dir}" \
    2>&1 | tee "${log_path}"
}

cmd_latest() {
  local exp_name="${1:-$(default_exp_name 30000)}"
  latest_run_dir "${exp_name}"
}

cmd_show_best() {
  local run_arg="${1:-latest}"
  local run_dir
  run_dir="$(resolve_run_dir_arg "${run_arg}" "${2:-$(default_exp_name 30000)}")"
  local best_metric_path="${run_dir}/logs/best_metric.json"
  if [[ ! -f "${best_metric_path}" ]]; then
    echo "best_metric.json not found: ${best_metric_path}" >&2
    return 1
  fi
  cat "${best_metric_path}"
}

main() {
  local cmd="${1:-help}"
  shift || true

  case "${cmd}" in
    train)
      cmd_train "$@"
      ;;
    test)
      cmd_test "$@"
      ;;
    latest)
      cmd_latest "$@"
      ;;
    show-best)
      cmd_show_best "$@"
      ;;
    help|-h|--help)
      usage
      ;;
    *)
      echo "Unknown command: ${cmd}" >&2
      usage
      return 1
      ;;
  esac
}

main "$@"
