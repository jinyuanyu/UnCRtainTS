#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'EOF'
用法: auto_pipeline.sh [选项]
自动化完成数据下载/预处理并运行训练或推理。

常用参数（亦可通过环境变量覆盖）:
  --data-dir PATH         数据根目录，默认 $REPO_ROOT/data_store
  --region NAME           区域 [all|africa|america|asiaEast|asiaWest|europa]，默认 all
  --with-s1               同时下载S1
  --with-mono             同时下载单时相SEN12MS-CR
  --skip-download         跳过下载（需提前准备好数据目录）
  --skip-precompute       跳过预计算cloud stats
  --mode MODE             train 或 test，默认 train
  --experiment NAME       实验名称，默认 auto_run
  --input-t N             输入时间点数量，默认 3
  --batch-size N          训练batch size，默认 4
  --epochs N              训练轮数，默认 20
  --device DEV            设备(cuda/cpu)，默认 cuda
  --python PATH           Python解释器，默认使用系统 python
  --precompute-dir PATH   预计算输出目录，默认 util/precomputed
  --res-dir PATH          训练结果目录，默认 model/results
  --weights-dir PATH      测试权重目录，默认同 --res-dir
  --resume-at N           测试时加载指定epoch的checkpoint
  --load-config PATH      测试时加载自定义conf.json
  --max-samples N         预计算最大样本数，默认 1e9
  --num-workers N         预计算DataLoader并行数，默认 0
  -h, --help              查看帮助
EOF
}

DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data_store}"
REGION="all"
DOWNLOAD_MT="yes"
DOWNLOAD_MONO="no"
DOWNLOAD_S1="no"
SKIP_DOWNLOAD="false"
SKIP_PRECOMPUTE="false"
MODE="train"
EXPERIMENT_NAME="auto_run"
INPUT_T=3
BATCH_SIZE=4
EPOCHS=20
DEVICE="cuda"
PYTHON_BIN="${PYTHON_BIN:-python}"
PRECOMPUTE_DIR="${PRECOMPUTE_DIR:-$REPO_ROOT/util/precomputed}"
RES_DIR="${RES_DIR:-$REPO_ROOT/model/results}"
WEIGHTS_DIR="$RES_DIR"
INFER_RES_DIR="${INFER_RES_DIR:-$REPO_ROOT/model/inference}"
RESUME_AT=""
LOAD_CONFIG=""
MAX_SAMPLES="${MAX_SAMPLES:-1000000000}"
NUM_WORKERS="${NUM_WORKERS:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-dir) DATA_ROOT="$2"; shift 2;;
    --region) REGION="$2"; shift 2;;
    --with-s1) DOWNLOAD_S1="yes"; shift 1;;
    --with-mono) DOWNLOAD_MONO="yes"; shift 1;;
    --skip-download) SKIP_DOWNLOAD="true"; shift 1;;
    --skip-precompute) SKIP_PRECOMPUTE="true"; shift 1;;
    --mode) MODE="$2"; shift 2;;
    --experiment) EXPERIMENT_NAME="$2"; shift 2;;
    --input-t) INPUT_T="$2"; shift 2;;
    --batch-size) BATCH_SIZE="$2"; shift 2;;
    --epochs) EPOCHS="$2"; shift 2;;
    --device) DEVICE="$2"; shift 2;;
    --python) PYTHON_BIN="$2"; shift 2;;
    --precompute-dir) PRECOMPUTE_DIR="$2"; shift 2;;
    --res-dir) RES_DIR="$2"; WEIGHTS_DIR="$2"; shift 2;;
    --weights-dir) WEIGHTS_DIR="$2"; shift 2;;
    --resume-at) RESUME_AT="$2"; shift 2;;
    --load-config) LOAD_CONFIG="$2"; shift 2;;
    --max-samples) MAX_SAMPLES="$2"; shift 2;;
    --num-workers) NUM_WORKERS="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "未知参数: $1"; usage; exit 1;;
  esac
done

SEN12MSCRTS_PATH="$DATA_ROOT/SEN12MSCRTS"
SEN12MSCR_PATH="$DATA_ROOT/SEN12MSCR"
mkdir -p "$DATA_ROOT"

download_data() {
  local mt_answer mono_answer s1_answer
  mt_answer=$([ "$DOWNLOAD_MT" == "yes" ] && echo "y" || echo "n")
  mono_answer=$([ "$DOWNLOAD_MONO" == "yes" ] && echo "y" || echo "n")
  s1_answer=$([ "$DOWNLOAD_S1" == "yes" ] && echo "y" || echo "n")

  if [[ "$DOWNLOAD_MT" == "yes" ]]; then
    printf "%s\n%s\n%s\n%s\n%s\n" "$mt_answer" "$REGION" "$mono_answer" "$s1_answer" "$DATA_ROOT" | bash "$REPO_ROOT/util/dl_data.sh"
  else
    printf "%s\n%s\n%s\n%s\n" "$mt_answer" "$mono_answer" "$s1_answer" "$DATA_ROOT" | bash "$REPO_ROOT/util/dl_data.sh"
  fi
}

precompute_stats() {
  mkdir -p "$PRECOMPUTE_DIR"
  for split in train val test; do
    local vary="random"
    [[ "$split" == "test" ]] && vary="fixed"
    "$PYTHON_BIN" "$REPO_ROOT/util/pre_compute_data_samples.py" \
      --root "$SEN12MSCRTS_PATH" \
      --split "$split" \
      --region "$REGION" \
      --sample-type generic \
      --input-t "$INPUT_T" \
      --export-data-path "$PRECOMPUTE_DIR" \
      --vary "$vary" \
      --n-epochs 1 \
      --max-samples "$MAX_SAMPLES" \
      --num-workers "$NUM_WORKERS"
  done
}

run_train() {
  "$PYTHON_BIN" "$REPO_ROOT/model/train_reconstruct.py" \
    --experiment_name "$EXPERIMENT_NAME" \
    --root1 "$SEN12MSCRTS_PATH" \
    --root2 "$SEN12MSCRTS_PATH" \
    --root3 "$SEN12MSCR_PATH" \
    --precomputed "$PRECOMPUTE_DIR" \
    --region "$REGION" \
    --input_t "$INPUT_T" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --device "$DEVICE" \
    --res_dir "$RES_DIR"
}

run_test() {
  cmd=(
    "$PYTHON_BIN" "$REPO_ROOT/model/test_reconstruct.py"
    --experiment_name "$EXPERIMENT_NAME"
    --weight_folder "$WEIGHTS_DIR"
    --root1 "$SEN12MSCRTS_PATH"
    --root2 "$SEN12MSCRTS_PATH"
    --root3 "$SEN12MSCR_PATH"
    --precomputed "$PRECOMPUTE_DIR"
    --region "$REGION"
    --input_t "$INPUT_T"
    --device "$DEVICE"
    --res_dir "$INFER_RES_DIR"
  )
  [[ -n "$RESUME_AT" ]] && cmd+=(--resume_at "$RESUME_AT")
  [[ -n "$LOAD_CONFIG" ]] && cmd+=(--load_config "$LOAD_CONFIG")
  "${cmd[@]}"
}

echo "==== UnCRtainTS 自动流程 ===="
echo "工作目录: $REPO_ROOT"
echo "数据目录: $DATA_ROOT"

if [[ "$SKIP_DOWNLOAD" != "true" ]]; then
  echo "[1/4] 下载数据 ..."
  download_data
else
  echo "[1/4] 跳过下载"
fi

if [[ "$SKIP_PRECOMPUTE" != "true" ]]; then
  echo "[2/4] 预计算cloud stats ..."
  precompute_stats
else
  echo "[2/4] 跳过预计算"
fi

echo "[3/4] 运行程序 ($MODE) ..."
case "$MODE" in
  train) run_train;;
  test) run_test;;
  *) echo "不支持的mode: $MODE"; exit 1;;
esac

echo "[4/4] 完成。"

