#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Chi chay warm-up + GAN cho bien the `full`, KHONG chay SCST.
#
# Ly do: SCST voi cau hinh hien tai bi reward hacking (oracle AMP cham chuoi
# toan W/F la 0.997 trong khi cham AMP that chi 0.360). Trong khi cho quyet dinh
# ve reward, phan warm-up + GAN van can cho moi phuong an, nen chay truoc.
#
# Checkpoint sinh ra dung duoc cho: bien the `full` (khi SCST duoc sua) va
# bien the `no_scst` trong ma tran ablation.
# ---------------------------------------------------------------------------
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-config/revision.yaml}"
# Thu muc dataset. Mac dinh la ban dung tu nguon tai that ngay 2026-09-10
# (config/dataset_manifest.json). Bo cu $DATA_DIR/ tu khai INTERIM,
# NOT FOR SUBMISSION nen khong duoc dung cho so lieu bai bao.
DATA_DIR="${DATA_DIR:-dataset/rebuilt_2026-09-10}"
# Nhan lan chay: driver bo qua buoc nao da co file, nen dung chung duong dan
# voi lan chay truoc se AM THAM tai dung checkpoint huan luyen tren dataset cu.
RUN_TAG="${RUN_TAG:-$(basename "$DATA_DIR")}"
ABL="results/ablations"
DATA_REPORT="${DATA_REPORT:-$DATA_DIR/dataset_build_report.json}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-10}"
GAN_EPOCHS="${GAN_EPOCHS:-50}"
SEEDS=(${SEEDS:-123 456 789 1337})

test -f "$DATA_REPORT" || { echo "Thieu $DATA_REPORT" >&2; exit 2; }

variant=full
for seed in "${SEEDS[@]}"; do
  echo "==================== full / seed ${seed} ===================="
  root="${ABL}/${variant}__${RUN_TAG}/seed${seed}"
  warmup="$root/warmup.pt"
  gan_dir="$root/gan"
  mkdir -p "$root" "$gan_dir"

  if [ ! -f "$warmup" ]; then
    echo "--- [1/2] MLE warm-up ---"
    "$PYTHON_BIN" scripts/mle_warmup.py --config "$CONFIG" --conditional \
      --esm-attention-contacts --contact-thresh 0.5 \
      --dataset-report "$DATA_REPORT" --epochs "$WARMUP_EPOCHS" \
      --batch-size 64 --seed "$seed" --out "$warmup"
  else
    echo "--- [1/2] warm-up da co, bo qua ---"
  fi

  if [ ! -f "$gan_dir/best_model.pt" ]; then
    echo "--- [2/2] Huan luyen GAN ---"
    "$PYTHON_BIN" train.py --config "$CONFIG" --conditional \
      --resume "$warmup" --dataset-report "$DATA_REPORT" \
      --epochs "$GAN_EPOCHS" --batch-size 64 --seed "$seed" \
      --checkpoint-dir "$gan_dir"
  else
    echo "--- [2/2] GAN da co, bo qua ---"
  fi
  echo "=== xong seed ${seed} (warm-up + GAN) ==="
done

echo "HOAN TAT phan GAN. Con lai: SCST sau khi chot cau hinh reward."
