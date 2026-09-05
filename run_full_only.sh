#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Giai doan 1 cua ban sua: CHI chay bien the `full`, du 5 seed.
#
# Khac voi run_revision_experiments.sh o dung hai diem, va ca hai deu duoc khai
# bao minh bach trong bai:
#   1. Khong yeu cau oracle hemolysis. Trong so reward hemolysis dat = 0, phan
#      con lai chia deu cho II-screen va AMP (0.5 / 0.5). Ly do: chua co du lieu
#      hemolysis co endpoint va nguong duoc tac gia xac dinh.
#   2. Chi chay bien the `full`. Cac bien the ablation chay sau bang driver goc.
#
# Moi tham so khac (epoch, seed, sampling, ESM revision) giu nguyen nhu driver goc.
# ---------------------------------------------------------------------------
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-config/revision.yaml}"
DATA_REPORT="${DATA_REPORT:-dataset/rebuilt/dataset_build_report.json}"
AMP_ORACLE="${AMP_ORACLE:-results/oracles/amp/oracle_amp.pkl}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-10}"
GAN_EPOCHS="${GAN_EPOCHS:-50}"
SCST_STEPS="${SCST_STEPS:-2000}"
NUM_GEN="${NUM_GEN:-1000}"
CONTROLLABILITY_N_PER="${CONTROLLABILITY_N_PER:-200}"
ESM2_REVISION="${ESM2_REVISION:-6fbf070e65b0b7291e7bbcd451118c216cff79d8}"
SEEDS=(${SEEDS:-42 123 456 789 1337})

test -f "$DATA_REPORT" || { echo "Thieu $DATA_REPORT" >&2; exit 2; }
test -f "$AMP_ORACLE"  || { echo "Thieu AMP oracle: $AMP_ORACLE" >&2; exit 2; }

variant=full

for seed in "${SEEDS[@]}"; do
  echo "==================== full / seed ${seed} ===================="
  root="results/ablations/${variant}/seed${seed}"
  warmup="$root/warmup.pt"
  gan_dir="$root/gan"
  final="$root/scst_model.pt"
  mkdir -p "$root" "$gan_dir" "results/ablations/${variant}/gen"

  if [ ! -f "$warmup" ]; then
    echo "--- [1/4] MLE warm-up ---"
    "$PYTHON_BIN" scripts/mle_warmup.py --config "$CONFIG" --conditional \
      --esm-attention-contacts --contact-thresh 0.5 \
      --dataset-report "$DATA_REPORT" --epochs "$WARMUP_EPOCHS" \
      --batch-size 64 --seed "$seed" --out "$warmup"
  else
    echo "--- [1/4] warm-up da co, bo qua ---"
  fi

  if [ ! -f "$gan_dir/best_model.pt" ]; then
    echo "--- [2/4] Huan luyen GAN (buoc lau nhat) ---"
    "$PYTHON_BIN" train.py --config "$CONFIG" --conditional \
      --resume "$warmup" --dataset-report "$DATA_REPORT" \
      --epochs "$GAN_EPOCHS" --batch-size 64 --seed "$seed" \
      --checkpoint-dir "$gan_dir"
  else
    echo "--- [2/4] GAN da co, bo qua ---"
  fi

  if [ ! -f "$final" ]; then
    echo "--- [3/4] SCST (khong co hemolysis) ---"
    "$PYTHON_BIN" scripts/scst_finetune.py --checkpoint "$gan_dir/best_model.pt" \
      --condition-csv dataset/rebuilt/train.csv --amp-oracle "$AMP_ORACLE" \
      --oracle-id ESM2Oracle_AMP_reward --steps "$SCST_STEPS" \
      --batch-size 64 --lr 1e-5 \
      --w-ii-screen 0.5 --w-amp 0.5 --w-hemolysis 0 \
      --seed "$seed" --out "$final"
  else
    echo "--- [3/4] SCST da co, bo qua ---"
  fi

  echo "--- [4/4] Sinh ${NUM_GEN} chuoi ---"
  "$PYTHON_BIN" generate.py --checkpoint "$final" --model-id "$variant" \
    --num "$NUM_GEN" --seed "$seed" \
    --temperature 1.0 --top-p 0.9 --min-length 5 --max-length 50 \
    --output "results/ablations/${variant}/gen/${variant}_seed${seed}.fasta"

  echo "--- Tinh dieu khien duoc ---"
  mkdir -p results/controllability
  "$PYTHON_BIN" scripts/controllability.py --checkpoint "$final" \
    --train-csv dataset/rebuilt/train.csv --n-per "$CONTROLLABILITY_N_PER" \
    --temperature 1.0 --top-p 0.9 --seed "$seed" \
    --out "results/controllability/full_seed${seed}.json"

  echo "=== xong seed ${seed} ==="
done

echo "==================== Danh gia gop 5 seed ===================="
"$PYTHON_BIN" scripts/evaluate_generated.py \
  --gen-dir "results/ablations/${variant}/gen" --reference "$variant" \
  --expected-models "$variant" --expected-n "$NUM_GEN" \
  --train-fasta dataset/rebuilt/train.fasta --sequence-plausibility \
  --esm-model-revision "$ESM2_REVISION" \
  --out "results/ablations/${variant}/benchmark.json"

echo "HOAN TAT. Ket qua chinh: results/ablations/full/benchmark.json"
