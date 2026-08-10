#!/usr/bin/env bash
# =============================================================================
# run_all.sh — full LightweightPeptideGen pipeline (Phases 1-3), Linux/Colab.
# On a >=16 GB GPU set ESM=esm2_t33_650M_UR50D for the paper-grade backbone.
#   bash run_all.sh
# =============================================================================
set -euo pipefail
ESM="${ESM:-esm2_t12_35M_UR50D}"   # paper-grade: ESM=esm2_t33_650M_UR50D
SEEDS=(42 123 456 789 1337)
NUM_GEN="${NUM_GEN:-1000}"
EPOCHS="${EPOCHS:-40}"             # GAN epochs (autoregressive gen is slow; raise if budget allows)
GSTEPS="${GSTEPS:-4}"             # generator steps/iter (config is 10; 4 is much faster, similar quality)
BATCH="${BATCH:-64}"             # fits a 16 GB T4 comfortably
WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
SCST_STEPS="${SCST_STEPS:-2000}"
mkdir -p checkpoints results results/gen

echo "==> [A1b/A2] MLE warm-up (ESM-2 fusion + KNN<8A contact graph)"
python scripts/mle_warmup.py --config config/config.yaml --conditional \
  --esm-model "$ESM" --contact-graph --contact-thresh 0.5 \
  --epochs "$WARMUP_EPOCHS" --batch-size 256 --out checkpoints/warmup.pt

echo "==> [A1/A3] Adversarial training (resumes warm-up; A3 objective from config)"
python train.py --config config/config.yaml --conditional --resume checkpoints/warmup.pt \
  --epochs "$EPOCHS" --g-steps "$GSTEPS" --batch-size "$BATCH"

echo "==> [B6] Train AMP oracle (real test-set AUC)"
python scripts/train_oracle.py amp --train dataset/train.csv --val dataset/val.csv \
  --test dataset/test.csv --model "$ESM" --out results/oracle_amp.pkl --batch-size 128
# Hemolysis (external HemoPI/DBAASP labels):
# python scripts/train_oracle.py hemo --train data/hemopi_train.csv --test data/hemopi_test.csv --model "$ESM" --out results/oracle_hemo.pkl

echo "==> [A4] SCST multi-objective fine-tune"
HEMO=""; [ -f results/oracle_hemo.pkl ] && HEMO="--hemo-oracle results/oracle_hemo.pkl"
python scripts/scst_finetune.py --checkpoint checkpoints/best_model.pt \
  --amp-oracle results/oracle_amp.pkl $HEMO \
  --steps "$SCST_STEPS" --batch-size 64 --lr 1e-5 \
  --w-stability 0.34 --w-amp 0.33 --w-hemolysis 0.33 --out checkpoints/scst_model.pt

echo "==> Generate 5 seeds x $NUM_GEN"
for s in "${SEEDS[@]}"; do
  python generate.py --checkpoint checkpoints/scst_model.pt --num "$NUM_GEN" --seed "$s" \
    --output "results/gen/LightweightPeptideGen_seed${s}.fasta"
done

echo "==> [B5/B6] Evaluate generated corpus (foldability + oracle, mean+/-std + significance)"
python scripts/evaluate_generated.py --gen-dir results/gen --foldability \
  --amp-oracle results/oracle_amp.pkl --esm-model "$ESM" --out results/benchmark.json

echo "==> [C10] Controllability"
python scripts/controllability.py --checkpoint checkpoints/scst_model.pt \
  --train-csv dataset/train.csv \
  --features charge_at_pH7 instability_index gravy aromaticity aliphatic_index \
  --n-per 200 --out results/controllability.json

echo "==> [C11] Lightweight: active-param report (+ optional backbone ablation)"
python scripts/backbone_ablation.py params --config config/config.yaml
# python scripts/backbone_ablation.py oracle --train dataset/train.csv --test dataset/test.csv --out results/backbone_oracle.json

echo "DONE. Metrics written to results/*.json"
