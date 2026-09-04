#!/usr/bin/env bash
# Reproducible major-revision experiment matrix. Run from the repository root.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-config/revision.yaml}"
DATA_REPORT="${DATA_REPORT:-dataset/rebuilt/dataset_build_report.json}"
AMP_ORACLE="${AMP_ORACLE:-results/oracles/amp/oracle_amp.pkl}"
HEMO_ORACLE="${HEMO_ORACLE:-results/oracles/hemolysis/oracle_hemo.pkl}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-10}"
GAN_EPOCHS="${GAN_EPOCHS:-50}"
SCST_STEPS="${SCST_STEPS:-2000}"
NUM_GEN="${NUM_GEN:-1000}"
CONTROLLABILITY_N_PER="${CONTROLLABILITY_N_PER:-200}"
ESM2_REVISION="${ESM2_REVISION:-6fbf070e65b0b7291e7bbcd451118c216cff79d8}"
SEEDS=(42 123 456 789 1337)

test -f "$DATA_REPORT" || { echo "Missing $DATA_REPORT; rebuild/audit data first" >&2; exit 2; }
test -f "$AMP_ORACLE" || { echo "Missing independent AMP reward oracle: $AMP_ORACLE" >&2; exit 2; }
test -f "$HEMO_ORACLE" || { echo "Missing hemolysis-specific reward oracle: $HEMO_ORACLE" >&2; exit 2; }

run_variant_seed() {
  local variant="$1"
  local seed="$2"
  local root="results/ablations/${variant}/seed${seed}"
  local warmup="$root/warmup.pt"
  local gan_dir="$root/gan"
  local parent="$gan_dir/best_model.pt"
  local final="$root/scst_model.pt"
  local conditional=(--conditional)
  local warmup_flags=(--esm-attention-contacts --contact-thresh 0.5)
  local gan_flags=()
  local scst_weights=(--w-ii-screen 0.34 --w-amp 0.33 --w-hemolysis 0.33)

  case "$variant" in
    no_esm2) warmup_flags=(--no-esm) ;;
    no_gatv2) warmup_flags=(--no-gat); gan_flags+=(--no-gat) ;;
    concat_fusion) warmup_flags+=(--fusion-type concat); gan_flags+=(--fusion-type concat) ;;
    no_conditioning) conditional=() ;;
    softmax) gan_flags+=(--discrete-relaxation softmax) ;;
    wgan_gp) gan_flags+=(--gan-loss wgan_gp) ;;
  esac

  mkdir -p "$root" "$gan_dir" "results/ablations/$variant/gen"
  "$PYTHON_BIN" scripts/mle_warmup.py --config "$CONFIG" "${conditional[@]}" \
    "${warmup_flags[@]}" --dataset-report "$DATA_REPORT" --epochs "$WARMUP_EPOCHS" \
    --batch-size 64 --seed "$seed" --out "$warmup"

  if [[ "$variant" == "no_adversarial" ]]; then
    parent="$warmup"
  else
    "$PYTHON_BIN" train.py --config "$CONFIG" "${conditional[@]}" \
      --resume "$warmup" "${gan_flags[@]}" --dataset-report "$DATA_REPORT" \
      --epochs "$GAN_EPOCHS" --batch-size 64 --seed "$seed" --checkpoint-dir "$gan_dir"
  fi

  if [[ "$variant" == "no_scst" ]]; then
    final="$parent"
  else
    case "$variant" in
      no_reward_stability) scst_weights=(--w-ii-screen 0 --w-amp 0.5 --w-hemolysis 0.5) ;;
      no_reward_amp) scst_weights=(--w-ii-screen 0.5 --w-amp 0 --w-hemolysis 0.5) ;;
      no_reward_hemolysis) scst_weights=(--w-ii-screen 0.5 --w-amp 0.5 --w-hemolysis 0) ;;
    esac
    "$PYTHON_BIN" scripts/scst_finetune.py --checkpoint "$parent" \
      --condition-csv dataset/rebuilt/train.csv --amp-oracle "$AMP_ORACLE" \
      --hemo-oracle "$HEMO_ORACLE" --oracle-id ESM2Oracle_AMP_reward \
      --hemo-oracle-id ESM2Oracle_Hemolysis_reward --steps "$SCST_STEPS" \
      --batch-size 64 --lr 1e-5 "${scst_weights[@]}" --seed "$seed" --out "$final"
  fi

  "$PYTHON_BIN" generate.py --checkpoint "$final" --model-id "$variant" \
    --num "$NUM_GEN" --seed "$seed" \
    --temperature 1.0 --top-p 0.9 --min-length 5 --max-length 50 \
    --output "results/ablations/$variant/gen/${variant}_seed${seed}.fasta"
  if [[ "$variant" == "full" ]]; then
    mkdir -p results/controllability
    "$PYTHON_BIN" scripts/controllability.py --checkpoint "$final" \
      --train-csv dataset/rebuilt/train.csv --n-per "$CONTROLLABILITY_N_PER" \
      --temperature 1.0 --top-p 0.9 --seed "$seed" \
      --out "results/controllability/full_seed${seed}.json"
  fi
}

# Primary ablations requested by the reviewers plus objective/relaxation controls.
VARIANTS=(
  full no_esm2 no_gatv2 concat_fusion no_conditioning no_adversarial no_scst
  softmax wgan_gp no_reward_stability no_reward_amp no_reward_hemolysis
)

for variant in "${VARIANTS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    # Reward-term ablations reuse the corresponding full GAN checkpoint, while
    # still receiving an independent SCST run for each training seed.
    if [[ "$variant" == no_reward_* ]]; then
      full_parent="results/ablations/full/seed${seed}/gan/best_model.pt"
      test -f "$full_parent" || { echo "Run full seed $seed before $variant" >&2; exit 2; }
      root="results/ablations/${variant}/seed${seed}"
      mkdir -p "$root" "results/ablations/$variant/gen"
      weights=(--w-ii-screen 0.34 --w-amp 0.33 --w-hemolysis 0.33)
      [[ "$variant" == "no_reward_stability" ]] && weights=(--w-ii-screen 0 --w-amp 0.5 --w-hemolysis 0.5)
      [[ "$variant" == "no_reward_amp" ]] && weights=(--w-ii-screen 0.5 --w-amp 0 --w-hemolysis 0.5)
      [[ "$variant" == "no_reward_hemolysis" ]] && weights=(--w-ii-screen 0.5 --w-amp 0.5 --w-hemolysis 0)
      "$PYTHON_BIN" scripts/scst_finetune.py --checkpoint "$full_parent" \
        --condition-csv dataset/rebuilt/train.csv --amp-oracle "$AMP_ORACLE" \
        --hemo-oracle "$HEMO_ORACLE" --oracle-id ESM2Oracle_AMP_reward \
        --hemo-oracle-id ESM2Oracle_Hemolysis_reward --steps "$SCST_STEPS" \
        --batch-size 64 --lr 1e-5 "${weights[@]}" --seed "$seed" --out "$root/scst_model.pt"
      "$PYTHON_BIN" generate.py --checkpoint "$root/scst_model.pt" \
        --model-id "$variant" --num "$NUM_GEN" \
        --seed "$seed" --temperature 1.0 --top-p 0.9 --min-length 5 --max-length 50 \
        --output "results/ablations/$variant/gen/${variant}_seed${seed}.fasta"
    else
      run_variant_seed "$variant" "$seed"
    fi
  done
  "$PYTHON_BIN" scripts/evaluate_generated.py \
    --gen-dir "results/ablations/$variant/gen" --reference "$variant" \
    --expected-models "$variant" --expected-n "$NUM_GEN" \
    --train-fasta dataset/rebuilt/train.fasta --sequence-plausibility \
    --esm-model-revision "$ESM2_REVISION" \
    --out "results/ablations/$variant/benchmark.json"
done

"$PYTHON_BIN" scripts/run_ablations.py --manifest config/ablation_manifest.json
"$PYTHON_BIN" scripts/aggregate_controllability.py
echo "Core experiment matrix complete. Run external validators and ESMFold import/aggregation next."
