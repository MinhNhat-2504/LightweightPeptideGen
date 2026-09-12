#!/usr/bin/env bash
# Reproducible major-revision experiment matrix. Run from the repository root.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-config/revision.yaml}"
# Thu muc dataset. Mac dinh la ban dung tu nguon tai that ngay 2026-09-10
# (config/dataset_manifest.json). Bo cu dataset/rebuilt/ tu khai INTERIM,
# NOT FOR SUBMISSION nen khong duoc dung cho so lieu bai bao.
DATA_DIR="${DATA_DIR:-dataset/rebuilt_2026-09-10}"
# Nhan lan chay: driver bo qua buoc nao da co file, nen dung chung duong dan
# voi lan chay truoc se AM THAM tai dung checkpoint huan luyen tren dataset cu.
RUN_TAG="${RUN_TAG:-$(basename "$DATA_DIR")}"
ABL="results/ablations"
DATA_REPORT="${DATA_REPORT:-$DATA_DIR/dataset_build_report.json}"
AMP_ORACLE="${AMP_ORACLE:-results/oracles/amp_2026-09-10/oracle_amp.pkl}"
HEMO_ORACLE="${HEMO_ORACLE:-results/oracles/hemolysis/oracle_hemo.pkl}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-10}"
GAN_EPOCHS="${GAN_EPOCHS:-50}"
SCST_STEPS="${SCST_STEPS:-2000}"
# Neo KL ve chinh sach GAN goc. BAT BUOC: khong co no, SCST khai thac oracle AMP
# (cham chuoi toan W/F la 0.997 nhung cham AMP that chi 0.360) va sup che do.
KL_COEF="${KL_COEF:-0.1}"
# SCST batch = 16, KHONG phai 64. Day la con so bai bao khai trong bang cau hinh
# huan luyen. De mac dinh 64 se cho ra ket qua khong khop bai da viet.
SCST_BATCH="${SCST_BATCH:-16}"
NUM_GEN="${NUM_GEN:-1000}"
CONTROLLABILITY_N_PER="${CONTROLLABILITY_N_PER:-200}"
ESM2_REVISION="${ESM2_REVISION:-6fbf070e65b0b7291e7bbcd451118c216cff79d8}"
SEEDS=(42 123 456 789 1337)

test -f "$DATA_REPORT" || { echo "Missing $DATA_REPORT; rebuild/audit data first" >&2; exit 2; }
test -f "$AMP_ORACLE" || { echo "Missing independent AMP reward oracle: $AMP_ORACLE" >&2; exit 2; }
# Hemolysis KHONG con nam trong reward SCST (quyet dinh tac gia 2026-09). No duoc
# danh gia sau bang predictor doc lap held-out huan luyen tren Hemolytik2, nen
# driver khong con doi hoi oracle hemolysis lam reward.

run_variant_seed() {
  local variant="$1"
  local seed="$2"
  local root="${ABL}/${variant}__${RUN_TAG}/seed${seed}"
  local warmup="$root/warmup.pt"
  local gan_dir="$root/gan"
  local parent="$gan_dir/best_model.pt"
  local final="$root/scst_model.pt"
  local conditional=(--conditional)
  local warmup_flags=(--esm-attention-contacts --contact-thresh 0.5)
  local gan_flags=()
  local scst_weights=(--w-ii-screen 0.5 --w-amp 0.5 --w-hemolysis 0)

  case "$variant" in
    no_esm2) warmup_flags=(--no-esm) ;;
    no_gatv2) warmup_flags=(--no-gat); gan_flags+=(--no-gat) ;;
    concat_fusion) warmup_flags+=(--fusion-type concat); gan_flags+=(--fusion-type concat) ;;
    no_conditioning) conditional=() ;;
    softmax) gan_flags+=(--discrete-relaxation softmax) ;;
    wgan_gp) gan_flags+=(--gan-loss wgan_gp) ;;
  esac

  mkdir -p "$root" "$gan_dir" "${ABL}/${variant}__${RUN_TAG}/gen"
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
      no_reward_stability) scst_weights=(--w-ii-screen 0 --w-amp 1.0 --w-hemolysis 0) ;;
      no_reward_amp) scst_weights=(--w-ii-screen 1.0 --w-amp 0 --w-hemolysis 0) ;;
    esac
    "$PYTHON_BIN" scripts/scst_finetune.py --checkpoint "$parent" \
      --condition-csv "$DATA_DIR/train.csv" --amp-oracle "$AMP_ORACLE" \
      --oracle-id ESM2Oracle_AMP_reward --steps "$SCST_STEPS" \
      --batch-size "$SCST_BATCH" --lr 1e-5 --kl-coef "$KL_COEF" \
      "${scst_weights[@]}" --seed "$seed" --out "$final"
  fi

  "$PYTHON_BIN" generate.py --checkpoint "$final" --model-id "$variant" \
    --num "$NUM_GEN" --seed "$seed" \
    --temperature 1.0 --top-p 0.9 --min-length 5 --max-length 50 \
    --output "${ABL}/${variant}__${RUN_TAG}/gen/${variant}_seed${seed}.fasta"
  if [[ "$variant" == "full" ]]; then
    mkdir -p results/controllability
    "$PYTHON_BIN" scripts/controllability.py --checkpoint "$final" \
      --train-csv "$DATA_DIR/train.csv" --n-per "$CONTROLLABILITY_N_PER" \
      --temperature 1.0 --top-p 0.9 --seed "$seed" \
      --out "results/controllability/full_seed${seed}.json"
  fi
}

# Primary ablations requested by the reviewers plus objective/relaxation controls.
VARIANTS=(
  full no_esm2 no_gatv2 concat_fusion no_conditioning no_adversarial no_scst
  softmax wgan_gp no_reward_stability no_reward_amp
  # no_reward_hemolysis da bo: hemolysis khong con nam trong reward SCST nen
  # bien the nay se giong het full. Bang ablation trong bai phai sua tu
  # 12 bien the xuong 11.
)

for variant in "${VARIANTS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    # Reward-term ablations reuse the corresponding full GAN checkpoint, while
    # still receiving an independent SCST run for each training seed.
    if [[ "$variant" == no_reward_* ]]; then
      full_parent="${ABL}/full__${RUN_TAG}/seed${seed}/gan/best_model.pt"
      test -f "$full_parent" || { echo "Run full seed $seed before $variant" >&2; exit 2; }
      root="${ABL}/${variant}__${RUN_TAG}/seed${seed}"
      mkdir -p "$root" "${ABL}/${variant}__${RUN_TAG}/gen"
      weights=(--w-ii-screen 0.5 --w-amp 0.5 --w-hemolysis 0)
      [[ "$variant" == "no_reward_stability" ]] && weights=(--w-ii-screen 0 --w-amp 1.0 --w-hemolysis 0)
      [[ "$variant" == "no_reward_amp" ]] && weights=(--w-ii-screen 1.0 --w-amp 0 --w-hemolysis 0)
      "$PYTHON_BIN" scripts/scst_finetune.py --checkpoint "$full_parent" \
        --condition-csv "$DATA_DIR/train.csv" --amp-oracle "$AMP_ORACLE" \
        --oracle-id ESM2Oracle_AMP_reward --steps "$SCST_STEPS" \
        --batch-size "$SCST_BATCH" --lr 1e-5 --kl-coef "$KL_COEF" \
        "${weights[@]}" --seed "$seed" --out "$root/scst_model.pt"
      "$PYTHON_BIN" generate.py --checkpoint "$root/scst_model.pt" \
        --model-id "$variant" --num "$NUM_GEN" \
        --seed "$seed" --temperature 1.0 --top-p 0.9 --min-length 5 --max-length 50 \
        --output "${ABL}/${variant}__${RUN_TAG}/gen/${variant}_seed${seed}.fasta"
    else
      run_variant_seed "$variant" "$seed"
    fi
  done
  "$PYTHON_BIN" scripts/evaluate_generated.py \
    --gen-dir "${ABL}/${variant}__${RUN_TAG}/gen" --reference "$variant" \
    --expected-models "$variant" --expected-n "$NUM_GEN" \
    --train-fasta "$DATA_DIR/train.fasta" --sequence-plausibility \
    --esm-model-revision "$ESM2_REVISION" \
    --out "${ABL}/${variant}__${RUN_TAG}/benchmark.json"
done

"$PYTHON_BIN" scripts/run_ablations.py --manifest config/ablation_manifest.json
"$PYTHON_BIN" scripts/aggregate_controllability.py
echo "Core experiment matrix complete. Run external validators and ESMFold import/aggregation next."
