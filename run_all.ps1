# =============================================================================
# run_all.ps1 - full LightweightPeptideGen pipeline (Phases 1-3)
# Windows PowerShell. Tune $Esm / epochs for your VRAM (defaults suit 8 GB).
# Run from the repo root:  powershell -ExecutionPolicy Bypass -File run_all.ps1
# Quick smoke (small subset, fast) to check the pipeline runs:  -File run_all.ps1 -Quick
# =============================================================================
param([switch]$Quick)
$ErrorActionPreference = "Stop"

$Esm    = "esm2_t12_35M_UR50D"   # 8 GB-friendly; use esm2_t33_650M_UR50D on a bigger GPU
$Seeds  = @(42, 123, 456, 789, 1337)
$NumGen = 1000
$WarmupEpochs = 5
$ScstSteps    = 2000
$Batch  = 64                     # GAN batch (fusion Transformer is heavier than the legacy GRU;
                                 # config's 8192 is for the GRU and OOMs an 8 GB card here)
# arrays so PowerShell expands each element as a separate argv token to python
$EsmCli = @()
$TrainExtra = @()

if ($Quick) {
    Write-Host "QUICK mode: tiny subset / few steps - pipeline smoke only (numbers NOT publishable)" -ForegroundColor Yellow
    $Esm = "esm2_t6_8M_UR50D"
    $Seeds = @(42, 123)
    $NumGen = 100
    $WarmupEpochs = 1
    $ScstSteps = 50
    $Batch = 16
    $EsmCli = @("--max-samples", "2000")
    $TrainExtra = @("--epochs", "2", "--max-samples", "2000")
}

New-Item -ItemType Directory -Force -Path checkpoints, results, "results/gen" | Out-Null

Write-Host "==> [A1b/A2] MLE warm-up (ESM-2 fusion + KNN<8A contact graph)" -ForegroundColor Cyan
python scripts/mle_warmup.py --config config/config.yaml --conditional `
  --esm-model $Esm --contact-graph --contact-thresh 0.5 `
  --epochs $WarmupEpochs --batch-size 256 $EsmCli --out checkpoints/warmup.pt
if ($LASTEXITCODE -ne 0) { throw "warm-up failed" }

Write-Host "==> [A1/A3] Adversarial training (resumes warm-up; esm_dim auto-read)" -ForegroundColor Cyan
# A3 objective is set in config.yaml: training.gan_loss (bce|wgan_gp), discrete_relaxation (softmax|gumbel)
python train.py --config config/config.yaml --conditional --resume checkpoints/warmup.pt --batch-size $Batch $TrainExtra
if ($LASTEXITCODE -ne 0) { throw "GAN training failed" }

Write-Host "==> [B6] Train AMP oracle (real test-set AUC)" -ForegroundColor Cyan
python scripts/train_oracle.py amp --train dataset/train.csv --val dataset/val.csv `
  --test dataset/test.csv --model $Esm --out results/oracle_amp.pkl --batch-size 128
if ($LASTEXITCODE -ne 0) { throw "oracle training failed" }
# Hemolysis oracle (needs external labelled set, columns sequence,label):
# python scripts/train_oracle.py hemo --train data/hemopi_train.csv --test data/hemopi_test.csv --model $Esm --out results/oracle_hemo.pkl

Write-Host "==> [A4] SCST multi-objective fine-tune" -ForegroundColor Cyan
$HemoArg = @()
if (Test-Path results/oracle_hemo.pkl) { $HemoArg = @("--hemo-oracle", "results/oracle_hemo.pkl") }
python scripts/scst_finetune.py --checkpoint checkpoints/best_model.pt `
  --amp-oracle results/oracle_amp.pkl $HemoArg `
  --steps $ScstSteps --batch-size 64 --lr 1e-5 `
  --w-stability 0.34 --w-amp 0.33 --w-hemolysis 0.33 --out checkpoints/scst_model.pt
if ($LASTEXITCODE -ne 0) { throw "SCST failed" }

Write-Host "==> Generate seeds x $NumGen sequences" -ForegroundColor Cyan
foreach ($s in $Seeds) {
  python generate.py --checkpoint checkpoints/scst_model.pt --num $NumGen --seed $s `
    --output "results/gen/LightweightPeptideGen_seed$s.fasta"
}

Write-Host "==> [B5/B6] Evaluate generated corpus (foldability + oracle, mean/std + significance)" -ForegroundColor Cyan
python scripts/evaluate_generated.py --gen-dir results/gen --foldability `
  --amp-oracle results/oracle_amp.pkl --esm-model $Esm --out results/benchmark.json

Write-Host "==> [C10] Controllability" -ForegroundColor Cyan
python scripts/controllability.py --checkpoint checkpoints/scst_model.pt `
  --train-csv dataset/train.csv `
  --features charge_at_pH7 instability_index gravy aromaticity aliphatic_index `
  --n-per 200 --out results/controllability.json

Write-Host "==> [C11] Lightweight: active-param report" -ForegroundColor Cyan
python scripts/backbone_ablation.py params --config config/config.yaml
# Optional (downloads 4 ESM models): backbone-size oracle ablation
# python scripts/backbone_ablation.py oracle --train dataset/train.csv --test dataset/test.csv --out results/backbone_oracle.json

Write-Host "DONE. Metrics written to results/*.json" -ForegroundColor Green
