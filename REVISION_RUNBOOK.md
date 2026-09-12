# Major-revision execution runbook

This runbook is for manuscript 7448295. A red `AUDIT HOLD` in the manuscript
means that the corresponding server artifacts have not passed verification.
Do not manually type numerical results into the LaTeX files.

## 1. Freeze code and environment

Install and test the revision code first. Complete the source manifest in Step
2, then commit both the code and manifest before training either oracle or any
generative model. Generated data and results are ignored by Git, so the working
tree must remain clean throughout all reportable model runs.

```bash
conda activate lightweight
python -m pip install -r requirements-revision.txt
git status --short
pytest tests -q
```

The pinned ESM-2 identity used by the revision is:

```text
facebook/esm2_t12_35M_UR50D
commit 6fbf070e65b0b7291e7bbcd451118c216cff79d8
```

## 2. Rebuild the AMP/non-AMP dataset

Copy `config/dataset_manifest.example.json` to
`config/dataset_manifest.json`. For each unchanged source file, replace every
placeholder with the database release, retrieval date, URL, license/access
terms, evidence rule, label definition, and SHA-256. The builder rejects a
source whose bytes do not match the declared hash.

For every positive source, choose exactly the applicable evidence route: either
declare a row-level `evidence_column` and a non-empty `evidence_accept` list, or
declare `prefiltered: true` and identify the database query or versioned script
in `prefilter_provenance`. A policy label alone is not accepted as evidence.

Prepare exact-deduplicated input for external homology clustering:

```bash
python -m peptidegen.data \
  --manifest config/dataset_manifest.json \
  --prepare-clustering-fasta dataset/clustering_input.fasta
```

Run the exact command declared in the manifest:

```bash
mmseqs easy-cluster dataset/clustering_input.fasta dataset/mmseqs \
  dataset/mmseqs_tmp --min-seq-id 0.4 -c 0.8 --cov-mode 0
```

Build the fixed splits from native MMseqs2 representative/member output:

```bash
python -m peptidegen.data \
  --manifest config/dataset_manifest.json \
  --cluster-tsv dataset/mmseqs_cluster.tsv \
  --cluster-format mmseqs_rep_member \
  --output-dir dataset/rebuilt_2026-09-10 --seed 42 --fractions 0.70 0.15 0.15
```

Do not continue if the builder reports an exact label conflict or a
mixed-label homology cluster. Those cases require a documented curation
decision, not automatic relabelling.

The dataset report stores paths relative to its own directory. Keep the
manifest, raw source files, cluster map, ledger, and split artifacts together
in the release archive so the recorded hashes can be revalidated after moving
the archive to another machine.

Commit the completed source manifest and all revision code now, then verify:

```bash
git status --short
git rev-parse HEAD
```

The first command must produce no output before Step 3.

## 3. Build and train the reward oracles

The AMP reward oracle can use the audited AMP/non-AMP splits. It is a training
signal only and must not be reused as independent AMP evidence.

```bash
python scripts/train_oracle.py amp \
  --train dataset/rebuilt_2026-09-10/train.csv \
  --val dataset/rebuilt_2026-09-10/validation.csv \
  --test dataset/rebuilt_2026-09-10/test.csv \
  --dataset-report dataset/rebuilt_2026-09-10/dataset_build_report.json \
  --model esm2_t12_35M_UR50D \
  --model-revision 6fbf070e65b0b7291e7bbcd451118c216cff79d8 \
  --seed 42 \
  --out results/oracles/amp/oracle_amp.pkl
```

Build hemolysis train/validation/test splits with their own source manifest and
homology clustering, then run:

```bash
python scripts/train_oracle.py hemo \
  --train dataset/hemolysis_rebuilt/train.csv \
  --val dataset/hemolysis_rebuilt/validation.csv \
  --test dataset/hemolysis_rebuilt/test.csv \
  --dataset-report dataset/hemolysis_rebuilt/dataset_build_report.json \
  --model esm2_t12_35M_UR50D \
  --model-revision 6fbf070e65b0b7291e7bbcd451118c216cff79d8 \
  --seed 42 \
  --out results/oracles/hemolysis/oracle_hemo.pkl
```

Each pickle must have a neighbouring `_report.json` marked
`reportable=true`. SCST verifies both the report and the pickle hash.

## 4. Run the full model and ablation matrix

The driver trains 12 variants under five independent seeds, samples exactly
1,000 sequences per resulting checkpoint, evaluates seed-level summaries, and
runs the conditional controllability sweeps for the full model.
It also binds every generated FASTA to a required `model_id`, checkpoint hash,
dataset-report hash, sampling configuration, and clean Git commit.

```bash
PYTHON_BIN=python \
ESM2_REVISION=6fbf070e65b0b7291e7bbcd451118c216cff79d8 \
WARMUP_EPOCHS=10 GAN_EPOCHS=50 SCST_STEPS=2000 NUM_GEN=1000 \
bash run_revision_experiments.sh
```

Do not lower epochs, sequence count, or seed count for manuscript runs.
`--max-samples`, heuristic rewards, post-generation II filtering, dirty Git
state, or missing provenance automatically make an artifact non-reportable.

## 5. Run ESMFold

Run every full-model FASTA. The frozen ESMFold identity is
`facebook/esmfold_v1` at commit
`75a3841ee059df2bf4d56688166c8fb459ddd97a`.

```bash
mkdir -p results/esmfold
for seed in 42 123 456 789 1337; do
  python scripts/esmfold_plddt.py \
    --input "results/ablations/full/gen/full_seed${seed}.fasta" \
    --output "results/esmfold/full_seed${seed}.csv" \
    --seed "$seed" \
    --model facebook/esmfold_v1 \
    --model-revision 75a3841ee059df2bf4d56688166c8fb459ddd97a
done

python scripts/aggregate_esmfold.py \
  --csv-dir results/esmfold \
  --fasta-dir results/ablations/full/gen \
  --expected-models full \
  --out results/esmfold_summary.json \
  --tex-out results/esmfold_summary.tex
```

## 6. Import genuinely independent predictors

Copy `config/external_validation_manifest.example.json` to
`config/external_validation_manifest.json`. Fill every version, citation,
command/service setting, run date, threshold, task definition, and independence
statement. Provide one raw per-sequence CSV for every seed for:

- an independent AMP predictor;
- ToxinPred3 as a general-toxicity screen only;
- a separate hemolysis-specific predictor.

Then run:

```bash
python scripts/evaluate_external_validators.py \
  --manifest config/external_validation_manifest.json \
  --out results/external_validation_report.json
```

The importer requires each prediction sequence multiset to exactly match its
reportable generated FASTA. These predictors must not have been used in
training, model selection, reward design, or hyperparameter tuning.

## 7. Create the immutable release and export paper tables

Reserve the archive DOI first. Create the ignored file
`config/release_manifest.json` from the example and fill the current Git commit,
archive DOI, licenses, data-availability statement, and environment-lock hash.
Set `baseline_policy` explicitly to `official_retraining`,
`inspired_controls_only`, or `literature_context_only`, and state in
`baseline_evidence` exactly which numerical baseline artifacts, if any, enter
the manuscript. Local inspired controls must not be named as official HydrAMP
or M3-CAD reproductions.
For the committed core lock in this audit, the current SHA-256 is:

```text
requirements-revision.txt
2fafdd28f03a2635af6b2047441c6c47f3f08b7ca8cd2aea7b0453973aca1aab
```

Run the final gates:

```bash
python scripts/audit_artifacts.py --root . \
  --out results/artifact_audit.json --require-complete

python scripts/export_verified_manuscript_tables.py \
  --root . --paper-dir ../../Revised/Audited
```

Only after both commands succeed should `main_clean.tex`,
`main_highlighted.tex`, and `respond_to_reviewer.tex` be compiled. Inspect the
final numerical prose as well as every table before submission; the exported
values are artifact-derived, but their scientific interpretation still
requires author approval.
