# Changes made after the archived release

The archived release (commit `624c130`, DOI recorded in the manuscript) is the
exact code that produced every reported number. The commits listed here landed
**after** those runs and are present on `main` but not in the archive.

**None of them alters any reported value.** The per-file justification is below,
so the claim can be checked rather than taken on trust.

Reproducing the reported results: use the archived DOI snapshot.
Re-running from scratch, or extending the work: use `main`.

---

## Why these changes exist

Eight paths changed. Two fix defects in the reporting tools, found while
preparing the release and after the reported runs had already completed. Four
repair the architecture-inspired control models, which had never been run
against the rebuilt corpus. Two are housekeeping: the ignore rules and this
repository's licence file.

Fixing them before the release would have meant either shipping a release whose
commit does not match the artifacts, or discarding roughly 51 GPU-hours of
completed runs to regenerate identical numbers under a new commit hash. Neither
serves reproducibility, so the split is declared instead of hidden.

---

## Files changed, and why none affects a reported number

### `scripts/run_ablations.py` — table rendering only

The paired-contrast column printed `full - variant` under a heading that did not
state the direction, so a negative value — meaning the ablated variant scored
*higher* than the full model — could be read backwards. The column heading and
caption now state the convention and that the tests are two-sided.

*No numerical effect:* only two LaTeX strings in `latex_table()` changed. Every
statistic, p-value and Holm adjustment is computed by untouched code.

### `scripts/aggregate_controllability.py` — file discovery only

The aggregator required the exact filename `{model}_seed{N}.json`, but the
experiment drivers prefix that name with a run tag derived from the dataset
directory, so a rerun on new data cannot be confused with an earlier one. The
aggregator therefore failed with "missing controllability runs" at the very end
of a multi-day campaign while the files sat beside it under their tagged names.
It now resolves the tagged names, and refuses to aggregate if the files come
from different campaigns.

*No numerical effect:* only path resolution changed. The reportability check,
seed matching, feature-set comparison, training-CSV provenance check and all
statistics are untouched. The reported controllability numbers were produced by
passing the tag explicitly on the command line, which the archived code accepts.

### `baselines/train_baseline.py` — control models only

Four defects, all in code that predates the audit and had never been run against
the rebuilt corpus:

1. No training seed existed and none was set anywhere, so the script trained a
   single model and the downstream generator drew five *sampling* seeds from that
   one checkpoint. Those five values are pseudoreplicates of one run, not five
   independent runs, and the statistical protocol takes the independent training
   seed as the unit of analysis.
2. The dataset loader was not given a class filter, so the controls would have
   trained on the whole corpus rather than the antimicrobial subset the proposed
   model uses — a confound in the very comparison the controls exist to support.
3. Checkpoint and log paths were fixed per model, so five seeds would have
   overwritten one directory.
4. Models were constructed with an eight-dimensional conditioning vector while
   the ratified schema has six, which would fail on the first forward pass.

It now takes `--seed`, seeds `random`/`numpy`/`torch`/CUDA before the dataloaders
are built, reads the conditioning width from the data, writes checkpoints under a
per-seed directory, and records a provenance file next to each checkpoint stating
explicitly that control runs are not reportable artifacts.

*No numerical effect:* no control model had been trained when the release was
archived, so no reported number depends on this file.

### `baselines/common/data_utils.py` — control models only

Added the class-filter pass-through that defect 2 above required.

*No numerical effect:* used only by the control models.

### `scripts/gen_baseline.py` — control models only

Conditioning width was hard-coded to eight in three model constructors and in the
sampled condition vector. It now resolves the width from the provenance record
written beside the checkpoint, with an explicit override, and refuses to guess.

*No numerical effect:* used only by the control models.

### `baselines/evaluate_baseline.py` — control models only

Its default checkpoint path became unreachable once checkpoints moved under
per-seed directories. Added `--seed` and corrected the default.

*No numerical effect:* this script is not on the path that produces the reported
comparison, which runs every model through one identical evaluation protocol.

### `.gitignore` — housekeeping

Added the dated dataset build directories, the clustering inputs, and the
control models' output directory. Without the last of these, running the control
evaluation leaves untracked files in the tree, which marks the worktree dirty and
causes every artifact generated afterwards to be recorded as non-reportable.

*No numerical effect.*

### `LICENSE` — added after the release

The archived snapshot predates this file. The licence terms are stated in the
manuscript's Data and Code Availability section and apply to the archived code as
well; this file records them in the repository.

*No numerical effect.*

---

## Known remaining item

`baselines/evaluate_baseline.py` still constructs models with an
eight-dimensional conditioning vector in four places. It is deliberately left
alone: it is not on the path that produces the reported comparison, because it
computes a different metric vocabulary that overlaps the main evaluator's in only
two keys, and the between-model test intersects metric keys across models.
Changing a script that nothing depends on is a way to introduce faults, not
remove them.
