# Baseline implementation audit

The modules in this directory are local, architecture-inspired controls. They
are **not** verified copies of the official HydrAMP or M3-CAD implementations:

- `baselines/hydramp` is a project-local conditional VAE approximation.
- `baselines/m3cad` explicitly omits the original 3D voxel branch and replaces
  it with an eight-feature MLP.
- `baselines/esm2gen` and `baselines/pepgraphormer` are project-local controls.

Consequently, results from these modules must be labelled
“HydrAMP-inspired CVAE” and “M3-CAD-inspired multimodal CVAE”; they cannot be
reported as retrained HydrAMP/M3-CAD results. A direct named-method comparison
requires the official repository/version, a frozen commit hash, documented
adaptation to the audited train/validation/test split, checkpoint hashes, and
generation commands. Published numbers may be cited only as published results
on their original datasets and must not be presented as same-split results.
