# Third-party resources

The MIT License in [`LICENSE`](LICENSE) covers **only the source code in this
repository**. It does not extend to the external databases and tools this work
depends on. None of them are redistributed here: each is cited and linked, and
its own terms govern it.

| Resource | Terms as we found them | How this work uses it |
|---|---|---|
| DRAMP 3.0 | free for academic use under terms stated by the provider | cited and linked |
| APD6 | free for academic use under terms stated by the provider | cited and linked |
| dbAMP 3.0 | free for academic use under terms stated by the provider | cited and linked |
| AMPBenchmark | **no licence declared** in its repository (checked 2026-09-10) | cited, not redistributed |
| UniProtKB/Swiss-Prot | CC BY 4.0 (UniProt Consortium) | cited and linked |
| Hemolytik2 | **GPL-3.0** | cited and linked, not redistributed |
| amPEPpy | **GPL-3.0** | invoked as an external tool, not vendored into this code |
| ToxinPred3 | per its own repository | invoked as an external package, not vendored |

## On the GPL-3.0 dependencies

Because no GPL-3.0 material is copied into, or linked against, the code in this
repository, the copyleft terms of Hemolytik2 and amPEPpy do not extend to it.
They are used as separate, externally installed tools, and their outputs are
recorded as data.

Anyone who bundles those resources into a derivative work must revisit that
analysis for themselves.

## On the training corpus

The training corpus is rebuilt from primary database downloads rather than
redistributed. `config/dataset_manifest.json` records, for every source, its
URL, release, retrieval date, licence and SHA-256, so the corpus can be
reconstructed from the named releases without this repository hosting any of
the source data.
