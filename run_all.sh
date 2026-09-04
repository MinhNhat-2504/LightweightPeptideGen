#!/usr/bin/env bash
# Backward-compatible entry point for the audited major-revision pipeline.
set -euo pipefail
exec bash "$(dirname "$0")/run_revision_experiments.sh" "$@"
