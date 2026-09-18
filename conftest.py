"""
Repo-root pytest configuration.

Provisions a small, fixed set of gitignored evidence/figure/checkpoint
result directories from the sibling conference repository
(`uav-defense-3d-resubmission`) into this repository's `results/`/`models/`
tree, IF (and only if) they are not already present locally. Runs once, at
conftest import time (see bottom of this file for why).

Why this exists
----------------
`results/` and `*.zip` are entirely gitignored (see .gitignore). A number of
tests read already-computed evidence CSVs/JSONs/figures via hardcoded
`PROJECT_ROOT / "results" / "<subdir>" / ...` paths defined inside production
scripts (e.g. `experiments/plot_paper_core_figures.py`,
`experiments/build_resubmission_evidence.py`), and some tests additionally
require a frozen checkpoint's path to be repo-relative (e.g.
`model_path.relative_to(PROJECT_ROOT)`, to cross-check it against
`results/final_nominal/model_hashes.json`). Those scripts' own path
constants are intentionally NOT modified here (some tests assert their exact
literal source text / equality), so the only safe way to make the data
available locally -- without regenerating it or touching any protocol/logic
-- is to copy the existing, already-produced artifacts into place before
the test session starts.

This fixture:
  * Never overwrites a directory that already exists locally (idempotent,
    and never clobbers a locally-regenerated/updated copy).
  * Only ever copies from the sibling conference repo -- the same repo
    that originally produced these artifacts -- so every provisioned file
    is byte-identical provenance, never a regenerated approximation.
  * Is a pure provisioning step (file copy), not a code path -- it changes
    no hashes, equations, protocols, or experiment logic. Frozen-checkpoint
    lookups also go through `experiments/artifact_paths.py`'s resolver as an
    independent second line of defense (it prefers a local copy, such as the
    one provisioned here, over the sibling repo).
"""

from __future__ import annotations

import shutil

from experiments.artifact_paths import PROJECT_ROOT, SIBLING_REPO_NAME

# Repo-relative result/checkpoint directories that some tests expect to
# already exist locally -- either because a test hard-codes an exact,
# repo-relative path (e.g. `model_path.relative_to(PROJECT_ROOT)`), or
# because it reads registry-listed evidence files under these trees. Each
# is copied verbatim, in full, from the sibling conference repo if missing
# locally. Frozen-checkpoint lookups also go through
# `experiments/artifact_paths.py`'s resolver (which prefers a local copy,
# such as the one provisioned here, over the sibling repo) as a second,
# independent line of defense.
_PROVISIONED_RESULT_DIRS: tuple[str, ...] = (
    "results/paper_evidence",
    "results/paper_figures",
    "results/lead_residual_expanded_final_5000",
    "results/final_nominal",
    "results/training_400k",
    "results/training_post_detection_400k",
    "models/lead_residual_post_detection",
    "results/corrections",
    "results/acquisition_ablation",
    "results/robustness",
    "results/audits",
)


def _provision_missing_result_dirs() -> None:
    sibling_root = PROJECT_ROOT.parent / SIBLING_REPO_NAME
    for relative_dir in _PROVISIONED_RESULT_DIRS:
        local_dir = PROJECT_ROOT / relative_dir
        if local_dir.exists():
            continue
        sibling_dir = sibling_root / relative_dir
        if not sibling_dir.exists():
            # Nothing we can do; leave missing so the affected test fails
            # with its normal, unambiguous FileNotFoundError/AssertionError.
            continue
        shutil.copytree(sibling_dir, local_dir)


# Run provisioning at conftest IMPORT time (module level), not inside a
# fixture. pytest imports conftest.py -- and therefore runs this -- before
# test collection begins, i.e. before any `experiments.*` module (which may
# build module-level, path-resolved dicts such as FROZEN_MODELS) is
# imported. A fixture (even a session-scoped autouse one) would only run
# during the first test's *setup* phase, which is too late: collection has
# already imported those modules and cached their resolved paths by then.
_provision_missing_result_dirs()
