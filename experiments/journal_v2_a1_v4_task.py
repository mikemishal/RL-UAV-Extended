"""Restartable runner for the discrete-tactical-interface (action-v4) task. Safe to re-run after any interruption:
verified stages are skipped, the first unverified stage is re-run. See experiments/journal_v2_a1_task_state.py.

    python -m experiments.journal_v2_a1_v4_task            # run/resume until done
    python -m experiments.journal_v2_a1_v4_task --status
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

from experiments import journal_v2_a1_task_state as TS

ROOT = TS.PROJECT_ROOT
OUT = TS.STATE_DIR
REL = lambda p: str(Path(p).resolve().relative_to(ROOT)).replace("\\", "/")  # noqa: E731
# Never executed by this task: PPO-learning tests (the two historical ones and the tiny-PPO resume validation of the A1 runner).
REGRESSION_EXCLUDES = ("tests/test_defender_adaptation_training_runner.py", "tests/test_vectorized_clutter_training.py",
                       "tests/test_journal_v2_a1_r2_training_runner.py")


def stage_01(state):
    from experiments import journal_v2_a1_r2 as R
    from experiments.journal_v2_protocol import assert_canonical_v2_training_config, build_v2_env_config

    assert TS.git("branch", "--show-current") == "research/journal-corrected-attacker-v2"
    head = TS.git("rev-parse", "HEAD")
    TS.git("merge-base", "--is-ancestor", "0ecf78e2d7e0727cdbdc1c6ca8e9e17b70699db5", "HEAD")  # raises if history diverged
    assert TS.git("rev-parse", "journal-v2-a0-canonical-v1^{commit}") == "bdbee45dc08a583a0da5ca882dc6b1269207a8cd"
    assert TS.git("diff", "--name-only", "journal-v2-a0-canonical-v1", "HEAD", "--", "uav_defend") == ""
    for level in ("C0", "C1", "C2", "C3"):
        for estimator in ("measurement", "kalman"):
            assert_canonical_v2_training_config(build_v2_env_config(level, estimator))
    assert (R.REWARD_VERSION, R.STEP_COST, R.W_PROGRESS, R.W_PATH, R.TERMINAL_SUCCESS, R.TERMINAL_FAILURE, R.GAMMA) == (
        "a1_v2_reward_v2", -0.1, 2.0, 1.0, 10.0, -10.0, 0.99)
    for rel in ("docs/journal_v2_a1_campaign_v1/README.md", "docs/journal_v2_a1_campaign_v2/README.md",
                "docs/journal_v2_a1_campaign_v2/data/theta_study_action_v3_summary.json",
                "docs/journal_v2_a1_campaign_v2/data/theta_study_summary.json"):
        assert (ROOT / rel).is_file(), rel
    report = {"head": head, "branch": TS.git("branch", "--show-current"), "canonical_a0_commit": "bdbee45dc08a583a0da5ca882dc6b1269207a8cd",
              "uav_defend_diff_vs_a0_tag": 0, "canonical_v2_guard": "8 configs pass", "reward": R.REWARD_VERSION,
              "failed_evidence_preserved": ["a1_v2_training_v1 (docs + results)", "action-v2 theta study", "action-v3 theta study"]}
    TS.atomic_write_json(OUT / "stage01_repository_verification.json", report)
    return [REL(OUT / "stage01_repository_verification.json")]


def stage_02(state):
    from experiments import journal_v2_a1_v4_primitive_audit as P

    result = P.build()
    TS.atomic_write_json(OUT / "primitive_audit.json", result)
    TS.halt(state, "02_primitive_audit", "D", "; ".join(result["conclusion"]["reasons"][:2]))
    return [REL(OUT / "primitive_audit.json")]


def stage_08(state):
    cmd = [sys.executable, "-m", "pytest", "tests", "-q", "--no-header", "-p", "no:cacheprovider", "-x", "--tb=short",
           *[f"--ignore={p}" for p in REGRESSION_EXCLUDES]]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env={**__import__("os").environ, "PYTHONPATH": ""})
    tail = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    match = re.search(r"(\d+) passed", tail)
    summary = {"command": " ".join(cmd[2:]), "returncode": proc.returncode, "summary_line": tail,
               "passed": int(match.group(1)) if match else None, "excluded_as_ppo_learning_tests": list(REGRESSION_EXCLUDES)}
    TS.atomic_write_json(OUT / "stage08_regression_summary.json", summary)
    if proc.returncode != 0:
        raise RuntimeError(f"non-training regression failed: {tail}\n{proc.stdout[-1500:]}")
    return [REL(OUT / "stage08_regression_summary.json")]


def stage_09(state):
    from experiments import journal_v2_a1_r2 as R

    registry = json.loads((ROOT / "results" / "defender_adaptation_model_registry.json").read_text())
    entries = list(registry["defenders"].values()) + list(registry["attackers"].values())
    bad = []
    for e in entries:
        path = Path(e["path"]) if Path(e["path"]).is_absolute() else ROOT / e["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != e["sha256"]:
            bad.append(str(path))
    failed_v1 = json.loads((R.FAILED_V1_RESULTS_ROOT / "a1_v2_model_registry.json").read_text())
    failed_bad = []
    for name, e in failed_v1["attackers"].items():
        if hashlib.sha256((ROOT / e["path"]).read_bytes()).hexdigest() != e["sha256"]:
            failed_bad.append(name)
    report = {"historical_registered_checkpoints": len(entries), "historical_mismatches": bad,
              "failed_a1_v1_checkpoints": len(failed_v1["attackers"]), "failed_a1_v1_mismatches": failed_bad,
              "no_action_v2_v3_or_v4_models_exist": not (R.RESULTS_ROOT / "seed_201" / "checkpoints" / "final_model.zip").exists()}
    TS.atomic_write_json(OUT / "stage09_checkpoint_integrity.json", report)
    if bad or failed_bad:
        raise RuntimeError(f"checkpoint integrity failure: {bad} {failed_bad}")
    return [REL(OUT / "stage09_checkpoint_integrity.json")]


RUNNERS = {"01_repository_verification": stage_01, "02_primitive_audit": stage_02, "08_non_training_regression": stage_08,
           "09_checkpoint_integrity": stage_09}


def main() -> int:
    state = TS.load_state()
    if "--status" in sys.argv:
        print(json.dumps({"next_stage": TS.next_stage(state), "halt": state.get("halt"),
                          "statuses": {k: v["status"] for k, v in state["stages"].items()}}, indent=1))
        return 0
    while True:
        state = TS.load_state()
        name = TS.next_stage(state)
        if name is None or name == "10_commit_tag_push":
            print(f"pipeline stopped before {name}: stage 10 (commit/push) is performed by hand after review")
            return 0
        print(f"[{name}] running", flush=True)
        TS.begin(state, name)
        try:
            artifacts = RUNNERS[name](TS.load_state())
        except Exception as exc:
            TS.fail(TS.load_state(), name, repr(exc)[:2000])
            print(f"[{name}] FAILED: {exc!r}", flush=True)
            return 1
        TS.complete(TS.load_state(), name, artifacts)
        print(f"[{name}] complete", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
