"""Stage runner for the A1-CBF foundation (pursuit/dynamics only; no obstacle/defender CBF, no RL). Safe to re-run
after ANY interruption.

    python -m experiments.journal_cbf_attacker_run all     # stages 1-7 (restartable)
    python -m experiments.journal_cbf_attacker_run status
"""

from __future__ import annotations

import argparse
import json
import subprocess

from experiments import journal_cbf_attacker_foundation as F
from experiments import journal_cbf_attacker_state as TS

DOCS = F.DOCS
START_HEAD = "2f45914"
# This stage, uniquely among A1 experiments, extends an EXISTING tracked file (SoldierEnv gains the opt-in
# enemy_control_mode constructor switch + the attacker-package import) rather than adding only new files;
# verify_repository() allows exactly this file to be modified relative to HEAD and flags anything else as an
# unexpected change. enemy_control_mode is deliberately a SoldierEnv.__init__ argument, NOT an EnvConfig field --
# EnvConfig itself is therefore untouched (net-zero vs. this stage's branch point), which also means it never
# perturbs any prior curriculum stage's frozen config/protocol-hash guard tests (see README.md section 5).
EXPECTED_MODIFIED_TRACKED_FILES = ("uav_defend/envs/soldier_env.py",)


def verify_repository() -> dict:
    head = TS.git("rev-parse", "HEAD")
    porcelain = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=TS.PROJECT_ROOT, capture_output=True, text=True, check=True).stdout
    modified = [line[3:] for line in porcelain.splitlines() if line]
    unexpected_modifications = [f for f in modified if f not in EXPECTED_MODIFIED_TRACKED_FILES]
    rec = {"head": head, "expected_start_head": START_HEAD, "head_is_expected_or_descendant": subprocess.run(["git", "merge-base", "--is-ancestor", START_HEAD, "HEAD"], cwd=TS.PROJECT_ROOT).returncode == 0,
           "branch": TS.git("rev-parse", "--abbrev-ref", "HEAD"), "modified_tracked_files": modified, "expected_modified_tracked_files": list(EXPECTED_MODIFIED_TRACKED_FILES),
           "unexpected_modifications": unexpected_modifications, "tracked_tree_clean_except_expected_cbf_files": len(unexpected_modifications) == 0,
           "untracked_at_start": [l[3:] for l in TS.git("status", "--porcelain", "--untracked-files=all").splitlines() if l.startswith("??")],
           "uav_defend_diff_stat_vs_a0_tag": TS.git("diff", "--stat", "journal-v2-a0-canonical-v1", "--", "uav_defend"),
           "namespace_overlaps": F.namespace_overlaps(), "scope": "pursuit/dynamics foundation only; no obstacle CBF, no defender CBF, no RL training"}
    assert rec["head_is_expected_or_descendant"] and rec["tracked_tree_clean_except_expected_cbf_files"]
    assert not rec["namespace_overlaps"]["foundation_vs_legacy_noop_overlap"] and rec["namespace_overlaps"]["gap_to_curriculum_m"] > 100_000
    return rec


def run_all() -> None:
    state = TS.load_state()
    psha = F.protocol_sha256()
    scratch = F.RESULTS_ROOT
    print(f"[cbf foundation] results={scratch} protocol_sha={psha[:16]}", flush=True)
    if not TS.stage_is_verified(state, "01_repository_verification"):
        TS.begin(state, "01_repository_verification")
        TS.atomic_write_json(TS.STATE_DIR / "repository_verification.json", verify_repository())
        TS.complete(state, "01_repository_verification", [DOCS + "repository_verification.json"], f"HEAD descendant of {START_HEAD}, tracked tree clean, namespaces disjoint")
    if not TS.stage_is_verified(state, "02_protocol_freeze", psha):
        TS.begin(state, "02_protocol_freeze", psha)
        TS.atomic_write_json(TS.STATE_DIR / "protocol.json", F.protocol_specification() | {"protocol_sha256": psha})
        TS.complete(state, "02_protocol_freeze", [DOCS + "protocol.json"], "protocol sha " + psha[:16])
    if not TS.stage_is_verified(state, "03_legacy_noop_validation", psha):
        TS.begin(state, "03_legacy_noop_validation", psha)
        result = F.run_legacy_noop_validation(scratch / "legacy_noop")
        TS.atomic_write_json(TS.STATE_DIR / "legacy_noop_result.json", result)
        artifacts = [DOCS + "legacy_noop_result.json"]
        if not result["identical"]:
            TS.fail(state, "03_legacy_noop_validation", f"LEGACY NO-OP FAILED: {len(result['mismatches'])} mismatches out of {result['seeds_checked']} seeds", artifacts)
            raise SystemExit("legacy no-op validation failed: fix before proceeding")
        TS.complete(state, "03_legacy_noop_validation", artifacts, f"{result['seeds_checked']} seeds identical to parent commit {result['parent_commit']}")
    test_path = TS.STATE_DIR / "unit_test_result.json"
    if not TS.stage_is_verified(state, "04_targeted_tests", psha):
        if not test_path.exists() or json.loads(test_path.read_text()).get("failed", 1) != 0:
            raise SystemExit("run the targeted tests first and write unit_test_result.json (failed == 0)")
        TS.begin(state, "04_targeted_tests", psha)
        TS.complete(state, "04_targeted_tests", [DOCS + "unit_test_result.json"], "targeted tests passed")
    if not TS.stage_is_verified(state, "05_foundation_validation", psha):
        TS.begin(state, "05_foundation_validation", psha)
        a0_eps = F.run_foundation_validation(scratch / "foundation", enemy_control_mode="legacy")
        cbf_eps = F.run_foundation_validation(scratch / "foundation", enemy_control_mode="cbf_velocity")
        m0, mcbf = F.metrics(a0_eps), F.metrics(cbf_eps)
        record = {"scenarios": list(F.FOUNDATION_SEEDS), "a0": {"metrics": m0}, "a1_cbf": {"metrics": mcbf}}
        TS.atomic_write_json(TS.STATE_DIR / "foundation_validation.json", record)
        TS.complete(state, "05_foundation_validation", [DOCS + "foundation_validation.json"], f"A0 capture {m0['capture']:.3f}; A1-CBF capture {mcbf['capture']:.3f}")
    if not TS.stage_is_verified(state, "06_constraint_verification", psha):
        TS.begin(state, "06_constraint_verification", psha)
        cbf_eps = F.run_foundation_validation(scratch / "foundation", enemy_control_mode="cbf_velocity")
        result = F.constraint_verification(cbf_eps, limits=F.CBFAttackerLimits())
        TS.atomic_write_json(TS.STATE_DIR / "constraint_verification.json", result)
        TS.complete(state, "06_constraint_verification", [DOCS + "constraint_verification.json"], f"passed={result['passed']}")
    if not TS.stage_is_verified(state, "07_acceptance_decision", psha):
        TS.begin(state, "07_acceptance_decision", psha)
        cbf_eps = F.run_foundation_validation(scratch / "foundation", enemy_control_mode="cbf_velocity")
        mcbf = F.metrics(cbf_eps)
        gate = F.gate_status(mcbf)
        constraint = json.loads((TS.STATE_DIR / "constraint_verification.json").read_text())
        decision = {"gate": gate, "constraint_verification_passed": constraint["passed"], "passed": bool(gate["passed"] and constraint["passed"]),
                    "verdict": "A" if (gate["passed"] and constraint["passed"]) else ("B" if mcbf["capture"] >= 0.80 else "C")}
        TS.atomic_write_json(TS.STATE_DIR / "acceptance_decision.json", decision)
        TS.complete(state, "07_acceptance_decision", [DOCS + "acceptance_decision.json"], f"verdict {decision['verdict']}")
    print(json.dumps(json.loads((TS.STATE_DIR / "acceptance_decision.json").read_text()), indent=1))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("all", "status"))
    args = parser.parse_args(argv)
    TS.STATE_DIR.mkdir(parents=True, exist_ok=True)
    if args.command == "status":
        print(json.dumps({s: TS.load_state()["stages"].get(s, {}).get("status") for s in TS.STAGES}, indent=1))
    else:
        run_all()


if __name__ == "__main__":
    main()
