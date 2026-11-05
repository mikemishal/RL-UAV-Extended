"""Stage runner for A1-CBF static-obstacle navigation (Prompt 2; no defender/dynamic-obstacle CBF, no RL). Safe to
re-run after ANY interruption.

    python -m experiments.journal_cbf_attacker_obstacle_run all     # stages 1-7 (restartable)
    python -m experiments.journal_cbf_attacker_obstacle_run status
"""

from __future__ import annotations

import argparse
import json
import subprocess

from experiments import journal_cbf_attacker_obstacle_foundation as F
from experiments import journal_cbf_attacker_obstacle_state as TS

DOCS = F.DOCS
START_HEAD = "2beaf77"
EXPECTED_MODIFIED_TRACKED_FILES = ("uav_defend/attackers/cbf_attacker.py", "uav_defend/envs/soldier_env.py")


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
           "namespace_overlaps": F.namespace_overlaps(), "scope": "static-obstacle navigation only; no defender CBF, no dynamic-obstacle CBF, no RL training"}
    assert rec["head_is_expected_or_descendant"] and rec["tracked_tree_clean_except_expected_cbf_files"]
    assert not rec["namespace_overlaps"]["obstacle_vs_legacy_noop_overlap"] and not rec["namespace_overlaps"]["obstacle_vs_foundation_legacy_noop_overlap"]
    return rec


def run_all() -> None:
    state = TS.load_state()
    psha = F.protocol_sha256()
    scratch = F.RESULTS_ROOT
    print(f"[cbf obstacle] results={scratch} protocol_sha={psha[:16]}", flush=True)
    if not TS.stage_is_verified(state, "01_repository_verification"):
        TS.begin(state, "01_repository_verification")
        TS.atomic_write_json(TS.STATE_DIR / "repository_verification.json", verify_repository())
        TS.complete(state, "01_repository_verification", [DOCS + "repository_verification.json"], f"HEAD descendant of {START_HEAD}, tracked tree clean, namespaces disjoint")
    if not TS.stage_is_verified(state, "02_protocol_freeze", psha):
        TS.begin(state, "02_protocol_freeze", psha)
        TS.atomic_write_json(TS.STATE_DIR / "protocol.json", F.protocol_specification() | {"protocol_sha256": psha})
        TS.complete(state, "02_protocol_freeze", [DOCS + "protocol.json"], "protocol sha " + psha[:16])
    if not TS.stage_is_verified(state, "03_static_obstacle_adapter", psha):
        TS.begin(state, "03_static_obstacle_adapter", psha)
        c0 = F.c0_identity_check(range(*F.C0_IDENTITY_SEEDS))
        TS.atomic_write_json(TS.STATE_DIR / "c0_identity_result.json", c0)
        artifacts = [DOCS + "c0_identity_result.json"]
        if not c0["identical"]:
            TS.fail(state, "03_static_obstacle_adapter", f"C0 IDENTITY REGRESSION FAILED: {len(c0['mismatches'])} mismatches", artifacts)
            raise SystemExit("C0 identity regression failed: fix before proceeding")
        TS.complete(state, "03_static_obstacle_adapter", artifacts, f"{c0['seeds_checked']} C0 seeds identical to Prompt-1 cached results")
    if not TS.stage_is_verified(state, "04_route_planner_integration", psha):
        TS.begin(state, "04_route_planner_integration", psha)
        result = F.run_legacy_noop_validation(scratch / "legacy_noop")
        TS.atomic_write_json(TS.STATE_DIR / "legacy_noop_result.json", result)
        artifacts = [DOCS + "legacy_noop_result.json"]
        if not result["identical"]:
            TS.fail(state, "04_route_planner_integration", f"LEGACY NO-OP FAILED: {len(result['mismatches'])} mismatches", artifacts)
            raise SystemExit("legacy no-op validation failed: fix before proceeding")
        TS.complete(state, "04_route_planner_integration", artifacts, f"{result['seeds_checked']} seeds identical to parent commit {result['parent_commit']}")
    test_path = TS.STATE_DIR / "unit_test_result.json"
    if not TS.stage_is_verified(state, "05_cbf_qp_implementation", psha):
        if not test_path.exists() or json.loads(test_path.read_text()).get("failed", 1) != 0:
            raise SystemExit("run the targeted tests first and write unit_test_result.json (failed == 0)")
        TS.begin(state, "05_cbf_qp_implementation", psha)
        TS.complete(state, "05_cbf_qp_implementation", [DOCS + "unit_test_result.json"], "targeted tests passed")
    if not TS.stage_is_verified(state, "06_deterministic_unit_validation", psha):
        TS.begin(state, "06_deterministic_unit_validation", psha)
        cbf_eps = F.run_obstacle_validation(scratch / "obstacle", enemy_control_mode="cbf_velocity")
        result = F.constraint_verification(cbf_eps, limits=F.CBFAttackerLimits())
        TS.atomic_write_json(TS.STATE_DIR / "constraint_verification.json", result)
        TS.complete(state, "06_deterministic_unit_validation", [DOCS + "constraint_verification.json"], f"passed={result['passed']}")
    if not TS.stage_is_verified(state, "07_monte_carlo_evaluation", psha):
        TS.begin(state, "07_monte_carlo_evaluation", psha)
        a0_eps = F.run_obstacle_validation(scratch / "obstacle", enemy_control_mode="legacy")
        cbf_eps = F.run_obstacle_validation(scratch / "obstacle", enemy_control_mode="cbf_velocity")
        m0, mcbf = F.metrics(a0_eps), F.metrics(cbf_eps)
        gate = F.gate_status(mcbf)
        constraint = json.loads((TS.STATE_DIR / "constraint_verification.json").read_text())
        decision = {"gate": gate, "constraint_verification_passed": constraint["passed"], "passed": bool(gate["passed"] and constraint["passed"]),
                    "verdict": "A" if (gate["passed"] and constraint["passed"]) else ("B" if mcbf["capture"] >= 0.80 and mcbf["obstacle_collision"] <= 3 else "C")}
        record = {"scenarios": list(F.OBSTACLE_SEEDS), "a0": {"metrics": m0}, "a1_cbf": {"metrics": mcbf}, "gate": gate, "decision": decision}
        TS.atomic_write_json(TS.STATE_DIR / "monte_carlo_result.json", record)
        TS.atomic_write_json(TS.STATE_DIR / "acceptance_decision.json", decision)
        TS.complete(state, "07_monte_carlo_evaluation", [DOCS + "monte_carlo_result.json", DOCS + "acceptance_decision.json"],
                    f"A0 capture {m0['capture']:.3f}; A1-CBF capture {mcbf['capture']:.3f}; collisions {mcbf['obstacle_collision']}; verdict {decision['verdict']}")
    print(json.dumps(json.loads((TS.STATE_DIR / "acceptance_decision.json").read_text()), indent=1))


def print_status() -> None:
    state = TS.load_state()
    for name in TS.STAGES:
        print(name, state["stages"].get(name, {}).get("status", "pending"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", choices=["all", "status"])
    args = parser.parse_args()
    if args.cmd == "all":
        run_all()
    else:
        print_status()
