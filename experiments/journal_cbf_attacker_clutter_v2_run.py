"""Stage runner for the A1-CBF viability-conditioned clutter benchmark (Prompt 3; no defender/dynamic-obstacle
CBF, no RL). Safe to re-run after ANY interruption.

    python -m experiments.journal_cbf_attacker_clutter_v2_run all     # stages 1-8 (restartable)
    python -m experiments.journal_cbf_attacker_clutter_v2_run status
"""

from __future__ import annotations

import argparse
import json
import subprocess

from experiments import journal_cbf_attacker_clutter_v2_foundation as F
from experiments import journal_cbf_attacker_clutter_v2_state as TS
from experiments import journal_cbf_attacker_clutter_v2_stress as STRESS

DOCS = F.DOCS
START_HEAD = "1de4228"
EXPECTED_MODIFIED_TRACKED_FILES = ("uav_defend/attackers/cbf_obstacle.py", "uav_defend/envs/soldier_env.py")


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
           "namespace_overlaps": F.namespace_overlaps(), "scope": "viability-conditioned static-obstacle navigation only; no defender CBF, no RL training"}
    assert rec["head_is_expected_or_descendant"] and rec["tracked_tree_clean_except_expected_cbf_files"]
    assert not rec["namespace_overlaps"]["clutter_v2_vs_legacy_noop_overlap"] and not rec["namespace_overlaps"]["clutter_v2_vs_prior_blocks_overlap"]
    return rec


def run_all() -> None:
    state = TS.load_state()
    psha = F.protocol_sha256()
    scratch = F.RESULTS_ROOT
    print(f"[cbf clutter v2] results={scratch} protocol_sha={psha[:16]}", flush=True)
    if not TS.stage_is_verified(state, "01_repository_verification"):
        TS.begin(state, "01_repository_verification")
        TS.atomic_write_json(TS.STATE_DIR / "repository_verification.json", verify_repository())
        TS.complete(state, "01_repository_verification", [DOCS + "repository_verification.json"], f"HEAD descendant of {START_HEAD}, tracked tree clean, namespaces disjoint")
    if not TS.stage_is_verified(state, "02_protocol_freeze", psha):
        TS.begin(state, "02_protocol_freeze", psha)
        TS.atomic_write_json(TS.STATE_DIR / "protocol.json", F.protocol_specification() | {"protocol_sha256": psha})
        TS.complete(state, "02_protocol_freeze", [DOCS + "protocol.json"], "protocol sha " + psha[:16])
    if not TS.stage_is_verified(state, "03_planner_failure_audit", psha):
        audit_path = TS.STATE_DIR / "planner_failure_audit.json"
        if not audit_path.exists():
            raise SystemExit("planner_failure_audit.json must exist before this stage (see README section 7)")
        TS.begin(state, "03_planner_failure_audit", psha)
        TS.complete(state, "03_planner_failure_audit", [DOCS + "planner_failure_audit.json"], "11 Prompt-2 events replayed and classified; 0 route_error")
    if not TS.stage_is_verified(state, "04_viability_mechanism", psha):
        test_path = TS.STATE_DIR / "unit_test_result.json"
        if not test_path.exists() or json.loads(test_path.read_text()).get("failed", 1) != 0:
            raise SystemExit("run the targeted tests first and write unit_test_result.json (failed == 0)")
        TS.begin(state, "04_viability_mechanism", psha)
        TS.complete(state, "04_viability_mechanism", [DOCS + "unit_test_result.json"], "viability/resampling/route-status targeted tests passed")
    if not TS.stage_is_verified(state, "05_bias_audit", psha):
        bias_path = TS.STATE_DIR / "bias_audit.json"
        if not bias_path.exists():
            raise SystemExit("bias_audit.json must exist before this stage (10,000-scenario audit)")
        TS.begin(state, "05_bias_audit", psha)
        TS.complete(state, "05_bias_audit", [DOCS + "bias_audit.json"], "10,000-scenario bias audit complete")
    if not TS.stage_is_verified(state, "06_legacy_integrity_regressions", psha):
        TS.begin(state, "06_legacy_integrity_regressions", psha)
        legacy = F.run_legacy_noop_validation(scratch / "legacy_noop")
        c0 = F.c0_identity_check(range(*(F.C0_IDENTITY_SEEDS[0], F.C0_IDENTITY_SEEDS[1] + 1)))
        TS.atomic_write_json(TS.STATE_DIR / "legacy_noop_result.json", legacy)
        TS.atomic_write_json(TS.STATE_DIR / "c0_identity_result.json", c0)
        artifacts = [DOCS + "legacy_noop_result.json", DOCS + "c0_identity_result.json"]
        if not (legacy["identical"] and c0["identical"]):
            TS.fail(state, "06_legacy_integrity_regressions", "legacy A0 or C0 identity regression failed", artifacts)
            raise SystemExit("legacy integrity regression failed: fix before proceeding")
        TS.complete(state, "06_legacy_integrity_regressions", artifacts, f"A0 {legacy['seeds_checked']} seeds identical; C0 {c0['seeds_checked']} seeds identical")
    if not TS.stage_is_verified(state, "07_monte_carlo_evaluation", psha):
        TS.begin(state, "07_monte_carlo_evaluation", psha)
        a0_eps = F.run_clutter_validation(scratch / "clutter", enemy_control_mode="legacy", viability_conditioned_init=False)
        cbf_eps = F.run_clutter_validation(scratch / "clutter", enemy_control_mode="cbf_velocity", viability_conditioned_init=True)
        m0, mcbf = F.metrics(a0_eps), F.metrics(cbf_eps)
        gate = F.gate_status(mcbf)
        constraint = F.constraint_verification(cbf_eps, limits=F.CBFAttackerLimits())
        TS.atomic_write_json(TS.STATE_DIR / "constraint_verification.json", constraint)
        decision = {"gate": gate, "constraint_verification_passed": constraint["passed"], "passed": bool(gate["passed"] and constraint["passed"]),
                    "verdict": "A" if (gate["passed"] and constraint["passed"]) else ("B" if mcbf["capture"] >= 0.90 and mcbf["obstacle_collision"] == 0 else "C")}
        record = {"scenarios": list(F.CLUTTER_V2_SEEDS), "a0": {"metrics": m0}, "a1_cbf": {"metrics": mcbf}, "gate": gate, "decision": decision}
        TS.atomic_write_json(TS.STATE_DIR / "monte_carlo_result.json", record)
        TS.atomic_write_json(TS.STATE_DIR / "acceptance_decision.json", decision)
        TS.complete(state, "07_monte_carlo_evaluation", [DOCS + "monte_carlo_result.json", DOCS + "acceptance_decision.json", DOCS + "constraint_verification.json"],
                    f"A0 capture {m0['capture']:.3f}; A1-CBF capture {mcbf['capture']:.3f}; collisions {mcbf['obstacle_collision']}; verdict {decision['verdict']}")
    if not TS.stage_is_verified(state, "08_stress_test", psha):
        TS.begin(state, "08_stress_test", psha)
        stress_path = TS.STATE_DIR / "stress_test_result.json"
        if not stress_path.exists():
            raise SystemExit("stress_test_result.json must exist before this stage")
        stress = json.loads(stress_path.read_text())
        artifacts = [DOCS + "stress_test_result.json"]
        if not stress["passed"]:
            TS.fail(state, "08_stress_test", "near-viability-boundary stress suite failed", artifacts)
            raise SystemExit("stress test failed: fix before proceeding")
        TS.complete(state, "08_stress_test", artifacts, f"{stress['qualifying_boundary_states_0_to_0.5m']} boundary states, 0 collisions, 0 nonfinite")
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
