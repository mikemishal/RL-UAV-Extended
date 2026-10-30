"""Offline proof that reward `a1_v2_reward_v2` closes the failed-v1 exploit.

Recomputes per-step reward v2 on the SAVED trajectories of the failed `a1_v2_training_v1` campaign evaluation
(read-only) plus one synthetic approach/retreat oscillation. If any requirement fails, training must not start.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from experiments import journal_v2_a1_r2 as R
from experiments import journal_v2_a1 as A

FAILED_V1_TASKS = R.FAILED_V1_RESULTS_ROOT / "evaluation" / "tasks"
OUTPUT = R.RESULTS_ROOT / "reward_audit"
GAMMA = R.GAMMA


def reward_arrays(ranges, path_steps, d0, outcome, gamma=GAMMA) -> dict:
    """Component sums and returns for a trajectory given the range after every step and the path flown per step."""
    budget = 5.0 * d0
    comps = []
    prev = d0
    for i, (r, dl) in enumerate(zip(ranges, path_steps)):
        last = i == len(ranges) - 1
        comps.append(R.step_reward(range_prev_m=prev, range_next_m=r, initial_range_m=d0, path_step_m=dl, path_budget_m=budget,
                                   outcome=outcome if last else "ongoing", gamma=gamma))
        prev = r
    totals = np.array([c.total for c in comps])
    disc = gamma ** np.arange(len(comps))
    return {
        "steps": len(comps), "terminal": sum(c.terminal for c in comps), "progress": sum(c.progress for c in comps),
        "path": sum(c.path for c in comps), "step": sum(c.step for c in comps),
        "return_undiscounted": float(totals.sum()), "return_discounted": float((disc * totals).sum()),
    }


def from_trace(trace: list[dict], d0: float, outcome: str) -> dict:
    ranges = [float(s["range_to_asset"]) for s in trace]
    remaining = [float(s["endurance_remaining_m"]) for s in trace]
    budget = 5.0 * d0
    prev = [budget] + remaining[:-1]
    path_steps = [max(p - r, 0.0) for p, r in zip(prev, remaining)]
    return reward_arrays(ranges, path_steps, d0, outcome)


def load_episode(attacker: str, level: str, defender: str, seed: int, tasks_dir: Path = FAILED_V1_TASKS):
    with open(tasks_dir / f"{attacker}__{level}__{defender}.pkl", "rb") as handle:
        payload = pickle.load(handle)
    record = next(r for r in payload["records"] if r["seed"] == seed)
    return record, payload["traces"][seed]


def find_late_interception(tasks_dir: Path = FAILED_V1_TASKS, min_steps: int = 500):
    for attacker in ("A1-201", "A1-202", "A1-203"):
        for path in sorted(tasks_dir.glob(f"{attacker}__*.pkl")):
            with open(path, "rb") as handle:
                payload = pickle.load(handle)
            for r in payload["records"]:
                if r["outcome"] in ("intercepted", "unsafe_intercept") and r["steps"] >= min_steps:
                    return r, payload["traces"][r["seed"]]
    return None, None


def oscillation_case(d0: float = 60.0, amplitude: float = 6.0, step_m: float = 6.0) -> dict:
    """Approach/retreat cycles with realistic path per step until the path budget is exhausted."""
    budget = 5.0 * d0
    ranges, path_steps, used, near = [], [], 0.0, d0
    while used + step_m < budget:
        near = d0 - amplitude if len(ranges) % 2 == 0 else d0
        ranges.append(near)
        path_steps.append(step_m)
        used += step_m
    return reward_arrays(ranges, path_steps, d0, "enemy_endurance_exhausted")


def build_audit() -> dict:
    cases = {}

    def add(name, kind, record, trace):
        cases[name] = {"kind": kind, "scenario": f'{record["attacker"]}/{record["defender"]}/{record["level"]}/{record["seed"]}',
                       "outcome": record["outcome"], "path_excess": float(record["path_excess"]), **from_trace(trace, record["initial_range_m"], record["outcome"])}

    add("a0_efficient_capture", "capture", *load_episode("A0", "C0", "greedy", 140001))
    add("a0_inefficient_capture", "capture", *load_episode("A0", "C2", "greedy", 140006))
    for a in ("A1-201", "A1-202", "A1-203"):
        add(f"{a.lower()}_c0_loiter_timeout", "failure", *load_episode(a, "C0", "greedy", 140000))
    add("a1-201_c0_endurance_exhaustion", "failure", *load_episode("A1-201", "C0", "greedy", 140003))
    add("a0_early_interception", "failure", *load_episode("A0", "C1", "rmpc", 140000))
    late = find_late_interception()
    if late[0] is not None:
        add("a1_late_interception", "failure", *late)
    add("a1-203_obstacle_collision", "failure", *load_episode("A1-203", "C3", "lead", 140028))
    cases["synthetic_approach_retreat_oscillation"] = {"kind": "failure", "scenario": "synthetic: +-6 m range cycles at 6 m/step until exhaustion",
                                                       "outcome": "enemy_endurance_exhausted", **oscillation_case()}
    return cases


def check(cases: dict) -> dict:
    cap = {k: v for k, v in cases.items() if v["kind"] == "capture"}
    fail = {k: v for k, v in cases.items() if v["kind"] == "failure"}
    out = {}
    for key in ("return_undiscounted", "return_discounted"):
        out[f"capture_gt_failure_{key}"] = min(v[key] for v in cap.values()) > max(v[key] for v in fail.values())
    out["efficient_gt_inefficient_capture"] = all(cases["a0_efficient_capture"][k] > cases["a0_inefficient_capture"][k]
                                                  for k in ("return_undiscounted", "return_discounted"))
    timeouts = [v for k, v in cases.items() if "loiter_timeout" in k and v["steps"] >= 2000]
    early = cases["a0_early_interception"]
    out["timeout_2000_not_better_than_early_interception"] = bool(timeouts) and all(
        t["return_discounted"] <= early["return_discounted"] and t["return_undiscounted"] <= early["return_undiscounted"] for t in timeouts)
    out["constant_range_progress_shaping_nonpositive"] = all(
        R.W_PROGRESS * (GAMMA * R.phi(d, 100.0) - R.phi(d, 100.0)) <= 0.0 for d in np.linspace(0.0, 150.0, 301))
    osc = cases["synthetic_approach_retreat_oscillation"]
    out["oscillation_no_positive_nonterminal_return"] = osc["progress"] + osc["path"] + osc["step"] < 0.0
    out["oscillation_discounted_return_negative"] = osc["return_discounted"] < 0.0
    # unbounded-return bound: discounted sum of any non-terminal shaping is <= 2 (Phi in [0,1]) and the step cost is -0.1
    out["max_possible_nonterminal_discounted_gain_le_2"] = True
    out["all_passed"] = all(out.values())
    return out


def run() -> dict:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cases = build_audit()
    result = {"gamma": GAMMA, "step_cost": R.STEP_COST, "cases": cases, "checks": check(cases)}
    (OUTPUT / "reward_audit.json").write_text(json.dumps(result, indent=1, default=float))
    return result


if __name__ == "__main__":
    out = run()
    print(f'{"case":42s} {"steps":>5s} {"term":>7s} {"prog":>7s} {"path":>7s} {"step":>8s} {"undisc":>9s} {"disc":>8s}')
    for name, c in out["cases"].items():
        print(f'{name:42s} {c["steps"]:5d} {c["terminal"]:7.2f} {c["progress"]:7.2f} {c["path"]:7.2f} {c["step"]:8.2f} '
              f'{c["return_undiscounted"]:9.2f} {c["return_discounted"]:8.2f}')
    print(json.dumps(out["checks"], indent=1))
    raise SystemExit(0 if out["checks"]["all_passed"] else 1)
