"""Phase 9-13 analysis for the mission-endurance population sweep.

Reads results/mission_endurance_study/<label>/<level>_<seed>.json produced by
experiments.mission_endurance_population_sweep and reports, per clutter level and
factor: outcome tallies, path-length stats, and every seed whose outcome changed
relative to the endurance-disabled baseline.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

OUT = Path("results/mission_endurance_study")
LABELS = ("disabled", "k3", "k4", "k5")
LEVELS = ("C0", "C1", "C2", "C3")


def load(label: str, level: str, seed: int) -> dict:
    return json.loads((OUT / label / f"{level}_{seed}.json").read_text(encoding="utf-8"))


def summarize_level(label: str, level: str) -> dict:
    outcomes: dict[str, int] = {}
    paths = []
    excess = []
    durations = []
    for seed in range(130000, 130100):
        rec = load(label, level, seed)
        s = rec["summary"]
        outcomes[s["outcome"]] = outcomes.get(s["outcome"], 0) + 1
        paths.append(s["path_length_m"])
        excess.append(s["path_excess_ratio"])
        durations.append(s["flight_time_s"])
    return {
        "outcomes": outcomes,
        "mean_path": statistics.mean(paths),
        "median_path": statistics.median(paths),
        "max_path": max(paths),
        "mean_excess": statistics.mean(excess),
        "max_excess": max(excess),
        "mean_duration": statistics.mean(durations),
    }


def changed_seeds(label: str, level: str) -> list[dict]:
    changes = []
    for seed in range(130000, 130100):
        base = load("disabled", level, seed)["summary"]
        other = load(label, level, seed)["summary"]
        if base["outcome"] != other["outcome"]:
            changes.append({
                "seed": seed,
                "base_outcome": base["outcome"],
                "new_outcome": other["outcome"],
                "base_path": base["path_length_m"],
                "new_path": other["path_length_m"],
            })
    return changes


def main() -> None:
    for label in LABELS:
        print(f"=== {label} ===")
        for level in LEVELS:
            s = summarize_level(label, level)
            print(level, s["outcomes"],
                  "mean_path=%.1f median_path=%.1f max_path=%.1f mean_excess=%.3f max_excess=%.3f mean_dur=%.1f" % (
                      s["mean_path"], s["median_path"], s["max_path"], s["mean_excess"], s["max_excess"], s["mean_duration"]))
    print()
    print("=== changed seeds relative to disabled ===")
    for label in ("k3", "k4", "k5"):
        print(f"--- {label} ---")
        total_changed = 0
        converted_captures = 0
        for level in LEVELS:
            changes = changed_seeds(label, level)
            total_changed += len(changes)
            for c in changes:
                if c["base_outcome"] in ("soldier_caught", "interception"):
                    converted_captures += 1
                print(f"  {level} seed {c['seed']}: {c['base_outcome']} -> {c['new_outcome']} "
                      f"(path {c['base_path']:.1f} -> {c['new_path']:.1f})")
        print(f"  total changed={total_changed} converted_successful_captures={converted_captures}")


if __name__ == "__main__":
    main()
