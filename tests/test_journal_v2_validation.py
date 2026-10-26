"""Tests for the V2 validation instrumentation (observational code only)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from experiments.journal_v2_validation_core import (
    TERMINAL_ORBIT_FLYBY_THRESHOLD,
    TERMINAL_ORBIT_REVOLUTION_THRESHOLD,
    count_overshoots,
    dense_min_distance,
    deviation_segments,
    orbit_revolutions,
    record_episode,
    summarize,
    terminal_flyby_metrics,
)
from uav_defend.obstacles.swept_contact import first_contact


# ---------------------------------------------------------- definitions
def test_overshoot_counts_one_approach_then_recede():
    ranges = np.array([50, 38, 26, 14, 4, 4.2, 12, 20.0])
    assert count_overshoots(ranges, dt=0.5) == 1


def test_overshoot_ignores_monotone_closing_and_initial_recede():
    assert count_overshoots(np.array([50, 40, 30, 20, 10, 2.0]), dt=0.5) == 0
    assert count_overshoots(np.array([10, 18, 26, 34.0]), dt=0.5) == 0  # never approached first


def test_overshoot_deadband_ignores_jitter():
    ranges = np.array([30, 24, 18, 12, 6, 6.1, 5.9, 6.2, 6.0])
    assert count_overshoots(ranges, dt=0.5) == 0


def test_overshoot_counts_repeated_passes():
    ranges = np.array([40, 30, 20, 10, 3, 12, 22, 14, 6, 3, 14, 26.0])
    assert count_overshoots(ranges, dt=0.5) == 2


def _circle(n_rev: float, radius: float = 10.0, n: int = 200):
    t = np.linspace(0, 2 * math.pi * n_rev, n)
    return np.stack([radius * np.cos(t), radius * np.sin(t), np.zeros(n)], axis=1)


def test_orbit_detects_full_revolution_and_ignores_straight_pass():
    asset = np.zeros((200, 3))
    assert orbit_revolutions(_circle(1.25), asset)[0] == pytest.approx(1.25, abs=0.02)
    straight = np.stack([np.linspace(-15, 15, 200), np.full(200, 3.0), np.zeros(200)], axis=1)
    assert orbit_revolutions(straight, np.zeros((200, 3)))[0] < 0.51  # half-turn of bearing at most


def test_orbit_ignores_trajectories_outside_local_radius():
    far = _circle(3.0, radius=40.0)
    assert orbit_revolutions(far, np.zeros((200, 3))) == (0.0, 0.0)


def test_orbit_direction_is_irrelevant():
    asset = np.zeros((200, 3))
    assert orbit_revolutions(_circle(1.0)[::-1], asset)[0] == pytest.approx(orbit_revolutions(_circle(1.0), asset)[0], abs=1e-6)


def test_terminal_flyby_detects_the_known_sustained_fly_by_case():
    """C3 seed 130017 at the accepted foundation (873026ea): a 27x path-excess episode in which
    the attacker repeatedly swings outside the fixed 20 m `orbit_revolutions` radius between near
    misses, so that metric alone reports no orbit even though the attacker visibly loiters around
    the asset for hundreds of steps (see docs/journal_v2_terminal_speed_governor.md).

    Finite mission endurance (journal extension; see docs/journal_v2_mission_endurance.md) is now
    canonical for V2 and deliberately truncates this exact episode before the fly-by pattern can
    run its full course -- that is the fix. This test validates the underlying
    `terminal_flyby_metrics` diagnostic against the full (pre-truncation) pathological
    trajectory, so it explicitly disables endurance to reproduce the historical episode."""
    import dataclasses

    rec = record_episode("C3", "V2", "passive", 130017, detour_cf=False,
                          cfg_transform=lambda cfg: dataclasses.replace(cfg, enemy_endurance_enabled=False))
    metrics = terminal_flyby_metrics(rec)
    assert metrics["fly_by_count"] >= TERMINAL_ORBIT_FLYBY_THRESHOLD
    assert metrics["angular_travel_revolutions"] >= TERMINAL_ORBIT_REVOLUTION_THRESHOLD
    assert metrics["detected"] is True
    # the existing fixed-radius metric misses it (motivating this diagnostic in the first place)
    assert summarize(rec)["orbit_event"] is False


@pytest.mark.parametrize("level,seed", [("C2", 130065), ("C3", 130063), ("C3", 130081), ("C3", 130064), ("C2", 130054), ("C0", 130000)])
def test_terminal_flyby_does_not_flag_normal_captures(level, seed):
    rec = record_episode(level, "V2", "passive", seed, detour_cf=False)
    assert terminal_flyby_metrics(rec)["detected"] is False


def test_dense_audit_agrees_with_analytic_solver():
    rng = np.random.default_rng(3)
    for _ in range(300):
        p0a, p1a, p0b, p1b = (rng.uniform(-8, 8, 3) for _ in range(4))
        d, _ = dense_min_distance(p0a, p1a, p0b, p1b)
        analytic = first_contact(p0a, p1a, p0b, p1b, 2.0)
        if analytic is not None:
            assert d <= 2.0 + 1e-3
        elif d <= 2.0:
            pytest.fail("dense sampling found a contact the analytic solver missed")


# ------------------------------------------------------ segment analysis
def _row(mode, perceived=()):
    return {"behavior_mode": mode, "perceived_static_ids": list(perceived), "attacker_pos": [0, 0, 0],
            "asset_pos": [0, 0, 0], "attacker_mission_direction": None, "attacker_final_command": None, "detected": False}


def _extra(blocked=None, dynamic=None, otype=None, oid=None):
    return {"line_blocked_static": blocked, "line_blocked_dynamic": dynamic, "avoid_obstacle_type": otype,
            "avoid_obstacle_id": oid, "dynamic_obstacles": [], "defender_standby": False, "weave_bias": 0.0,
            "evasion_active": False}


def test_deviation_segment_entry_exit_and_classification():
    modes = ["DIRECT", "DIRECT", "DIRECT", "OBSTACLE_AVOID", "OBSTACLE_AVOID", "DIRECT", "DIRECT"]
    rows = [_row(m, perceived=(4,)) for m in modes]
    ex = [_extra() for _ in modes]
    ex[2] = _extra(blocked=4)               # line blocked at the decision before step 3
    ex[3] = _extra(blocked=4, otype="static", oid=4)
    ex[4] = _extra(blocked=None, otype="static", oid=4)
    segs = deviation_segments({"rows": rows, "extras": ex})
    assert len(segs) == 1
    s = segs[0]
    assert (s["activation_step"], s["last_avoid_step"], s["return_to_direct_step"]) == (3, 4, 5)
    assert s["duration_steps"] == 2 and s["kind"] == "static_line_blocked"
    assert s["associated_with_perceived_obstacle"] and s["line_clear_at_return"] is True


def test_unassociated_and_evasion_segments_are_flagged():
    rows = [_row("DIRECT"), _row("EVADE"), _row("DIRECT")]
    segs = deviation_segments({"rows": rows, "extras": [_extra(), _extra(), _extra()]})
    assert segs[0]["kind"] == "evasion" and not segs[0]["associated_with_perceived_obstacle"]
    rows = [_row("DIRECT"), _row("OBSTACLE_AVOID", perceived=(1,)), _row("DIRECT")]
    ex = [_extra(), _extra(otype="static", oid=9), _extra()]  # id 9 not in the perceived set
    segs = deviation_segments({"rows": rows, "extras": ex})
    assert not segs[0]["associated_with_perceived_obstacle"] and segs[0]["kind"] == "static_prediction_only"


def test_segment_terminated_while_avoiding_has_no_return_step():
    rows = [_row("DIRECT"), _row("OBSTACLE_AVOID", perceived=(2,))]
    segs = deviation_segments({"rows": rows, "extras": [_extra(), _extra(otype="static", oid=2)]})
    assert segs[0]["return_to_direct_step"] is None and segs[0]["terminated_while_avoiding"]


def test_linger_run_counts_avoiding_steps_with_a_clear_line():
    modes = ["DIRECT"] + ["OBSTACLE_AVOID"] * 5 + ["DIRECT"]
    rows = [_row(m, perceived=(1,)) for m in modes]
    ex = [_extra(), _extra(blocked=1, otype="static", oid=1)] + [_extra(otype="static", oid=1) for _ in range(5)]
    segs = deviation_segments({"rows": rows, "extras": ex})
    assert segs[0]["max_linger_run_steps"] == 3  # decisions for steps 3..5 saw a clear line (ex[0] is blocked at activation)


# ------------------------------------------------- recorder invariants
@pytest.fixture(scope="module")
def c0_records():
    return {c: record_episode("C0", "V2", c, 130000, counterfactual=True) for c in ("passive", "greedy")}


def test_recorder_matches_environment_counters(c0_records):
    for rec in c0_records.values():
        s = summarize(rec)
        assert s["overshoots"] == s["overshoots_from_trace"]
        assert s["steps"] == len(rec["rows"]) - 1
        assert rec["rows"][-1]["outcome"] == rec["final_info"]["outcome"]


def test_c0_v2_attacker_is_direct_and_unweaving(c0_records):
    s = summarize(c0_records["passive"])
    assert s["frac_direct"] == 1.0 and s["max_abs_weave_bias"] == 0.0 and s["max_direct_command_error_deg"] < 0.1
    assert s["overshoots"] == 0 and not s["orbit_event"] and s["captured"]
    assert s["missed_attacker_contacts"] == 0 and s["missed_defender_contacts"] == 0


def test_attacker_track_is_independent_of_the_defender(c0_records):
    a = np.array([r["attacker_pos"] for r in c0_records["passive"]["rows"]])
    b = np.array([r["attacker_pos"] for r in c0_records["greedy"]["rows"]])
    m = min(len(a), len(b)) - 1
    assert m > 5 and np.allclose(a[:m], b[:m], atol=1e-6)


def test_counterfactual_endpoint_semantics_would_not_have_ended_the_step(c0_records):
    rec = c0_records["greedy"]
    assert rec["final_info"]["terminal_event"] == "interception"
    cf = rec["counterfactual"]
    assert cf is not None and cf["terminal"] is False and cf["defender_enemy_dist"] > rec["config"]["intercept_radius"]
    # The swept contact fraction places the interception strictly inside the step.
    assert 0.0 < rec["final_info"]["terminal_event_fraction"] < 1.0


def test_recording_does_not_perturb_the_episode():
    plain = record_episode("C2", "V2", "greedy", 130001, counterfactual=False)
    with_cf = record_episode("C2", "V2", "greedy", 130001, counterfactual=True)
    assert [r["attacker_pos"] for r in plain["rows"]] == [r["attacker_pos"] for r in with_cf["rows"]]
    assert plain["final_info"]["outcome"] == with_cf["final_info"]["outcome"]


def test_v1_endpoint_audit_reports_the_known_missed_crossings():
    rec = record_episode("C0", "V1", "passive", 130000)
    assert rec["final_info"]["mission_contact_mode"] == "endpoint"
    assert len(rec["missed_attacker"]) >= 4  # the near-passes seen in the original visualization
    assert all(m["min_distance"] <= rec["config"]["threat_radius"] < m["endpoint_distance"] for m in rec["missed_attacker"])


# ------------------------------------------------ constructed obstacle scenarios
def _scenario(spec, start=(-20.0, 0.0, 3.0), asset=(45.0, 0.0, 0.0), asset_step=None):
    """Hand-placed obstacles, attacker at rest at `start`, stationary asset unless `asset_step` is given."""
    import dataclasses
    import types

    def transform(cfg):
        return dataclasses.replace(cfg, obstacle_layout_mode="fixed", obstacle_fixed_spec=spec, dynamic_obstacles_enabled=False)

    def setup(env):
        env._enemy_pos = np.array(start, dtype=np.float32)
        env._enemy_vel = np.zeros(3, dtype=np.float32)
        env._soldier_pos = np.array(asset, dtype=np.float32)
        if asset_step is None:
            env._move_soldier = lambda: None
        else:
            # Bound to the env so the counterfactual deep copies move their own asset.
            delta = np.array(asset_step, dtype=np.float32)
            env._move_soldier = types.MethodType(
                lambda self: setattr(self, "_soldier_pos", (self._soldier_pos + delta).astype(np.float32)), env)

    return record_episode("C2", "V2", "passive", 0, detour_cf=True, cfg_transform=transform, setup=setup)


def _min_obstacle_clearance(rec) -> float:
    return min(r["nearest_obstacle_clearance"] for r in rec["rows"] if r["nearest_obstacle_clearance"] is not None)


def _assert_valid_avoidance_run(rec):
    """Invariants shared by the obstacle-negotiation scenarios (no clearance-margin claim)."""
    s = summarize(rec)
    assert s["captured"] and s["outcome"] == "soldier_caught"  # mission completed, no obstacle collision
    assert "collision" not in s["outcome"] and _min_obstacle_clearance(rec) > 0.0  # physical non-penetration only
    assert s["segments_unassociated"] == 0
    assert s["segments_not_returned"] == 0 and not any(g["terminated_while_avoiding"] for g in deviation_segments(rec))
    assert not s["orbit_event"] and s["orbit_revolutions_max"] < 0.5  # no orbiting
    assert s["n_avoidance_segments"] >= 1
    return s


def test_blocking_obstacle_detour_is_counterfactually_necessary_and_returns_to_direct():
    rec = _scenario(((15.0, 0.0, 5.0, 4.0, 4.0, 5.0),))
    segs = deviation_segments(rec)
    assert len(segs) == 1 and segs[0]["necessity"] == "necessary_collision"
    # DIRECT is rejoined when the actual turn-limited DIRECT rollout is collision-free (not when an
    # uninflated straight line is): the line may still clip the box top while descent-rate limits keep
    # the vehicle above it. The invariant is that the rest of the flight is DIRECT and never touches it.
    assert segs[0]["return_to_direct_step"] is not None
    after = rec["rows"][segs[0]["return_to_direct_step"]:]
    assert after and all(r["behavior_mode"] == "DIRECT" for r in after[:-1])
    assert all(r["nearest_obstacle_clearance"] is None or r["nearest_obstacle_clearance"] > 0.0 for r in after)
    s = summarize(rec)
    assert s["captured"] and s["max_lateral_deviation_m"] > 3.0 and s["detours_unnecessary"] == 0


def test_nearby_nonblocking_obstacle_causes_no_deviation():
    rec = _scenario(((15.0, 14.0, 5.0, 4.0, 4.0, 5.0),))
    s = summarize(rec)
    assert s["n_avoidance_segments"] == 0 and s["frac_direct"] == 1.0 and s["captured"]
    assert s["max_lateral_deviation_m"] < 1e-3


def test_staggered_obstacles_are_negotiated_and_return_to_direct(record_property):
    """Two obstacles on the straight line to the asset, 30 m apart.

    The planner may negotiate them as two excursions or as one continuous maneuver
    (the faithful predictor currently climbs over both); the scientific content is
    that the obstacles are really in the way, are negotiated without a collision,
    and the attacker ends up back in DIRECT and completes the mission. The number
    of excursions is deliberately NOT asserted.

    KNOWN LIMITATION, recorded rather than hidden: the minimum obstacle clearance is
    reported as a diagnostic only. It is about 0.02 m in the current maneuver (the
    final DIRECT approach skirts the second obstacle's corner) because the trigger
    and release rules use physical geometry only. That is NOT a validated operational
    safety margin; robust clearance semantics are unresolved and are to be addressed
    together with the waypoint-geometry redesign.
    """
    from uav_defend.obstacles.geometry import AABBObstacle

    spec = ((5.0, 2.0, 5.0, 3.0, 3.0, 5.0), (35.0, -2.0, 5.0, 3.0, 3.0, 5.0))
    start, asset = (-20.0, 0.0, 3.0), (45.0, 0.0, 0.0)
    rec = _scenario(spec, start, asset)
    s = _assert_valid_avoidance_run(rec)
    # Both obstacles genuinely lie on the straight attacker->asset line ...
    for obstacle in spec:
        box = AABBObstacle.from_center_half_extents(obstacle[:3], obstacle[3:])
        assert box.segment_intersects(np.array(start), np.array(asset))
    # ... and both were perceived by the attacker during the flight.
    perceived = {i for row in rec["rows"] for i in row["perceived_static_ids"]}
    assert perceived >= {0, 1}
    assert deviation_segments(rec)[0]["necessity"] == "necessary_collision"
    assert rec["rows"][-1]["behavior_mode"] == "DIRECT" or rec["rows"][-1]["outcome"] == "soldier_caught"
    record_property("min_obstacle_clearance_m", _min_obstacle_clearance(rec))
    record_property("avoidance_excursions", s["n_avoidance_segments"])


def test_separated_obstacles_require_two_distinct_returned_excursions(record_property):
    """AVOID -> DIRECT -> AVOID -> DIRECT.

    Geometry: a full-height pillar 10 m ahead of the attacker blocks the initial line,
    and a second full-height pillar sits off that line. The asset drifts steadily
    sideways (a deterministic stand-in for its random walk), so the straight line to
    it only swings into the second pillar after the first has been cleared and the
    attacker is flying DIRECT. Full-height pillars (top at the 30 m ceiling) exclude
    a climb, so each encounter is a lateral bypass. Chosen so the second encounter is
    genuinely required, not forced by production code.
    """
    spec = ((-38.0, 0.0, 15.0, 3.0, 3.0, 15.0), (38.0, -10.0, 15.0, 3.0, 3.0, 15.0))
    rec = _scenario(spec, start=(-48.0, 0.0, 3.0), asset=(48.0, 0.0, 0.0), asset_step=(0.0, -0.5, 0.0))
    s = _assert_valid_avoidance_run(rec)
    segs = deviation_segments(rec)
    assert len(segs) >= 2
    first, second = segs[0], segs[1]
    # each is a real obstacle conflict (direct flight would have collided) with a distinct obstacle
    assert first["necessity"] == second["necessity"] == "necessary_collision"
    assert first["obstacle_id"] != second["obstacle_id"] and {first["obstacle_id"], second["obstacle_id"]} <= {0, 1}
    # it returned to DIRECT after the first, and stayed there for a meaningful interval
    assert first["return_to_direct_step"] is not None
    assert second["activation_step"] - first["return_to_direct_step"] >= 5
    modes = [r["behavior_mode"] for r in rec["rows"][1:]]
    gap = modes[first["return_to_direct_step"] - 1 : second["activation_step"] - 1]
    assert gap and all(m == "DIRECT" for m in gap)
    assert second["return_to_direct_step"] is not None
    assert modes[-1] == "DIRECT"
    record_property("min_obstacle_clearance_m", _min_obstacle_clearance(rec))
    record_property("avoidance_excursions", s["n_avoidance_segments"])


def test_detour_counterfactual_does_not_perturb_the_recorded_episode():
    a = record_episode("C2", "V2", "passive", 130002, detour_cf=False)
    b = record_episode("C2", "V2", "passive", 130002, detour_cf=True)
    assert [r["attacker_pos"] for r in a["rows"]] == [r["attacker_pos"] for r in b["rows"]]
    assert b["detour_cf"] and all(v["collides"] for v in b["detour_cf"].values())


def test_unnecessary_detours_are_detected_in_the_legacy_attacker():
    rec = record_episode("C2", "V1", "passive", 130000, detour_cf=True)
    s = summarize(rec)
    assert s["detours_unnecessary"] > 0  # V1's persistent-release avoidance makes pointless detours
