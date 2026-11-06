"""A1-CBF defender avoidance (journal_cbf_attacker, defender_cbf / Prompt 4): the defender UAV as an adversarial
MOVING hazard, handled LOCALLY by the CBF-QP (never by the static visibility-graph route planner -- re-planning a
global route around a constantly-moving interceptor would be architecturally wrong, see README section 3).

Information contract (brief section 4): A1-CBF receives defender state ONLY through the SAME established gate the
PPO/BC curriculum's attacker observation already uses (`EnvConfig.enemy_defender_detection_range`, canonical
25 m -- see `experiments.adaptive_attacker_env.build_attacker_observation`): the TRUE 3-D attacker-defender range
must be <= this gate for ANY defender-relative information to exist. Ground-truth defender velocity is technically
also exposed by that SAME PPO-era contract once gated, but A1-CBF deliberately does NOT read it directly --
instead it estimates defender velocity by finite-differencing consecutive GATED POSITION observations
(`DefenderVelocityEstimator`), a strictly more conservative sensing assumption appropriate for a safety-critical
filter (an onboard position sensor is a far more defensible real-world assumption than a perfect velocity feed).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

_EPS = 1e-9

# Established attacker-visible defender-information gate (journal PPO/BC curriculum; see
# EnvConfig.enemy_defender_detection_range's canonical default and
# experiments.adaptive_attacker_env.build_attacker_observation). A1-CBF reuses this SAME gate -- it is NOT a new
# parameter invented for this stage.
DEFENDER_DETECTION_RANGE = 25.0

# Defender CBF safety radius (brief section 5): uses the SAME NUMERIC VALUE as the canonical
# EnvConfig.unsafe_intercept_radius (3.5 m, verified in uav_defend/config/env_config.py) -- but note this is a
# DIFFERENT geometric quantity. unsafe_intercept_radius gates the ENEMY-SOLDIER (attacker-asset) distance at the
# moment of physical interception (dist_es in SoldierEnv.step()'s termination logic); R_D here gates the
# ATTACKER-DEFENDER distance for the CBF barrier (brief section 6: d_D = ||p_A - p_D|| - R_D). The brief directs
# reuse of this SAME frozen 3.5 m number -- 1.0 m of margin OUTSIDE the actual physical interception radius
# (EnvConfig.intercept_radius, 2.5 m, UNCHANGED, also attacker-defender) -- applied to the new attacker-defender
# relationship. Frozen; never tuned after evaluation.
DEFENDER_SAFETY_RADIUS = 3.5
CBF_ALPHA_DEFENDER = 1.0


@dataclass
class DefenderVelocityEstimator:
    """Finite-difference defender-velocity estimator over GATED position observations only (never ground-truth
    velocity). Stateful across one episode's external 0.5 s steps; `dt` is the external step interval (NOT
    `dt_sub` -- gated position observations only refresh once per external step, since the detection gate itself
    is evaluated once per `SoldierEnv.step()`, matching the established attacker-observation contract's own
    refresh cadence).

    Contract (brief section 4, frozen):
      * First-ever valid observation: velocity estimate = zero, `valid=False` (no prior sample to differentiate
        against -- NOT an arbitrary guess).
      * A gap (the defender becomes unobservable for >=1 step, then observable again): treated as a FRESH first
        observation -- NEVER differentiated across the gap (would silently divide a real displacement by a
        multiple of `dt`, understating the TRUE closing speed in exactly the scenario where safety matters most).
      * Deterministic: a pure function of the observation sequence, no randomness.
    """

    dt: float
    _last_valid_pos: np.ndarray | None = field(default=None, init=False)
    _last_valid_step: int | None = field(default=None, init=False)

    def reset(self) -> None:
        self._last_valid_pos = None
        self._last_valid_step = None

    def update(self, step_index: int, position: np.ndarray | None) -> tuple[np.ndarray, bool]:
        """`position` is the GATED defender position this step, or `None` if not observable (gate failed).
        Returns `(velocity_estimate, valid)`."""
        if position is None:
            # Gap: do NOT advance `_last_valid_*` (a later re-acquisition must not differentiate across this gap).
            return np.zeros(3, dtype=np.float64), False
        position = np.asarray(position, dtype=np.float64).reshape(3)
        if self._last_valid_pos is None or (step_index - self._last_valid_step) != 1:
            velocity = np.zeros(3, dtype=np.float64)
            valid = False
        else:
            velocity = (position - self._last_valid_pos) / self.dt
            valid = True
        self._last_valid_pos = position.copy()
        self._last_valid_step = step_index
        return velocity, valid


def gated_defender_position(attacker_pos: np.ndarray, defender_pos: np.ndarray, *, detection_range: float = DEFENDER_DETECTION_RANGE) -> np.ndarray | None:
    """The established attacker-visible defender-position contract: the TRUE 3-D attacker-defender range must be
    <= `detection_range`, else `None` (no leakage of ground-truth state before the gate)."""
    attacker_pos = np.asarray(attacker_pos, dtype=np.float64)
    defender_pos = np.asarray(defender_pos, dtype=np.float64)
    if float(np.linalg.norm(attacker_pos - defender_pos)) <= detection_range:
        return defender_pos.copy()
    return None


@dataclass(frozen=True)
class DefenderHazard:
    """One active defender-avoidance CBF constraint (brief sections 6-8). `normal` points from the defender toward
    the attacker (mirrors `StaticHazard`'s outward-normal convention)."""

    normal: np.ndarray        # unit vector from defender toward attacker
    d: float                  # ||r_D|| - R_D (signed clearance to the safety radius)
    h: float                  # barrier margin: d - stopping_distance_margin
    closing_speed: float      # s_D = max(0, -n_D . v_rel)
    stopping_margin: float    # d_stop,D
    distance: float           # raw ||r_D|| (attacker-defender range), for diagnostics
    defender_velocity: np.ndarray  # the ESTIMATED defender velocity used (not ground truth)


def build_defender_hazard(
    attacker_pos: np.ndarray, attacker_vel: np.ndarray, defender_pos: np.ndarray, defender_vel: np.ndarray, *,
    safety_radius: float = DEFENDER_SAFETY_RADIUS, max_accel: float, dt_sub: float,
) -> DefenderHazard:
    """Builds the defender hazard from the ATTACKER-AVAILABLE (gated position, estimated velocity) state only.
    Caller is responsible for gating (only call this when the defender is observable this step)."""
    attacker_pos = np.asarray(attacker_pos, dtype=np.float64).reshape(3)
    attacker_vel = np.asarray(attacker_vel, dtype=np.float64).reshape(3)
    defender_pos = np.asarray(defender_pos, dtype=np.float64).reshape(3)
    defender_vel = np.asarray(defender_vel, dtype=np.float64).reshape(3)

    r = attacker_pos - defender_pos
    distance = float(np.linalg.norm(r))
    if distance > _EPS:
        normal = r / distance
    else:
        # Numerically co-located (never expected in practice -- physical interception would already have
        # terminated the episode well before this): a safe, deterministic fallback (no RNG call), matching the
        # SAME convention used elsewhere in this module family (e.g. CBFRouteNavigator's degenerate-range
        # fallback) -- never NaN, never a function of simulation history.
        normal = np.array([1.0, 0.0, 0.0])
    d = distance - safety_radius

    v_rel = attacker_vel - defender_vel
    s = max(0.0, float(-np.dot(normal, v_rel)))
    d_stop = (s * s) / (2.0 * max_accel) + s * dt_sub
    h = d - d_stop
    return DefenderHazard(normal=normal, d=d, h=h, closing_speed=s, stopping_margin=d_stop, distance=distance, defender_velocity=defender_vel.copy())


def predict_defender_positions(defender_pos: np.ndarray, defender_vel: np.ndarray, *, dt_sub: float, n_substeps: int) -> list[np.ndarray]:
    """Constant-VELOCITY extrapolation of the defender's position over the substeps of one external step (brief
    section 9): `p_D(tau) = p_D(t) + v_D(t) * tau` for `tau in {0, dt_sub, 2*dt_sub, ...}`. Uses ONLY the
    currently-estimated defender velocity -- no acceleration is predicted. Refreshed from the newly observed
    defender state at the START of the NEXT external step (never carried over)."""
    defender_pos = np.asarray(defender_pos, dtype=np.float64).reshape(3)
    defender_vel = np.asarray(defender_vel, dtype=np.float64).reshape(3)
    return [defender_pos + defender_vel * (i * dt_sub) for i in range(n_substeps)]
