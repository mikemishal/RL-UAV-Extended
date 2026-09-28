# Defender Command-Pipeline Audit (Phase 2) & Collision-Semantics Audit (Phase 21)

Traced directly from `uav_defend/envs/soldier_env.py` as of this sub-task
(building on `journal-behavior-arbitration-v1`). No controller equations
were changed to produce this document.

## Phase 2: defender command pipeline

```
controller.act(obs, policy_info)          <- Greedy / PN / Lead / RMPC /
        |                                     PPO / PPO-Kalman / LR-PPO
        v                                     (ALL external to SoldierEnv;
   action: np.ndarray shape (3,)               action is the ONLY channel)
        |
        v
SoldierEnv.step(action)
        |
        v
SoldierEnv._move_defender(action)   <-- SINGLE entry point for EVERY
        |                                controller family, unmodified
        |                                upstream of this call
        v
   u_desired = normalize(action)     (direction; zero if action norm<=eps)
        |
        v
   [NEW] shared obstacle-safety layer (this sub-task):
       - defender_obstacle_avoidance_enabled == False -> u_safe = u_desired
         (bit-for-bit no-op, exact conference/legacy behavior)
       - == True -> EnemyObstaclePerception.update() (defender's OWN
         instance) -> PerceivedObstacleSet -> ObstacleAvoidancePlanner.plan(
         u_base=u_desired, target_position=enemy_pos, ...) -> u_safe
        |
        v
   desired_velocity = v_d * u_safe
        |
        v
   _advance_velocity(...)             (defender_max_accel/turn_rate/
        |                                climb_rate/descent_rate limits --
        |                                UNCHANGED, same call as before)
        v
   new_pos = defender_pos + next_vel * dt
        |
        v
   _apply_boundary(...)                (domain reflection -- unchanged)
        |
        v
   self._defender_pos / self._defender_vel updated (ground truth)
        |
        v
   [existing, unchanged] swept-motion ground-truth collision diagnostics
   (defender_obstacle_collision / defender_dynamic_obstacle_collision,
   defender_min_obstacle_clearance, etc.) computed from the NEW position,
   using the FULL ground-truth world -- independent of awareness mode.
        |
        v
   _get_info() assembles the observation/diagnostics dict returned to the
   caller (RL wrappers additionally build a fixed-dimension obs vector
   from a SUBSET of info -- untouched by this sub-task, see Phase 12 below).
```

### Per-controller-family notes
- **Greedy / PN / Lead**: hand-designed, stateless-per-step (Lead keeps a
  small amount of internal state for the lead-time estimate). `action`
  is a unit-ish direction vector; `_move_defender` normalizes it again
  regardless.
- **RMPC**: `action` is the first control of a CEM-optimized sequence
  (`RMPCPolicy.act()`); `_move_defender` treats it identically to any
  other controller's output -- RMPC has no way to see the obstacle-safety
  layer or know it exists.
- **PPO / PPO-Kalman**: `action` is the raw SB3 `model.predict()` output
  (already in the environment's `[-1, 1]^3` action space); passed through
  `_move_defender` unmodified upstream.
- **LR-PPO**: `action = compose_lead_residual(lead_direction, r_raw,
  residual_max_angle_deg)` -- the CANONICAL Lead + bounded residual
  composition (30 degree bound), computed ENTIRELY inside
  `LeadResidualPPOPolicyWrapper.act()`, before `_move_defender` ever sees
  it. The obstacle-safety layer operates strictly AFTER this composition
  (Phase 11: "obstacle navigation occurs AFTER canonical LR-PPO
  composition") -- confirmed by inspection: `_move_defender` receives
  `action` as an opaque `np.ndarray`, with zero knowledge of how it was
  produced.

### Fairness/isolation properties (Phase 26), confirmed by inspection
- Every controller family's `action` passes through the exact SAME
  `_move_defender()` call -- there is no per-family branch anywhere in the
  safety-layer code.
- `uav_defend/policies/sanitize.py::build_policy_info()` uses an
  ALLOW-list (`_MEASUREMENT_KEYS`/`_KALMAN_KEYS`), and no obstacle-related
  key was added to either list -- so no controller (frozen or otherwise)
  can access `defender_awareness_mode`, `defender_avoidance_active`, or
  any other new diagnostic field, even if it tried to read `info` directly
  (Phase 12).
- All controllers/instances in a given benchmark run share the identical
  `env.reset(seed=...)` call, which alone determines obstacle layout,
  hostile behavior, and soldier behavior -- matched scenarios (Phase 26).

## Phase 21: physical collision semantics (audit, no changes made)

Directly confirmed by inspecting `SoldierEnv.step()`'s termination logic
(the `intercepted` / `soldier_caught` / `unsafe_intercept` / `timeout`
outcome computation): **it references ONLY `dist_de` (defender-enemy
distance) and `dist_es` (enemy-soldier distance) plus the step counter.**
No obstacle-collision field of any kind (`defender_obstacle_collision`,
`defender_dynamic_obstacle_collision`, `enemy_..._collision`, or the new
`defender_avoidance_*` diagnostics) is read anywhere in this logic.

This matches the SAME convention already explicitly documented for the
attacker side and for the existing (pre-defender-awareness) clutter
benchmark (see `experiments/clutter_benchmark_protocol.py`'s own
"OBSTACLE SEMANTICS" docstring section, written for an earlier sub-task):

> Defender-obstacle and hostile-obstacle collisions are DIAGNOSTIC ONLY:
> they never stop motion, never alter position/velocity, never set
> `terminated`, and never alter `reward`/`outcome`.

**Current state, for all four combinations**:
| | static obstacle | dynamic obstacle |
|---|---|---|
| defender | diagnostic only (no effect on physics/outcome/reward) | diagnostic only |
| attacker (hostile) | diagnostic only | diagnostic only |

No semantics were changed in this sub-task. This audit is a read-only
confirmation, per the explicit instruction not to silently change this
before a separate, controlled decision.

**Recommendation for the next phase** (not implemented here): before any
RL retraining that is meant to teach a controller to actually AVOID
obstacles (rather than just react at the shared low-level safety layer),
physical collision consequences should be introduced deliberately and
explicitly -- e.g. a bounded, clearly-labeled reward penalty and/or a rare
hard-collision terminal condition -- as its OWN reviewed change, not
bundled into observation-design or retraining work.
