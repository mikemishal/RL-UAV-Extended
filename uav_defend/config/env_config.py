"""Environment configuration parameters."""

from dataclasses import dataclass


@dataclass
class EnvConfig:
    """Configuration for the UAV Defend environment."""
    
    # Domain: 3D engagement volume.
    #   x, y in [-L, L] (horizontal extent)
    #   z in [0, max_altitude] (altitude)
    L: float = 50.0  # Half-size of the horizontal square domain
    max_altitude: float = 30.0  # H: Upper vertical bound of the engagement volume

    # Enemy spawn altitude range (must lie within [0, max_altitude])
    enemy_spawn_altitude_min: float = 10.0
    enemy_spawn_altitude_max: float = 30.0
    
    # Time parameters
    max_steps: int = 2000  # Tmax: Maximum steps before episode ends
    dt: float = 0.5  # Time step duration (∆t)
    
    # Numerical stability
    eps: float = 1e-8  # ε: Small value for numerical stability
    
    # =========================================================================
    # Speed Configuration (realistic but training-friendly defaults)
    # - Soldier is intentionally much slower than drones (human on foot)
    # - Defender is faster than enemy so RL training remains feasible
    # - These values work well with dt=0.5 timestep for numerical stability
    # =========================================================================
    
    # Soldier movement parameters
    v_s: float = 1.5  # Soldier speed (slow human movement, much slower than drones)
    # Protected-asset (soldier) stochastic motion model (journal model-study extension):
    #   "legacy_gaussian_random_walk" (default): per-step IID Gaussian displacement, scale
    #     v_s*dt -- the ORIGINAL, zero-velocity-persistence model. Bit-identical to all prior
    #     behavior; this is the only model V1 and the canonical V2 benchmark use.
    #   "gauss_markov_velocity": an optional, config-gated alternative with a temporally
    #     correlated (AR(1)/Gauss-Markov) velocity process instead of an IID position walk --
    #     see uav_defend/envs/soldier_env.py:_move_soldier for the exact recurrence. Not
    #     selected by any canonical builder; a model-study artifact only.
    asset_motion_model: str = "legacy_gaussian_random_walk"
    # Velocity persistence time constant (s) for "gauss_markov_velocity"; rho = exp(-dt/tau).
    # Unused (and not validated) under the legacy model.
    asset_motion_tau_s: float = 1.0
    
    # Enemy drone parameters (weaving pursuit)
    v_e: float = 12.0  # Enemy drone speed (realistic and threatening)
    rho: float = 0.85  # AR(1) coefficient for weave bias persistence (lower = faster direction changes)
    sigma_a: float = 0.5  # Weave bias noise standard deviation (higher = more lateral movement)
    sigma_e: float = 0.15  # Heading noise standard deviation
    weave_amplitude: float = 1.5  # Multiplier for lateral weave component
    
    # Defender drone parameters
    v_d: float = 18.0  # Defender speed (faster than enemy, makes interception feasible for RL training)
    
    # =========================================================================
    # Constrained 3D Point-Mass Dynamics
    # - These are PROVISIONAL SIMULATION DEFAULTS, not validated hardware specs.
    # - v_d / v_e (above) are the maximum total speeds for defender / enemy.
    # - Each vehicle maintains a persistent 3D velocity that approaches a
    #   commanded desired velocity subject to: max acceleration, max
    #   horizontal turn rate, and max climb/descent rate (vertical speed).
    # - This is a 3D constrained point-mass model, NOT six-DOF or
    #   aerodynamic flight dynamics.
    # =========================================================================
    
    # Defender dynamics
    defender_max_accel: float = 8.0            # m/s^2
    defender_max_turn_rate_deg: float = 90.0   # deg/s (horizontal heading-rate limit)
    defender_max_climb_rate: float = 6.0       # m/s
    defender_max_descent_rate: float = 6.0     # m/s
    
    # Hostile-UAV dynamics
    enemy_max_accel: float = 6.0                # m/s^2
    enemy_max_turn_rate_deg: float = 75.0       # deg/s (horizontal heading-rate limit)
    enemy_max_climb_rate: float = 5.0           # m/s
    enemy_max_descent_rate: float = 5.0         # m/s
    
    # =========================================================================
    # Reactive Hostile-UAV Evasion (provisional simulation defaults, NOT
    # hardware-derived quantities)
    # - The hostile UAV keeps its primary mission (reach the soldier) but
    #   increasingly biases its desired direction away from the defender as
    #   the defender closes within enemy_evasion_radius. This is a
    #   parameterized reactive evasive policy, not intelligent adversarial
    #   RL, optimal-attacker, or game-theoretic behavior.
    # - Evasion is independent of defender detection/sensor state: it models
    #   the hostile UAV's own local awareness of an approaching interceptor,
    #   not the defender's tracking of the hostile UAV.
    # =========================================================================
    enemy_evasion_enabled: bool = True    # If False, hostile guidance reduces to pursuit + weave (no evasion)
    enemy_evasion_radius: float = 20.0    # m; 3D defender-enemy range at which evasion begins
    enemy_evasion_gain: float = 0.75      # Dimensionless relative evasion strength, lambda_e in [0, 1]

    # =========================================================================
    # Adaptive Attacker v1 defender-sensing contract (journal extension).
    # Explicit, separate from the reactive-evasion parameters above: this
    # field gates ONLY the defender-relative features exposed in the
    # LEARNED attacker's OWN observation vector (see
    # experiments/adaptive_attacker_*.py) -- it does NOT affect
    # `_compute_enemy_evasion()`'s pre-existing, ground-truth-based reactive
    # evasion mechanics used by the hand-designed hostile model above
    # (audited and intentionally left unchanged; no correctness issue was
    # found there). Below this range, the learned attacker's observation
    # zeros out all defender-relative features (as if undetected); at or
    # within it, exact defender-relative position/velocity/range/closing-
    # rate are exposed (perfect measurement after detection, consistent
    # with the existing hostile model's post-detection assumptions --
    # Phase 7 explicitly permits this for Adaptive Attacker v1).
    # =========================================================================
    enemy_defender_detection_range: float = 25.0  # m; 3D range at which the LEARNED attacker's observation gains defender-relative features
    
    # RL reward shaping parameters
    # NOTE: there is no Kalman-only "tracking error improvement" reward term.
    # Tracking error is an EVALUATION metric only (see info["tracking_error"]),
    # never a training reward, so Direct and Kalman tracks optimize exactly
    # the same task reward (see SoldierEnv.step()).
    reward_intercept: float = 100.0  # Reward for intercepting enemy (WIN)
    reward_soldier_caught: float = -100.0  # Penalty for enemy catching soldier (LOSS)
    reward_unsafe_intercept: float = -150.0  # Severe penalty for unsafe intercept (discourage risky behavior)
    reward_timeout: float = -100.0  # Penalty for timeout (failed to intercept)
    reward_progress_scale: float = 5.0  # Scale for distance progress reward (closing on enemy)
    reward_time_penalty: float = -0.05  # Small time penalty per step
    reward_proximity_warning: float = -0.5  # Per-step penalty when enemy is close to soldier (increased)

    # Physical obstacle-collision terminal rewards (journal extension,
    # Phase 3/4/5). Defender collisions are the defender's OWN failure
    # (destroyed vehicle) and default to the same severity as the existing
    # worst-case penalty (reward_unsafe_intercept). Hostile (enemy)
    # collisions are an ENVIRONMENTAL failure of the attacker's mission,
    # NOT a defender interception -- they default to NEUTRAL (0.0) so the
    # defender is never automatically rewarded as if it had intercepted;
    # mission-outcome accounting (info["outcome"]) is kept separate from
    # this RL reward choice, and both are independently configurable.
    reward_defender_static_obstacle_collision: float = -150.0
    reward_defender_dynamic_obstacle_collision: float = -150.0
    reward_enemy_static_obstacle_collision: float = 0.0
    reward_enemy_dynamic_obstacle_collision: float = 0.0
    
    # =========================================================================
    # Distance Thresholds (geometry parameters)
    # - detection_radius >> intercept_radius: early warning for reaction time
    # - threat_radius: immediate danger zone around soldier
    # - intercept_radius: neutralization distance (defender catches enemy)
    # - unsafe_intercept_radius: collateral risk zone around soldier
    # =========================================================================
    detection_radius: float = 15.0  # Maximum sensing range for detecting the enemy drone
    intercept_radius: float = 2.5   # Distance at which defender neutralizes/intercepts the enemy
    threat_radius: float = 2.0      # Enemy reaches soldier and causes mission failure
    unsafe_intercept_radius: float = 3.5  # If intercept occurs this close to soldier, it's a failure

    # =========================================================================
    # Pre-Detection Engagement Semantics (revised canonical scenario)
    # - The defender begins co-located with the protected soldier
    #   (see SoldierEnv.reset()). Before hostile detection, the intended
    #   engagement scenario is a stationary standby/escort posture: the
    #   defender has not yet launched to intercept anything.
    # - defender_standby_until_detection=True (the REVISED canonical
    #   default): while the hostile is undetected, the defender is held
    #   EXACTLY co-located with the soldier (position and velocity), and the
    #   supplied controller action has NO effect. Control effectively hands
    #   off to the selected interception controller beginning with the
    #   FIRST step whose action is applied after detection already held true
    #   at the start of that step (see SoldierEnv.step() for the exact
    #   discrete-time handoff). This applies identically to every
    #   controller (Greedy, PN, Lead, PPO, PPO-Kalman, ...).
    # - defender_standby_until_detection=False (LEGACY mode, retained only
    #   for historical reproducibility / diagnostics, e.g.
    #   experiments/policy_wrappers.py's EscortUntilDetectionPolicy-based
    #   acquisition-ablation study): the supplied action is applied to the
    #   defender every step from the very start of the episode, exactly as
    #   in the original (pre-revision) environment behavior.
    # - See docs/revised_predetection_protocol.md for the full rationale.
    # =========================================================================
    defender_standby_until_detection: bool = True

    
    # =========================================================================
    # Kalman Tracking and Sensing Noise Configuration
    # - measurement_var is the single official sensor-noise parameter.
    # - After detection, a noisy enemy-position measurement is generated each step
    #   with covariance measurement_var * I for all controller modes.
    # - The Direct/measurement track additionally derives a finite-difference
    #   measurement velocity from consecutive measurements (see
    #   SoldierEnv._update_detection); no filtering is applied to it.
    # - If use_kalman_tracking=True, Kalman filter smooths these measurements.
    # - If use_kalman_tracking=False (the default), policies/observations receive
    #   the raw noisy measurement plus its finite-difference velocity estimate.
    #   Kalman filtering is an explicit opt-in estimator, not the environment's
    #   implicit default.
    # =========================================================================
    use_kalman_tracking: bool = False  # If True, use Kalman filter for enemy state estimation
    process_var: float = 1.0           # Process noise variance (higher = trust measurements more)
    measurement_var: float = 0.5       # Enemy position measurement noise variance (per axis)
    lead_time: float = 0.0             # Prediction lead time for extrapolating enemy position (seconds)

    # =========================================================================
    # Obstacle Configuration (journal extension: obstacle-aware 3-D
    # interception; see uav_defend/obstacles/ for geometry/layout code).
    #
    # OFF BY DEFAULT: obstacles_enabled=False reproduces the conference-
    # baseline environment exactly -- no obstacle layout is generated, no
    # RNG stream is consumed for obstacles, and no other behavior changes.
    # This phase adds obstacle GEOMETRY/DATA ONLY: no collision
    # termination, no avoidance guidance, no observation changes.
    # =========================================================================
    obstacles_enabled: bool = False
    obstacle_layout_mode: str = "none"  # "none" | "fixed" | "random"
    obstacle_count: int = 0             # number of obstacles to place ("random" mode only)
    obstacle_min_size: float = 2.0      # min horizontal (x/y) full width/length, meters
    obstacle_max_size: float = 6.0      # max horizontal (x/y) full width/length, meters
    obstacle_min_height: float = 2.0    # min full vertical extent (z-height), meters; obstacles are ground-based (z_min=0)
    obstacle_max_height: float = 10.0   # max full vertical extent (z-height), meters
    obstacle_clearance_from_asset: float = 5.0   # min clearance (m) from the protected asset / defender initial position
    obstacle_clearance_from_spawn: float = 3.0   # min clearance (m) from a known enemy spawn point ("random" mode)
    obstacle_min_separation: float = 0.0         # min clearance (m) required between distinct obstacles (0 = no overlap only)
    obstacle_max_placement_attempts: int = 500   # rejection-sampling attempt budget PER obstacle ("random" mode)
    # "fixed" mode only: tuple of (center_x, center_y, center_z, half_x, half_y, half_z) per obstacle
    obstacle_fixed_spec: tuple[tuple[float, float, float, float, float, float], ...] = ()
    # Static-obstacle map-knowledge classification (journal extension: attacker
    # perception layer; see uav_defend/obstacles/perception.py). Pure metadata
    # -- never affects geometry/physics. "random" mode: fraction (by DETERMINISTIC
    # placement order, no RNG draw) of obstacles classified map_known=False;
    # default 0.0 preserves prior behavior (all obstacles map-known) exactly,
    # including RNG-consumption sequence. "fixed" mode: explicit per-obstacle
    # tuple (empty = all map-known).
    obstacle_unmapped_fraction: float = 0.0
    obstacle_fixed_map_known: tuple[bool, ...] = ()

    # =========================================================================
    # Obstacle-compatible exogenous motion (journal extension, Phase 3):
    #
    # Protected asset: the existing planar Gaussian random-walk PROPOSAL is
    # unchanged (same RNG draw, same sequence). If the proposed displacement
    # would enter/cross an obstacle footprint, the displacement is REJECTED
    # (soldier holds its previous valid position for that step) -- never
    # resampled, so the soldier RNG stream is never perturbed by obstacles.
    #
    # Hostile UAS (Phase 3B): a deterministic, RNG-free, short-horizon local
    # obstacle-navigation layer (see uav_defend/obstacles/avoidance.py) can
    # temporarily redirect the hostile's base guidance command (pursuit +
    # weave + heading noise + defender evasion) when that base command is
    # predicted -- under the SAME constrained point-mass dynamics used
    # elsewhere -- to collide with an obstacle. OFF BY DEFAULT
    # (enemy_obstacle_avoidance_enabled=False) so Phase-2 hostile
    # trajectories are reproduced exactly unless explicitly enabled.
    # =========================================================================
    enemy_obstacle_avoidance_enabled: bool = False
    enemy_obstacle_clearance: float = 2.0                  # m; inflates obstacles for candidate bypass targets
    enemy_obstacle_prediction_horizon_steps: int = 12       # simulation steps of short-horizon collision prediction
    enemy_obstacle_release_steps: int = 4                  # consecutive collision-free checks required to release avoidance

    # =========================================================================
    # Defender UAS (journal extension): the SAME deterministic, RNG-free,
    # short-horizon local obstacle-navigation layer as the hostile's above
    # (see uav_defend/obstacles/avoidance.py -- ObstacleAvoidancePlanner is
    # explicitly mover-agnostic and reused here, never duplicated), applied
    # to whatever action ANY defender controller family (Greedy, PN, Lead,
    # RMPC, PPO, PPO-Kalman, LR-PPO) produces -- a single, controller-blind
    # insertion point in SoldierEnv._move_defender(). OFF BY DEFAULT
    # (defender_obstacle_avoidance_enabled=False) so conference/frozen
    # defender behavior is reproduced exactly unless explicitly enabled.
    # =========================================================================
    defender_obstacle_avoidance_enabled: bool = False
    defender_obstacle_clearance: float = 2.0
    defender_obstacle_prediction_horizon_steps: int = 12
    defender_obstacle_release_steps: int = 4

    # =========================================================================
    # Dynamic obstacle configuration (journal extension: simple moving
    # aerial clutter -- birds / other non-combat airborne objects). See
    # uav_defend/obstacles/dynamic.py. OFF BY DEFAULT: dynamic_obstacles_enabled
    # =False reproduces existing behavior exactly -- no dynamic obstacles are
    # generated, no dedicated RNG stream is consumed, no other behavior changes.
    # =========================================================================
    dynamic_obstacles_enabled: bool = False
    dynamic_obstacle_count: int = 0
    dynamic_obstacle_min_radius: float = 0.3     # m; spherical collision radius
    dynamic_obstacle_max_radius: float = 1.0     # m
    dynamic_obstacle_min_speed: float = 1.0       # m/s; approximately constant speed
    dynamic_obstacle_max_speed: float = 4.0       # m/s
    dynamic_obstacle_min_altitude: float = 5.0    # m
    dynamic_obstacle_max_altitude: float = 25.0   # m
    dynamic_obstacle_heading_noise: float = 0.05  # rad; bounded per-step stochastic heading change (std dev)
    dynamic_obstacle_clearance: float = 2.0       # m; safety margin added to (enemy_radius + obstacle radius)
    dynamic_obstacle_prediction_horizon_steps: int = 12  # simulation steps of short-horizon conflict prediction

    # =========================================================================
    # Hostile-UAS environmental-awareness/perception layer (journal extension;
    # see uav_defend/obstacles/perception.py). Governs what the AVOIDANCE
    # PLANNER may see -- ground-truth physics/collision checking is always
    # computed against the FULL world regardless of this setting (perception
    # state != physics state; see module docstring).
    #
    # "disabled" (DEFAULT): backward-compatible mode -- avoidance sees the
    #     full ground-truth static layout + all current dynamic obstacles,
    #     exactly as in the pre-perception-layer implementation. Not treated
    #     as a realistic journal attacker mode.
    # "local"     (Level B): only obstacles within sensor range are perceived;
    #     no prior map knowledge.
    # "map_local" (Level C): map-known static obstacles are always perceived
    #     (from episode start) PLUS locally-sensed unmapped-static/dynamic
    #     obstacles.
    # =========================================================================
    enemy_obstacle_awareness_mode: str = "disabled"
    enemy_obstacle_sensor_range: float = 15.0          # m; static-obstacle local sensing radius (AABB-surface distance)
    enemy_dynamic_obstacle_sensor_range: float = 15.0  # m; dynamic-obstacle local sensing radius (center distance)
    enemy_static_obstacle_memory_enabled: bool = True  # once locally detected, a static obstacle is remembered for the rest of the episode
    enemy_dynamic_obstacle_memory_steps: int = 6       # steps a dynamic-obstacle track survives without re-observation before expiring

    # =========================================================================
    # Defender environmental-awareness/perception layer (journal extension;
    # reuses uav_defend/obstacles/perception.py's EnemyObstaclePerception --
    # mover-agnostic despite the class name -- via a SEPARATE instance owning
    # the defender's own memory state). Same "disabled"/"local"/"map_local"
    # semantics as the hostile's mode above. Defaults mirror the hostile's
    # defaults (Phase 5: "use attacker defaults initially unless there is a
    # concrete reason not to").
    # =========================================================================
    defender_obstacle_awareness_mode: str = "disabled"
    defender_obstacle_sensor_range: float = 15.0
    defender_dynamic_obstacle_sensor_range: float = 15.0
    defender_static_obstacle_memory_enabled: bool = True
    defender_dynamic_obstacle_memory_steps: int = 6

    # =========================================================================
    # Perception-constrained multi-obstacle local route planner (journal
    # extension; see uav_defend/obstacles/route_planner.py). Requires
    # enemy_obstacle_avoidance_enabled=True (the reactive layer remains the
    # last-mile safety mechanism). OFF BY DEFAULT: enemy_route_planning_enabled
    # =False reproduces the pre-route-planner behavior exactly.
    # =========================================================================
    enemy_route_planning_enabled: bool = False
    enemy_route_planning_range: float = 40.0           # m; bounded local planning horizon (Phase 7)
    enemy_route_max_graph_nodes: int = 150              # cap on visibility-graph size (>= 5 nodes/obstacle x dense-cluster obstacle counts)
    enemy_route_waypoint_tolerance: float = 2.0         # m; distance at which a waypoint counts as reached
    enemy_progress_window_steps: int = 40               # rolling window for stall detection (Phase 12)
    enemy_route_stall_distance_threshold: float = 2.0   # m; negligible goal-distance reduction over the window
    # Behavior-arbitration (journal extension, follow-up task): when True
    # (default), route following is SUSPENDED (not replanned/judged) while
    # defender evasion is active or within its post-evasion hysteresis hold
    # (reusing enemy_obstacle_release_steps), then REJOINED (not blindly
    # replanned) on resuming. When False, reproduces the original
    # always-persistent route-planning behavior (pre-arbitration ablation).
    enemy_route_arbitration_enabled: bool = True

    # =========================================================================
    # Journal V2 corrected-semantics switches (see
    # docs/corrected_attacker_semantics_v2.md). DEFAULTS REPRODUCE V1 EXACTLY;
    # publication V2 experiments must select the corrected values explicitly
    # via experiments/journal_v2_protocol.build_v2_env_config.
    #
    # mission_contact_mode:
    #   "endpoint" (V1): attacker->asset capture and defender->attacker
    #       interception are tested only at the step's final positions, so a
    #       fast pass through the contact sphere inside one dt is missed.
    #   "swept"   (V2): both contacts are tested continuously over the step
    #       with a moving-sphere (relative-motion) test; the earliest event
    #       in the step (obstacle collision, capture, interception) wins.
    # enemy_nominal_guidance_mode:
    #   "legacy" (V1): pursuit + AR(1) weave + heading noise + reactive
    #       defender evasion, with the persistent-release avoidance planner.
    #   "direct_obstacle_only" (V2 scripted A0): direct unit pursuit of the
    #       asset; deviates only while an obstacle predictively blocks the
    #       direct path and returns to direct pursuit as soon as the straight
    #       line to the asset is again clear. No weave, noise or evasion.
    #       A learned tactical override, when present, still replaces the
    #       command exactly as in V1.
    # =========================================================================
    mission_contact_mode: str = "endpoint"
    enemy_nominal_guidance_mode: str = "legacy"

    def __post_init__(self) -> None:
        """Validate altitude and dynamics configuration."""
        if self.mission_contact_mode not in ("endpoint", "swept"):
            raise ValueError(
                f"mission_contact_mode must be 'endpoint' or 'swept', got {self.mission_contact_mode!r}"
            )
        if self.enemy_nominal_guidance_mode not in ("legacy", "direct_obstacle_only"):
            raise ValueError(
                "enemy_nominal_guidance_mode must be 'legacy' or 'direct_obstacle_only', got "
                f"{self.enemy_nominal_guidance_mode!r}"
            )
        if self.enemy_spawn_altitude_min < 0:
            raise ValueError(
                f"enemy_spawn_altitude_min must be >= 0, got {self.enemy_spawn_altitude_min}"
            )
        if self.enemy_spawn_altitude_min > self.enemy_spawn_altitude_max:
            raise ValueError(
                "enemy_spawn_altitude_min must be <= enemy_spawn_altitude_max, got "
                f"{self.enemy_spawn_altitude_min} > {self.enemy_spawn_altitude_max}"
            )
        if self.enemy_spawn_altitude_max > self.max_altitude:
            raise ValueError(
                "enemy_spawn_altitude_max must be <= max_altitude, got "
                f"{self.enemy_spawn_altitude_max} > {self.max_altitude}"
            )

        if self.v_d <= 0:
            raise ValueError(f"v_d must be > 0, got {self.v_d}")
        if self.v_e <= 0:
            raise ValueError(f"v_e must be > 0, got {self.v_e}")
        if self.asset_motion_model not in ("legacy_gaussian_random_walk", "gauss_markov_velocity"):
            raise ValueError(
                "asset_motion_model must be 'legacy_gaussian_random_walk' or 'gauss_markov_velocity', got "
                f"{self.asset_motion_model!r}"
            )
        if self.asset_motion_tau_s <= 0:
            raise ValueError(f"asset_motion_tau_s must be > 0, got {self.asset_motion_tau_s}")

        if self.defender_max_accel <= 0:
            raise ValueError(
                f"defender_max_accel must be > 0, got {self.defender_max_accel}"
            )
        if self.enemy_max_accel <= 0:
            raise ValueError(f"enemy_max_accel must be > 0, got {self.enemy_max_accel}")

        if not (0 < self.defender_max_turn_rate_deg <= 360):
            raise ValueError(
                "defender_max_turn_rate_deg must be in (0, 360], got "
                f"{self.defender_max_turn_rate_deg}"
            )
        if not (0 < self.enemy_max_turn_rate_deg <= 360):
            raise ValueError(
                f"enemy_max_turn_rate_deg must be in (0, 360], got {self.enemy_max_turn_rate_deg}"
            )

        if self.defender_max_climb_rate <= 0:
            raise ValueError(
                f"defender_max_climb_rate must be > 0, got {self.defender_max_climb_rate}"
            )
        if self.defender_max_descent_rate <= 0:
            raise ValueError(
                f"defender_max_descent_rate must be > 0, got {self.defender_max_descent_rate}"
            )
        if self.enemy_max_climb_rate <= 0:
            raise ValueError(
                f"enemy_max_climb_rate must be > 0, got {self.enemy_max_climb_rate}"
            )
        if self.enemy_max_descent_rate <= 0:
            raise ValueError(
                f"enemy_max_descent_rate must be > 0, got {self.enemy_max_descent_rate}"
            )

        if self.enemy_evasion_radius <= 0:
            raise ValueError(
                f"enemy_evasion_radius must be > 0, got {self.enemy_evasion_radius}"
            )
        if not (0 <= self.enemy_evasion_gain <= 1):
            raise ValueError(
                f"enemy_evasion_gain must be in [0, 1], got {self.enemy_evasion_gain}"
            )
        if self.enemy_defender_detection_range <= 0:
            raise ValueError(
                f"enemy_defender_detection_range must be > 0, got {self.enemy_defender_detection_range}"
            )

        if self.obstacle_layout_mode not in ("none", "fixed", "random"):
            raise ValueError(
                f"obstacle_layout_mode must be one of 'none', 'fixed', 'random', got "
                f"{self.obstacle_layout_mode!r}"
            )
        if self.obstacle_count < 0:
            raise ValueError(f"obstacle_count must be >= 0, got {self.obstacle_count}")
        if self.obstacle_min_size <= 0 or self.obstacle_max_size <= 0:
            raise ValueError(
                "obstacle_min_size and obstacle_max_size must be > 0, got "
                f"{self.obstacle_min_size}, {self.obstacle_max_size}"
            )
        if self.obstacle_min_size > self.obstacle_max_size:
            raise ValueError(
                "obstacle_min_size must be <= obstacle_max_size, got "
                f"{self.obstacle_min_size} > {self.obstacle_max_size}"
            )
        if self.obstacle_min_height <= 0 or self.obstacle_max_height <= 0:
            raise ValueError(
                "obstacle_min_height and obstacle_max_height must be > 0, got "
                f"{self.obstacle_min_height}, {self.obstacle_max_height}"
            )
        if self.obstacle_min_height > self.obstacle_max_height:
            raise ValueError(
                "obstacle_min_height must be <= obstacle_max_height, got "
                f"{self.obstacle_min_height} > {self.obstacle_max_height}"
            )
        if self.obstacle_min_height > self.max_altitude:
            raise ValueError(
                "obstacle_min_height must be <= max_altitude, got "
                f"{self.obstacle_min_height} > {self.max_altitude}"
            )
        if self.obstacle_clearance_from_asset < 0:
            raise ValueError(
                f"obstacle_clearance_from_asset must be >= 0, got {self.obstacle_clearance_from_asset}"
            )
        if self.obstacle_clearance_from_spawn < 0:
            raise ValueError(
                f"obstacle_clearance_from_spawn must be >= 0, got {self.obstacle_clearance_from_spawn}"
            )
        if self.obstacle_min_separation < 0:
            raise ValueError(
                f"obstacle_min_separation must be >= 0, got {self.obstacle_min_separation}"
            )
        if self.obstacle_max_placement_attempts <= 0:
            raise ValueError(
                f"obstacle_max_placement_attempts must be > 0, got {self.obstacle_max_placement_attempts}"
            )
        for spec in self.obstacle_fixed_spec:
            if len(spec) != 6:
                raise ValueError(
                    "each obstacle_fixed_spec entry must be a 6-tuple "
                    "(center_x, center_y, center_z, half_x, half_y, half_z), got "
                    f"{spec}"
                )
            if any(h <= 0 for h in spec[3:]):
                raise ValueError(
                    f"obstacle_fixed_spec half-extents must be > 0, got {spec}"
                )

        if self.enemy_obstacle_clearance < 0:
            raise ValueError(
                f"enemy_obstacle_clearance must be >= 0, got {self.enemy_obstacle_clearance}"
            )
        if self.enemy_obstacle_prediction_horizon_steps < 1:
            raise ValueError(
                "enemy_obstacle_prediction_horizon_steps must be >= 1, got "
                f"{self.enemy_obstacle_prediction_horizon_steps}"
            )
        if self.enemy_obstacle_release_steps < 1:
            raise ValueError(
                f"enemy_obstacle_release_steps must be >= 1, got {self.enemy_obstacle_release_steps}"
            )

        if self.defender_obstacle_clearance < 0:
            raise ValueError(
                f"defender_obstacle_clearance must be >= 0, got {self.defender_obstacle_clearance}"
            )
        if self.defender_obstacle_prediction_horizon_steps < 1:
            raise ValueError(
                "defender_obstacle_prediction_horizon_steps must be >= 1, got "
                f"{self.defender_obstacle_prediction_horizon_steps}"
            )
        if self.defender_obstacle_release_steps < 1:
            raise ValueError(
                f"defender_obstacle_release_steps must be >= 1, got {self.defender_obstacle_release_steps}"
            )

        if self.dynamic_obstacle_count < 0:
            raise ValueError(f"dynamic_obstacle_count must be >= 0, got {self.dynamic_obstacle_count}")
        if self.dynamic_obstacle_min_radius <= 0 or self.dynamic_obstacle_max_radius <= 0:
            raise ValueError(
                "dynamic_obstacle_min_radius and dynamic_obstacle_max_radius must be > 0, got "
                f"{self.dynamic_obstacle_min_radius}, {self.dynamic_obstacle_max_radius}"
            )
        if self.dynamic_obstacle_min_radius > self.dynamic_obstacle_max_radius:
            raise ValueError(
                "dynamic_obstacle_min_radius must be <= dynamic_obstacle_max_radius, got "
                f"{self.dynamic_obstacle_min_radius} > {self.dynamic_obstacle_max_radius}"
            )
        if self.dynamic_obstacle_min_speed <= 0 or self.dynamic_obstacle_max_speed <= 0:
            raise ValueError(
                "dynamic_obstacle_min_speed and dynamic_obstacle_max_speed must be > 0, got "
                f"{self.dynamic_obstacle_min_speed}, {self.dynamic_obstacle_max_speed}"
            )
        if self.dynamic_obstacle_min_speed > self.dynamic_obstacle_max_speed:
            raise ValueError(
                "dynamic_obstacle_min_speed must be <= dynamic_obstacle_max_speed, got "
                f"{self.dynamic_obstacle_min_speed} > {self.dynamic_obstacle_max_speed}"
            )
        if self.dynamic_obstacle_min_altitude < 0 or self.dynamic_obstacle_max_altitude > self.max_altitude:
            raise ValueError(
                "dynamic obstacle altitude range must lie within [0, max_altitude], got "
                f"[{self.dynamic_obstacle_min_altitude}, {self.dynamic_obstacle_max_altitude}] "
                f"vs max_altitude={self.max_altitude}"
            )
        if self.dynamic_obstacle_min_altitude > self.dynamic_obstacle_max_altitude:
            raise ValueError(
                "dynamic_obstacle_min_altitude must be <= dynamic_obstacle_max_altitude, got "
                f"{self.dynamic_obstacle_min_altitude} > {self.dynamic_obstacle_max_altitude}"
            )
        if self.dynamic_obstacle_heading_noise < 0:
            raise ValueError(
                f"dynamic_obstacle_heading_noise must be >= 0, got {self.dynamic_obstacle_heading_noise}"
            )
        if self.dynamic_obstacle_clearance < 0:
            raise ValueError(f"dynamic_obstacle_clearance must be >= 0, got {self.dynamic_obstacle_clearance}")
        if self.dynamic_obstacle_prediction_horizon_steps < 1:
            raise ValueError(
                "dynamic_obstacle_prediction_horizon_steps must be >= 1, got "
                f"{self.dynamic_obstacle_prediction_horizon_steps}"
            )

        if not (0.0 <= self.obstacle_unmapped_fraction <= 1.0):
            raise ValueError(
                f"obstacle_unmapped_fraction must be in [0, 1], got {self.obstacle_unmapped_fraction}"
            )
        if self.obstacle_fixed_map_known and len(self.obstacle_fixed_map_known) != len(self.obstacle_fixed_spec):
            raise ValueError(
                "obstacle_fixed_map_known must be empty or match obstacle_fixed_spec's length, got "
                f"{len(self.obstacle_fixed_map_known)} vs {len(self.obstacle_fixed_spec)}"
            )

        if self.enemy_obstacle_awareness_mode not in ("disabled", "local", "map_local"):
            raise ValueError(
                "enemy_obstacle_awareness_mode must be one of 'disabled', 'local', 'map_local', got "
                f"{self.enemy_obstacle_awareness_mode!r}"
            )
        if self.enemy_obstacle_sensor_range <= 0:
            raise ValueError(f"enemy_obstacle_sensor_range must be > 0, got {self.enemy_obstacle_sensor_range}")
        if self.enemy_dynamic_obstacle_sensor_range <= 0:
            raise ValueError(
                f"enemy_dynamic_obstacle_sensor_range must be > 0, got {self.enemy_dynamic_obstacle_sensor_range}"
            )
        if self.enemy_dynamic_obstacle_memory_steps < 0:
            raise ValueError(
                f"enemy_dynamic_obstacle_memory_steps must be >= 0, got {self.enemy_dynamic_obstacle_memory_steps}"
            )

        if self.defender_obstacle_awareness_mode not in ("disabled", "local", "map_local"):
            raise ValueError(
                "defender_obstacle_awareness_mode must be one of 'disabled', 'local', 'map_local', got "
                f"{self.defender_obstacle_awareness_mode!r}"
            )
        if self.defender_obstacle_sensor_range <= 0:
            raise ValueError(f"defender_obstacle_sensor_range must be > 0, got {self.defender_obstacle_sensor_range}")
        if self.defender_dynamic_obstacle_sensor_range <= 0:
            raise ValueError(
                f"defender_dynamic_obstacle_sensor_range must be > 0, got {self.defender_dynamic_obstacle_sensor_range}"
            )
        if self.defender_dynamic_obstacle_memory_steps < 0:
            raise ValueError(
                f"defender_dynamic_obstacle_memory_steps must be >= 0, got {self.defender_dynamic_obstacle_memory_steps}"
            )

        if self.enemy_route_planning_range <= 0:
            raise ValueError(f"enemy_route_planning_range must be > 0, got {self.enemy_route_planning_range}")
        if self.enemy_route_max_graph_nodes < 2:
            raise ValueError(f"enemy_route_max_graph_nodes must be >= 2, got {self.enemy_route_max_graph_nodes}")
        if self.enemy_route_waypoint_tolerance <= 0:
            raise ValueError(f"enemy_route_waypoint_tolerance must be > 0, got {self.enemy_route_waypoint_tolerance}")
        if self.enemy_progress_window_steps < 1:
            raise ValueError(f"enemy_progress_window_steps must be >= 1, got {self.enemy_progress_window_steps}")
        if self.enemy_route_stall_distance_threshold < 0:
            raise ValueError(
                f"enemy_route_stall_distance_threshold must be >= 0, got {self.enemy_route_stall_distance_threshold}"
            )
