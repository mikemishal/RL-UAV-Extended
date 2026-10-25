"""
Gymnasium-compatible environment for UAV defense RL training.

This is the UNIFIED environment used for BOTH:
  1. Baseline evaluation with hand-designed policies (e.g., GreedyInterceptPolicy)
  2. RL training with algorithms like PPO, SAC, TD3

The environment itself contains NO policy logic. All defender control is
provided externally via the `step(action)` interface.

Design Philosophy:
  - Environment provides dynamics and observations
  - Policy provides actions (whether scripted or learned)
  - Same environment instance works for evaluation and training
"""

from __future__ import annotations

from typing import Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from uav_defend.config.env_config import EnvConfig
from uav_defend.tracking import EnemyKalmanFilter
from uav_defend.dynamics.constrained_point_mass import advance_velocity, apply_boundary
from uav_defend.obstacles.swept_contact import first_contact
from uav_defend.obstacles import DynamicsLimits, ObstacleAvoidancePlanner, ObstacleLayout, generate_layout
from uav_defend.obstacles.stateful_avoidance import StatefulAvoidanceController
from uav_defend.obstacles.dynamic import (
    DynamicObstacle,
    dynamic_obstacle_swept_aabb,
    dynamic_obstacle_threatens,
    generate_dynamic_obstacles,
    step_dynamic_obstacles,
    swept_sphere_vs_sphere_first_contact,
    swept_sphere_vs_sphere_min_distance,
)
from uav_defend.obstacles.perception import EnemyObstaclePerception, PerceivedObstacleSet
from uav_defend.obstacles.route_planner import RoutePlanner

# Same-instant tolerance (in fractions of one step) for swept event ordering.
EVENT_TIE_TOLERANCE = 1e-9
# Range-rate deadband (m/s) for counting approach->recede transitions.
OVERSHOOT_RANGE_RATE_DEADBAND = 1.0
# Horizontal speed (m/s) below which a heading is undefined.
MIN_HEADING_SPEED = 0.1

_DEFENDER_COLLISION_ATTRS = (
    "_defender_obstacle_collision", "_defender_collision_obstacle_index", "_defender_collision_point",
    "_defender_collision_count", "_defender_ever_collided", "_defender_was_in_collision",
    "_defender_dynamic_obstacle_collision", "_defender_collision_obstacle_id",
    "_defender_dynamic_collision_count", "_defender_dynamic_ever_collided", "_defender_was_in_dynamic_collision",
)
_ENEMY_COLLISION_ATTRS = (
    "_enemy_obstacle_collision", "_enemy_collision_obstacle_index", "_enemy_collision_point",
    "_enemy_collision_count", "_enemy_ever_collided", "_enemy_was_in_collision",
    "_enemy_dynamic_obstacle_collision", "_enemy_collision_obstacle_id",
    "_enemy_dynamic_collision_count", "_enemy_dynamic_ever_collided", "_enemy_was_in_dynamic_collision",
)

class SoldierEnv(gym.Env):
    """
    A three-dimensional constrained point-mass engagement environment for RL
    training with partial observability.

    NOTE: This is a 3D constrained point-mass UAV dynamics simulator. Both
    aircraft (defender and hostile UAV) maintain a persistent 3D velocity
    that approaches a controller/guidance-commanded desired velocity subject
    to bounded acceleration, a bounded horizontal turn rate, a maximum total
    speed, and bounded climb/descent rates. This is NOT six-DOF or
    aerodynamic flight dynamics, and the parameter defaults are provisional
    simulation values rather than validated hardware specifications.
    
    ============================================================================
    UNIFIED ENVIRONMENT FOR BASELINE AND RL
    ============================================================================
    This environment is designed to work identically for:
      - Scripted/baseline policies (e.g., GreedyInterceptPolicy)
      - RL algorithms (e.g., PPO from Stable-Baselines3)
    
    The environment contains NO hardcoded policy logic. The defender drone
    is controlled entirely by external actions passed to step().
    
    Usage with baseline policy:
        env = SoldierEnv()
        policy = GreedyInterceptPolicy()
        obs, info = env.reset(seed=42)
        while True:
            action = policy.act(obs, info)  # Policy provides action
            obs, reward, done, _, info = env.step(action)
            if done:
                break
    
    Usage with RL (Stable-Baselines3):
        from stable_baselines3 import PPO
        env = SoldierEnv()
        model = PPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=100000)
    ============================================================================
    
    Domain: 3D engagement volume, x, y ∈ [-L, L] (horizontal) and z ∈ [0, H]
    (altitude), where H = max_altitude.
    
    Entities:
        - Soldier (s ∈ ℝ³): Planar Gaussian random walk (uncontrolled); always
          remains a ground entity with s_z = 0 at all times.
        - Defender (d ∈ ℝ³): Persistent 3D velocity that tracks a
          controller-commanded desired velocity subject to dynamic limits
          (acceleration, turn rate, climb/descent rate, max speed).
        - Enemy (e ∈ ℝ³): Persistent 3D velocity that tracks a weaving
          pursuit-commanded desired velocity, subject to the same class of
          dynamic limits; naturally descends toward the ground target while
          pursuing. Uses pursuit toward the protected soldier with
          stochastic weaving and proximity-triggered reactive evasion from
          the defender, executed through constrained 3D point-mass
          dynamics -- a parameterized reactive evasive policy, NOT
          intelligent adversarial RL, optimal-attacker, or game-theoretic
          behavior (see _compute_enemy_evasion and _move_enemy).
    
    Partial Observability with Shared Sensor Noise:
        - Before detection: Enemy state is MASKED (zeros)
        - After detection: a common noisy enemy-position measurement is
          generated every step from a dedicated sensor RNG stream, shared by
          BOTH the Direct and Kalman tracks (same sensor realization; only
          downstream processing differs)
        - If Kalman is enabled, e_hat/v_hat come from filtered estimates of that measurement
        - If Kalman is disabled, e_hat mirrors the raw measurement and v_hat
          mirrors its finite-difference velocity estimate (see below) --
          giving Direct and Kalman IDENTICAL observation semantics
          (position estimate + velocity estimate), differing only in the
          estimator used
        - Detection occurs when defender is within detection_radius of enemy (3D distance)
        - Policies never receive the true enemy state through observation channels
        - Defender motion is controlled entirely by external policy (action-driven)
    
    Observation Space (spaces.Box, shape=16, dtype=float32):
        All values normalized to [-1, 1]:
            [soldier_x, soldier_y, soldier_z,
             defender_x, defender_y, defender_z,
             defender_vx, defender_vy, defender_vz,
             detected_flag,
             hostile_pos_1..3, hostile_vel_1..3]
        - defender velocity is normalized by v_d (component-wise)
        - detected_flag: 0.0 (not detected) or 1.0 (detected)
        - Both tracks share IDENTICAL semantics: hostile_pos = position
          estimate of the hostile UAV, hostile_vel = velocity estimate,
          normalized by L/max_altitude (position) and v_e (velocity):
                - If use_kalman_tracking=True:
                    hostile_pos_1..3 = e_hat (Kalman estimated position),
                    hostile_vel_1..3 = v_hat (Kalman estimated velocity)
                - If use_kalman_tracking=False (environment default):
                    hostile_pos_1..3 = raw noisy enemy position measurement,
                    hostile_vel_1..3 = finite-difference measurement velocity
                    (zeros until a second measurement is available)
        - All hostile-related values are 0.0 before detection
    
    Action Space (spaces.Box, shape=3, dtype=float32):
        3D continuous action vector in [-1, 1]³, interpreted as a DESIRED
        velocity direction (not an instantaneous displacement command):
        - Normalized to unit vector if norm > eps -> desired_velocity = v_d * direction
        - If action norm <= eps, desired_velocity = [0,0,0] (command to stop)
        - The environment's constrained-velocity dynamics determine how
          quickly the defender's actual persistent velocity can turn,
          accelerate, or decelerate toward that desired velocity.
    
    Reward (dense shaping for RL, IDENTICAL for Direct and Kalman tracks --
    no estimator-dependent reward term, no hidden clipping):
        +100 for intercepting enemy safely (WIN)
        -100 for soldier caught (LOSS)
        -150 for unsafe intercept (intercept too close to soldier)
        -100 for timeout
        +5.0 * (prev_dist - curr_dist) for closing distance to enemy (using TRUE distance)
        -0.05 per step (encourages efficiency)
        proximity warning penalty scaled by closeness of enemy to soldier
        Tracking error (info["tracking_error"]) is an EVALUATION metric only
        and is never added to the reward.
    
    Termination (all distances are true 3D Euclidean distances):
        - Intercepted safely: dist_de < intercept_radius AND dist_es > unsafe_intercept_radius (WIN)
        - Soldier caught: dist_es < threat_radius (LOSS)
        - Unsafe intercept: dist_de < intercept_radius AND dist_es <= unsafe_intercept_radius (LOSS)
        - Timeout: step_count >= max_steps
    
    Gymnasium/SB3 Compatibility:
        - Passes gymnasium.utils.env_checker.check_env()
        - Compatible with Stable-Baselines3 algorithms
        - Fixed-size observation and action spaces
        - Proper reset() and step() signatures
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    # V2 attacker obstacle avoidance: stateful receding-horizon controller (True) or the legacy
    # persisted-waypoint planner (False, kept to reproduce the 3c61845 foundation). Read at construction.
    STATEFUL_ENEMY_AVOIDANCE: bool = True
    
    def __init__(
        self,
        config: EnvConfig | None = None,
        render_mode: str | None = None,
    ):
        """
        Initialize the SoldierEnv.
        
        This environment is policy-agnostic: it accepts any 3D action vector
        for defender control, whether from a scripted baseline or RL algorithm.
        
        Args:
            config: Environment configuration. Uses defaults if None (which
                    now means use_kalman_tracking=False, i.e. the Direct/
                    measurement track -- always pass an explicit EnvConfig in
                    training/evaluation scripts rather than relying on this
                    default).
                    - If config.use_kalman_tracking=True: observation contains
                      Kalman estimates (e_hat, v_hat) - Kalman track
                    - If config.use_kalman_tracking=False: observation contains
                      raw measurement + finite-difference velocity - Direct track
            render_mode: One of "human", "rgb_array", or None.
        
        Example:
            # Direct track (noisy measurement + finite-difference velocity)
            config = EnvConfig(use_kalman_tracking=False)
            env = SoldierEnv(config=config)
            
            # Kalman track (Kalman estimates in observations)
            config = EnvConfig(use_kalman_tracking=True)
            env = SoldierEnv(config=config)
        """
        super().__init__()
        
        self.config = config if config is not None else EnvConfig()
        self.render_mode = render_mode
        
        # =====================================================================
        # OBSERVATION SPACE (fixed-size, normalized for RL)
        # =====================================================================
        # Shape: (16,), all values in [-1, 1]
        # Layout: [soldier(3), defender(3), defender_vel(3), detected_flag(1), hostile_pos(3), hostile_vel(3)]
        #
        # defender_vel is normalized by v_d (component-wise); since total
        # defender speed is dynamically constrained to <= v_d, each component
        # remains within [-1, 1].
        #
        # Both tracks share IDENTICAL semantics (hostile position estimate +
        # hostile velocity estimate); only the estimator differs:
        #
        # If config.use_kalman_tracking=True (Kalman track):
        #   - hostile_pos: Kalman estimated enemy position (zeros before detection)
        #   - hostile_vel: Kalman estimated enemy velocity (zeros before detection)
        #   Policy acts on ESTIMATED state, not ground truth.
        #
        # If config.use_kalman_tracking=False (Direct track, environment default):
        #   - hostile_pos: noisy enemy position measurement (zeros before detection)
        #   - hostile_vel: finite-difference measurement velocity (zeros before
        #     detection AND for the first step after detection, since two
        #     measurements are required)
        #   Policy acts on measured state without filtering.
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(16,),
            dtype=np.float32,
        )
        
        # Track previous distance for reward shaping
        self._prev_defender_enemy_dist: float = 0.0
        
        # =====================================================================
        # ACTION SPACE (3D desired-velocity direction)
        # =====================================================================
        # Shape: (3,), values in [-1, 1]
        # Interpretation: 3D DESIRED velocity direction for the defender.
        #   - Normalized to unit vector if norm > eps -> desired_velocity = v_d * direction
        #   - If norm <= eps, desired_velocity = [0,0,0] (command to stop)
        #   - The defender's actual persistent velocity approaches this
        #     desired velocity subject to bounded acceleration, bounded
        #     horizontal turn rate, and bounded climb/descent rate. The
        #     action does NOT instantaneously set the defender's velocity.
        #
        # This is the same interface for both:
        #   - Scripted policies: policy.act(obs, info) -> action
        #   - RL algorithms: model.predict(obs) -> action
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32,
        )
        
        # Internal state
        self._soldier_pos: np.ndarray | None = None
        self._soldier_vel: np.ndarray = np.zeros(3, dtype=np.float32)
        self._soldier_gauss_markov_initialized: bool = False
        self._defender_pos: np.ndarray | None = None
        self._enemy_pos: np.ndarray | None = None
        self._defender_vel: np.ndarray | None = None  # Persistent 3D velocity [vx,vy,vz], m/s
        self._enemy_vel: np.ndarray | None = None      # Persistent 3D velocity [vx,vy,vz], m/s
        self._defender_dynamics_info: dict = {}  # Per-step dynamics diagnostics (see _advance_velocity)
        self._enemy_dynamics_info: dict = {}
        self._enemy_evasion_info: dict = {}  # Per-step evasion diagnostics (see _compute_enemy_evasion)
        self._enemy_pursuit_direction: np.ndarray | None = None  # Diagnostic only, shape (3,)
        self._weave_bias: float = 0.0  # AR(1) lateral weave bias 'a'
        self._step_count: int = 0
        self._enemy_detected: bool = False  # Tracking state: enemy detected?
        # Independent deterministic RNG streams derived from the episode seed
        # (see reset()), so that controller-dependent sensor RNG consumption
        # cannot perturb unrelated exogenous randomness (spawn, soldier walk,
        # hostile weave/noise). See test_experiment_fairness.py for the
        # regression test that guards this property.
        self._rng_spawn: np.random.Generator | None = None
        self._rng_soldier: np.random.Generator | None = None
        self._rng_enemy_motion: np.random.Generator | None = None
        self._rng_sensor: np.random.Generator | None = None
        # Dedicated obstacle RNG stream (journal extension; see reset()).
        # Never shared with the four streams above -- see
        # test_obstacle_env_integration.py's baseline-reproduction regression
        # test for the property this separation guarantees.
        self._rng_obstacles: np.random.Generator | None = None
        self._obstacle_layout: ObstacleLayout = ObstacleLayout()
        # Dedicated dynamic-obstacle RNG stream (journal extension; see
        # reset()). Spawned AFTER the static-obstacle stream so that
        # enabling/disabling dynamic obstacles never perturbs static-obstacle
        # or exogenous-motion trajectories.
        self._rng_dynamic_obstacles: np.random.Generator | None = None
        self._dynamic_obstacles: tuple[DynamicObstacle, ...] = ()

        # Obstacle collision/clearance diagnostics (journal extension,
        # Phase 2). DIAGNOSTIC ONLY in this phase: never affects movement,
        # reward, or termination (see step()/reset()).
        self._defender_obstacle_collision: bool = False
        self._enemy_obstacle_collision: bool = False
        self._defender_collision_obstacle_index: int | None = None
        self._enemy_collision_obstacle_index: int | None = None
        self._defender_collision_point: np.ndarray | None = None
        self._enemy_collision_point: np.ndarray | None = None
        self._defender_min_obstacle_clearance: float = float("inf")
        self._enemy_min_obstacle_clearance: float = float("inf")
        # Collision EVENT counters: an "outside -> collision" transition
        # counts once; continued occupancy of the same (or any) obstacle on
        # subsequent steps is NOT re-counted until clearance is regained.
        self._defender_collision_count: int = 0
        self._enemy_collision_count: int = 0
        self._defender_ever_collided: bool = False
        self._enemy_ever_collided: bool = False
        self._defender_was_in_collision: bool = False
        self._enemy_was_in_collision: bool = False

        # Protected-asset obstacle-blocked-motion diagnostics (Phase 3).
        # DIAGNOSTIC ONLY plus the (data-preserving) blocked-motion rule
        # itself -- see _move_soldier().
        self._soldier_obstacle_blocked: bool = False
        self._soldier_obstacle_block_count: int = 0

        # Hostile obstacle-navigation planner (Phase 3B): a deterministic,
        # RNG-free, short-horizon local avoidance layer (see
        # uav_defend/obstacles/avoidance.py). Persistence/hysteresis state
        # lives inside the planner instance and is cleared every reset().
        if self.STATEFUL_ENEMY_AVOIDANCE and self.config.enemy_nominal_guidance_mode == "direct_obstacle_only":
            # Corrected (V2) A0 only: stateful receding-horizon guidance instead of a persisted waypoint.
            self._enemy_obstacle_planner = StatefulAvoidanceController(
                clearance=self.config.enemy_obstacle_clearance,
                horizon_steps=self.config.enemy_obstacle_prediction_horizon_steps,
                max_altitude=self.config.max_altitude,
                eps=self.config.eps,
                mission_stop_radius=self.config.threat_radius,
            )
        else:
            self._enemy_obstacle_planner = ObstacleAvoidancePlanner(
                clearance=self.config.enemy_obstacle_clearance,
                horizon_steps=self.config.enemy_obstacle_prediction_horizon_steps,
                release_steps=self.config.enemy_obstacle_release_steps,
                max_altitude=self.config.max_altitude,
                eps=self.config.eps,
                immediate_release=self.config.enemy_nominal_guidance_mode == "direct_obstacle_only",
                mission_stop_radius=(
                    self.config.threat_radius if self.config.enemy_nominal_guidance_mode == "direct_obstacle_only" else None
                ),
                executed_prediction=self.config.enemy_nominal_guidance_mode == "direct_obstacle_only",
            )
        # Environmental-awareness/perception layer (journal extension): the
        # planner above is fed ONLY this object's output (see _move_enemy()),
        # never the ground-truth world directly -- see
        # uav_defend/obstacles/perception.py.
        self._enemy_perception = EnemyObstaclePerception(
            mode=self.config.enemy_obstacle_awareness_mode,
            static_sensor_range=self.config.enemy_obstacle_sensor_range,
            dynamic_sensor_range=self.config.enemy_dynamic_obstacle_sensor_range,
            static_memory_enabled=self.config.enemy_static_obstacle_memory_enabled,
            dynamic_memory_steps=self.config.enemy_dynamic_obstacle_memory_steps,
        )
        self._enemy_last_perceived: PerceivedObstacleSet = PerceivedObstacleSet()
        # Multi-obstacle local route planner (journal extension): sits
        # BETWEEN mission pursuit and the reactive avoidance layer above --
        # see _move_enemy(). Consumes only PerceivedObstacleSet, never
        # ground truth.
        self._enemy_route_planner = RoutePlanner(
            clearance=self.config.enemy_obstacle_clearance,
            max_altitude=self.config.max_altitude,
            planning_range=self.config.enemy_route_planning_range,
            max_graph_nodes=self.config.enemy_route_max_graph_nodes,
            waypoint_tolerance=self.config.enemy_route_waypoint_tolerance,
            progress_window_steps=self.config.enemy_progress_window_steps,
            stall_distance_threshold=self.config.enemy_route_stall_distance_threshold,
            dt=self.config.dt,
        )
        self._enemy_route_last_result: dict | None = None
        # Command-conflict diagnostics (Phase 3, behavior-arbitration
        # audit): raw per-stage commands + angular disagreement + episode
        # counters. Purely observational -- never fed back into control.
        self._enemy_mission_command: np.ndarray = np.zeros(3, dtype=np.float32)
        self._enemy_evasion_command: np.ndarray = np.zeros(3, dtype=np.float32)
        self._enemy_route_command: np.ndarray | None = None
        self._enemy_reactive_avoidance_command: np.ndarray | None = None
        self._enemy_final_command: np.ndarray = np.zeros(3, dtype=np.float32)
        self._enemy_evasion_active: bool = False
        self._enemy_evasion_streak_steps: int = 0
        self._enemy_evasion_streak_start_pos: np.ndarray | None = None
        self._enemy_evasion_streak_start_goal_dist: float | None = None
        self._enemy_replans_while_evading: int = 0
        self._enemy_waypoint_invalidations_from_evasion: int = 0
        self._enemy_route_resumptions_after_evasion: int = 0
        self._enemy_obstacle_conflicts_during_evasion: int = 0
        self._enemy_behavior_mode: str = "MISSION"
        self._enemy_behavior_mode_switch_count: int = 0
        self._enemy_angle_mission_evasion: float | None = None
        self._enemy_angle_mission_route: float | None = None
        self._enemy_angle_evasion_route: float | None = None
        self._enemy_angle_route_final: float | None = None
        self._enemy_angle_evasion_final: float | None = None
        # Behavior arbitration (Phase 5-7): route-suspension-during-evasion
        # state (see _move_enemy()). Initialized at (not below) the release
        # threshold so route planning is NOT spuriously suspended at episode
        # start merely because evasion hasn't happened yet.
        self._enemy_evasion_clear_streak: int = self.config.enemy_obstacle_release_steps
        self._enemy_route_was_suspended: bool = False
        self._enemy_route_suspended: bool = False
        self._reset_v2_episode_state(0.0)
        # Adaptive Attacker v1 hook (learned tactical policy, journal
        # extension): when set (by an external training wrapper, NEVER by
        # SoldierEnv itself), _move_enemy() replaces its ENTIRE hand-designed
        # tactical command (pursuit+evasion+route+weave+noise) with
        # normalize(enemy_tactical_override()), while leaving obstacle
        # safety/dynamics/collision (and all diagnostic fields, computed as
        # usual from the hand-designed model for comparison purposes)
        # completely unchanged. None (the default, set here and never
        # touched by reset()) reproduces EXACT prior behavior -- this
        # attribute does not exist anywhere else in the codebase yet, so no
        # other caller can be affected.
        self.enemy_tactical_override: Callable[[], np.ndarray] | None = None
        # Hostile obstacle-navigation diagnostics (Phase 3B). Deterministic
        # and RNG-free -- see _move_enemy()/ObstacleAvoidancePlanner.plan().
        self._enemy_obstacle_avoidance_active: bool = False
        self._enemy_obstacle_avoidance_mode: str | None = None
        self._enemy_obstacle_avoidance_vector: np.ndarray = np.zeros(3, dtype=np.float32)
        self._enemy_predicted_obstacle_index: int | None = None
        self._enemy_predicted_collision_step: int | None = None
        self._enemy_predicted_min_clearance: float = float("inf")
        self._enemy_obstacle_fallback_used: bool = False

        # Defender obstacle-navigation planner (journal extension): the
        # SAME mover-agnostic ObstacleAvoidancePlanner as the hostile's
        # above, in a SEPARATE instance so avoidance persistence/hysteresis
        # never leaks between the two movers. See _move_defender().
        self._defender_obstacle_planner = ObstacleAvoidancePlanner(
            clearance=self.config.defender_obstacle_clearance,
            horizon_steps=self.config.defender_obstacle_prediction_horizon_steps,
            release_steps=self.config.defender_obstacle_release_steps,
            max_altitude=self.config.max_altitude,
            eps=self.config.eps,
        )
        # Defender perception (journal extension): reuses
        # EnemyObstaclePerception (mover-agnostic despite the class name --
        # see uav_defend/obstacles/perception.py) with the defender's OWN
        # memory state. The mission target (the hostile UAV) is NEVER
        # added to this perceived set -- it remains the interception
        # target, not environmental clutter (Phase 14).
        self._defender_perception = EnemyObstaclePerception(
            mode=self.config.defender_obstacle_awareness_mode,
            static_sensor_range=self.config.defender_obstacle_sensor_range,
            dynamic_sensor_range=self.config.defender_dynamic_obstacle_sensor_range,
            static_memory_enabled=self.config.defender_static_obstacle_memory_enabled,
            dynamic_memory_steps=self.config.defender_dynamic_obstacle_memory_steps,
        )
        self._defender_last_perceived: PerceivedObstacleSet = PerceivedObstacleSet()
        self._defender_obstacle_avoidance_active: bool = False
        self._defender_obstacle_avoidance_mode: str | None = None
        self._defender_desired_command: np.ndarray = np.zeros(3, dtype=np.float32)
        self._defender_safe_command: np.ndarray = np.zeros(3, dtype=np.float32)
        self._defender_avoidance_obstacle_type: str | None = None
        self._defender_avoidance_obstacle_id: int | None = None
        self._defender_avoidance_angle_deg: float = 0.0
        self._defender_predicted_min_clearance: float = float("inf")

        # Hostile avoidance diagnostics extended for dynamic obstacles
        # (journal extension): obstacle_type/angle/time-to-conflict
        # distinguish a static-AABB avoidance decision from a
        # dynamic-obstacle one, and record the actual (not just predicted)
        # per-step dynamic-obstacle collision/clearance state.
        self._enemy_avoidance_obstacle_type: str | None = None
        self._enemy_avoidance_obstacle_id: int | None = None
        self._enemy_avoidance_angle_deg: float = 0.0
        self._enemy_predicted_time_to_conflict: float | None = None
        self._enemy_dynamic_obstacle_collision: bool = False
        self._enemy_collision_obstacle_id: int | None = None
        self._enemy_min_dynamic_obstacle_clearance: float = float("inf")
        self._enemy_dynamic_collision_count: int = 0
        self._enemy_dynamic_ever_collided: bool = False
        self._enemy_was_in_dynamic_collision: bool = False
        self._defender_dynamic_obstacle_collision: bool = False
        self._defender_collision_obstacle_id: int | None = None
        self._defender_min_dynamic_obstacle_clearance: float = float("inf")
        self._defender_dynamic_collision_count: int = 0
        self._defender_dynamic_ever_collided: bool = False
        self._defender_was_in_dynamic_collision: bool = False

        # Physical-collision termination diagnostics (journal extension,
        # Phase 3/4): the entity's FIRST contact this step (static or
        # dynamic, whichever is earlier -- see step()), used to terminate
        # the episode and clamp position/velocity at the contact point.
        # None/0.0/None whenever no physical collision occurred this step.
        self._defender_collision_event_type: str | None = None
        self._defender_collision_event_point: np.ndarray | None = None
        self._defender_collision_event_speed: float = 0.0
        self._enemy_collision_event_type: str | None = None
        self._enemy_collision_event_point: np.ndarray | None = None
        self._enemy_collision_event_speed: float = 0.0
        
        # Kalman filter for enemy tracking (initialized on first detection)
        self._kf: EnemyKalmanFilter | None = None
        self._e_hat: np.ndarray | None = None  # Estimated enemy position
        self._v_hat: np.ndarray | None = None  # Estimated enemy velocity
        self._enemy_measurement: np.ndarray | None = None  # Latest noisy position measurement
        
        # Standardized Direct/measurement-track finite-difference velocity
        # estimate (see _update_detection). This is the single common
        # estimator used by the Direct observation and any measurement-mode
        # controller (Greedy, PN, Lead) -- no independent per-policy finite
        # differencing.
        self._prev_enemy_measurement: np.ndarray | None = None
        self._enemy_measurement_velocity: np.ndarray = np.zeros(3, dtype=np.float32)
        self._enemy_measurement_velocity_valid: bool = False
        
        # Measurement noise standard deviation for enemy position sensing
        # Derived from config measurement_var (variance = std^2)
        self._measurement_noise_std: float = np.sqrt(self.config.measurement_var)
    
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        """
        Reset the environment to an initial state.
        
        The soldier is initialized at the origin (0, 0, 0) and remains a
        ground entity (z = 0) for the entire episode.
        
        Args:
            seed: Random seed for reproducibility.
            options: Additional options (unused).
        
        Returns:
            observation: Shape (16,) normalized state vector.
            info: Additional information dict.
        """
        super().reset(seed=seed)
        # Independent deterministic RNG streams derived from the same episode
        # seed via SeedSequence.spawn(): sensor-RNG consumption (which varies
        # with detection timing, itself controller-dependent) must never
        # perturb unrelated exogenous randomness (spawn, soldier walk,
        # hostile weave/noise).
        seed_seq = np.random.SeedSequence(seed)
        spawn_seed, soldier_seed, enemy_seed, sensor_seed = seed_seq.spawn(4)
        self._rng_spawn = np.random.default_rng(spawn_seed)
        self._rng_soldier = np.random.default_rng(soldier_seed)
        self._rng_enemy_motion = np.random.default_rng(enemy_seed)
        self._rng_sensor = np.random.default_rng(sensor_seed)
        # Obstacle RNG (journal extension): spawned AFTER the four streams
        # above so their entropy/output is bit-for-bit unchanged regardless
        # of whether obstacles are enabled (numpy SeedSequence.spawn() calls
        # on the same sequence never alter earlier children -- see
        # test_obstacle_env_integration.py).
        (obstacle_seed,) = seed_seq.spawn(1)
        self._rng_obstacles = np.random.default_rng(obstacle_seed)
        # Dynamic-obstacle RNG (journal extension): spawned AFTER the
        # static-obstacle stream, for the same reproducibility reason.
        (dynamic_obstacle_seed,) = seed_seq.spawn(1)
        self._rng_dynamic_obstacles = np.random.default_rng(dynamic_obstacle_seed)
        
        # Initialize soldier at the origin, on the ground (z = 0)
        self._soldier_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        # Soldier velocity state (journal model-study extension, asset_motion_model ==
        # "gauss_markov_velocity" only; unused/not maintained under the legacy model so legacy
        # behavior and its RNG draw sequence are completely untouched).
        self._soldier_vel = np.zeros(3, dtype=np.float32)
        self._soldier_gauss_markov_initialized = False
        
        # Initialize defender co-located with soldier: d0 = s0 = [x, y, 0]
        self._defender_pos = self._soldier_pos.copy()
        
        # Defender begins stationary: it must accelerate from rest rather
        # than instantly travel at v_d.
        self._defender_vel = np.zeros(3, dtype=np.float32)
        
        # Initialize enemy at random position on boundary edge
        self._enemy_pos = self._spawn_enemy_at_edge()
        
        # The hostile UAV represents an aircraft already entering the
        # engagement volume: initialize its velocity approximately toward
        # the soldier at its maximum speed, rather than from rest, to avoid
        # an artificial launch-from-rest transient.
        pursuit_dir = self._soldier_pos - self._enemy_pos
        pursuit_norm = np.linalg.norm(pursuit_dir)
        if pursuit_norm > self.config.eps:
            pursuit_dir = pursuit_dir / pursuit_norm
        else:
            pursuit_dir = np.array([1.0, 0.0, 0.0])  # Safe fallback (numerically degenerate case)
        self._enemy_pursuit_direction = pursuit_dir.astype(np.float32)
        initial_enemy_vel = self.config.v_e * pursuit_dir
        # Respect the vertical-rate invariant maintained everywhere else:
        # clip the initial vertical component to [-max_descent, max_climb]
        # (this can only reduce the total initial speed slightly below v_e,
        # never increase it, so the max-speed bound also remains satisfied).
        initial_enemy_vel[2] = np.clip(
            initial_enemy_vel[2],
            -self.config.enemy_max_descent_rate,
            self.config.enemy_max_climb_rate,
        )
        self._enemy_vel = initial_enemy_vel.astype(np.float32)
        
        # Obstacle layout (journal extension): DATA ONLY in this phase --
        # does not affect movement, detection, reward, or termination.
        # Generated from its own dedicated RNG stream (see above), using
        # the now-known enemy spawn position for spawn-clearance rejection
        # in "random" mode.
        self._obstacle_layout = generate_layout(
            config=self.config,
            rng=self._rng_obstacles,
            asset_position=self._soldier_pos,
            defender_position=self._defender_pos,
            enemy_spawn_position=self._enemy_pos,
        )

        # Dynamic obstacles (journal extension): DATA ONLY unless
        # `dynamic_obstacles_enabled` -- generated from their own dedicated
        # RNG stream, never affecting behavior when disabled.
        if self.config.dynamic_obstacles_enabled:
            self._dynamic_obstacles = generate_dynamic_obstacles(
                config=self.config,
                rng=self._rng_dynamic_obstacles,
                static_layout=self._obstacle_layout,
                asset_position=self._soldier_pos,
                defender_position=self._defender_pos,
                enemy_position=self._enemy_pos,
            )
        else:
            self._dynamic_obstacles = ()

        # Reset per-episode collision/clearance diagnostics (Phase 2).
        # Initial positions are validated (both "random" via clearance
        # margins and "fixed" via hard containment rejection in
        # generate_layout()) to never begin inside an obstacle, so the
        # initial collision flags below are always False by construction.
        self._defender_obstacle_collision = self._obstacle_layout.contains_point(self._defender_pos)
        self._enemy_obstacle_collision = self._obstacle_layout.contains_point(self._enemy_pos)
        self._defender_collision_obstacle_index = self._obstacle_layout.containing_obstacle_index(self._defender_pos)
        self._enemy_collision_obstacle_index = self._obstacle_layout.containing_obstacle_index(self._enemy_pos)
        self._defender_collision_point = None
        self._enemy_collision_point = None
        self._defender_min_obstacle_clearance = self._obstacle_layout.min_clearance(self._defender_pos)
        self._enemy_min_obstacle_clearance = self._obstacle_layout.min_clearance(self._enemy_pos)
        self._defender_collision_count = 0
        self._enemy_collision_count = 0
        self._defender_ever_collided = False
        self._enemy_ever_collided = False
        self._defender_was_in_collision = self._defender_obstacle_collision
        self._enemy_was_in_collision = self._enemy_obstacle_collision

        # Reset per-episode protected-asset/hostile obstacle diagnostics (Phase 3).
        self._soldier_obstacle_blocked = False
        self._soldier_obstacle_block_count = 0
        self._enemy_obstacle_planner.reset()
        self._enemy_perception.reset()
        self._enemy_last_perceived = PerceivedObstacleSet()
        self._enemy_route_planner.reset()
        self._enemy_route_last_result = None
        self._enemy_mission_command = np.zeros(3, dtype=np.float32)
        self._enemy_evasion_command = np.zeros(3, dtype=np.float32)
        self._enemy_route_command = None
        self._enemy_reactive_avoidance_command = None
        self._enemy_final_command = np.zeros(3, dtype=np.float32)
        self._enemy_evasion_active = False
        self._enemy_evasion_streak_steps = 0
        self._enemy_evasion_streak_start_pos = None
        self._enemy_evasion_streak_start_goal_dist = None
        self._enemy_replans_while_evading = 0
        self._enemy_waypoint_invalidations_from_evasion = 0
        self._enemy_route_resumptions_after_evasion = 0
        self._enemy_obstacle_conflicts_during_evasion = 0
        self._enemy_behavior_mode = "MISSION"
        self._enemy_behavior_mode_switch_count = 0
        self._enemy_angle_mission_evasion = None
        self._enemy_angle_mission_route = None
        self._enemy_angle_evasion_route = None
        self._enemy_angle_route_final = None
        self._enemy_angle_evasion_final = None
        self._enemy_evasion_clear_streak = self.config.enemy_obstacle_release_steps
        self._enemy_route_was_suspended = False
        self._enemy_route_suspended = False
        self._enemy_obstacle_avoidance_active = False
        self._enemy_obstacle_avoidance_mode = None
        self._enemy_obstacle_avoidance_vector = np.zeros(3, dtype=np.float32)
        self._enemy_predicted_obstacle_index = None
        self._enemy_predicted_collision_step = None
        self._enemy_predicted_min_clearance = self._obstacle_layout.min_clearance(self._enemy_pos)
        self._enemy_obstacle_fallback_used = False

        self._defender_obstacle_planner.reset()
        self._defender_perception.reset()
        self._defender_last_perceived = PerceivedObstacleSet()
        self._defender_obstacle_avoidance_active = False
        self._defender_obstacle_avoidance_mode = None
        self._defender_desired_command = np.zeros(3, dtype=np.float32)
        self._defender_safe_command = np.zeros(3, dtype=np.float32)
        self._defender_avoidance_obstacle_type = None
        self._defender_avoidance_obstacle_id = None
        self._defender_avoidance_angle_deg = 0.0
        self._defender_predicted_min_clearance = self._obstacle_layout.min_clearance(self._defender_pos)

        # Reset dynamic-obstacle diagnostics (journal extension).
        self._enemy_avoidance_obstacle_type = None
        self._enemy_avoidance_obstacle_id = None
        self._enemy_avoidance_angle_deg = 0.0
        self._enemy_predicted_time_to_conflict = None
        self._enemy_dynamic_obstacle_collision = False
        self._enemy_collision_obstacle_id = None
        self._enemy_min_dynamic_obstacle_clearance = self._dynamic_min_clearance(self._enemy_pos)
        self._enemy_dynamic_collision_count = 0
        self._enemy_dynamic_ever_collided = False
        self._enemy_was_in_dynamic_collision = False
        self._defender_dynamic_obstacle_collision = False
        self._defender_collision_obstacle_id = None
        self._defender_min_dynamic_obstacle_clearance = self._dynamic_min_clearance(self._defender_pos)
        self._defender_dynamic_collision_count = 0
        self._defender_dynamic_ever_collided = False
        self._defender_was_in_dynamic_collision = False

        # Reset physical-collision-event diagnostics (journal extension,
        # Phase 3/4).
        self._defender_collision_event_type = None
        self._defender_collision_event_point = None
        self._defender_collision_event_speed = 0.0
        self._enemy_collision_event_type = None
        self._enemy_collision_event_point = None
        self._enemy_collision_event_speed = 0.0
        
        # Reset per-step dynamics diagnostics (see _advance_velocity)
        self._defender_dynamics_info = {
            "accel_used": 0.0,
            "turn_rate_used_deg": 0.0,
            "accel_saturated": False,
            "turn_saturated": False,
            "climb_saturated": False,
        }
        self._enemy_dynamics_info = dict(self._defender_dynamics_info)
        
        # Reset per-step evasion diagnostics (see _compute_enemy_evasion)
        self._enemy_evasion_info = self._compute_enemy_evasion(self._defender_pos, self._enemy_pos)
        
        # Initialize weave bias to zero: a0 = 0
        self._weave_bias = 0.0
        
        self._step_count = 0
        
        # Initialize detection state: enemy not detected at start
        self._enemy_detected = False
        
        # Reset Kalman filter state
        self._kf = None
        self._e_hat = None
        self._v_hat = None
        self._enemy_measurement = None
        
        # Reset standardized Direct-track measurement-velocity state
        self._prev_enemy_measurement = None
        self._enemy_measurement_velocity = np.zeros(3, dtype=np.float32)
        self._enemy_measurement_velocity_valid = False
        
        # Calculate initial distances
        initial_enemy_soldier_dist = np.linalg.norm(self._enemy_pos - self._soldier_pos)
        initial_defender_enemy_dist = np.linalg.norm(self._defender_pos - self._enemy_pos)
        
        # Store for reward shaping
        self._prev_defender_enemy_dist = initial_defender_enemy_dist
        self._reset_v2_episode_state(float(initial_enemy_soldier_dist))
        
        return self._get_obs(), self._get_info(
            "ongoing", initial_enemy_soldier_dist, initial_defender_enemy_dist,
            defender_standby=self.config.defender_standby_until_detection,
            controller_action_executed=False,
            just_detected=False,
        )
    
    # ------------------------------------------------------------------
    # V2 corrected-semantics helpers (see docs/corrected_attacker_semantics_v2.md)
    # ------------------------------------------------------------------
    def _empty_contact_info(self) -> dict:
        return {
            "mission_contact_mode": self.config.mission_contact_mode,
            "capture_contact_fraction": None,
            "capture_contact_position": None,
            "capture_contact_distance": None,
            "interception_contact_fraction": None,
            "interception_contact_position": None,
            "interception_contact_distance": None,
            "terminal_event": None,
            "terminal_event_fraction": None,
        }

    def _reset_v2_episode_state(self, initial_range: float = 0.0) -> None:
        """Per-episode contact record and attacker mission-efficiency
        counters. Observational only: never read by guidance, reward or
        termination."""
        self._contact_info = self._empty_contact_info()
        self._attacker_initial_range = float(initial_range)
        self._attacker_path_length = 0.0
        self._attacker_heading_change_deg = 0.0
        self._attacker_prev_heading_rad: float | None = None
        self._attacker_overshoot_count = 0
        self._attacker_prev_range = float(initial_range)
        self._attacker_range_trend = 0  # -1 closing, +1 receding, 0 undetermined
        self._episode_time_s = 0.0

    def _update_attacker_efficiency(self, enemy_prev_pos: np.ndarray, event_fraction: float) -> None:
        dt = self.config.dt
        self._attacker_path_length += float(np.linalg.norm(self._enemy_pos - enemy_prev_pos))
        self._episode_time_s += dt * float(event_fraction)

        vel = np.asarray(self._enemy_vel, dtype=np.float64)
        if float(np.hypot(vel[0], vel[1])) > MIN_HEADING_SPEED:
            heading = float(np.arctan2(vel[1], vel[0]))
            if self._attacker_prev_heading_rad is not None:
                delta = (heading - self._attacker_prev_heading_rad + np.pi) % (2.0 * np.pi) - np.pi
                self._attacker_heading_change_deg += abs(float(np.degrees(delta)))
            self._attacker_prev_heading_rad = heading

        current_range = float(np.linalg.norm(self._enemy_pos - self._soldier_pos))
        range_rate = (current_range - self._attacker_prev_range) / dt
        if range_rate < -OVERSHOOT_RANGE_RATE_DEADBAND:
            self._attacker_range_trend = -1
        elif range_rate > OVERSHOOT_RANGE_RATE_DEADBAND:
            if self._attacker_range_trend == -1:
                self._attacker_overshoot_count += 1
            self._attacker_range_trend = 1
        self._attacker_prev_range = current_range

    def _snapshot_collision_state(self) -> dict:
        return {name: getattr(self, name, None) for name in _DEFENDER_COLLISION_ATTRS + _ENEMY_COLLISION_ATTRS}

    def _restore_collision_state(self, snapshot: dict, who: str) -> None:
        names = _DEFENDER_COLLISION_ATTRS if who == "defender" else _ENEMY_COLLISION_ATTRS
        for name in names:
            setattr(self, name, snapshot[name])

    def _attacker_efficiency_info(self) -> dict:
        path = self._attacker_path_length
        efficiency = (self._attacker_initial_range / path) if path > 1e-9 else None
        return {
            "attacker_initial_range": self._attacker_initial_range,
            "attacker_path_length": path,
            "attacker_flight_time_s": self._episode_time_s,
            "attacker_heading_change_total_deg": self._attacker_heading_change_deg,
            "attacker_overshoot_count": self._attacker_overshoot_count,
            # initial straight-line range / actual path length. The asset
            # random-walks (sigma = v_s*dt per axis per step), so this ratio
            # can marginally exceed 1; the capped value is also reported.
            "attacker_path_efficiency": efficiency,
            "attacker_path_efficiency_capped": None if efficiency is None else min(1.0, efficiency),
        }

    def _spawn_enemy_at_edge(self) -> np.ndarray:
        """
        Spawn enemy at a random location on the vertical lateral boundary of
        the 3D engagement volume, i.e. on the perimeter of the horizontal
        square [-L, L]² at a randomly sampled altitude.
        
        Uniformly choose one of four horizontal edges (x = ±L or y = ±L), then
        sample the other horizontal coordinate uniformly along that edge, and
        independently sample altitude z ~ Uniform(enemy_spawn_altitude_min,
        enemy_spawn_altitude_max).
        
        Returns:
            Enemy position array of shape (3,).
        """
        L = self.config.L
        edge = self._rng_spawn.integers(0, 4)  # 0=top, 1=bottom, 2=left, 3=right
        coord = self._rng_spawn.uniform(-L, L)  # Position along the edge
        z = self._rng_spawn.uniform(
            self.config.enemy_spawn_altitude_min,
            self.config.enemy_spawn_altitude_max,
        )
        
        if edge == 0:  # Top edge: y = L
            return np.array([coord, L, z], dtype=np.float32)
        elif edge == 1:  # Bottom edge: y = -L
            return np.array([coord, -L, z], dtype=np.float32)
        elif edge == 2:  # Left edge: x = -L
            return np.array([-L, coord, z], dtype=np.float32)
        else:  # Right edge: x = L
            return np.array([L, coord, z], dtype=np.float32)

    def _dynamic_min_clearance(self, point: np.ndarray) -> float:
        """Minimum (surface-to-point) clearance from `point` to any current
        dynamic obstacle; `+inf` if none exist. Point-vs-sphere-surface
        distance, consistent with `ObstacleLayout.min_clearance`'s
        point-vs-box-surface convention for static obstacles."""
        if not self._dynamic_obstacles:
            return float("inf")
        return min(
            float(np.linalg.norm(np.asarray(point, dtype=np.float64) - obs.position)) - obs.radius
            for obs in self._dynamic_obstacles
        )

    def _combined_avoidance_layout(
        self, position: np.ndarray, velocity: np.ndarray, horizon_steps: int, clearance: float,
        static_layout: ObstacleLayout, dynamic_obstacles: tuple,
    ) -> ObstacleLayout:
        """`static_layout` plus one conservative swept-AABB per obstacle in
        `dynamic_obstacles` that a closest-point-of-approach pre-check
        (using the mover's CURRENT velocity as a constant-velocity proxy --
        the subsequent full constrained-dynamics candidate evaluation is
        the authoritative check) predicts could come within `clearance` of
        `position` within the horizon. Obstacles already receding/never
        converging are excluded, so a merely-nearby-but-diverging obstacle
        can never spuriously trigger avoidance (see
        `dynamic_obstacle_swept_aabb`'s full-path-box conservativeness).

        NOTE: `static_layout`/`dynamic_obstacles` are the PERCEIVED
        obstacle set (see uav_defend/obstacles/perception.py), NOT
        necessarily the ground-truth world -- this function itself is
        perception-agnostic; the caller decides what it may see."""
        if not dynamic_obstacles:
            return static_layout
        threatening = [
            dynamic_obstacle_swept_aabb(obs, horizon_steps, self.config.dt, clearance)
            for obs in dynamic_obstacles
            if dynamic_obstacle_threatens(position, velocity, obs, horizon_steps, self.config.dt, clearance)
        ]
        if not threatening:
            return static_layout
        return ObstacleLayout(obstacles=static_layout.obstacles + tuple(threatening))

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one time step in the environment.
        
        This is the unified interface for both baseline and RL control.
        The environment applies the action to move the defender, then
        updates all other entities (soldier, enemy) and computes reward.
        
        Args:
            action: 3D continuous action vector in [-1, 1]³, shape=(3,),
                   interpreted as a DESIRED velocity direction (see class
                   docstring). This can come from:
                     - Scripted policy: policy.act(obs, info) -> np.ndarray
                     - RL algorithm: model.predict(obs) -> (action, state)
                   The action is normalized to a unit vector if norm > eps.
                   If norm <= eps, the desired velocity is [0,0,0] (a command
                   to decelerate toward a stop, not an instantaneous stop).
        
        Returns:
            observation: Shape (16,) normalized state vector.
            reward: Scalar reward (dense shaping for RL training).
            terminated: True if episode ended (win/loss/timeout).
            truncated: Always False (no external truncation).
            info: Dict with additional state for debugging/visualization.
        
        Note:
            The environment does NOT contain any policy logic.
            All defender control comes from the external action.

        Pre-detection standby semantics (config.defender_standby_until_detection,
        REVISED canonical default=True):
            Let detected_at_step_start = self._enemy_detected BEFORE this
            step's mutations. The discrete-time sequence is:
                A. Move the soldier (existing stochastic random walk).
                B. If standby AND NOT detected_at_step_start: synchronize the
                   defender exactly onto the (post-move) soldier position
                   with zero velocity -- the supplied `action` has NO effect
                   this step.
                C. Move the hostile (existing pursuit+weave+evasion model);
                   evasion sees the CURRENT (possibly just-synced) defender
                   position.
                D. Update detection using the resulting current positions.
                E. Execute the supplied `action` through the defender's
                   constrained dynamics ONLY IF detected_at_step_start was
                   True. If detected_at_step_start was False -- including
                   the very transition step where detection newly becomes
                   True in D -- the action is NOT executed this step.
            Consequently: the first observation with detected_flag=1 is
            returned with the defender still exactly co-located with the
            soldier (zero velocity); the controller's first EXECUTED
            interception action occurs on the NEXT call to step() (whose
            detected_at_step_start is now True). This gives every
            controller (Greedy, PN, Lead, PPO, PPO-Kalman, ...) an
            identical, action-independent pre-detection state history.
            When defender_standby_until_detection=False (legacy mode), the
            supplied action is applied every step from the start of the
            episode, exactly as in the original (pre-revision) environment.
        """
        assert self._soldier_pos is not None, "Call reset() before step()"

        standby_mode = self.config.defender_standby_until_detection
        detected_at_step_start = self._enemy_detected

        # Positions BEFORE this step's motion (journal extension, Phase 2):
        # needed for swept-motion obstacle-collision detection below, since
        # checking only the endpoint position can miss a fast entity
        # tunneling through a thin obstacle within one dt.
        defender_prev_pos = self._defender_pos.copy()
        enemy_prev_pos = self._enemy_pos.copy()
        soldier_prev_pos = self._soldier_pos.copy()
        swept_contact_mode = self.config.mission_contact_mode == "swept"
        self._contact_info = self._empty_contact_info()

        # Advance dynamic obstacles (journal extension) BEFORE the enemy's
        # own move, so its avoidance planning this step reacts to their
        # current (this-step) position/velocity. Prev positions are kept
        # for the swept collision diagnostics computed below.
        dynamic_obstacles_prev = self._dynamic_obstacles
        if self.config.dynamic_obstacles_enabled:
            self._dynamic_obstacles = step_dynamic_obstacles(
                self._dynamic_obstacles, self._rng_dynamic_obstacles, self.config
            )

        # A. Move soldier stochastically (uncontrolled) -- unchanged model.
        self._move_soldier()

        # B. Standby hand-off: while undetected, hold the defender exactly
        # co-located with the (just-moved) soldier, ignoring the supplied
        # action entirely. Must happen BEFORE the hostile moves so that
        # reactive evasion (which reads self._defender_pos) sees the
        # CURRENT defender location during standby.
        if standby_mode and not detected_at_step_start:
            self._defender_pos = self._soldier_pos.copy()
            self._defender_vel = np.zeros(3, dtype=np.float32)
            self._defender_dynamics_info = {
                "accel_used": 0.0,
                "turn_rate_used_deg": 0.0,
                "accel_saturated": False,
                "turn_saturated": False,
                "climb_saturated": False,
            }

        # C. Move enemy with weaving pursuit + reactive evasion toward soldier (uncontrolled)
        self._move_enemy()

        # D. Update enemy detection state and Kalman tracking estimates.
        # Called here, in step(), so that it is a first-class environment
        # behavior available to ANY controller — not tied to RL logic.
        # Uses the positions after this step's entity movements but before
        # the defender displacement, matching pre-refactor semantics.
        self._update_detection()

        # just_detected: True only on the transition where detection changes
        # False -> True during this very step (computed from D's result).
        just_detected = (not detected_at_step_start) and self._enemy_detected

        # E. Defender action: execute the supplied action through the
        # existing constrained dynamics ONLY when the controller was already
        # in control at the START of this step (detected_at_step_start).
        # In legacy mode (standby_mode=False), the action is always executed
        # -- the original, pre-revision behavior.
        if standby_mode:
            controller_action_executed = bool(detected_at_step_start)
            if controller_action_executed:
                self._move_defender(action)
            # else: standby / detection-transition step -- defender remains
            # exactly at the position set in step B (co-located with the
            # soldier), with zero velocity; the supplied action is discarded.
        else:
            self._move_defender(action)
            controller_action_executed = True

        # Swept-motion obstacle-collision diagnostics (journal extension,
        # Phase 2). Starting with the physical-collision-semantics phase
        # (Phase 3/4/6/7 below), a detected collision is no longer purely
        # diagnostic: it clamps the colliding entity's position to its
        # first physical-contact point and terminates the episode -- see
        # the precedence block after distance computation, below.
        #
        # Phase-2 semantic fix: while the defender is held in pre-detection
        # standby (controller_action_executed=False), synchronizing its
        # position onto the just-moved soldier is bookkeeping, NOT physical
        # UAV flight -- the segment (previous defender position) -> (synced
        # soldier position) must NOT be swept-tested, or a fictitious
        # "flight through a wall" could be reported even though the
        # defender never actually flew anywhere. In that case we test only
        # the current (synchronized) position for containment (a
        # zero-length segment), never the bookkeeping displacement. Once
        # controller_action_executed=True, normal swept-motion checking
        # against the real pre-step position resumes.
        if controller_action_executed:
            defender_collision_segment_start = defender_prev_pos
        else:
            defender_collision_segment_start = self._defender_pos

        # Swept mission contacts (mission_contact_mode == "swept"): all three
        # bodies move linearly over the step. Standby synchronization of the
        # defender onto the asset is bookkeeping, not flight, so (exactly as
        # for the obstacle sweep above) the defender is treated as
        # stationary at its post-sync position that step.
        capture_hit = interception_hit = None
        collision_snapshot = None
        if swept_contact_mode:
            capture_hit = first_contact(
                enemy_prev_pos, self._enemy_pos, soldier_prev_pos, self._soldier_pos, self.config.threat_radius,
            )
            interception_hit = first_contact(
                defender_collision_segment_start, self._defender_pos, enemy_prev_pos, self._enemy_pos,
                self.config.intercept_radius,
            )
            collision_snapshot = self._snapshot_collision_state()
        defender_collision = self._obstacle_layout.first_segment_intersection(
            defender_collision_segment_start, self._defender_pos
        )
        enemy_collision = self._obstacle_layout.first_segment_intersection(enemy_prev_pos, self._enemy_pos)

        self._defender_obstacle_collision = defender_collision.intersects
        self._enemy_obstacle_collision = enemy_collision.intersects
        self._defender_collision_obstacle_index = defender_collision.obstacle_index
        self._enemy_collision_obstacle_index = enemy_collision.obstacle_index
        self._defender_collision_point = defender_collision.point
        self._enemy_collision_point = enemy_collision.point
        self._defender_min_obstacle_clearance = self._obstacle_layout.min_clearance(self._defender_pos)
        self._enemy_min_obstacle_clearance = self._obstacle_layout.min_clearance(self._enemy_pos)

        # Collision EVENT counting: only an "outside -> collision" transition
        # increments the count. An entity that remains inside/overlapping an
        # obstacle across multiple steps (possible if a caller keeps
        # stepping past a collision-terminated episode) is counted once,
        # not once per step.
        if self._defender_obstacle_collision and not self._defender_was_in_collision:
            self._defender_collision_count += 1
            self._defender_ever_collided = True
        self._defender_was_in_collision = self._defender_obstacle_collision
        if self._enemy_obstacle_collision and not self._enemy_was_in_collision:
            self._enemy_collision_count += 1
            self._enemy_ever_collided = True
        self._enemy_was_in_collision = self._enemy_obstacle_collision

        # Dynamic-obstacle swept collision diagnostics (journal extension).
        # ACTUAL (not predicted) per-step collision. Tracks the EARLIEST
        # dynamic-obstacle first-contact time/point per mover (Phase 6/7:
        # first-event-wins), for comparison against the static contact time
        # below.
        self._enemy_min_dynamic_obstacle_clearance = self._dynamic_min_clearance(self._enemy_pos)
        self._defender_min_dynamic_obstacle_clearance = self._dynamic_min_clearance(self._defender_pos)
        enemy_dynamic_collision = False
        enemy_collision_obstacle_id = None
        enemy_dynamic_hit_t: float | None = None
        enemy_dynamic_hit_point: np.ndarray | None = None
        defender_dynamic_collision = False
        defender_collision_obstacle_id = None
        defender_dynamic_hit_t: float | None = None
        defender_dynamic_hit_point: np.ndarray | None = None
        for obs_prev, obs in zip(dynamic_obstacles_prev, self._dynamic_obstacles):
            enemy_t_hit = swept_sphere_vs_sphere_first_contact(
                enemy_prev_pos, self._enemy_pos, obs_prev.position, obs.position, obs.radius
            )
            if enemy_t_hit is not None:
                enemy_dynamic_collision = True
                if enemy_dynamic_hit_t is None or enemy_t_hit < enemy_dynamic_hit_t:
                    enemy_dynamic_hit_t = enemy_t_hit
                    enemy_dynamic_hit_point = enemy_prev_pos + enemy_t_hit * (self._enemy_pos - enemy_prev_pos)
                    enemy_collision_obstacle_id = obs.id
            defender_t_hit = swept_sphere_vs_sphere_first_contact(
                defender_prev_pos, self._defender_pos, obs_prev.position, obs.position, obs.radius
            )
            if defender_t_hit is not None:
                defender_dynamic_collision = True
                if defender_dynamic_hit_t is None or defender_t_hit < defender_dynamic_hit_t:
                    defender_dynamic_hit_t = defender_t_hit
                    defender_dynamic_hit_point = defender_prev_pos + defender_t_hit * (self._defender_pos - defender_prev_pos)
                    defender_collision_obstacle_id = obs.id
        self._enemy_dynamic_obstacle_collision = enemy_dynamic_collision
        if enemy_dynamic_collision:
            self._enemy_collision_obstacle_id = enemy_collision_obstacle_id
        self._defender_dynamic_obstacle_collision = defender_dynamic_collision
        if defender_dynamic_collision:
            self._defender_collision_obstacle_id = defender_collision_obstacle_id
        if enemy_dynamic_collision and not self._enemy_was_in_dynamic_collision:
            self._enemy_dynamic_collision_count += 1
            self._enemy_dynamic_ever_collided = True
        self._enemy_was_in_dynamic_collision = enemy_dynamic_collision
        if defender_dynamic_collision and not self._defender_was_in_dynamic_collision:
            self._defender_dynamic_collision_count += 1
            self._defender_dynamic_ever_collided = True
        self._defender_was_in_dynamic_collision = defender_dynamic_collision

        # Physical-collision termination (journal extension, Phase 3/4/6/7):
        # for each mover independently, compare the static entry time
        # against the dynamic first-contact time and keep whichever is
        # EARLIER (first-event-wins) as that mover's collision event this
        # step. The colliding mover's position is clamped to its
        # first-contact point (never left past the obstacle surface) and
        # its velocity zeroed; see the outcome precedence block below for
        # how this terminates the episode.
        self._defender_collision_event_type = None
        self._defender_collision_event_point = None
        self._defender_collision_event_speed = 0.0
        self._enemy_collision_event_type = None
        self._enemy_collision_event_point = None
        self._enemy_collision_event_speed = 0.0

        defender_static_t = defender_collision.t_enter if defender_collision.intersects else None
        if defender_static_t is not None and (
            defender_dynamic_hit_t is None or defender_static_t <= defender_dynamic_hit_t
        ):
            self._defender_collision_event_type = "static"
            self._defender_collision_event_point = defender_collision.point
        elif defender_dynamic_hit_t is not None:
            self._defender_collision_event_type = "dynamic"
            self._defender_collision_event_point = defender_dynamic_hit_point

        enemy_static_t = enemy_collision.t_enter if enemy_collision.intersects else None
        if enemy_static_t is not None and (
            enemy_dynamic_hit_t is None or enemy_static_t <= enemy_dynamic_hit_t
        ):
            self._enemy_collision_event_type = "static"
            self._enemy_collision_event_point = enemy_collision.point
        elif enemy_dynamic_hit_t is not None:
            self._enemy_collision_event_type = "dynamic"
            self._enemy_collision_event_point = enemy_dynamic_hit_point

        # Swept-contact event arbitration (V2): the physically EARLIEST event
        # in this step wins -- defender/attacker obstacle collision, attacker
        # capture of the asset, or defender interception. Ties within
        # EVENT_TIE_TOLERANCE (in step fractions) are broken by the fixed
        # priority: defender collision > attacker collision > asset capture
        # > interception (mission failure outranks mission success, and
        # collisions outrank both, matching the V1 end-of-step precedence).
        # Events that lose are discarded: a later collision is un-counted
        # (collision bookkeeping above is rolled back) so it can never end
        # or annotate an episode a mission event already ended.
        terminal_event: str | None = None
        terminal_fraction = 1.0
        if swept_contact_mode:
            def _collision_fraction(event_type, static_t, dynamic_t):
                if event_type == "static":
                    return static_t
                if event_type == "dynamic":
                    return dynamic_t
                return None

            candidates = []
            defender_col_t = _collision_fraction(
                self._defender_collision_event_type, defender_static_t, defender_dynamic_hit_t,
            )
            enemy_col_t = _collision_fraction(
                self._enemy_collision_event_type, enemy_static_t, enemy_dynamic_hit_t,
            )
            if defender_col_t is not None:
                candidates.append((float(defender_col_t), 0, "defender_collision"))
            if enemy_col_t is not None:
                candidates.append((float(enemy_col_t), 1, "enemy_collision"))
            if capture_hit is not None:
                candidates.append((capture_hit.fraction, 2, "capture"))
            if interception_hit is not None:
                candidates.append((interception_hit.fraction, 3, "interception"))
            if candidates:
                earliest = min(c[0] for c in candidates)
                terminal_fraction, _, terminal_event = min(
                    (c for c in candidates if c[0] <= earliest + EVENT_TIE_TOLERANCE),
                    key=lambda c: c[1],
                )
            if terminal_event != "defender_collision":
                self._restore_collision_state(collision_snapshot, "defender")
                self._defender_collision_event_type = None
                self._defender_collision_event_point = None
            if terminal_event != "enemy_collision":
                self._restore_collision_state(collision_snapshot, "enemy")
                self._enemy_collision_event_type = None
                self._enemy_collision_event_point = None
            if terminal_event in ("capture", "interception"):
                # State at the contact instant; distances/classification
                # below are evaluated on these interpolated positions.
                tau = terminal_fraction
                self._enemy_pos = (enemy_prev_pos + tau * (self._enemy_pos - enemy_prev_pos)).astype(np.float32)
                self._defender_pos = (
                    defender_collision_segment_start + tau * (self._defender_pos - defender_collision_segment_start)
                ).astype(np.float32)
                self._soldier_pos = (soldier_prev_pos + tau * (self._soldier_pos - soldier_prev_pos)).astype(np.float32)
            self._contact_info.update(
                capture_contact_fraction=None if capture_hit is None else capture_hit.fraction,
                capture_contact_position=None if capture_hit is None else capture_hit.position_a.copy(),
                capture_contact_distance=None if capture_hit is None else capture_hit.distance,
                interception_contact_fraction=None if interception_hit is None else interception_hit.fraction,
                interception_contact_position=None if interception_hit is None else interception_hit.position_b.copy(),
                interception_contact_distance=None if interception_hit is None else interception_hit.distance,
                terminal_event=terminal_event,
                terminal_event_fraction=terminal_fraction if terminal_event is not None else None,
            )

        if self._defender_collision_event_type is not None:
            self._defender_collision_event_speed = float(np.linalg.norm(self._defender_vel))
            self._defender_pos = self._defender_collision_event_point.astype(np.float32)
            self._defender_vel = np.zeros(3, dtype=np.float32)
        if self._enemy_collision_event_type is not None:
            self._enemy_collision_event_speed = float(np.linalg.norm(self._enemy_vel))
            self._enemy_pos = self._enemy_collision_event_point.astype(np.float32)
            self._enemy_vel = np.zeros(3, dtype=np.float32)

        self._step_count += 1
        
        # Calculate distances (uses the post-clamp position on a collision
        # step; irrelevant to the outcome in that case since the collision
        # precedence branch below wins regardless of these values).
        enemy_soldier_dist = np.linalg.norm(self._enemy_pos - self._soldier_pos)
        defender_enemy_dist = np.linalg.norm(self._defender_pos - self._enemy_pos)
        defender_soldier_dist = np.linalg.norm(self._defender_pos - self._soldier_pos)
        
        # Check termination conditions using dist_de and dist_es
        dist_de = defender_enemy_dist  # defender to enemy
        dist_es = enemy_soldier_dist   # enemy to soldier
        
        # Check if defender is close enough to intercept
        can_intercept = dist_de <= self.config.intercept_radius
        
        # Check if intercept would be unsafe (too close to soldier)
        unsafe_zone = dist_es <= self.config.unsafe_intercept_radius
        
        # WIN: Safe intercept (defender catches enemy, far enough from soldier)
        intercepted = can_intercept and not unsafe_zone
        
        # LOSS: Enemy reaches soldier (threat zone)
        soldier_caught = dist_es <= self.config.threat_radius
        
        # LOSS: Unsafe intercept (caught enemy but too close to soldier - collateral risk)
        unsafe_intercept = can_intercept and unsafe_zone and not soldier_caught

        if swept_contact_mode:
            # Contact outcomes come from the swept event arbitration above;
            # the end-of-step flags just computed are superseded. The
            # safe/unsafe classification still uses the existing
            # unsafe_intercept_radius, evaluated at the interpolated
            # interception instant (positions were set to that instant).
            soldier_caught = terminal_event == "capture"
            can_intercept = terminal_event == "interception"
            intercepted = can_intercept and not unsafe_zone
            unsafe_intercept = can_intercept and unsafe_zone
        
        truncated = False

        # Step-order precedence (journal extension, Phase 6): a physical
        # obstacle collision this step takes priority over EVERY
        # mission-geometry outcome that would otherwise also be true this
        # same step (e.g. a defender that clips a wall exactly as it
        # reaches interception range has crashed, not intercepted; a
        # hostile that clips a wall exactly as it reaches the soldier has
        # suffered an environmental failure, not "caught" the soldier).
        # Ties between the defender's own event and the hostile's are
        # broken in favor of the defender's event -- an explicit,
        # documented convention (not a claim of physical earliness across
        # the two independent movers).
        if self._defender_collision_event_type == "static":
            outcome = "defender_static_obstacle_collision"
            reward = self.config.reward_defender_static_obstacle_collision
            terminated = True
        elif self._defender_collision_event_type == "dynamic":
            outcome = "defender_dynamic_obstacle_collision"
            reward = self.config.reward_defender_dynamic_obstacle_collision
            terminated = True
        elif self._enemy_collision_event_type == "static":
            outcome = "enemy_static_obstacle_collision"
            reward = self.config.reward_enemy_static_obstacle_collision
            terminated = True
        elif self._enemy_collision_event_type == "dynamic":
            outcome = "enemy_dynamic_obstacle_collision"
            reward = self.config.reward_enemy_dynamic_obstacle_collision
            terminated = True
        else:
            # Episode terminates on any terminal condition or timeout
            terminated = intercepted or soldier_caught or unsafe_intercept or (self._step_count >= self.config.max_steps)

            # Determine outcome and reward
            if soldier_caught:
                # Check soldier_caught FIRST (highest priority failure)
                outcome = "soldier_caught"
                reward = self.config.reward_soldier_caught  # LOSS
            elif unsafe_intercept:
                # Unsafe intercept: caught enemy but too close to soldier
                outcome = "unsafe_intercept"
                reward = self.config.reward_unsafe_intercept  # LOSS
            elif intercepted:
                # Safe intercept (WIN)
                outcome = "intercepted"
                reward = self.config.reward_intercept  # WIN
            elif self._step_count >= self.config.max_steps:
                outcome = "timeout"
                reward = self.config.reward_timeout  # LOSS (failed to intercept)
            else:
                outcome = "ongoing"
                reward = 0.0
                
                # 1. Progress reward: positive for closing distance to enemy (using TRUE position)
                progress = self._prev_defender_enemy_dist - dist_de
                reward += self.config.reward_progress_scale * progress
                
                # 2. Small time penalty (encourages efficiency)
                reward += self.config.reward_time_penalty
                
                # 3. Proximity warning: penalty when enemy gets close to soldier
                # Scaled by how close enemy is (inverse distance)
                proximity_threshold = self.config.unsafe_intercept_radius * 3.0  # ~10.5 units
                if dist_es < proximity_threshold:
                    # Stronger penalty as enemy gets closer
                    proximity_factor = 1.0 - (dist_es / proximity_threshold)
                    reward += self.config.reward_proximity_warning * proximity_factor
                
                # NOTE: no hidden reward clipping. The executed ongoing reward is
                # exactly r_t = reward_progress_scale*(d_{t-1}-d_t) + reward_time_penalty
                # + proximity term. This is identical for Direct and Kalman tracks
                # -- neither track receives a Kalman-only tracking-error reward
                # (tracking error remains an EVALUATION metric only, see
                # info["tracking_error"]). During standby, action-independence
                # follows automatically: dist_de/dist_es evolve purely from the
                # soldier/hostile motion models, so this reward term cannot be
                # influenced by the (discarded) supplied action.
        
        # Update previous distance for next step
        self._prev_defender_enemy_dist = defender_enemy_dist
        self._update_attacker_efficiency(enemy_prev_pos, terminal_fraction)
        
        return self._get_obs(), reward, terminated, truncated, self._get_info(
            outcome, enemy_soldier_dist, defender_enemy_dist,
            defender_standby=(standby_mode and not controller_action_executed),
            controller_action_executed=controller_action_executed,
            just_detected=just_detected,
        )
    
    def _advance_velocity(
        self,
        current_velocity: np.ndarray,
        desired_velocity: np.ndarray,
        max_speed: float,
        max_accel: float,
        max_turn_rate_rad: float,
        max_climb_rate: float,
        max_descent_rate: float,
    ) -> tuple[np.ndarray, dict]:
        """
        Advance a persistent 3D point-mass velocity toward a desired velocity,
        subject to a maximum horizontal turn rate, maximum acceleration,
        maximum total speed, and maximum climb/descent rate.

        This is the single reusable constrained-velocity update shared by
        both the defender and the hostile UAV (a 3D constrained point-mass
        dynamics model — NOT six-DOF or aerodynamic flight dynamics).

        Order of operations:
          1. Vertical-rate command limiting: clip desired_velocity[2] into
             [-max_descent_rate, max_climb_rate].
          2. Horizontal turn-rate limiting: rotate the current horizontal
             heading toward the desired horizontal heading by at most
             max_turn_rate_rad * dt (shortest signed angular path, wrapped
             to [-pi, pi]), adopting the desired horizontal speed magnitude.
             Special cases (documented explicitly, not merely defaulted):
               - If current horizontal speed <= eps, the previous heading is
                 undefined; heading initializes directly toward the desired
                 heading with NO rate limit applied (there is nothing to
                 rate-limit against). Acceleration limiting (step 3) still
                 governs how quickly speed itself can rise.
               - If desired horizontal speed <= eps (e.g. a "stop" command),
                 heading has no defined target; the current heading is held
                 while horizontal speed is commanded toward zero.
             The result is the "dynamically permissible target velocity".
          3. Acceleration limiting: clip the change from current_velocity to
             the target velocity to at most max_accel * dt in magnitude,
             preserving the direction of the change. Because this is
             equivalent to next = current + t*(target - current) for some
             t in [0, 1], the result is a convex combination of current and
             target.
          4. Total-speed / vertical-rate safety enforcement: since both
             current_velocity and the target velocity already satisfy the
             max_speed ball and the climb/descent bounds (both are convex
             constraints), and step 3 produces a convex combination of the
             two, the result satisfies both bounds automatically. A
             defensive clip is applied regardless, to guard against
             floating-point overshoot (never used to silently paper over a
             design error).

        Args:
            current_velocity: Current persistent velocity [vx, vy, vz], shape (3,).
            desired_velocity: Controller/guidance-commanded desired velocity
                [vx, vy, vz], shape (3,); ||desired_velocity|| is expected to
                be <= max_speed (e.g. max_speed * unit_direction).
            max_speed: Maximum total speed (m/s).
            max_accel: Maximum acceleration magnitude (m/s^2).
            max_turn_rate_rad: Maximum horizontal heading-rate magnitude (rad/s).
            max_climb_rate: Maximum upward vertical speed (m/s), positive.
            max_descent_rate: Maximum downward vertical speed magnitude (m/s), positive.

        Returns:
            (next_velocity, diagnostics):
                next_velocity: Constrained velocity, shape (3,), float32.
                diagnostics: dict with keys "accel_used" (m/s^2, actually
                    applied this step), "turn_rate_used_deg" (deg/s, actually
                    applied this step), "accel_saturated" (bool),
                    "turn_saturated" (bool), "climb_saturated" (bool, True if
                    the vertical-rate command limit in step 1 was active).
        """
        return advance_velocity(
            current_velocity=current_velocity,
            desired_velocity=desired_velocity,
            max_speed=max_speed,
            max_accel=max_accel,
            max_turn_rate_rad=max_turn_rate_rad,
            max_climb_rate=max_climb_rate,
            max_descent_rate=max_descent_rate,
            dt=self.config.dt,
            eps=self.config.eps,
        )
    
    def _apply_boundary(self, pos: np.ndarray, vel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Clip position to the 3D engagement volume (x, y \u2208 [-L, L], z \u2208 [0, H])
        and zero ONLY the outward-normal velocity component at any boundary
        the position has reached. Tangential velocity components are
        preserved unchanged (no bounce/reflection for persistent-velocity
        vehicles).
        
        For example, at z = 0 with vz < 0 (descending into the ground),
        vz is set to 0 while vx, vy are untouched. This prevents outward
        velocity from continuing to accumulate against a boundary while the
        vehicle is resting against it.
        
        Args:
            pos: Position array of shape (3,).
            vel: Persistent velocity array of shape (3,).
        
        Returns:
            (clipped_pos, clipped_vel), each of shape (3,).
        """
        return apply_boundary(pos, vel, L=self.config.L, max_altitude=self.config.max_altitude)
    
    def _move_defender(self, action: np.ndarray) -> None:
        """
        Move the defender drone using constrained 3D point-mass dynamics.
        
        Args:
            action: 3D continuous action vector in [-1, 1]³, interpreted as
                   a DESIRED velocity direction. Normalized to a unit vector
                   if norm > eps -> desired_velocity = v_d * direction. If
                   action norm <= eps, desired_velocity = [0,0,0] (command
                   to decelerate toward a stop).
        
        The defender's persistent velocity (self._defender_vel) approaches
        this desired velocity subject to defender_max_accel,
        defender_max_turn_rate_deg, defender_max_climb_rate, and
        defender_max_descent_rate (see _advance_velocity). Position is then
        integrated as p_{t+1} = p_t + v_{t+1} * dt, followed by physically
        consistent boundary handling (_apply_boundary): outward velocity is
        zeroed only if the position actually reached a boundary.
        
        Note:
            Detection and Kalman tracking are managed independently by
            _update_detection(), called from step() before this method.
        """
        eps = self.config.eps
        
        action = np.asarray(action, dtype=np.float32)
        action_norm = np.linalg.norm(action)
        
        if action_norm > eps:
            u_desired = action / action_norm
        else:
            u_desired = np.zeros(3, dtype=np.float32)
        self._defender_desired_command = u_desired.astype(np.float32)

        # Shared, controller-agnostic obstacle-safety layer (journal
        # extension): identical mechanism to the hostile's, applied to
        # whatever `action` ANY defender controller family produced
        # (Greedy/PN/Lead/RMPC/PPO/PPO-Kalman/LR-PPO all funnel through
        # this SAME call site, unmodified upstream -- see
        # docs/defender_command_pipeline_audit.md). Exact no-op
        # (u_safe == u_desired, bit-for-bit) when disabled or when no
        # collision is predicted.
        u_safe = u_desired
        if self.config.defender_obstacle_avoidance_enabled:
            perceived = self._defender_perception.update(
                self._defender_pos, self._obstacle_layout, self._dynamic_obstacles
            )
            self._defender_last_perceived = perceived
            perceived_static_layout = perceived.to_static_layout()
            avoidance_layout = self._combined_avoidance_layout(
                self._defender_pos, self._defender_vel,
                self.config.defender_obstacle_prediction_horizon_steps, self.config.defender_obstacle_clearance,
                perceived_static_layout, perceived.dynamic,
            )
            if (
                self._defender_obstacle_planner.active
                and (
                    self._defender_obstacle_planner.obstacle_index is None
                    or self._defender_obstacle_planner.obstacle_index >= len(avoidance_layout)
                )
            ):
                self._defender_obstacle_planner.reset()
            # Mission target (Phase 14): the hostile UAV is the
            # interception target, NEVER environmental clutter -- it is
            # passed here only as the candidate-selection tie-break
            # reference (matching the hostile's own use of its mission
            # target), never inserted into `avoidance_layout`.
            plan = self._defender_obstacle_planner.plan(
                position=self._defender_pos,
                velocity=self._defender_vel,
                u_base=u_desired,
                layout=avoidance_layout,
                target_position=self._enemy_pos,
                limits=self._defender_dynamics_limits(),
            )
            u_safe = np.asarray(plan["direction"], dtype=np.float64)
            self._defender_obstacle_avoidance_active = plan["active"]
            self._defender_obstacle_avoidance_mode = plan["mode"]
            self._defender_predicted_min_clearance = plan["predicted_min_clearance"]
            n_static = len(perceived.static)
            if plan["obstacle_index"] is None:
                self._defender_avoidance_obstacle_type = None
                self._defender_avoidance_obstacle_id = None
            elif plan["obstacle_index"] < n_static:
                self._defender_avoidance_obstacle_type = "static"
                self._defender_avoidance_obstacle_id = perceived.static[plan["obstacle_index"]].id
            else:
                self._defender_avoidance_obstacle_type = perceived.dynamic[plan["obstacle_index"] - n_static].obstacle_type
                self._defender_avoidance_obstacle_id = perceived.dynamic[plan["obstacle_index"] - n_static].id
            cos_angle = float(np.clip(np.dot(np.asarray(u_safe, dtype=np.float64), u_desired.astype(np.float64)), -1.0, 1.0))
            self._defender_avoidance_angle_deg = float(np.degrees(np.arccos(cos_angle)))
        else:
            self._defender_obstacle_avoidance_active = False
            self._defender_obstacle_avoidance_mode = None
            self._defender_avoidance_obstacle_type = None
            self._defender_avoidance_obstacle_id = None
            self._defender_avoidance_angle_deg = 0.0
            self._defender_predicted_min_clearance = self._obstacle_layout.min_clearance(self._defender_pos)
        self._defender_safe_command = np.asarray(u_safe, dtype=np.float32)

        desired_velocity = self.config.v_d * np.asarray(u_safe, dtype=np.float32)
        
        next_vel, diag = self._advance_velocity(
            current_velocity=self._defender_vel,
            desired_velocity=desired_velocity,
            max_speed=self.config.v_d,
            max_accel=self.config.defender_max_accel,
            max_turn_rate_rad=np.radians(self.config.defender_max_turn_rate_deg),
            max_climb_rate=self.config.defender_max_climb_rate,
            max_descent_rate=self.config.defender_max_descent_rate,
        )
        self._defender_dynamics_info = diag
        
        new_pos = self._defender_pos + next_vel * self.config.dt
        new_pos, next_vel = self._apply_boundary(new_pos, next_vel)
        
        self._defender_pos = new_pos.astype(np.float32)
        self._defender_vel = next_vel.astype(np.float32)
    
    def _defender_dynamics_limits(self) -> DynamicsLimits:
        """Bundles the defender's constrained-dynamics parameters for
        reuse by the obstacle-navigation predictor, mirroring
        `_enemy_dynamics_limits()` -- guarantees the predictor's dynamics
        model can never drift out of sync with the real per-step update
        above."""
        return DynamicsLimits(
            max_speed=self.config.v_d,
            max_accel=self.config.defender_max_accel,
            max_turn_rate_rad=np.radians(self.config.defender_max_turn_rate_deg),
            max_climb_rate=self.config.defender_max_climb_rate,
            max_descent_rate=self.config.defender_max_descent_rate,
            dt=self.config.dt,
            eps=self.config.eps,
            L=self.config.L,
            max_altitude=self.config.max_altitude,
        )
    
    def _update_detection(self) -> None:
        """
        Update enemy detection state, the standardized Direct-track
        measurement-velocity estimate, and Kalman tracking estimates.

        Called once per step() BEFORE defender displacement, after all entity
        positions have been updated. This is a standalone environment behavior:
        the results are available to any controller through info['e_hat'],
        info['v_hat'], info['enemy_measurement'], info['enemy_measurement_velocity'],
        and info['tracking_error'] — no RL wrapper required.

        Detection occurs when the defender is within detection_radius of the
        enemy. Once detected, tracking persists for the rest of the episode.

        Every step after detection, a common noisy 3-D position measurement
        is drawn from the dedicated sensor RNG stream (self._rng_sensor),
        shared by ALL controller modes -- this is the single sensor
        realization consumed by both the Direct and Kalman tracks. From
        consecutive measurements, a raw finite-difference measurement
        velocity is derived (no smoothing/filtering):

            v_hat_meas(t) = (z_t - z_{t-1}) / dt

        valid only from the second measurement onward (see
        self._enemy_measurement_velocity_valid).

        If use_kalman_tracking=True (KalmanGreedyInterceptPolicy, PPO-Kalman,
        PN-Kalman, Lead-Kalman):
            - EnemyKalmanFilter is initialized on first detection.
            - predict() + update() called every subsequent step using the
              SAME measurement drawn above.
            - e_hat and v_hat carry filtered estimates with proper uncertainty.
            - tracking_error in info is the Euclidean filter error (an
              EVALUATION metric only -- never part of the training reward).

        If use_kalman_tracking=False (Greedy, PN, Lead, PPO Direct -- the
        environment default):
            - e_hat mirrors the raw measurement; v_hat mirrors the raw
              finite-difference measurement velocity (zeros until valid).
            - tracking_error in info is the measurement error magnitude.
        """
        was_detected = self._enemy_detected
        if not was_detected:
            defender_enemy_dist = np.linalg.norm(self._enemy_pos - self._defender_pos)
            if defender_enemy_dist > self.config.detection_radius:
                return
            self._enemy_detected = True  # Start tracking!

        # Common sensor step (dedicated sensor RNG stream): the SAME noisy
        # measurement realization is used by every controller mode; only
        # subsequent estimator processing (Kalman vs raw) differs.
        self._enemy_measurement = (
            self._enemy_pos
            + self._rng_sensor.normal(0.0, self._measurement_noise_std, size=(3,))
        ).astype(np.float32)

        # Standardized Direct-track finite-difference measurement velocity:
        # the single common estimator shared by the Direct observation and
        # any measurement-mode controller (Greedy, PN, Lead). No smoothing.
        if self._prev_enemy_measurement is not None:
            self._enemy_measurement_velocity = (
                (self._enemy_measurement.astype(np.float64) - self._prev_enemy_measurement.astype(np.float64))
                / self.config.dt
            ).astype(np.float32)
            self._enemy_measurement_velocity_valid = True
        else:
            # First-ever measurement: velocity not yet observable.
            self._enemy_measurement_velocity = np.zeros(3, dtype=np.float32)
            self._enemy_measurement_velocity_valid = False
        self._prev_enemy_measurement = self._enemy_measurement.copy()

        if self.config.use_kalman_tracking:
            if not was_detected:
                # Initialize Kalman filter with first noisy measurement
                self._kf = EnemyKalmanFilter(
                    dt=self.config.dt,
                    process_var=self.config.process_var,
                    measurement_var=self.config.measurement_var,
                )
                self._kf.initialize(self._enemy_measurement.astype(np.float64))
            else:
                # Kalman predict + update with this step's noisy measurement
                self._kf.predict()
                self._kf.update(self._enemy_measurement.astype(np.float64))

            # Apply lead time prediction if configured
            if self.config.lead_time > 0.0:
                # Extrapolate position forward by lead_time
                pos = self._kf.get_position()
                vel = self._kf.get_velocity()
                self._e_hat = (pos + vel * self.config.lead_time).astype(np.float32)
            else:
                self._e_hat = self._kf.get_position().astype(np.float32)
            self._v_hat = self._kf.get_velocity().astype(np.float32)
        else:
            # Direct/measurement track: e_hat/v_hat mirror the standardized
            # raw-measurement + finite-difference-velocity estimator, giving
            # Direct and Kalman identical observation SEMANTICS (position +
            # velocity estimate of the hostile UAV) -- only the estimator
            # differs.
            self._e_hat = self._enemy_measurement.copy()
            self._v_hat = self._enemy_measurement_velocity.copy()
    
    def _move_soldier(self) -> None:
        """
        Move the soldier using `self.config.asset_motion_model`.

        "legacy_gaussian_random_walk" (default; the ONLY model V1 and the canonical V2 benchmark
        use) is UNCHANGED from before this journal model-study extension existed -- see
        `_move_soldier_legacy` below, bit-identical including its RNG draw sequence.

        "gauss_markov_velocity" is an optional, config-gated alternative studied for whether a
        temporally correlated asset-motion process changes benchmark behavior -- see
        `_move_soldier_gauss_markov` below. It is never selected by any canonical config builder.
        """
        if self.config.asset_motion_model == "gauss_markov_velocity":
            self._move_soldier_gauss_markov()
        else:
            self._move_soldier_legacy()

    def _move_soldier_legacy(self) -> None:
        """
        Move the soldier with a true Gaussian random walk.
        
        The soldier remains a ground entity: only x, y are perturbed and
        z is held fixed at 0.0 at all times (no vertical motion).
        
        - Sample 2D (horizontal) Gaussian displacement with scale σ = v_s * dt
        - Variable step magnitude (Gaussian-distributed)
        - Reflecting boundary conditions at [-L, L]² (horizontal only)

        Obstacle-blocked motion (journal extension, Phase 3): the proposed
        displacement (and its RNG draw) is computed EXACTLY as before,
        regardless of obstacles. If the resulting candidate segment
        (previous position -> candidate position, both at z=0) would
        enter/cross any obstacle footprint, the displacement is REJECTED
        and the soldier holds its previous valid position for this step --
        it is NEVER resampled, so the soldier RNG stream's sequence of
        draws is identical whether or not obstacles are enabled (see
        test_soldier_obstacle_motion.py).
        """
        # True Gaussian random walk: sample horizontal displacement directly
        # Scale (standard deviation) determines typical step size
        sigma = self.config.v_s * self.config.dt
        displacement_xy = self._rng_soldier.normal(loc=0.0, scale=sigma, size=(2,))
        displacement = np.array(
            [displacement_xy[0], displacement_xy[1], 0.0], dtype=np.float32
        )
        
        # Update position
        new_pos = self._soldier_pos + displacement
        
        # Reflecting boundary conditions at [-L, L]² × [0, H]
        new_pos = self._reflect_boundary(new_pos)
        
        # Soldier is a ground entity: enforce z = 0 explicitly
        new_pos[2] = 0.0

        # Obstacle-validity rule: reuse the existing layout geometry (never
        # duplicate intersection math) to swept-test the ground-plane
        # segment; boundary contact counts as blocked (AABBObstacle
        # convention). Reject-in-place, never resample.
        if self._obstacle_layout.segment_intersects(self._soldier_pos, new_pos):
            self._soldier_obstacle_blocked = True
            self._soldier_obstacle_block_count += 1
            return

        self._soldier_obstacle_blocked = False
        self._soldier_pos = new_pos.astype(np.float32)

    def _move_soldier_gauss_markov(self) -> None:
        """
        Journal model-study extension (asset_motion_model == "gauss_markov_velocity"): an
        optional, config-gated alternative to the legacy IID Gaussian random walk. Not selected
        by any canonical config builder; see docs/journal_v2_asset_motion_model_study.md.

        Replaces the per-step IID position displacement with a temporally correlated (Gauss-
        Markov / AR(1)) horizontal velocity process:

            v_{t+1} = rho * v_t + sqrt(1 - rho^2) * v_s * xi_t,   xi_t ~ N(0, I)
            p_{t+1} = p_t + v_{t+1} * dt

        with rho = exp(-dt / tau) (`asset_motion_tau_s`), so that at stationarity each velocity
        component retains standard deviation v_s -- the same per-component scale implied by the
        legacy model's displacement (std = v_s * dt over one step) -- while the PROCESS itself is
        correlated across steps instead of independent. This changes temporal persistence, not
        magnitude.

        Initialization (Phase 3): on the first call this episode there is no previous velocity to
        decay from, so v_0 = v_s * xi_0 (the stationary-scale case of the recurrence with the
        "rho * v_t" term dropped) is used instead, consuming the SAME one draw of size (2,) the
        recurrence would have consumed on every other step -- the number and shape of per-step RNG
        draws from `_rng_soldier` is identical to the legacy model's `size=(2,)` draw on every
        step, so swapping models does not change how many random numbers are consumed per step
        (only their application differs).

        Horizontal-only, ground entity (z held at 0), reflecting boundary with the corresponding
        velocity component ALSO reflected (not left pointing outward) on any axis that reflects.

        Obstacle-blocked motion: the candidate segment is tested exactly as the legacy model's is
        (reject-in-place, never resampled); the velocity update itself is NOT rejected (it is the
        stochastic process state, evolved unconditionally, mirroring the legacy model's rule that
        the proposed draw is computed "regardless of obstacles").
        """
        dt = self.config.dt
        v_s = self.config.v_s
        rho = float(np.exp(-dt / self.config.asset_motion_tau_s))

        xi = self._rng_soldier.normal(loc=0.0, scale=1.0, size=(2,))
        if not self._soldier_gauss_markov_initialized:
            new_vel_xy = v_s * xi
            self._soldier_gauss_markov_initialized = True
        else:
            prev_vel_xy = self._soldier_vel[:2].astype(np.float64)
            new_vel_xy = rho * prev_vel_xy + np.sqrt(max(1.0 - rho * rho, 0.0)) * v_s * xi

        displacement_xy = new_vel_xy * dt
        candidate_pos = self._soldier_pos.astype(np.float64)
        candidate_pos[:2] += displacement_xy
        candidate_pos[2] = 0.0

        reflected_xy, reflected_vel_xy = self._reflect_horizontal_boundary_with_velocity(
            candidate_pos[:2], new_vel_xy
        )
        new_pos = np.array([reflected_xy[0], reflected_xy[1], 0.0], dtype=np.float32)
        self._soldier_vel = np.array([reflected_vel_xy[0], reflected_vel_xy[1], 0.0], dtype=np.float32)

        if self._obstacle_layout.segment_intersects(self._soldier_pos, new_pos):
            self._soldier_obstacle_blocked = True
            self._soldier_obstacle_block_count += 1
            return

        self._soldier_obstacle_blocked = False
        self._soldier_pos = new_pos

    def _reflect_horizontal_boundary_with_velocity(
        self, pos_xy: np.ndarray, vel_xy: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Like `_reflect_boundary`, but horizontal-only (the soldier never moves vertically) and
        also reflects the velocity component on every axis that undergoes a position reflection --
        `_reflect_boundary` itself, used everywhere else (enemy, defender), is untouched."""
        L = self.config.L
        pos_xy = np.array(pos_xy, dtype=np.float64, copy=True)
        vel_xy = np.array(vel_xy, dtype=np.float64, copy=True)
        for i in range(2):
            while pos_xy[i] < -L or pos_xy[i] > L:
                if pos_xy[i] < -L:
                    pos_xy[i] = 2 * (-L) - pos_xy[i]
                    vel_xy[i] = -vel_xy[i]
                if pos_xy[i] > L:
                    pos_xy[i] = 2 * L - pos_xy[i]
                    vel_xy[i] = -vel_xy[i]
        return pos_xy, vel_xy
    
    def _compute_enemy_evasion(
        self,
        defender_pos: np.ndarray,
        enemy_pos: np.ndarray,
    ) -> dict:
        """
        Compute the hostile UAV's reactive evasion component away from the
        defender's current position.

        This models the hostile UAV's OWN local awareness of an approaching
        interceptor (e.g. proximity/threat sensing on the attacking drone
        itself), NOT the defender's detection/tracking state. It is
        intentionally independent of self._enemy_detected -- the hostile
        UAV's behavior is not coupled to whether the defender has acquired
        a track on it.

        Activation increases smoothly (no binary switch) as the true 3D
        defender-enemy distance closes within enemy_evasion_radius:

            alpha(d) = clip(1 - d / R_e, 0, 1)

        so alpha=0 at d >= R_e, alpha=0.5 at d = R_e/2, and alpha -> 1 as
        d -> 0. The evasion component (before combination with pursuit/weave
        and before final normalization) is:

            evasion_component = enemy_evasion_gain * alpha * u_away

        where u_away = (enemy_pos - defender_pos) / ||enemy_pos - defender_pos||
        is the full 3D direction from the defender toward the hostile UAV.

        When enemy_evasion_enabled is False, activation (and therefore the
        evasion component) is forced to zero regardless of distance -- this
        provides an exact no-evasion ablation.

        Args:
            defender_pos: Defender position, shape (3,).
            enemy_pos: Hostile UAV position, shape (3,).

        Returns:
            dict with keys:
                "distance" (float): true 3D defender-enemy range.
                "activation" (float): alpha, in [0, 1].
                "effective_weight" (float): enemy_evasion_gain * alpha, in
                    [0, enemy_evasion_gain].
                "away_direction" (np.ndarray, shape (3,)): u_away.
                "evasion_component" (np.ndarray, shape (3,)): the
                    pre-combination, pre-normalization evasion vector.
        """
        eps = self.config.eps
        R_e = self.config.enemy_evasion_radius
        lambda_e = self.config.enemy_evasion_gain

        diff = enemy_pos - defender_pos
        distance = float(np.linalg.norm(diff))

        if distance > eps:
            away_direction = diff / distance
        else:
            # Defender and enemy numerically co-located: safe deterministic
            # fallback (no RNG call -- preserves reproducibility) rather
            # than dividing by zero or producing NaN.
            away_direction = np.array([0.0, 0.0, 1.0])

        if self.config.enemy_evasion_enabled:
            activation = float(np.clip(1.0 - distance / R_e, 0.0, 1.0))
        else:
            activation = 0.0

        effective_weight = lambda_e * activation
        evasion_component = effective_weight * away_direction

        return {
            "distance": distance,
            "activation": activation,
            "effective_weight": effective_weight,
            "away_direction": away_direction.astype(np.float32),
            "evasion_component": evasion_component.astype(np.float32),
        }

    def _move_enemy(self) -> None:
        """
        Move the enemy with a stochastic weaving pursuit policy in 3D, with
        proximity-triggered reactive evasion from the defender, using
        constrained point-mass dynamics.
        
        The hostile UAV retains its primary mission -- reach the protected
        soldier -- but increasingly biases its desired direction away from
        the defender as the defender closes within enemy_evasion_radius.
        This is a parameterized reactive evasive policy layered onto the
        attack mission (NOT a retreat policy, NOT intelligent adversarial
        RL, NOT game-theoretic or optimal-attacker behavior, and NOT
        proportional navigation / lead pursuit / constant-bearing guidance).
        
        Weaving pursuit GUIDANCE (unchanged from the 3D-foundation task):
        1. Compute vector to soldier: r = s - e (full 3D; since the soldier
           is always at z=0, this naturally drives the enemy to descend
           toward the ground target as it pursues)
        2. Unit vector toward soldier: r_hat = r / (||r|| + eps)  -- the
           pursuit component u_p, preserved unchanged and never replaced.
        3. Horizontal lateral unit vector: lateral = [-r_hat_y, r_hat_x, 0],
           normalized; falls back to a fixed horizontal direction if the
           pursuit direction is (near-)vertical and the horizontal
           component is degenerate (norm <= eps)
        4. Update weave bias (AR(1)): a <- rho * a + sigma_a * eta, eta ~ N(0,1)
        5. Heading noise: z ~ N(0, I_3) (generalized to 3 dimensions)
        
        REACTIVE EVASION (new in this task): _compute_enemy_evasion()
        computes a smoothly-activated 3D component pointing away from the
        defender's current position (see that method's docstring for the
        activation formula). This is added alongside pursuit/weave/noise,
        NOT in place of them.
        
        6. Unnormalized direction: u_base_raw = r_hat + a * lateral +
           sigma_e * z + evasion_component
        7. Normalize: u_base = u_base_raw / (||u_base_raw|| + eps)
        
        OBSTACLE NAVIGATION (journal extension, Phase 3B): u_base is the
        "base command" the hostile would fly if obstacles did not exist.
        When enemy_obstacle_avoidance_enabled, u_base (together with the
        hostile's current position/velocity) is handed to a deterministic,
        RNG-free `ObstacleAvoidancePlanner` (uav_defend/obstacles/avoidance.py),
        which predicts -- using the SAME constrained point-mass dynamics as
        below, over a short fixed horizon -- whether flying u_base would
        collide with an obstacle, and if so temporarily substitutes a
        deterministic bypass direction. See that module for the prediction/
        candidate-generation/selection/persistence algorithm. When
        disabled, or when no collision is predicted, the commanded
        direction u is exactly u_base (bit-for-bit): obstacle navigation
        NEVER touches pursuit/weave/noise/evasion or the RNG streams that
        produce them.
        
        DYNAMICS (unchanged from the constrained-dynamics task): guidance
        produces a DESIRED velocity direction, not instantaneous motion.
        desired_velocity = v_e * u is fed through the same reusable
        constrained-velocity update used by the defender (_advance_velocity),
        subject to enemy_max_accel, enemy_max_turn_rate_deg,
        enemy_max_climb_rate, and enemy_max_descent_rate -- evasion cannot
        bypass these limits. Position is then integrated as
        p_{t+1} = p_t + v_{t+1} * dt, followed by physically consistent
        boundary handling (_apply_boundary).
        
        Note: vertical weaving is NOT modeled here; altitude change results
        from 3D pursuit of the ground-level soldier target and/or 3D
        evasive maneuvering. Proportional navigation and lead pursuit are
        NOT modeled here (deferred to a later task).
        """
        eps = self.config.eps
        
        # Vector from enemy to soldier (full 3D)
        r = self._soldier_pos - self._enemy_pos
        r_norm = np.linalg.norm(r)
        
        # Unit vector toward soldier (pursuit component u_p)
        if r_norm > eps:
            r_hat = r / r_norm
        else:
            # Enemy is on top of soldier, pick a random 3D unit direction
            rand_vec = self._rng_enemy_motion.normal(0.0, 1.0, size=(3,))
            rand_norm = np.linalg.norm(rand_vec)
            if rand_norm > eps:
                r_hat = rand_vec / rand_norm
            else:
                r_hat = np.array([1.0, 0.0, 0.0])
        self._enemy_pursuit_direction = r_hat.astype(np.float32)
        # Command-conflict diagnostics (Phase 3): raw mission command,
        # captured BEFORE any downstream stage may overwrite r_hat.
        self._enemy_mission_command = r_hat.astype(np.float32)

        # Perception layer (journal extension): computed ONCE per step so
        # both the route planner below and the reactive avoidance layer
        # further down share the IDENTICAL PerceivedObstacleSet -- calling
        # EnemyObstaclePerception.update() twice per step would incorrectly
        # double-age dynamic-obstacle track memory. "disabled" mode
        # perceives the full ground truth (bit-for-bit prior behavior).
        perceived = None
        if self.config.enemy_obstacle_avoidance_enabled:
            perceived = self._enemy_perception.update(
                self._enemy_pos, self._obstacle_layout, self._dynamic_obstacles
            )
            self._enemy_last_perceived = perceived

        # Reactive evasion (moved earlier, Phase 5 arbitration): computed
        # BEFORE the route-planning decision below so the arbitration layer
        # knows whether an immediate defender threat is present THIS step.
        # Intentionally independent of self._enemy_detected (see
        # _compute_enemy_evasion docstring).
        evasion_info = self._compute_enemy_evasion(self._defender_pos, self._enemy_pos)
        direct_guidance = self.config.enemy_nominal_guidance_mode == "direct_obstacle_only"
        if direct_guidance:
            # Corrected scripted A0: no tactical defender evasion. The
            # defender-proximity diagnostics (range, activation) are kept for
            # logging, but the component never enters the command.
            evasion_info = dict(evasion_info)
            evasion_info["evasion_component"] = np.zeros(3, dtype=np.float32)
        self._enemy_evasion_info = evasion_info
        self._enemy_evasion_command = evasion_info["evasion_component"].astype(np.float32)
        evasion_active_this_step = (not direct_guidance) and bool(evasion_info["activation"] > 1e-6)

        # Behavior-arbitration hysteresis (Phase 7): reuse the EXISTING
        # obstacle-avoidance release-steps hysteresis parameter rather than
        # inventing a new threshold -- route suspension persists for
        # `enemy_obstacle_release_steps` consecutive non-evading steps
        # after evasion ends, preventing ROUTE<->EVADE ping-pong right at
        # the activation boundary.
        if evasion_active_this_step:
            self._enemy_evasion_clear_streak = 0
        else:
            self._enemy_evasion_clear_streak += 1
        route_suspended = self.config.enemy_route_arbitration_enabled and (
            evasion_active_this_step or self._enemy_evasion_clear_streak < self.config.enemy_obstacle_release_steps
        )
        if direct_guidance:
            route_suspended = False
        just_resumed = (not route_suspended) and self._enemy_route_was_suspended
        self._enemy_route_was_suspended = route_suspended

        # Multi-obstacle route planning (journal extension, Phase 3-15;
        # arbitration added Phase 5-6): replaces the naive straight-line
        # pursuit direction with a locally-planned waypoint direction
        # around perceived obstacle clusters -- UNLESS route following is
        # currently suspended in favor of defender evasion (Priority 2 over
        # Priority 3, Phase 5), in which case the route is left untouched
        # (never judged/replanned against evasion-induced deviation, Phase
        # 6) and the pursuit term remains the raw mission direction, still
        # combined with weave/noise/evasion below exactly as without route
        # planning. On resuming (evasion hysteresis just cleared), the
        # route is REJOINED from the current position (skipping any
        # effectively-passed waypoints) or replanned if no longer safe --
        # never forced back to an obsolete waypoint behind the attacker.
        self._enemy_route_last_result = None if not route_suspended else self._enemy_route_last_result
        self._enemy_route_suspended = route_suspended and self.config.enemy_route_planning_enabled
        prev_route_replan_count = (
            self._enemy_route_planner._replan_count if self.config.enemy_route_planning_enabled else 0
        )
        self._enemy_route_command = None
        if self.config.enemy_obstacle_avoidance_enabled and self.config.enemy_route_planning_enabled and perceived is not None and not route_suspended:
            if direct_guidance and self._enemy_route_planner.direct_path_clear(
                self._enemy_pos, self._soldier_pos, perceived,
            ):
                # Corrected A0: the direct line to the asset is feasible, so
                # no route is followed and any previous bypass is discarded.
                self._enemy_route_planner.invalidate_route()
                route_result = None
            elif just_resumed:
                route_result = self._enemy_route_planner.resume_or_rejoin(
                    self._enemy_pos, self._soldier_pos, perceived, self.config.enemy_obstacle_awareness_mode,
                )
            else:
                route_result = self._enemy_route_planner.plan(
                    self._enemy_pos, self._soldier_pos, perceived, self.config.enemy_obstacle_awareness_mode,
                )
            if route_result is not None:
                self._enemy_route_last_result = route_result
                route_direction = np.asarray(route_result["direction"], dtype=np.float64)
                self._enemy_route_command = route_direction.astype(np.float32)
                if np.linalg.norm(route_direction) > eps:
                    r_hat = route_direction
        
        # Horizontal lateral vector (perpendicular to the horizontal
        # projection of the pursuit direction, 90° CCW rotation in XY)
        lateral_raw = np.array([-r_hat[1], r_hat[0], 0.0])
        lateral_norm = np.linalg.norm(lateral_raw)
        if lateral_norm > eps:
            lateral = lateral_raw / lateral_norm
        else:
            # Pursuit direction is (near-)vertical: horizontal weave axis is
            # degenerate. Fall back to a fixed horizontal direction rather
            # than dividing by zero.
            lateral = np.array([1.0, 0.0, 0.0])
        
        # Update weave bias with AR(1) process: a <- rho * a + sigma_a * eta
        # (legacy only; corrected A0 draws no randomness and has no weave).
        if direct_guidance:
            self._weave_bias = 0.0
            u_base = r_hat / np.linalg.norm(r_hat)
        else:
            eta = self._rng_enemy_motion.normal(0.0, 1.0)
            self._weave_bias = self.config.rho * self._weave_bias + self.config.sigma_a * eta

            # Heading noise, generalized to 3 dimensions
            z = self._rng_enemy_motion.normal(0.0, 1.0, size=(3,))

            # Base command: exactly the pre-obstacle hostile model (pursuit +
            # weave + heading noise + evasion), unnormalized then normalized.
            weave_component = self._weave_bias * self.config.weave_amplitude * lateral
            u_base_raw = r_hat + weave_component + self.config.sigma_e * z + evasion_info["evasion_component"]
            u_base_norm = np.linalg.norm(u_base_raw)
            if u_base_norm > eps:
                u_base = u_base_raw / u_base_norm
            else:
                u_base = r_hat  # Fallback to direct pursuit

        # Learned-attacker tactical override (Adaptive Attacker v1, Phase
        # 3/8, journal extension): when set, REPLACES the entire
        # hand-designed tactical command computed above with an
        # externally-supplied desired direction. The diagnostic fields
        # already captured above (_enemy_mission_command,
        # _enemy_evasion_command, _enemy_route_command, etc.) still reflect
        # what the hand-designed model WOULD have flown this step -- useful
        # for scripted-vs-learned command comparison (Phase 23) at zero
        # extra risk to the existing, unchanged code path above. Obstacle
        # safety/dynamics/collision below are UNCHANGED and always apply to
        # whichever u_base is in effect. Falls back to the hand-designed
        # u_base (rather than producing NaN/degenerate motion) if the
        # override returns a near-zero vector.
        if self.enemy_tactical_override is not None:
            override_raw = np.asarray(self.enemy_tactical_override(), dtype=np.float64)
            override_norm = np.linalg.norm(override_raw)
            if override_norm > eps:
                u_base = override_raw / override_norm

        # Deterministic, RNG-free obstacle navigation (journal extension,
        # Phase 3B). Operates on u_base (the command the hostile would
        # actually execute if obstacles did not exist) plus current
        # position/velocity -- NOT merely the instantaneous pursuit ray.
        # Exact no-op (u = u_base, bit-for-bit) when disabled or when no
        # obstacles exist -- see ObstacleAvoidancePlanner.plan().
        if self.config.enemy_obstacle_avoidance_enabled:
            # perceived was already computed once, above (shared with the
            # route planner) -- never recomputed here.
            perceived_static_layout = perceived.to_static_layout()
            avoidance_layout = self._combined_avoidance_layout(
                self._enemy_pos, self._enemy_vel,
                self.config.enemy_obstacle_prediction_horizon_steps, self.config.enemy_obstacle_clearance,
                perceived_static_layout, perceived.dynamic,
            )
            # Defensive guard: the dynamic portion of `avoidance_layout` is
            # rebuilt every step from only the CURRENTLY-threatening
            # dynamic obstacles (see `_combined_avoidance_layout`), so a
            # persisted `obstacle_index` from a prior step can point past
            # the end of (or, rarely, to a different obstacle within) this
            # step's layout once the threatening set changes. Resetting
            # here (re-activating fresh against the current layout, same
            # step) avoids an IndexError; static-obstacle persistence is
            # unaffected since the static portion's indices never change.
            if (
                self._enemy_obstacle_planner.active
                and (
                    self._enemy_obstacle_planner.obstacle_index is None
                    or self._enemy_obstacle_planner.obstacle_index >= len(avoidance_layout)
                )
            ):
                self._enemy_obstacle_planner.reset()
            plan = self._enemy_obstacle_planner.plan(
                position=self._enemy_pos,
                velocity=self._enemy_vel,
                u_base=u_base,
                layout=avoidance_layout,
                target_position=self._soldier_pos,
                limits=self._enemy_dynamics_limits(),
            )
        else:
            plan = {
                "direction": u_base, "active": False, "mode": None, "obstacle_index": None,
                "predicted_collision_step": None, "predicted_min_clearance": float("inf"),
                "fallback_used": False,
            }
        self._enemy_obstacle_avoidance_active = plan["active"]
        self._enemy_obstacle_avoidance_mode = plan["mode"]
        self._enemy_obstacle_avoidance_vector = np.asarray(plan["direction"], dtype=np.float32)
        self._enemy_reactive_avoidance_command = (
            np.asarray(plan["direction"], dtype=np.float32) if plan["active"] else None
        )
        self._enemy_predicted_obstacle_index = plan["obstacle_index"]
        self._enemy_predicted_collision_step = plan["predicted_collision_step"]
        self._enemy_predicted_min_clearance = plan["predicted_min_clearance"]
        self._enemy_obstacle_fallback_used = plan["fallback_used"]

        # Distinguish a static-AABB avoidance decision from a
        # dynamic-obstacle one: indices >= len(perceived static layout)
        # refer to the ephemeral swept-AABBs appended by
        # _combined_avoidance_layout(). Falls back to the ground-truth
        # static layout length when avoidance is disabled (no perception
        # update ran this step, so n_static is irrelevant either way).
        n_static = len(self._enemy_last_perceived.static) if self.config.enemy_obstacle_avoidance_enabled else len(self._obstacle_layout)
        if plan["obstacle_index"] is None:
            self._enemy_avoidance_obstacle_type = None
            self._enemy_avoidance_obstacle_id = None
        elif plan["obstacle_index"] < n_static:
            self._enemy_avoidance_obstacle_type = "static"
            self._enemy_avoidance_obstacle_id = (
                self._enemy_last_perceived.static[plan["obstacle_index"]].id
                if self.config.enemy_obstacle_avoidance_enabled else plan["obstacle_index"]
            )
        else:
            self._enemy_avoidance_obstacle_type = self._enemy_last_perceived.dynamic[plan["obstacle_index"] - n_static].obstacle_type
            self._enemy_avoidance_obstacle_id = self._enemy_last_perceived.dynamic[plan["obstacle_index"] - n_static].id
        cos_angle = float(np.clip(np.dot(self._enemy_obstacle_avoidance_vector.astype(np.float64), u_base), -1.0, 1.0))
        self._enemy_avoidance_angle_deg = float(np.degrees(np.arccos(cos_angle)))
        self._enemy_predicted_time_to_conflict = (
            plan["predicted_collision_step"] * self.config.dt if plan["predicted_collision_step"] is not None else None
        )

        u = np.asarray(plan["direction"], dtype=np.float64)
        self._enemy_final_command = u.astype(np.float32)

        # Command-conflict diagnostics (Phase 3): angular disagreement
        # between every pair of per-stage commands actually computed this
        # step (angles in degrees; None where a stage did not fire).
        def _angle_deg(a, b):
            if a is None or b is None:
                return None
            a_n, b_n = np.linalg.norm(a), np.linalg.norm(b)
            if a_n <= eps or b_n <= eps:
                return None
            cos_a = float(np.clip(np.dot(a, b) / (a_n * b_n), -1.0, 1.0))
            return float(np.degrees(np.arccos(cos_a)))

        self._enemy_angle_mission_evasion = _angle_deg(self._enemy_mission_command, self._enemy_evasion_command)
        self._enemy_angle_mission_route = _angle_deg(self._enemy_mission_command, self._enemy_route_command)
        self._enemy_angle_evasion_route = _angle_deg(self._enemy_evasion_command, self._enemy_route_command)
        self._enemy_angle_route_final = _angle_deg(self._enemy_route_command, self._enemy_final_command)
        self._enemy_angle_evasion_final = _angle_deg(self._enemy_evasion_command, self._enemy_final_command)

        # Evasion activation transition + streak tracking.
        self._enemy_evasion_active = evasion_active_this_step
        if evasion_active_this_step:
            if self._enemy_evasion_streak_steps == 0:
                self._enemy_evasion_streak_start_pos = self._enemy_pos.copy()
                self._enemy_evasion_streak_start_goal_dist = float(np.linalg.norm(self._soldier_pos - self._enemy_pos))
            self._enemy_evasion_streak_steps += 1
            if self._enemy_obstacle_avoidance_active:
                self._enemy_obstacle_conflicts_during_evasion += 1
        else:
            if self._enemy_evasion_streak_steps > 0 and self.config.enemy_route_planning_enabled:
                # Evasion just ended: did the SAME route resume (no replan
                # forced this step) or did it require a fresh replan?
                if self._enemy_route_last_result is not None and self._enemy_route_last_result["route_replan_count"] == prev_route_replan_count:
                    self._enemy_route_resumptions_after_evasion += 1
            self._enemy_evasion_streak_steps = 0
            self._enemy_evasion_streak_start_pos = None
            self._enemy_evasion_streak_start_goal_dist = None

        # Route replans / waypoint invalidations attributable to evasion.
        if self._enemy_route_last_result is not None and self._enemy_route_last_result["route_replan_count"] > prev_route_replan_count:
            if evasion_active_this_step:
                self._enemy_replans_while_evading += 1
                if self._enemy_route_last_result["route_replan_reason"] in ("route_unsafe_or_new_obstacle", "progress_stall"):
                    self._enemy_waypoint_invalidations_from_evasion += 1

        # Behavior mode (Phase 5-7): highest-priority ACTIVE stage this
        # step, in the enforced arbitration priority order collision-avoid
        # > evade > route > mission (route is suspended, not merely
        # deprioritized, while evasion is active or within its hysteresis
        # hold -- see route_suspended above).
        if self._enemy_obstacle_avoidance_active:
            new_mode = "COLLISION_AVOID"
        elif evasion_active_this_step:
            new_mode = "EVADE"
        elif not route_suspended and self._enemy_route_last_result is not None:
            new_mode = "ROUTE"
        else:
            new_mode = "MISSION"
        if new_mode != self._enemy_behavior_mode:
            self._enemy_behavior_mode_switch_count += 1
        self._enemy_behavior_mode = new_mode

        # Guidance produces a DESIRED velocity; dynamics determine the
        # actual persistent velocity subject to acceleration/turn/vertical limits.
        desired_velocity = self.config.v_e * u
        
        next_vel, diag = self._advance_velocity(
            current_velocity=self._enemy_vel,
            desired_velocity=desired_velocity,
            max_speed=self.config.v_e,
            max_accel=self.config.enemy_max_accel,
            max_turn_rate_rad=np.radians(self.config.enemy_max_turn_rate_deg),
            max_climb_rate=self.config.enemy_max_climb_rate,
            max_descent_rate=self.config.enemy_max_descent_rate,
        )
        self._enemy_dynamics_info = diag
        
        new_pos = self._enemy_pos + next_vel * self.config.dt
        new_pos, next_vel = self._apply_boundary(new_pos, next_vel)
        
        self._enemy_pos = new_pos.astype(np.float32)
        self._enemy_vel = next_vel.astype(np.float32)

    def _enemy_dynamics_limits(self) -> DynamicsLimits:
        """Bundles the hostile's constrained-dynamics parameters for reuse
        by the obstacle-navigation predictor (uav_defend/obstacles/avoidance.py),
        guaranteeing the predictor's dynamics model can never drift out of
        sync with the real per-step update above."""
        return DynamicsLimits(
            max_speed=self.config.v_e,
            max_accel=self.config.enemy_max_accel,
            max_turn_rate_rad=np.radians(self.config.enemy_max_turn_rate_deg),
            max_climb_rate=self.config.enemy_max_climb_rate,
            max_descent_rate=self.config.enemy_max_descent_rate,
            dt=self.config.dt,
            eps=self.config.eps,
            L=self.config.L,
            max_altitude=self.config.max_altitude,
        )
    
    def _reflect_boundary(self, pos: np.ndarray) -> np.ndarray:
        """
        Apply reflecting boundary conditions within the 3D engagement volume:
        x, y ∈ [-L, L] and z ∈ [0, H] where H = max_altitude.
        
        If position exceeds a boundary, reflect it back into the domain.
        This is applied independently per axis, with the ground (z=0) and
        ceiling (z=H) acting as reflecting boundaries for altitude, exactly
        as the horizontal walls do for x and y.
        
        Args:
            pos: Position array of shape (3,).
        
        Returns:
            Reflected position within [-L, L]² × [0, H].
        """
        L = self.config.L
        H = self.config.max_altitude
        lo = (-L, -L, 0.0)
        hi = (L, L, H)
        
        for i in range(3):
            # Reflect until within bounds (handles multiple reflections)
            while pos[i] < lo[i] or pos[i] > hi[i]:
                if pos[i] < lo[i]:
                    pos[i] = 2 * lo[i] - pos[i]  # Reflect off lower boundary
                if pos[i] > hi[i]:
                    pos[i] = 2 * hi[i] - pos[i]  # Reflect off upper boundary
        
        return pos
    
    def _normalize_pos(self, pos: np.ndarray) -> np.ndarray:
        """
        Normalize a 3D position vector to a dimensionally-appropriate scale.
        
        Horizontal coordinates (x, y) are normalized by L (placing them in
        [-1, 1]); altitude (z) is normalized by max_altitude (placing it in
        [0, 1], a valid subset of the overall [-1, 1] Box).
        
        Args:
            pos: Position array of shape (3,).
        
        Returns:
            Normalized position array of shape (3,).
        """
        L = self.config.L
        H = self.config.max_altitude
        return np.array([pos[0] / L, pos[1] / L, pos[2] / H], dtype=np.float32)
    
    def _get_obs(self) -> np.ndarray:
        """
        Get normalized observation for RL with partial observability.
        
        The common layout is identical for both estimator tracks -- Direct
        and Kalman observations differ ONLY in how the hostile
        position/velocity fields are estimated, never in semantics:
        
        Returns:
            obs: Array of shape (16,), all values normalized to [-1, 1].
            
            [soldier(3), defender(3), defender_vel(3), detected_flag(1),
             hostile_position_info(3), hostile_velocity_info(3)]
            
            - defender_vel: defender's persistent 3D velocity normalized by
              v_d (component-wise); always within [-1, 1] since total
              defender speed is dynamically constrained to <= v_d.
            
            If config.use_kalman_tracking=True (Kalman track):
                hostile_position_info = e_hat (Kalman estimated position,
                    x,y normalized by L, z normalized by max_altitude)
                hostile_velocity_info = v_hat (Kalman estimated velocity,
                    normalized by v_e)
                
            If config.use_kalman_tracking=False (Direct/measurement track,
            the environment default):
                hostile_position_info = raw noisy hostile-position measurement
                    (x,y normalized by L, z normalized by max_altitude)
                hostile_velocity_info = finite-difference measurement
                    velocity (normalized by v_e; zeros until a second
                    measurement makes it valid, see
                    _enemy_measurement_velocity_valid)
            
            Both tracks: all hostile-related values are 0.0 before detection.
            Policies act on ESTIMATED/MEASURED state, never ground truth.
        """
        v_e = self.config.v_e
        v_d = self.config.v_d
        
        # Normalize positions
        soldier_norm = self._normalize_pos(self._soldier_pos)
        defender_norm = self._normalize_pos(self._defender_pos)
        
        # Normalize defender's own persistent velocity by its max speed.
        # Guard against a misconfigured (non-positive) v_d.
        if v_d > 0:
            defender_vel_norm = np.clip(self._defender_vel / v_d, -1.0, 1.0).astype(np.float32)
        else:
            defender_vel_norm = np.zeros(3, dtype=np.float32)
        
        # Detection flag
        detected_flag = np.array([1.0 if self._enemy_detected else 0.0])
        
        if self.config.use_kalman_tracking:
            # KALMAN TRACK: Use Kalman filter estimates
            if self._enemy_detected and self._e_hat is not None:
                hostile_pos_norm = np.clip(self._normalize_pos(self._e_hat), -1.0, 1.0)
                hostile_vel_norm = np.clip(self._v_hat / v_e, -1.0, 1.0)
            else:
                hostile_pos_norm = np.zeros(3, dtype=np.float32)
                hostile_vel_norm = np.zeros(3, dtype=np.float32)
        else:
            # DIRECT/MEASUREMENT TRACK: raw measurement + finite-difference velocity
            if self._enemy_detected and self._enemy_measurement is not None:
                hostile_pos_norm = np.clip(self._normalize_pos(self._enemy_measurement), -1.0, 1.0)
                hostile_vel_norm = np.clip(self._enemy_measurement_velocity / v_e, -1.0, 1.0)
            else:
                hostile_pos_norm = np.zeros(3, dtype=np.float32)
                hostile_vel_norm = np.zeros(3, dtype=np.float32)
        
        return np.concatenate([
            soldier_norm,
            defender_norm,
            defender_vel_norm,
            detected_flag,
            hostile_pos_norm,
            hostile_vel_norm,
        ]).astype(np.float32)
    
    def _get_info(self, outcome: str = "ongoing", enemy_soldier_dist: float = 0.0, 
                  defender_enemy_dist: float = 0.0, defender_standby: bool = False,
                  controller_action_executed: bool = False, just_detected: bool = False) -> dict:
        """Get additional info dict.
        
        Args:
            outcome: Episode outcome - "ongoing", "intercepted", "soldier_caught", 
                    "unsafe_intercept", or "timeout".
            enemy_soldier_dist: Distance between enemy and soldier.
            defender_enemy_dist: Distance between defender and enemy.
            defender_standby: True if THIS transition used pre-detection
                standby behavior (defender held co-located with the
                soldier, supplied action discarded) -- see step().
            controller_action_executed: True iff the supplied action was
                actually applied to the defender's constrained dynamics
                this transition (always True in legacy/non-standby mode).
            just_detected: True only on the transition where
                self._enemy_detected changed False -> True this step.
        
        Dynamics diagnostic fields (units: velocity in m/s, acceleration in
        m/s^2, turn rate in deg/s) are computed by _advance_velocity() during
        _move_defender() / _move_enemy() and are NOT part of the RL
        observation (only defender_vel is observable, per the observation
        layout above).
        
        Hostile-evasion diagnostic fields (enemy_evasion_*, enemy_pursuit_direction)
        are computed by _compute_enemy_evasion() during _move_enemy() and are
        likewise NOT part of the RL observation -- the defender must respond
        to evasion only through its normal sensor/estimation observations.
        """
        # Check if current state is in unsafe intercept zone
        unsafe_zone = enemy_soldier_dist <= self.config.unsafe_intercept_radius
        
        # Compute tracking error whenever an estimated/measured enemy position is available.
        if self._enemy_detected and self._e_hat is not None:
            tracking_error = float(np.linalg.norm(self._enemy_pos - self._e_hat))
        else:
            tracking_error = None
        
        return {
            "step_count": self._step_count,
            "soldier_pos": self._soldier_pos.copy(),
            "defender_pos": self._defender_pos.copy(),
            "enemy_pos": self._enemy_pos.copy(),
            "defender_vel": self._defender_vel.copy(),  # m/s, shape (3,)
            "enemy_vel": self._enemy_vel.copy(),        # m/s, shape (3,)
            "defender_speed": float(np.linalg.norm(self._defender_vel)),  # m/s
            "enemy_speed": float(np.linalg.norm(self._enemy_vel)),        # m/s
            "defender_accel": self._defender_dynamics_info["accel_used"],  # m/s^2
            "enemy_accel": self._enemy_dynamics_info["accel_used"],        # m/s^2
            "defender_turn_rate": self._defender_dynamics_info["turn_rate_used_deg"],  # deg/s
            "enemy_turn_rate": self._enemy_dynamics_info["turn_rate_used_deg"],        # deg/s
            "defender_accel_saturated": self._defender_dynamics_info["accel_saturated"],
            "enemy_accel_saturated": self._enemy_dynamics_info["accel_saturated"],
            "defender_turn_saturated": self._defender_dynamics_info["turn_saturated"],
            "enemy_turn_saturated": self._enemy_dynamics_info["turn_saturated"],
            "defender_climb_saturated": self._defender_dynamics_info["climb_saturated"],
            "enemy_climb_saturated": self._enemy_dynamics_info["climb_saturated"],
            # Hostile-evasion diagnostics (NOT part of the RL observation)
            "enemy_evasion_active": bool(
                self.config.enemy_evasion_enabled
                and self._enemy_evasion_info.get("activation", 0.0) > 0.0
                and self.config.enemy_evasion_gain > 0.0
            ),
            "enemy_evasion_activation": self._enemy_evasion_info.get("activation", 0.0),
            "enemy_evasion_gain": self.config.enemy_evasion_gain,
            "enemy_evasion_weight": self._enemy_evasion_info.get("effective_weight", 0.0),
            "enemy_evasion_vector": self._enemy_evasion_info.get(
                "evasion_component", np.zeros(3, dtype=np.float32)
            ).copy(),
            "enemy_pursuit_direction": (
                self._enemy_pursuit_direction.copy()
                if self._enemy_pursuit_direction is not None
                else np.zeros(3, dtype=np.float32)
            ),
            "weave_bias": self._weave_bias,
            "enemy_soldier_dist": enemy_soldier_dist,
            "defender_enemy_dist": defender_enemy_dist,
            "enemy_detected": self._enemy_detected,  # Use stored tracking state
            "unsafe_intercept": unsafe_zone,  # True if enemy is in unsafe zone around soldier
            "outcome": outcome,
            # Kalman filter tracking info
            "detected": self._enemy_detected,
            "enemy_measurement": (
                self._enemy_measurement.copy() if self._enemy_measurement is not None else None
            ),
            # Standardized Direct-track finite-difference measurement velocity.
            # Legitimate sensor-derived information (NOT ground truth).
            "enemy_measurement_velocity": self._enemy_measurement_velocity.copy(),
            "enemy_measurement_velocity_valid": self._enemy_measurement_velocity_valid,
            "e_hat": self._e_hat.copy() if self._e_hat is not None else None,
            "v_hat": self._v_hat.copy() if self._v_hat is not None else None,
            "tracking_error": tracking_error,
            # Pre-detection standby diagnostics (see step() docstring).
            "defender_standby": defender_standby,
            "controller_action_executed": controller_action_executed,
            "just_detected": just_detected,
            # Obstacle diagnostics (journal extension; NOT part of the RL
            # observation in this phase -- see uav_defend/obstacles/).
            "obstacles_enabled": self.config.obstacles_enabled,
            "obstacle_count": len(self._obstacle_layout),
            "obstacle_layout": self._obstacle_layout.to_dict(),
            # Obstacle collision/clearance diagnostics (Phase 2). DIAGNOSTIC
            # ONLY -- does not affect reward or termination in this phase.
            # Clearance is +inf when there are no obstacles (chosen
            # consistently over None to simplify minimum-distance math).
            "defender_obstacle_collision": self._defender_obstacle_collision,
            "enemy_obstacle_collision": self._enemy_obstacle_collision,
            "defender_collision_obstacle_index": self._defender_collision_obstacle_index,
            "enemy_collision_obstacle_index": self._enemy_collision_obstacle_index,
            "defender_collision_point": (
                self._defender_collision_point.copy() if self._defender_collision_point is not None else None
            ),
            "enemy_collision_point": (
                self._enemy_collision_point.copy() if self._enemy_collision_point is not None else None
            ),
            "defender_min_obstacle_clearance": self._defender_min_obstacle_clearance,
            "enemy_min_obstacle_clearance": self._enemy_min_obstacle_clearance,
            # Collision EVENT counts (outside -> collision transitions only;
            # see step() for the exact definition), not per-step occupancy.
            "defender_collision_count": self._defender_collision_count,
            "enemy_collision_count": self._enemy_collision_count,
            "defender_ever_collided": self._defender_ever_collided,
            "enemy_ever_collided": self._enemy_ever_collided,
            # Protected-asset obstacle-blocked-motion diagnostics (Phase 3).
            "soldier_obstacle_blocked": self._soldier_obstacle_blocked,
            "soldier_obstacle_block_count": self._soldier_obstacle_block_count,
            # Hostile obstacle-navigation diagnostics (Phase 3B). Deterministic,
            # RNG-free; NOT part of the policy observation.
            "enemy_obstacle_avoidance_active": self._enemy_obstacle_avoidance_active,
            "enemy_obstacle_avoidance_mode": self._enemy_obstacle_avoidance_mode,
            "enemy_obstacle_avoidance_vector": self._enemy_obstacle_avoidance_vector.copy(),
            "enemy_predicted_obstacle_index": self._enemy_predicted_obstacle_index,
            "enemy_predicted_collision_step": self._enemy_predicted_collision_step,
            "enemy_predicted_min_clearance": self._enemy_predicted_min_clearance,
            "enemy_obstacle_fallback_used": self._enemy_obstacle_fallback_used,
            # Dynamic-obstacle diagnostics (journal extension). DIAGNOSTIC
            # ONLY -- same never-blocking/never-terminal convention as the
            # static case; NOT part of the RL observation.
            "dynamic_obstacles_enabled": self.config.dynamic_obstacles_enabled,
            "dynamic_obstacle_count": len(self._dynamic_obstacles),
            "enemy_avoidance_active": self._enemy_obstacle_avoidance_active,
            "enemy_avoidance_obstacle_id": self._enemy_avoidance_obstacle_id,
            "enemy_avoidance_obstacle_type": self._enemy_avoidance_obstacle_type,
            "enemy_avoidance_angle_deg": self._enemy_avoidance_angle_deg,
            "enemy_predicted_time_to_conflict": self._enemy_predicted_time_to_conflict,
            "enemy_actual_min_clearance": min(self._enemy_min_obstacle_clearance, self._enemy_min_dynamic_obstacle_clearance),
            "enemy_min_dynamic_obstacle_clearance": self._enemy_min_dynamic_obstacle_clearance,
            "defender_min_dynamic_obstacle_clearance": self._defender_min_dynamic_obstacle_clearance,
            "enemy_dynamic_obstacle_collision": self._enemy_dynamic_obstacle_collision,
            "defender_dynamic_obstacle_collision": self._defender_dynamic_obstacle_collision,
            "enemy_collision_obstacle_id": self._enemy_collision_obstacle_id,
            "defender_collision_obstacle_id": self._defender_collision_obstacle_id,
            "enemy_dynamic_collision_count": self._enemy_dynamic_collision_count,
            "defender_dynamic_collision_count": self._defender_dynamic_collision_count,
            "enemy_dynamic_ever_collided": self._enemy_dynamic_ever_collided,
            "defender_dynamic_ever_collided": self._defender_dynamic_ever_collided,
            # Physical-collision-event diagnostics (journal extension,
            # Phase 3/4): the entity's FIRST contact THIS STEP (None if no
            # physical collision occurred this step -- see step() for the
            # static-vs-dynamic first-event-wins precedence). Distinct from
            # the ever/count fields above, which persist across the whole
            # episode.
            "defender_collision_event_type": self._defender_collision_event_type,
            "defender_collision_event_point": (
                self._defender_collision_event_point.copy()
                if self._defender_collision_event_point is not None else None
            ),
            "defender_collision_event_speed": self._defender_collision_event_speed,
            "defender_collision_event_step": (
                self._step_count if self._defender_collision_event_type is not None else None
            ),
            "enemy_collision_event_type": self._enemy_collision_event_type,
            "enemy_collision_event_point": (
                self._enemy_collision_event_point.copy()
                if self._enemy_collision_event_point is not None else None
            ),
            "enemy_collision_event_speed": self._enemy_collision_event_speed,
            "enemy_collision_event_step": (
                self._step_count if self._enemy_collision_event_type is not None else None
            ),
            # Attacker perception/awareness diagnostics (journal extension;
            # see uav_defend/obstacles/perception.py). NEVER exposed to any
            # defender controller/observation -- ground-truth collision
            # diagnostics above are unaffected by awareness mode.
            "enemy_awareness_mode": self.config.enemy_obstacle_awareness_mode,
            "enemy_num_mapped_obstacles_known": sum(
                1 for o in self._enemy_last_perceived.static if o.source == "map"
            ),
            "enemy_num_locally_sensed_static": sum(
                1 for o in self._enemy_last_perceived.static if o.source == "local_sensor"
            ),
            "enemy_num_locally_sensed_dynamic": len(self._enemy_last_perceived.dynamic),
            "enemy_total_perceived_obstacles": (
                len(self._enemy_last_perceived.static) + len(self._enemy_last_perceived.dynamic)
            ),
            "enemy_new_obstacle_detections": self._enemy_perception.new_static_detections,
            "enemy_perceived_static_ids": tuple(o.id for o in self._enemy_last_perceived.static),
            "enemy_perceived_dynamic_ids": tuple(o.id for o in self._enemy_last_perceived.dynamic),
            # Defender perception/awareness + obstacle-safety diagnostics
            # (journal extension; see uav_defend/obstacles/perception.py,
            # uav_defend/obstacles/avoidance.py). NEVER exposed to any
            # controller/observation -- purely diagnostic. The hostile UAV
            # (mission target) is NEVER counted here (Phase 14).
            "defender_awareness_mode": self.config.defender_obstacle_awareness_mode,
            "defender_num_mapped_obstacles_known": sum(
                1 for o in self._defender_last_perceived.static if o.source == "map"
            ),
            "defender_num_locally_sensed_static": sum(
                1 for o in self._defender_last_perceived.static if o.source == "local_sensor"
            ),
            "defender_num_locally_sensed_dynamic": len(self._defender_last_perceived.dynamic),
            "defender_total_perceived_obstacles": (
                len(self._defender_last_perceived.static) + len(self._defender_last_perceived.dynamic)
            ),
            "defender_avoidance_active": self._defender_obstacle_avoidance_active,
            "defender_avoidance_obstacle_id": self._defender_avoidance_obstacle_id,
            "defender_avoidance_obstacle_type": self._defender_avoidance_obstacle_type,
            "defender_avoidance_angle_deg": self._defender_avoidance_angle_deg,
            "defender_desired_command": self._defender_desired_command.copy(),
            "defender_safe_command": self._defender_safe_command.copy(),
            "defender_predicted_min_clearance": self._defender_predicted_min_clearance,
            "defender_actual_min_clearance": min(
                self._obstacle_layout.min_clearance(self._defender_pos), self._defender_min_dynamic_obstacle_clearance
            ),
            # Multi-obstacle route-planner diagnostics (journal extension;
            # see uav_defend/obstacles/route_planner.py). None of these
            # affect physics/collision -- purely the mission-command source
            # fed into the (unchanged) reactive avoidance layer.
            "enemy_route_active": bool(self._enemy_route_last_result is not None),
            "enemy_route_waypoint_count": (
                self._enemy_route_last_result["route_waypoint_count"] if self._enemy_route_last_result else 0
            ),
            "enemy_route_current_waypoint": (
                self._enemy_route_last_result["route_current_waypoint"] if self._enemy_route_last_result else 0
            ),
            "enemy_route_replan_count": (
                self._enemy_route_last_result["route_replan_count"] if self._enemy_route_last_result else 0
            ),
            "enemy_route_replan_reason": (
                self._enemy_route_last_result["route_replan_reason"] if self._enemy_route_last_result else None
            ),
            "enemy_route_obstacle_ids": (
                self._enemy_route_last_result["route_obstacle_ids"] if self._enemy_route_last_result else ()
            ),
            "enemy_route_estimated_length": (
                self._enemy_route_last_result["route_estimated_length"] if self._enemy_route_last_result else 0.0
            ),
            "enemy_progress_stall_active": (
                self._enemy_route_last_result["progress_stall_active"] if self._enemy_route_last_result else False
            ),
            "enemy_route_revisit": (
                self._enemy_route_last_result["route_revisit"] if self._enemy_route_last_result else False
            ),
            "enemy_vertical_bypass_count": (
                self._enemy_route_last_result["vertical_bypass_count"] if self._enemy_route_last_result else 0
            ),
            # Command-conflict diagnostics (Phase 3, behavior-arbitration
            # audit). Purely observational.
            "enemy_mission_command": self._enemy_mission_command,
            "enemy_evasion_command": self._enemy_evasion_command,
            "enemy_route_command": self._enemy_route_command,
            "enemy_reactive_avoidance_command": self._enemy_reactive_avoidance_command,
            "enemy_final_command": self._enemy_final_command,
            "enemy_evasion_active": self._enemy_evasion_active,
            "enemy_angle_mission_evasion_deg": self._enemy_angle_mission_evasion,
            "enemy_angle_mission_route_deg": self._enemy_angle_mission_route,
            "enemy_angle_evasion_route_deg": self._enemy_angle_evasion_route,
            "enemy_angle_route_final_deg": self._enemy_angle_route_final,
            "enemy_angle_evasion_final_deg": self._enemy_angle_evasion_final,
            "enemy_replans_while_evading": self._enemy_replans_while_evading,
            "enemy_waypoint_invalidations_from_evasion": self._enemy_waypoint_invalidations_from_evasion,
            "enemy_route_resumptions_after_evasion": self._enemy_route_resumptions_after_evasion,
            "enemy_evasion_streak_steps": self._enemy_evasion_streak_steps,
            "enemy_evasion_distance_since_streak_start": (
                float(np.linalg.norm(self._enemy_pos - self._enemy_evasion_streak_start_pos))
                if self._enemy_evasion_streak_start_pos is not None else 0.0
            ),
            "enemy_evasion_net_progress_since_streak_start": (
                self._enemy_evasion_streak_start_goal_dist - float(np.linalg.norm(self._soldier_pos - self._enemy_pos))
                if self._enemy_evasion_streak_start_goal_dist is not None else 0.0
            ),
            "enemy_obstacle_conflicts_during_evasion": self._enemy_obstacle_conflicts_during_evasion,
            "enemy_behavior_mode": self._enemy_behavior_mode,
            "enemy_behavior_mode_switch_count": self._enemy_behavior_mode_switch_count,
            "enemy_route_suspended": self._enemy_route_suspended,
            "enemy_nominal_guidance_mode": self.config.enemy_nominal_guidance_mode,
            **self._contact_info,
            **self._attacker_efficiency_info(),
        }
    
    def render(self) -> np.ndarray | None:
        """Render the environment (placeholder for future implementation)."""
        if self.render_mode == "rgb_array":
            # Return a simple representation (to be expanded later)
            return self._render_frame()
        return None
    
    def _render_frame(self) -> np.ndarray:
        """Render a frame as an RGB array."""
        # Simple placeholder: 100x100 black image with entities
        size = 100
        frame = np.zeros((size, size, 3), dtype=np.uint8)
        L = self.config.L

        # Draw obstacle footprints (journal extension, Phase 2): this
        # renderer is fundamentally 2-D/top-down, so obstacles are drawn as
        # filled x/y-footprint rectangles UNDER the entity markers below.
        # Shade encodes height (taller obstacle -> darker gray) as a simple
        # height annotation without adding a 3-D rendering dependency.
        for obstacle in self._obstacle_layout:
            x_min, x_max = np.clip(
                np.round([(obstacle.min_corner[0] + L) / (2 * L) * (size - 1),
                          (obstacle.max_corner[0] + L) / (2 * L) * (size - 1)]).astype(int),
                0, size - 1,
            )
            y_min, y_max = np.clip(
                np.round([(obstacle.min_corner[1] + L) / (2 * L) * (size - 1),
                          (obstacle.max_corner[1] + L) / (2 * L) * (size - 1)]).astype(int),
                0, size - 1,
            )
            height_frac = float(np.clip(obstacle.max_corner[2] / max(self.config.max_altitude, self.config.eps), 0.0, 1.0))
            shade = int(round(150 - 90 * height_frac))  # taller obstacle -> darker gray
            frame[y_min:y_max + 1, x_min:x_max + 1] = [shade, shade, shade]
        
        # Draw soldier as blue square
        if self._soldier_pos is not None:
            x = int((self._soldier_pos[0] + L) / (2 * L) * (size - 1))
            y = int((self._soldier_pos[1] + L) / (2 * L) * (size - 1))
            x = np.clip(x, 0, size - 1)
            y = np.clip(y, 0, size - 1)
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    px, py = x + dx, y + dy
                    if 0 <= px < size and 0 <= py < size:
                        frame[py, px] = [0, 0, 255]  # Blue
        
        # Draw defender as green square
        if self._defender_pos is not None:
            x = int((self._defender_pos[0] + L) / (2 * L) * (size - 1))
            y = int((self._defender_pos[1] + L) / (2 * L) * (size - 1))
            x = np.clip(x, 0, size - 1)
            y = np.clip(y, 0, size - 1)
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    px, py = x + dx, y + dy
                    if 0 <= px < size and 0 <= py < size:
                        frame[py, px] = [0, 255, 0]  # Green
        
        # Draw enemy as red square
        if self._enemy_pos is not None:
            x = int((self._enemy_pos[0] + L) / (2 * L) * (size - 1))
            y = int((self._enemy_pos[1] + L) / (2 * L) * (size - 1))
            x = np.clip(x, 0, size - 1)
            y = np.clip(y, 0, size - 1)
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    px, py = x + dx, y + dy
                    if 0 <= px < size and 0 <= py < size:
                        frame[py, px] = [255, 0, 0]  # Red
        
        return frame
    
    def close(self) -> None:
        """Clean up resources."""
        pass
