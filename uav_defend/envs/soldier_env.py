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

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from uav_defend.config.env_config import EnvConfig
from uav_defend.tracking import EnemyKalmanFilter
from uav_defend.dynamics.constrained_point_mass import advance_velocity, apply_boundary
from uav_defend.obstacles import DynamicsLimits, ObstacleAvoidancePlanner, ObstacleLayout, generate_layout


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
        self._enemy_obstacle_planner = ObstacleAvoidancePlanner(
            clearance=self.config.enemy_obstacle_clearance,
            horizon_steps=self.config.enemy_obstacle_prediction_horizon_steps,
            release_steps=self.config.enemy_obstacle_release_steps,
            max_altitude=self.config.max_altitude,
            eps=self.config.eps,
        )
        # Hostile obstacle-navigation diagnostics (Phase 3B). Deterministic
        # and RNG-free -- see _move_enemy()/ObstacleAvoidancePlanner.plan().
        self._enemy_obstacle_avoidance_active: bool = False
        self._enemy_obstacle_avoidance_mode: str | None = None
        self._enemy_obstacle_avoidance_vector: np.ndarray = np.zeros(3, dtype=np.float32)
        self._enemy_predicted_obstacle_index: int | None = None
        self._enemy_predicted_collision_step: int | None = None
        self._enemy_predicted_min_clearance: float = float("inf")
        self._enemy_obstacle_fallback_used: bool = False
        
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
        
        # Initialize soldier at the origin, on the ground (z = 0)
        self._soldier_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        
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
        self._enemy_obstacle_avoidance_active = False
        self._enemy_obstacle_avoidance_mode = None
        self._enemy_obstacle_avoidance_vector = np.zeros(3, dtype=np.float32)
        self._enemy_predicted_obstacle_index = None
        self._enemy_predicted_collision_step = None
        self._enemy_predicted_min_clearance = self._obstacle_layout.min_clearance(self._enemy_pos)
        self._enemy_obstacle_fallback_used = False
        
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
        
        return self._get_obs(), self._get_info(
            "ongoing", initial_enemy_soldier_dist, initial_defender_enemy_dist,
            defender_standby=self.config.defender_standby_until_detection,
            controller_action_executed=False,
            just_detected=False,
        )
    
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
        # Phase 2). DIAGNOSTIC ONLY: does not stop movement, modify
        # position/velocity, terminate the episode, or alter reward -- see
        # module-level design notes in uav_defend/obstacles/.
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
        # obstacle across multiple steps (possible in Phase 2 since motion is
        # never blocked) is counted once, not once per step.
        if self._defender_obstacle_collision and not self._defender_was_in_collision:
            self._defender_collision_count += 1
            self._defender_ever_collided = True
        self._defender_was_in_collision = self._defender_obstacle_collision
        if self._enemy_obstacle_collision and not self._enemy_was_in_collision:
            self._enemy_collision_count += 1
            self._enemy_ever_collided = True
        self._enemy_was_in_collision = self._enemy_obstacle_collision

        self._step_count += 1
        
        # Calculate distances
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
        
        # Episode terminates on any terminal condition or timeout
        terminated = intercepted or soldier_caught or unsafe_intercept or (self._step_count >= self.config.max_steps)
        truncated = False
        
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
            desired_direction = action / action_norm
            desired_velocity = self.config.v_d * desired_direction
        else:
            desired_velocity = np.zeros(3, dtype=np.float32)
        
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
        eta = self._rng_enemy_motion.normal(0.0, 1.0)
        self._weave_bias = self.config.rho * self._weave_bias + self.config.sigma_a * eta
        
        # Heading noise, generalized to 3 dimensions
        z = self._rng_enemy_motion.normal(0.0, 1.0, size=(3,))
        
        # Reactive evasion: intentionally independent of self._enemy_detected
        # (see _compute_enemy_evasion docstring).
        evasion_info = self._compute_enemy_evasion(self._defender_pos, self._enemy_pos)
        self._enemy_evasion_info = evasion_info

        # Base command: exactly the pre-obstacle hostile model (pursuit +
        # weave + heading noise + evasion), unnormalized then normalized.
        weave_component = self._weave_bias * self.config.weave_amplitude * lateral
        u_base_raw = r_hat + weave_component + self.config.sigma_e * z + evasion_info["evasion_component"]
        u_base_norm = np.linalg.norm(u_base_raw)
        if u_base_norm > eps:
            u_base = u_base_raw / u_base_norm
        else:
            u_base = r_hat  # Fallback to direct pursuit

        # Deterministic, RNG-free obstacle navigation (journal extension,
        # Phase 3B). Operates on u_base (the command the hostile would
        # actually execute if obstacles did not exist) plus current
        # position/velocity -- NOT merely the instantaneous pursuit ray.
        # Exact no-op (u = u_base, bit-for-bit) when disabled or when no
        # obstacles exist -- see ObstacleAvoidancePlanner.plan().
        if self.config.enemy_obstacle_avoidance_enabled:
            plan = self._enemy_obstacle_planner.plan(
                position=self._enemy_pos,
                velocity=self._enemy_vel,
                u_base=u_base,
                layout=self._obstacle_layout,
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
        self._enemy_predicted_obstacle_index = plan["obstacle_index"]
        self._enemy_predicted_collision_step = plan["predicted_collision_step"]
        self._enemy_predicted_min_clearance = plan["predicted_min_clearance"]
        self._enemy_obstacle_fallback_used = plan["fallback_used"]

        u = np.asarray(plan["direction"], dtype=np.float64)
        
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
