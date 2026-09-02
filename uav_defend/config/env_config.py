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

    def __post_init__(self) -> None:
        """Validate altitude and dynamics configuration."""
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
