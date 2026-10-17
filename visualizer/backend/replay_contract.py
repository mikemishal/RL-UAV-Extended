"""Version 1 schema for a single authoritative native-3D replay."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, JsonValue, model_validator

Vec3 = tuple[FiniteFloat, FiniteFloat, FiniteFloat]
NonNegative = Annotated[FiniteFloat, Field(ge=0)]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Commit = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
Outcome = Literal[
    "intercepted", "soldier_caught", "unsafe_intercept", "timeout",
    "defender_static_obstacle_collision", "defender_dynamic_obstacle_collision",
    "enemy_static_obstacle_collision", "enemy_dynamic_obstacle_collision",
]


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Checkpoint(ContractModel):
    artifact_id: str = Field(min_length=1)
    research_relative_path: str = Field(min_length=1)
    sha256: Sha256
    role: Literal["d0_start", "final", "secondary_best", "conference_best", "attacker_best"]


class PolicySelection(ContractModel):
    policy_kind: Literal["analytical", "learned", "scripted"]
    policy_id: str
    architecture: str | None = None
    generation: Literal["conference", "d0", "d0_plus", "d1"] | None = None
    training_seed: int | None = None
    checkpoint: Checkpoint | None = None
    observation_shape: tuple[Annotated[int, Field(gt=0)], ...] | None = None
    action_shape: tuple[Literal[3]] = (3,)
    estimator: Literal["measurement", "kalman", "attacker_local", "scripted_ground_truth"]
    parameters: dict[str, JsonValue]


class EnvironmentSpec(ContractModel):
    clutter: Literal["C0", "C1", "C2", "C3"]
    awareness: Literal["local", "map_local"]
    scenario_kind: Literal["canonical"] = "canonical"
    builder: Literal["experiments.journal_clutter_protocol.build_canonical_env_config"]
    config: dict[str, JsonValue]
    config_sha256: Sha256
    dt_s: Annotated[FiniteFloat, Field(gt=0)]


class EntityState(ContractModel):
    position_m: Vec3
    velocity_mps: Vec3 | None = None
    speed_mps: NonNegative | None = None
    horizontal_heading_deg: FiniteFloat | None = None
    movement_direction: Vec3 | None = None


class StaticObstacle(ContractModel):
    obstacle_id: str
    min_corner_m: Vec3
    max_corner_m: Vec3
    map_known: bool
    visual_kind: Literal["block", "tree"] = "block"


class MovingObstacle(ContractModel):
    obstacle_id: str
    kind: Literal["bird", "airborne_clutter"]
    position_m: Vec3
    velocity_mps: Vec3
    radius_m: Annotated[FiniteFloat, Field(gt=0)]


class PolicyVisible(ContractModel):
    estimator: Literal["measurement", "kalman"]
    observation: list[FiniteFloat] = Field(min_length=16, max_length=16)
    soldier_pos_m: Vec3
    defender_pos_m: Vec3
    defender_vel_mps: Vec3
    enemy_detected: bool
    enemy_measurement_m: Vec3 | None = None
    enemy_measurement_velocity_mps: Vec3 | None = None
    enemy_measurement_velocity_valid: bool | None = None
    e_hat_m: Vec3 | None = None
    v_hat_mps: Vec3 | None = None
    sector_features: list[FiniteFloat] | None = Field(default=None, min_length=20, max_length=20)
    unavailable_fields: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def enforce_estimator_channels(self):
        if self.estimator == "measurement" and (self.e_hat_m is not None or self.v_hat_mps is not None):
            raise ValueError("Measurement policy-visible inputs must not contain Kalman-only fields.")
        if self.estimator == "kalman" and any(value is not None for value in (
            self.enemy_measurement_m, self.enemy_measurement_velocity_mps, self.enemy_measurement_velocity_valid
        )):
            raise ValueError("Kalman policy-visible inputs must not contain measurement-only fields.")
        return self


class LRPPOGuidance(ContractModel):
    lead_direction: Vec3
    residual_action: Vec3 | None = None
    residual_norm_raw: NonNegative | None = None
    residual_perpendicular: Vec3 | None = None
    residual_norm_clipped: bool | None = None
    native_adjustment_deg: Annotated[FiniteFloat, Field(ge=0, le=30.0001)] | None = None
    signed_horizontal_adjustment_deg: Annotated[FiniteFloat, Field(ge=-180, le=180)] | None = None
    max_adjustment_deg: Literal[30.0] = 30.0
    near_bound: bool | None = None
    near_bound_margin_deg: NonNegative | None = None


class LeadGuidance(ContractModel):
    guidance_mode: str | None = None
    target_position_m: Vec3 | None = None
    target_velocity_mps: Vec3 | None = None
    intercept_time_s: NonNegative | None = None
    intercept_point_m: Vec3 | None = None
    discriminant: FiniteFloat | None = None


class PNGuidance(ContractModel):
    guidance_mode: str | None = None
    range_m: NonNegative | None = None
    closing_speed_mps: FiniteFloat | None = None
    los_unit: Vec3 | None = None
    los_rate_radps: Vec3 | None = None
    acceleration_mps2: Vec3 | None = None


class RMPCGuidance(ContractModel):
    guidance_mode: str | None = None
    initialization_mode: str | None = None
    scenario_scores: dict[str, FiniteFloat] | None = None
    robust_score: FiniteFloat | None = None
    best_sequence: list[Vec3] | None = None
    objective_evaluations: Annotated[int, Field(ge=0)] | None = None
    decision_time_s: NonNegative | None = None


class PolicyTelemetry(ContractModel):
    decision_step: Annotated[int, Field(ge=0)]
    decision_policy_visible: PolicyVisible
    requested_action: Vec3
    requested_action_norm: NonNegative
    defender_policy_input: list[FiniteFloat] = Field(min_length=1)
    defender_model_input: list[FiniteFloat] | None = None
    attacker_observation: list[FiniteFloat] | None = Field(default=None, min_length=36, max_length=36)
    attacker_tactical_command: Vec3 | None = None
    attacker_command_source: Literal["a0_scripted", "a1_learned"] | None = None
    attacker_mission_command: Vec3 | None = None
    attacker_final_command: Vec3 | None = None
    desired_command: Vec3 | None = None
    safe_command: Vec3 | None = None
    requested_to_desired_delta: Vec3 | None = None
    requested_to_desired_angle_deg: Annotated[FiniteFloat, Field(ge=0, le=180)] | None = None
    desired_to_safe_delta: Vec3 | None = None
    desired_to_safe_angle_deg: Annotated[FiniteFloat, Field(ge=0, le=180)] | None = None
    requested_to_velocity_angle_deg: Annotated[FiniteFloat, Field(ge=0, le=180)] | None = None
    controller_action_executed: bool
    defender_standby: bool
    avoidance_active: bool
    avoidance_angle_deg: NonNegative | None = None
    executed_velocity_mps: Vec3
    executed_acceleration_mps2: NonNegative | None = None
    executed_turn_rate_degps: NonNegative | None = None
    acceleration_saturated: bool | None = None
    turn_saturated: bool | None = None
    climb_saturated: bool | None = None
    requested_horizontal_heading_deg: FiniteFloat | None = None
    executed_horizontal_heading_deg: FiniteFloat | None = None
    heading_change_deg: FiniteFloat | None = None
    heading_reversal: bool | None = None
    lrppo: LRPPOGuidance | None = None
    lead: LeadGuidance | None = None
    pn: PNGuidance | None = None
    rmpc: RMPCGuidance | None = None
    unavailable_fields: dict[str, str] = Field(default_factory=dict)


class PrivilegedDiagnostics(ContractModel):
    attacker_position_m: Vec3
    attacker_velocity_mps: Vec3
    tracking_error_m: NonNegative | None = None
    defender_attacker_range_m: NonNegative
    attacker_asset_range_m: NonNegative
    closing_speed_mps: FiniteFloat | None = None
    simulator_info: dict[str, JsonValue]
    unavailable_fields: dict[str, str] = Field(default_factory=dict)


class ReplayFrame(ContractModel):
    step: Annotated[int, Field(ge=0)]
    time_s: NonNegative
    defender: EntityState
    attacker: EntityState
    asset: EntityState
    moving_obstacles: list[MovingObstacle]
    policy_visible: PolicyVisible
    telemetry: PolicyTelemetry | None = None
    privileged_diagnostics: PrivilegedDiagnostics
    detection_radius_m: NonNegative
    detection_time_step: Annotated[int, Field(ge=0)] | None = None
    defender_action_applied: bool | None = None
    cumulative_reward: FiniteFloat | None = None
    environmental_state: dict[str, JsonValue] = Field(default_factory=dict)
    unavailable_fields: dict[str, str] = Field(default_factory=dict)
    reward: FiniteFloat | None = None
    terminated: bool
    truncated: bool
    outcome: Literal["ongoing"] | Outcome


class ReplayEvent(ContractModel):
    kind: Literal["detection", "interception", "unsafe_interception", "asset_reached", "collision", "timeout"]
    step: Annotated[int, Field(ge=0)]
    time_s: NonNegative
    source: str
    entity: Literal["defender", "attacker", "asset"] | None = None
    obstacle_id: str | None = None
    position_m: Vec3 | None = None
    contact_fraction: Annotated[FiniteFloat, Field(ge=0, le=1)] | None = None
    unavailable_fields: dict[str, str] = Field(default_factory=dict)


class EpisodeSummary(ContractModel):
    outcome: Outcome
    steps: Annotated[int, Field(gt=0)]
    duration_s: NonNegative
    detection_time_step: Annotated[int, Field(ge=0)] | None = None
    detection_time_s: NonNegative | None = None
    interception_step: Annotated[int, Field(ge=0)] | None = None
    interception_time_s: NonNegative | None = None
    collision_type: str | None = None
    collision_step: Annotated[int, Field(ge=0)] | None = None
    collision_time_s: NonNegative | None = None
    final_defender_attacker_distance_m: NonNegative
    final_attacker_asset_distance_m: NonNegative
    total_reward: FiniteFloat
    safety_metrics: dict[str, JsonValue] = Field(default_factory=dict)


class ReplayEpisode(ContractModel):
    episode_id: str
    simulation_seed: Annotated[int, Field(ge=0)]
    defender: PolicySelection
    attacker: PolicySelection
    environment: EnvironmentSpec
    terminal_flags: dict[str, bool]
    static_obstacles: list[StaticObstacle]
    frames: list[ReplayFrame] = Field(min_length=2)
    events: list[ReplayEvent]
    summary: EpisodeSummary


class ReplayProvenance(ContractModel):
    simulation_source_commit: Commit
    visualizer_source_commit: Commit
    source_dirty: bool
    visualizer_dirty: bool
    deterministic_inference: Literal[True] = True
    python_version: str
    package_versions: dict[str, str]
    model_device: str
    registry_sha256: Sha256


class ReplayBundle(ContractModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    coordinate_system: Literal["simulation_xyz_z_up_m"] = "simulation_xyz_z_up_m"
    presentation: Literal["horizontal_projection_fixed_heights"] = "horizontal_projection_fixed_heights"
    provenance: ReplayProvenance
    episode_count: Literal[1] = 1
    seed_start: Annotated[int, Field(ge=0)]
    episodes: list[ReplayEpisode]

    @model_validator(mode="after")
    def validate_sequence(self):
        if len(self.episodes) != self.episode_count:
            raise ValueError("episode_count must match the number of recorded episodes.")
        if [episode.simulation_seed for episode in self.episodes] != list(
            range(self.seed_start, self.seed_start + self.episode_count)
        ):
            raise ValueError("Episodes must use consecutive simulation seeds in recording order.")
        return self
