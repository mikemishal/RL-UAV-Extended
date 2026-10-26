export interface CheckpointSeed {
  training_seed: number;
  available: boolean;
  checkpoint_role: string;
  checkpoint_sha256: string | null;
  unavailability_reason: string | null;
  observation_shape: number[];
  estimator: string;
}

export interface DefenderGeneration {
  generation: string;
  display_name: string;
  architecture: string;
  checkpoint_role: string;
  available: boolean;
  training_seeds: CheckpointSeed[];
  estimator: string;
  observation_shape: number[];
}

export interface DefenderPolicy {
  policy_id: string;
  display_name: string;
  classification: 'analytical' | 'learned';
  architecture: string | null;
  available: boolean;
  unavailability_reason: string | null;
  estimator: string | null;
  observation_shape: number[] | null;
  generations: DefenderGeneration[];
  supported_scenarios: { clutter: string; perception_modes: string[] }[];
  diagnostic_fields: string[];
}

export interface AttackerPolicy {
  attacker_id: string;
  display_name: string;
  classification: 'scripted' | 'learned';
  architecture: string;
  training_seed: number | null;
  held_out: boolean;
  available: boolean;
  checkpoint_sha256: string | null;
  unavailability_reason: string | null;
}

export interface PolicyCatalog {
  source_commit: string;
  registry_sha256: string;
  deterministic_inference: true;
  defenders: DefenderPolicy[];
  attackers: AttackerPolicy[];
}

export type ScenarioParameters = Record<string, unknown>;

/** V1: original environment. V2: corrected swept-contact semantics with direct obstacle-only A0. */
export type Semantics = 'V1' | 'V2';

export interface ScenarioConfiguration {
  config_sha256: string;
  parameters: ScenarioParameters;
}

export interface ScenarioPerception {
  id: 'local' | 'map_local';
  description: string;
  configurations: {
    measurement: ScenarioConfiguration;
    kalman: ScenarioConfiguration;
  };
}

export interface ScenarioLevel {
  clutter: 'C0' | 'C1' | 'C2' | 'C3';
  display_name: string;
  perceptions: ScenarioPerception[];
}

export interface ScenarioCatalog {
  scenarios: ScenarioLevel[];
}

export interface ReplayJobRequest {
  defender_policy_id: string;
  defender_generation?: string;
  defender_training_seed?: number;
  attacker_condition: string;
  clutter_level: string;
  perception_mode: string;
  semantics?: Semantics;
  seed_start: number;
  episode_count: 1 | 3 | 5 | 10;
  deterministic_inference: true;
}

export interface ReplayJob {
  job_id: string;
  request: ReplayJobRequest;
  state: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled' | 'interrupted';
  requested_episode_count: number;
  completed_count: number;
  current_episode_seed: number | null;
  completed_replay_ids: string[];
  error: string | null;
  cancellation_requested: boolean;
  created_at: string;
  started_at: string | null;
  updated_at: string;
  completed_at: string | null;
}

export interface ReplayListItem {
  replay_id: string;
  simulation_seed: number;
  defender: { policy_id: string; architecture: string | null; generation: string | null; training_seed: number | null; checkpoint: { sha256: string; role: string } | null };
  attacker: { policy_id: string; checkpoint: { sha256: string; role: string } | null };
  clutter: string;
  perception: string;
  semantics?: Semantics;
  outcome: string;
  steps: number;
  duration_s: number;
  detection_time_s: number | null;
  total_reward: number;
  collision_type: string | null;
  interception_time_s: number | null;
  provenance: {
    simulation_source_commit: string;
    visualizer_source_commit: string;
    source_dirty: boolean;
    visualizer_dirty: boolean;
    registry_sha256: string;
  };
}

export interface ReplayCatalog {
  replays: ReplayListItem[];
  warnings: string[];
}

export interface ResearchBookmark {
  bookmark_id: string;
  label: string;
  time_s: number;
  frame_step: number;
  decision_step: number | null;
  sync_mode: 'absolute' | 'detection';
  local_time_a_s: number | null;
  local_time_b_s: number | null;
  observation: string;
  interpretation: string;
  tags: string[];
  camera_pose: Record<string, unknown> | null;
  created_at: string;
}

export interface ResearchCase {
  case_id: string;
  case_type: 'single' | 'comparison';
  title: string;
  description: string;
  tags: string[];
  notes: string;
  replay_ids: string[];
  replay_references: { replay_id: string; artifact_sha256: string; integrity: string }[];
  integrity: string;
  comparison_status: 'matched' | 'exploratory' | 'incompatible' | null;
  comparison_reasons: string[];
  metadata: {
    defender_families: string[];
    generations: string[];
    training_seeds: number[];
    attackers: string[];
    clutter: string[];
    perception: string[];
    simulation_seeds: number[];
    outcomes: string[];
    source_commits: string[];
    dirty_provenance: boolean;
    created_at: string;
  };
  bookmarks: ResearchBookmark[];
  version: number;
}

export interface CaseCreateRequest {
  case_type: 'single' | 'comparison';
  title: string;
  description?: string;
  tags?: string[];
  notes?: string;
  replay_ids: string[];
}

export interface BookmarkCreateRequest {
  label: string;
  time_s: number;
  frame_step: number;
  decision_step?: number | null;
  sync_mode: 'absolute' | 'detection';
  local_time_a_s: number;
  local_time_b_s?: number | null;
  observation?: string;
  interpretation?: string;
  tags?: string[];
  camera_pose?: Record<string, unknown> | null;
}

export interface ReplaySummary {
  replay_id: string;
  schema_version: string;
  simulation_seed: number;
  defender: {
    policy_id: string;
    architecture: string | null;
    generation: string | null;
    training_seed: number | null;
    checkpoint: { sha256: string; role: string } | null;
  };
  attacker: { policy_id: string; training_seed: number | null; checkpoint: { sha256: string; role: string } | null };
  environment: { clutter: string; awareness: string; semantics?: Semantics; config_sha256: string; dt_s: number };
  summary: {
    outcome: string;
    steps: number;
    duration_s: number;
    detection_time_step: number | null;
    detection_time_s: number | null;
    interception_time_s: number | null;
    collision_type: string | null;
    total_reward: number;
  };
  provenance: {
    simulation_source_commit: string;
    visualizer_source_commit: string;
    source_dirty: boolean;
    visualizer_dirty: boolean;
    registry_sha256: string;
  };
}

export type Vec3 = readonly [number, number, number];

export interface ReplayCheckpoint {
  artifact_id: string;
  research_relative_path: string;
  sha256: string;
  role: 'd0_start' | 'final' | 'secondary_best' | 'conference_best' | 'attacker_best';
}

export interface ReplayPolicySelection {
  policy_kind: 'analytical' | 'learned' | 'scripted';
  policy_id: string;
  architecture: string | null;
  generation: 'conference' | 'd0' | 'd0_plus' | 'd1' | null;
  training_seed: number | null;
  checkpoint: ReplayCheckpoint | null;
  observation_shape: number[] | null;
  action_shape: [3];
  estimator: 'measurement' | 'kalman' | 'attacker_local' | 'scripted_ground_truth';
  parameters: Record<string, unknown>;
}

export interface ReplayEntityState {
  position_m: Vec3;
  velocity_mps: Vec3 | null;
  speed_mps: number | null;
  horizontal_heading_deg: number | null;
  movement_direction: Vec3 | null;
}

export type EntityState = ReplayEntityState;

export interface ReplayStaticObstacle {
  obstacle_id: string;
  min_corner_m: Vec3;
  max_corner_m: Vec3;
  map_known: boolean;
  visual_kind: 'block' | 'tree';
}

export interface ReplayMovingObstacle {
  obstacle_id: string;
  kind: 'bird' | 'airborne_clutter';
  position_m: Vec3;
  velocity_mps: Vec3;
  radius_m: number;
}

export interface ReplayPolicyVisible {
  estimator: 'measurement' | 'kalman';
  observation: number[];
  soldier_pos_m: Vec3;
  defender_pos_m: Vec3;
  defender_vel_mps: Vec3;
  enemy_detected: boolean;
  enemy_measurement_m: Vec3 | null;
  enemy_measurement_velocity_mps: Vec3 | null;
  enemy_measurement_velocity_valid: boolean | null;
  e_hat_m: Vec3 | null;
  v_hat_mps: Vec3 | null;
  sector_features: number[] | null;
  unavailable_fields: Record<string, string>;
}

export interface ReplayLeadGuidance {
  guidance_mode: string | null;
  target_position_m: Vec3 | null;
  target_velocity_mps: Vec3 | null;
  intercept_time_s: number | null;
  intercept_point_m: Vec3 | null;
  discriminant: number | null;
}

export interface ReplayPNGuidance {
  guidance_mode: string | null;
  range_m: number | null;
  closing_speed_mps: number | null;
  los_unit: Vec3 | null;
  los_rate_radps: Vec3 | null;
  acceleration_mps2: Vec3 | null;
}

export interface ReplayRMPCGuidance {
  guidance_mode: string | null;
  initialization_mode: string | null;
  scenario_scores: Record<string, number> | null;
  robust_score: number | null;
  best_sequence: Vec3[] | null;
  objective_evaluations: number | null;
  decision_time_s: number | null;
}

export interface ReplayLRPPOGuidance {
  lead_direction: Vec3;
  residual_action: Vec3 | null;
  residual_norm_raw: number | null;
  residual_perpendicular: Vec3 | null;
  residual_norm_clipped: boolean | null;
  native_adjustment_deg: number | null;
  signed_horizontal_adjustment_deg: number | null;
  max_adjustment_deg: 30;
  near_bound: boolean | null;
  near_bound_margin_deg: number | null;
}

export interface ReplayPolicyTelemetry {
  decision_step: number;
  decision_policy_visible: ReplayPolicyVisible;
  requested_action: Vec3;
  requested_action_norm: number;
  defender_policy_input: number[];
  defender_model_input: number[] | null;
  attacker_observation: number[] | null;
  attacker_tactical_command: Vec3 | null;
  attacker_command_source: 'a0_scripted' | 'a1_learned' | null;
  attacker_mission_command: Vec3 | null;
  attacker_final_command: Vec3 | null;
  desired_command: Vec3 | null;
  safe_command: Vec3 | null;
  requested_to_desired_delta: Vec3 | null;
  requested_to_desired_angle_deg: number | null;
  desired_to_safe_delta: Vec3 | null;
  desired_to_safe_angle_deg: number | null;
  requested_to_velocity_angle_deg: number | null;
  controller_action_executed: boolean;
  defender_standby: boolean;
  avoidance_active: boolean;
  avoidance_angle_deg: number | null;
  executed_velocity_mps: Vec3;
  executed_acceleration_mps2: number | null;
  executed_turn_rate_degps: number | null;
  acceleration_saturated: boolean | null;
  turn_saturated: boolean | null;
  climb_saturated: boolean | null;
  requested_horizontal_heading_deg: number | null;
  executed_horizontal_heading_deg: number | null;
  heading_change_deg: number | null;
  heading_reversal: boolean | null;
  lrppo: ReplayLRPPOGuidance | null;
  lead: ReplayLeadGuidance | null;
  pn: ReplayPNGuidance | null;
  rmpc: ReplayRMPCGuidance | null;
  unavailable_fields: Record<string, string>;
}

export interface ReplayEvent {
  kind: 'detection' | 'interception' | 'unsafe_interception' | 'asset_reached' | 'collision' | 'timeout' | 'endurance_exhausted';
  step: number;
  time_s: number;
  source: string;
  entity: 'defender' | 'attacker' | 'asset' | null;
  obstacle_id: string | null;
  position_m: Vec3 | null;
  contact_fraction: number | null;
  unavailable_fields: Record<string, string>;
}

export interface ReplayFrame {
  step: number;
  time_s: number;
  defender: ReplayEntityState;
  attacker: ReplayEntityState;
  asset: ReplayEntityState;
  moving_obstacles: ReplayMovingObstacle[];
  policy_visible: ReplayPolicyVisible;
  telemetry: ReplayPolicyTelemetry | null;
  privileged_diagnostics: {
    attacker_position_m: Vec3;
    attacker_velocity_mps: Vec3;
    tracking_error_m: number | null;
    defender_attacker_range_m: number;
    attacker_asset_range_m: number;
    closing_speed_mps: number | null;
    simulator_info: Record<string, unknown>;
    unavailable_fields: Record<string, string>;
  };
  detection_radius_m: number;
  detection_time_step: number | null;
  defender_action_applied: boolean | null;
  cumulative_reward: number | null;
  environmental_state: Record<string, unknown>;
  unavailable_fields: Record<string, string>;
  reward: number | null;
  terminated: boolean;
  truncated: boolean;
  outcome: 'ongoing' | 'intercepted' | 'soldier_caught' | 'unsafe_intercept' | 'timeout'
    | 'defender_static_obstacle_collision' | 'defender_dynamic_obstacle_collision'
    | 'enemy_static_obstacle_collision' | 'enemy_dynamic_obstacle_collision' | 'enemy_endurance_exhausted';
}

export interface ReplayEpisode {
  episode_id: string;
  simulation_seed: number;
  defender: ReplayPolicySelection;
  attacker: ReplayPolicySelection;
  environment: {
    clutter: string;
    awareness: string;
    scenario_kind: 'canonical';
    builder: 'experiments.journal_clutter_protocol.build_canonical_env_config'
      | 'experiments.journal_v2_protocol.build_v2_env_config';
    semantics?: Semantics;
    config: Record<string, unknown>;
    config_sha256: string;
    dt_s: number;
  };
  terminal_flags: { terminated: boolean; truncated: boolean };
  static_obstacles: ReplayStaticObstacle[];
  frames: ReplayFrame[];
  events: ReplayEvent[];
  summary: {
    outcome: Exclude<ReplayFrame['outcome'], 'ongoing'>;
    steps: number;
    duration_s: number;
    detection_time_step: number | null;
    detection_time_s: number | null;
    interception_step: number | null;
    interception_time_s: number | null;
    collision_type: string | null;
    collision_step: number | null;
    collision_time_s: number | null;
    final_defender_attacker_distance_m: number;
    final_attacker_asset_distance_m: number;
    total_reward: number;
    safety_metrics: Record<string, unknown>;
  };
}

export interface ReplayBundle {
  schema_version: '1.0.0';
  coordinate_system: 'simulation_xyz_z_up_m';
  presentation: 'horizontal_projection_fixed_heights';
  provenance: ReplaySummary['provenance'] & {
    deterministic_inference: true;
    python_version: string;
    package_versions: Record<string, string>;
    model_device: string;
  };
  episode_count: 1;
  seed_start: number;
  episodes: [ReplayEpisode];
}

export class ApiError extends Error {
  constructor(message: string, readonly status: number, readonly code: string) {
    super(message);
  }
}

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(path, init);
  } catch {
    throw new ApiError('Backend unavailable. Start the localhost API and try again.', 0, 'BACKEND_UNAVAILABLE');
  }
  const payload: unknown = await response.json().catch(() => null);
  if (!response.ok) {
    const error = typeof payload === 'object' && payload !== null && 'error' in payload
      ? (payload as { error?: { code?: unknown; message?: unknown } }).error
      : undefined;
    throw new ApiError(
      typeof error?.message === 'string' ? error.message : `Request failed (HTTP ${response.status}).`,
      response.status,
      typeof error?.code === 'string' ? error.code : 'API_ERROR',
    );
  }
  return payload as T;
}

export function getPolicies(): Promise<PolicyCatalog> {
  return api('/api/policies');
}

export function getScenarios(): Promise<ScenarioCatalog> {
  return api('/api/scenarios');
}

export function getReplays(limit = 100): Promise<ReplayCatalog> {
  return api(`/api/replays?limit=${limit}`);
}

export function getReplaySummary(replayId: string): Promise<ReplaySummary> {
  return api(`/api/replays/${encodeURIComponent(replayId)}/summary`);
}

export function getReplay(replayId: string): Promise<ReplayBundle> {
  return api(`/api/replays/${encodeURIComponent(replayId)}`);
}

export function createReplayJob(request: ReplayJobRequest): Promise<ReplayJob> {
  return api('/api/replay-jobs', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
}

export function getReplayJob(jobId: string): Promise<ReplayJob> {
  return api(`/api/replay-jobs/${encodeURIComponent(jobId)}`);
}

export function cancelReplayJob(jobId: string): Promise<ReplayJob> {
  return api(`/api/replay-jobs/${encodeURIComponent(jobId)}/cancel`, { method: 'POST' });
}

export function getResearchCases(
  query = '',
  filters: Partial<Record<'architecture' | 'generation' | 'training_seed' | 'attacker' | 'clutter' | 'perception' | 'evaluation_seed' | 'outcome' | 'case_type' | 'tag' | 'annotations' | 'bookmarked' | 'sort', string>> = {},
): Promise<{ cases: ResearchCase[]; total: number }> {
  const search = new URLSearchParams();
  if (query.trim()) search.set('q', query.trim());
  Object.entries(filters).forEach(([key, value]) => {
    if (value !== undefined && value !== '') search.set(key, value);
  });
  return api(`/api/cases?${search.toString()}`);
}

export function createResearchCase(request: CaseCreateRequest): Promise<ResearchCase> {
  return api('/api/cases', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
}

export function updateResearchCase(
  caseId: string,
  patch: Partial<Pick<ResearchCase, 'title' | 'description' | 'tags' | 'notes'>> & { expected_version: number },
): Promise<ResearchCase> {
  return api(`/api/cases/${encodeURIComponent(caseId)}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(patch),
  });
}

export function deleteResearchCase(caseId: string, version: number): Promise<void> {
  return api(`/api/cases/${encodeURIComponent(caseId)}?expected_version=${version}`, { method: 'DELETE' });
}

export function addResearchBookmark(
  caseId: string,
  version: number,
  bookmark: BookmarkCreateRequest,
): Promise<ResearchCase> {
  return api(`/api/cases/${encodeURIComponent(caseId)}/bookmarks`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      expected_version: version,
      bookmark: {
        ...bookmark,
        title: bookmark.label,
        camera: bookmark.camera_pose ?? {},
      },
    }),
  });
}

export function updateResearchBookmark(
  caseId: string,
  bookmarkId: string,
  version: number,
  patch: Partial<ResearchBookmark>,
): Promise<ResearchCase> {
  const { label, camera_pose, ...fields } = patch;
  return api(`/api/cases/${encodeURIComponent(caseId)}/bookmarks/${encodeURIComponent(bookmarkId)}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      expected_version: version,
      ...fields,
      ...(label !== undefined ? { title: label } : {}),
      ...(camera_pose !== undefined ? { camera: camera_pose } : {}),
    }),
  });
}

export function deleteResearchBookmark(caseId: string, bookmarkId: string, version: number): Promise<ResearchCase> {
  return api(`/api/cases/${encodeURIComponent(caseId)}/bookmarks/${encodeURIComponent(bookmarkId)}?expected_version=${version}`, {
    method: 'DELETE',
  });
}

export function createCaseExport(
  caseId: string,
  request: { kind: 'report' | 'episode_csv' | 'comparison_csv' | 'trajectory_svg' | 'telemetry_svg' | 'package_zip'; start_time_s?: number; end_time_s?: number; sync_mode?: 'absolute' | 'detection'; camera?: string },
): Promise<{ export_id: string; case_id: string; kind: string; files: { name: string; sha256: string; bytes: number; download_url: string }[] }> {
  return api(`/api/cases/${encodeURIComponent(caseId)}/exports`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
}

export function createChartCapture(
  caseId: string,
  exportId: string,
  request: { source_filename: string; source_sha256: string; png_base64: string; width: 2100; height: number; dpi: 300 },
): Promise<{ export_id: string; files: { name: string; sha256: string; bytes: number; download_url: string; kind?: string }[] }> {
  return api(`/api/cases/${encodeURIComponent(caseId)}/exports/${encodeURIComponent(exportId)}/chart-capture`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
}

export function createSceneCapture(
  caseId: string,
  request: {
    replay_id: string;
    replay_sha256: string;
    png_base64: string;
    width: number;
    height: number;
    time_s: number;
    camera: string;
    projection: 'perspective' | 'top-down';
  },
): Promise<{ export_id: string; files: { name: string; sha256: string; bytes: number; download_url: string }[] }> {
  return api(`/api/cases/${encodeURIComponent(caseId)}/exports/scene-capture`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
}

export function getHealth(): Promise<{ service: string; phase: number; status: 'ok' | 'degraded'; data_root: string }> {
  return api('/api/health');
}
