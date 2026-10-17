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
  environment: { clutter: string; awareness: string; config_sha256: string; dt_s: number };
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

export function getHealth(): Promise<{ service: string; phase: number; status: 'ok' | 'degraded'; data_root: string }> {
  return api('/api/health');
}
