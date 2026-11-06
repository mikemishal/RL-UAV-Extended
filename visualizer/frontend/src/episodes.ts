/**
 * Simplified two-scenario episode API client (brief: "simplify the application backend around exactly TWO
 * user-facing environments"). Fetch helpers and types for the additive `/api/episodes*` and
 * `/api/scenario-presets` endpoints -- see `visualizer/backend/episode_models.py` / `main.py`. Deliberately
 * independent of `api.ts`'s `ReplayBundle`/checkpoint-oriented types: episodes created through this simplified
 * API have no checkpoint provenance to verify and a much smaller, simpler replay schema.
 */

export type ScenarioId = 'no_clutter' | 'clutter';
export type ResolvedAttacker = 'A0' | 'A1-CBF';

export interface ScenarioPreset {
  scenario: ScenarioId;
  scenario_label: string;
  clutter_level: 'C0' | 'C2';
  resolved_attacker: ResolvedAttacker;
  attacker_label: string;
  dynamic_obstacles_enabled: boolean;
  viability_conditioned_init: boolean;
  defender_cbf_enabled: boolean;
}

export interface EpisodeMetadata {
  episode_id: string;
  created_at: string;
  scenario: ScenarioId;
  scenario_label: string;
  resolved_attacker: ResolvedAttacker;
  defender: string;
  seed: number;
  status: 'completed' | 'failed';
  outcome: string;
  duration: number;
  steps: number;
  asset_captured: boolean;
  defender_interception: boolean;
  obstacle_collision: boolean;
  replay_path: string;
  source_commit: string;
  controller_version: string;
  is_new: boolean;
  min_defender_separation_m: number | null;
  min_obstacle_clearance_m: number | null;
  path_length_m: number | null;
  error_message: string | null;
}

export type Vec3 = readonly [number, number, number];

export interface EpisodeReplayStep {
  t: number;
  attacker_pos: Vec3;
  defender_pos: Vec3;
  asset_pos: Vec3;
}

export interface EpisodeObstacle {
  min: Vec3;
  max: Vec3;
}

export interface EpisodeReplay {
  schema: string;
  outcome: string;
  obstacles: EpisodeObstacle[];
  steps: EpisodeReplayStep[];
}

export interface EpisodeDetail {
  metadata: EpisodeMetadata;
  replay: EpisodeReplay;
}

export interface EpisodeListResponse {
  episodes: EpisodeMetadata[];
  total: number;
  new_count: number;
  saved_count: number;
}

export interface EpisodeCreateRequest {
  scenario: ScenarioId;
  defender: string;
  seed?: number;
}

export interface DeleteOutcome {
  episode_id: string;
  deleted: boolean;
  reason: string | null;
}

export interface BulkDeleteResponse {
  results: DeleteOutcome[];
}

export class EpisodesApiError extends Error {
  constructor(message: string, readonly status: number, readonly code: string) {
    super(message);
  }
}

async function episodesApi<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(path, init);
  } catch {
    throw new EpisodesApiError('Backend unavailable. Start the localhost API and try again.', 0, 'BACKEND_UNAVAILABLE');
  }
  const payload: unknown = await response.json().catch(() => null);
  if (!response.ok) {
    const error = typeof payload === 'object' && payload !== null && 'error' in payload
      ? (payload as { error?: { code?: unknown; message?: unknown } }).error
      : undefined;
    throw new EpisodesApiError(
      typeof error?.message === 'string' ? error.message : `Request failed (HTTP ${response.status}).`,
      response.status,
      typeof error?.code === 'string' ? error.code : 'API_ERROR',
    );
  }
  return payload as T;
}

export function getScenarioPresets(): Promise<{ presets: ScenarioPreset[] }> {
  return episodesApi('/api/scenario-presets');
}

export function createEpisode(request: EpisodeCreateRequest): Promise<EpisodeMetadata> {
  return episodesApi('/api/episodes', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
}

export interface EpisodeListFilters {
  new?: boolean;
  scenario?: ScenarioId;
  defender?: string;
}

export function listEpisodes(filters: EpisodeListFilters = {}): Promise<EpisodeListResponse> {
  const params = new URLSearchParams();
  if (filters.new !== undefined) params.set('new', String(filters.new));
  if (filters.scenario) params.set('scenario', filters.scenario);
  if (filters.defender) params.set('defender', filters.defender);
  const query = params.toString();
  return episodesApi(`/api/episodes${query ? `?${query}` : ''}`);
}

export function getEpisode(episodeId: string): Promise<EpisodeDetail> {
  return episodesApi(`/api/episodes/${encodeURIComponent(episodeId)}`);
}

export function deleteEpisode(episodeId: string): Promise<DeleteOutcome> {
  return episodesApi(`/api/episodes/${encodeURIComponent(episodeId)}`, { method: 'DELETE' });
}

export function deleteEpisodes(episodeIds: string[]): Promise<BulkDeleteResponse> {
  return episodesApi('/api/episodes/delete', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ episode_ids: episodeIds }),
  });
}

/** Friendly, stable display labels for the 9 supported defenders (brief section 5) -- kept local rather than
 * depending on the heavier, checkpoint-verification-oriented `/api/policies` catalog, which the simplified UI
 * uses only opportunistically for availability information (see `SimplifiedApp.tsx`). */
export const DEFENDER_LABELS: Record<string, string> = {
  passive: 'Passive',
  greedy: 'Greedy',
  kalman_greedy: 'Kalman Greedy',
  pn: 'Proportional Navigation (PN)',
  lead: 'Lead',
  rmpc: 'RMPC',
  ppo: 'PPO',
  ppo_kalman: 'PPO + Kalman',
  lr_ppo: 'LR-PPO',
};

export const DEFENDER_IDS = Object.keys(DEFENDER_LABELS);

/** Friendly outcome labels (brief section 10) -- never show the raw backend enum string when a friendly label
 * exists; fall back to a humanized version of the raw string for anything unrecognized rather than hiding it. */
export function friendlyOutcome(outcome: string): string {
  const known: Record<string, string> = {
    soldier_caught: 'Asset Captured',
    intercepted: 'Defender Intercepted',
    unsafe_intercept: 'Defender Intercepted (unsafe)',
    enemy_static_obstacle_collision: 'Obstacle Collision',
    enemy_dynamic_obstacle_collision: 'Obstacle Collision',
    enemy_endurance_exhausted: 'Endurance Exhausted',
    timeout: 'Timeout',
    ongoing: 'Ongoing',
    episode_failed: 'Episode Failed',
  };
  return known[outcome] ?? outcome.replace(/_/g, ' ').replace(/^./, c => c.toUpperCase());
}
