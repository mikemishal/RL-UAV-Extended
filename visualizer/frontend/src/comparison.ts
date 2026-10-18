import type { ReplayBundle, ReplayEpisode, ReplayListItem, ReplayPolicySelection } from './api';
import { sampleAtTime } from './playback';

export type ComparisonSyncMode = 'absolute' | 'detection';
export type PairValidationStatus = 'matched' | 'exploratory' | 'incompatible';

export interface ComparisonValidation {
  status: PairValidationStatus;
  reasons: string[];
}

export interface DefenderChoice {
  policyId: string;
  generation: string | null;
  trainingSeed: number | null;
}

export interface ExpectedCheckpoint {
  sha256: string | null;
  role: string;
}

function stableValue(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(stableValue);
  if (typeof value !== 'object' || value === null) return value;
  return Object.fromEntries(Object.entries(value as Record<string, unknown>)
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([key, item]) => [key, stableValue(item)]));
}

function stableJson(value: unknown): string {
  return JSON.stringify(stableValue(value));
}

function physicalConfig(episode: ReplayEpisode): Record<string, unknown> {
  const config = episode.environment.config;
  return Object.fromEntries(Object.entries(config).filter(([key]) => key !== 'use_kalman_tracking'));
}

function initialWorld(episode: ReplayEpisode): unknown {
  const frame = episode.frames[0];
  return {
    defender: frame.defender,
    attacker: frame.attacker,
    asset: frame.asset,
    staticObstacles: [...episode.static_obstacles].sort((a, b) => a.obstacle_id.localeCompare(b.obstacle_id)),
    movingObstacles: [...frame.moving_obstacles].sort((a, b) => a.obstacle_id.localeCompare(b.obstacle_id)),
  };
}

function checkpointIdentity(selection: ReplayPolicySelection): string | null {
  return selection.checkpoint?.sha256 ?? null;
}

export function validateComparisonPair(left: ReplayBundle, right: ReplayBundle): ComparisonValidation {
  const a = left.episodes[0];
  const b = right.episodes[0];
  const incompatible: string[] = [];
  const exploratory: string[] = [];

  if (left.schema_version !== right.schema_version || left.schema_version !== '1.0.0') {
    incompatible.push('Replay schema versions do not match or are unsupported.');
  }
  if (left.coordinate_system !== right.coordinate_system
      || left.presentation !== right.presentation) {
    incompatible.push('Coordinate or presentation contracts differ.');
  }
  if (a.simulation_seed !== b.simulation_seed) incompatible.push('Evaluation seeds differ.');
  if (a.attacker.policy_id !== b.attacker.policy_id
      || checkpointIdentity(a.attacker) !== checkpointIdentity(b.attacker)) {
    incompatible.push('Attacker identity or frozen checkpoint hash differs.');
  }
  if (a.environment.clutter !== b.environment.clutter
      || a.environment.awareness !== b.environment.awareness
      || a.environment.scenario_kind !== b.environment.scenario_kind
      || a.environment.builder !== b.environment.builder) {
    incompatible.push('Canonical clutter or perception scenario differs.');
  }
  if (a.environment.dt_s !== b.environment.dt_s) incompatible.push('Simulator time steps differ.');
  if (stableJson(physicalConfig(a)) !== stableJson(physicalConfig(b))) {
    incompatible.push('Resolved physical environment parameters differ.');
  }
  if (stableJson(initialWorld(a)) !== stableJson(initialWorld(b))) {
    incompatible.push('Initial entity state or obstacle realization differs.');
  }
  if (left.provenance.simulation_source_commit !== right.provenance.simulation_source_commit) {
    incompatible.push('Simulator source revisions differ.');
  }
  if (left.provenance.python_version !== right.provenance.python_version
      || stableJson(left.provenance.package_versions) !== stableJson(right.provenance.package_versions)) {
    incompatible.push('Python or scientific package versions differ; behavior equivalence is not established.');
  }
  if (left.provenance.model_device !== right.provenance.model_device) {
    incompatible.push('Model inference devices differ; numerical behavior equivalence is not established.');
  }
  if (left.provenance.deterministic_inference !== true || right.provenance.deterministic_inference !== true) {
    incompatible.push('Deterministic evaluation inference is not recorded for both episodes.');
  }
  if (a.defender.estimator !== b.defender.estimator) {
    exploratory.push(`Policy-visible estimator contracts differ (${a.defender.estimator} vs ${b.defender.estimator}); each pane retains its own recorded observations.`);
  }
  if (a.environment.config_sha256 !== b.environment.config_sha256) {
    exploratory.push('Full resolved configuration hashes differ; only the explicit estimator tracking flag is excluded from the physical-parameter comparison.');
  }
  if (left.provenance.source_dirty || right.provenance.source_dirty) {
    exploratory.push('At least one replay was generated from a dirty simulator worktree; source equivalence beyond the recorded commit cannot be proven.');
  }
  if (left.provenance.visualizer_dirty || right.provenance.visualizer_dirty) {
    exploratory.push('At least one replay records a dirty visualizer worktree.');
  }
  if (left.provenance.visualizer_source_commit !== right.provenance.visualizer_source_commit) {
    exploratory.push('Replay recorder source revisions differ.');
  }

  if (incompatible.length) return { status: 'incompatible', reasons: incompatible };
  if (exploratory.length) return { status: 'exploratory', reasons: exploratory };
  return {
    status: 'matched',
    reasons: ['Same evaluation seed, attacker checkpoint, physical configuration, initial world state, time step, and clean verified simulator provenance.'],
  };
}

export function matchesDefenderChoice(
  item: ReplayListItem,
  choice: DefenderChoice,
  expectedCheckpoint?: ExpectedCheckpoint | null,
): boolean {
  const selectionMatches = item.defender.policy_id === choice.policyId
    && item.defender.generation === choice.generation
    && item.defender.training_seed === choice.trainingSeed;
  if (!selectionMatches || expectedCheckpoint === undefined) return selectionMatches;
  if (expectedCheckpoint === null) return item.defender.checkpoint === null;
  return item.defender.checkpoint?.sha256 === expectedCheckpoint.sha256
    && item.defender.checkpoint?.role === expectedCheckpoint.role;
}

export function mapAxisTimeToEpisodeTime(
  axisTimeS: number,
  episode: ReplayEpisode,
  mode: ComparisonSyncMode,
): number {
  const detectionOffset = mode === 'detection' ? episode.summary.detection_time_s : 0;
  return Math.max(0, Math.min(episode.summary.duration_s, axisTimeS + (detectionOffset ?? 0)));
}

export function comparisonAxisRange(
  a: ReplayEpisode,
  b: ReplayEpisode,
  mode: ComparisonSyncMode,
): { minimum: number; maximum: number } | null {
  if (mode === 'absolute') {
    return { minimum: 0, maximum: Math.max(a.summary.duration_s, b.summary.duration_s) };
  }
  const detectionA = a.summary.detection_time_s;
  const detectionB = b.summary.detection_time_s;
  if (detectionA === null || detectionB === null) return null;
  return {
    minimum: Math.max(-detectionA, -detectionB),
    maximum: Math.max(a.summary.duration_s - detectionA, b.summary.duration_s - detectionB),
  };
}

export function defenderPathLength(episode: ReplayEpisode): number {
  let distance = 0;
  for (let index = 1; index < episode.frames.length; index += 1) {
    const previous = episode.frames[index - 1].defender.position_m;
    const current = episode.frames[index].defender.position_m;
    distance += Math.hypot(current[0] - previous[0], current[1] - previous[1], current[2] - previous[2]);
  }
  return distance;
}

export function firstPositionDivergence(
  a: ReplayEpisode,
  b: ReplayEpisode,
  mode: ComparisonSyncMode,
  thresholdM: number,
): { axisTimeS: number; separationM: number } | null {
  if (!(thresholdM > 0) || !Number.isFinite(thresholdM)) return null;
  const range = comparisonAxisRange(a, b, mode);
  if (!range) return null;
  for (const frame of a.frames) {
    const axisTimeS = mode === 'detection'
      ? frame.time_s - (a.summary.detection_time_s ?? 0)
      : frame.time_s;
    if (axisTimeS < range.minimum || axisTimeS > range.maximum) continue;
    const otherTimeS = mapAxisTimeToEpisodeTime(axisTimeS, b, mode);
    const otherIndex = sampleAtTime(b, otherTimeS).lowerIndex;
    const otherFrame = b.frames[otherIndex];
    if (Math.abs(otherFrame.time_s - otherTimeS) > 1e-9) continue;
    const first = frame.defender.position_m;
    const second = otherFrame.defender.position_m;
    const separationM = Math.hypot(first[0] - second[0], first[1] - second[1], first[2] - second[2]);
    if (separationM >= thresholdM) return { axisTimeS, separationM };
  }
  return null;
}
