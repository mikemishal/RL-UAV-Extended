import type { ReplayEpisode, ReplayFrame, ReplayPolicyTelemetry, Vec3 } from './api';
import { sampleAtTime } from './playback';

const VECTOR_EPSILON = 1e-9;

function norm(vector: Vec3): number {
  return Math.hypot(vector[0], vector[1], vector[2]);
}

export function angleBetween3dDeg(first: Vec3 | null, second: Vec3 | null): number | null {
  if (!first || !second) return null;
  const firstNorm = norm(first);
  const secondNorm = norm(second);
  if (!Number.isFinite(firstNorm) || !Number.isFinite(secondNorm)
      || firstNorm < VECTOR_EPSILON || secondNorm < VECTOR_EPSILON) return null;
  const dot = first[0] * second[0] + first[1] * second[1] + first[2] * second[2];
  const cosine = Math.max(-1, Math.min(1, dot / (firstNorm * secondNorm)));
  return Math.acos(cosine) * 180 / Math.PI;
}

export function signedAzimuthDifferenceDeg(lead: Vec3 | null, requested: Vec3 | null): number | null {
  if (!lead || !requested) return null;
  const leadHorizontal = Math.hypot(lead[0], lead[1]);
  const requestedHorizontal = Math.hypot(requested[0], requested[1]);
  if (!Number.isFinite(leadHorizontal) || !Number.isFinite(requestedHorizontal)
      || leadHorizontal < VECTOR_EPSILON || requestedHorizontal < VECTOR_EPSILON) return null;
  const leadAzimuth = Math.atan2(lead[1], lead[0]);
  const requestedAzimuth = Math.atan2(requested[1], requested[0]);
  const difference = (requestedAzimuth - leadAzimuth + Math.PI) % (2 * Math.PI);
  return ((difference < 0 ? difference + 2 * Math.PI : difference) - Math.PI) * 180 / Math.PI;
}

export function elevationDifferenceDeg(lead: Vec3 | null, requested: Vec3 | null): number | null {
  if (!lead || !requested) return null;
  const leadNorm = norm(lead);
  const requestedNorm = norm(requested);
  if (!Number.isFinite(leadNorm) || !Number.isFinite(requestedNorm)
      || leadNorm < VECTOR_EPSILON || requestedNorm < VECTOR_EPSILON) return null;
  const elevation = (value: Vec3) => Math.atan2(value[2], Math.hypot(value[0], value[1]));
  return (elevation(requested) - elevation(lead)) * 180 / Math.PI;
}

export function euclideanDistance(first: Vec3 | null, second: Vec3 | null): number | null {
  if (!first || !second) return null;
  const distance = Math.hypot(first[0] - second[0], first[1] - second[1], first[2] - second[2]);
  return Number.isFinite(distance) ? distance : null;
}

export interface DecisionInterval {
  decisionStep: number;
  startFrame: ReplayFrame;
  endFrame: ReplayFrame;
  telemetry: ReplayPolicyTelemetry;
}

export function decisionIntervalAtTime(episode: ReplayEpisode, timeS: number): DecisionInterval | null {
  const sample = sampleAtTime(episode, timeS);
  if (sample.lowerIndex >= episode.frames.length - 1) return null;
  const startFrame = episode.frames[sample.lowerIndex];
  const endFrame = episode.frames[sample.lowerIndex + 1];
  const telemetry = endFrame.telemetry;
  if (!telemetry || telemetry.decision_step !== startFrame.step) return null;
  return { decisionStep: startFrame.step, startFrame, endFrame, telemetry };
}

export interface SeriesPoint {
  timeS: number;
  step: number;
  value: number | null;
}

export interface DiagnosticSeries {
  defenderAttackerDistance: SeriesPoint[];
  attackerAssetDistance: SeriesPoint[];
  reward: SeriesPoint[];
  cumulativeReward: SeriesPoint[];
  leadRequestedAngle: SeriesPoint[];
  horizontalAzimuth: SeriesPoint[];
  elevationChange: SeriesPoint[];
  requestedSafeAngle: SeriesPoint[];
  requestedVelocityAngle: SeriesPoint[];
  trackingError: SeriesPoint[];
  velocityEstimateError: SeriesPoint[];
  requestedActionTurn: SeriesPoint[];
}

function point(frame: ReplayFrame, timeS: number, value: number | null): SeriesPoint {
  return { timeS, step: frame.step, value: value !== null && Number.isFinite(value) ? value : null };
}

function decisionTarget(telemetry: ReplayPolicyTelemetry): Vec3 | null {
  const visible = telemetry.decision_policy_visible;
  if (!visible.enemy_detected) return null;
  return visible.estimator === 'kalman' ? visible.e_hat_m : visible.enemy_measurement_m;
}

function decisionVelocity(telemetry: ReplayPolicyTelemetry): Vec3 | null {
  const visible = telemetry.decision_policy_visible;
  if (!visible.enemy_detected) return null;
  return visible.estimator === 'kalman' ? visible.v_hat_mps : visible.enemy_measurement_velocity_mps;
}

export function deriveDiagnosticSeries(episode: ReplayEpisode): DiagnosticSeries {
  const distances = episode.frames.map(frame => ({
    defenderAttackerDistance: point(frame, frame.time_s, frame.privileged_diagnostics.defender_attacker_range_m),
    attackerAssetDistance: point(frame, frame.time_s, frame.privileged_diagnostics.attacker_asset_range_m),
    reward: point(frame, frame.time_s, frame.reward),
    cumulativeReward: point(frame, frame.time_s, frame.cumulative_reward),
  }));
  const transitions = episode.frames.slice(1).flatMap(endFrame => {
    const telemetry = endFrame.telemetry;
    if (!telemetry || telemetry.decision_step !== endFrame.step - 1) return [];
    const startFrame = episode.frames[endFrame.step - 1];
    const timeS = startFrame.time_s;
    const target = decisionTarget(telemetry);
    const estimateVelocity = decisionVelocity(telemetry);
    const previousTelemetry = startFrame.telemetry;
    return [{
      telemetry,
      timeS,
      trackingError: point(startFrame, timeS, euclideanDistance(
        target,
        startFrame.privileged_diagnostics.attacker_position_m,
      )),
      velocityEstimateError: point(startFrame, timeS, euclideanDistance(
        estimateVelocity,
        startFrame.privileged_diagnostics.attacker_velocity_mps,
      )),
      requestedActionTurn: point(startFrame, timeS, angleBetween3dDeg(
        previousTelemetry?.requested_action ?? null,
        telemetry.requested_action,
      )),
    }];
  });
  return {
    defenderAttackerDistance: distances.map(value => value.defenderAttackerDistance),
    attackerAssetDistance: distances.map(value => value.attackerAssetDistance),
    reward: distances.map(value => value.reward),
    cumulativeReward: distances.map(value => value.cumulativeReward),
    leadRequestedAngle: transitions.map(({ telemetry, timeS }) => point(
      episode.frames[telemetry.decision_step],
      timeS,
      telemetry.lrppo
        ? angleBetween3dDeg(telemetry.lrppo.lead_direction, telemetry.requested_action)
        : null,
    )),
    horizontalAzimuth: transitions.map(({ telemetry, timeS }) => point(
      episode.frames[telemetry.decision_step],
      timeS,
      telemetry.lrppo
        ? signedAzimuthDifferenceDeg(telemetry.lrppo.lead_direction, telemetry.requested_action)
        : null,
    )),
    elevationChange: transitions.map(({ telemetry, timeS }) => point(
      episode.frames[telemetry.decision_step],
      timeS,
      telemetry.lrppo
        ? elevationDifferenceDeg(telemetry.lrppo.lead_direction, telemetry.requested_action)
        : null,
    )),
    requestedSafeAngle: transitions.map(({ telemetry, timeS }) => point(
      episode.frames[telemetry.decision_step],
      timeS,
      telemetry.desired_to_safe_angle_deg,
    )),
    requestedVelocityAngle: transitions.map(({ telemetry, timeS }) => point(
      episode.frames[telemetry.decision_step],
      timeS,
      telemetry.requested_to_velocity_angle_deg,
    )),
    trackingError: transitions.map(({ trackingError }) => trackingError),
    velocityEstimateError: transitions.map(({ velocityEstimateError }) => velocityEstimateError),
    requestedActionTurn: transitions.map(({ requestedActionTurn }) => requestedActionTurn),
  };
}
