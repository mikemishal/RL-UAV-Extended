import { expect, test, type Page, type Route } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import { gunzipSync } from 'node:zlib';
import type { ReplayBundle, ReplayEntityState, ReplayEpisode, ReplayFrame, ReplayListItem, ReplayPolicyTelemetry } from '../src/api';
import {
  angleBetween3dDeg,
  decisionIntervalAtTime,
  deriveDiagnosticSeries,
  elevationDifferenceDeg,
  signedAzimuthDifferenceDeg,
} from '../src/diagnostics';
import { sampleAtTime, scenarioBounds, simToThree, simVectorToThree, staticObstacleBounds } from '../src/playback';
import {
  comparisonAxisRange,
  defenderPathLength,
  firstPositionDivergence,
  mapAxisTimeToEpisodeTime,
  matchesDefenderChoice,
  validateComparisonPair,
} from '../src/comparison';

const fixtureReplayId = 'f'.repeat(32);
function fixtureTelemetry(decisionStep: number, detected: boolean): ReplayPolicyTelemetry {
  const estimatedPosition = detected ? [9.4, 2, 13] as const : null;
  return {
    decision_step: decisionStep,
    decision_policy_visible: {
      estimator: 'measurement',
      observation: Array(39).fill(0),
      soldier_pos_m: [0, decisionStep, 0],
      defender_pos_m: [decisionStep, 0, 4 + decisionStep],
      defender_vel_mps: [2, 0, 2],
      enemy_detected: detected,
      enemy_measurement_m: estimatedPosition,
      enemy_measurement_velocity_mps: detected ? [-2, 4, 2] : null,
      enemy_measurement_velocity_valid: detected,
      e_hat_m: null,
      v_hat_mps: null,
      sector_features: null,
      unavailable_fields: {},
    },
    requested_action: [Math.sqrt(3) / 2, 0, 0.5],
    requested_action_norm: 1,
    defender_policy_input: Array(39).fill(0),
    defender_model_input: Array(39).fill(0),
    attacker_observation: null,
    attacker_tactical_command: null,
    attacker_command_source: null,
    attacker_mission_command: null,
    attacker_final_command: null,
    desired_command: [Math.sqrt(3) / 2, 0, 0.5],
    safe_command: [Math.sqrt(3) / 2, 0, 0.5],
    requested_to_desired_delta: [0, 0, 0],
    requested_to_desired_angle_deg: 0,
    desired_to_safe_delta: [0, 0, 0],
    desired_to_safe_angle_deg: 0,
    requested_to_velocity_angle_deg: 60,
    controller_action_executed: true,
    defender_standby: false,
    avoidance_active: false,
    avoidance_angle_deg: null,
    executed_velocity_mps: [0, 0, 4],
    executed_acceleration_mps2: null,
    executed_turn_rate_degps: null,
    acceleration_saturated: null,
    turn_saturated: null,
    climb_saturated: null,
    requested_horizontal_heading_deg: null,
    executed_horizontal_heading_deg: null,
    heading_change_deg: null,
    heading_reversal: null,
    lrppo: {
      lead_direction: [1, 0, 0],
      residual_action: null,
      residual_norm_raw: null,
      residual_perpendicular: null,
      residual_norm_clipped: null,
      native_adjustment_deg: 30,
      signed_horizontal_adjustment_deg: 0,
      max_adjustment_deg: 30,
      near_bound: false,
      near_bound_margin_deg: null,
    },
    lead: null,
    pn: null,
    rmpc: null,
    unavailable_fields: {
      'telemetry.lrppo.residual_action': 'raw PPO output is not exposed by the current wrapper',
      'telemetry.lrppo.residual_norm_clipped': 'internal clipping status is not exposed by the current wrapper',
    },
  };
}
const fixtureFrames: ReplayFrame[] = [0, 1, 2].map(step => {
  const defender: ReplayEntityState = {
    position_m: [step, 0, 4 + step],
    velocity_mps: [2, 0, 2],
    speed_mps: 2.828,
    horizontal_heading_deg: null,
    movement_direction: null,
  };
  const attacker: ReplayEntityState = {
    position_m: [10 - step, step * 2, 12 + step],
    velocity_mps: [-2, 4, 2],
    speed_mps: 4.9,
    horizontal_heading_deg: null,
    movement_direction: null,
  };
  const asset: ReplayEntityState = {
    position_m: [0, step, 0],
    velocity_mps: [0, 2, 0],
    speed_mps: 2,
    horizontal_heading_deg: null,
    movement_direction: null,
  };
  return {
    step,
    time_s: step * 0.5,
    defender,
    attacker,
    asset,
    moving_obstacles: [{
      obstacle_id: 'dynamic-0',
      kind: 'bird',
      position_m: [2 + step, 3, 5],
      velocity_mps: [2, 0, 0],
      radius_m: 0.5,
    }],
    policy_visible: {
      estimator: 'measurement',
      observation: Array(16).fill(0),
      soldier_pos_m: asset.position_m,
      defender_pos_m: defender.position_m,
      defender_vel_mps: defender.velocity_mps!,
      enemy_detected: step >= 1,
      enemy_measurement_m: step >= 1 ? attacker.position_m : null,
      enemy_measurement_velocity_mps: step >= 1 ? attacker.velocity_mps : null,
      enemy_measurement_velocity_valid: step >= 1,
      e_hat_m: null,
      v_hat_mps: null,
      sector_features: null,
      unavailable_fields: {},
    },
    telemetry: step === 0 ? null : fixtureTelemetry(step - 1, step >= 2),
    privileged_diagnostics: {
      attacker_position_m: attacker.position_m,
      attacker_velocity_mps: attacker.velocity_mps!,
      tracking_error_m: null,
      defender_attacker_range_m: 12 - step,
      attacker_asset_range_m: 16 - step,
      closing_speed_mps: null,
      simulator_info: {},
      unavailable_fields: {},
    },
    detection_radius_m: 15,
    detection_time_step: step >= 1 ? 1 : null,
    defender_action_applied: true,
    cumulative_reward: -step,
    environmental_state: {},
    unavailable_fields: {},
    reward: -1,
    terminated: step === 2,
    truncated: false,
    outcome: step === 2 ? 'intercepted' : 'ongoing',
  };
});

const fixtureEpisode: ReplayEpisode = {
  episode_id: fixtureReplayId,
  simulation_seed: 20,
  defender: {
    policy_kind: 'learned',
    policy_id: 'lr_ppo',
    architecture: 'lrppo_clutter_v1',
    generation: 'd1',
    training_seed: 77,
    checkpoint: {
      artifact_id: 'lrppo-d1-seed-77',
      research_relative_path: 'experiments/checkpoints/example.zip',
      sha256: 'a'.repeat(64),
      role: 'final',
    },
    observation_shape: [39],
    action_shape: [3],
    estimator: 'measurement',
    parameters: {},
  },
  attacker: {
    policy_kind: 'learned',
    policy_id: 'A1-203',
    architecture: 'adaptive_attacker_v1',
    generation: null,
    training_seed: 203,
    checkpoint: {
      artifact_id: 'a1-203',
      research_relative_path: 'experiments/checkpoints/attacker.zip',
      sha256: 'c'.repeat(64),
      role: 'attacker_best',
    },
    observation_shape: [36],
    action_shape: [3],
    estimator: 'attacker_local',
    parameters: {},
  },
  environment: {
    clutter: 'C3',
    awareness: 'map_local',
    scenario_kind: 'canonical',
    builder: 'experiments.journal_clutter_protocol.build_canonical_env_config',
    config: { L: 40, max_altitude: 30 },
    config_sha256: 'd'.repeat(64),
    dt_s: 0.5,
  },
  terminal_flags: { terminated: true, truncated: false },
  static_obstacles: [{
    obstacle_id: 'building-0',
    min_corner_m: [4, -2, 0],
    max_corner_m: [8, 2, 6],
    map_known: true,
    visual_kind: 'block',
  }],
  frames: fixtureFrames,
  events: [
    {
      kind: 'detection',
      step: 1,
      time_s: 0.5,
      source: 'info.just_detected',
      entity: 'attacker',
      obstacle_id: null,
      position_m: fixtureFrames[1].attacker.position_m,
      contact_fraction: null,
      unavailable_fields: {},
    },
    {
      kind: 'interception',
      step: 2,
      time_s: 1,
      source: 'terminated',
      entity: 'attacker',
      obstacle_id: null,
      position_m: null,
      contact_fraction: null,
      unavailable_fields: {},
    },
  ],
  summary: {
    outcome: 'intercepted',
    steps: 2,
    duration_s: 1,
    detection_time_step: 1,
    detection_time_s: 0.5,
    interception_step: 2,
    interception_time_s: 1,
    collision_type: null,
    collision_step: null,
    collision_time_s: null,
    final_defender_attacker_distance_m: 1,
    final_attacker_asset_distance_m: 15,
    total_reward: -2,
    safety_metrics: {},
  },
};

const kalmanFixtureEpisode: ReplayEpisode = {
  ...fixtureEpisode,
  defender: { ...fixtureEpisode.defender, policy_id: 'ppo_kalman', estimator: 'kalman' },
  frames: fixtureEpisode.frames.map(frame => ({
    ...frame,
    policy_visible: {
      ...frame.policy_visible,
      estimator: 'kalman',
      e_hat_m: frame.policy_visible.enemy_detected ? [9.4, 2, 13] : null,
      v_hat_mps: frame.policy_visible.enemy_detected ? [-1, 4, 2] : null,
    },
    telemetry: frame.telemetry ? {
      ...frame.telemetry,
      lrppo: null,
      decision_policy_visible: {
        ...frame.telemetry.decision_policy_visible,
        estimator: 'kalman',
        e_hat_m: frame.telemetry.decision_policy_visible.enemy_detected ? [9.4, 2, 13] : null,
        v_hat_mps: frame.telemetry.decision_policy_visible.enemy_detected ? [-1, 4, 2] : null,
      },
    } : null,
  })),
};

const fixtureBundle: ReplayBundle = {
  schema_version: '1.0.0',
  coordinate_system: 'simulation_xyz_z_up_m',
  presentation: 'horizontal_projection_fixed_heights',
  provenance: {
    simulation_source_commit: 'a'.repeat(40),
    visualizer_source_commit: 'b'.repeat(40),
    source_dirty: false,
    visualizer_dirty: false,
    deterministic_inference: true,
    python_version: '3.13.0',
    package_versions: {},
    model_device: 'cpu',
    registry_sha256: 'b'.repeat(64),
  },
  episode_count: 1,
  seed_start: 20,
  episodes: [fixtureEpisode],
};

function fixtureBundleFor(replayId: string, seedOffset = 0, episode: ReplayEpisode = fixtureEpisode): ReplayBundle {
  return {
    ...fixtureBundle,
    seed_start: fixtureBundle.seed_start + seedOffset,
    episodes: [{
      ...episode,
      episode_id: replayId,
      simulation_seed: episode.simulation_seed + seedOffset,
    }],
  };
}

function fixtureSummaryFor(replayId: string, seedOffset = 0) {
  return {
    ...summary,
    replay_id: replayId,
    simulation_seed: summary.simulation_seed + seedOffset,
    summary: {
      ...summary.summary,
      outcome: 'intercepted',
      steps: fixtureEpisode.summary.steps,
      duration_s: fixtureEpisode.summary.duration_s,
      detection_time_step: fixtureEpisode.summary.detection_time_step,
      detection_time_s: fixtureEpisode.summary.detection_time_s,
      interception_time_s: fixtureEpisode.summary.interception_time_s,
      collision_type: null,
      total_reward: fixtureEpisode.summary.total_reward,
    },
  };
}

async function mockFixtureEpisodes(page: Page, replayIds: string[], episode: ReplayEpisode = fixtureEpisode) {
  const items = replayIds.map((id, index) => ({
    replay_id: id,
    simulation_seed: 20 + index,
    defender: { policy_id: 'lr_ppo', architecture: 'lrppo_clutter_v1', generation: 'd1', training_seed: 77, checkpoint: { sha256: 'a'.repeat(64), role: 'final' } },
    attacker: { policy_id: 'A1-203', checkpoint: { sha256: 'c'.repeat(64), role: 'attacker_best' } },
    clutter: 'C3',
    perception: 'map_local',
    outcome: 'intercepted',
    steps: 2,
    duration_s: 1,
    detection_time_s: 0.5,
    total_reward: -2,
    collision_type: null,
    interception_time_s: 1,
    provenance: {
      simulation_source_commit: 'a'.repeat(40),
      visualizer_source_commit: 'b'.repeat(40),
      source_dirty: false,
      visualizer_dirty: false,
      registry_sha256: 'b'.repeat(64),
    },
  }));
  await page.route('**/api/replays?limit=100', route => route.fulfill({ json: { replays: items, warnings: [] } }));
  await page.route('**/api/replays/*', async route => {
    const path = new URL(route.request().url()).pathname;
    if (path.endsWith('/summary')) return route.fallback();
    const id = path.split('/').at(-1) ?? replayIds[0];
    const index = replayIds.indexOf(id);
    await route.fulfill({ json: fixtureBundleFor(id, Math.max(0, index), episode) });
  });
  await page.route('**/api/replays/*/summary', route => {
    const id = new URL(route.request().url()).pathname.split('/').at(-2) ?? replayIds[0];
    const index = replayIds.indexOf(id);
    return route.fulfill({ json: fixtureSummaryFor(id, Math.max(0, index)) });
  });
}

test('coordinate transforms, geometry, sampling, and interpolation preserve recorded state', () => {
  expect(simToThree([2, -3, 5])).toEqual([2, 5, 3]);
  expect(simVectorToThree([2, -3, 5])).toEqual([2, 5, 3]);
  const first: [number, number, number] = [3, -2, 4];
  const second: [number, number, number] = [1, 5, -2];
  const firstTransformed = simToThree(first);
  const secondTransformed = simToThree(second);
  const length = (value: number[]) => Math.hypot(...value);
  const cosine = (left: number[], right: number[]) => (
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
  ) / (length(left) * length(right));
  expect(length(first)).toBeCloseTo(length(firstTransformed), 12);
  expect(length(second)).toBeCloseTo(length(secondTransformed), 12);
  expect(cosine(first, second)).toBeCloseTo(cosine(firstTransformed, secondTransformed), 12);
  expect(staticObstacleBounds([4, -2, 0], [8, 2, 6])).toEqual({
    min: [4, 0, -2],
    max: [8, 6, 2],
    center: [6, 3, 0],
    extent: 6,
  });

  expect(scenarioBounds(fixtureEpisode).center).toEqual([5, 7, -1]);
  const original = JSON.stringify(fixtureEpisode.frames);
  const exact = sampleAtTime(fixtureEpisode, 0.5);
  expect(exact.defender).toEqual([1, 5, 0]);
  expect(exact.attacker).toEqual([9, 13, -2]);
  expect(exact.asset).toEqual([0, 0, -1]);
  expect(exact.dynamicObstacles.get('dynamic-0')).toEqual([3, 5, -3]);
  const between = sampleAtTime(fixtureEpisode, 0.25);
  expect(between.defender).toEqual([0.5, 4.5, 0]);
  expect(between.dynamicObstacles.get('dynamic-0')).toEqual([2.5, 5, -3]);
  expect(sampleAtTime(fixtureEpisode, -5).defender).toEqual(simToThree(fixtureFrames[0].defender.position_m));
  expect(sampleAtTime(fixtureEpisode, 50).defender).toEqual(simToThree(fixtureFrames[2].defender.position_m));
  expect(JSON.stringify(fixtureEpisode.frames)).toBe(original);
});

test('policy diagnostics calculate native angles and align telemetry to recorded transitions', () => {
  const lead: [number, number, number] = [1, 0, 0];
  const requested: [number, number, number] = [Math.sqrt(3) / 2, 0, 0.5];
  expect(angleBetween3dDeg(lead, requested)).toBeCloseTo(30, 10);
  expect(elevationDifferenceDeg(lead, requested)).toBeCloseTo(30, 10);
  expect(signedAzimuthDifferenceDeg([1, 0, 0], [0, 1, 0])).toBeCloseTo(90, 10);
  expect(signedAzimuthDifferenceDeg([1, 0, 0], [0, -1, 0])).toBeCloseTo(-90, 10);
  expect(angleBetween3dDeg([1e-12, 0, 0], requested)).toBeNull();
  expect(angleBetween3dDeg([Number.NaN, 0, 0], requested)).toBeNull();
  expect(signedAzimuthDifferenceDeg([0, 0, 1], requested)).toBeNull();
  expect(angleBetween3dDeg([0, 0, 0], requested)).toBeNull();
  expect(angleBetween3dDeg(lead, null)).toBeNull();

  const first = decisionIntervalAtTime(fixtureEpisode, 0);
  expect(first).not.toBeNull();
  expect(first?.decisionStep).toBe(0);
  expect(first?.startFrame.step).toBe(0);
  expect(first?.endFrame.step).toBe(1);
  expect(first?.telemetry.controller_action_executed).toBe(true);
  expect(decisionIntervalAtTime(fixtureEpisode, fixtureEpisode.summary.duration_s)).toBeNull();

  const before = JSON.stringify(fixtureEpisode);
  const series = deriveDiagnosticSeries(fixtureEpisode);
  const leadRequestedAngles = series.leadRequestedAngle.map(item => item.value);
  expect(leadRequestedAngles).toHaveLength(2);
  leadRequestedAngles.forEach(value => expect(value).toBeCloseTo(30, 10));
  expect(series.trackingError[0].value).toBeNull();
  expect(series.trackingError[1].value).toBeCloseTo(0.4, 10);
  expect(series.requestedVelocityAngle[0].value).toBe(60);
  expect(fixtureFrames[1].telemetry?.lrppo?.residual_action).toBeNull();
  expect(fixtureFrames[1].telemetry?.lrppo?.residual_norm_clipped).toBeNull();
  expect(JSON.stringify(fixtureEpisode)).toBe(before);
});

test('comparison eligibility distinguishes exact initial matching from exploratory estimator differences', () => {
  const kalmanD0: ReplayEpisode = {
    ...kalmanFixtureEpisode,
    defender: {
      ...kalmanFixtureEpisode.defender,
      generation: 'd0',
      training_seed: 74,
      architecture: 'ppo_kalman_clutter_v1',
      checkpoint: {
        ...kalmanFixtureEpisode.defender.checkpoint!,
        artifact_id: 'ppo-kalman-d0-seed-74',
        sha256: 'd'.repeat(64),
        role: 'd0_start',
      },
    },
    environment: {
      ...kalmanFixtureEpisode.environment,
      config: { ...kalmanFixtureEpisode.environment.config, use_kalman_tracking: true },
      config_sha256: 'e'.repeat(64),
    },
  };
  const left = fixtureBundleFor('a'.repeat(32));
  const right = fixtureBundleFor('b'.repeat(32), 0, kalmanD0);
  const validation = validateComparisonPair(left, right);
  expect(validation.status).toBe('exploratory');
  expect(validation.reasons.join(' ')).toContain('estimator contracts differ');

  const differentWorld: ReplayBundle = {
    ...right,
    episodes: [{
      ...kalmanD0,
      frames: kalmanD0.frames.map((frame, index) => index === 0
        ? { ...frame, asset: { ...frame.asset, position_m: [1, 0, 0] as const } }
        : frame),
    }],
  };
  expect(validateComparisonPair(left, differentWorld).status).toBe('incompatible');
  const differentAttacker: ReplayBundle = {
    ...right,
    episodes: [{
      ...kalmanD0,
      attacker: {
        ...kalmanD0.attacker,
        checkpoint: { ...kalmanD0.attacker.checkpoint!, sha256: 'f'.repeat(64) },
      },
    }],
  };
  expect(validateComparisonPair(left, differentAttacker).reasons.join(' ')).toContain('Attacker identity');
  const differentRuntime: ReplayBundle = {
    ...right,
    provenance: { ...right.provenance, package_versions: { numpy: '2.0' } },
  };
  expect(validateComparisonPair(left, differentRuntime).status).toBe('incompatible');
  expect(validateComparisonPair(left, differentRuntime).reasons.join(' ')).toContain('scientific package versions differ');

  const replayItem: ReplayListItem = {
    replay_id: 'a'.repeat(32),
    simulation_seed: 20,
    defender: { policy_id: 'lr_ppo', architecture: 'lrppo_clutter_v1', generation: 'd1', training_seed: 77, checkpoint: { sha256: 'a'.repeat(64), role: 'final' } },
    attacker: { policy_id: 'A1-203', checkpoint: { sha256: 'c'.repeat(64), role: 'attacker_best' } },
    clutter: 'C3',
    perception: 'map_local',
    outcome: 'intercepted',
    steps: 2,
    duration_s: 1,
    detection_time_s: 0.5,
    total_reward: -2,
    collision_type: null,
    interception_time_s: 1,
    provenance: {
      simulation_source_commit: 'a'.repeat(40),
      visualizer_source_commit: 'b'.repeat(40),
      source_dirty: false,
      visualizer_dirty: false,
      registry_sha256: 'b'.repeat(64),
    },
  };
  expect(matchesDefenderChoice(replayItem, { policyId: 'lr_ppo', generation: 'd1', trainingSeed: 77 })).toBe(true);
  expect(matchesDefenderChoice(replayItem, { policyId: 'lr_ppo', generation: 'd1', trainingSeed: 77 }, {
    sha256: 'a'.repeat(64), role: 'final',
  })).toBe(true);
  expect(matchesDefenderChoice(replayItem, { policyId: 'lr_ppo', generation: 'd1', trainingSeed: 77 }, {
    sha256: 'f'.repeat(64), role: 'final',
  })).toBe(false);
  expect(matchesDefenderChoice(replayItem, { policyId: 'lr_ppo', generation: 'd1', trainingSeed: 77 }, {
    sha256: 'a'.repeat(64), role: 'd0_start',
  })).toBe(false);
  expect(matchesDefenderChoice(replayItem, { policyId: 'lr_ppo', generation: 'd0', trainingSeed: 77 })).toBe(false);
});

test('comparison clock maps actual durations and detection offsets without post-terminal extrapolation', () => {
  const shorter: ReplayEpisode = {
    ...fixtureEpisode,
    frames: fixtureEpisode.frames.slice(0, 2).map((frame, index) => ({
      ...frame,
      terminated: index === 1,
      outcome: index === 1 ? 'intercepted' : 'ongoing',
    })),
    summary: { ...fixtureEpisode.summary, steps: 1, duration_s: 0.5 },
  };
  expect(comparisonAxisRange(shorter, fixtureEpisode, 'absolute')).toEqual({ minimum: 0, maximum: 1 });
  expect(mapAxisTimeToEpisodeTime(1, shorter, 'absolute')).toBe(0.5);
  expect(comparisonAxisRange(shorter, fixtureEpisode, 'detection')).toEqual({ minimum: -0.5, maximum: 0.5 });
  expect(mapAxisTimeToEpisodeTime(-0.25, fixtureEpisode, 'detection')).toBe(0.25);
  expect(mapAxisTimeToEpisodeTime(50, shorter, 'detection')).toBe(0.5);
  expect(defenderPathLength(fixtureEpisode)).toBeCloseTo(Math.sqrt(2) * 2, 10);
  const separated: ReplayEpisode = {
    ...fixtureEpisode,
    frames: fixtureEpisode.frames.map((frame, index) => index === 1
      ? { ...frame, defender: { ...frame.defender, position_m: [6, 0, 5] as const } }
      : frame),
  };
  expect(firstPositionDivergence(fixtureEpisode, separated, 'absolute', 5)?.axisTimeS).toBe(0.5);
  const asynchronouslySampled: ReplayEpisode = {
    ...separated,
    frames: separated.frames.map((frame, index) => index === 1 ? { ...frame, time_s: 0.75 } : frame),
  };
  expect(firstPositionDivergence(fixtureEpisode, asynchronouslySampled, 'absolute', 5)).toBeNull();
  expect(firstPositionDivergence(fixtureEpisode, separated, 'absolute', 50)).toBeNull();
});

const scenarioLevels = ['C0', 'C1', 'C2', 'C3'];
const parameters = {
  v_d: 18, v_e: 12, v_s: 1.5, detection_radius: 15, intercept_radius: 2.5,
  threat_radius: 2, unsafe_intercept_radius: 3.5, dt: 0.5, max_steps: 2000,
  max_altitude: 30, defender_max_accel: 8, defender_max_climb_rate: 6,
  defender_max_descent_rate: 6, defender_max_turn_rate_deg: 90,
  obstacle_count: 15, dynamic_obstacle_count: 4, obstacle_min_size: 3,
  obstacle_max_size: 7, obstacle_min_height: 4, obstacle_max_height: 12,
  dynamic_obstacle_min_radius: 0.3, dynamic_obstacle_max_radius: 1,
};

const supportedScenarios = scenarioLevels.map(clutter => ({ clutter, perception_modes: ['local', 'map_local'] }));
const seed = (training_seed: number) => ({
  training_seed, available: true, checkpoint_role: 'final', checkpoint_sha256: 'a'.repeat(64),
  unavailability_reason: null, observation_shape: [39], estimator: 'measurement',
});
const policies = {
  source_commit: 'a'.repeat(40),
  registry_sha256: 'b'.repeat(64),
  deterministic_inference: true,
  defenders: [
    {
      policy_id: 'greedy', display_name: 'Greedy', classification: 'analytical', architecture: 'greedy',
      available: true, unavailability_reason: null, estimator: 'measurement', observation_shape: [16],
      generations: [], supported_scenarios: supportedScenarios, diagnostic_fields: ['requested_action'],
    },
    {
      policy_id: 'lr_ppo', display_name: 'LR-PPO (clutter trained)', classification: 'learned', architecture: null,
      available: true, unavailability_reason: null, estimator: null, observation_shape: null,
      generations: [
        {
          generation: 'd0', display_name: 'D0 — original clutter-trained', architecture: 'lrppo_clutter_v1',
          checkpoint_role: 'd0_start', available: true, training_seeds: [{ ...seed(77), checkpoint_role: 'd0_start' }],
          estimator: 'measurement', observation_shape: [39],
        },
        {
          generation: 'd1', display_name: 'D1 — adversarial-mixture continuation', architecture: 'lrppo_clutter_v1',
          checkpoint_role: 'final', available: true, training_seeds: [seed(77)],
          estimator: 'measurement', observation_shape: [39],
        },
      ],
      supported_scenarios: supportedScenarios, diagnostic_fields: ['lead_direction'],
    },
  ],
  attackers: [
    { attacker_id: 'A0', display_name: 'A0 — Scripted attacker', classification: 'scripted', architecture: 'canonical_scripted', training_seed: null, held_out: false, available: true, checkpoint_sha256: null, unavailability_reason: null },
    { attacker_id: 'A1-203', display_name: 'A1-203 — Held-out learned attacker', classification: 'learned', architecture: 'adaptive_attacker_v1', training_seed: 203, held_out: true, available: true, checkpoint_sha256: 'c'.repeat(64), unavailability_reason: null },
  ],
};

const scenarios = {
  scenarios: scenarioLevels.map(clutter => ({
    clutter,
    display_name: { C0: 'Unobstructed', C1: 'Low clutter', C2: 'Medium clutter', C3: 'High clutter' }[clutter],
    perceptions: ['local', 'map_local'].map(id => ({
      id,
      description: id === 'local' ? 'Local sensing' : 'Known map plus local sensing',
      configurations: {
        measurement: { config_sha256: 'd'.repeat(64), parameters },
        kalman: { config_sha256: 'e'.repeat(64), parameters: { ...parameters, use_kalman_tracking: true } },
      },
    })),
  })),
};

const jobBase = {
  job_id: 'f'.repeat(32),
  request: {
    defender_policy_id: 'lr_ppo', defender_generation: 'd1', defender_training_seed: 77,
    attacker_condition: 'A1-203', clutter_level: 'C3', perception_mode: 'map_local',
    seed_start: 20, episode_count: 3, deterministic_inference: true,
  },
  requested_episode_count: 3, completed_count: 0, current_episode_seed: null,
  completed_replay_ids: [], error: null, cancellation_requested: false,
  created_at: '2026-10-16T10:00:00Z', started_at: null, updated_at: '2026-10-16T10:00:00Z', completed_at: null,
};

const replayId = '9'.repeat(32);
const replayIds = [replayId, '8'.repeat(32), '7'.repeat(32)];
const summary = {
  replay_id: replayId,
  schema_version: '1.0.0',
  simulation_seed: 20,
  defender: {
    policy_id: 'lr_ppo', architecture: 'lrppo_clutter_v1', generation: 'd1', training_seed: 77,
    checkpoint: { sha256: 'a'.repeat(64), role: 'final' },
  },
  attacker: { policy_id: 'A1-203', training_seed: 203, checkpoint: { sha256: 'c'.repeat(64), role: 'attacker_best' } },
  environment: { clutter: 'C3', awareness: 'map_local', config_sha256: 'd'.repeat(64), dt_s: 0.5 },
  summary: {
    outcome: 'enemy_static_obstacle_collision', steps: 21, duration_s: 10.5,
    detection_time_step: 4, detection_time_s: 2, interception_time_s: null,
    collision_type: 'enemy_static_obstacle_collision', total_reward: -3.4,
  },
  provenance: {
    simulation_source_commit: 'a'.repeat(40), visualizer_source_commit: 'b'.repeat(40),
    source_dirty: true, visualizer_dirty: false, registry_sha256: 'b'.repeat(64),
  },
};

async function mockCatalog(page: Page) {
  await page.route('**/api/health', route => route.fulfill({
    json: { service: 'counter-uas-visualizer', phase: 3, status: 'ok', data_root: 'configured', simulation_enabled: true, warnings: [] },
  }));
  await page.route('**/api/policies', route => route.fulfill({ json: policies }));
  await page.route('**/api/scenarios', route => route.fulfill({ json: scenarios }));
  await page.route('**/api/replays?limit=100', route => route.fulfill({ json: { replays: [], warnings: [] } }));
}

async function openAnalysis(page: Page) {
  await page.getByRole('button', { name: 'Analysis', exact: true }).click();
  const expand = page.getByRole('button', { name: 'Expand diagnostics panel' });
  if (await expand.count()) await expand.click();
}

test('renders the no-episode battlefield, supports camera reset, and keeps Phase 3 selectors', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  await mockCatalog(page);
  await page.goto('/');
  await expect(page.getByRole('heading', { name: 'No episode loaded' })).toBeVisible();
  const viewport = page.getByTestId('battlefield-viewport');
  await expect(viewport).toBeVisible();
  const canvas = page.locator('canvas');
  await expect(canvas).toBeVisible();
  expect(await canvas.evaluate(element => {
    if (!(element instanceof HTMLCanvasElement)) throw new Error('Expected a canvas element');
    const gl = element.getContext('webgl2');
    return Boolean(gl && !gl.isContextLost() && gl.getParameter(gl.VERSION));
  })).toBe(true);
  const cameraState = page.getByTestId('camera-state');
  await expect(cameraState).toHaveAttribute('data-camera-position', /,/);
  const initialPosition = (await cameraState.getAttribute('data-camera-position'))!.split(',').map(Number);
  const initial = await canvas.screenshot();
  const box = await canvas.boundingBox();
  if (!box) throw new Error('Canvas has no layout box');
  await page.mouse.move(box.x + 80, box.y + 80);
  await page.mouse.down();
  await page.mouse.move(box.x + 220, box.y + 120, { steps: 12 });
  await page.mouse.up();
  await expect(async () => expect((await canvas.screenshot()).equals(initial)).toBe(false)).toPass();
  await expect.poll(async () => {
    const current = (await cameraState.getAttribute('data-camera-position'))?.split(',').map(Number);
    return current ? Math.hypot(...current.map((value, index) => value - initialPosition[index])) : 0;
  }).toBeGreaterThan(1);
  await page.getByRole('button', { name: 'Reset camera' }).click();
  await expect.poll(async () => {
    const current = (await cameraState.getAttribute('data-camera-position'))?.split(',').map(Number);
    return current ? Math.hypot(...current.map((value, index) => value - initialPosition[index])) : Infinity;
  }).toBeLessThan(0.1);
  await expect(page.getByRole('button', { name: 'Simulation Viewer', exact: true })).toHaveAttribute('aria-pressed', 'true');
  await expect(page.getByRole('button', { name: 'Run Episode', exact: true })).toBeEnabled();
  await expect(page.getByRole('button', { name: 'Watch Recorded Episode' })).toBeVisible();
  await expect(page.getByLabel('Episode count', { exact: true })).toHaveValue('1');
  await expect(page.getByLabel('Attacker policy')).toHaveValue('A0');
  await expect(page.getByLabel('Clutter level')).toHaveValue('C0');
  await expect(page.getByRole('button', { name: 'Expand diagnostics panel' })).toBeVisible();
  await expect(page.getByLabel('Defender architecture')).toHaveValue('greedy');
  await expect(page.getByText('Defender speed: 18 m/s')).toBeVisible();
  await page.screenshot({ path: 'test-results/phase4-empty-battlefield.png', fullPage: true });
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(canvas).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.screenshot({ path: 'test-results/phase4-mobile.png', fullPage: true });
  expect(errors).toEqual([]);
});

async function scrubTimeline(page: Page, timeS: number) {
  await page.getByLabel('Episode timeline').evaluate((element, value) => {
    const input = element as HTMLInputElement;
    const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set;
    if (!setter) throw new Error('The range input has no native value setter.');
    setter.call(input, String(value));
    input.dispatchEvent(new Event('input', { bubbles: true }));
  }, timeS);
}

async function focusDefenderAndZoom(page: Page) {
  await page.getByRole('button', { name: 'Follow defender' }).click();
  await expect(page.getByRole('button', { name: 'Follow defender' })).toHaveAttribute('aria-pressed', 'true');
  await page.waitForTimeout(500);
  const viewport = page.getByTestId('battlefield-viewport');
  const box = await viewport.boundingBox();
  if (!box) throw new Error('Battlefield viewport has no layout box');
  const cameraState = page.getByTestId('camera-state');
  const distance = async () => {
    const position = (await cameraState.getAttribute('data-camera-position'))?.split(',').map(Number);
    const target = (await cameraState.getAttribute('data-camera-target'))?.split(',').map(Number);
    return position && target ? Math.hypot(...position.map((value, index) => value - target[index])) : Infinity;
  };
  const initialDistance = await distance();
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  for (let index = 0; index < 16; index += 1) await page.mouse.wheel(0, -120);
  await expect.poll(distance).toBeLessThan(initialDistance * 0.9);
}

test('plays real-schema frames, scrubs exactly, maps entities and obstacles, and switches episodes', async ({ page }) => {
  test.setTimeout(90_000);
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  await mockCatalog(page);
  const ids = [fixtureReplayId, 'e'.repeat(32)];
  await mockFixtureEpisodes(page, ids);
  await page.goto('/');
  await openAnalysis(page);
  await page.locator(`[data-replay-id="${fixtureReplayId}"]`).click();
  const rendered = page.getByTestId('rendered-replay-state');
  await expect(rendered).toHaveAttribute('data-step', '0');
  expect((await rendered.getAttribute('data-defender-position'))?.split(',').map(Number)).toEqual(
    simToThree(fixtureFrames[0].defender.position_m),
  );
  expect((await rendered.getAttribute('data-attacker-position'))?.split(',').map(Number)).toEqual(
    simToThree(fixtureFrames[0].attacker.position_m),
  );
  await expect(rendered).toHaveAttribute('data-static-obstacle-count', '1');
  await expect(rendered).toHaveAttribute('data-moving-obstacle-count', '1');
  await page.screenshot({ path: 'test-results/phase4-fixture-start.png', fullPage: true });

  await page.getByLabel('Playback speed').selectOption('8');
  await page.getByRole('button', { name: 'Play' }).click();
  await expect(rendered).toHaveAttribute('data-step', '2', { timeout: 3000 });
  await page.getByRole('button', { name: 'Restart episode' }).click();
  await expect(rendered).toHaveAttribute('data-step', '0');
  await scrubTimeline(page, 0.5);
  await expect(rendered).toHaveAttribute('data-step', '1');
  await expect(rendered).toHaveAttribute('data-interpolated', 'false');
  expect((await rendered.getAttribute('data-defender-position'))?.split(',').map(Number)).toEqual(
    simToThree(fixtureFrames[1].defender.position_m),
  );
  await expect(rendered).toHaveAttribute('data-visible-event-count', '1');
  await expect(page.getByText('Detected at 0.5 s')).toBeVisible();

  await page.getByRole('button', { name: 'Top-down' }).click();
  await expect(page.getByRole('button', { name: 'Top-down' })).toHaveAttribute('aria-pressed', 'true');
  await page.screenshot({ path: 'test-results/phase4-fixture-top-down.png', fullPage: true });
  await page.getByRole('button', { name: 'Return to 3D' }).click();
  await expect(page.getByRole('button', { name: 'Free 3D' })).toHaveAttribute('aria-pressed', 'true');

  await page.getByRole('checkbox', { name: 'Lead vector' }).uncheck();
  await page.getByRole('button', { name: 'Jump to end' }).click();
  await expect(rendered).toHaveAttribute('data-step', '2');
  await expect(page.getByText(/Terminal simulator outcome: intercepted/)).toBeVisible();
  await expect(page.getByText(/no policy decision is recorded for this selected terminal time/)).toBeVisible();
  await page.getByRole('button', { name: 'Next episode' }).click();
  await expect(page.getByText(/Episode 2 \/ 2 · seed 21/)).toBeVisible();
  await expect(rendered).toHaveAttribute('data-step', '0');
  await expect(page.getByRole('checkbox', { name: 'Lead vector' })).toBeChecked();
  await page.screenshot({ path: 'test-results/phase4-fixture-episode-switch.png', fullPage: true });
  expect(errors).toEqual([]);
});

test('synchronizes the decision inspector and charts, and masks privileged fields in Policy-Visible mode', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  await mockCatalog(page);
  await mockFixtureEpisodes(page, [fixtureReplayId]);
  await page.goto('/');
  await openAnalysis(page);
  await page.locator(`[data-replay-id="${fixtureReplayId}"]`).click();
  await expect(page.getByText(/command applies during step 0 → 1/)).toBeVisible();
  await expect(page.getByText('Raw PPO residual')).toBeVisible();
  await expect(page.getByText(/Unavailable: raw PPO output/)).toBeVisible();
  await expect(page.getByLabel('Lead vector')).toBeChecked();
  await expect(page.getByLabel('30° correction cone')).toBeChecked();

  await scrubTimeline(page, 0.5);
  await expect(page.getByText(/command applies during step 1 → 2/)).toBeVisible();
  await expect(page.getByText(/Native 3D Lead → requested angle/).first()).toBeVisible();
  const chart = page.getByRole('img', { name: 'LR-PPO native 3D Lead → requested angle against simulation time' });
  await expect(chart).toBeVisible();
  await chart.scrollIntoViewIfNeeded();
  const chartBox = await chart.boundingBox();
  if (!chartBox) throw new Error('Diagnostic chart has no layout box');
  const fraction = 0.75;
  await page.mouse.click(
    chartBox.x + ((35 + fraction * 760) / 830) * chartBox.width,
    chartBox.y + chartBox.height * 0.5,
  );
  await expect(page.getByLabel('Episode timeline')).toHaveValue('0.75');
  await expect(page.getByTestId('rendered-replay-state')).toHaveAttribute('data-step', '1');
  const selectedTimeS = Number.parseFloat((await page.locator('.inspector-field').filter({ hasText: 'Playback state time' }).locator('dd').textContent()) ?? '');
  const cursorX = Number(await chart.locator('.chart-cursor').getAttribute('x1'));
  expect(selectedTimeS).toBeGreaterThan(0.7);
  expect(selectedTimeS).toBeLessThan(0.8);
  expect(cursorX).toBeCloseTo(35 + selectedTimeS * 760, 0);

  await page.getByRole('button', { name: 'Policy-Visible' }).click();
  const scene = page.getByTestId('rendered-replay-state');
  await expect(scene).toHaveAttribute('data-attacker-position', 'unavailable');
  await expect(scene).toHaveAttribute('data-visible-event-count', '0');
  await expect(page.getByText(/privileged separation is hidden/).first()).toBeVisible();
  await expect(page.getByText(/reward is not part of the policy-visible scene/).first()).toBeVisible();
  await expect(page.getByRole('button', { name: 'Follow attacker' })).toBeDisabled();
  await expect(page.locator('.chart-event')).toHaveCount(0);
  await expect(page.getByText(/Researcher-only episode outcomes and provenance remain hidden/)).toBeVisible();
  await page.getByRole('button', { name: 'Collapse diagnostics panel' }).click();
  await expect(page.getByRole('button', { name: 'Expand diagnostics panel' })).toBeVisible();
  await page.getByRole('button', { name: 'Expand diagnostics panel' }).click();
  await expect(page.getByRole('button', { name: 'Collapse diagnostics panel' })).toBeVisible();
  expect(errors).toEqual([]);
});

test('keeps Kalman estimates separate from privileged truth and masks estimates before detection', async ({ page }) => {
  await mockCatalog(page);
  await mockFixtureEpisodes(page, [fixtureReplayId], kalmanFixtureEpisode);
  await page.goto('/');
    await openAnalysis(page);
    await page.locator(`[data-replay-id="${fixtureReplayId}"]`).click();
  await expect(page.getByText(/Policy-visible target estimate unavailable before detection/)).toBeVisible();
  await expect(page.getByText(/Derived target position tracking error/)).toBeVisible();

  await scrubTimeline(page, 0.5);
  await expect(page.getByText('Kalman estimate position')).toBeVisible();
  await expect(page.getByText('[9.4, 2, 13] m')).toBeVisible();
  await expect(page.getByText('Privileged ground-truth position')).toBeVisible();
  await expect(page.getByText('[9, 2, 13] m')).toBeVisible();
  await expect(page.getByText(/Derived Euclidean tracking error/)).toBeVisible();
  await expect(page.getByText('0.40 m')).toBeVisible();

  await page.getByRole('button', { name: 'Policy-Visible' }).click();
  await expect(page.getByText(/never supplied to the policy/).first()).toBeVisible();
  await expect(page.getByLabel('Ground-truth marker')).toHaveCount(0);
  await expect(page.getByTestId('rendered-replay-state')).toHaveAttribute('data-attacker-position', 'unavailable');
});

test('submits a compatible learned selection and shows actual job progress and results', async ({ page }) => {
  await mockCatalog(page);
  let postBody: Record<string, unknown> | null = null;
  let statusPolls = 0;
  await page.route('**/api/replay-jobs', async (route: Route) => {
    postBody = route.request().postDataJSON() as Record<string, unknown>;
    await route.fulfill({ status: 202, json: { ...jobBase, state: 'queued' } });
  });
  await page.route(`**/api/replay-jobs/${jobBase.job_id}`, async route => {
    statusPolls += 1;
    if (statusPolls === 1) {
      await route.fulfill({
        json: {
          ...jobBase, state: 'running', completed_count: 1, current_episode_seed: 21,
          started_at: jobBase.created_at, updated_at: '2026-10-16T10:00:01Z',
        },
      });
    } else {
      await route.fulfill({
        json: {
          ...jobBase, state: 'completed', completed_count: 3, current_episode_seed: null,
          completed_replay_ids: replayIds, started_at: jobBase.created_at,
          completed_at: '2026-10-16T10:00:03Z', updated_at: '2026-10-16T10:00:03Z',
        },
      });
    }
  });
  await page.route('**/api/replays/*', async route => {
    if (new URL(route.request().url()).pathname.endsWith('/summary')) return route.fallback();
    await route.fulfill({ json: fixtureBundle });
  });
  await page.route('**/api/replays/*/summary', route => {
    const id = new URL(route.request().url()).pathname.split('/').at(-2) ?? replayId;
    const index = replayIds.indexOf(id);
    const outcomes = ['enemy_static_obstacle_collision', 'soldier_caught', 'intercepted'];
    const item = {
      ...summary,
      replay_id: id,
      simulation_seed: 20 + Math.max(index, 0),
      summary: {
        ...summary.summary,
        outcome: outcomes[Math.max(index, 0)],
        collision_type: index === 0 ? 'enemy_static_obstacle_collision' : null,
        duration_s: 10.5 + Math.max(index, 0),
        total_reward: -3.4 + Math.max(index, 0),
      },
    };
    return route.fulfill({ json: item });
  });

  await page.goto('/');
  await openAnalysis(page);
  await page.getByLabel('Defender architecture').selectOption('lr_ppo');
  await page.getByLabel('Defender generation').selectOption('d1');
  await page.getByLabel('Defender training seed').selectOption('77');
  await page.getByLabel('Attacker policy').selectOption('A1-203');
  await page.getByLabel('Clutter level').selectOption('C3');
  await page.getByLabel('Episode count').selectOption('3');
  await page.getByLabel('Starting evaluation seed').fill('20');
  await page.getByRole('button', { name: 'Run 3 Episodes' }).click();
  await expect.poll(() => postBody).toEqual({
    defender_policy_id: 'lr_ppo', defender_generation: 'd1', defender_training_seed: 77,
    attacker_condition: 'A1-203', clutter_level: 'C3', perception_mode: 'map_local',
    seed_start: 20, episode_count: 3, deterministic_inference: true,
  });
  await expect(page.getByText('Episode 2 of 3 running — 1 completed / seed 21')).toBeVisible();
  await expect(page.getByRole('heading', { name: 'enemy_static_obstacle_collision — Attacker / static obstacle collision' })).toBeVisible({ timeout: 10000 });
  await expect(page.getByText('Obstacle collisions: 1')).toBeVisible();
  await expect(page.getByText('Attacker reached asset: 1')).toBeVisible();
  await expect(page.getByText('Source dirty flag: true')).toBeVisible();
  await expect(page.getByTestId('rendered-replay-state')).toBeAttached();
  await expect(page.getByLabel('Current simulation job')).toContainText('3 of 3 episodes completed');
  await page.screenshot({ path: 'test-results/phase4-results.png', fullPage: true });
});

test('shows actionable submission and service errors without invented results', async ({ page }) => {
  await mockCatalog(page);
  await page.route('**/api/replay-jobs', route => route.fulfill({
    status: 409,
    json: { error: { code: 'REPLAY_JOB_BUSY', message: 'Another replay batch is running; retry when it finishes.' } },
  }));
  await page.goto('/');
  await page.getByRole('button', { name: 'Run Episode', exact: true }).click();
  await expect(page.getByRole('alert')).toContainText('Another replay batch is running');
  await openAnalysis(page);
  await expect(page.getByText('No completed replay episodes are available yet.')).toBeVisible();
});

test('backend outage is visible without inventing episode results', async ({ page }) => {
  await page.route('**/api/health', route => route.abort());
  await page.goto('/');
  await expect(page.getByRole('status')).toContainText('Backend unavailable');
  await expect(page.getByRole('heading', { name: 'No episode loaded' })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Run Episode', exact: true })).toBeDisabled();
  await openAnalysis(page);
  await expect(page.getByText('No completed replay episodes are available yet.')).toBeVisible();
});

test('unconfigured research data root is visible', async ({ page }) => {
  await page.route('**/api/health', route => route.fulfill({
    json: { service: 'counter-uas-visualizer', phase: 3, status: 'degraded', data_root: 'not_configured', simulation_enabled: false },
  }));
  await page.goto('/');
  await expect(page.getByRole('status')).toContainText('research data root not configured');
});

test('reports WebGL loss and clears the warning after actual context restoration', async ({ page }) => {
  await page.route('**/api/health', route => route.fulfill({
    json: { service: 'counter-uas-visualizer', phase: 3, status: 'ok', data_root: 'configured', simulation_enabled: true },
  }));
  await page.route('**/api/policies', route => route.fulfill({ json: policies }));
  await page.route('**/api/scenarios', route => route.fulfill({ json: scenarios }));
  await page.route('**/api/replays?limit=100', route => route.fulfill({ json: { replays: [], warnings: [] } }));
  await page.goto('/');
  await expect(page.getByTestId('battlefield-viewport')).toBeVisible();
  const extension = await page.locator('canvas').evaluateHandle(element => {
    if (!(element instanceof HTMLCanvasElement)) throw new Error('Expected canvas');
    const extension = element.getContext('webgl2')?.getExtension('WEBGL_lose_context');
    if (!extension) throw new Error('Browser cannot test WebGL context loss');
    return extension;
  });
  await extension.evaluate(value => value.loseContext());
  await expect(page.locator('.inline-error[role="alert"]')).toContainText('WebGL context was lost');
  await expect(page.getByRole('button', { name: 'Reset camera' })).toBeEnabled();
  await extension.evaluate(value => value.restoreContext());
  await expect(page.locator('.inline-error[role="alert"]')).toHaveCount(0);
  await expect(page.getByTestId('battlefield-viewport')).toBeVisible();
  await extension.dispose();
});

test('loads and scrubs all four existing real replay cases, including the 974-frame trace', async ({ page }) => {
  test.skip(process.env.VISUALIZER_ENABLE_REAL_REPLAY_TESTS !== '1', 'Set VISUALIZER_ENABLE_REAL_REPLAY_TESTS=1 with the Phase 3 API running.');
  test.setTimeout(120_000);
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  const replayListResponse = page.waitForResponse(response => response.url().includes('/api/replays?limit=100'));
  await page.goto('/');
  const listResponse = await replayListResponse;
  await openAnalysis(page);
  expect(listResponse.ok()).toBe(true);
  const replayList = await listResponse.json() as { replays: { replay_id: string }[] };
  expect(replayList.replays.some(item => item.replay_id === '32b7d3bf27c755da0d73821c9c495603')).toBe(true);
  await expect(page.getByTestId('episode-result').first()).toBeVisible({ timeout: 15000 });
  const cases = [
    { id: '02b97c96328d9bdc772f791ec21e6d32', frames: 282, static: 0, moving: 0 },
    { id: '6f36d9fdab26b18a06036496d1e6e74d', frames: 170, static: 8, moving: 2 },
    { id: '2edecf334ae265f70e1981c3e872b462', frames: 330, static: 15, moving: 4 },
    { id: '32b7d3bf27c755da0d73821c9c495603', frames: 974, static: 15, moving: 4 },
  ];
  const rendered = page.getByTestId('rendered-replay-state');
  for (const [index, replayCase] of cases.entries()) {
    const response = await page.request.get(`http://127.0.0.1:8000/api/replays/${replayCase.id}`);
    expect(response.ok()).toBe(true);
    const bundle = await response.json() as ReplayBundle;
    const episode = bundle.episodes[0];
    expect(episode.frames).toHaveLength(replayCase.frames);
    await page.locator(`[data-replay-id="${replayCase.id}"]`).click();
    await expect(rendered).toHaveAttribute('data-step', '0', { timeout: 15000 });
    expect((await rendered.getAttribute('data-defender-position'))?.split(',').map(Number)).toEqual(
      simToThree(episode.frames[0].defender.position_m),
    );
    expect((await rendered.getAttribute('data-attacker-position'))?.split(',').map(Number)).toEqual(
      simToThree(episode.frames[0].attacker.position_m),
    );
    expect((await rendered.getAttribute('data-asset-position'))?.split(',').map(Number)).toEqual(
      simToThree(episode.frames[0].asset.position_m),
    );
    await expect(rendered).toHaveAttribute('data-static-obstacle-count', String(replayCase.static));
    await expect(rendered).toHaveAttribute('data-moving-obstacle-count', String(replayCase.moving));
    if (index === 0) await page.screenshot({ path: 'test-results/phase4-real-beginning.png', fullPage: true });
    if (index === 2) await page.screenshot({ path: 'test-results/phase4-real-obstacles.png', fullPage: true });
    if (index === 3) {
      const middleTime = episode.summary.duration_s / 2;
      await scrubTimeline(page, middleTime);
      await expect(rendered).toHaveAttribute('data-interpolated', 'true');
      await expect(rendered).toHaveAttribute('data-step', '486');
      await page.screenshot({ path: 'test-results/phase4-real-mid-trails.png', fullPage: true });
      await page.getByRole('button', { name: 'Top-down' }).click();
      await expect(page.getByRole('button', { name: 'Top-down' })).toHaveAttribute('aria-pressed', 'true');
      await expect.poll(() => page.getByTestId('camera-state').getAttribute('data-camera-up')).toBe('0,0,-1');
      await expect.poll(async () => {
        const forward = await page.getByTestId('camera-state').getAttribute('data-camera-forward');
        if (!forward) return 1;
        const [x, y, z] = forward.split(',').map(Number);
        return Math.max(Math.abs(x), Math.abs(y + 1), Math.abs(z));
      }).toBeLessThan(0.001);
      const cameraPosition = (await page.getByTestId('camera-state').getAttribute('data-camera-position'))!
        .split(',').map(Number);
      const cameraTarget = (await page.getByTestId('camera-state').getAttribute('data-camera-target'))!
        .split(',').map(Number);
      expect(cameraPosition[1]).toBeGreaterThan(cameraTarget[1]);
      await page.screenshot({ path: 'test-results/phase4-real-top-down.png', fullPage: true });
      await page.getByRole('button', { name: 'Jump to end' }).click();
      await expect(rendered).toHaveAttribute('data-step', '973');
      await expect(page.getByText(/Terminal simulator outcome: intercepted/)).toBeVisible();
      await page.screenshot({ path: 'test-results/phase4-real-terminal.png', fullPage: true });
      await page.screenshot({ path: 'test-results/phase4-real-episode-list.png', fullPage: true });
    }
  }
  expect(errors).toEqual([]);
});

test('captures Phase 5 diagnostics from the four existing local replay artifacts', async ({ page }) => {
  test.skip(process.env.VISUALIZER_ENABLE_LOCAL_REPLAY_SCREENSHOTS !== '1', 'Set VISUALIZER_ENABLE_LOCAL_REPLAY_SCREENSHOTS=1 to load the four existing gzip replay artifacts.');
  test.setTimeout(300_000);
  const artifactCases = [
    { id: '02b97c96328d9bdc772f791ec21e6d32', file: 'phase2-final-greedy-a0-c0.json.gz', frames: 282 },
    { id: '6f36d9fdab26b18a06036496d1e6e74d', file: 'phase2-final-lrppo-d0-a0-c2.json.gz', frames: 170 },
    { id: '2edecf334ae265f70e1981c3e872b462', file: 'phase2-final-ppo-kalman-d0-a1-203-c3.json.gz', frames: 330 },
    { id: '32b7d3bf27c755da0d73821c9c495603', file: 'phase2-final-lrppo-d1-a1-203-c3.json.gz', frames: 974 },
  ];
  const bundles = await Promise.all(artifactCases.map(async item => {
    const url = new URL(`../../data/local/${item.file}`, import.meta.url);
    const compressed = await readFile(url);
    return JSON.parse(gunzipSync(compressed).toString('utf8')) as ReplayBundle;
  }));
  const replayRecords = artifactCases.map((item, index) => {
    const bundle = bundles[index];
    const episode = bundle.episodes[0];
    const checkpoint = episode.defender.checkpoint;
    const attackerCheckpoint = episode.attacker.checkpoint;
    return {
      replay_id: item.id,
      simulation_seed: episode.simulation_seed,
      defender: {
        policy_id: episode.defender.policy_id,
        architecture: episode.defender.architecture,
        generation: episode.defender.generation,
        training_seed: episode.defender.training_seed,
        checkpoint: checkpoint ? { sha256: checkpoint.sha256, role: checkpoint.role } : null,
      },
      attacker: {
        policy_id: episode.attacker.policy_id,
        checkpoint: attackerCheckpoint ? { sha256: attackerCheckpoint.sha256, role: attackerCheckpoint.role } : null,
      },
      clutter: episode.environment.clutter,
      perception: episode.environment.awareness,
      outcome: episode.summary.outcome,
      steps: episode.summary.steps,
      duration_s: episode.summary.duration_s,
      detection_time_s: episode.summary.detection_time_s,
      total_reward: episode.summary.total_reward,
      collision_type: episode.summary.collision_type,
      interception_time_s: episode.summary.interception_time_s,
      provenance: {
        simulation_source_commit: bundle.provenance.simulation_source_commit,
        visualizer_source_commit: bundle.provenance.visualizer_source_commit,
        source_dirty: bundle.provenance.source_dirty,
        visualizer_dirty: bundle.provenance.visualizer_dirty,
        registry_sha256: bundle.provenance.registry_sha256,
      },
    };
  });

  await page.route('**/api/health', route => route.fulfill({
    json: { service: 'counter-uas-visualizer', phase: 3, status: 'ok', data_root: 'configured', simulation_enabled: true, warnings: [] },
  }));
  await page.route('**/api/policies', route => route.fulfill({ json: policies }));
  await page.route('**/api/scenarios', route => route.fulfill({ json: scenarios }));
  await page.route('**/api/replays?limit=100', route => route.fulfill({ json: { replays: replayRecords, warnings: [] } }));
  await page.route('**/api/replays/*', route => {
    const path = new URL(route.request().url()).pathname;
    const id = path.split('/').at(-1) ?? '';
    const index = artifactCases.findIndex(item => item.id === id);
    return index < 0
      ? route.fulfill({ status: 404, json: { error: { code: 'NOT_FOUND', message: 'Unknown local replay.' } } })
      : route.fulfill({ json: bundles[index] });
  });
  await page.route('**/api/replays/*/summary', route => {
    const path = new URL(route.request().url()).pathname;
    const id = path.split('/').at(-2) ?? '';
    const index = artifactCases.findIndex(item => item.id === id);
    if (index < 0) return route.fulfill({ status: 404, json: { error: { code: 'NOT_FOUND', message: 'Unknown local replay.' } } });
    const item = replayRecords[index];
    const bundle = bundles[index];
    const episode = bundle.episodes[0];
    return route.fulfill({ json: {
      replay_id: item.replay_id,
      schema_version: bundle.schema_version,
      simulation_seed: item.simulation_seed,
      defender: item.defender,
      attacker: { policy_id: episode.attacker.policy_id, training_seed: episode.attacker.training_seed, checkpoint: episode.attacker.checkpoint },
      environment: { clutter: item.clutter, awareness: item.perception, config_sha256: episode.environment.config_sha256, dt_s: episode.environment.dt_s },
      summary: {
        outcome: item.outcome,
        steps: item.steps,
        duration_s: item.duration_s,
        detection_time_step: episode.summary.detection_time_step,
        detection_time_s: item.detection_time_s,
        interception_time_s: item.interception_time_s,
        collision_type: item.collision_type,
        total_reward: item.total_reward,
      },
      provenance: item.provenance,
    } });
  });

  await page.goto('/');
  const rendered = page.getByTestId('rendered-replay-state');
  const greedy = artifactCases[0];
  expect(bundles[0].episodes[0].frames).toHaveLength(greedy.frames);
  await page.locator(`[data-replay-id="${greedy.id}"]`).click();
  await expect(rendered).toHaveAttribute('data-step', '0');
  await expect(page.getByRole('heading', { name: 'LR-PPO native 3D correction' })).toHaveCount(0);
  await focusDefenderAndZoom(page);
  await page.locator('.battlefield').screenshot({ path: 'test-results/phase5-greedy-no-lrppo-overlays.png' });

  const lrppoD0 = artifactCases[1];
  expect(bundles[1].episodes[0].frames).toHaveLength(lrppoD0.frames);
  await page.locator(`[data-replay-id="${lrppoD0.id}"]`).click();
  await expect(page.getByRole('heading', { name: 'LR-PPO native 3D correction' })).toBeVisible();
  await scrubTimeline(page, Math.min(2, bundles[1].episodes[0].summary.duration_s));
  await expect(rendered).toHaveAttribute('data-step', /[1-9]/);

  const kalman = artifactCases[2];
  const kalmanEpisode = bundles[2].episodes[0];
  expect(kalmanEpisode.frames).toHaveLength(kalman.frames);
  await page.locator(`[data-replay-id="${kalman.id}"]`).click();
  const kalmanTime = Math.min(kalmanEpisode.summary.duration_s, (kalmanEpisode.summary.detection_time_s ?? 1) + 0.5);
  await scrubTimeline(page, kalmanTime);
  await expect(page.getByRole('heading', { name: /Policy-visible target \/ privileged truth/ })).toBeVisible();
  await focusDefenderAndZoom(page);
  await page.locator('.battlefield').screenshot({ path: 'test-results/phase5-ppo-kalman-estimate-truth.png' });
  await page.locator('.policy-inspector').screenshot({ path: 'test-results/phase5-paused-policy-inspector.png' });

  const lrppoD1 = artifactCases[3];
  const longEpisode = bundles[3].episodes[0];
  expect(longEpisode.frames).toHaveLength(974);
  const decisionFrame = longEpisode.frames.find(frame =>
    frame.telemetry?.lrppo && frame.telemetry.decision_policy_visible.enemy_detected
      && frame.telemetry.controller_action_executed);
  if (!decisionFrame?.telemetry) throw new Error('The real LR-PPO trace has no detectable applied decision interval.');
  const selectedDecisionTime = longEpisode.frames[decisionFrame.telemetry.decision_step].time_s;
  await page.locator(`[data-replay-id="${lrppoD1.id}"]`).click();
  await scrubTimeline(page, selectedDecisionTime);
  await expect(page.getByRole('heading', { name: 'LR-PPO native 3D correction' })).toBeVisible();
  await expect(page.getByText(/Native 3D Lead → requested angle/).first()).toBeVisible();
  await expect(page.getByRole('checkbox', { name: 'Lead vector' })).toBeChecked();
  await expect(page.getByRole('checkbox', { name: 'Requested direction' })).toBeChecked();
  await expect(page.getByRole('checkbox', { name: 'Actual velocity' })).toBeChecked();
  await expect(page.getByRole('checkbox', { name: '30° correction cone' })).toBeChecked();
  await focusDefenderAndZoom(page);
  await page.locator('.battlefield').screenshot({ path: 'test-results/phase5-lrppo-three-vectors-cone.png' });
  const angleChart = page.getByRole('img', { name: 'LR-PPO native 3D Lead → requested angle against simulation time' });
  await angleChart.scrollIntoViewIfNeeded();
  await angleChart.locator('xpath=..').screenshot({ path: 'test-results/phase5-lrppo-angle-chart.png' });
  await scrubTimeline(page, 0);
  await expect(rendered).toHaveAttribute('data-step', '0');
  await scrubTimeline(page, longEpisode.summary.duration_s);
  await expect(rendered).toHaveAttribute('data-step', '973');
  await expect(page.getByText(/no policy decision is recorded for this selected terminal time\/state/i)).toBeVisible();
  await scrubTimeline(page, selectedDecisionTime);
  await page.getByRole('button', { name: 'Policy-Visible' }).click();
  await expect(rendered).toHaveAttribute('data-attacker-position', 'unavailable');
  await expect(page.getByText('Attacker UAV · privileged ground truth')).toHaveCount(0);
  await expect(page.getByTestId('attacker-truth-label')).toHaveCount(0);
  await page.locator('.battlefield').screenshot({ path: 'test-results/phase5-policy-visible-mode.png' });
});

test('verifies synchronized comparison against live API replays in Chromium', async ({ page }) => {
  test.skip(process.env.VISUALIZER_ENABLE_LIVE_COMPARISON_TESTS !== '1', 'Set VISUALIZER_ENABLE_LIVE_COMPARISON_TESTS=1 to verify two validated real API replays.');
  test.setTimeout(240_000);
  const pageErrors: string[] = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  page.on('response', response => {
    if (response.url().startsWith('http://127.0.0.1:5173/api/') && response.status() >= 500) {
      pageErrors.push(`API HTTP ${response.status()} ${response.url()}`);
    }
  });
  await page.goto('/');
  await page.getByRole('button', { name: 'Compare policies' }).click();
  const pairStatus = page.locator('.comparison-pair .pair-status');
  await expect(pairStatus).toHaveText(/exploratory|matched/, { timeout: 60_000 });
  await expect(page.getByTestId('rendered-replay-state-A')).toBeVisible();
  await expect(page.getByTestId('rendered-replay-state-B')).toBeVisible();
  await expect(page.locator('.comparison-scenes canvas')).toHaveCount(2);
  const livePair = await page.evaluate(async () => {
    const listResponse = await fetch('/api/replays?limit=100');
    if (!listResponse.ok) throw new Error(`Live replay index returned HTTP ${listResponse.status}`);
    const listing = await listResponse.json() as { replays: Array<{ replay_id: string; defender: { policy_id: string; generation: string; training_seed: number }; simulation_seed: number; attacker: { policy_id: string }; clutter: string; perception: string }> };
    const find = (policy: string, generation: string, trainingSeed: number) => listing.replays.find(item =>
      item.defender.policy_id === policy && item.defender.generation === generation
      && item.defender.training_seed === trainingSeed && item.simulation_seed === 0
      && item.attacker.policy_id === 'A1-203' && item.clutter === 'C3' && item.perception === 'map_local');
    const left = find('lr_ppo', 'd1', 77);
    const right = find('ppo_kalman', 'd0', 74);
    if (!left || !right) throw new Error('The live API does not contain the previously validated seed-0 replay pair.');
    const [leftResponse, rightResponse] = await Promise.all([
      fetch(`/api/replays/${left.replay_id}`),
      fetch(`/api/replays/${right.replay_id}`),
    ]);
    if (!leftResponse.ok || !rightResponse.ok) throw new Error('The live API could not load both full replay bundles.');
    const [a, b] = await Promise.all([leftResponse.json(), rightResponse.json()]) as Array<{ episodes: Array<{ frames: Array<{ defender: { position_m: number[] } }> }> }>;
    return { a: a.episodes[0], b: b.episodes[0] };
  });
  expect(await page.getByTestId('rendered-replay-state-A').getAttribute('data-defender-position'))
    .toBe(livePair.a.frames[0].defender.position_m.join(','));
  expect(await page.getByTestId('rendered-replay-state-B').getAttribute('data-defender-position'))
    .toBe(livePair.b.frames[0].defender.position_m.join(','));
  await page.screenshot({ path: 'test-results/phase6-real-lrppo-d1-vs-ppo-kalman-d0.png', fullPage: true });

  const stateA = page.getByTestId('rendered-replay-state-A');
  const stateB = page.getByTestId('rendered-replay-state-B');
  await page.getByRole('button', { name: 'Step forward' }).click();
  await expect(stateA).toHaveAttribute('data-step', '1');
  await expect(stateB).toHaveAttribute('data-step', '1');
  await page.getByRole('button', { name: 'Play', exact: true }).click();
  await expect.poll(async () => Number(await stateA.getAttribute('data-step'))).toBeGreaterThan(1);
  await expect.poll(async () => Number(await stateB.getAttribute('data-step'))).toBeGreaterThan(1);
  await page.getByRole('button', { name: 'Pause', exact: true }).click();
  await page.getByRole('button', { name: 'End', exact: true }).click();
  const lengths = await page.evaluate(async () => {
    const ids = ['32b7d3bf27c755da0d73821c9c495603', '2edecf334ae265f70e1981c3e872b462'];
    return Promise.all(ids.map(async id => {
      const response = await fetch(`/api/replays/${id}`);
      if (!response.ok) throw new Error(`Could not load replay ${id}`);
      const data = await response.json() as { episodes: Array<{ frames: unknown[] }> };
      return data.episodes[0].frames.length - 1;
    }));
  });
  await expect(stateA).toHaveAttribute('data-step', String(lengths[0]));
  await expect(stateB).toHaveAttribute('data-step', String(lengths[1]));
  await expect(page.getByText(/Episode ended · frozen at/)).toHaveCount(2);
  await page.screenshot({ path: 'test-results/phase6-real-different-terminal-outcomes.png', fullPage: true });

  await page.getByLabel('Alignment').selectOption('detection');
  await expect(page.getByRole('heading', { name: 'Seconds relative to first detection' })).toBeVisible();
  await page.screenshot({ path: 'test-results/phase6-real-detection-aligned.png', fullPage: true });
  await page.getByRole('button', { name: /top down/i }).click();
  await expect(page.getByTestId('camera-state-A')).toHaveAttribute('data-camera-position', /,/);
  await expect(page.getByTestId('camera-state-B')).toHaveAttribute('data-camera-position', /,/);
  await page.screenshot({ path: 'test-results/phase6-real-linked-top-down.png', fullPage: true });
  await page.getByRole('checkbox', { name: /Link free \/ top-down cameras/ }).uncheck();
  await page.getByRole('checkbox', { name: /Link free \/ top-down cameras/ }).check();

  await page.getByRole('button', { name: 'Policy-Visible', exact: true }).click();
  await expect(stateA).toHaveAttribute('data-attacker-position', 'unavailable');
  await expect(stateB).toHaveAttribute('data-attacker-position', 'unavailable');
  await expect(page.getByText(/Canonical outcome:/)).toHaveCount(0);
  await expect(page.locator('.comparison-scenes > div').nth(0).getByTestId('guidance-legend')).toHaveCount(1);
  await expect(page.locator('.comparison-scenes > div').nth(1).getByTestId('guidance-legend')).toHaveCount(0);
  await expect.poll(() => pageErrors).toEqual([]);
});

test('generates only missing requested comparison replays sequentially through the live job API', async ({ page }) => {
  test.skip(process.env.VISUALIZER_ENABLE_LIVE_COMPARISON_GENERATION !== '1', 'Set VISUALIZER_ENABLE_LIVE_COMPARISON_GENERATION=1 to run the three minimum missing seed-0 real rollouts.');
  test.setTimeout(540_000);
  await page.goto('/');
  await page.getByRole('button', { name: 'Compare policies' }).click();
  await expect(page.locator('.comparison-pair')).toBeVisible();
  const choices = page.locator('.comparison-policy-choice');
  const choose = async (side: number, policy: string, generation?: string, trainingSeed?: number) => {
    const selector = choices.nth(side);
    await selector.getByLabel('Controller family').selectOption(policy);
    if (generation) {
      await selector.getByLabel('Generation').selectOption(generation);
      if (trainingSeed !== undefined) await selector.getByLabel('Training seed').selectOption(String(trainingSeed));
    }
  };
  const generateOneMissing = async (screenshot: string) => {
    await expect(page.getByText('1 rollout(s) are missing from the replay index; valid existing replays will be reused.')).toBeVisible();
    await page.getByRole('button', { name: 'Generate missing replays' }).click();
    await expect(page.getByRole('button', { name: 'Generating sequentially…' })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Generate missing replays' })).toBeDisabled({ timeout: 420_000 });
    await expect(page.locator('.comparison-pair .pair-status')).toHaveText(/exploratory|matched/, { timeout: 90_000 });
    await expect(page.locator('.comparison-scenes canvas')).toHaveCount(2);
    await page.screenshot({ path: `test-results/${screenshot}`, fullPage: true });
  };

  await choose(0, 'lr_ppo', 'd0', 77);
  await choose(1, 'lr_ppo', 'd1', 77);
  await generateOneMissing('phase6-real-lrppo-d0-vs-d1.png');

  await choose(0, 'lr_ppo', 'd1', 77);
  await choose(1, 'ppo_kalman', 'd1', 74);
  await generateOneMissing('phase6-real-lrppo-d1-vs-ppo-kalman-d1.png');

  await choose(0, 'lead');
  await choose(1, 'lr_ppo', 'd1', 77);
  await generateOneMissing('phase6-real-lead-vs-lrppo-d1.png');
  await page.screenshot({ path: 'test-results/phase6-real-paired-seed-list.png', fullPage: true });
});

test('captures requested side-by-side screenshots from validated real replay pairs', async ({ page }) => {
  test.skip(process.env.VISUALIZER_ENABLE_COMPARISON_SCREENSHOTS !== '1', 'Set VISUALIZER_ENABLE_COMPARISON_SCREENSHOTS=1 to capture validated local replay comparisons.');
  test.setTimeout(300_000);
  await page.goto('/');
  await page.getByRole('button', { name: 'Compare policies' }).click();
  const choices = page.locator('.comparison-policy-choice');
  const choose = async (side: number, policy: string, generation?: string, trainingSeed?: number) => {
    const selector = choices.nth(side);
    await selector.getByLabel('Controller family').selectOption(policy);
    if (generation) {
      await selector.getByLabel('Generation').selectOption(generation);
      if (trainingSeed !== undefined) await selector.getByLabel('Training seed').selectOption(String(trainingSeed));
    }
  };
  const waitForPair = async (left: string, right: string) => {
    await expect(page.locator('.comparison-pair .pair-status')).toHaveText(/exploratory|matched/, { timeout: 90_000 });
    await expect(page.locator('.comparison-scenes canvas')).toHaveCount(2);
    await expect(page.locator('.comparison-pane-title').nth(0)).toContainText(new RegExp(left, 'i'), { timeout: 90_000 });
    await expect(page.locator('.comparison-pane-title').nth(1)).toContainText(new RegExp(right, 'i'), { timeout: 90_000 });
  };

  await choose(0, 'lr_ppo', 'd0', 77);
  await choose(1, 'lr_ppo', 'd1', 77);
  await waitForPair('d0', 'd1');
  await page.screenshot({ path: 'test-results/phase6-real-lrppo-d0-vs-d1.png', fullPage: true });

  await choose(0, 'lr_ppo', 'd1', 77);
  await choose(1, 'ppo_kalman', 'd1', 74);
  await waitForPair('d1', 'ppo_kalman');
  await page.screenshot({ path: 'test-results/phase6-real-lrppo-d1-vs-ppo-kalman-d1.png', fullPage: true });
  await page.screenshot({ path: 'test-results/phase6-real-synchronized-timeline-and-charts.png', fullPage: true });
  await page.getByLabel('Alignment').selectOption('detection');
  await expect(page.getByRole('heading', { name: 'Seconds relative to first detection' })).toBeVisible();
  await page.screenshot({ path: 'test-results/phase6-real-detection-aligned.png', fullPage: true });
  await page.getByRole('button', { name: /top down/i }).click();
  await page.screenshot({ path: 'test-results/phase6-real-linked-top-down.png', fullPage: true });

  await page.getByLabel('Alignment').selectOption('absolute');
  await choose(0, 'lead');
  await choose(1, 'lr_ppo', 'd1', 77);
  await waitForPair('lead', 'd1');
  await page.screenshot({ path: 'test-results/phase6-real-lead-vs-lrppo-d1.png', fullPage: true });

  await choose(0, 'lr_ppo', 'd0', 77);
  await choose(1, 'lr_ppo', 'd1', 77);
  await waitForPair('d0', 'd1');
  await page.getByRole('button', { name: 'End', exact: true }).click();
  await expect(page.getByText(/Episode ended · frozen at/)).toHaveCount(2);
  await page.screenshot({ path: 'test-results/phase6-real-different-terminal-outcomes.png', fullPage: true });
  const seedList = page.locator('.comparison-seed-list');
  await seedList.scrollIntoViewIfNeeded();
  await seedList.screenshot({ path: 'test-results/phase6-real-paired-episode-list.png' });
});
