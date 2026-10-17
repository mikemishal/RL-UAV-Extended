import { expect, test, type Page, type Route } from '@playwright/test';
import type { ReplayBundle, ReplayEntityState, ReplayEpisode, ReplayFrame } from '../src/api';
import { sampleAtTime, scenarioBounds, simToThree, staticObstacleBounds } from '../src/playback';

const fixtureReplayId = 'f'.repeat(32);
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
    telemetry: null,
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
    config: {},
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

function fixtureBundleFor(replayId: string, seedOffset = 0): ReplayBundle {
  return {
    ...fixtureBundle,
    seed_start: fixtureBundle.seed_start + seedOffset,
    episodes: [{
      ...fixtureEpisode,
      episode_id: replayId,
      simulation_seed: fixtureEpisode.simulation_seed + seedOffset,
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

async function mockFixtureEpisodes(page: Page, replayIds: string[]) {
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
    await route.fulfill({ json: fixtureBundleFor(id, Math.max(0, index)) });
  });
  await page.route('**/api/replays/*/summary', route => {
    const id = new URL(route.request().url()).pathname.split('/').at(-2) ?? replayIds[0];
    const index = replayIds.indexOf(id);
    return route.fulfill({ json: fixtureSummaryFor(id, Math.max(0, index)) });
  });
}

test('coordinate transforms, geometry, sampling, and interpolation preserve recorded state', () => {
  expect(simToThree([2, -3, 5])).toEqual([2, 5, 3]);
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
  await expect(page.getByRole('button', { name: 'Generate Episodes' })).toBeEnabled();
  await expect(page.getByLabel('Episode count', { exact: true })).toHaveValue('5');
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

test('plays real-schema frames, scrubs exactly, maps entities and obstacles, and switches episodes', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  await mockCatalog(page);
  const ids = [fixtureReplayId, 'e'.repeat(32)];
  await mockFixtureEpisodes(page, ids);
  await page.goto('/');
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

  await page.getByLabel('Playback speed').selectOption('0.5');
  await page.getByRole('button', { name: 'Play' }).click();
  await expect(rendered).toHaveAttribute('data-step', '1', { timeout: 3000 });
  await page.getByRole('button', { name: 'Pause' }).click();
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

  await page.getByRole('button', { name: 'Jump to end' }).click();
  await expect(rendered).toHaveAttribute('data-step', '2');
  await expect(page.getByText(/Terminal simulator outcome: intercepted/)).toBeVisible();
  await page.getByRole('button', { name: 'Next episode' }).click();
  await expect(page.getByText(/Episode 2 \/ 2 · seed 21/)).toBeVisible();
  await expect(rendered).toHaveAttribute('data-step', '0');
  await page.screenshot({ path: 'test-results/phase4-fixture-episode-switch.png', fullPage: true });
  expect(errors).toEqual([]);
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
  await page.getByLabel('Defender architecture').selectOption('lr_ppo');
  await page.getByLabel('Defender generation').selectOption('d1');
  await page.getByLabel('Defender training seed').selectOption('77');
  await page.getByLabel('Attacker policy').selectOption('A1-203');
  await page.getByLabel('Clutter level').selectOption('C3');
  await page.getByLabel('Episode count').selectOption('3');
  await page.getByLabel('Starting evaluation seed').fill('20');
  await page.getByRole('button', { name: 'Generate Episodes' }).click();
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
  await expect(page.getByTestId('rendered-replay-state')).toHaveAttribute('data-step', '0');
  await page.screenshot({ path: 'test-results/phase4-results.png', fullPage: true });
});

test('shows actionable submission and service errors without invented results', async ({ page }) => {
  await mockCatalog(page);
  await page.route('**/api/replay-jobs', route => route.fulfill({
    status: 409,
    json: { error: { code: 'REPLAY_JOB_BUSY', message: 'Another replay batch is running; retry when it finishes.' } },
  }));
  await page.goto('/');
  await page.getByRole('button', { name: 'Generate Episodes' }).click();
  await expect(page.getByRole('alert')).toContainText('Another replay batch is running');
  await expect(page.getByText('No completed replay episodes are available yet.')).toBeVisible();
});

test('backend outage is visible without inventing episode results', async ({ page }) => {
  await page.route('**/api/health', route => route.abort());
  await page.goto('/');
  await expect(page.getByRole('status')).toContainText('Backend unavailable');
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
