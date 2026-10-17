import { expect, test, type Page, type Route } from '@playwright/test';

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

test('renders an empty interactive battlefield and functional Phase 3 selectors', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  await mockCatalog(page);
  await page.goto('/');
  await expect(page.getByRole('heading', { name: 'No episode loaded for playback' })).toBeVisible();
  const viewport = page.getByTestId('viewport');
  await expect(viewport).toHaveAttribute('data-scene-ready', 'true');
  const canvas = page.locator('canvas');
  await expect(canvas).toBeVisible();
  expect(await canvas.evaluate(element => {
    if (!(element instanceof HTMLCanvasElement)) throw new Error('Expected a canvas element');
    const gl = element.getContext('webgl2');
    return Boolean(gl && !gl.isContextLost() && gl.getParameter(gl.VERSION));
  })).toBe(true);
  const initial = await canvas.screenshot();
  const box = await canvas.boundingBox();
  if (!box) throw new Error('Canvas has no layout box');
  await page.mouse.move(box.x + 80, box.y + 80);
  await page.mouse.down();
  await page.mouse.move(box.x + 220, box.y + 120, { steps: 12 });
  await page.mouse.up();
  await expect(async () => expect((await canvas.screenshot()).equals(initial)).toBe(false)).toPass();
  await page.getByRole('button', { name: 'Reset camera' }).click();
  await expect(async () => expect((await canvas.screenshot()).equals(initial)).toBe(true)).toPass();
  await expect(page.getByRole('button', { name: 'Generate Episodes' })).toBeEnabled();
  await expect(page.getByLabel('Episode count', { exact: true })).toHaveValue('5');
  await expect(page.getByLabel('Defender architecture')).toHaveValue('greedy');
  await expect(page.getByText('Defender speed: 18 m/s')).toBeVisible();
  await page.screenshot({ path: 'test-results/phase3-policy-selection.png', fullPage: true });
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(canvas).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.screenshot({ path: 'test-results/phase3-mobile.png', fullPage: true });
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
  await expect(page.getByRole('heading', { name: 'No episode loaded for playback' })).toBeVisible();
  await page.screenshot({ path: 'test-results/phase3-results.png', fullPage: true });
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
  await expect(page.getByTestId('viewport')).toHaveAttribute('data-scene-ready', 'true');
  const extension = await page.locator('canvas').evaluateHandle(element => {
    if (!(element instanceof HTMLCanvasElement)) throw new Error('Expected canvas');
    const extension = element.getContext('webgl2')?.getExtension('WEBGL_lose_context');
    if (!extension) throw new Error('Browser cannot test WebGL context loss');
    return extension;
  });
  await extension.evaluate(value => value.loseContext());
  await expect(page.getByRole('alert')).toContainText('WebGL context lost');
  await expect(page.getByRole('button', { name: 'Reset camera' })).toBeDisabled();
  await extension.evaluate(value => value.restoreContext());
  await expect(page.getByRole('alert')).toHaveCount(0);
  await expect(page.getByTestId('viewport')).toHaveAttribute('data-scene-ready', 'true');
  await extension.dispose();
});
