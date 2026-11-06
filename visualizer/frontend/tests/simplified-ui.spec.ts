import { expect, test, type Page } from '@playwright/test';

/**
 * Tests for the simplified two-scenario UI (brief section 31). Uses an in-memory fake backend (via
 * `page.route()`) that mirrors the real `/api/episodes*` contract (`visualizer/backend/episode_models.py`)
 * closely enough to exercise the full create -> list -> replay -> viewed -> delete lifecycle deterministically,
 * without needing the real research simulation environment.
 */

interface FakeEpisode {
  episode_id: string;
  created_at: string;
  scenario: 'no_clutter' | 'clutter';
  scenario_label: string;
  resolved_attacker: 'A0' | 'A1-CBF';
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

let idCounter = 0;
function makeEpisode(overrides: Partial<FakeEpisode> = {}): FakeEpisode {
  idCounter += 1;
  return {
    episode_id: idCounter.toString(16).padStart(32, '0'),
    created_at: new Date(2024, 0, 1, 0, 0, idCounter).toISOString(),
    scenario: 'clutter',
    scenario_label: 'Clutter',
    resolved_attacker: 'A1-CBF',
    defender: 'greedy',
    seed: 42,
    status: 'completed',
    outcome: 'soldier_caught',
    duration: 5.5,
    steps: 11,
    asset_captured: true,
    defender_interception: false,
    obstacle_collision: false,
    replay_path: idCounter.toString(16).padStart(32, '0'),
    source_commit: 'a989217b38dc0316718d04e41b065943fec29ce7',
    controller_version: 'journal-cbf-attacker-v1',
    is_new: true,
    min_defender_separation_m: 3.2,
    min_obstacle_clearance_m: 2.1,
    path_length_m: 40.5,
    error_message: null,
    ...overrides,
  };
}

function replayFor(episode: FakeEpisode) {
  return {
    schema: 'app_episode_replay/1',
    outcome: episode.outcome,
    obstacles: [{ min: [-5, -5, 0], max: [5, 5, 6] }],
    steps: Array.from({ length: episode.steps }, (_, i) => ({
      t: i * 0.5,
      attacker_pos: [i, i * 0.5, 2],
      defender_pos: [10 - i, 5, 2],
      asset_pos: [0, 20, 0],
    })),
  };
}

const PRESETS = {
  presets: [
    { scenario: 'no_clutter', scenario_label: 'No Clutter', clutter_level: 'C0', resolved_attacker: 'A0', attacker_label: 'A0', dynamic_obstacles_enabled: false, viability_conditioned_init: false, defender_cbf_enabled: false },
    { scenario: 'clutter', scenario_label: 'Clutter', clutter_level: 'C2', resolved_attacker: 'A1-CBF', attacker_label: 'A1-CBF', dynamic_obstacles_enabled: false, viability_conditioned_init: true, defender_cbf_enabled: true },
  ],
};

function policiesFixture(overrides: Record<string, { available: boolean; unavailability_reason: string | null }> = {}) {
  const ids = ['greedy', 'kalman_greedy', 'pn', 'lead', 'rmpc', 'passive', 'ppo', 'ppo_kalman', 'lr_ppo'];
  return {
    source_commit: 'a'.repeat(40), registry_sha256: 'b'.repeat(64), deterministic_inference: true,
    defenders: ids.map(id => ({
      policy_id: id, display_name: id, classification: 'analytical', architecture: null,
      available: overrides[id]?.available ?? true, unavailability_reason: overrides[id]?.unavailability_reason ?? null,
      estimator: null, observation_shape: null, generations: [], supported_scenarios: [], diagnostic_fields: [],
    })),
    attackers: [],
  };
}

interface BackendOptions {
  episodes?: FakeEpisode[];
  policyOverrides?: Record<string, { available: boolean; unavailability_reason: string | null }>;
  createError?: { code: string; message: string } | null;
  createResult?: FakeEpisode | null;
  getEpisodeError?: { code: string; message: string } | null;
  deleteFailIds?: string[];
}

async function mockBackend(page: Page, options: BackendOptions = {}) {
  const episodes: FakeEpisode[] = options.episodes ?? [];
  await page.route('**/api/health', route => route.fulfill({
    json: {
      service: 'counter-uas-visualizer', phase: 3, status: 'ok', simulation_enabled: true, data_root: 'configured', warnings: [],
      research_source_available: true, a0_available: true, a1_cbf_available: true, episode_storage_writable: true,
      new_episode_count: episodes.filter(e => e.is_new).length, saved_episode_count: episodes.filter(e => !e.is_new).length,
    },
  }));
  await page.route('**/api/scenario-presets', route => route.fulfill({ json: PRESETS }));
  await page.route('**/api/policies', route => route.fulfill({ json: policiesFixture(options.policyOverrides) }));

  await page.route('**/api/episodes/delete', async route => {
    const body = route.request().postDataJSON() as { episode_ids: string[] };
    const results = body.episode_ids.map(id => {
      const fail = options.deleteFailIds?.includes(id);
      if (!fail) {
        const index = episodes.findIndex(e => e.episode_id === id);
        if (index >= 0) episodes.splice(index, 1);
      }
      return { episode_id: id, deleted: !fail, reason: fail ? 'not_found' : null };
    });
    await route.fulfill({ json: { results } });
  });

  await page.route(/\/api\/episodes(\?.*)?$/, async route => {
    const request = route.request();
    if (request.method() === 'POST') {
      if (options.createError) {
        await route.fulfill({ status: 400, json: { error: options.createError } });
        return;
      }
      const body = request.postDataJSON() as { scenario: 'no_clutter' | 'clutter'; defender: string; seed?: number };
      const created = options.createResult ?? makeEpisode({
        scenario: body.scenario,
        scenario_label: body.scenario === 'no_clutter' ? 'No Clutter' : 'Clutter',
        resolved_attacker: body.scenario === 'no_clutter' ? 'A0' : 'A1-CBF',
        defender: body.defender,
        seed: body.seed ?? 777,
      });
      episodes.push(created);
      await route.fulfill({ status: 201, json: created });
      return;
    }
    const url = new URL(request.url());
    const newParam = url.searchParams.get('new');
    const scenarioParam = url.searchParams.get('scenario');
    const defenderParam = url.searchParams.get('defender');
    let filtered = episodes;
    if (newParam !== null) filtered = filtered.filter(e => e.is_new === (newParam === 'true'));
    if (scenarioParam) filtered = filtered.filter(e => e.scenario === scenarioParam);
    if (defenderParam) filtered = filtered.filter(e => e.defender === defenderParam);
    await route.fulfill({
      json: {
        episodes: filtered, total: filtered.length,
        new_count: episodes.filter(e => e.is_new).length, saved_count: episodes.filter(e => !e.is_new).length,
      },
    });
  });

  await page.route(/\/api\/episodes\/[0-9a-f]{32}$/, async route => {
    const request = route.request();
    const id = request.url().split('/').pop()!;
    if (request.method() === 'DELETE') {
      const index = episodes.findIndex(e => e.episode_id === id);
      const existed = index >= 0;
      if (existed) episodes.splice(index, 1);
      if (!existed) {
        await route.fulfill({ status: 404, json: { error: { code: 'EPISODE_NOT_FOUND', message: 'not found' } } });
        return;
      }
      await route.fulfill({ json: { episode_id: id, deleted: true, reason: null } });
      return;
    }
    if (options.getEpisodeError) {
      await route.fulfill({ status: 404, json: { error: options.getEpisodeError } });
      return;
    }
    const episode = episodes.find(e => e.episode_id === id);
    if (!episode) {
      await route.fulfill({ status: 404, json: { error: { code: 'EPISODE_NOT_FOUND', message: 'not found' } } });
      return;
    }
    episode.is_new = false;
    await route.fulfill({ json: { metadata: episode, replay: replayFor(episode) } });
  });
}

test.beforeEach(() => { idCounter = 0; });

// ----------------------------------------------------------------------------- 1/2: scenario options + default
test('shows exactly two scenario options, defaulting to Clutter', async ({ page }) => {
  await mockBackend(page);
  await page.goto('/');
  await expect(page.getByRole('button', { name: /No Clutter/ })).toBeVisible();
  const clutterCard = page.getByRole('button', { name: /^Clutter/ });
  await expect(clutterCard).toBeVisible();
  await expect(clutterCard).toHaveAttribute('aria-pressed', 'true');
  await expect(page.getByRole('button', { name: /No Clutter/ })).toHaveAttribute('aria-pressed', 'false');
});

// ----------------------------------------------------------------------------- 3: no attacker selector
test('never shows a separate attacker selector', async ({ page }) => {
  await mockBackend(page);
  await page.goto('/');
  await expect(page.getByText('Attacker:')).toBeVisible();
  await expect(page.locator('select', { hasText: 'A0' })).toHaveCount(0);
  await expect(page.locator('.control-column').getByLabel('Defender')).toBeVisible();
});

// ----------------------------------------------------------------------------- 4: defender dropdown
test('defender dropdown lists all 9 supported defenders with friendly labels', async ({ page }) => {
  await mockBackend(page);
  await page.goto('/');
  const select = page.locator('.control-column').getByLabel('Defender');
  const options = await select.locator('option').allTextContents();
  expect(options.length).toBe(9);
  expect(options.some(o => o.includes('Proportional Navigation'))).toBe(true);
  expect(options.some(o => o.includes('LR-PPO'))).toBe(true);
});

// ----------------------------------------------------------------------------- 5: optional seed
test('seed is optional and the randomize button fills a value', async ({ page }) => {
  await mockBackend(page);
  await page.goto('/');
  const seedInput = page.getByLabel('Seed (optional)');
  await expect(seedInput).toHaveValue('');
  await page.getByRole('button', { name: 'Randomize seed' }).click();
  await expect(seedInput).not.toHaveValue('');
});

// ----------------------------------------------------------------------------- 6: run No Clutter payload
test('running No Clutter sends the correct payload and resolves to Scripted', async ({ page }) => {
  await mockBackend(page);
  await page.goto('/');
  await page.getByRole('button', { name: /No Clutter/ }).click();
  await expect(page.getByText('Attacker: Scripted')).toBeVisible();
  const [request] = await Promise.all([
    page.waitForRequest(r => r.url().includes('/api/episodes') && r.method() === 'POST'),
    page.getByRole('button', { name: 'Run Episode' }).click(),
  ]);
  expect(request.postDataJSON()).toMatchObject({ scenario: 'no_clutter', defender: 'lr_ppo' });
});

// ----------------------------------------------------------------------------- 7: run Clutter payload
test('running Clutter (default) sends the correct payload and resolves to Evasive CBF', async ({ page }) => {
  await mockBackend(page);
  await page.goto('/');
  await expect(page.getByText('Attacker: Evasive CBF')).toBeVisible();
  const [request] = await Promise.all([
    page.waitForRequest(r => r.url().includes('/api/episodes') && r.method() === 'POST'),
    page.getByRole('button', { name: 'Run Episode' }).click(),
  ]);
  expect(request.postDataJSON()).toMatchObject({ scenario: 'clutter', defender: 'lr_ppo' });
});

// ----------------------------------------------------------------------------- 8: new episode appears in New (without marking viewed)
test('a newly generated episode appears under New without being marked viewed', async ({ page }) => {
  await mockBackend(page);
  await page.goto('/');
  await page.getByRole('button', { name: 'Run Episode' }).click();
  await expect(page.getByRole('tab', { name: /New \(1\)/ })).toBeVisible();
  await expect(page.locator('.episode-row .badge-new')).toBeVisible();
});

// ----------------------------------------------------------------------------- 9/10/11: replay loads + moves to Saved + NEW badge behavior
test('Replay loads the stored trajectory and moves the episode from New to Saved', async ({ page }) => {
  const episode = makeEpisode({ is_new: true });
  await mockBackend(page, { episodes: [episode] });
  await page.goto('/');
  await expect(page.getByRole('tab', { name: /New \(1\)/ })).toBeVisible();
  await expect(page.locator('.episode-row .badge-new')).toBeVisible();
  await page.getByRole('button', { name: 'Replay' }).first().click();
  await expect(page.locator('.trajectory-viewer')).toBeVisible();
  await expect(page.getByRole('tab', { name: /New \(0\)/ })).toBeVisible();
  await expect(page.getByRole('tab', { name: /Saved \(1\)/ })).toBeVisible();
});

// ----------------------------------------------------------------------------- 12: Saved list
test('Saved tab lists viewed episodes', async ({ page }) => {
  const episode = makeEpisode({ is_new: false });
  await mockBackend(page, { episodes: [episode] });
  await page.goto('/');
  await page.getByRole('tab', { name: /Saved/ }).click();
  await expect(page.locator('.episode-row')).toHaveCount(1);
});

// ----------------------------------------------------------------------------- 13/14: filters
test('scenario and defender filters narrow the visible list', async ({ page }) => {
  const episodes = [
    makeEpisode({ is_new: false, scenario: 'clutter', defender: 'greedy' }),
    makeEpisode({ is_new: false, scenario: 'no_clutter', defender: 'pn' }),
  ];
  await mockBackend(page, { episodes });
  await page.goto('/');
  await page.getByRole('tab', { name: /Saved/ }).click();
  await expect(page.locator('.episode-row')).toHaveCount(2);
  await page.locator('.library-filters').getByLabel('Scenario').selectOption('clutter');
  await expect(page.locator('.episode-row')).toHaveCount(1);
  await page.locator('.library-filters').getByLabel('Scenario').selectOption('all');
  await page.locator('.library-filters').getByLabel('Defender').selectOption('pn');
  await expect(page.locator('.episode-row')).toHaveCount(1);
});

// ----------------------------------------------------------------------------- 15/16: single delete + cancel
test('single delete requires confirmation and removes the episode on confirm', async ({ page }) => {
  const episode = makeEpisode({ is_new: false });
  await mockBackend(page, { episodes: [episode] });
  await page.goto('/');
  await page.getByRole('tab', { name: /Saved/ }).click();
  await page.getByRole('button', { name: 'Delete episode' }).click();
  await expect(page.getByRole('dialog')).toBeVisible();
  await page.getByRole('dialog').getByRole('button', { name: 'Cancel' }).click();
  await expect(page.locator('.episode-row')).toHaveCount(1);

  await page.getByRole('button', { name: 'Delete episode' }).click();
  await page.getByRole('dialog').getByRole('button', { name: 'Delete' }).click();
  await expect(page.locator('.episode-row')).toHaveCount(0);
  await expect(page.getByText('No saved episodes yet.')).toBeVisible();
});

// ----------------------------------------------------------------------------- 17/18: bulk select + delete
test('bulk select and delete removes only the selected episodes', async ({ page }) => {
  const episodes = [makeEpisode({ is_new: false }), makeEpisode({ is_new: false })];
  await mockBackend(page, { episodes });
  await page.goto('/');
  await page.getByRole('tab', { name: /Saved/ }).click();
  await page.getByRole('button', { name: 'Select' }).click();
  await page.locator('.episode-row input[type=checkbox]').first().check();
  await page.getByRole('button', { name: /Delete selected/ }).click();
  await page.getByRole('dialog').getByRole('button', { name: 'Delete' }).click();
  await expect(page.locator('.episode-row')).toHaveCount(1);
});

// ----------------------------------------------------------------------------- 19: bulk partial failure
test('bulk delete reports per-id failures instead of assuming total success', async ({ page }) => {
  const episodes = [makeEpisode({ is_new: false }), makeEpisode({ is_new: false })];
  await mockBackend(page, { episodes, deleteFailIds: [episodes[0].episode_id] });
  await page.goto('/');
  await page.getByRole('tab', { name: /Saved/ }).click();
  await page.getByRole('button', { name: 'Select' }).click();
  await page.getByRole('button', { name: 'Select all visible' }).click();
  await page.getByRole('button', { name: /Delete selected/ }).click();
  await page.getByRole('dialog').getByRole('button', { name: 'Delete' }).click();
  await expect(page.getByText(/Could not delete:/)).toBeVisible();
});

// ----------------------------------------------------------------------------- 20: empty states
test('shows clean empty-state messages', async ({ page }) => {
  await mockBackend(page, { episodes: [] });
  await page.goto('/');
  await expect(page.getByText('No new episodes. Run an episode to create one.')).toBeVisible();
  await page.getByRole('tab', { name: /Saved/ }).click();
  await expect(page.getByText('No saved episodes yet.')).toBeVisible();
});

// ----------------------------------------------------------------------------- 21: backend run error
test('shows a friendly error when running an episode fails', async ({ page }) => {
  await mockBackend(page, { createError: { code: 'SIMULATION_RUNTIME_UNAVAILABLE', message: 'Cannot start the simulation runtime.' } });
  await page.goto('/');
  await page.getByRole('button', { name: 'Run Episode' }).click();
  await expect(page.getByText('Cannot start the simulation runtime.')).toBeVisible();
});

// ----------------------------------------------------------------------------- 22: replay error
test('shows a friendly error when loading a replay fails', async ({ page }) => {
  const episode = makeEpisode({ is_new: true });
  await mockBackend(page, { episodes: [episode], getEpisodeError: { code: 'EPISODE_REPLAY_UNREADABLE', message: 'This replay could not be read.' } });
  await page.goto('/');
  await page.getByRole('button', { name: 'Replay' }).first().click();
  await expect(page.getByText('This replay could not be read.')).toBeVisible();
});

// ----------------------------------------------------------------------------- 23: existing viewer functionality
test('trajectory viewer supports play/pause and the timeline scrubber', async ({ page }) => {
  const episode = makeEpisode({ is_new: true, steps: 5 });
  await mockBackend(page, { episodes: [episode] });
  await page.goto('/');
  await page.getByRole('button', { name: 'Replay' }).first().click();
  await expect(page.locator('.trajectory-viewer-step-count')).toHaveText('1 / 5');
  await page.getByLabel('Timeline').fill('3');
  await expect(page.locator('.trajectory-viewer-step-count')).toHaveText('4 / 5');
  await page.getByRole('button', { name: 'Play' }).click();
  await expect(page.getByRole('button', { name: 'Pause' })).toBeVisible();
});

// ----------------------------------------------------------------------------- 24: Details drawer
test('Details drawer is collapsed by default and reveals provenance metadata', async ({ page }) => {
  const episode = makeEpisode({ is_new: true });
  await mockBackend(page, { episodes: [episode] });
  await page.goto('/');
  await page.getByRole('button', { name: 'Replay' }).first().click();
  await expect(page.getByText('journal-cbf-attacker-v1')).toHaveCount(0);
  await page.getByRole('button', { name: /Details/ }).click();
  await expect(page.getByText('journal-cbf-attacker-v1')).toBeVisible();
  await expect(page.getByText(episode.source_commit)).toBeVisible();
});

// ----------------------------------------------------------------------------- 25: legacy compatibility route
test('legacy interface remains available at /legacy', async ({ page }) => {
  await page.route('**/api/health', route => route.fulfill({
    json: { service: 'counter-uas-visualizer', phase: 3, status: 'ok', data_root: 'configured', simulation_enabled: true, warnings: [] },
  }));
  await page.route('**/api/policies', route => route.fulfill({ json: policiesFixture() }));
  await page.route('**/api/scenarios', route => route.fulfill({ json: { scenarios: [] } }));
  await page.route('**/api/replays?limit=100', route => route.fulfill({ json: { replays: [], warnings: [] } }));
  await page.goto('/legacy');
  await expect(page.getByRole('heading', { name: 'No episode loaded' })).toBeVisible();
  await expect(page.locator('.simplified-app')).toHaveCount(0);
});
