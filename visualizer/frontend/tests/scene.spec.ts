import { expect, test } from '@playwright/test';

test('renders a real empty WebGL scene, orbits, resets, and responds to viewport size', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto('/');
  await expect(page.getByRole('heading', { name: 'No episode loaded' })).toBeVisible();
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
  await expect(page.getByRole('button', { name: 'Run episodes / Phase 3' })).toBeDisabled();
  await expect(page.getByRole('button', { name: 'Play', exact: true })).toBeDisabled();
  await expect(page.getByLabel('Episode count', { exact: true })).toHaveValue('5');
  await page.screenshot({ path: 'test-results/phase1-desktop.png', fullPage: true });
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(canvas).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.screenshot({ path: 'test-results/phase1-mobile.png', fullPage: true });
  expect(errors).toEqual([]);
});

test('backend outage is visible without inventing episode results', async ({ page }) => {
  await page.route('**/api/health', route => route.abort());
  await page.goto('/');
  await expect(page.getByRole('status')).toContainText('Backend unavailable');
  await expect(page.getByText('No frames / no outcome')).toBeVisible();
});

test('unconfigured research data root is visible', async ({ page }) => {
  await page.route('**/api/health', route => route.fulfill({
    json: { service: 'counter-uas-visualizer', phase: 1, status: 'degraded' },
  }));
  await page.goto('/');
  await expect(page.getByRole('status')).toContainText('research data root not configured');
});

test('reports WebGL loss and clears the warning after actual context restoration', async ({ page }) => {
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
