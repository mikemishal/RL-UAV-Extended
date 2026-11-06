/**
 * A1-CBF defender-review precomputed trajectory path (journal_cbf_attacker, defender_cbf / Prompt 4 manual
 * review). Fetch helpers and types for the SEPARATE, additive /api/cbf-review/* endpoints -- see
 * visualizer/backend/precomputed_cases.py. Deliberately independent of api.ts's ReplayBundle/ResearchCase types:
 * those model the strict, SB3-oriented replay contract, which a CBF-only trajectory cannot honestly satisfy.
 */

export interface CbfReviewCaseSummary {
  case_id: string;
  rank: number | null;
  requested_category: string | null;
  category_match: 'exact' | 'substitute' | null;
  category_rationale: string | null;
  case_name: string;
  attacker_variant: 'A1-CBF-full' | 'A1-CBF-obstacle-only';
  defender_method: string | null;
  seed: number | null;
  outcome: string | null;
  asset_captured: boolean | null;
  defender_intercepted: boolean | null;
  obstacle_collision: boolean | null;
  min_defender_separation_m: number | null;
  min_obstacle_clearance_m: number | null;
  episode_duration_s: number | null;
  step_count: number;
  what_to_look_for: string | null;
}

export interface CbfReviewCollection {
  collection_id: string;
  label: string;
  description: string;
  research_pin_commit: string;
  research_relative_dir: string;
  opening_case_id: string | null;
  defender_safety_radius_m: number;
  defender_intercept_radius_m: number;
  step_dt_s: number;
  cases: CbfReviewCaseSummary[];
  warnings: string[];
}

export type Vec3 = readonly [number, number, number];

export interface CbfReviewObstacle {
  min: Vec3;
  max: Vec3;
}

export interface CbfReviewStep {
  t: number;
  attacker_pos: Vec3;
  defender_pos: Vec3;
  asset_pos: Vec3;
  defender_observable: boolean;
  intervention_source: 'none' | 'obstacle' | 'defender' | 'joint';
  cbf_qp_status: 'inactive' | 'hard' | 'soft_fallback' | 'emergency_brake';
  defender_barrier_h: number | null;
  min_inflated_obstacle_clearance: number | null;
  /** Derived (computed by the backend from recorded positions, never recorded directly by the simulator). */
  attacker_speed_mps_derived: number | null;
  attacker_asset_distance_m_derived: number | null;
  attacker_defender_distance_m_derived: number | null;
  obstacle_cbf_active_derived: boolean | null;
  defender_cbf_active_derived: boolean | null;
}

export interface CbfReviewCaseDetail extends CbfReviewCaseSummary {
  obstacles: CbfReviewObstacle[];
  steps: CbfReviewStep[];
}

class CbfReviewApiError extends Error {
  constructor(message: string, readonly status: number, readonly code: string) {
    super(message);
  }
}

async function cbfApi<T>(path: string): Promise<T> {
  let response: Response;
  try {
    response = await fetch(path);
  } catch {
    throw new CbfReviewApiError('Backend unavailable. Start the localhost API and try again.', 0, 'BACKEND_UNAVAILABLE');
  }
  const payload: unknown = await response.json().catch(() => null);
  if (!response.ok) {
    const error = typeof payload === 'object' && payload !== null && 'error' in payload
      ? (payload as { error?: { code?: unknown; message?: unknown } }).error
      : undefined;
    throw new CbfReviewApiError(
      typeof error?.message === 'string' ? error.message : `Request failed (HTTP ${response.status}).`,
      response.status,
      typeof error?.code === 'string' ? error.code : 'API_ERROR',
    );
  }
  return payload as T;
}

export function getCbfReviewCollection(): Promise<CbfReviewCollection> {
  return cbfApi('/api/cbf-review/collection');
}

export function getCbfReviewCase(caseId: string): Promise<CbfReviewCaseDetail> {
  return cbfApi(`/api/cbf-review/cases/${encodeURIComponent(caseId)}`);
}

export { CbfReviewApiError };
