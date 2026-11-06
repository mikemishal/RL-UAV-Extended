import { useEffect, useMemo, useRef, useState } from 'react';
import {
  getCbfReviewCase,
  getCbfReviewCollection,
  CbfReviewApiError,
} from './cbfReview';
import type { CbfReviewCaseDetail, CbfReviewCaseSummary, CbfReviewCollection, CbfReviewStep } from './cbfReview';

interface CbfDefenderReviewProps {
  /** Case id to open initially (e.g. from a `?view=cbf-review&case=<id>` deep link). */
  initialCaseId?: string | null;
  /** Called whenever the selected case changes, so the parent can keep the URL in sync. */
  onCaseChange?: (caseId: string | null) => void;
}

const WORLD_HALF_EXTENT_M = 60; // a little larger than the canonical L=50 domain, for margin.
const SVG_SIZE = 560;

function worldToSvg(x: number, y: number): [number, number] {
  // Simulation is x-right, y-"up" (north); SVG y grows downward, so flip y for a conventional top-down map.
  const sx = ((x + WORLD_HALF_EXTENT_M) / (2 * WORLD_HALF_EXTENT_M)) * SVG_SIZE;
  const sy = ((WORLD_HALF_EXTENT_M - y) / (2 * WORLD_HALF_EXTENT_M)) * SVG_SIZE;
  return [sx, sy];
}

function scaleLength(meters: number): number {
  return (meters / (2 * WORLD_HALF_EXTENT_M)) * SVG_SIZE;
}

function formatOutcome(outcome: string | null): string {
  if (!outcome) return 'unknown';
  return outcome.replace(/_/g, ' ');
}

function formatNum(value: number | null | undefined, digits = 2, suffix = ''): string {
  return value === null || value === undefined || !Number.isFinite(value) ? '\u2014' : `${value.toFixed(digits)}${suffix}`;
}

function CaseListItem({
  item,
  active,
  onSelect,
}: {
  item: CbfReviewCaseSummary;
  active: boolean;
  onSelect: () => void;
}) {
  return (
    <li>
      <button type="button" className="case-list-button cbf-review-list-button" aria-pressed={active} onClick={onSelect}>
        <span className="cbf-review-rank">#{item.rank ?? '\u2014'}</span>
        <strong>{item.case_name}</strong>
        <span>
          {item.defender_method ?? 'unknown defender'} &middot; {item.attacker_variant} &middot; seed {item.seed ?? '\u2014'}
        </span>
        <span>
          Outcome: {formatOutcome(item.outcome)}
          {item.category_match === 'substitute' && <em className="cbf-review-substitute-flag"> (closest available substitute)</em>}
        </span>
        <small>Requested category: {item.requested_category ?? '\u2014'}</small>
      </button>
    </li>
  );
}

export function CbfDefenderReview({ initialCaseId, onCaseChange }: CbfDefenderReviewProps) {
  const [collection, setCollection] = useState<CbfReviewCollection | null>(null);
  const [collectionError, setCollectionError] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(initialCaseId ?? null);
  const [detail, setDetail] = useState<CbfReviewCaseDetail | null>(null);
  const [detailError, setDetailError] = useState<string | null>(null);
  const [stepIndex, setStepIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [showDefenderSafety, setShowDefenderSafety] = useState(true);
  const [showInterceptRadius, setShowInterceptRadius] = useState(true);
  const [showCbfActive, setShowCbfActive] = useState(true);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    let cancelled = false;
    getCbfReviewCollection()
      .then(result => {
        if (cancelled) return;
        setCollection(result);
        setCollectionError(null);
        setSelectedId(previous => previous ?? result.opening_case_id ?? result.cases[0]?.case_id ?? null);
      })
      .catch(error => {
        if (cancelled) return;
        setCollectionError(error instanceof CbfReviewApiError ? error.message : 'Could not load the A1-CBF review collection.');
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!selectedId) return;
    let cancelled = false;
    setDetail(null);
    setDetailError(null);
    setStepIndex(0);
    setPlaying(false);
    getCbfReviewCase(selectedId)
      .then(result => {
        if (cancelled) return;
        setDetail(result);
      })
      .catch(error => {
        if (cancelled) return;
        setDetailError(error instanceof CbfReviewApiError ? error.message : `Could not load case ${selectedId}.`);
      });
    onCaseChange?.(selectedId);
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedId]);

  useEffect(() => {
    if (!playing || !detail) {
      if (timerRef.current) clearInterval(timerRef.current);
      return;
    }
    timerRef.current = setInterval(() => {
      setStepIndex(previous => {
        if (previous >= detail.steps.length - 1) {
          setPlaying(false);
          return previous;
        }
        return previous + 1;
      });
    }, 400);
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [playing, detail]);

  const step: CbfReviewStep | null = detail ? detail.steps[Math.min(stepIndex, detail.steps.length - 1)] : null;
  const trail = useMemo(() => {
    if (!detail) return { attacker: '', defender: '', asset: '' };
    const upTo = detail.steps.slice(0, stepIndex + 1);
    const toPoints = (pick: (s: CbfReviewStep) => readonly [number, number, number]) =>
      upTo.map(s => worldToSvg(pick(s)[0], pick(s)[1]).join(',')).join(' ');
    return {
      attacker: toPoints(s => s.attacker_pos),
      defender: toPoints(s => s.defender_pos),
      asset: toPoints(s => s.asset_pos),
    };
  }, [detail, stepIndex]);

  if (collectionError) {
    return (
      <section className="cbf-review panel" aria-label="A1-CBF defender review">
        <p className="inline-error" role="alert">{collectionError}</p>
      </section>
    );
  }
  if (!collection) {
    return (
      <section className="cbf-review panel" aria-label="A1-CBF defender review">
        <p className="muted">Loading the A1-CBF review collection&hellip;</p>
      </section>
    );
  }

  return (
    <section className="cbf-review panel" aria-label="A1-CBF defender review">
      <div className="cbf-review-banner" role="status">
        <strong>{collection.label}</strong>
        <p>{collection.description}</p>
        <p className="muted">
          Research source: <code>{collection.research_pin_commit}</code> &middot; precomputed trajectories read from{' '}
          <code>{collection.research_relative_dir}</code> &mdash; the browser never re-runs or reconstructs the controller.
        </p>
        {collection.warnings.length > 0 && (
          <ul className="cbf-review-warnings">
            {collection.warnings.map(w => <li key={w}>{w}</li>)}
          </ul>
        )}
      </div>
      <div className="case-columns cbf-review-columns">
        <div>
          <h3>Review cases ({collection.cases.length})</h3>
          <ul className="case-list">
            {collection.cases.map(item => (
              <CaseListItem key={item.case_id} item={item} active={item.case_id === selectedId} onSelect={() => setSelectedId(item.case_id)} />
            ))}
          </ul>
        </div>
        <div>
          {detailError && <p className="inline-error" role="alert">{detailError}</p>}
          {!detail && !detailError && <p className="muted">Loading case&hellip;</p>}
          {detail && (
            <>
              <div className="case-detail-heading">
                <div>
                  <h3>{detail.case_name}</h3>
                  <p>
                    Rank #{detail.rank ?? '\u2014'} &middot; requested category: {detail.requested_category ?? '\u2014'}
                    {detail.category_match === 'substitute' ? ' (substitute \u2014 see rationale below)' : ' (exact match)'}
                  </p>
                </div>
              </div>
              {detail.category_rationale && <p className="muted cbf-review-rationale">{detail.category_rationale}</p>}
              <div className="cbf-review-summary-grid">
                <span>Attacker variant: <strong>{detail.attacker_variant}</strong></span>
                <span>Defender method: <strong>{detail.defender_method ?? '\u2014'}</strong></span>
                <span>Seed: <strong>{detail.seed ?? '\u2014'}</strong></span>
                <span>Outcome: <strong>{formatOutcome(detail.outcome)}</strong></span>
                <span>Asset captured: <strong>{detail.asset_captured === null ? '\u2014' : detail.asset_captured ? 'yes' : 'no'}</strong></span>
                <span>Defender interception: <strong>{detail.defender_intercepted === null ? '\u2014' : detail.defender_intercepted ? 'yes' : 'no'}</strong></span>
                <span>Obstacle collision: <strong>{detail.obstacle_collision === null ? '\u2014' : detail.obstacle_collision ? 'yes' : 'no'}</strong></span>
                <span>Min defender separation: <strong>{formatNum(detail.min_defender_separation_m, 2, ' m')}</strong></span>
                <span>Min obstacle clearance: <strong>{formatNum(detail.min_obstacle_clearance_m, 2, ' m')}</strong></span>
                <span>Duration: <strong>{formatNum(detail.episode_duration_s, 1, ' s')}</strong> ({detail.step_count} steps)</span>
              </div>
              {detail.what_to_look_for && (
                <div className="cbf-review-watch">
                  <h4>What to look for</h4>
                  <p>{detail.what_to_look_for}</p>
                </div>
              )}

              <div className="cbf-review-overlay-toggles" role="group" aria-label="Overlay toggles">
                <label><input type="checkbox" checked={showDefenderSafety} onChange={e => setShowDefenderSafety(e.target.checked)} /> Defender safety region ({collection.defender_safety_radius_m} m)</label>
                <label><input type="checkbox" checked={showInterceptRadius} onChange={e => setShowInterceptRadius(e.target.checked)} /> Defender interception region ({collection.defender_intercept_radius_m} m)</label>
                <label><input type="checkbox" checked={showCbfActive} onChange={e => setShowCbfActive(e.target.checked)} /> CBF active indicator</label>
              </div>

              <svg viewBox={`0 0 ${SVG_SIZE} ${SVG_SIZE}`} width="100%" height={SVG_SIZE} role="img" aria-label="Top-down trajectory view" className="cbf-review-scene">
                <rect x={0} y={0} width={SVG_SIZE} height={SVG_SIZE} className="cbf-review-scene-bg" />
                {detail.obstacles.map((o, i) => {
                  const [x0, y0] = worldToSvg(o.min[0], o.min[1]);
                  const [x1, y1] = worldToSvg(o.max[0], o.max[1]);
                  return <rect key={i} x={Math.min(x0, x1)} y={Math.min(y0, y1)} width={Math.abs(x1 - x0)} height={Math.abs(y1 - y0)} className="cbf-review-obstacle" />;
                })}
                <polyline points={trail.asset} className="cbf-review-trail cbf-review-trail-asset" />
                <polyline points={trail.defender} className="cbf-review-trail cbf-review-trail-defender" />
                <polyline points={trail.attacker} className="cbf-review-trail cbf-review-trail-attacker" />
                {step && (
                  <>
                    {showDefenderSafety && (
                      <circle cx={worldToSvg(step.defender_pos[0], step.defender_pos[1])[0]} cy={worldToSvg(step.defender_pos[0], step.defender_pos[1])[1]} r={scaleLength(collection.defender_safety_radius_m)} className="cbf-review-safety-ring" />
                    )}
                    {showInterceptRadius && (
                      <circle cx={worldToSvg(step.defender_pos[0], step.defender_pos[1])[0]} cy={worldToSvg(step.defender_pos[0], step.defender_pos[1])[1]} r={scaleLength(collection.defender_intercept_radius_m)} className="cbf-review-intercept-ring" />
                    )}
                    <circle cx={worldToSvg(step.asset_pos[0], step.asset_pos[1])[0]} cy={worldToSvg(step.asset_pos[0], step.asset_pos[1])[1]} r={4} className="cbf-review-marker cbf-review-marker-asset" />
                    <circle cx={worldToSvg(step.defender_pos[0], step.defender_pos[1])[0]} cy={worldToSvg(step.defender_pos[0], step.defender_pos[1])[1]} r={5} className="cbf-review-marker cbf-review-marker-defender" />
                    <circle
                      cx={worldToSvg(step.attacker_pos[0], step.attacker_pos[1])[0]}
                      cy={worldToSvg(step.attacker_pos[0], step.attacker_pos[1])[1]}
                      r={6}
                      className={`cbf-review-marker cbf-review-marker-attacker${showCbfActive && (step.defender_cbf_active_derived || step.obstacle_cbf_active_derived) ? ' cbf-active' : ''}`}
                    />
                  </>
                )}
              </svg>

              <div className="cbf-review-transport">
                <button type="button" onClick={() => setStepIndex(0)} disabled={stepIndex === 0}>&#8676; Start</button>
                <button type="button" onClick={() => setStepIndex(i => Math.max(0, i - 1))} disabled={stepIndex === 0}>&#8592; Step</button>
                <button type="button" onClick={() => setPlaying(p => !p)}>{playing ? 'Pause' : 'Play'}</button>
                <button type="button" onClick={() => setStepIndex(i => Math.min(detail.steps.length - 1, i + 1))} disabled={stepIndex >= detail.steps.length - 1}>Step &#8594;</button>
                <input
                  type="range" min={0} max={detail.steps.length - 1} value={stepIndex}
                  onChange={e => setStepIndex(Number(e.target.value))}
                  aria-label="Step scrubber"
                />
                <span className="muted">step {stepIndex + 1} / {detail.steps.length}</span>
              </div>

              {step && (
                <div className="cbf-review-playback-info">
                  <span>t = {formatNum(step.t, 1, ' s')}</span>
                  <span>attacker speed (derived): {formatNum(step.attacker_speed_mps_derived, 2, ' m/s')}</span>
                  <span>attacker&rarr;asset (derived): {formatNum(step.attacker_asset_distance_m_derived, 2, ' m')}</span>
                  <span>attacker&rarr;defender (derived): {formatNum(step.attacker_defender_distance_m_derived, 2, ' m')}</span>
                  <span>defender observable: {String(step.defender_observable)}</span>
                  <span>defender CBF active (derived): {step.defender_cbf_active_derived === null ? '\u2014' : String(step.defender_cbf_active_derived)}</span>
                  <span>obstacle CBF active (derived): {step.obstacle_cbf_active_derived === null ? '\u2014' : String(step.obstacle_cbf_active_derived)}</span>
                  <span>intervention source: <strong>{step.intervention_source}</strong></span>
                  <span>QP status: <strong>{step.cbf_qp_status}</strong></span>
                  {step.defender_barrier_h !== null && <span>defender barrier h: {formatNum(step.defender_barrier_h, 2)}</span>}
                  {step.min_inflated_obstacle_clearance !== null && <span>min inflated obstacle clearance: {formatNum(step.min_inflated_obstacle_clearance, 2, ' m')}</span>}
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </section>
  );
}
