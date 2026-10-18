import { useMemo } from 'react';
import type { MouseEvent as ReactMouseEvent, PointerEvent as ReactPointerEvent, ReactNode } from 'react';
import type {
  ReplayEpisode,
  ReplayEvent,
  ReplayFrame,
  ReplayPolicyTelemetry,
  Vec3,
} from './api';
import type { PlaybackReadout } from './ReplayPlayback';
import {
  angleBetween3dDeg,
  elevationDifferenceDeg,
  euclideanDistance,
  signedAzimuthDifferenceDeg,
} from './diagnostics';
import type { DiagnosticSeries, SeriesPoint } from './diagnostics';
import { usePlaybackClock } from './playbackClock';

export type InformationMode = 'ground-truth' | 'policy-visible';

export interface DiagnosticVisibility {
  lead: boolean;
  requested: boolean;
  executed: boolean;
  cone: boolean;
  estimate: boolean;
  truthMarker: boolean;
  trackingError: boolean;
  estimateVelocity: boolean;
  safetyMarker: boolean;
}

interface PolicyInspectorProps {
  episode: ReplayEpisode | null;
  replayId: string | null;
  readout: PlaybackReadout | null;
  diagnosticSeries: DiagnosticSeries | null;
  informationMode: InformationMode;
  visibility: DiagnosticVisibility;
  onInformationMode: (mode: InformationMode) => void;
  onVisibility: (visibility: DiagnosticVisibility) => void;
}

function formatVector(vector: Vec3 | null | undefined, unit: string): string {
  if (!vector || vector.some(value => !Number.isFinite(value))) return 'Unavailable';
  return `[${vector.map(value => Number(value.toFixed(3))).join(', ')}] ${unit}`;
}

function formatNumber(value: number | null | undefined, unit = '', digits = 2): string {
  return value === null || value === undefined || !Number.isFinite(value)
    ? 'Unavailable'
    : `${value.toFixed(digits)}${unit ? ` ${unit}` : ''}`;
}

function readablePolicy(policyId: string): string {
  const labels: Record<string, string> = {
    greedy: 'Greedy',
    kalman_greedy: 'Kalman Greedy',
    pn: 'Proportional Navigation',
    pn_kalman: 'Proportional Navigation (Kalman)',
    lead: 'Lead',
    lead_kalman: 'Lead (Kalman)',
    rmpc: 'Robust MPC',
    ppo: 'PPO',
    ppo_kalman: 'PPO-Kalman',
    lr_ppo: 'LR-PPO',
  };
  return labels[policyId] ?? policyId;
}

function DataField({ label, value, title }: { label: string; value: ReactNode; title?: string }) {
  return <div className="inspector-field" title={title}><dt>{label}</dt><dd>{value}</dd></div>;
}

function Toggle({
  label,
  checked,
  onChange,
}: {
  label: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
}) {
  return (
    <label className="diagnostic-toggle">
      <input type="checkbox" checked={checked} onChange={event => onChange(event.target.checked)} />
      {label}
    </label>
  );
}

function vectorDirection(a: Vec3 | null | undefined, b: Vec3 | null | undefined): number | null {
  return angleBetween3dDeg(a ?? null, b ?? null);
}

function estimatePosition(telemetry: ReplayPolicyTelemetry | null): Vec3 | null {
  if (!telemetry?.decision_policy_visible.enemy_detected) return null;
  const visible = telemetry.decision_policy_visible;
  return visible.estimator === 'kalman' ? visible.e_hat_m : visible.enemy_measurement_m;
}

function estimateVelocity(telemetry: ReplayPolicyTelemetry | null): Vec3 | null {
  if (!telemetry?.decision_policy_visible.enemy_detected) return null;
  const visible = telemetry.decision_policy_visible;
  return visible.estimator === 'kalman' ? visible.v_hat_mps : visible.enemy_measurement_velocity_mps;
}

function fieldReason(telemetry: ReplayPolicyTelemetry | null, field: string, fallback: string): string {
  return telemetry?.unavailable_fields[field] ?? telemetry?.decision_policy_visible.unavailable_fields[field] ?? fallback;
}

function safetyFields(
  frame: ReplayFrame,
  commandEndFrame: ReplayFrame | null,
  telemetry: ReplayPolicyTelemetry | null,
  informationMode: InformationMode,
): ReactNode {
  const environment = (telemetry ? commandEndFrame ?? frame : frame).environmental_state;
  const clearance = environment.defender_min_obstacle_clearance;
  const dynamicClearance = environment.defender_min_dynamic_obstacle_clearance;
  return (
    <>
      <DataField label="Avoidance active (recorded)" value={telemetry ? (telemetry.avoidance_active ? 'Yes' : 'No') : 'Unavailable: no decision for this terminal time/state.'} />
      <DataField label="Safety-adjusted direction (desired → safe)" value={formatNumber(telemetry?.desired_to_safe_angle_deg, '°')} />
      <DataField label="Requested → executed velocity angle" value={formatNumber(telemetry?.requested_to_velocity_angle_deg, '°')} title="Recorded comparison; velocity reflects the constrained post-transition dynamics." />
      <DataField label="Post-transition minimum static clearance" value={informationMode === 'ground-truth' ? formatNumber(typeof clearance === 'number' ? clearance : null, 'm') : 'Unavailable: privileged full-scene clearance is hidden in Policy-Visible mode.'} />
      <DataField label="Post-transition minimum moving-obstacle clearance" value={informationMode === 'ground-truth' ? formatNumber(typeof dynamicClearance === 'number' ? dynamicClearance : null, 'm') : 'Unavailable: moving-obstacle state is hidden in Policy-Visible mode.'} />
      <DataField label="Defender collision event at resulting frame" value={informationMode === 'ground-truth' ? String(environment.defender_collision_event_type ?? 'No collision event recorded at this frame') : 'Unavailable: privileged event information is hidden in Policy-Visible mode.'} />
      <DataField label="Arbitration" value={telemetry
        ? telemetry.avoidance_active
          ? informationMode === 'ground-truth' && environment.defender_avoidance_obstacle_id
            ? `Avoidance active / ${String(environment.defender_avoidance_obstacle_id)}`
            : 'Avoidance active; obstacle identity unavailable in this view'
          : 'No avoidance override recorded'
        : 'Unavailable: no recorded command interval.'} />
    </>
  );
}

function policyParameterRows(episode: ReplayEpisode): ReactNode {
  const values = episode.defender.parameters;
  const entries = Object.entries(values).filter(([, value]) => value !== null && value !== undefined);
  if (!entries.length) return <DataField label="Policy parameters" value="Unavailable: no policy-specific parameters are recorded." />;
  return entries.map(([key, value]) => (
    <DataField key={key} label={key.replaceAll('_', ' ')} value={typeof value === 'object' ? JSON.stringify(value) : String(value)} />
  ));
}

export function PolicyInspector({
  episode,
  replayId,
  readout,
  diagnosticSeries,
  informationMode,
  visibility,
  onInformationMode,
  onVisibility,
}: PolicyInspectorProps) {
  const liveTimeS = usePlaybackClock(replayId);
  if (!episode) return null;
  const currentReadout = readout?.replayId === replayId ? readout : null;
  const frame = currentReadout?.frame ?? episode.frames[0];
  const telemetry = currentReadout?.decision?.telemetry ?? null;
  const currentTimeS = liveTimeS ?? currentReadout?.timeS ?? 0;
  const isLrppo = episode.defender.policy_id === 'lr_ppo';
  const isKalman = episode.defender.estimator === 'kalman';
  const lead = telemetry?.lrppo?.lead_direction ?? null;
  const requested = telemetry?.requested_action ?? null;
  const nativeAngle = isLrppo ? vectorDirection(lead, requested) : null;
  const horizontalAngle = isLrppo ? signedAzimuthDifferenceDeg(lead, requested) : null;
  const elevationAngle = isLrppo ? elevationDifferenceDeg(lead, requested) : null;
  const targetEstimate = estimatePosition(telemetry);
  const targetVelocity = estimateVelocity(telemetry);
  const truth = telemetry ? telemetry.decision_policy_visible.enemy_detected
    ? episode.frames[telemetry.decision_step]?.privileged_diagnostics.attacker_position_m ?? null
    : null : null;
  const truthVelocity = telemetry ? telemetry.decision_policy_visible.enemy_detected
    ? episode.frames[telemetry.decision_step]?.privileged_diagnostics.attacker_velocity_mps ?? null
    : null : null;
  const trackingError = euclideanDistance(targetEstimate, truth);
  const velocityError = euclideanDistance(targetVelocity, truthVelocity);
  const show = (key: keyof DiagnosticVisibility, checked: boolean) => onVisibility({ ...visibility, [key]: checked });
  return (
    <section className="policy-inspector" aria-label="Policy decision inspector">
      <details open>
        <summary>Policy Decision Inspector</summary>
        <div className="inspector-content">
          <div className="information-mode">
            <span>Research display mode</span>
            <div role="group" aria-label="Research display mode">
              <button type="button" aria-pressed={informationMode === 'ground-truth'} onClick={() => onInformationMode('ground-truth')}>Ground Truth</button>
              <button type="button" aria-pressed={informationMode === 'policy-visible'} onClick={() => onInformationMode('policy-visible')}>Policy-Visible</button>
            </div>
            <p>{informationMode === 'policy-visible'
              ? 'Partial policy-visible reconstruction: recorded state, target observations, and sector features are available; full local sensing geometry and the complete controller observation scene are not.'
              : 'Researcher privileged view. Actual attacker state is shown separately from policy-visible estimate/measurement data.'}</p>
          </div>
          <dl className="inspector-grid">
            <DataField label="Controller" value={readablePolicy(episode.defender.policy_id)} />
            <DataField label="Generation / training seed" value={`${episode.defender.generation ?? 'Not applicable'} / ${episode.defender.training_seed ?? 'Not applicable'}`} />
            <DataField label="Checkpoint role" value={episode.defender.checkpoint?.role ?? 'Not applicable / analytical'} />
            <DataField label="Playback state time" value={formatNumber(currentTimeS, 's', 3)} />
            <DataField label="Displayed state frame" value={`${frame.step} / ${episode.summary.steps}`} />
            <DataField label="Decision-time detection" value={telemetry ? (telemetry.decision_policy_visible.enemy_detected ? 'Detected' : 'Not detected') : (frame.policy_visible.enemy_detected ? 'Detected' : 'Not detected')} />
            <DataField label="Defender → attacker range" value={informationMode === 'ground-truth' ? formatNumber(frame.privileged_diagnostics.defender_attacker_range_m, 'm') : 'Unavailable: privileged separation is hidden in Policy-Visible mode.'} />
            <DataField label="Attacker → asset range" value={informationMode === 'ground-truth' ? formatNumber(frame.privileged_diagnostics.attacker_asset_range_m, 'm') : 'Unavailable: privileged separation is hidden in Policy-Visible mode.'} />
            <DataField label="Reward at displayed recorded frame" value={informationMode === 'ground-truth' ? formatNumber(frame.reward) : 'Unavailable: reward is not part of the policy-visible scene.'} title="No value is interpolated between recorded transitions." />
          </dl>

          <h3>Decision and action</h3>
          {telemetry && readout?.decision ? (
            <>
              <p className="decision-interval">Decision at step {readout.decision.decisionStep}; command applies during step {readout.decision.decisionStep} → {readout.decision.decisionStep + 1} ({readout.decision.startFrame.time_s.toFixed(3)}–{readout.decision.endFrame.time_s.toFixed(3)} s).</p>
              <dl className="inspector-grid">
                <DataField label="Requested action (recorded)" value={formatVector(telemetry.requested_action, 'direction')} />
                <DataField label="Environment desired direction (recorded)" value={formatVector(telemetry.desired_command, 'direction')} />
                <DataField label="Safety-adjusted direction (recorded)" value={formatVector(telemetry.safe_command, 'direction')} />
                <DataField label="Actual post-transition velocity (recorded)" value={formatVector(telemetry.executed_velocity_mps, 'm/s')} />
                <DataField label="Action disposition" value={telemetry.controller_action_executed ? 'Controller action applied' : 'Decision recorded, but not applied (simulator standby)'} />
                <DataField label="Current reward (recorded transition)" value={informationMode === 'ground-truth' ? formatNumber(readout.decision.endFrame.reward) : 'Unavailable: reward is not part of the policy-visible scene.'} />
                <DataField label="Consecutive requested-action change" value={formatNumber(diagnosticSeries?.requestedActionTurn.find(item => item.step === currentReadout?.decision?.decisionStep)?.value, '°')} />
              </dl>
            </>
          ) : (
            <p className="unavailable">Unavailable: no policy decision is recorded for this selected terminal time/state.</p>
          )}

          {isLrppo && (
            <>
              <h3>LR-PPO native 3D correction</h3>
              <dl className="inspector-grid">
                <DataField label="Analytical Lead direction" value={formatVector(lead, 'unit direction')} />
                <DataField label="Final requested direction" value={formatVector(requested, 'unit direction')} />
                <DataField label="Actual post-transition velocity" value={formatVector(telemetry?.executed_velocity_mps, 'm/s')} />
                <DataField label="Native 3D Lead → requested angle" value={formatNumber(nativeAngle, '°')} title="Nonnegative angle calculated from the recorded Lead and final requested vectors." />
                <DataField label="Canonical maximum 3D correction" value={`${telemetry?.lrppo?.max_adjustment_deg ?? 30}° (verified implementation limit)`} />
                <DataField label="Signed horizontal azimuth difference" value={formatNumber(horizontalAngle, '°')} />
                <DataField label="Elevation difference (derived)" value={formatNumber(elevationAngle, '°')} />
                <DataField label="Raw PPO residual" value={`Unavailable: ${fieldReason(telemetry, 'telemetry.lrppo.residual_action', 'not exposed by the recorded wrapper')}`} />
                <DataField label="Internal residual clipping" value={`Unavailable: ${fieldReason(telemetry, 'telemetry.lrppo.residual_norm_clipped', 'not exposed by the recorded wrapper')}`} />
                <DataField label="Observed safety override" value={telemetry?.avoidance_active ? `Active / desired-to-safe ${formatNumber(telemetry.desired_to_safe_angle_deg, '°')}` : telemetry ? 'No avoidance override recorded' : 'Unavailable'} />
              </dl>
              <div className="diagnostic-toggles" aria-label="LR-PPO guidance overlays">
                <Toggle label="Lead vector" checked={visibility.lead} onChange={value => show('lead', value)} />
                <Toggle label="Requested direction" checked={visibility.requested} onChange={value => show('requested', value)} />
                <Toggle label="Actual velocity" checked={visibility.executed} onChange={value => show('executed', value)} />
                <Toggle label="30° correction cone" checked={visibility.cone} onChange={value => show('cone', value)} />
              </div>
              <p className="telemetry-note">Direction arrows are equal-length normalized indicators, not command magnitudes or vehicle motion distances.</p>
            </>
          )}

          {isKalman && (
            <>
              <h3>Policy-visible target / privileged truth</h3>
              {telemetry?.decision_policy_visible.enemy_detected ? (
                <dl className="inspector-grid">
                  <DataField label={`${telemetry.decision_policy_visible.estimator === 'kalman' ? 'Kalman estimate' : 'Recorded measurement'} position`} value={formatVector(targetEstimate, 'm')} />
                  <DataField label="Estimated velocity" value={formatVector(targetVelocity, 'm/s')} />
                  <DataField label="Privileged ground-truth position" value={informationMode === 'ground-truth' ? formatVector(truth, 'm') : 'Unavailable: hidden in Policy-Visible mode; never supplied to the policy.'} title="Research-only; not provided to defender policy." />
                  <DataField label="Privileged ground-truth velocity" value={informationMode === 'ground-truth' ? formatVector(truthVelocity, 'm/s') : 'Unavailable: hidden in Policy-Visible mode; never supplied to the policy.'} title="Research-only; not provided to defender policy." />
                  <DataField label="Derived Euclidean tracking error" value={informationMode === 'ground-truth' ? formatNumber(trackingError, 'm') : 'Unavailable: requires privileged truth.'} />
                  <DataField label="Derived velocity-estimate error" value={informationMode === 'ground-truth' ? formatNumber(velocityError, 'm/s') : 'Unavailable: requires privileged truth.'} />
                </dl>
              ) : (
                <p className="unavailable">Policy-visible target estimate unavailable before detection. Privileged truth remains separate.</p>
              )}
              <div className="diagnostic-toggles" aria-label="Kalman and tracking overlays">
                <Toggle label="Policy-visible estimate" checked={visibility.estimate} onChange={value => show('estimate', value)} />
                {informationMode === 'ground-truth' && <Toggle label="Ground-truth marker" checked={visibility.truthMarker} onChange={value => show('truthMarker', value)} />}
                {informationMode === 'ground-truth' && <Toggle label="Tracking-error vector" checked={visibility.trackingError} onChange={value => show('trackingError', value)} />}
                <Toggle label="Estimated velocity" checked={visibility.estimateVelocity} onChange={value => show('estimateVelocity', value)} />
              </div>
              <p className="telemetry-note">The policy receives the estimate/measurement channel only. Ground truth and derived errors are researcher diagnostics.</p>
            </>
          )}

          {!isLrppo && !isKalman && telemetry?.lead && (
            <>
              <h3>Lead guidance (recorded)</h3>
              <dl className="inspector-grid">
                <DataField label="Guidance mode" value={telemetry.lead.guidance_mode ?? 'Unavailable'} />
                <DataField label="Target position" value={formatVector(telemetry.lead.target_position_m, 'm')} />
                <DataField label="Target velocity" value={formatVector(telemetry.lead.target_velocity_mps, 'm/s')} />
                <DataField label="Intercept point" value={formatVector(telemetry.lead.intercept_point_m, 'm')} />
                <DataField label="Intercept time" value={formatNumber(telemetry.lead.intercept_time_s, 's')} />
              </dl>
            </>
          )}
          {!isLrppo && telemetry?.pn && (
            <>
              <h3>Proportional-navigation guidance (recorded)</h3>
              <dl className="inspector-grid">
                <DataField label="Guidance mode" value={telemetry.pn.guidance_mode ?? 'Unavailable'} />
                <DataField label="Range" value={formatNumber(telemetry.pn.range_m, 'm')} />
                <DataField label="Closing speed" value={formatNumber(telemetry.pn.closing_speed_mps, 'm/s')} />
                <DataField label="LOS unit vector" value={formatVector(telemetry.pn.los_unit, 'unit direction')} />
                <DataField label="LOS rate" value={formatVector(telemetry.pn.los_rate_radps, 'rad/s')} />
                <DataField label="PN acceleration command" value={formatVector(telemetry.pn.acceleration_mps2, 'm/s²')} />
              </dl>
            </>
          )}
          {!isLrppo && telemetry?.rmpc && (
            <>
              <h3>RMPC diagnostics (recorded)</h3>
              <dl className="inspector-grid">
                <DataField label="Guidance / initialization mode" value={`${telemetry.rmpc.guidance_mode ?? 'Unavailable'} / ${telemetry.rmpc.initialization_mode ?? 'Unavailable'}`} />
                <DataField label="Robust score" value={formatNumber(telemetry.rmpc.robust_score)} />
                <DataField label="Objective evaluations" value={telemetry.rmpc.objective_evaluations ?? 'Unavailable'} />
                <DataField label="Decision time" value={formatNumber(telemetry.rmpc.decision_time_s, 's')} />
                <DataField label="Scenario scores" value={telemetry.rmpc.scenario_scores ? JSON.stringify(telemetry.rmpc.scenario_scores) : 'Unavailable'} />
              </dl>
            </>
          )}

          <h3>Safety, obstacle arbitration, and collisions</h3>
          <dl className="inspector-grid">{safetyFields(frame, currentReadout?.decision?.endFrame ?? null, telemetry, informationMode)}</dl>
          <h3>Recorded controller parameters</h3>
          <dl className="inspector-grid">{policyParameterRows(episode)}</dl>

          <h3>Scene information controls</h3>
          {!isLrppo && <p className="muted">LR-PPO-specific guidance overlays are unavailable for this controller.</p>}
          {!isKalman && <p className="muted">Kalman estimate and tracking overlays are unavailable for this estimator.</p>}
          <div className="diagnostic-toggles">
            {isKalman && <Toggle label="Show estimate in scene" checked={visibility.estimate} onChange={value => show('estimate', value)} />}
            {isKalman && <Toggle label="Show truth in scene" checked={visibility.truthMarker} onChange={value => show('truthMarker', value)} />}
            <Toggle label="Show safety intervention marker" checked={visibility.safetyMarker} onChange={value => show('safetyMarker', value)} />
          </div>
          {informationMode === 'policy-visible' && (
            <p className="unavailable">Partial reconstruction only: static geometry is filtered to map-known obstacles. Local sensing geometry, exact perception occlusion, and the complete policy-visible scene are not recorded by this replay schema.</p>
          )}
        </div>
      </details>
    </section>
  );
}

interface ChartLine {
  label: string;
  points: SeriesPoint[];
  color: string;
}

const emptyEvents: ReplayEvent[] = [];
const lrPpoAngleReference = { value: 30, label: '30° canonical maximum' };

function pathFor(points: SeriesPoint[], durationS: number, min: number, max: number): string {
  const width = 760;
  const height = 110;
  const left = 35;
  const top = 12;
  const finite = points.filter(item => item.value !== null);
  if (finite.length < 2) return '';
  const range = max - min || 1;
  let active = false;
  const commands: string[] = [];
  for (const item of points) {
    if (item.value === null) {
      active = false;
      continue;
    }
    const x = left + Math.max(0, Math.min(1, item.timeS / durationS)) * width;
    const y = top + height - ((item.value - min) / range) * height;
    commands.push(`${active ? 'L' : 'M'}${x.toFixed(2)},${y.toFixed(2)}`);
    active = true;
  }
  return commands.join(' ');
}

function SeriesChart({
  title,
  unit,
  lines,
  events,
  timeS,
  durationS,
  onSeek,
  reference,
}: {
  title: string;
  unit: string;
  lines: ChartLine[];
  events: ReplayEvent[];
  timeS: number;
  durationS: number;
  onSeek: (timeS: number) => void;
  reference?: { value: number; label: string };
}) {
  const chartModel = useMemo(() => {
    const values = lines.flatMap(line => line.points.flatMap(item => item.value === null ? [] : [item.value]));
    if (reference) values.push(reference.value);
    const rawMin = values.length ? Math.min(...values) : 0;
    const rawMax = values.length ? Math.max(...values) : 1;
    const padding = (rawMax - rawMin || Math.max(1, Math.abs(rawMax) * 0.05)) * 0.08;
    const min = rawMin - padding;
    const max = rawMax + padding;
    return {
      hasValues: values.length > 0,
      min,
      max,
      referenceY: reference ? 12 + 110 - ((reference.value - min) / (max - min || 1)) * 110 : null,
      paths: lines.map(line => pathFor(line.points, durationS, min, max)),
    };
  }, [durationS, lines, reference]);
  const eventMarkers = useMemo(() => events.map(event => ({
    event,
    x: 35 + Math.max(0, Math.min(1, event.time_s / Math.max(durationS, 1e-9))) * 760,
  })), [durationS, events]);
  const cursorX = 35 + Math.max(0, Math.min(1, timeS / Math.max(durationS, 1e-9))) * 760;
  function seekFromPointer(event: ReactMouseEvent<SVGSVGElement> | ReactPointerEvent<SVGSVGElement>) {
    const rect = event.currentTarget.getBoundingClientRect();
    const chartX = (event.clientX - rect.left) / rect.width * 830;
    const normalized = Math.max(0, Math.min(1, (chartX - 35) / 760));
    onSeek(normalized * durationS);
  }
  return (
    <article className="diagnostic-chart">
      <h3>{title} <span>({unit}, recorded samples)</span></h3>
      {!chartModel.hasValues ? <p className="unavailable">Unavailable: this replay has no valid samples for this chart.</p> : (
        <>
          <svg
            viewBox="0 0 830 150"
            role="img"
            aria-label={`${title} against simulation time`}
            data-testid="diagnostic-chart"
            data-chart-title={title}
            onClick={seekFromPointer}
            onPointerDown={event => {
              event.currentTarget.setPointerCapture(event.pointerId);
              seekFromPointer(event);
            }}
            onPointerMove={event => {
              if (event.buttons > 0) seekFromPointer(event);
            }}
            onPointerUp={event => {
              if (event.currentTarget.hasPointerCapture(event.pointerId)) {
                event.currentTarget.releasePointerCapture(event.pointerId);
              }
            }}
          >
            {[0, 0.5, 1].map(fraction => {
              const y = 12 + fraction * 110;
              return <line key={fraction} x1="35" x2="795" y1={y} y2={y} className="chart-grid" />;
            })}
            {reference && chartModel.referenceY !== null && (
              <g>
                <line x1="35" x2="795" y1={chartModel.referenceY} y2={chartModel.referenceY} className="chart-reference" />
                <text x="42" y={chartModel.referenceY - 3} className="chart-reference-label">{reference.label}</text>
              </g>
            )}
            {lines.map((line, index) => (
              <path key={line.label} d={chartModel.paths[index]} fill="none" stroke={line.color} strokeWidth="2" vectorEffect="non-scaling-stroke" />
            ))}
            {eventMarkers.map(({ event, x }, index) => {
              return <line key={`${event.kind}-${event.step}-${index}`} x1={x} x2={x} y1="12" y2="122" className={`chart-event ${event.kind}`} />;
            })}
            <line x1={cursorX} x2={cursorX} y1="8" y2="126" className="chart-cursor" />
            <text x="35" y="144" className="chart-axis-label">0 s</text>
            <text x="795" y="144" textAnchor="end" className="chart-axis-label">{durationS.toFixed(1)} s</text>
          </svg>
          <div className="chart-legend">
            {lines.map(line => <span key={line.label}><i style={{ background: line.color }} />{line.label}</span>)}
            {reference && <span><i className="reference-swatch" />{reference.label}</span>}
            <span><i className="event-swatch" />Recorded events</span>
          </div>
        </>
      )}
    </article>
  );
}

export function DiagnosticCharts({
  episode,
  series,
  timeS,
  onSeek,
  informationMode,
}: {
  episode: ReplayEpisode | null;
  series: DiagnosticSeries | null;
  timeS: number;
  onSeek: (timeS: number) => void;
  informationMode: InformationMode;
}) {
  const distanceLines = useMemo<ChartLine[]>(() => series ? [
    { label: 'Defender → attacker', points: series.defenderAttackerDistance, color: '#65acff' },
    { label: 'Attacker → asset', points: series.attackerAssetDistance, color: '#fa777d' },
  ] : [], [series]);
  const rewardLines = useMemo<ChartLine[]>(() => series
    ? [{ label: 'Reward', points: series.reward, color: '#65c998' }]
    : [], [series]);
  const cumulativeRewardLines = useMemo<ChartLine[]>(() => series
    ? [{ label: 'Cumulative reward', points: series.cumulativeReward, color: '#a78bfa' }]
    : [], [series]);
  const leadAngleLines = useMemo<ChartLine[]>(() => series
    ? [{ label: '3D angular separation', points: series.leadRequestedAngle, color: '#c084fc' }]
    : [], [series]);
  const horizontalAngleLines = useMemo<ChartLine[]>(() => series
    ? [{ label: 'Horizontal azimuth', points: series.horizontalAzimuth, color: '#f0abfc' }]
    : [], [series]);
  const executionAngleLines = useMemo<ChartLine[]>(() => series ? [
    { label: 'Desired → safe direction', points: series.requestedSafeAngle, color: '#f59e0b' },
    { label: 'Requested direction → actual velocity', points: series.requestedVelocityAngle, color: '#22d3ee' },
  ] : [], [series]);
  const trackingErrorLines = useMemo<ChartLine[]>(() => series
    ? [{ label: 'Euclidean estimate → truth error', points: series.trackingError, color: '#c084fc' }]
    : [], [series]);
  const velocityErrorLines = useMemo<ChartLine[]>(() => series
    ? [{ label: 'Euclidean estimate → truth error', points: series.velocityEstimateError, color: '#60a5fa' }]
    : [], [series]);
  const actionTurnLines = useMemo<ChartLine[]>(() => series
    ? [{ label: '3D requested action turn', points: series.requestedActionTurn, color: '#f59e0b' }]
    : [], [series]);
  const chartEvents = useMemo(() => episode && informationMode === 'ground-truth' ? episode.events : emptyEvents, [episode, informationMode]);
  if (!episode || !series) return null;
  const policyId = episode.defender.policy_id;
  const isLrppo = policyId === 'lr_ppo';
  const isKalman = episode.defender.estimator === 'kalman';
  const charts: ReactNode[] = informationMode === 'ground-truth' ? [
    <SeriesChart key="distance" title="Recorded separation" unit="m" lines={distanceLines} events={chartEvents} timeS={timeS} durationS={episode.summary.duration_s} onSeek={onSeek} />,
    <SeriesChart key="reward" title="Transition reward" unit="reward" lines={rewardLines} events={chartEvents} timeS={timeS} durationS={episode.summary.duration_s} onSeek={onSeek} />,
    <SeriesChart key="cumulative" title="Cumulative recorded reward" unit="reward" lines={cumulativeRewardLines} events={chartEvents} timeS={timeS} durationS={episode.summary.duration_s} onSeek={onSeek} />,
  ] : [];
  if (isLrppo) {
    charts.push(
      <SeriesChart key="lr-angle" title="LR-PPO native 3D Lead → requested angle" unit="degrees" lines={leadAngleLines} reference={lrPpoAngleReference} events={chartEvents} timeS={timeS} durationS={episode.summary.duration_s} onSeek={onSeek} />,
      <SeriesChart key="lr-horizontal" title="LR-PPO horizontal azimuth difference" unit="signed degrees" lines={horizontalAngleLines} events={chartEvents} timeS={timeS} durationS={episode.summary.duration_s} onSeek={onSeek} />,
      <SeriesChart key="lr-execution" title="Requested direction vs execution" unit="degrees" lines={executionAngleLines} events={chartEvents} timeS={timeS} durationS={episode.summary.duration_s} onSeek={onSeek} />,
    );
  } else if (isKalman && informationMode === 'ground-truth') {
    charts.push(
      <SeriesChart key="kalman-position" title="Derived target position tracking error" unit="m" lines={trackingErrorLines} events={chartEvents} timeS={timeS} durationS={episode.summary.duration_s} onSeek={onSeek} />,
      <SeriesChart key="kalman-velocity" title="Derived target velocity estimate error" unit="m/s" lines={velocityErrorLines} events={chartEvents} timeS={timeS} durationS={episode.summary.duration_s} onSeek={onSeek} />,
      <SeriesChart key="kalman-turn" title="Requested action change between decisions" unit="degrees" lines={actionTurnLines} events={chartEvents} timeS={timeS} durationS={episode.summary.duration_s} onSeek={onSeek} />,
    );
  } else {
    charts.push(
      <SeriesChart key="action-turn" title="Requested action change between decisions" unit="degrees" lines={actionTurnLines} events={chartEvents} timeS={timeS} durationS={episode.summary.duration_s} onSeek={onSeek} />,
    );
  }
  return (
    <section className="diagnostic-charts panel" aria-label="Synchronized telemetry charts">
      <div className="diagnostic-charts-heading">
        <div><span className="eyebrow">RECORDED DECISION TELEMETRY</span><h2>Synchronized charts</h2></div>
        <p>{informationMode === 'ground-truth'
          ? 'Click any chart to seek. Cursor time follows playback; decision series contain only actual policy decisions.'
          : 'Policy-Visible mode omits privileged ranges, outcomes, and estimate-to-truth diagnostics.'}</p>
      </div>
      <div className="diagnostic-chart-grid">{charts}</div>
    </section>
  );
}
