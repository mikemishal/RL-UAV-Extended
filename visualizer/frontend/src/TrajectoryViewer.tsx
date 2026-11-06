/**
 * Reusable top-down trajectory player (brief: "Reuse the existing trajectory viewer. Do not redesign simulation
 * rendering unless necessary for compatibility."). Extracted from the SAME rendering approach and CSS classes
 * already used by `CbfDefenderReview.tsx` (`cbf-review-scene`/`cbf-review-trail-*`/`cbf-review-marker-*`, see
 * `styles.css`) so the simplified episode viewer looks and behaves consistently with the existing A1-CBF review
 * viewer, without modifying that existing, independently-tested component at all.
 */

import { useEffect, useMemo, useRef, useState } from 'react';
import type { EpisodeObstacle, EpisodeReplayStep, Vec3 } from './episodes';

const WORLD_HALF_EXTENT_M = 60;
const SVG_SIZE = 560;
const BASE_INTERVAL_MS = 400;

function worldToSvg(x: number, y: number): [number, number] {
  const sx = ((x + WORLD_HALF_EXTENT_M) / (2 * WORLD_HALF_EXTENT_M)) * SVG_SIZE;
  const sy = ((WORLD_HALF_EXTENT_M - y) / (2 * WORLD_HALF_EXTENT_M)) * SVG_SIZE;
  return [sx, sy];
}

export interface TrajectoryViewerProps {
  steps: EpisodeReplayStep[];
  obstacles: EpisodeObstacle[];
  /** Reset the viewer to step 0 whenever this changes (e.g. a new episode was loaded). */
  resetKey: string;
}

const SPEED_OPTIONS = [0.5, 1, 2, 4] as const;

export function TrajectoryViewer({ steps, obstacles, resetKey }: TrajectoryViewerProps) {
  const [stepIndex, setStepIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState<(typeof SPEED_OPTIONS)[number]>(1);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    setStepIndex(0);
    setPlaying(false);
  }, [resetKey]);

  useEffect(() => {
    if (!playing || steps.length === 0) {
      if (timerRef.current) clearInterval(timerRef.current);
      return;
    }
    timerRef.current = setInterval(() => {
      setStepIndex(previous => {
        if (previous >= steps.length - 1) {
          setPlaying(false);
          return previous;
        }
        return previous + 1;
      });
    }, BASE_INTERVAL_MS / speed);
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [playing, steps, speed]);

  const clampedIndex = Math.min(stepIndex, Math.max(steps.length - 1, 0));
  const step: EpisodeReplayStep | null = steps.length > 0 ? steps[clampedIndex] : null;

  const trail = useMemo(() => {
    if (steps.length === 0) return { attacker: '', defender: '', asset: '' };
    const upTo = steps.slice(0, clampedIndex + 1);
    const toPoints = (pick: (s: EpisodeReplayStep) => Vec3) =>
      upTo.map(s => worldToSvg(pick(s)[0], pick(s)[1]).join(',')).join(' ');
    return {
      attacker: toPoints(s => s.attacker_pos),
      defender: toPoints(s => s.defender_pos),
      asset: toPoints(s => s.asset_pos),
    };
  }, [steps, clampedIndex]);

  if (steps.length === 0) {
    return (
      <div className="trajectory-viewer trajectory-viewer-empty" role="status">
        <p className="muted">No replay data to display.</p>
      </div>
    );
  }

  return (
    <div className="trajectory-viewer">
      <svg viewBox={`0 0 ${SVG_SIZE} ${SVG_SIZE}`} width="100%" height={SVG_SIZE} role="img" aria-label="Top-down trajectory view" className="cbf-review-scene">
        <rect x={0} y={0} width={SVG_SIZE} height={SVG_SIZE} className="cbf-review-scene-bg" />
        {obstacles.map((o, i) => {
          const [x0, y0] = worldToSvg(o.min[0], o.min[1]);
          const [x1, y1] = worldToSvg(o.max[0], o.max[1]);
          return <rect key={i} x={Math.min(x0, x1)} y={Math.min(y0, y1)} width={Math.abs(x1 - x0)} height={Math.abs(y1 - y0)} className="cbf-review-obstacle" />;
        })}
        <polyline points={trail.asset} className="cbf-review-trail cbf-review-trail-asset" />
        <polyline points={trail.defender} className="cbf-review-trail cbf-review-trail-defender" />
        <polyline points={trail.attacker} className="cbf-review-trail cbf-review-trail-attacker" />
        {step && (
          <>
            <circle cx={worldToSvg(step.asset_pos[0], step.asset_pos[1])[0]} cy={worldToSvg(step.asset_pos[0], step.asset_pos[1])[1]} r={4} className="cbf-review-marker cbf-review-marker-asset" />
            <circle cx={worldToSvg(step.defender_pos[0], step.defender_pos[1])[0]} cy={worldToSvg(step.defender_pos[0], step.defender_pos[1])[1]} r={5} className="cbf-review-marker cbf-review-marker-defender" />
            <circle cx={worldToSvg(step.attacker_pos[0], step.attacker_pos[1])[0]} cy={worldToSvg(step.attacker_pos[0], step.attacker_pos[1])[1]} r={6} className="cbf-review-marker cbf-review-marker-attacker" />
          </>
        )}
      </svg>
      <div className="cbf-review-transport" role="group" aria-label="Playback controls">
        <button type="button" onClick={() => setStepIndex(0)} aria-label="Restart">⏮</button>
        <button
          type="button"
          onClick={() => setPlaying(p => !p)}
          aria-label={playing ? 'Pause' : 'Play'}
          aria-pressed={playing}
        >
          {playing ? 'Pause' : 'Play'}
        </button>
        <input
          type="range"
          min={0}
          max={Math.max(steps.length - 1, 0)}
          value={clampedIndex}
          onChange={e => setStepIndex(Number(e.target.value))}
          aria-label="Timeline"
        />
        <span className="trajectory-viewer-step-count">{clampedIndex + 1} / {steps.length}</span>
        <label className="trajectory-viewer-speed">
          Speed
          <select value={speed} onChange={e => setSpeed(Number(e.target.value) as (typeof SPEED_OPTIONS)[number])} aria-label="Playback speed">
            {SPEED_OPTIONS.map(option => (
              <option key={option} value={option}>{option}x</option>
            ))}
          </select>
        </label>
      </div>
    </div>
  );
}
