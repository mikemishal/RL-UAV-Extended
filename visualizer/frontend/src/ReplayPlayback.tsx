import { useEffect, useMemo, useRef, useState } from 'react';
import { Battlefield } from './Battlefield';
import type { CameraMode, TrailVisibility } from './Battlefield';
import type { ReplayEpisode, ReplayFrame, ReplaySummary } from './api';
import { sampleAtTime } from './playback';

const playbackSpeeds = [0.25, 0.5, 1, 2, 4, 8] as const;

export interface PlaybackReadout {
  replayId: string;
  frame: ReplayFrame;
  interpolated: boolean;
}

interface ReplayPlaybackProps {
  replayId: string | null;
  episode: ReplayEpisode | null;
  summary: ReplaySummary | null;
  traceLoading: boolean;
  traceError: string | null;
  episodeNumber: number;
  episodeCount: number;
  onSelectPrevious: () => void;
  onSelectNext: () => void;
  onReadout: (readout: PlaybackReadout | null) => void;
  onSceneError: (message: string) => void;
}

function clockLabel(seconds: number): string {
  const minutes = Math.floor(seconds / 60);
  const remainder = seconds - minutes * 60;
  return `${minutes.toString().padStart(2, '0')}:${remainder.toFixed(1).padStart(4, '0')}`;
}

export function ReplayPlayback({
  replayId,
  episode,
  summary,
  traceLoading,
  traceError,
  episodeNumber,
  episodeCount,
  onSelectPrevious,
  onSelectNext,
  onReadout,
  onSceneError,
}: ReplayPlaybackProps) {
  const [timeS, setTimeS] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState<number>(1);
  const [loop, setLoop] = useState(false);
  const [cameraMode, setCameraMode] = useState<CameraMode>('free');
  const [previous3dMode, setPrevious3dMode] = useState<CameraMode>('free');
  const [resetCameraToken, setResetCameraToken] = useState(0);
  const [showFullTrajectory, setShowFullTrajectory] = useState(false);
  const [showDetectionVolume, setShowDetectionVolume] = useState(true);
  const [wireframeObstacles, setWireframeObstacles] = useState(false);
  const [transparentObstacles, setTransparentObstacles] = useState(false);
  const [trails, setTrails] = useState<TrailVisibility>({
    defender: true,
    attacker: true,
    asset: true,
    obstacles: false,
  });
  const sample = useMemo(() => episode ? sampleAtTime(episode, timeS) : null, [episode, timeS]);
  const duration = episode?.summary.duration_s ?? 0;
  const dt = episode?.environment.dt_s ?? 0.5;
  const lastTick = useRef<number | null>(null);
  const readoutRef = useRef(onReadout);
  readoutRef.current = onReadout;

  useEffect(() => {
    setTimeS(0);
    setPlaying(false);
    setSpeed(1);
    setCameraMode('free');
    setPrevious3dMode('free');
    setResetCameraToken(value => value + 1);
    setShowFullTrajectory(false);
    readoutRef.current(null);
  }, [replayId]);

  useEffect(() => {
    if (!episode || !sample || !replayId) {
      readoutRef.current(null);
      return;
    }
    readoutRef.current({
      replayId,
      frame: episode.frames[sample.lowerIndex],
      interpolated: sample.fraction > 0,
    });
  }, [episode, replayId, sample]);

  useEffect(() => {
    if (!playing || !episode) {
      lastTick.current = null;
      return;
    }
    let requestId = 0;
    const tick = (now: number) => {
      const previous = lastTick.current;
      lastTick.current = now;
      if (previous !== null) {
        const elapsedSimulatedSeconds = Math.max(0, (now - previous) / 1000) * speed;
        setTimeS(current => {
          const next = current + elapsedSimulatedSeconds;
          return next >= duration ? (loop ? 0 : duration) : next;
        });
      }
      requestId = window.requestAnimationFrame(tick);
    };
    requestId = window.requestAnimationFrame(tick);
    return () => window.cancelAnimationFrame(requestId);
  }, [duration, episode, loop, playing, speed]);

  useEffect(() => {
    if (!loop && timeS >= duration && playing) setPlaying(false);
  }, [duration, loop, playing, timeS]);

  useEffect(() => {
    if (traceError) setPlaying(false);
  }, [traceError]);

  const lowerIndex = sample?.lowerIndex ?? 0;
  const atBeginning = !episode || timeS <= 0;
  const atEnd = !episode || timeS >= duration;
  const terminalReached = Boolean(episode && atEnd);

  function setCamera(next: CameraMode) {
    if (next === 'top-down') {
      if (cameraMode !== 'top-down') setPrevious3dMode(cameraMode);
      setCameraMode(next);
      return;
    }
    setCameraMode(next);
  }

  function goTopDown() {
    if (cameraMode !== 'top-down') setPrevious3dMode(cameraMode);
    setCameraMode('top-down');
  }

  function resetCamera() {
    setResetCameraToken(value => value + 1);
  }

  function seekToFrame(index: number) {
    if (!episode) return;
    const safeIndex = Math.max(0, Math.min(episode.frames.length - 1, index));
    setTimeS(episode.frames[safeIndex].time_s);
  }

  function setTrail(key: keyof TrailVisibility, checked: boolean) {
    setTrails(current => ({ ...current, [key]: checked }));
  }

  const selectedOutcome = summary?.summary.outcome ?? episode?.summary.outcome;

  return (
    <>
      <Battlefield
        episode={episode}
        timeS={timeS}
        cameraMode={cameraMode}
        resetCameraToken={resetCameraToken}
        showFullTrajectory={showFullTrajectory}
        trails={trails}
        showDetectionVolume={showDetectionVolume}
        wireframeObstacles={wireframeObstacles}
        transparentObstacles={transparentObstacles}
        onWebGlError={onSceneError}
        onResetCamera={resetCamera}
      />
      {traceLoading && <p className="trace-loading" role="status">Loading and validating recorded replay…</p>}
      {traceError && <div className="inline-error" role="alert">Replay trace could not be loaded: {traceError}</div>}
      <section className="playback panel" aria-label="Playback controls">
        <div className="playback-heading">
          <div><span className="eyebrow">RECORDED PLAYBACK</span><h2>{episode ? `Episode ${episodeNumber} / ${episodeCount} · seed ${episode.simulation_seed}` : 'Choose a completed episode'}</h2></div>
          <span className="playback-clock" aria-live="polite">
            {episode ? `${clockLabel(timeS)} / ${clockLabel(duration)} · step ${sample ? episode.frames[sample.lowerIndex].step : 0}${sample?.fraction ? `→${episode.frames[sample.upperIndex].step}` : ''}` : 'No episode loaded'}
          </span>
        </div>
        <div className="playback-buttons" aria-label="Episode transport controls">
          <button type="button" aria-label="Previous episode" onClick={onSelectPrevious} disabled={episodeNumber <= 1}>Previous episode</button>
          <button type="button" aria-label="Jump to beginning" onClick={() => seekToFrame(0)} disabled={!episode || atBeginning}>Beginning</button>
          <button type="button" aria-label="Previous frame" onClick={() => seekToFrame(sample?.fraction ? lowerIndex : lowerIndex - 1)} disabled={!episode || atBeginning}>− Frame</button>
          <button type="button" aria-label={playing ? 'Pause' : 'Play'} onClick={() => setPlaying(value => !value)} disabled={!episode || traceLoading || Boolean(traceError) || (atEnd && !loop)}>
            {playing ? 'Pause' : 'Play'}
          </button>
          <button type="button" aria-label="Next frame" onClick={() => seekToFrame(lowerIndex + 1)} disabled={!episode || atEnd}>+ Frame</button>
          <button type="button" aria-label="Restart episode" onClick={() => { setPlaying(false); seekToFrame(0); }} disabled={!episode}>Restart</button>
          <button type="button" aria-label="Jump to end" onClick={() => seekToFrame(episode ? episode.frames.length - 1 : 0)} disabled={!episode || atEnd}>End</button>
          <button type="button" aria-label="Next episode" onClick={onSelectNext} disabled={episodeNumber >= episodeCount}>Next episode</button>
          <label className="speed-select">
            Speed
            <select aria-label="Playback speed" value={speed} onChange={event => setSpeed(Number(event.target.value))} disabled={!episode}>
              {playbackSpeeds.map(value => <option key={value} value={value}>{value}×</option>)}
            </select>
          </label>
          <label className="loop-toggle"><input type="checkbox" checked={loop} onChange={event => setLoop(event.target.checked)} /> Loop</label>
        </div>
        <label className="timeline-label">
          <span>Timeline</span><span>{sample ? `${sample.lowerIndex + 1} / ${episode?.frames.length ?? 0} frames` : '—'}</span>
          <input
            aria-label="Episode timeline"
            data-testid="episode-timeline"
            type="range"
            min="0"
            max={duration}
            step={Math.max(dt / 100, 0.001)}
            value={Math.min(timeS, duration)}
            disabled={!episode || traceLoading}
            onChange={event => setTimeS(Number(event.target.value))}
          />
        </label>
        {episode && (
          <>
            <div className="viewer-options" aria-label="Scene display options">
              <div className="camera-presets" role="group" aria-label="Camera mode">
                <button type="button" aria-pressed={cameraMode === 'free'} onClick={() => setCamera('free')}>Free 3D</button>
                <button type="button" aria-pressed={cameraMode === 'follow-defender'} onClick={() => setCamera('follow-defender')}>Follow defender</button>
                <button type="button" aria-pressed={cameraMode === 'follow-attacker'} onClick={() => setCamera('follow-attacker')}>Follow attacker</button>
                <button type="button" aria-pressed={cameraMode === 'top-down'} onClick={goTopDown}>Top-down</button>
                {cameraMode === 'top-down' && (
                  <button type="button" onClick={() => setCamera(previous3dMode === 'top-down' ? 'free' : previous3dMode)}>Return to 3D</button>
                )}
              </div>
              <div className="scene-toggles" role="group" aria-label="Scene visibility">
                <label><input type="checkbox" checked={trails.defender && trails.attacker && trails.asset} onChange={event => {
                  const checked = event.target.checked;
                  setTrails(current => ({ ...current, defender: checked, attacker: checked, asset: checked }));
                }} /> All trajectories</label>
                <label><input type="checkbox" checked={trails.defender} onChange={event => setTrail('defender', event.target.checked)} /> Defender trail</label>
                <label><input type="checkbox" checked={trails.attacker} onChange={event => setTrail('attacker', event.target.checked)} /> Attacker trail</label>
                <label><input type="checkbox" checked={trails.asset} onChange={event => setTrail('asset', event.target.checked)} /> Asset trail</label>
                <label><input type="checkbox" checked={trails.obstacles} onChange={event => setTrail('obstacles', event.target.checked)} /> Moving-obstacle trails</label>
                <label><input type="checkbox" checked={showFullTrajectory} onChange={event => setShowFullTrajectory(event.target.checked)} /> Show full trajectory</label>
                <label><input type="checkbox" checked={showDetectionVolume} onChange={event => setShowDetectionVolume(event.target.checked)} /> Detection volume</label>
                <label><input type="checkbox" checked={wireframeObstacles} onChange={event => setWireframeObstacles(event.target.checked)} /> Wireframe obstacles</label>
                <label><input type="checkbox" checked={transparentObstacles} onChange={event => setTransparentObstacles(event.target.checked)} /> Transparent obstacles</label>
              </div>
            </div>
            <div className="playback-status" aria-live="polite">
              <span>{episode.frames[lowerIndex].policy_visible.enemy_detected ? `Detected at ${episode.summary.detection_time_s?.toFixed(1) ?? 'recorded'} s` : 'Not detected at selected recorded frame'}</span>
              <span>{terminalReached ? `Terminal outcome: ${episode.summary.outcome} · ${duration.toFixed(1)} simulated s` : `Recorded outcome known: ${selectedOutcome ?? 'unavailable'} (not yet reached)`}</span>
              <span>Continuous positions between recorded samples are display-only interpolation; step controls select exact recorded frames.</span>
            </div>
          </>
        )}
      </section>
    </>
  );
}
