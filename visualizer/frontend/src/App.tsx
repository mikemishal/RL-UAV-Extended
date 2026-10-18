import { lazy, Suspense, useCallback, useEffect, useMemo, useState } from 'react';
import { ReplayPlayback } from './ReplayPlayback';
import type { PlaybackReadout } from './ReplayPlayback';
import { PolicyInspector } from './PolicyDiagnostics';
import type { DiagnosticVisibility, InformationMode } from './PolicyDiagnostics';
import { deriveDiagnosticSeries } from './diagnostics';
import {
  ApiError,
  cancelReplayJob,
  createReplayJob,
  getHealth,
  getPolicies,
  getReplayJob,
  getReplay,
  getReplaySummary,
  getReplays,
  getScenarios,
  createSceneCapture,
} from './api';
import type {
  DefenderPolicy,
  PolicyCatalog,
  ReplayJob,
  ReplayJobRequest,
  ReplayBundle,
  ReplayListItem,
  ReplaySummary,
  ScenarioCatalog,
  ResearchBookmark,
  ResearchCase,
} from './api';
import { ResearchCases } from './ResearchCases';

type BackendStatus = { kind: 'checking' | 'ok' | 'degraded' | 'error'; message: string };
type EpisodeCount = 1 | 3 | 5 | 10;
const MAX_SEED = 2 ** 32 - 1;
const ComparisonExplorer = lazy(() => import('./ComparisonExplorer').then(module => ({ default: module.ComparisonExplorer })));
const defaultDiagnosticVisibility: DiagnosticVisibility = {
  lead: true,
  requested: true,
  executed: true,
  cone: true,
  estimate: true,
  truthMarker: true,
  trackingError: true,
  estimateVelocity: true,
  safetyMarker: true,
};

const parameterLabels: Record<string, string> = {
  v_d: 'Defender speed',
  v_e: 'Attacker speed',
  v_s: 'Asset speed',
  detection_radius: 'Detection radius',
  intercept_radius: 'Interception radius',
  threat_radius: 'Threat radius',
  unsafe_intercept_radius: 'Unsafe interception radius',
  dt: 'Time step',
  max_steps: 'Maximum steps',
  max_altitude: 'Maximum altitude',
  defender_max_accel: 'Defender maximum acceleration',
  defender_max_climb_rate: 'Maximum climb rate',
  defender_max_descent_rate: 'Maximum descent rate',
  defender_max_turn_rate_deg: 'Maximum turn rate',
  obstacle_count: 'Static obstacle count',
  dynamic_obstacle_count: 'Moving obstacle count',
  obstacle_min_size: 'Minimum building size',
  obstacle_max_size: 'Maximum building size',
  obstacle_min_height: 'Minimum building height',
  obstacle_max_height: 'Maximum building height',
  dynamic_obstacle_min_radius: 'Minimum moving-obstacle radius',
  dynamic_obstacle_max_radius: 'Maximum moving-obstacle radius',
};

const units: Record<string, string> = {
  v_d: 'm/s',
  v_e: 'm/s',
  v_s: 'm/s',
  detection_radius: 'm',
  intercept_radius: 'm',
  threat_radius: 'm',
  unsafe_intercept_radius: 'm',
  dt: 's',
  max_altitude: 'm',
  defender_max_accel: 'm/s²',
  defender_max_climb_rate: 'm/s',
  defender_max_descent_rate: 'm/s',
  defender_max_turn_rate_deg: '°/s',
  obstacle_min_size: 'm',
  obstacle_max_size: 'm',
  obstacle_min_height: 'm',
  obstacle_max_height: 'm',
  dynamic_obstacle_min_radius: 'm',
  dynamic_obstacle_max_radius: 'm',
};

const outcomeLabels: Record<string, string> = {
  intercepted: 'Intercepted',
  soldier_caught: 'Attacker reached protected asset',
  unsafe_intercept: 'Unsafe interception',
  timeout: 'Timeout',
  defender_static_obstacle_collision: 'Defender / static obstacle collision',
  defender_dynamic_obstacle_collision: 'Defender / moving obstacle collision',
  enemy_static_obstacle_collision: 'Attacker / static obstacle collision',
  enemy_dynamic_obstacle_collision: 'Attacker / moving obstacle collision',
};

function readableOutcome(outcome: string): string {
  return `${outcome} — ${outcomeLabels[outcome] ?? 'Recorded simulator outcome'}`;
}

function useBackendHealth(): BackendStatus {
  const [status, setStatus] = useState<BackendStatus>({ kind: 'checking', message: 'Checking backend' });
  useEffect(() => {
    let active = true;
    getHealth().then(data => {
      if (!active) return;
      if (data.phase !== 3 || data.service !== 'counter-uas-visualizer') {
        throw new Error('Backend returned an unexpected health response');
      }
      setStatus(data.status === 'ok'
        ? { kind: 'ok', message: 'Backend connected / research data root configured' }
        : { kind: 'degraded', message: 'Backend connected / research data root not configured' });
    }).catch(error => {
      if (!active) return;
      console.error('Backend health check failed', error);
      setStatus({ kind: 'error', message: 'Backend unavailable / start the localhost API on port 8000' });
    });
    return () => { active = false; };
  }, []);
  return status;
}

function formatParameter(value: unknown, key: string): string {
  if (typeof value !== 'number') return String(value);
  const digits = Math.abs(value) < 10 ? 2 : 1;
  return `${Number(value.toFixed(digits))}${units[key] ? ` ${units[key]}` : ''}`;
}

function isActive(job: ReplayJob | null): boolean {
  return job?.state === 'queued' || job?.state === 'running';
}

function DefenderControls({
  policies,
  policy,
  generation,
  trainingSeed,
  onPolicy,
  onGeneration,
  onTrainingSeed,
}: {
  policies: DefenderPolicy[];
  policy: DefenderPolicy | undefined;
  generation: string;
  trainingSeed: number;
  onPolicy: (policy: DefenderPolicy) => void;
  onGeneration: (generation: string) => void;
  onTrainingSeed: (seed: number) => void;
}) {
  const generations = policy?.generations.filter(item => item.available) ?? [];
  const selectedGeneration = generations.find(item => item.generation === generation) ?? generations[0];
  const seeds = selectedGeneration?.training_seeds.filter(item => item.available) ?? [];
  return (
    <>
      <label>
        Defender architecture / policy
        <select
          aria-label="Defender architecture"
          value={policy?.policy_id ?? ''}
          onChange={event => {
            const selected = policies.find(item => item.policy_id === event.target.value);
            if (selected) onPolicy(selected);
          }}
        >
          {policies.map(item => <option key={item.policy_id} value={item.policy_id}>{item.display_name}</option>)}
        </select>
      </label>
      {policy?.classification === 'learned' && (
        <>
          <label>
            Model generation
            <select
              aria-label="Defender generation"
              value={selectedGeneration?.generation ?? ''}
              onChange={event => onGeneration(event.target.value)}
            >
              {generations.map(item => <option key={item.generation} value={item.generation}>{item.display_name}</option>)}
            </select>
          </label>
          <label>
            Training seed
            <select
              aria-label="Defender training seed"
              value={seeds.some(item => item.training_seed === trainingSeed) ? trainingSeed : (seeds[0]?.training_seed ?? '')}
              onChange={event => onTrainingSeed(Number(event.target.value))}
            >
              {seeds.map(item => (
                <option key={item.training_seed} value={item.training_seed}>
                  Seed {item.training_seed} / {item.checkpoint_role}
                </option>
              ))}
            </select>
          </label>
        </>
      )}
    </>
  );
}

function EpisodeSelection({
  items,
  selectedId,
  onSelect,
  informationMode,
}: {
  items: { id: string; seed: number; defender: string; attacker: string; clutter: string; outcome: string; steps: number; duration: number; detection: number | null; reward: number; collision: string | null; intercept: number | null }[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  informationMode: InformationMode;
}) {
  if (!items.length) return <p className="muted">No completed replay episodes are available yet.</p>;
  return (
    <div className="episode-list" role="list" aria-label="Episode results">
      {items.map((item, index) => (
        <button
          className={`episode-result ${selectedId === item.id ? 'selected' : ''}`}
          data-testid="episode-result"
          data-replay-id={item.id}
          key={item.id}
          onClick={() => onSelect(item.id)}
          role="listitem"
          type="button"
        >
          <span className="episode-result-heading">
            <strong>Episode {index + 1} / seed {item.seed}</strong>
            {informationMode === 'ground-truth' && <span>{readableOutcome(item.outcome)}</span>}
          </span>
          {informationMode === 'ground-truth' ? (
            <>
              <span>{item.defender} vs {item.attacker} / {item.clutter}</span>
              <span>{item.steps} steps / {item.duration.toFixed(1)} s / reward {item.reward.toFixed(2)}</span>
              <span>{item.detection === null ? 'Detection: not acquired' : `Detection: ${item.detection.toFixed(1)} s`}{item.collision ? ` / ${item.collision}` : ''}{item.intercept === null ? '' : ` / interception ${item.intercept.toFixed(1)} s`}</span>
            </>
          ) : (
            <span>{item.defender} / {item.clutter} / {item.steps} recorded steps / {item.duration.toFixed(1)} s</span>
          )}
        </button>
      ))}
    </div>
  );
}

export function App() {
  const health = useBackendHealth();
  const [viewMode, setViewMode] = useState<'single' | 'comparison' | 'cases'>('single');
  const [caseOrigin, setCaseOrigin] = useState<'single' | 'comparison'>('single');
  const [layoutMode, setLayoutMode] = useState<'focus' | 'analysis'>('analysis');
  const [comparisonContext, setComparisonContext] = useState<{
    replayIds: string[];
    axisTimeS: number;
    localTimeA: number;
    localTimeB: number;
    syncMode: 'absolute' | 'detection';
    frameStep: number;
    decisionStep: number | null;
  } | null>(null);
  const [caseRestore, setCaseRestore] = useState<{ item: ResearchCase; bookmark?: ResearchBookmark; nonce: number } | null>(null);
  const [sceneCaptureRequest, setSceneCaptureRequest] = useState<{
    token: number;
    caseId: string;
    replayId: string;
    replaySha256: string;
  } | null>(null);
  const [sceneCaptureStatus, setSceneCaptureStatus] = useState<string | null>(null);
  const [caseSeekNonce, setCaseSeekNonce] = useState(0);
  const [catalog, setCatalog] = useState<PolicyCatalog | null>(null);
  const [scenarios, setScenarios] = useState<ScenarioCatalog | null>(null);
  const [catalogError, setCatalogError] = useState<string | null>(null);
  const [defenderId, setDefenderId] = useState('greedy');
  const [generation, setGeneration] = useState('d0');
  const [trainingSeed, setTrainingSeed] = useState(71);
  const [attackerId, setAttackerId] = useState('A0');
  const [clutter, setClutter] = useState('C0');
  const [perception, setPerception] = useState<'local' | 'map_local'>('map_local');
  const [episodeCount, setEpisodeCount] = useState<EpisodeCount>(5);
  const [seedStart, setSeedStart] = useState('0');
  const [job, setJob] = useState<ReplayJob | null>(null);
  const [jobError, setJobError] = useState<string | null>(null);
  const [formError, setFormError] = useState<string | null>(null);
  const [replays, setReplays] = useState<ReplayListItem[]>([]);
  const [replayWarnings, setReplayWarnings] = useState<string[]>([]);
  const [batchSummaries, setBatchSummaries] = useState<ReplaySummary[]>([]);
  const [selectedReplayId, setSelectedReplayId] = useState<string | null>(null);
  const [selectedSummary, setSelectedSummary] = useState<ReplaySummary | null>(null);
  const [summaryError, setSummaryError] = useState<string | null>(null);
  const [traceBundle, setTraceBundle] = useState<ReplayBundle | null>(null);
  const [traceReplayId, setTraceReplayId] = useState<string | null>(null);
  const [traceLoading, setTraceLoading] = useState(false);
  const [traceError, setTraceError] = useState<string | null>(null);
  const [sceneError, setSceneError] = useState<string | null>(null);
  const [playbackReadout, setPlaybackReadout] = useState<PlaybackReadout | null>(null);
  const [informationMode, setInformationMode] = useState<InformationMode>('ground-truth');
  const [diagnosticVisibility, setDiagnosticVisibility] = useState(defaultDiagnosticVisibility);
  const [diagnosticsCollapsed, setDiagnosticsCollapsed] = useState(false);
  const [seekRequest, setSeekRequest] = useState<{ timeS: number; nonce: number } | null>(null);
  const [loading, setLoading] = useState(true);
  const handlePlaybackReadout = useCallback((readout: PlaybackReadout | null) => {
    setPlaybackReadout(previous => previous?.replayId === readout?.replayId
      && previous?.frame.step === readout?.frame.step
      && previous?.decision?.decisionStep === readout?.decision?.decisionStep
      && previous?.interpolated === readout?.interpolated
      ? previous
      : readout);
  }, []);
  const handleSceneError = useCallback((message: string) => {
    setSceneError(message || null);
  }, []);
  const beginSceneCapture = useCallback((item: ResearchCase, replayId: string, timeS: number) => {
    const reference = item.replay_references.find(entry => entry.replay_id === replayId);
    if (!reference) {
      setSceneCaptureStatus('Scene capture was not started: the selected replay is not referenced by this case.');
      return;
    }
    setSceneCaptureStatus('Rendering a high-resolution capture from the recorded Three.js scene…');
    setCaseOrigin('single');
    setSelectedReplayId(replayId);
    setSelectedSummary(null);
    setPlaybackReadout(null);
    setSeekRequest(previous => ({ timeS, nonce: (previous?.nonce ?? 0) + 1 }));
    setSceneCaptureRequest({
      token: Date.now(),
      caseId: item.case_id,
      replayId,
      replaySha256: reference.artifact_sha256,
    });
    setViewMode('single');
  }, []);
  const handleSceneCaptureResult = useCallback(async (
    result: { pngBase64: string; width: number; height: number; projection: 'perspective' | 'top-down'; camera: string; timeS: number } | null,
    captureError?: string,
  ) => {
    const request = sceneCaptureRequest;
    if (!request) return;
    try {
      if (!result) throw new Error(captureError || 'WebGL scene capture failed.');
      const saved = await createSceneCapture(request.caseId, {
        replay_id: request.replayId,
        replay_sha256: request.replaySha256,
        png_base64: result.pngBase64,
        width: result.width,
        height: result.height,
        time_s: result.timeS,
        camera: result.camera,
        projection: result.projection,
      });
      const file = saved.files[0];
      if (!file) throw new Error('The API saved no scene capture file.');
      const anchor = document.createElement('a');
      anchor.href = file.download_url;
      anchor.download = file.name;
      anchor.click();
      setSceneCaptureStatus(`Saved and downloaded ${file.name} (${result.width} × ${result.height}) from replay ${request.replayId.slice(0, 8)} at ${result.timeS.toFixed(2)} s.`);
    } catch (cause) {
      setSceneCaptureStatus(cause instanceof Error ? `Scene capture failed: ${cause.message}` : 'Scene capture failed.');
    } finally {
      setSceneCaptureRequest(null);
      setViewMode('cases');
    }
  }, [sceneCaptureRequest]);
  const handleComparisonCaseContext = useCallback((context: typeof comparisonContext) => {
    setComparisonContext(previous => {
      if (previous?.replayIds.join('|') === context?.replayIds.join('|')
          && previous?.axisTimeS === context?.axisTimeS
          && previous?.localTimeA === context?.localTimeA
          && previous?.localTimeB === context?.localTimeB
          && previous?.syncMode === context?.syncMode
          && previous?.frameStep === context?.frameStep
          && previous?.decisionStep === context?.decisionStep) return previous;
      return context;
    });
  }, []);
  const refreshReplayIndex = useCallback(async () => {
    const data = await getReplays();
    setReplays(data.replays);
    setReplayWarnings(data.warnings);
    return data.replays;
  }, []);

  useEffect(() => {
    let active = true;
    Promise.all([getPolicies(), getScenarios()])
      .then(([policyData, scenarioData]) => {
        if (!active) return;
        setCatalog(policyData);
        setScenarios(scenarioData);
        const defaultPolicy = policyData.defenders.find(item => item.policy_id === 'greedy' && item.available)
          ?? policyData.defenders.find(item => item.available);
        if (defaultPolicy) {
          setDefenderId(defaultPolicy.policy_id);
          const availableGeneration = defaultPolicy.generations.find(item => item.available);
          if (availableGeneration) {
            setGeneration(availableGeneration.generation);
            const availableSeed = availableGeneration.training_seeds.find(item => item.available);
            if (availableSeed) setTrainingSeed(availableSeed.training_seed);
          }
        }
        const defaultAttacker = policyData.attackers.find(item => item.attacker_id === 'A0' && item.available)
          ?? policyData.attackers.find(item => item.available);
        if (defaultAttacker) setAttackerId(defaultAttacker.attacker_id);
      })
      .catch(error => {
        if (active) setCatalogError(error instanceof Error ? error.message : 'Policy catalog could not be loaded.');
      })
      .finally(() => { if (active) setLoading(false); });
    getReplays().then(data => {
      if (active) {
        setReplays(data.replays);
        setReplayWarnings(data.warnings);
      }
    })
      .catch(error => { if (active) console.error('Replay listing failed', error); });
    return () => { active = false; };
  }, []);

  const availableDefenders = catalog?.defenders.filter(item => item.available) ?? [];
  const defender = availableDefenders.find(item => item.policy_id === defenderId) ?? availableDefenders[0];
  const availableAttackers = catalog?.attackers.filter(item => item.available) ?? [];
  const attacker = availableAttackers.find(item => item.attacker_id === attackerId) ?? availableAttackers[0];
  const currentGenerations = defender?.generations.filter(item => item.available) ?? [];
  const selectedGeneration = currentGenerations.find(item => item.generation === generation) ?? currentGenerations[0];
  const currentSeeds = selectedGeneration?.training_seeds.filter(item => item.available) ?? [];
  const activeSeed = currentSeeds.some(item => item.training_seed === trainingSeed)
    ? trainingSeed
    : currentSeeds[0]?.training_seed;
  const estimator = defender?.classification === 'learned'
    ? selectedGeneration?.estimator
    : defender?.estimator ?? 'measurement';
  const selectedScenario = scenarios?.scenarios.find(item => item.clutter === clutter);
  const selectedPerception = selectedScenario?.perceptions.find(item => item.id === perception);
  const resolvedConfiguration = estimator === 'kalman'
    ? selectedPerception?.configurations.kalman
    : selectedPerception?.configurations.measurement;
  const activeJob = isActive(job);
  const batchIdsKey = job?.completed_replay_ids.join(',') ?? '';

  useEffect(() => {
    if (!job || !isActive(job)) return;
    let active = true;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const poll = async () => {
      try {
        const updated = await getReplayJob(job.job_id);
        if (!active) return;
        setJob(updated);
        if (isActive(updated)) timer = setTimeout(() => { void poll(); }, 1000);
      } catch (error) {
        if (!active) return;
        setJobError(error instanceof Error ? error.message : 'Could not refresh replay job status.');
        timer = setTimeout(() => { void poll(); }, 1500);
      }
    };
    timer = setTimeout(() => { void poll(); }, 900);
    return () => { active = false; if (timer) clearTimeout(timer); };
  }, [job?.job_id, job?.state, job?.updated_at]);

  useEffect(() => {
    if (!job || !batchIdsKey) return;
    let active = true;
    Promise.all(job.completed_replay_ids.map(id => getReplaySummary(id)))
      .then(items => {
        if (!active) return;
        setBatchSummaries(items);
        if (!selectedReplayId && items[0]) setSelectedReplayId(items[0].replay_id);
      })
      .catch(error => {
        if (active) setJobError(error instanceof Error ? error.message : 'Could not load completed replay summaries.');
      });
    return () => { active = false; };
  }, [job?.job_id, batchIdsKey]);

  useEffect(() => {
    if (!selectedReplayId) {
      setSelectedSummary(null);
      setTraceBundle(null);
      setTraceReplayId(null);
      setTraceError(null);
      setTraceLoading(false);
      return;
    }
    let active = true;
    setSelectedSummary(null);
    setSummaryError(null);
    getReplaySummary(selectedReplayId)
      .then(summary => { if (active) setSelectedSummary(summary); })
      .catch(error => {
        if (active) {
          setSelectedSummary(null);
          setSummaryError(error instanceof Error ? error.message : 'Selected replay metadata could not be loaded.');
        }
      });
    return () => { active = false; };
  }, [selectedReplayId]);

  useEffect(() => {
      if (!selectedReplayId) {
        setTraceBundle(null);
        setTraceReplayId(null);
        setTraceError(null);
        setTraceLoading(false);
        return;
      }
      let active = true;
      setTraceBundle(null);
      setTraceError(null);
      setTraceLoading(true);
      getReplay(selectedReplayId)
        .then(bundle => {
          if (bundle.schema_version !== '1.0.0' || bundle.episodes.length !== 1 || bundle.episodes[0].frames.length < 2) {
            throw new Error('Replay must contain one validated version 1.0.0 episode with at least two recorded frames.');
          }
          if (active) {
            setTraceBundle(bundle);
            setTraceReplayId(selectedReplayId);
          }
        })
        .catch(error => {
          if (active) {
            setTraceBundle(null);
            setTraceReplayId(null);
            setTraceError(error instanceof Error ? error.message : 'Selected replay trace could not be loaded.');
          }
        })
        .finally(() => { if (active) setTraceLoading(false); });
      return () => { active = false; };
  }, [selectedReplayId]);

  useEffect(() => {
    if (!job || !['completed', 'failed', 'cancelled', 'interrupted'].includes(job.state)) return;
    let active = true;
    getReplays().then(data => {
      if (active) {
        setReplays(data.replays);
        setReplayWarnings(data.warnings);
      }
    })
      .catch(error => { if (active) setJobError(error instanceof Error ? error.message : 'Replay list refresh failed.'); });
    return () => { active = false; };
  }, [job?.state]);

  const resultItems = useMemo(() => {
    if (job) {
      return batchSummaries.map((item, index) => ({
        id: item.replay_id,
        seed: item.simulation_seed,
        defender: item.defender.policy_id,
        attacker: item.attacker.policy_id,
        clutter: item.environment.clutter,
        outcome: item.summary.outcome,
        steps: item.summary.steps,
        duration: item.summary.duration_s,
        detection: item.summary.detection_time_s,
        reward: item.summary.total_reward,
        collision: item.summary.collision_type,
        intercept: item.summary.interception_time_s,
        index,
      }));
    }
    return replays.map((item, index) => ({
      id: item.replay_id,
      seed: item.simulation_seed,
      defender: item.defender.policy_id,
      attacker: item.attacker.policy_id,
      clutter: item.clutter,
      outcome: item.outcome,
      steps: item.steps,
      duration: item.duration_s,
      detection: item.detection_time_s,
      reward: item.total_reward,
      collision: item.collision_type,
      intercept: item.interception_time_s,
      index,
    }));
  }, [batchSummaries, job, replays]);
  const selectedReplayIndex = resultItems.findIndex(item => item.id === selectedReplayId);
  const selectedEpisode = traceReplayId === selectedReplayId ? traceBundle?.episodes[0] ?? null : null;
  const diagnosticSeries = useMemo(
    () => selectedEpisode ? deriveDiagnosticSeries(selectedEpisode) : null,
    [selectedEpisode],
  );
  const currentPlaybackReadout = playbackReadout?.replayId === selectedReplayId ? playbackReadout : null;

  const batchStats = useMemo(() => {
    const summaries = batchSummaries.map(item => item.summary);
    if (!summaries.length) return null;
    return {
      intercepted: summaries.filter(item => item.outcome === 'intercepted').length,
      attackerSuccess: summaries.filter(item => item.outcome === 'soldier_caught').length,
      collisions: summaries.filter(item => item.outcome.endsWith('_obstacle_collision')).length,
      timeouts: summaries.filter(item => item.outcome === 'timeout').length,
      meanDuration: summaries.reduce((total, item) => total + item.duration_s, 0) / summaries.length,
    };
  }, [batchSummaries]);

  async function submitJob() {
    setFormError(null);
    setJobError(null);
    setBatchSummaries([]);
    const parsedSeed = Number(seedStart);
    if (!Number.isSafeInteger(parsedSeed) || parsedSeed < 0 || parsedSeed + episodeCount - 1 > MAX_SEED) {
      setFormError(`Starting seed must be an integer from 0 through ${MAX_SEED - episodeCount + 1}.`);
      return;
    }
    if (!defender || !attacker) {
      setFormError('No compatible available defender and attacker selection is loaded.');
      return;
    }
    const request: ReplayJobRequest = {
      defender_policy_id: defender.policy_id,
      attacker_condition: attacker.attacker_id,
      clutter_level: clutter,
      perception_mode: perception,
      seed_start: parsedSeed,
      episode_count: episodeCount,
      deterministic_inference: true,
    };
    if (defender.classification === 'learned') {
      if (!selectedGeneration || activeSeed === undefined) {
        setFormError('Select an available learned generation and checkpoint seed.');
        return;
      }
      request.defender_generation = selectedGeneration.generation;
      request.defender_training_seed = activeSeed;
    }
    try {
      setJob(await createReplayJob(request));
      setSelectedReplayId(null);
    } catch (error) {
      const message = error instanceof ApiError ? error.message : 'Replay job could not be started.';
      setFormError(message);
    }
  }

  async function cancelJob() {
    if (!job) return;
    setJobError(null);
    try {
      setJob(await cancelReplayJob(job.job_id));
    } catch (error) {
      setJobError(error instanceof Error ? error.message : 'Cancellation request failed.');
    }
  }

  function selectReplay(id: string) {
    if (id !== selectedReplayId) {
      setSelectedSummary(null);
      setPlaybackReadout(null);
      setInformationMode('ground-truth');
      setDiagnosticVisibility(defaultDiagnosticVisibility);
    }
    setSelectedReplayId(id);
  }

  function restoreResearchCase(item: ResearchCase, bookmark?: ResearchBookmark) {
    if (item.case_type === 'comparison') {
      setCaseRestore({ item, bookmark, nonce: caseSeekNonce + 1 });
      setCaseSeekNonce(value => value + 1);
      setViewMode('comparison');
      return;
    }
    setCaseRestore(null);
    setSelectedReplayId(item.replay_ids[0] ?? null);
    setSelectedSummary(null);
    setPlaybackReadout(null);
    setViewMode('single');
    const timeS = bookmark?.local_time_a_s ?? bookmark?.time_s ?? 0;
    setSeekRequest(previous => ({ timeS, nonce: (previous?.nonce ?? 0) + 1 }));
  }

  const scenarioValues = resolvedConfiguration?.parameters ?? {};
  const physicalKeys = Object.keys(parameterLabels).filter(key => key in scenarioValues);
  const caseReplayIds = caseOrigin === 'comparison'
    ? comparisonContext?.replayIds ?? []
    : selectedReplayId ? [selectedReplayId] : [];
  const caseTimeA = caseOrigin === 'comparison'
    ? comparisonContext?.localTimeA ?? 0
    : playbackReadout?.replayId === selectedReplayId ? playbackReadout.timeS : 0;
  const caseFrame = caseOrigin === 'comparison'
    ? comparisonContext?.frameStep ?? 0
    : playbackReadout?.replayId === selectedReplayId ? playbackReadout.frame.step : 0;
  const caseDecision = caseOrigin === 'comparison'
    ? comparisonContext?.decisionStep ?? null
    : playbackReadout?.replayId === selectedReplayId ? playbackReadout.decision?.decisionStep ?? null : null;

  return (
    <main className={`app-shell ${layoutMode}-layout`}>
      <header className="app-header">
        <div className="brand-mark" aria-hidden="true">CU</div>
        <div className="brand"><span className="eyebrow">LOCAL RESEARCH WORKSPACE</span><h1>Counter-UAS <span>Simulation Explorer</span></h1></div>
        <div className="app-mode-switch" role="group" aria-label="Explorer mode">
          <button type="button" aria-pressed={viewMode === 'single'} onClick={() => setViewMode('single')}>Single episode</button>
          <button type="button" aria-pressed={viewMode === 'comparison'} onClick={() => setViewMode('comparison')}>Compare policies</button>
          <button type="button" aria-pressed={viewMode === 'cases'} onClick={() => {
            if (viewMode !== 'cases') setCaseOrigin(viewMode);
            setViewMode('cases');
          }}>Research cases</button>
        </div>
        {viewMode !== 'cases' && <div className="app-mode-switch" role="group" aria-label="Workspace layout">
          <button type="button" aria-pressed={layoutMode === 'focus'} onClick={() => setLayoutMode('focus')}>Focus</button>
          <button type="button" aria-pressed={layoutMode === 'analysis'} onClick={() => setLayoutMode('analysis')}>Analysis</button>
        </div>}
        <span className="phase-badge">PHASE 7 / RESEARCH CASES</span>
      </header>
      <div className="status-bar"><span className={`status-dot ${health.kind}`} /><span role="status">{health.message}</span><span className="localhost-label">LOCALHOST ONLY</span></div>
      {viewMode === 'cases' && (
        <ResearchCases
          replayCatalog={replays}
          replayIds={caseReplayIds}
          comparison={caseOrigin === 'comparison'}
          syncMode={comparisonContext?.syncMode ?? 'absolute'}
          axisTimeS={caseOrigin === 'comparison' ? comparisonContext?.axisTimeS ?? 0 : caseTimeA}
          localTimeA={caseTimeA}
          localTimeB={caseOrigin === 'comparison' ? comparisonContext?.localTimeB ?? null : null}
          frameStep={caseFrame}
          decisionStep={caseDecision}
          onRestore={restoreResearchCase}
          onCaptureScene={beginSceneCapture}
          captureStatus={sceneCaptureStatus}
        />
      )}
      {viewMode === 'comparison' ? (
        <Suspense fallback={<p className="muted">Loading the comparison workspace…</p>}>
          <ComparisonExplorer
            catalog={catalog}
            scenarios={scenarios}
            replays={replays}
            replayJobActive={activeJob}
            onJob={setJob}
            refreshReplays={refreshReplayIndex}
            restoreCase={caseRestore}
            onCaseContext={handleComparisonCaseContext}
          />
        </Suspense>
      ) : <div className="single-workspace" hidden={viewMode === 'cases'}>
      <div className={`dashboard${diagnosticsCollapsed ? ' diagnostics-collapsed' : ''}`}>
        <aside className="configuration panel">
          <span className="eyebrow">EXPERIMENT SETUP</span><h2>Episode configuration</h2>
          <p className="muted">Select a verified policy and canonical scenario. Physics parameters are read-only.</p>
          {catalogError && <div className="inline-error" role="alert">Policy catalog unavailable: {catalogError}</div>}
          <fieldset disabled={loading || !catalog || activeJob || availableDefenders.length === 0 || availableAttackers.length === 0}>
            <legend className="sr-only">Configure deterministic real simulation episodes</legend>
            <DefenderControls
              policies={availableDefenders}
              policy={defender}
              generation={selectedGeneration?.generation ?? generation}
              trainingSeed={activeSeed ?? trainingSeed}
              onPolicy={next => {
                setDefenderId(next.policy_id);
                const nextGeneration = next.generations.find(item => item.available);
                setGeneration(nextGeneration?.generation ?? '');
                setTrainingSeed(nextGeneration?.training_seeds.find(item => item.available)?.training_seed ?? 0);
              }}
              onGeneration={value => {
                setGeneration(value);
                const next = currentGenerations.find(item => item.generation === value)?.training_seeds.find(item => item.available);
                if (next) setTrainingSeed(next.training_seed);
              }}
              onTrainingSeed={setTrainingSeed}
            />
            <label>
              Attacker policy
              <select aria-label="Attacker policy" value={attacker?.attacker_id ?? ''} onChange={event => setAttackerId(event.target.value)}>
                {availableAttackers.map(item => <option key={item.attacker_id} value={item.attacker_id}>{item.display_name}</option>)}
              </select>
            </label>
            <label>
              Clutter level
              <select aria-label="Clutter level" value={clutter} onChange={event => setClutter(event.target.value)}>
                {scenarios?.scenarios.map(item => <option key={item.clutter} value={item.clutter}>{item.clutter} — {item.display_name}</option>)}
              </select>
            </label>
            <label>
              Obstacle perception
              <select aria-label="Perception mode" value={perception} onChange={event => setPerception(event.target.value as 'local' | 'map_local')}>
                <option value="local">Local sensing</option>
                <option value="map_local">Known static map + local sensing</option>
              </select>
            </label>
            <div className="resolved-parameters" aria-label="Resolved simulation parameters">
              <strong>Resolved canonical parameters / read-only</strong>
              <span>Estimator track: {estimator ?? 'not resolved'}</span>
              {physicalKeys.map(key => (
                <span key={key}>{parameterLabels[key]}: {formatParameter(scenarioValues[key], key)}</span>
              ))}
              {resolvedConfiguration && <small>Scenario config SHA-256: {resolvedConfiguration.config_sha256}</small>}
            </div>
            <label>
              Episode count
              <select aria-label="Episode count" value={episodeCount} onChange={event => setEpisodeCount(Number(event.target.value) as EpisodeCount)}>
                <option value="1">1 episode</option><option value="3">3 episodes</option>
                <option value="5">5 episodes (default)</option><option value="10">10 episodes</option>
              </select>
            </label>
            <label>
              Starting evaluation seed
              <input aria-label="Starting evaluation seed" type="number" min="0" max={MAX_SEED} step="1" value={seedStart} onChange={event => setSeedStart(event.target.value)} />
              <small>Episodes use consecutive seeds. Deterministic inference is fixed on.</small>
            </label>
            <button className="primary-button" type="button" onClick={() => void submitJob()} disabled={activeJob || !catalog || loading}>
              {activeJob ? 'Generating episodes…' : 'Generate Episodes'}
            </button>
          </fieldset>
          {formError && <div className="inline-error" role="alert">{formError}</div>}
          <div className="integrity-note"><strong>Research integrity</strong><p>Allowlisted frozen artifacts, hash verification, deterministic inference, and native Python simulation. No training or browser-side dynamics.</p></div>
        </aside>
        <div className="scene-column">
          <ReplayPlayback
            replayId={selectedReplayId}
            episode={selectedEpisode}
            summary={selectedSummary}
            traceLoading={traceLoading}
            traceError={traceError}
            episodeNumber={selectedReplayIndex + 1}
            episodeCount={resultItems.length}
            onSelectPrevious={() => {
              if (selectedReplayIndex > 0) selectReplay(resultItems[selectedReplayIndex - 1].id);
            }}
            onSelectNext={() => {
              if (selectedReplayIndex >= 0 && selectedReplayIndex < resultItems.length - 1) {
                selectReplay(resultItems[selectedReplayIndex + 1].id);
              }
            }}
            onReadout={handlePlaybackReadout}
            onSceneError={handleSceneError}
            informationMode={informationMode}
            diagnosticSeries={diagnosticSeries}
            diagnosticVisibility={diagnosticVisibility}
            seekRequest={seekRequest}
            onDiagnosticSeek={timeS => setSeekRequest(previous => ({ timeS, nonce: (previous?.nonce ?? 0) + 1 }))}
            captureRequestToken={sceneCaptureRequest?.replayId === selectedReplayId && selectedEpisode ? sceneCaptureRequest.token : 0}
            sceneActive={viewMode === 'single'}
            onCaptureResult={handleSceneCaptureResult}
          />
          {sceneError && <p className="inline-error" role="alert">Scene error: {sceneError}</p>}
          <section className="results panel" aria-label="Episode generation progress and results">
            <div className="results-heading">
              <div><span className="eyebrow">REAL SIMULATION JOB</span><h2>Generation progress &amp; episode results</h2></div>
              {job && <span className={`job-state ${job.state}`}>{job.state}</span>}
            </div>
            {job && (
              <div className="job-progress" aria-live="polite">
                <p><strong>Job ID:</strong> <code>{job.job_id}</code></p>
                {job.state === 'running' && job.current_episode_seed !== null
                  ? <p>Episode {job.completed_count + 1} of {job.requested_episode_count} running — {job.completed_count} completed / seed {job.current_episode_seed}</p>
                  : <p>{job.completed_count} of {job.requested_episode_count} episodes completed</p>}
                {job.cancellation_requested && <p className="muted">Cancellation requested; waiting for the simulation process to stop safely.</p>}
                {job.error && <p className="inline-error" role="alert">{job.error}</p>}
                {jobError && <p className="inline-error" role="alert">{jobError}</p>}
                {activeJob && <button type="button" onClick={() => void cancelJob()}>Cancel batch</button>}
              </div>
            )}
            {batchStats && batchSummaries.length > 0 && (
              <div className="batch-statistics" aria-label="Exploratory descriptive batch summaries">
                <strong>Exploratory descriptive summaries / not publication-grade estimates</strong>
                <span>Intercepted: {batchStats.intercepted}</span>
                <span>Attacker reached asset: {batchStats.attackerSuccess}</span>
                <span>Obstacle collisions: {batchStats.collisions}</span>
                <span>Timeouts: {batchStats.timeouts}</span>
                <span>Mean simulated duration: {batchStats.meanDuration.toFixed(1)} s</span>
              </div>
            )}
            {replayWarnings.map(warning => <p className="inline-error" role="alert" key={warning}>{warning}</p>)}
            <EpisodeSelection items={resultItems} selectedId={selectedReplayId} onSelect={selectReplay} informationMode={informationMode} />
            {summaryError && <p className="inline-error" role="alert">{summaryError}</p>}
          </section>
        </div>
        <aside className={`diagnostics panel${diagnosticsCollapsed ? ' collapsed' : ''}`}>
          <button
            className="diagnostics-collapse"
            type="button"
            aria-expanded={!diagnosticsCollapsed}
            aria-label={diagnosticsCollapsed ? 'Expand diagnostics panel' : 'Collapse diagnostics panel'}
            onClick={() => setDiagnosticsCollapsed(value => !value)}
          >
            {diagnosticsCollapsed ? '›' : 'Collapse diagnostics'}
          </button>
          {!diagnosticsCollapsed && <>
          <span className="eyebrow">LIVE RECORDED STATE</span><h2>Episode information</h2>
          <PolicyInspector
            episode={selectedEpisode}
            replayId={selectedReplayId}
            readout={currentPlaybackReadout}
            diagnosticSeries={diagnosticSeries}
            informationMode={informationMode}
            visibility={diagnosticVisibility}
            onInformationMode={setInformationMode}
            onVisibility={setDiagnosticVisibility}
          />
          {!selectedEpisode && <p className="muted">{traceLoading ? 'Loading the complete recorded trace…' : traceError ? 'The selected trace is unavailable; other episodes remain accessible.' : 'Select a completed episode to inspect its recorded state.'}</p>}
          {selectedEpisode && (
            <div className="diagnostic-card" aria-label="Current episode information">
              <h3>{selectedEpisode.defender.policy_id} {selectedEpisode.defender.generation ?? ''}{informationMode === 'ground-truth' ? ` vs ${selectedEpisode.attacker.policy_id}` : ''}</h3>
              <p>Defender generation: {selectedEpisode.defender.generation ?? 'Not applicable'}</p>
              <p>Defender training seed: {selectedEpisode.defender.training_seed ?? 'Unavailable'} / checkpoint role: {selectedEpisode.defender.checkpoint?.role ?? 'none / analytical'}</p>
              {informationMode === 'ground-truth' && <p>Attacker: {selectedEpisode.attacker.policy_id}{selectedEpisode.attacker.training_seed === null ? '' : ` / training seed ${selectedEpisode.attacker.training_seed}`}</p>}
              <p>Clutter: {selectedEpisode.environment.clutter} / perception: {selectedEpisode.environment.awareness}</p>
              <p>Evaluation seed: {selectedEpisode.simulation_seed} / time step: {selectedEpisode.environment.dt_s.toFixed(3)} s</p>
              {currentPlaybackReadout && (
                <>
                  <p>Current recorded step: {currentPlaybackReadout.frame.step}{currentPlaybackReadout.interpolated ? ' (display between this and the next recorded frame)' : ''}</p>
                  <p>Detection at this recorded step: {currentPlaybackReadout.frame.policy_visible.enemy_detected ? 'detected' : 'not detected'}{informationMode === 'ground-truth' ? ` / detection time: ${selectedEpisode.summary.detection_time_s === null ? 'not recorded' : `${selectedEpisode.summary.detection_time_s.toFixed(1)} s`}` : ''}</p>
                  {informationMode === 'ground-truth' ? (
                    <>
                      <p>Defender–attacker range at recorded step {currentPlaybackReadout.frame.step}: {currentPlaybackReadout.frame.privileged_diagnostics.defender_attacker_range_m.toFixed(2)} m</p>
                      <p>Attacker–asset range at recorded step {currentPlaybackReadout.frame.step}: {currentPlaybackReadout.frame.privileged_diagnostics.attacker_asset_range_m.toFixed(2)} m</p>
                    </>
                  ) : <p>Privileged entity ranges are hidden in Policy-Visible mode.</p>}
                  {informationMode === 'ground-truth' && (currentPlaybackReadout.frame.terminated || currentPlaybackReadout.frame.truncated
                    ? <p className="terminal-outcome">Terminal simulator outcome: {readableOutcome(currentPlaybackReadout.frame.outcome)}</p>
                    : <p>Recorded final outcome (known from complete trace): {readableOutcome(selectedEpisode.summary.outcome)}</p>)}
                </>
              )}
              {selectedEpisode && !currentPlaybackReadout && <p className="muted">Current step: waiting for trace playback initialization.</p>}
              <p className="privileged-note">{informationMode === 'ground-truth'
                ? 'The attacker remains visible to the researcher before policy detection; this is privileged visualization, not defender information.'
                : 'Policy-Visible mode hides privileged attacker state and trajectories. The reconstructed scene is partial because full sensing geometry is not recorded.'}</p>
            </div>
          )}
          <h2>Recorded summary &amp; provenance</h2>
          {informationMode === 'policy-visible' && (
            <p className="privileged-note">Policy-Visible mode presents the partial policy-known scene and diagnostics. Researcher-only episode outcomes and provenance remain hidden until Ground Truth mode is selected.</p>
          )}
          {informationMode === 'ground-truth' && <>
          {!selectedSummary && <p className="muted">Select a completed result to inspect its validated summary and provenance.</p>}
          {selectedSummary && (
            <>
              <div className="diagnostic-card">
                <h3>{readableOutcome(selectedSummary.summary.outcome)}</h3>
                <p>Seed {selectedSummary.simulation_seed} / {selectedSummary.defender.policy_id} {selectedSummary.defender.generation ?? ''} vs {selectedSummary.attacker.policy_id}</p>
                <p>{selectedSummary.environment.clutter} / {selectedSummary.environment.awareness} / {selectedSummary.summary.steps} steps</p>
                <p>Duration {selectedSummary.summary.duration_s.toFixed(1)} s / total reward {selectedSummary.summary.total_reward.toFixed(3)}</p>
                <p>Detection {selectedSummary.summary.detection_time_s === null ? 'not acquired' : `${selectedSummary.summary.detection_time_s.toFixed(1)} s`}</p>
                <p>{selectedSummary.summary.collision_type ?? (selectedSummary.summary.interception_time_s === null ? 'No collision/interception event' : `Interception at ${selectedSummary.summary.interception_time_s.toFixed(1)} s`)}</p>
              </div>
              <div className="diagnostic-card provenance">
                <h3>Reproducibility provenance</h3>
                <p>Schema {selectedSummary.schema_version}</p>
                <p>Simulator {selectedSummary.provenance.simulation_source_commit}</p>
                <p>Visualizer {selectedSummary.provenance.visualizer_source_commit}{selectedSummary.provenance.visualizer_dirty ? ' / dirty at run time' : ''}</p>
                <p>Source dirty flag: {String(selectedSummary.provenance.source_dirty)}</p>
                <p>Defender checkpoint: {selectedSummary.defender.checkpoint?.sha256 ?? 'none / analytical'}</p>
                <p>Attacker checkpoint: {selectedSummary.attacker.checkpoint?.sha256 ?? 'none / scripted'}</p>
                <p>Scenario config SHA-256: {selectedSummary.environment.config_sha256}</p>
              </div>
            </>
          )}
          </>}
          <div className="legend"><h3>Scene entity colors</h3><span><i className="defender" />Defender UAV / trail</span><span><i className="attacker" />Attacker UAV / trail</span><span><i className="asset" />Protected mobile asset / trail</span><span><i className="obstacle" />Moving obstacles</span></div>
          </>}
          <p className="projection-note">The browser interpolates only for display. Native recorded 3D state and simulator collision semantics remain unchanged.</p>
        </aside>
      </div>
      <footer className="app-footer"><span>Real simulation. Recorded evidence. No synthetic results.</span><span>{catalog?.source_commit ? `Research source ${catalog.source_commit}` : 'Waiting for verified policy catalog'}</span></footer>
      </div>}
    </main>
  );
}
