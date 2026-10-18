import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { ChangeEvent } from 'react';
import {
  cancelReplayJob,
  createReplayJob,
  getReplay,
  getReplayJob,
} from './api';
import type {
  DefenderPolicy,
  PolicyCatalog,
  ReplayBundle,
  ReplayJob,
  ReplayJobRequest,
  ReplayListItem,
  ReplayPolicySelection,
  ReplayEpisode,
  ResearchBookmark,
  ResearchCase,
  ScenarioCatalog,
} from './api';
import { Battlefield } from './Battlefield';
import type { CameraMode, CameraPose, TrailVisibility } from './Battlefield';
import { decisionIntervalAtTime, deriveDiagnosticSeries } from './diagnostics';
import {
  comparisonAxisRange,
  defenderPathLength,
  firstPositionDivergence,
  mapAxisTimeToEpisodeTime,
  matchesDefenderChoice,
  validateComparisonPair,
} from './comparison';
import type { ComparisonSyncMode, ComparisonValidation, DefenderChoice } from './comparison';
import type { ExpectedCheckpoint } from './comparison';
import { sampleAtTime } from './playback';
import { ComparisonDiagnosticCharts, PolicyInspector } from './PolicyDiagnostics';
import type { DiagnosticVisibility, InformationMode } from './PolicyDiagnostics';
import type { PlaybackReadout } from './ReplayPlayback';

const counts = [1, 3, 5, 10] as const;
const speeds = [0.25, 0.5, 1, 2, 4, 8] as const;
const maxEvaluationSeed = 2 ** 32 - 1;
const defaultVisibility: DiagnosticVisibility = {
  lead: true,
  requested: true,
  executed: true,
  cone: true,
  estimate: true,
  truthMarker: false,
  trackingError: true,
  estimateVelocity: true,
  safetyMarker: true,
};
const trails: TrailVisibility = { defender: true, attacker: true, asset: true, obstacles: false };
type EpisodeCount = typeof counts[number];

interface PairResult {
  replayAId: string | null;
  replayBId: string | null;
  status: ComparisonValidation['status'] | 'pending' | 'incomplete';
  reasons: string[];
}

interface ComparisonExplorerProps {
  catalog: PolicyCatalog | null;
  scenarios: ScenarioCatalog | null;
  replays: ReplayListItem[];
  replayJobActive: boolean;
  onJob: (job: ReplayJob) => void;
  refreshReplays: () => Promise<ReplayListItem[]>;
  restoreCase: { item: ResearchCase; bookmark?: ResearchBookmark; nonce: number } | null;
  onCaseContext: (context: {
    replayIds: string[];
    axisTimeS: number;
    localTimeA: number;
    localTimeB: number;
    syncMode: ComparisonSyncMode;
    frameStep: number;
    decisionStep: number | null;
  } | null) => void;
}

function labelPolicy(policy: DefenderPolicy, generation: string | null, trainingSeed: number | null): string {
  if (policy.classification !== 'learned') return policy.display_name;
  const selected = policy.generations.find(item => item.generation === generation);
  const seed = selected?.training_seeds.find(item => item.training_seed === trainingSeed);
  return `${policy.display_name} · ${selected?.display_name ?? generation} · seed ${trainingSeed} · ${seed?.checkpoint_role ?? selected?.checkpoint_role ?? 'role unavailable'}`;
}

function safeChoice(policy: DefenderPolicy | undefined, generation?: string, trainingSeed?: number): DefenderChoice {
  if (!policy || policy.classification === 'analytical') {
    return { policyId: policy?.policy_id ?? 'greedy', generation: null, trainingSeed: null };
  }
  const availableGenerations = policy.generations.filter(item => item.available);
  const selectedGeneration = availableGenerations.find(item => item.generation === generation)
    ?? availableGenerations.find(item => item.generation === 'd1')
    ?? availableGenerations[0];
  const seed = selectedGeneration?.training_seeds.find(item => item.available && item.training_seed === trainingSeed)
    ?? selectedGeneration?.training_seeds.find(item => item.available);
  return {
    policyId: policy.policy_id,
    generation: selectedGeneration?.generation ?? null,
    trainingSeed: seed?.training_seed ?? null,
  };
}

function choicePolicy(catalog: PolicyCatalog | null, choice: DefenderChoice): DefenderPolicy | undefined {
  return catalog?.defenders.find(item => item.policy_id === choice.policyId && item.available);
}

function expectedCheckpoint(catalog: PolicyCatalog | null, choice: DefenderChoice): ExpectedCheckpoint | null {
  if (choice.generation === null) return null;
  const generation = choicePolicy(catalog, choice)?.generations.find(item => item.generation === choice.generation);
  const seed = generation?.training_seeds.find(item => item.training_seed === choice.trainingSeed && item.available);
  return { sha256: seed?.checkpoint_sha256 ?? null, role: seed?.checkpoint_role ?? '' };
}

function listMatches(
  replays: ReplayListItem[],
  choice: DefenderChoice,
  attackerId: string,
  attackerCheckpointHash: string | null,
  clutter: string,
  perception: string,
  seed: number,
  expectedDefenderCheckpoint: ExpectedCheckpoint | null | undefined,
): ReplayListItem[] {
  return replays.filter(item => matchesDefenderChoice(item, choice, expectedDefenderCheckpoint)
    && item.attacker.policy_id === attackerId
    && (item.attacker.checkpoint?.sha256 ?? null) === attackerCheckpointHash
    && item.clutter === clutter
    && item.perception === perception
    && item.simulation_seed === seed);
}

function selectionMatches(
  episode: ReplayEpisode,
  choice: DefenderChoice,
  attackerId: string,
  attackerCheckpointHash: string | null,
  clutter: string,
  perception: string,
  seed: number,
  expectedDefenderCheckpoint: ExpectedCheckpoint | null,
): boolean {
  return episode.defender.policy_id === choice.policyId
    && episode.defender.generation === choice.generation
    && episode.defender.training_seed === choice.trainingSeed
    && (expectedDefenderCheckpoint === null
      ? episode.defender.checkpoint === null
      : episode.defender.checkpoint?.sha256 === expectedDefenderCheckpoint.sha256
        && episode.defender.checkpoint?.role === expectedDefenderCheckpoint.role)
    && episode.attacker.policy_id === attackerId
    && (episode.attacker.checkpoint?.sha256 ?? null) === attackerCheckpointHash
    && episode.environment.clutter === clutter
    && episode.environment.awareness === perception
    && episode.simulation_seed === seed;
}

function currentReadout(episode: ReplayEpisode, replayId: string, timeS: number): PlaybackReadout {
  const sample = sampleAtTime(episode, timeS);
  return {
    replayId,
    frame: episode.frames[sample.lowerIndex],
    timeS: sample.timeS,
    decision: decisionIntervalAtTime(episode, sample.timeS),
    interpolated: sample.fraction > 0,
  };
}

function policyText(selection: ReplayPolicySelection): string {
  const generation = selection.generation ? ` ${selection.generation.toUpperCase()}` : '';
  const trainingSeed = selection.training_seed === null ? '' : ` / training seed ${selection.training_seed}`;
  const role = selection.checkpoint?.role ? ` / ${selection.checkpoint.role}` : '';
  const hash = selection.checkpoint?.sha256 ? ` / ${selection.checkpoint.sha256.slice(0, 12)}` : '';
  return `${selection.policy_id}${generation}${trainingSeed}${role}${hash}`;
}

function policySelect(
  title: string,
  choice: DefenderChoice,
  catalog: PolicyCatalog | null,
  disabled: boolean,
  onChange: (choice: DefenderChoice) => void,
) {
  const policies = catalog?.defenders.filter(item => item.available) ?? [];
  const policy = choicePolicy(catalog, choice);
  const generations = policy?.generations.filter(item => item.available) ?? [];
  const selectedGeneration = generations.find(item => item.generation === choice.generation);
  const seeds = selectedGeneration?.training_seeds.filter(item => item.available) ?? [];
  return (
    <fieldset className="comparison-policy-choice" disabled={disabled || !policy}>
      <legend>{title}</legend>
      <label>Controller family
        <select value={choice.policyId} onChange={event => {
          const next = policies.find(item => item.policy_id === event.target.value);
          if (next) onChange(safeChoice(next));
        }}>
          {policies.map(item => <option key={item.policy_id} value={item.policy_id}>{item.display_name}</option>)}
        </select>
      </label>
      {policy?.classification === 'learned' && <>
        <label>Generation
          <select value={selectedGeneration?.generation ?? ''} onChange={event => onChange(safeChoice(policy, event.target.value))}>
            {generations.map(item => <option key={item.generation} value={item.generation}>{item.display_name} · {item.checkpoint_role}</option>)}
          </select>
        </label>
        <label>Training seed
          <select value={choice.trainingSeed ?? ''} onChange={event => onChange(safeChoice(policy, choice.generation ?? undefined, Number(event.target.value)))}>
            {seeds.map(item => <option key={item.training_seed} value={item.training_seed}>{item.training_seed} · {item.checkpoint_role} · {item.checkpoint_sha256?.slice(0, 12) ?? 'hash unavailable'}</option>)}
          </select>
        </label>
        <p className="muted">Observation {selectedGeneration?.observation_shape.join(' × ') ?? 'unavailable'} · estimator {selectedGeneration?.estimator ?? 'unavailable'}.</p>
      </>}
    </fieldset>
  );
}

function outcomeClass(outcome: string): string {
  return outcome === 'intercepted' ? 'success' : 'failure';
}

function axisTimeForEpisode(episode: ReplayEpisode, timeS: number, mode: ComparisonSyncMode): number {
  return mode === 'detection' ? timeS - (episode.summary.detection_time_s ?? 0) : timeS;
}

function firstSafetyTime(episode: ReplayEpisode, mode: ComparisonSyncMode): number | null {
  const frame = episode.frames.find(item => item.telemetry?.avoidance_active);
  if (!frame?.telemetry) return null;
  const decisionFrame = episode.frames[frame.telemetry.decision_step];
  return axisTimeForEpisode(episode, decisionFrame?.time_s ?? frame.time_s, mode);
}

function firstCollisionTime(episode: ReplayEpisode, mode: ComparisonSyncMode): number | null {
  const event = episode.events
    .filter(item => item.kind === 'collision')
    .reduce((first, item) => first === null || item.time_s < first.time_s ? item : first, null as ReplayEpisode['events'][number] | null);
  return event ? axisTimeForEpisode(episode, event.time_s, mode) : null;
}

function seedValues(seedStart: number, count: EpisodeCount): number[] {
  return Array.from({ length: count }, (_, index) => seedStart + index);
}

function wait(milliseconds: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, milliseconds));
}

function cameraPoseChanged(previous: CameraPose | null, next: CameraPose): boolean {
  if (!previous) return true;
  const values = [
    ...previous.position, ...previous.quaternion, ...previous.up, ...previous.target,
    ...next.position, ...next.quaternion, ...next.up, ...next.target,
  ];
  return values.slice(0, values.length / 2).some((value, index) => Math.abs(value - values[index + values.length / 2]) > 0.002);
}

export function ComparisonExplorer({
  catalog,
  scenarios,
  replays,
  replayJobActive,
  onJob,
  refreshReplays,
  restoreCase,
  onCaseContext,
}: ComparisonExplorerProps) {
  const [choiceA, setChoiceA] = useState<DefenderChoice>({ policyId: 'lr_ppo', generation: 'd1', trainingSeed: 77 });
  const [choiceB, setChoiceB] = useState<DefenderChoice>({ policyId: 'ppo_kalman', generation: 'd0', trainingSeed: 74 });
  const [attackerId, setAttackerId] = useState('A1-203');
  const [clutter, setClutter] = useState('C3');
  const [perception, setPerception] = useState('map_local');
  const [seedStartText, setSeedStartText] = useState('0');
  const [episodeCount, setEpisodeCount] = useState<EpisodeCount>(1);
  const [seedIndex, setSeedIndex] = useState(0);
  const [pairResults, setPairResults] = useState<Record<number, PairResult>>({});
  const [overrides, setOverrides] = useState<Record<number, { a?: string | null; b?: string | null }>>({});
  const [bundleA, setBundleA] = useState<ReplayBundle | null>(null);
  const [bundleB, setBundleB] = useState<ReplayBundle | null>(null);
  const [pairValidation, setPairValidation] = useState<ComparisonValidation | null>(null);
  const [pairLoading, setPairLoading] = useState(false);
  const [pairError, setPairError] = useState<string | null>(null);
  const [axisTimeS, setAxisTimeS] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState<number>(1);
  const [syncMode, setSyncMode] = useState<ComparisonSyncMode>('absolute');
  const [informationMode, setInformationMode] = useState<InformationMode>('ground-truth');
  const [visibility, setVisibility] = useState(defaultVisibility);
  const [cameraMode, setCameraMode] = useState<CameraMode>('free');
  const [linkedCameras, setLinkedCameras] = useState(true);
  const [cameraPose, setCameraPose] = useState<CameraPose | null>(null);
  const [resetCameraToken, setResetCameraToken] = useState(0);
  const [divergenceThreshold, setDivergenceThreshold] = useState(5);
  const [generating, setGenerating] = useState(false);
  const [generationJob, setGenerationJob] = useState<ReplayJob | null>(null);
  const [generationError, setGenerationError] = useState<string | null>(null);
  const [sceneError, setSceneError] = useState<string | null>(null);
  const alive = useRef(true);
  const restoreConfigKey = useRef<string | null>(null);
  const pendingRestoreTime = useRef<number | null>(null);

  useEffect(() => {
    alive.current = true;
    return () => { alive.current = false; };
  }, []);
  useEffect(() => {
    if (!catalog) return;
    setChoiceA(current => safeChoice(
      catalog.defenders.find(item => item.policy_id === current.policyId && item.available)
        ?? catalog.defenders.find(item => item.policy_id === 'lr_ppo' && item.available)
        ?? catalog.defenders.find(item => item.available),
      current.generation ?? 'd0',
      current.trainingSeed ?? 77,
    ));
    setChoiceB(current => safeChoice(
      catalog.defenders.find(item => item.policy_id === current.policyId && item.available)
        ?? catalog.defenders.find(item => item.policy_id === 'ppo_kalman' && item.available)
        ?? catalog.defenders.find(item => item.available),
      current.generation ?? 'd1',
      current.trainingSeed ?? 74,
    ));
    const preferredAttacker = catalog.attackers.find(item => item.attacker_id === 'A1-203' && item.available)
      ?? catalog.attackers.find(item => item.available);
    if (preferredAttacker) setAttackerId(preferredAttacker.attacker_id);
  }, [catalog]);

  useEffect(() => {
    if (!restoreCase || restoreCase.item.case_type !== 'comparison' || replays.length === 0) return;
    const [recordA, recordB] = restoreCase.item.replay_ids.map(id => replays.find(item => item.replay_id === id));
    if (!recordA || !recordB) return;
    restoreConfigKey.current = `${recordA.defender.policy_id}/${recordA.defender.generation}/${recordA.defender.training_seed}|${recordB.defender.policy_id}/${recordB.defender.generation}/${recordB.defender.training_seed}|${recordA.attacker.policy_id}|${recordA.clutter}|${recordA.perception}|${recordA.simulation_seed}|1`;
    pendingRestoreTime.current = restoreCase.bookmark?.time_s ?? 0;
    setChoiceA({
      policyId: recordA.defender.policy_id,
      generation: recordA.defender.generation,
      trainingSeed: recordA.defender.training_seed,
    });
    setChoiceB({
      policyId: recordB.defender.policy_id,
      generation: recordB.defender.generation,
      trainingSeed: recordB.defender.training_seed,
    });
    setAttackerId(recordA.attacker.policy_id);
    setClutter(recordA.clutter);
    setPerception(recordA.perception);
    setSeedStartText(String(recordA.simulation_seed));
    setEpisodeCount(1);
    setSeedIndex(0);
    setOverrides({ [recordA.simulation_seed]: { a: recordA.replay_id, b: recordB.replay_id } });
    setPairResults({});
    setSyncMode(restoreCase.bookmark?.sync_mode ?? 'absolute');
    setPlaying(false);
  }, [replays, restoreCase]);

  const seedStartParsed = Number(seedStartText);
  const validSeedRange = Number.isSafeInteger(seedStartParsed)
    && seedStartParsed >= 0
    && seedStartParsed + episodeCount - 1 <= maxEvaluationSeed;
  const seeds = useMemo(
    () => validSeedRange ? seedValues(seedStartParsed, episodeCount) : [],
    [episodeCount, seedStartParsed, validSeedRange],
  );
  const selectedSeed = seeds[Math.min(seedIndex, Math.max(0, seeds.length - 1))] ?? 0;
  const policyA = choicePolicy(catalog, choiceA);
  const policyB = choicePolicy(catalog, choiceB);
  const attackers = catalog?.attackers.filter(item => item.available) ?? [];
  const selectedAttacker = attackers.find(item => item.attacker_id === attackerId);
  const attackerCheckpointHash = selectedAttacker?.checkpoint_sha256 ?? null;
  const defenderCheckpointA = expectedCheckpoint(catalog, choiceA);
  const defenderCheckpointB = expectedCheckpoint(catalog, choiceB);
  const selectedScenario = scenarios?.scenarios.find(item => item.clutter === clutter);
  const compatible = (policy: DefenderPolicy | undefined) => policy?.supported_scenarios
    .some(item => item.clutter === clutter && item.perception_modes.includes(perception)) ?? false;
  const scenarioCompatible = compatible(policyA) && compatible(policyB);
  const candidatesA = useMemo(
    () => listMatches(replays, choiceA, attackerId, attackerCheckpointHash, clutter, perception, selectedSeed, defenderCheckpointA),
    [attackerCheckpointHash, attackerId, choiceA, clutter, defenderCheckpointA, perception, replays, selectedSeed],
  );
  const candidatesB = useMemo(
    () => listMatches(replays, choiceB, attackerId, attackerCheckpointHash, clutter, perception, selectedSeed, defenderCheckpointB),
    [attackerCheckpointHash, attackerId, choiceB, clutter, defenderCheckpointB, perception, replays, selectedSeed],
  );
  const pairResult = pairResults[selectedSeed];
  const replayIdA = overrides[selectedSeed]?.a !== undefined
    ? overrides[selectedSeed].a ?? null
    : pairResult?.replayAId ?? candidatesA[0]?.replay_id ?? null;
  const replayIdB = overrides[selectedSeed]?.b !== undefined
    ? overrides[selectedSeed].b ?? null
    : pairResult?.replayBId ?? candidatesB[0]?.replay_id ?? null;
  const exactSameController = choiceA.policyId === choiceB.policyId
    && choiceA.generation === choiceB.generation
    && choiceA.trainingSeed === choiceB.trainingSeed;
  const configKey = `${choiceA.policyId}/${choiceA.generation}/${choiceA.trainingSeed}|${choiceB.policyId}/${choiceB.generation}/${choiceB.trainingSeed}|${attackerId}|${clutter}|${perception}|${seedStartText}|${episodeCount}`;

  useEffect(() => {
    setPairResults({});
    if (restoreConfigKey.current !== null) {
      if (restoreConfigKey.current === configKey) restoreConfigKey.current = null;
      setSeedIndex(0);
      setPlaying(false);
      setBundleA(null);
      setBundleB(null);
      setPairValidation(null);
      setPairError(null);
      return;
    }
    setOverrides({});
    setSeedIndex(0);
    setAxisTimeS(0);
    setPlaying(false);
    setBundleA(null);
    setBundleB(null);
    setPairValidation(null);
    setPairError(null);
  }, [configKey]);

  useEffect(() => {
    const restoredPairSelected = Boolean(restoreCase?.item.case_type === 'comparison'
      && replayIdA && replayIdB
      && restoreCase.item.replay_ids.includes(replayIdA)
      && restoreCase.item.replay_ids.includes(replayIdB));
    if ((!scenarioCompatible && !restoredPairSelected) || !replayIdA || !replayIdB || (exactSameController && !restoredPairSelected)) {
      setBundleA(null);
      setBundleB(null);
      setPairValidation(null);
      setPairLoading(false);
      return;
    }
    let active = true;
    setPairLoading(true);
    setPairError(null);
    setBundleA(null);
    setBundleB(null);
    setPairValidation(null);
    Promise.all([getReplay(replayIdA), getReplay(replayIdB)])
      .then(([a, b]) => {
        if (!a.episodes?.[0] || !b.episodes?.[0]
            || a.episodes[0].frames.length < 2 || b.episodes[0].frames.length < 2) {
          throw new Error('Each comparison replay must contain one schema-v1 episode and at least two validated frames.');
        }
        const episodeA = a.episodes[0];
        const episodeB = b.episodes[0];
        const isRestoredPair = Boolean(restoreCase?.item.case_type === 'comparison'
          && restoreCase.item.replay_ids.includes(replayIdA)
          && restoreCase.item.replay_ids.includes(replayIdB));
        if (!isRestoredPair && (!selectionMatches(episodeA, choiceA, attackerId, attackerCheckpointHash, clutter, perception, selectedSeed, defenderCheckpointA)
            || !selectionMatches(episodeB, choiceB, attackerId, attackerCheckpointHash, clutter, perception, selectedSeed, defenderCheckpointB))) {
          throw new Error('Replay contents do not match the selected controller, scenario, attacker, perception mode, and evaluation seed.');
        }
        const validation = validateComparisonPair(a, b);
        if (!active) return;
        setBundleA(a);
        setBundleB(b);
        setPairValidation(validation);
        setPairResults(previous => ({
          ...previous,
          [selectedSeed]: {
            replayAId: replayIdA,
            replayBId: replayIdB,
            status: validation.status,
            reasons: validation.reasons,
          },
        }));
      })
      .catch(error => {
        if (active) setPairError(error instanceof Error ? error.message : 'The selected replay pair could not be loaded and validated.');
      })
      .finally(() => { if (active) setPairLoading(false); });
    return () => { active = false; };
  }, [attackerId, choiceA, choiceB, clutter, exactSameController, perception, replayIdA, replayIdB, restoreCase, scenarioCompatible, selectedSeed]);

  const episodeA = bundleA?.episodes[0] ?? null;
  const episodeB = bundleB?.episodes[0] ?? null;
  const detectionUnavailable = Boolean(episodeA && episodeB
    && (episodeA.summary.detection_time_s === null || episodeB.summary.detection_time_s === null));
  const range = useMemo(
    () => episodeA && episodeB ? comparisonAxisRange(episodeA, episodeB, syncMode) : null,
    [episodeA, episodeB, syncMode],
  );
  const activeRange = useMemo(
    () => syncMode === 'detection' && detectionUnavailable ? null : range,
    [detectionUnavailable, range, syncMode],
  );

  useEffect(() => {
    if (syncMode === 'detection' && detectionUnavailable) setSyncMode('absolute');
  }, [detectionUnavailable, syncMode]);
  useEffect(() => {
    if (!activeRange) return;
    const requested = pendingRestoreTime.current;
    setAxisTimeS(requested === null
      ? activeRange.minimum
      : Math.max(activeRange.minimum, Math.min(activeRange.maximum, requested)));
    pendingRestoreTime.current = null;
  }, [episodeA?.episode_id, episodeB?.episode_id, activeRange?.maximum, activeRange?.minimum]);
  useEffect(() => {
    if (!playing || !activeRange) return;
    let previous = performance.now();
    const timer = window.setInterval(() => {
      const now = performance.now();
      const elapsedS = Math.min(0.1, (now - previous) / 1000);
      previous = now;
      setAxisTimeS(current => {
        const next = Math.min(activeRange.maximum, current + elapsedS * speed);
        if (next >= activeRange.maximum) setPlaying(false);
        return next;
      });
    }, 33);
    return () => window.clearInterval(timer);
  }, [activeRange, playing, speed]);

  const localTimeA = episodeA ? mapAxisTimeToEpisodeTime(axisTimeS, episodeA, syncMode) : 0;
  const localTimeB = episodeB ? mapAxisTimeToEpisodeTime(axisTimeS, episodeB, syncMode) : 0;
  const readoutA = episodeA && replayIdA ? currentReadout(episodeA, replayIdA, localTimeA) : null;
  const readoutB = episodeB && replayIdB ? currentReadout(episodeB, replayIdB, localTimeB) : null;
  useEffect(() => {
    if (!replayIdA || !replayIdB || !readoutA) {
      onCaseContext(null);
      return;
    }
    onCaseContext({
      replayIds: [replayIdA, replayIdB],
      axisTimeS,
      localTimeA,
      localTimeB,
      syncMode,
      frameStep: readoutA.frame.step,
      decisionStep: readoutA.decision?.decisionStep ?? null,
    });
  }, [axisTimeS, localTimeA, localTimeB, onCaseContext, readoutA, replayIdA, replayIdB, syncMode]);
  const seriesA = useMemo(() => episodeA ? deriveDiagnosticSeries(episodeA) : null, [episodeA]);
  const seriesB = useMemo(() => episodeB ? deriveDiagnosticSeries(episodeB) : null, [episodeB]);
  const eligiblePair = Boolean(episodeA && episodeB && pairValidation && pairValidation.status !== 'incompatible');
  const firstDivergence = useMemo(
    () => eligiblePair && episodeA && episodeB
      ? firstPositionDivergence(episodeA, episodeB, syncMode, divergenceThreshold)
      : null,
    [divergenceThreshold, eligiblePair, episodeA, episodeB, syncMode],
  );
  const summaries = useMemo(() => episodeA && episodeB ? [
    { episode: episodeA, label: 'Policy A', pathLength: defenderPathLength(episodeA) },
    { episode: episodeB, label: 'Policy B', pathLength: defenderPathLength(episodeB) },
  ] : [], [episodeA, episodeB]);
  const requiredSimulations = episodeCount * 2;
  const currentlyMissing = seeds.reduce((total, seed) => total
    + Number(listMatches(replays, choiceA, attackerId, attackerCheckpointHash, clutter, perception, seed, defenderCheckpointA).length === 0)
    + Number(listMatches(replays, choiceB, attackerId, attackerCheckpointHash, clutter, perception, seed, defenderCheckpointB).length === 0), 0);
  const existingRows = useMemo(() => seeds.map(seed => {
    const result = pairResults[seed];
    const matchesA = listMatches(replays, choiceA, attackerId, attackerCheckpointHash, clutter, perception, seed, defenderCheckpointA);
    const matchesB = listMatches(replays, choiceB, attackerId, attackerCheckpointHash, clutter, perception, seed, defenderCheckpointB);
    const a = matchesA.find(item => item.replay_id === result?.replayAId) ?? matchesA[0];
    const b = matchesB.find(item => item.replay_id === result?.replayBId) ?? matchesB[0];
    return { seed, result, a, b };
  }), [attackerCheckpointHash, attackerId, choiceA, choiceB, clutter, defenderCheckpointA, defenderCheckpointB, pairResults, perception, replays, seeds]);
  const validatedBatchCounts = useMemo(() => {
    const validated = existingRows.filter(row => row.a && row.b
      && (row.result?.status === 'matched' || row.result?.status === 'exploratory'));
    return {
      count: validated.length,
      aSuccesses: validated.filter(row => row.a?.outcome === 'intercepted').length,
      bSuccesses: validated.filter(row => row.b?.outcome === 'intercepted').length,
    };
  }, [existingRows]);

  const selectSeed = useCallback((seed: number) => {
    const index = seeds.indexOf(seed);
    if (index < 0) return;
    setSeedIndex(index);
    setAxisTimeS(0);
    setPlaying(false);
    setBundleA(null);
    setBundleB(null);
    setPairValidation(null);
    setPairError(null);
  }, [seeds]);

  async function runMissingReplays() {
    if (!scenarioCompatible || !validSeedRange || replayJobActive || generating || exactSameController) return;
    setGenerating(true);
    setGenerationError(null);
    setPairError(null);
    let currentReplays = replays;
    try {
      for (const seed of seeds) {
        for (const [choice, side] of [[choiceA, 'A'], [choiceB, 'B']] as const) {
          const expectedHash = selectedAttacker?.checkpoint_sha256 ?? null;
          const expectedPolicyCheckpoint = expectedCheckpoint(catalog, choice);
          if (listMatches(currentReplays, choice, attackerId, expectedHash, clutter, perception, seed, expectedPolicyCheckpoint).length) continue;
          const request: ReplayJobRequest = {
            defender_policy_id: choice.policyId,
            attacker_condition: attackerId,
            clutter_level: clutter,
            perception_mode: perception,
            seed_start: seed,
            episode_count: 1,
            deterministic_inference: true,
          };
          if (choice.generation && choice.trainingSeed !== null) {
            request.defender_generation = choice.generation;
            request.defender_training_seed = choice.trainingSeed;
          }
          let job = await createReplayJob(request);
          setGenerationJob(job);
          onJob(job);
          while (['queued', 'running'].includes(job.state)) {
            await wait(900);
            job = await getReplayJob(job.job_id);
            if (!alive.current) return;
            setGenerationJob(job);
            onJob(job);
          }
          if (job.state !== 'completed' || job.completed_replay_ids.length !== 1) {
            throw new Error(`Policy ${side} replay for seed ${seed} ${job.state}: ${job.error ?? 'no validated replay was persisted'}. Completed pairs remain available.`);
          }
          currentReplays = await refreshReplays();
        }
        const matchesA = listMatches(currentReplays, choiceA, attackerId, attackerCheckpointHash, clutter, perception, seed, defenderCheckpointA);
        const matchesB = listMatches(currentReplays, choiceB, attackerId, attackerCheckpointHash, clutter, perception, seed, defenderCheckpointB);
        if (!matchesA.length || !matchesB.length) {
          setPairResults(previous => ({
            ...previous,
            [seed]: {
              replayAId: matchesA[0]?.replay_id ?? null,
              replayBId: matchesB[0]?.replay_id ?? null,
              status: 'incomplete',
              reasons: ['One or both newly generated replay records are not available in the validated replay catalog.'],
            },
          }));
          throw new Error(`Seed ${seed} is incomplete: a paired replay could not be found. Previously completed seed pairs were retained.`);
        }
        const [a, b] = await Promise.all([getReplay(matchesA[0].replay_id), getReplay(matchesB[0].replay_id)]);
        const validation = validateComparisonPair(a, b);
        setPairResults(previous => ({
          ...previous,
          [seed]: {
            replayAId: matchesA[0].replay_id,
            replayBId: matchesB[0].replay_id,
            status: validation.status,
            reasons: validation.reasons,
          },
        }));
        if (validation.status === 'incompatible') {
          throw new Error(`Seed ${seed} traces were generated but failed comparison eligibility: ${validation.reasons.join(' ')}`);
        }
      }
      await refreshReplays();
      setGenerationError(null);
    } catch (error) {
      setGenerationError(error instanceof Error ? error.message : 'Comparison replay generation failed.');
      try { currentReplays = await refreshReplays(); } catch (refreshError) {
        setGenerationError(previous => `${previous ?? 'Comparison generation stopped.'} Replay catalog refresh also failed: ${refreshError instanceof Error ? refreshError.message : 'unknown error'}`);
      }
      void currentReplays;
    } finally {
      setGenerating(false);
      setGenerationJob(null);
    }
  }

  async function cancelGeneration() {
    if (!generationJob || !['queued', 'running'].includes(generationJob.state)) return;
    try {
      const updated = await cancelReplayJob(generationJob.job_id);
      setGenerationJob(updated);
      onJob(updated);
    } catch (error) {
      setGenerationError(error instanceof Error ? error.message : 'Comparison job cancellation failed.');
    }
  }

  function setChoice(which: 'A' | 'B', choice: DefenderChoice) {
    if (which === 'A') setChoiceA(choice);
    else setChoiceB(choice);
  }

  function selectReplay(which: 'A' | 'B', event: ChangeEvent<HTMLSelectElement>) {
    const replayId = event.target.value || null;
    setOverrides(previous => ({
      ...previous,
      [selectedSeed]: { ...previous[selectedSeed], [which.toLowerCase()]: replayId },
    }));
  }

  function seekAxis(time: number) {
    setAxisTimeS(Math.max(activeRange?.minimum ?? 0, Math.min(activeRange?.maximum ?? time, time)));
    setPlaying(false);
  }

  function jumpToEvents(kind: 'detection' | 'safety' | 'collision' | 'terminal' | 'divergence') {
    if (!episodeA || !episodeB || !activeRange) return;
    let time: number | null = null;
    if (kind === 'detection') {
      const times = [episodeA, episodeB].flatMap(episode => episode.summary.detection_time_s === null
        ? []
        : [axisTimeForEpisode(episode, episode.summary.detection_time_s, syncMode)]);
      time = times.length ? Math.min(...times) : null;
    } else if (kind === 'safety') {
      const times = [firstSafetyTime(episodeA, syncMode), firstSafetyTime(episodeB, syncMode)]
        .filter((value): value is number => value !== null);
      time = times.length ? Math.min(...times) : null;
    } else if (kind === 'collision') {
      const times = [firstCollisionTime(episodeA, syncMode), firstCollisionTime(episodeB, syncMode)]
        .filter((value): value is number => value !== null);
      time = times.length ? Math.min(...times) : null;
    } else if (kind === 'terminal') {
      time = Math.min(
        axisTimeForEpisode(episodeA, episodeA.summary.duration_s, syncMode),
        axisTimeForEpisode(episodeB, episodeB.summary.duration_s, syncMode),
      );
    } else if (firstDivergence) {
      time = firstDivergence.axisTimeS;
    }
    if (time !== null) seekAxis(time);
  }

  const canDisplay = Boolean(eligiblePair && episodeA && episodeB && readoutA && readoutB);
  const resetCamera = () => {
    setCameraPose(null);
    setResetCameraToken(value => value + 1);
  };
  const publishCameraPose = useCallback((pose: CameraPose) => {
    setCameraPose(previous => cameraPoseChanged(previous, pose) ? pose : previous);
  }, []);
  const firstDetectionAvailable = Boolean(episodeA?.summary.detection_time_s !== null && episodeB?.summary.detection_time_s !== null);
  const safetyAvailable = Boolean(episodeA?.frames.some(frame => frame.telemetry?.avoidance_active)
    || episodeB?.frames.some(frame => frame.telemetry?.avoidance_active));
  const collisionAvailable = Boolean(episodeA?.events.some(event => event.kind === 'collision')
    || episodeB?.events.some(event => event.kind === 'collision'));
  const terminalTimeAvailable = Boolean(episodeA && episodeB && (episodeA.summary.duration_s > 0 || episodeB.summary.duration_s > 0));
  const showPanels = canDisplay && pairValidation !== null;

  return (
    <div className="comparison-shell" aria-label="Compare policies">
      <section className="comparison-selection panel" aria-label="Comparison scenario and controller selection">
        <div className="comparison-heading">
          <div><span className="eyebrow">MATCHED-SEED REPLAY ANALYSIS</span><h2>Compare policies</h2></div>
          <span className="small-badge">REAL RECORDED EPISODES · NO BROWSER SIMULATION</span>
        </div>
        <p className="muted">Each pane uses its own validated trace and controller-visible observations. Existing artifacts are reused; generation occurs only when explicitly requested.</p>
        {policySelect('Policy A', choiceA, catalog, generating || replayJobActive, value => setChoice('A', value))}
        {policySelect('Policy B', choiceB, catalog, generating || replayJobActive, value => setChoice('B', value))}
        <div className="comparison-shared-controls">
          <label>Shared attacker
            <select value={attackerId} disabled={generating || replayJobActive} onChange={event => setAttackerId(event.target.value)}>
              {attackers.map(item => <option key={item.attacker_id} value={item.attacker_id}>{item.attacker_id} · {item.display_name}{item.held_out ? ' · held out' : ''}</option>)}
            </select>
          </label>
          <label>Canonical clutter
            <select value={clutter} disabled={generating || replayJobActive} onChange={event => setClutter(event.target.value)}>
              {(scenarios?.scenarios ?? []).map(item => <option key={item.clutter} value={item.clutter}>{item.clutter} · {item.display_name}</option>)}
            </select>
          </label>
          <label>Perception configuration
            <select value={perception} disabled={generating || replayJobActive} onChange={event => setPerception(event.target.value)}>
              {(selectedScenario?.perceptions ?? []).map(item => <option key={item.id} value={item.id}>{item.id} · {item.description}</option>)}
            </select>
          </label>
          <label>Evaluation seed start
            <input type="number" min="0" max={maxEvaluationSeed} step="1" value={seedStartText} disabled={generating || replayJobActive} onChange={event => setSeedStartText(event.target.value)} />
          </label>
          <label>Matched seed count
            <select value={episodeCount} disabled={generating || replayJobActive} onChange={event => setEpisodeCount(Number(event.target.value) as EpisodeCount)}>
              {counts.map(value => <option key={value} value={value}>{value}</option>)}
            </select>
          </label>
        </div>
        {!validSeedRange && <p className="inline-error" role="alert">Enter a valid unsigned evaluation-seed range.</p>}
        {!scenarioCompatible && <p className="inline-error" role="alert">The selected clutter and perception configuration is not supported by both controller selections.</p>}
        {exactSameController && <p className="inline-error" role="alert">Policy A and Policy B are the same checkpoint selection; choose distinct controller selections for a comparison.</p>}
        {replayJobActive && !generating && <p className="inline-error">Another replay batch is active. Comparison generation uses the existing one-active-job guard and is disabled until it finishes.</p>}
        <div className="comparison-generation">
          <div>
            <strong>{requiredSimulations} simulations required for a full {episodeCount}-seed comparison</strong>
            <p>{currentlyMissing} rollout(s) are missing from the replay index; valid existing replays will be reused.</p>
          </div>
          <button type="button" className="primary-button" disabled={!validSeedRange || !scenarioCompatible || replayJobActive || generating || exactSameController || currentlyMissing === 0} onClick={() => void runMissingReplays()}>
            {generating ? 'Generating sequentially…' : 'Generate missing replays'}
          </button>
          {generating && <button type="button" onClick={() => void cancelGeneration()}>Cancel comparison generation</button>}
        </div>
        {generationJob && <p className="comparison-job-state" aria-live="polite">One active job · {generationJob.state} · {generationJob.completed_count}/{generationJob.requested_episode_count} complete{generationJob.current_episode_seed === null ? '' : ` · seed ${generationJob.current_episode_seed}`}</p>}
        {generationError && <p className="inline-error" role="alert">{generationError}</p>}
      </section>

      <section className="comparison-pair panel" aria-label="Selected matched replay pair">
        <div className="comparison-heading">
          <div><span className="eyebrow">PAIRED ARTIFACTS</span><h2>Seed {selectedSeed} · {policyA ? labelPolicy(policyA, choiceA.generation, choiceA.trainingSeed) : choiceA.policyId} vs {policyB ? labelPolicy(policyB, choiceB.generation, choiceB.trainingSeed) : choiceB.policyId}</h2></div>
          {pairValidation && <span className={`pair-status ${pairValidation.status}`} role="status">{pairValidation.status}</span>}
        </div>
        <div className="comparison-replay-selectors">
          <label>Policy A replay
            <select value={replayIdA ?? ''} disabled={generating || candidatesA.length === 0} onChange={event => selectReplay('A', event)}>
              {candidatesA.length ? candidatesA.map(item => <option key={item.replay_id} value={item.replay_id}>{item.replay_id.slice(0, 12)} · {informationMode === 'ground-truth' ? item.outcome : 'outcome hidden'} · {item.provenance.source_dirty ? 'dirty source' : 'clean source'}</option>) : <option value="">No indexed replay for this selection</option>}
            </select>
          </label>
          <label>Policy B replay
            <select value={replayIdB ?? ''} disabled={generating || candidatesB.length === 0} onChange={event => selectReplay('B', event)}>
              {candidatesB.length ? candidatesB.map(item => <option key={item.replay_id} value={item.replay_id}>{item.replay_id.slice(0, 12)} · {informationMode === 'ground-truth' ? item.outcome : 'outcome hidden'} · {item.provenance.source_dirty ? 'dirty source' : 'clean source'}</option>) : <option value="">No indexed replay for this selection</option>}
            </select>
          </label>
        </div>
        {pairLoading && <p className="trace-loading" role="status">Loading and validating the full replay pair…</p>}
        {pairError && <p className="inline-error" role="alert">{pairError}</p>}
        {!pairLoading && !pairError && !replayIdA && !replayIdB && <p className="muted">No replay pair is indexed for seed {selectedSeed}. Use “Generate missing replays” to run the explicitly requested, sequential real rollouts.</p>}
        {!pairLoading && !pairError && (!replayIdA || !replayIdB) && (replayIdA || replayIdB) && <p className="inline-error" role="alert">Incomplete pair: one policy replay is missing. The available trace is not shown as a matched pair.</p>}
        {pairValidation?.reasons.map(reason => <p className={pairValidation.status === 'incompatible' ? 'inline-error' : 'comparison-note'} key={reason}>{reason}</p>)}
        {pairValidation?.status === 'incompatible' && <p className="inline-error" role="alert">Pairing rejected: this pair cannot be displayed as a shared-scenario comparison.</p>}
        {pairValidation && <p className="comparison-note">Matching initial conditions do not establish coupled future attacker or asset trajectories. Each pane remains an independent authoritative rollout; no path is copied between traces.</p>}
      </section>

      {showPanels && episodeA && episodeB && readoutA && readoutB && seriesA && seriesB && activeRange && (
        <>
          <section className="comparison-controls panel" aria-label="Synchronized comparison playback controls">
            <div className="comparison-heading">
              <div><span className="eyebrow">SHARED PLAYBACK CLOCK</span><h2>{syncMode === 'absolute' ? 'Absolute simulation time' : 'Seconds relative to first detection'}</h2></div>
              <div className="comparison-display-mode" role="group" aria-label="Shared information display mode">
                <button type="button" aria-pressed={informationMode === 'ground-truth'} onClick={() => setInformationMode('ground-truth')}>Ground Truth</button>
                <button type="button" aria-pressed={informationMode === 'policy-visible'} onClick={() => setInformationMode('policy-visible')}>Policy-Visible</button>
              </div>
            </div>
            <div className="comparison-toolbar">
              <button type="button" onClick={() => { setPlaying(false); setAxisTimeS(activeRange.minimum); }}>Restart</button>
              <button type="button" onClick={() => setPlaying(value => !value)}>{playing ? 'Pause' : 'Play'}</button>
              <button type="button" onClick={() => seekAxis(axisTimeS - Math.min(episodeA.environment.dt_s, episodeB.environment.dt_s))}>Step backward</button>
              <button type="button" onClick={() => seekAxis(axisTimeS + Math.min(episodeA.environment.dt_s, episodeB.environment.dt_s))}>Step forward</button>
              <button type="button" onClick={() => seekAxis(activeRange.minimum)}>Beginning</button>
              <button type="button" onClick={() => seekAxis(activeRange.maximum)}>End</button>
              <label className="speed-select">Speed
                <select value={speed} onChange={event => setSpeed(Number(event.target.value))}>
                  {speeds.map(value => <option key={value} value={value}>{value}×</option>)}
                </select>
              </label>
              <label className="speed-select">Alignment
                <select value={syncMode} onChange={event => setSyncMode(event.target.value as ComparisonSyncMode)} disabled={detectionUnavailable}>
                  <option value="absolute">Absolute simulation time</option>
                  <option value="detection">Time relative to first detection</option>
                </select>
              </label>
            </div>
            {detectionUnavailable && <p className="muted">Detection alignment is disabled because at least one replay has no recorded first-detection time.</p>}
            <label className="timeline-label">
              {syncMode === 'absolute' ? 'Shared simulation time' : 'Time relative to first detection'}
              <input type="range" min={activeRange.minimum} max={activeRange.maximum || activeRange.minimum + 1e-6} step="0.01" value={axisTimeS} onChange={event => seekAxis(Number(event.target.value))} aria-label="Shared comparison timeline" />
            </label>
            <div className="comparison-clock-readout">
              <span>Shared axis {axisTimeS.toFixed(2)} s</span>
              <span>Policy A local {localTimeA.toFixed(2)} s{localTimeA >= episodeA.summary.duration_s ? ' · episode ended' : ''}</span>
              <span>Policy B local {localTimeB.toFixed(2)} s{localTimeB >= episodeB.summary.duration_s ? ' · episode ended' : ''}</span>
            </div>
            <div className="comparison-toolbar">
              {(['free', 'follow-defender', 'follow-attacker', 'top-down'] as const).map(mode => (
                <button key={mode} type="button" aria-pressed={cameraMode === mode} onClick={() => setCameraMode(mode)}>{mode.replaceAll('-', ' ')}</button>
              ))}
              <label className="diagnostic-toggle" title="Free-orbit and top-down camera movement is linked; follow cameras track each pane's own entity."><input type="checkbox" checked={linkedCameras} onChange={event => {
                setCameraPose(null);
                setLinkedCameras(event.target.checked);
              }} />Link free / top-down cameras</label>
              <button type="button" onClick={resetCamera}>Reset both cameras</button>
            </div>
            <div className="comparison-events">
              <button type="button" disabled={!firstDetectionAvailable} onClick={() => jumpToEvents('detection')}>Jump to first detection</button>
              <label>Position divergence threshold
                <input type="number" min="1" max="50" step="1" value={divergenceThreshold} onChange={event => setDivergenceThreshold(Math.max(1, Math.min(50, Number(event.target.value) || 1)))} />
                <span>m</span>
              </label>
              <button type="button" disabled={!firstDivergence} onClick={() => jumpToEvents('divergence')}>Jump to first path divergence</button>
              <button type="button" disabled={!safetyAvailable} onClick={() => jumpToEvents('safety')}>Jump to first safety event</button>
              <button type="button" disabled={!collisionAvailable} onClick={() => jumpToEvents('collision')}>Jump to first collision</button>
              <button type="button" disabled={!terminalTimeAvailable} onClick={() => jumpToEvents('terminal')}>Jump to first terminal event</button>
              {firstDivergence && <span>First sampled path separation ≥ {divergenceThreshold} m: {firstDivergence.axisTimeS.toFixed(2)} s ({firstDivergence.separationM.toFixed(2)} m). Descriptive only; not causal.</span>}
            </div>
          </section>

          <section className="comparison-scenes" aria-label="Policy A and Policy B authoritative 3D battlefields">
            <div>
              <div className="comparison-pane-title"><strong>LEFT · POLICY A</strong><span>{policyText(episodeA.defender)} · {episodeA.attacker.policy_id} · {episodeA.environment.clutter} / {episodeA.environment.awareness} · eval seed {episodeA.simulation_seed}</span></div>
              <Battlefield
                panelId="A"
                heading="Policy A battlefield"
                episode={episodeA}
                timeS={localTimeA}
                cameraMode={cameraMode}
                resetCameraToken={resetCameraToken}
                showFullTrajectory={false}
                trails={trails}
                showDetectionVolume
                wireframeObstacles={false}
                transparentObstacles={false}
                informationMode={informationMode}
                diagnosticVisibility={visibility}
                decision={readoutA.decision}
                onWebGlError={setSceneError}
                onResetCamera={resetCamera}
                linkedCamera={linkedCameras && (cameraMode === 'free' || cameraMode === 'top-down')}
                cameraLeader
                cameraPose={cameraPose}
                onCameraPose={publishCameraPose}
                sharedBoundsEpisode={episodeB}
              />
              <p className="comparison-terminal">{localTimeA >= episodeA.summary.duration_s ? `Episode ended · frozen at ${episodeA.summary.duration_s.toFixed(2)} s` : `Recorded state ${readoutA.frame.step} · local ${localTimeA.toFixed(2)} s`}</p>
            </div>
            <div>
              <div className="comparison-pane-title"><strong>RIGHT · POLICY B</strong><span>{policyText(episodeB.defender)} · {episodeB.attacker.policy_id} · {episodeB.environment.clutter} / {episodeB.environment.awareness} · eval seed {episodeB.simulation_seed}</span></div>
              <Battlefield
                panelId="B"
                heading="Policy B battlefield"
                episode={episodeB}
                timeS={localTimeB}
                cameraMode={cameraMode}
                resetCameraToken={resetCameraToken}
                showFullTrajectory={false}
                trails={trails}
                showDetectionVolume
                wireframeObstacles={false}
                transparentObstacles={false}
                informationMode={informationMode}
                diagnosticVisibility={visibility}
                decision={readoutB.decision}
                onWebGlError={setSceneError}
                onResetCamera={resetCamera}
                linkedCamera={linkedCameras && (cameraMode === 'free' || cameraMode === 'top-down')}
                cameraPose={cameraPose}
                onCameraPose={publishCameraPose}
                sharedBoundsEpisode={episodeA}
              />
              <p className="comparison-terminal">{localTimeB >= episodeB.summary.duration_s ? `Episode ended · frozen at ${episodeB.summary.duration_s.toFixed(2)} s` : `Recorded state ${readoutB.frame.step} · local ${localTimeB.toFixed(2)} s`}</p>
            </div>
          </section>
          {sceneError && <p className="inline-error" role="alert">3D scene error: {sceneError}</p>}

          <section className="comparison-inspectors" aria-label="Independent policy diagnostics">
            <div className="diagnostics panel comparison-inspector">
              <h2>Policy A recorded diagnostics</h2>
              <PolicyInspector episode={episodeA} replayId={replayIdA} readout={readoutA} diagnosticSeries={seriesA}
                informationMode={informationMode} visibility={visibility} onInformationMode={setInformationMode}
                onVisibility={setVisibility} comparisonTimeS={localTimeA} hideInformationModeControl />
            </div>
            <div className="diagnostics panel comparison-inspector">
              <h2>Policy B recorded diagnostics</h2>
              <PolicyInspector episode={episodeB} replayId={replayIdB} readout={readoutB} diagnosticSeries={seriesB}
                informationMode={informationMode} visibility={visibility} onInformationMode={setInformationMode}
                onVisibility={setVisibility} comparisonTimeS={localTimeB} hideInformationModeControl />
            </div>
          </section>

          <section className="comparison-summary panel" aria-label="Descriptive policy outcome comparison">
            <div className="comparison-heading"><div><span className="eyebrow">PER-EPISODE RECORDED OUTCOMES</span><h2>Descriptive comparison · not a policy ranking</h2></div></div>
            {informationMode === 'ground-truth' ? (
              <div className="comparison-summary-grid">
                {summaries.map(({ episode, label, pathLength }) => {
                  const finalFrame = episode.frames[episode.frames.length - 1];
                  const collisions = episode.events.filter(event => event.kind === 'collision').length;
                  const safetyAvailable = episode.frames.some(frame => frame.telemetry?.avoidance_active !== undefined);
                  const safetyDecisions = new Set(episode.frames
                    .filter(frame => frame.telemetry?.avoidance_active)
                    .map(frame => frame.telemetry?.decision_step)
                    .filter((step): step is number => step !== undefined));
                  return (
                    <article key={label} className="comparison-summary-card">
                      <h3>{label} · <span className={outcomeClass(episode.summary.outcome)}>{episode.summary.outcome}</span></h3>
                      <p>Canonical outcome: {episode.summary.outcome === 'intercepted' ? 'Success' : 'Failure or non-success'} · duration {episode.summary.duration_s.toFixed(2)} s</p>
                      <p>Detection {episode.summary.detection_time_s === null ? 'not recorded' : `${episode.summary.detection_time_s.toFixed(2)} s`} · interception {episode.summary.interception_time_s === null ? 'not recorded' : `${episode.summary.interception_time_s.toFixed(2)} s`}</p>
                      <p>Defender path length {pathLength.toFixed(2)} m · recorded obstacle collisions {collisions} · safety-intervention decision frames {safetyAvailable ? safetyDecisions.size : 'unavailable (no safety telemetry)'}</p>
                      <p>Final defender–attacker range {finalFrame.privileged_diagnostics.defender_attacker_range_m.toFixed(2)} m · attacker–asset range {finalFrame.privileged_diagnostics.attacker_asset_range_m.toFixed(2)} m</p>
                    </article>
                  );
                })}
              </div>
            ) : <p className="muted">Privileged outcomes, event times, collision counts, final ranges, and attacker-relative metrics are hidden in Policy-Visible mode.</p>}
            <p className="comparison-note">Differences describe these two recorded episodes only. A matched seed and shared initial geometry do not guarantee identical subsequent attacker, asset, or stochastic trajectories.</p>
          </section>

          <ComparisonDiagnosticCharts
            episodeA={episodeA}
            episodeB={episodeB}
            seriesA={seriesA}
            seriesB={seriesB}
            axisTimeS={axisTimeS}
            axisStartS={activeRange.minimum}
            durationS={Math.max(1e-6, activeRange.maximum - activeRange.minimum)}
            detectionAligned={syncMode === 'detection'}
            informationMode={informationMode}
            onSeek={seekAxis}
          />
        </>
      )}

      <section className="comparison-seed-list panel" aria-label="Paired evaluation seeds">
        <div className="comparison-heading">
          <div><span className="eyebrow">MATCHED EVALUATION SEEDS</span><h2>Paired episode list</h2></div>
          <div><button type="button" disabled={seedIndex <= 0} onClick={() => selectSeed(seeds[seedIndex - 1])}>Previous seed</button> <button type="button" disabled={seedIndex >= seeds.length - 1} onClick={() => selectSeed(seeds[seedIndex + 1])}>Next seed</button></div>
        </div>
        <div className="paired-seed-table" role="table" aria-label="Evaluation seed outcome and replay validation">
          <div className="paired-seed-row paired-seed-header" role="row"><span>Seed</span><span>Policy A</span><span>Policy B</span><span>A−B duration</span><span>Replay availability</span><span>Pair status</span></div>
          {existingRows.map(({ seed, result, a, b }) => (
            <button type="button" className={`paired-seed-row${seed === selectedSeed ? ' selected' : ''}`} role="row" key={seed} onClick={() => selectSeed(seed)}>
              <span>{seed}</span><span>{informationMode === 'ground-truth' ? a?.outcome ?? 'missing' : a ? 'hidden' : 'missing'}</span><span>{informationMode === 'ground-truth' ? b?.outcome ?? 'missing' : b ? 'hidden' : 'missing'}</span>
              <span>{informationMode === 'ground-truth' && a && b
                && (result?.status === 'matched' || result?.status === 'exploratory')
                ? `${(a.duration_s - b.duration_s).toFixed(1)} s`
                : '—'}</span>
              <span>{a ? 'A ready' : 'A missing'} · {b ? 'B ready' : 'B missing'}</span>
              <span>{result?.status ?? (a && b ? 'awaiting validation' : 'incomplete')}</span>
            </button>
          ))}
        </div>
        {informationMode === 'ground-truth' && validatedBatchCounts.count > 0 && (
          <p className="muted">Descriptive outcomes from {validatedBatchCounts.count} validated pair(s): Policy A intercepted {validatedBatchCounts.aSuccesses}; Policy B intercepted {validatedBatchCounts.bSuccesses}. Exploratory batch counts are not publication-grade estimates.</p>
        )}
        {episodeA && episodeB && pairValidation?.status !== 'incompatible' && informationMode === 'ground-truth' && (
          <p className="muted">Current descriptive differences: duration {(episodeA.summary.duration_s - episodeB.summary.duration_s).toFixed(2)} s (A − B) · path length {(defenderPathLength(episodeA) - defenderPathLength(episodeB)).toFixed(2)} m (A − B).</p>
        )}
      </section>
    </div>
  );
}
