/**
 * Simplified research interface (brief: "Replace the normal app workflow with a very simple research
 * interface"). The default root page -- see `main.tsx` for the `/legacy` routing that keeps the full,
 * historical `App` component available, unchanged, for development use.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import {
  DEFENDER_IDS,
  DEFENDER_LABELS,
  EpisodesApiError,
  createEpisode,
  deleteEpisode,
  deleteEpisodes,
  friendlyOutcome,
  getEpisode,
  getScenarioPresets,
  listEpisodes,
} from './episodes';
import type { EpisodeDetail, EpisodeMetadata, ScenarioId, ScenarioPreset } from './episodes';
import { TrajectoryViewer } from './TrajectoryViewer';
import { getPolicies } from './api';

type Tab = 'new' | 'saved';
type SortOrder = 'newest' | 'oldest';

interface Toast {
  id: number;
  kind: 'success' | 'error';
  message: string;
}

const SCENARIO_DESCRIPTIONS: Record<ScenarioId, string> = {
  no_clutter: 'Open environment with the scripted attacker.',
  clutter: 'Obstacle-rich environment with the evasive CBF attacker.',
};

const SCENARIO_TITLES: Record<ScenarioId, string> = {
  no_clutter: 'No Clutter',
  clutter: 'Clutter',
};

function attackerDisplay(resolved: 'A0' | 'A1-CBF' | undefined): string {
  return resolved === 'A1-CBF' ? 'Evasive CBF' : 'Scripted';
}

function randomSeed(): number {
  const buffer = new Uint32Array(1);
  crypto.getRandomValues(buffer);
  return buffer[0];
}

function formatDateTime(iso: string): string {
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}

function useBackendHealthy(): boolean | null {
  const [healthy, setHealthy] = useState<boolean | null>(null);
  useEffect(() => {
    let cancelled = false;
    const check = () => {
      fetch('/api/health')
        .then(r => (r.ok ? r.json() : Promise.reject(new Error('unhealthy'))))
        .then(() => { if (!cancelled) setHealthy(true); })
        .catch(() => { if (!cancelled) setHealthy(false); });
    };
    check();
    const interval = setInterval(check, 20000);
    return () => { cancelled = true; clearInterval(interval); };
  }, []);
  return healthy;
}

function ScenarioCard({
  scenario, active, onSelect,
}: { scenario: ScenarioId; active: boolean; onSelect: () => void }) {
  return (
    <button
      type="button"
      className={`scenario-card${active ? ' scenario-card-active' : ''}`}
      aria-pressed={active}
      onClick={onSelect}
    >
      <strong>{SCENARIO_TITLES[scenario]}</strong>
      <span>{SCENARIO_DESCRIPTIONS[scenario]}</span>
    </button>
  );
}

interface ConfirmState {
  title: string;
  body: string;
  confirmLabel: string;
  onConfirm: () => void;
}

function ConfirmDialog({ state, onCancel }: { state: ConfirmState; onCancel: () => void }) {
  const confirmRef = useRef<HTMLButtonElement | null>(null);
  useEffect(() => { confirmRef.current?.focus(); }, []);
  return (
    <div className="modal-backdrop" role="presentation" onClick={onCancel}>
      <div
        className="modal-dialog"
        role="dialog"
        aria-modal="true"
        aria-label={state.title}
        onClick={e => e.stopPropagation()}
      >
        <h3>{state.title}</h3>
        <p>{state.body}</p>
        <div className="modal-actions">
          <button type="button" onClick={onCancel}>Cancel</button>
          <button type="button" className="danger" ref={confirmRef} onClick={state.onConfirm}>{state.confirmLabel}</button>
        </div>
      </div>
    </div>
  );
}

function EpisodeRow({
  episode, selectMode, selected, onToggleSelect, onReplay, onDelete,
}: {
  episode: EpisodeMetadata;
  selectMode: boolean;
  selected: boolean;
  onToggleSelect: () => void;
  onReplay: () => void;
  onDelete: () => void;
}) {
  return (
    <li className={`episode-row${episode.is_new ? ' episode-row-new' : ''}`}>
      {selectMode && (
        <input
          type="checkbox"
          checked={selected}
          onChange={onToggleSelect}
          aria-label={`Select episode ${episode.episode_id}`}
        />
      )}
      <div className="episode-row-main">
        <div className="episode-row-line1">
          <span className={`badge badge-scenario-${episode.scenario}`}>{episode.scenario_label}</span>
          <span className="episode-defender">{DEFENDER_LABELS[episode.defender] ?? episode.defender}</span>
          <span className="episode-outcome">{friendlyOutcome(episode.outcome)}</span>
          {episode.is_new && <span className="badge badge-new">NEW</span>}
        </div>
        <div className="episode-row-line2 muted">
          {formatDateTime(episode.created_at)} &middot; seed {episode.seed} &middot; {episode.duration.toFixed(1)}s &middot; {episode.steps} steps
        </div>
      </div>
      <div className="episode-row-actions">
        <button type="button" onClick={onReplay}>Replay</button>
        {!selectMode && (
          <button type="button" className="icon-button" aria-label="Delete episode" title="Delete" onClick={onDelete}>🗑</button>
        )}
      </div>
    </li>
  );
}

export function SimplifiedApp() {
  const healthy = useBackendHealthy();
  const [presets, setPresets] = useState<ScenarioPreset[]>([]);
  const [scenario, setScenario] = useState<ScenarioId>('clutter');
  const [defender, setDefender] = useState('lr_ppo');
  const [defenderTouched, setDefenderTouched] = useState(false);
  const [availability, setAvailability] = useState<Record<string, { available: boolean; reason: string | null }>>({});
  const [seedText, setSeedText] = useState('');
  const [running, setRunning] = useState(false);
  const [runError, setRunError] = useState<string | null>(null);

  const [tab, setTab] = useState<Tab>('new');
  const [newEpisodes, setNewEpisodes] = useState<EpisodeMetadata[]>([]);
  const [savedEpisodes, setSavedEpisodes] = useState<EpisodeMetadata[]>([]);
  const [newCount, setNewCount] = useState(0);
  const [savedCount, setSavedCount] = useState(0);
  const [listError, setListError] = useState<string | null>(null);
  const [scenarioFilter, setScenarioFilter] = useState<'all' | ScenarioId>('all');
  const [defenderFilter, setDefenderFilter] = useState<'all' | string>('all');
  const [sortOrder, setSortOrder] = useState<SortOrder>('newest');

  const [selectedEpisodeId, setSelectedEpisodeId] = useState<string | null>(null);
  const [selectedDetail, setSelectedDetail] = useState<EpisodeDetail | null>(null);
  const [detailError, setDetailError] = useState<string | null>(null);
  const [detailsOpen, setDetailsOpen] = useState(false);

  const [selectMode, setSelectMode] = useState(false);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [confirm, setConfirm] = useState<ConfirmState | null>(null);
  const [bulkFailures, setBulkFailures] = useState<string[]>([]);

  const [toasts, setToasts] = useState<Toast[]>([]);
  const toastIdRef = useRef(0);

  const pushToast = useCallback((kind: Toast['kind'], message: string) => {
    const id = ++toastIdRef.current;
    setToasts(current => [...current, { id, kind, message }]);
    setTimeout(() => setToasts(current => current.filter(t => t.id !== id)), 5000);
  }, []);

  useEffect(() => {
    let cancelled = false;
    getScenarioPresets().then(result => { if (!cancelled) setPresets(result.presets); }).catch(() => undefined);
    getPolicies().then(catalog => {
      if (cancelled) return;
      const map: Record<string, { available: boolean; reason: string | null }> = {};
      for (const d of catalog.defenders) map[d.policy_id] = { available: d.available, reason: d.unavailability_reason };
      setAvailability(map);
      if (!defenderTouched && map.lr_ppo && map.lr_ppo.available === false) setDefender('greedy');
    }).catch(() => undefined);
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const refreshLists = useCallback(() => {
    setListError(null);
    Promise.all([
      listEpisodes({ new: true, scenario: scenarioFilter === 'all' ? undefined : scenarioFilter, defender: defenderFilter === 'all' ? undefined : defenderFilter }),
      listEpisodes({ new: false, scenario: scenarioFilter === 'all' ? undefined : scenarioFilter, defender: defenderFilter === 'all' ? undefined : defenderFilter }),
    ]).then(([newResult, savedResult]) => {
      setNewEpisodes(newResult.episodes);
      setSavedEpisodes(savedResult.episodes);
      setNewCount(newResult.new_count);
      setSavedCount(newResult.saved_count);
    }).catch(error => {
      setListError(error instanceof EpisodesApiError ? error.message : 'Could not load episodes.');
    });
  }, [scenarioFilter, defenderFilter]);

  useEffect(() => { refreshLists(); }, [refreshLists]);

  // Default tab: New if any new episodes exist, otherwise Saved (brief section 9) -- only applied once,
  // on first successful load, so it never fights a user's manual tab choice afterward.
  const tabDefaultApplied = useRef(false);
  useEffect(() => {
    if (tabDefaultApplied.current) return;
    if (newEpisodes.length === 0 && savedEpisodes.length === 0 && newCount === 0 && savedCount === 0) return;
    tabDefaultApplied.current = true;
    setTab(newCount > 0 ? 'new' : 'saved');
  }, [newEpisodes, savedEpisodes, newCount, savedCount]);

  const sortedList = useCallback((items: EpisodeMetadata[]) => {
    const copy = [...items];
    copy.sort((a, b) => sortOrder === 'newest'
      ? b.created_at.localeCompare(a.created_at)
      : a.created_at.localeCompare(b.created_at));
    return copy;
  }, [sortOrder]);

  const visibleEpisodes = sortedList(tab === 'new' ? newEpisodes : savedEpisodes);

  const runEpisode = useCallback(() => {
    if (running) return;
    setRunning(true);
    setRunError(null);
    const seed = seedText.trim() === '' ? undefined : Number(seedText.trim());
    if (seed !== undefined && (!Number.isInteger(seed) || seed < 0)) {
      setRunError('Seed must be a non-negative whole number.');
      setRunning(false);
      return;
    }
    createEpisode({ scenario, defender, seed })
      .then(created => {
        pushToast('success', `Episode generated \u2014 ${friendlyOutcome(created.outcome)}.`);
        refreshLists();
        setTab('new');
      })
      .catch(error => {
        setRunError(error instanceof EpisodesApiError ? error.message : 'Could not run the episode.');
      })
      .finally(() => setRunning(false));
  }, [running, scenario, defender, seedText, pushToast, refreshLists]);

  const openEpisode = useCallback((episodeId: string) => {
    setSelectedEpisodeId(episodeId);
    setDetailError(null);
    getEpisode(episodeId)
      .then(detail => {
        setSelectedDetail(detail);
        refreshLists();
      })
      .catch(error => {
        setDetailError(error instanceof EpisodesApiError ? error.message : 'Could not load this episode.');
      });
  }, [refreshLists]);

  const removeFromViewerIfActive = useCallback((episodeId: string) => {
    if (selectedEpisodeId === episodeId) {
      setSelectedEpisodeId(null);
      setSelectedDetail(null);
    }
  }, [selectedEpisodeId]);

  const requestSingleDelete = useCallback((episode: EpisodeMetadata) => {
    setConfirm({
      title: episode.is_new ? 'Delete this episode?' : 'Delete this saved episode? This removes its stored replay.',
      body: 'This action cannot be undone.',
      confirmLabel: 'Delete',
      onConfirm: () => {
        setConfirm(null);
        deleteEpisode(episode.episode_id)
          .then(() => {
            pushToast('success', 'Episode deleted.');
            removeFromViewerIfActive(episode.episode_id);
            refreshLists();
          })
          .catch(error => pushToast('error', error instanceof EpisodesApiError ? error.message : 'Could not delete the episode.'));
      },
    });
  }, [pushToast, refreshLists, removeFromViewerIfActive]);

  const requestBulkDelete = useCallback(() => {
    const ids = [...selectedIds];
    if (ids.length === 0) return;
    setConfirm({
      title: `Delete ${ids.length} saved episode${ids.length === 1 ? '' : 's'}?`,
      body: 'Their stored replay data will also be removed.',
      confirmLabel: 'Delete',
      onConfirm: () => {
        setConfirm(null);
        setBulkFailures([]);
        deleteEpisodes(ids)
          .then(response => {
            const failed = response.results.filter(r => !r.deleted).map(r => r.episode_id);
            if (failed.length > 0) {
              setBulkFailures(failed);
              pushToast('error', `${failed.length} episode(s) could not be deleted.`);
            } else {
              pushToast('success', `${ids.length} episode(s) deleted.`);
            }
            for (const id of ids) removeFromViewerIfActive(id);
            setSelectedIds(new Set());
            setSelectMode(false);
            refreshLists();
          })
          .catch(error => pushToast('error', error instanceof EpisodesApiError ? error.message : 'Could not delete the selected episodes.'));
      },
    });
  }, [selectedIds, pushToast, refreshLists, removeFromViewerIfActive]);

  const toggleSelected = useCallback((episodeId: string) => {
    setSelectedIds(current => {
      const next = new Set(current);
      if (next.has(episodeId)) next.delete(episodeId); else next.add(episodeId);
      return next;
    });
  }, []);

  const defenderUnavailable = availability[defender]?.available === false;
  const activePreset = presets.find(p => p.scenario === scenario);

  return (
    <div className="simplified-app app-shell">
      <header className="app-header simplified-header">
        <div>
          <h1>Counter-UAS Simulation</h1>
          <p className="muted subtitle">Evaluate defender strategies against scripted and evasive attackers.</p>
        </div>
        <span className={`status-indicator ${healthy === false ? 'status-bad' : 'status-ok'}`} role="status">
          {healthy === false ? 'Simulation Unavailable' : 'Simulation Ready'}
        </span>
        <a className="legacy-link" href="/legacy">Legacy interface</a>
      </header>

      <div className="simplified-layout">
        <aside className="control-column panel">
          <h2>Experiment</h2>
          <div className="scenario-cards" role="group" aria-label="Scenario">
            <ScenarioCard scenario="no_clutter" active={scenario === 'no_clutter'} onSelect={() => setScenario('no_clutter')} />
            <ScenarioCard scenario="clutter" active={scenario === 'clutter'} onSelect={() => setScenario('clutter')} />
          </div>
          <p className="attacker-info muted">Attacker: {attackerDisplay(activePreset?.resolved_attacker)}</p>

          <label className="field">
            Defender
            <select value={defender} onChange={e => { setDefender(e.target.value); setDefenderTouched(true); }}>
              {DEFENDER_IDS.map(id => (
                <option key={id} value={id} disabled={availability[id]?.available === false}>
                  {DEFENDER_LABELS[id]}{availability[id]?.available === false ? ' (unavailable)' : ''}
                </option>
              ))}
            </select>
          </label>
          {defenderUnavailable && (
            <p className="inline-warning" role="alert">
              {availability[defender]?.reason ?? 'Selected defender is currently unavailable.'}
            </p>
          )}

          <label className="field seed-field">
            Seed (optional)
            <div className="seed-row">
              <input
                type="text"
                inputMode="numeric"
                value={seedText}
                onChange={e => setSeedText(e.target.value)}
                placeholder="auto"
                aria-label="Seed (optional)"
              />
              <button type="button" className="icon-button" title="Randomize" aria-label="Randomize seed" onClick={() => setSeedText(String(randomSeed()))}>🎲</button>
            </div>
          </label>

          <button type="button" className="run-button primary" onClick={runEpisode} disabled={running || defenderUnavailable}>
            {running ? 'Running\u2026' : 'Run Episode'}
          </button>
          {runError && <p className="inline-error" role="alert">{runError}</p>}
        </aside>

        <main className="viewer-column panel">
          <h2>Replay</h2>
          {detailError && <p className="inline-error" role="alert">{detailError}</p>}
          {!selectedDetail && !detailError && <p className="muted">Select an episode from the library and choose Replay.</p>}
          {selectedDetail && (
            <>
              <TrajectoryViewer
                steps={selectedDetail.replay.steps}
                obstacles={selectedDetail.replay.obstacles}
                resetKey={selectedDetail.metadata.episode_id}
              />
              <div className="current-episode-summary">
                <span>Scenario: <strong>{selectedDetail.metadata.scenario_label}</strong></span>
                <span>Attacker: <strong>{attackerDisplay(selectedDetail.metadata.resolved_attacker)}</strong></span>
                <span>Defender: <strong>{DEFENDER_LABELS[selectedDetail.metadata.defender] ?? selectedDetail.metadata.defender}</strong></span>
                <span>Seed: <strong>{selectedDetail.metadata.seed}</strong></span>
                <span>Outcome: <strong>{friendlyOutcome(selectedDetail.metadata.outcome)}</strong></span>
                <span>Duration: <strong>{selectedDetail.metadata.duration.toFixed(1)}s</strong></span>
                <span>Steps: <strong>{selectedDetail.metadata.steps}</strong></span>
                {selectedDetail.metadata.min_defender_separation_m !== null && (
                  <span>Min defender separation: <strong>{selectedDetail.metadata.min_defender_separation_m.toFixed(2)} m</strong></span>
                )}
                {selectedDetail.metadata.min_obstacle_clearance_m !== null && (
                  <span>Min obstacle clearance: <strong>{selectedDetail.metadata.min_obstacle_clearance_m.toFixed(2)} m</strong></span>
                )}
                {selectedDetail.metadata.path_length_m !== null && (
                  <span>Path length: <strong>{selectedDetail.metadata.path_length_m.toFixed(1)} m</strong></span>
                )}
              </div>
              <button type="button" className="details-toggle" aria-expanded={detailsOpen} onClick={() => setDetailsOpen(o => !o)}>
                {detailsOpen ? 'Hide details \u25B4' : 'Details \u25BE'}
              </button>
              {detailsOpen && (
                <div className="details-drawer">
                  <span>Resolved attacker: <strong>{selectedDetail.metadata.resolved_attacker}</strong></span>
                  <span>Source commit: <code>{selectedDetail.metadata.source_commit}</code></span>
                  <span>Controller version: <code>{selectedDetail.metadata.controller_version}</code></span>
                  <span>Defender id: <code>{selectedDetail.metadata.defender}</code></span>
                  <span>Episode id: <code>{selectedDetail.metadata.episode_id}</code></span>
                  <span>Asset captured: <strong>{selectedDetail.metadata.asset_captured ? 'yes' : 'no'}</strong></span>
                  <span>Defender interception: <strong>{selectedDetail.metadata.defender_interception ? 'yes' : 'no'}</strong></span>
                  <span>Obstacle collision: <strong>{selectedDetail.metadata.obstacle_collision ? 'yes' : 'no'}</strong></span>
                </div>
              )}
            </>
          )}
        </main>

        <aside className="library-column panel">
          <h2>Episodes</h2>
          <div className="library-tabs" role="tablist">
            <button type="button" role="tab" aria-selected={tab === 'new'} onClick={() => setTab('new')}>New ({newCount})</button>
            <button type="button" role="tab" aria-selected={tab === 'saved'} onClick={() => setTab('saved')}>Saved ({savedCount})</button>
          </div>

          <div className="library-filters">
            <label>
              Scenario
              <select value={scenarioFilter} onChange={e => setScenarioFilter(e.target.value as 'all' | ScenarioId)}>
                <option value="all">All</option>
                <option value="no_clutter">No Clutter</option>
                <option value="clutter">Clutter</option>
              </select>
            </label>
            <label>
              Defender
              <select value={defenderFilter} onChange={e => setDefenderFilter(e.target.value)}>
                <option value="all">All</option>
                {DEFENDER_IDS.map(id => <option key={id} value={id}>{DEFENDER_LABELS[id]}</option>)}
              </select>
            </label>
            <label>
              Sort
              <select value={sortOrder} onChange={e => setSortOrder(e.target.value as SortOrder)}>
                <option value="newest">Newest first</option>
                <option value="oldest">Oldest first</option>
              </select>
            </label>
            <button type="button" onClick={refreshLists}>Refresh</button>
          </div>

          {tab === 'saved' && (
            <div className="bulk-controls">
              {!selectMode ? (
                <button type="button" onClick={() => setSelectMode(true)} disabled={visibleEpisodes.length === 0}>Select</button>
              ) : (
                <>
                  <button type="button" onClick={() => setSelectedIds(new Set(visibleEpisodes.map(e => e.episode_id)))}>Select all visible</button>
                  <button type="button" className="danger" onClick={requestBulkDelete} disabled={selectedIds.size === 0}>Delete selected ({selectedIds.size})</button>
                  <button type="button" onClick={() => { setSelectMode(false); setSelectedIds(new Set()); }}>Cancel</button>
                </>
              )}
            </div>
          )}
          {bulkFailures.length > 0 && (
            <p className="inline-error" role="alert">Could not delete: {bulkFailures.join(', ')}</p>
          )}

          {listError && <p className="inline-error" role="alert">{listError}</p>}
          {!listError && visibleEpisodes.length === 0 && (
            <p className="muted">
              {tab === 'new'
                ? (newCount === 0 && scenarioFilter === 'all' && defenderFilter === 'all'
                  ? 'No new episodes. Run an episode to create one.'
                  : 'No episodes match these filters.')
                : (savedCount === 0 && scenarioFilter === 'all' && defenderFilter === 'all'
                  ? 'No saved episodes yet.'
                  : 'No episodes match these filters.')}
            </p>
          )}
          <ul className="episode-list">
            {visibleEpisodes.map(episode => (
              <EpisodeRow
                key={episode.episode_id}
                episode={episode}
                selectMode={selectMode && tab === 'saved'}
                selected={selectedIds.has(episode.episode_id)}
                onToggleSelect={() => toggleSelected(episode.episode_id)}
                onReplay={() => openEpisode(episode.episode_id)}
                onDelete={() => requestSingleDelete(episode)}
              />
            ))}
          </ul>
        </aside>
      </div>

      {confirm && <ConfirmDialog state={confirm} onCancel={() => setConfirm(null)} />}
      <div className="toast-stack" aria-live="polite">
        {toasts.map(toast => (
          <div key={toast.id} className={`toast toast-${toast.kind}`}>{toast.message}</div>
        ))}
      </div>
    </div>
  );
}
