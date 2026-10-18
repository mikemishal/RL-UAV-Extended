import { useEffect, useState } from 'react';
import {
  addResearchBookmark,
  createChartCapture,
  createCaseExport,
  createResearchCase,
  deleteResearchBookmark,
  deleteResearchCase,
  getResearchCases,
  updateResearchBookmark,
  updateResearchCase,
} from './api';
import type { ReplayListItem, ResearchBookmark, ResearchCase } from './api';

async function rasterizeChart(file: { download_url: string; name: string; sha256: string }): Promise<{
  png_base64: string; width: 2100; height: number;
}> {
  const response = await fetch(file.download_url);
  if (!response.ok) throw new Error(`Could not read SVG chart ${file.name} for PNG rasterization.`);
  const svg = await response.text();
  const viewBox = /viewBox=["']\s*([-\d.eE]+)[,\s]+([-\d.eE]+)[,\s]+([\d.eE]+)[,\s]+([\d.eE]+)\s*["']/.exec(svg);
  if (!viewBox) throw new Error(`SVG chart ${file.name} has no valid viewBox.`);
  const viewWidth = Number(viewBox[3]);
  const viewHeight = Number(viewBox[4]);
  const height = Math.round(2100 * viewHeight / viewWidth);
  if (!Number.isFinite(height) || height < 200 || height > 4096) throw new Error(`SVG chart ${file.name} exceeds supported PNG dimensions.`);
  const sourceUrl = URL.createObjectURL(new Blob([svg], { type: 'image/svg+xml;charset=utf-8' }));
  try {
    const image = new Image();
    image.src = sourceUrl;
    await image.decode();
    const canvas = document.createElement('canvas');
    canvas.width = 2100;
    canvas.height = height;
    const context = canvas.getContext('2d', { alpha: false });
    if (!context) throw new Error('Browser could not allocate a 300-DPI chart canvas.');
    context.fillStyle = '#ffffff';
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.imageSmoothingEnabled = true;
    context.imageSmoothingQuality = 'high';
    context.drawImage(image, 0, 0, canvas.width, canvas.height);
    const png = await new Promise<Blob>((resolve, reject) => {
      canvas.toBlob(blob => blob ? resolve(blob) : reject(new Error('Browser PNG encoding failed.')), 'image/png');
    });
    const dataUrl = await new Promise<string>((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => typeof reader.result === 'string' ? resolve(reader.result) : reject(new Error('PNG encoding returned no data URL.'));
      reader.onerror = () => reject(new Error('Browser could not read its encoded PNG.'));
      reader.readAsDataURL(png);
    });
    return { png_base64: dataUrl.slice(dataUrl.indexOf(',') + 1), width: 2100, height };
  } finally {
    URL.revokeObjectURL(sourceUrl);
  }
}

interface ResearchCasesProps {
  replayCatalog: ReplayListItem[];
  replayIds: string[];
  comparison: boolean;
  syncMode: 'absolute' | 'detection';
  axisTimeS: number;
  localTimeA: number;
  localTimeB: number | null;
  frameStep: number;
  decisionStep: number | null;
  onRestore: (item: ResearchCase, bookmark?: ResearchBookmark) => void;
  onCaptureScene: (item: ResearchCase, replayId: string, timeS: number) => void;
  captureStatus: string | null;
}

export function ResearchCases({
  replayCatalog, replayIds, comparison, syncMode, axisTimeS, localTimeA, localTimeB, frameStep, decisionStep, onRestore, onCaptureScene, captureStatus,
}: ResearchCasesProps) {
  const [cases, setCases] = useState<ResearchCase[]>([]);
  const [caseType, setCaseType] = useState<'single' | 'comparison'>(comparison ? 'comparison' : 'single');
  const [replayA, setReplayA] = useState('');
  const [replayB, setReplayB] = useState('');
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [query, setQuery] = useState('');
  const [filters, setFilters] = useState<Record<string, string>>({ sort: 'modified' });
  const [title, setTitle] = useState('');
  const [description, setDescription] = useState('');
  const [tags, setTags] = useState('');
  const [notes, setNotes] = useState('');
  const [bookmarkLabel, setBookmarkLabel] = useState('');
  const [observation, setObservation] = useState('');
  const [interpretation, setInterpretation] = useState('');
  const [bookmarkTags, setBookmarkTags] = useState('');
  const [exportStart, setExportStart] = useState('');
  const [exportEnd, setExportEnd] = useState('');
  const [editingBookmarkId, setEditingBookmarkId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const selected = cases.find(item => item.case_id === selectedId) ?? null;
  const activeSelectionMatches = Boolean(selected && selected.replay_ids.length === replayIds.length
    && selected.replay_ids.every((id, index) => id === replayIds[index]));

  async function refresh(search = query, selectedFilters = filters) {
    try {
      const response = await getResearchCases(search, selectedFilters);
      setCases(response.cases);
      setSelectedId(current => response.cases.some(item => item.case_id === current) ? current : null);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Research cases could not be loaded.');
    }
  }

  useEffect(() => { void refresh(); }, []);
  useEffect(() => {
    if (!replayCatalog.length) return;
    setReplayA(current => current || replayCatalog[0]?.replay_id || '');
    setReplayB(current => current || replayCatalog[1]?.replay_id || '');
  }, [replayCatalog]);
  useEffect(() => {
    if (comparison === (caseType === 'comparison') && replayIds.length === (caseType === 'comparison' ? 2 : 1)) {
      setReplayA(replayIds[0]);
      if (caseType === 'comparison') setReplayB(replayIds[1]);
    }
  }, [caseType, comparison, replayIds.join('|')]);

  async function saveCase() {
    const expectedCount = caseType === 'comparison' ? 2 : 1;
    const currentIds = [replayA, ...(caseType === 'comparison' ? [replayB] : [])];
    if (currentIds.length !== expectedCount || currentIds.some(id => !id)) {
      setError('Select existing validated replay artifact(s) before saving a case.');
      return;
    }
    setBusy(true); setError(null); setMessage(null);
    try {
      const saved = await createResearchCase({
        case_type: caseType,
        title: title.trim() || `${caseType === 'comparison' ? 'Comparison' : 'Replay'} case ${new Date().toLocaleString()}`,
        description,
        tags: tags.split(',').map(item => item.trim()).filter(Boolean),
        notes,
        replay_ids: currentIds,
      });
      setCases(current => [saved, ...current.filter(item => item.case_id !== saved.case_id)]);
      setSelectedId(saved.case_id);
      setMessage(`Saved ${saved.case_type} case. Replay artifacts remain unchanged.`);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Case could not be saved.');
    } finally { setBusy(false); }
  }

  async function saveBookmark() {
    if (!selected || !activeSelectionMatches) {
      setError('Open this case in its recorded workspace before adding an annotation.');
      return;
    }
    setBusy(true); setError(null); setMessage(null);
    try {
      const saved = await addResearchBookmark(selected.case_id, selected.version, {
        label: bookmarkLabel.trim() || `Bookmark at ${axisTimeS.toFixed(2)} s`,
        time_s: axisTimeS,
        frame_step: frameStep,
        decision_step: decisionStep,
        sync_mode: syncMode,
        local_time_a_s: localTimeA,
        local_time_b_s: comparison ? localTimeB : null,
        observation,
        interpretation,
        tags: bookmarkTags.split(',').map(item => item.trim()).filter(Boolean),
      });
      setCases(current => current.map(item => item.case_id === saved.case_id ? saved : item));
      setBookmarkLabel(''); setObservation(''); setInterpretation(''); setBookmarkTags('');
      setMessage('Timestamped annotation saved with the current frame and synchronization metadata.');
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Bookmark could not be saved.');
    } finally { setBusy(false); }
  }

  async function removeSelected() {
    if (!selected || !window.confirm(`Delete case “${selected.title}”? Original replay files will not be deleted.`)) return;
    setBusy(true); setError(null);
    try {
      await deleteResearchCase(selected.case_id, selected.version);
      setCases(current => current.filter(item => item.case_id !== selected.case_id));
      setSelectedId(null); setMessage('Case sidecar deleted; source replay artifacts were preserved.');
    } catch (cause) { setError(cause instanceof Error ? cause.message : 'Case could not be deleted.'); }
    finally { setBusy(false); }
  }

  async function saveMetadata() {
    if (!selected) return;
    setBusy(true); setError(null);
    try {
      const saved = await updateResearchCase(selected.case_id, {
        expected_version: selected.version,
        title: title.trim() || selected.title,
        description,
        tags: tags.split(',').map(item => item.trim()).filter(Boolean),
        notes,
      });
      setCases(current => current.map(item => item.case_id === saved.case_id ? saved : item));
      setMessage('Case metadata saved.');
    } catch (cause) { setError(cause instanceof Error ? cause.message : 'Case metadata could not be updated.'); }
    finally { setBusy(false); }
  }

  async function exportCase(kind: 'report' | 'episode_csv' | 'comparison_csv' | 'trajectory_svg' | 'telemetry_svg' | 'package_zip') {
    if (!selected) return;
    setBusy(true); setError(null);
    try {
      const start = exportStart.trim() ? Number(exportStart) : undefined;
      const end = exportEnd.trim() ? Number(exportEnd) : undefined;
      if ((start !== undefined && !Number.isFinite(start)) || (end !== undefined && !Number.isFinite(end))
          || (start !== undefined && end !== undefined && start >= end)) {
        throw new Error('Export interval must contain finite increasing timeline bounds.');
      }
      const request = {
        kind: kind === 'package_zip' ? 'report' as const : kind,
        ...(start !== undefined ? { start_time_s: start } : {}),
        ...(end !== undefined ? { end_time_s: end } : {}),
        sync_mode: selected.case_type === 'comparison' ? syncMode : 'absolute' as const,
        camera: 'recorded current view',
      };
      const chartExport = await createCaseExport(selected.case_id, request);
      const pngFiles = [];
      for (const file of chartExport.files.filter(item => item.name.endsWith('.svg'))) {
        const raster = await rasterizeChart(file);
        const capture = await createChartCapture(selected.case_id, chartExport.export_id, {
          source_filename: file.name,
          source_sha256: file.sha256,
          ...raster,
          dpi: 300,
        });
        const png = capture.files.find(item => item.name === file.name.replace(/\.svg$/, '.png'));
        if (!png) throw new Error(`Backend did not persist the PNG derived from ${file.name}.`);
        pngFiles.push(png);
      }
      const result = kind === 'package_zip'
        ? await createCaseExport(selected.case_id, { ...request, kind: 'package_zip' })
        : chartExport;
      for (const file of [...result.files, ...(kind === 'package_zip' ? [] : pngFiles)]) {
        const anchor = document.createElement('a');
        anchor.href = file.download_url;
        anchor.download = file.name;
        anchor.click();
      }
      setMessage(`Created ${kind.replaceAll('_', ' ')} with rasterized 300-DPI charts where applicable; checksums and source SVG references are recorded.`);
    } catch (cause) { setError(cause instanceof Error ? cause.message : 'Scientific export failed.'); }
    finally { setBusy(false); }
  }

  return (
    <section className="research-cases panel" aria-label="Research case library">
      <div className="research-cases-heading">
        <div><span className="eyebrow">LOCAL JSON SIDECARS · SOURCE REPLAYS READ-ONLY</span><h2>Research case library</h2></div>
        <label>Search title, notes, tags, and policy
          <input value={query} onChange={event => setQuery(event.target.value)} onKeyDown={event => {
            if (event.key === 'Enter') void refresh(query);
          }} />
        </label>
          <div className="case-filter-controls">
            <label>Case type<select value={filters.case_type ?? ''} onChange={event => setFilters(current => ({ ...current, case_type: event.target.value }))}>
              <option value="">All</option><option value="single">Single</option><option value="comparison">Comparison</option>
            </select></label>
            <label>Clutter<select value={filters.clutter ?? ''} onChange={event => setFilters(current => ({ ...current, clutter: event.target.value }))}>
              <option value="">All</option>{['C0', 'C1', 'C2', 'C3'].map(value => <option key={value}>{value}</option>)}
            </select></label>
            <label>Outcome<input value={filters.outcome ?? ''} onChange={event => setFilters(current => ({ ...current, outcome: event.target.value }))} /></label>
            <label>Policy architecture<input value={filters.architecture ?? ''} onChange={event => setFilters(current => ({ ...current, architecture: event.target.value }))} /></label>
            <label>Generation<input value={filters.generation ?? ''} onChange={event => setFilters(current => ({ ...current, generation: event.target.value }))} /></label>
            <label>Training seed<input type="number" value={filters.training_seed ?? ''} onChange={event => setFilters(current => ({ ...current, training_seed: event.target.value }))} /></label>
            <label>Attacker<input value={filters.attacker ?? ''} onChange={event => setFilters(current => ({ ...current, attacker: event.target.value }))} /></label>
            <label>Perception<input value={filters.perception ?? ''} onChange={event => setFilters(current => ({ ...current, perception: event.target.value }))} /></label>
            <label>Evaluation seed<input type="number" value={filters.evaluation_seed ?? ''} onChange={event => setFilters(current => ({ ...current, evaluation_seed: event.target.value }))} /></label>
            <label>Exact tag<input value={filters.tag ?? ''} onChange={event => setFilters(current => ({ ...current, tag: event.target.value }))} /></label>
            <label>Annotations<select value={filters.annotations ?? ''} onChange={event => setFilters(current => ({ ...current, annotations: event.target.value }))}>
              <option value="">Any</option><option value="true">Has annotations</option><option value="false">No annotations</option>
            </select></label>
            <label>Bookmarks<select value={filters.bookmarked ?? ''} onChange={event => setFilters(current => ({ ...current, bookmarked: event.target.value }))}>
              <option value="">Any</option><option value="true">Bookmarked</option><option value="false">No bookmarks</option>
            </select></label>
            <label>Sort<select value={filters.sort ?? 'modified'} onChange={event => setFilters(current => ({ ...current, sort: event.target.value }))}>
              {['modified', 'created', 'policy', 'outcome', 'duration'].map(value => <option key={value} value={value}>{value}</option>)}
            </select></label>
          </div>
          <button type="button" onClick={() => void refresh()} disabled={busy}>Search / filter</button>
      </div>
      {captureStatus && <p role="status" className="capture-status">{captureStatus}</p>}
      {error && <p className="inline-error" role="alert">{error}</p>}
      {message && <p className="case-success" role="status">{message}</p>}
      <div className="case-columns">
        <div>
          <h3>Save current recorded selection</h3>
          <p className="muted">Only existing replay IDs and their compressed-artifact hashes are stored; simulation output is never rewritten.</p>
          <label>Case title<input value={title} onChange={event => setTitle(event.target.value)} placeholder="Descriptive title" /></label>
          <label>Saved case type<select value={caseType} onChange={event => setCaseType(event.target.value as 'single' | 'comparison')}>
            <option value="single">Single replay</option><option value="comparison">Comparison pair</option>
          </select></label>
          <label>Replay A / single replay<select value={replayA} onChange={event => setReplayA(event.target.value)}>
            <option value="">Select validated replay</option>
            {replayCatalog.map(item => <option key={item.replay_id} value={item.replay_id}>
              {item.defender.policy_id} {item.defender.generation ?? ''} / {item.attacker.policy_id} / {item.clutter} {item.perception} / seed {item.simulation_seed} / {item.outcome} / {item.replay_id.slice(0, 8)}
            </option>)}
          </select></label>
          {caseType === 'comparison' && <label>Replay B<select value={replayB} onChange={event => setReplayB(event.target.value)}>
            <option value="">Select validated replay</option>
            {replayCatalog.map(item => <option key={item.replay_id} value={item.replay_id} disabled={item.replay_id === replayA}>
              {item.defender.policy_id} {item.defender.generation ?? ''} / {item.attacker.policy_id} / {item.clutter} {item.perception} / seed {item.simulation_seed} / {item.outcome} / {item.replay_id.slice(0, 8)}
            </option>)}
          </select></label>}
          <label>Description<input value={description} onChange={event => setDescription(event.target.value)} /></label>
          <label>Tags, comma separated<input value={tags} onChange={event => setTags(event.target.value)} /></label>
          <label>Research notes<textarea value={notes} onChange={event => setNotes(event.target.value)} rows={3} /></label>
          <button type="button" disabled={busy || !replayCatalog.length} onClick={() => void saveCase()}>
            Save {caseType === 'comparison' ? 'comparison' : 'single replay'} case
          </button>
          <h3 className="case-subheading">Saved cases ({cases.length})</h3>
          <ul className="case-list">
            {cases.map(item => <li key={item.case_id}>
              <button type="button" className="case-list-button" aria-pressed={item.case_id === selectedId} onClick={() => {
                setSelectedId(item.case_id); setTitle(item.title); setDescription(item.description); setTags(item.tags.join(', ')); setNotes(item.notes);
              }}>
                <strong>{item.title}</strong><span>{item.case_type} · {item.integrity} · {item.metadata.outcomes.join(', ') || 'outcome unavailable'}</span>
                <small>{item.metadata.defender_families.join(' / ')} · {item.metadata.clutter.join(', ')} · seeds {item.metadata.simulation_seeds.join(', ')}</small>
              </button>
            </li>)}
            {!cases.length && <li className="muted">No saved cases match this search.</li>}
          </ul>
        </div>
        <div>
          {!selected ? <p className="muted">Select a saved case to inspect provenance, restore its replay(s), annotate timestamps, or export analysis.</p> : <>
            <div className="case-detail-heading"><div><h3>{selected.title}</h3><p>{selected.case_type} · integrity: {selected.integrity} · {selected.comparison_status ?? 'single replay'}</p></div>
              <button type="button" onClick={() => onRestore(selected)} disabled={busy}>Open case</button>
            </div>
            {selected.comparison_reasons.length > 0 && <p className="muted">{selected.comparison_reasons.join(' ')}</p>}
            <p className="case-provenance">Policies: {selected.metadata.defender_families.join(' vs ')} · generations: {selected.metadata.generations.join(', ') || 'analytical'} · attacker: {selected.metadata.attackers.join(', ')} · {selected.metadata.clutter.join(', ')} / {selected.metadata.perception.join(', ')} · simulation seed: {selected.metadata.simulation_seeds.join(', ')}</p>
            <p className="case-provenance">Source commit(s): {selected.metadata.source_commits.join(', ')} · dirty provenance: {selected.metadata.dirty_provenance ? 'yes' : 'no'} · saved {new Date(selected.metadata.created_at).toLocaleString()}</p>
            <div className="case-form-grid">
              <label>Title<input value={title} onChange={event => setTitle(event.target.value)} /></label>
              <label>Description<input value={description} onChange={event => setDescription(event.target.value)} /></label>
              <label>Tags<input value={tags} onChange={event => setTags(event.target.value)} /></label>
              <label>Notes<textarea value={notes} onChange={event => setNotes(event.target.value)} rows={3} /></label>
            </div>
            <div className="case-actions">
              {selected.replay_ids.map((replayId, index) => (
                <button
                  type="button"
                  key={`capture-${replayId}`}
                  onClick={() => onCaptureScene(selected, replayId, index === 1 ? localTimeB ?? 0 : localTimeA)}
                  disabled={busy}
                >
                  Capture recorded scene · {index === 0 ? 'Policy A' : 'Policy B'}
                </button>
              ))}
              <button type="button" onClick={() => void saveMetadata()} disabled={busy}>Save metadata</button>
              <button type="button" onClick={() => void removeSelected()} disabled={busy}>Delete case only</button>
              <button type="button" onClick={() => void exportCase('report')} disabled={busy}>Markdown report</button>
              <button type="button" onClick={() => void exportCase(selected.case_type === 'comparison' ? 'comparison_csv' : 'episode_csv')} disabled={busy}>CSV</button>
              <button type="button" onClick={() => void exportCase('trajectory_svg')} disabled={busy}>Trajectory SVG</button>
              <button type="button" onClick={() => void exportCase('telemetry_svg')} disabled={busy}>Telemetry SVG</button>
              <button type="button" onClick={() => void exportCase('package_zip')} disabled={busy}>ZIP package</button>
            </div>
            <div className="case-export-range">
              <label>Export timeline start (s)<input type="number" step="any" value={exportStart} onChange={event => setExportStart(event.target.value)} placeholder="Beginning" /></label>
              <label>Export timeline end (s)<input type="number" step="any" value={exportEnd} onChange={event => setExportEnd(event.target.value)} placeholder="Episode end" /></label>
              <span className="muted">Blank bounds include the full recorded trace. Comparison exports use the {syncMode} shared timeline.</span>
            </div>
            <h3 className="case-subheading">Timestamped annotations</h3>
            <p className="muted">Current selection: axis {axisTimeS.toFixed(2)} s; Policy A local {localTimeA.toFixed(2)} s{comparison ? `; Policy B local ${localTimeB?.toFixed(2) ?? '—'} s (${syncMode})` : ''}; recorded frame {frameStep}.</p>
            <label>Bookmark title<input value={bookmarkLabel} onChange={event => setBookmarkLabel(event.target.value)} /></label>
            <label>Observed behavior<input value={observation} onChange={event => setObservation(event.target.value)} /></label>
            <label>Research interpretation<textarea value={interpretation} onChange={event => setInterpretation(event.target.value)} rows={2} /></label>
            <label>Annotation tags<input value={bookmarkTags} onChange={event => setBookmarkTags(event.target.value)} /></label>
            <button type="button" onClick={() => void saveBookmark()} disabled={busy || !activeSelectionMatches}>Add timestamped annotation</button>
            <ul className="bookmark-list">
              {selected.bookmarks.map(bookmark => <li key={bookmark.bookmark_id}>
                {editingBookmarkId === bookmark.bookmark_id ? <>
                  <label>Annotation title<input value={bookmarkLabel} onChange={event => setBookmarkLabel(event.target.value)} /></label>
                  <label>Observation<input value={observation} onChange={event => setObservation(event.target.value)} /></label>
                  <label>Interpretation<textarea rows={2} value={interpretation} onChange={event => setInterpretation(event.target.value)} /></label>
                  <label>Tags<input value={bookmarkTags} onChange={event => setBookmarkTags(event.target.value)} /></label>
                  <button type="button" disabled={busy} onClick={async () => {
                    setBusy(true); setError(null);
                    try {
                      const saved = await updateResearchBookmark(selected.case_id, bookmark.bookmark_id, selected.version, {
                        label: bookmarkLabel.trim(),
                        observation,
                        interpretation,
                        tags: bookmarkTags.split(',').map(item => item.trim()).filter(Boolean),
                      });
                      setCases(current => current.map(item => item.case_id === saved.case_id ? saved : item));
                      setEditingBookmarkId(null);
                    } catch (cause) { setError(cause instanceof Error ? cause.message : 'Annotation could not be updated.'); }
                    finally { setBusy(false); }
                  }}>Save annotation edits</button>
                </> : <>
                  <button type="button" onClick={() => onRestore(selected, bookmark)}>{bookmark.label} · axis {bookmark.time_s.toFixed(2)} s · frame {bookmark.frame_step}</button>
                  <span>{bookmark.observation || 'No observation'}{bookmark.interpretation ? ` — ${bookmark.interpretation}` : ''} {bookmark.tags.map(tag => `#${tag}`).join(' ')}</span>
                  <button type="button" disabled={busy} onClick={() => {
                    setEditingBookmarkId(bookmark.bookmark_id);
                    setBookmarkLabel(bookmark.label);
                    setObservation(bookmark.observation);
                    setInterpretation(bookmark.interpretation);
                    setBookmarkTags(bookmark.tags.join(', '));
                  }}>Edit annotation</button>
                </>}
                <button type="button" aria-label={`Delete annotation ${bookmark.label}`} disabled={busy} onClick={async () => {
                  setBusy(true); setError(null);
                  try {
                    const saved = await deleteResearchBookmark(selected.case_id, bookmark.bookmark_id, selected.version);
                    setCases(current => current.map(item => item.case_id === saved.case_id ? saved : item));
                  } catch (cause) { setError(cause instanceof Error ? cause.message : 'Annotation could not be deleted.'); }
                  finally { setBusy(false); }
                }}>Delete annotation</button>
              </li>)}
            </ul>
          </>}
        </div>
      </div>
    </section>
  );
}
