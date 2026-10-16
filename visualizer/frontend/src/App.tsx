import { useEffect, useState } from 'react';
import { Battlefield } from './Battlefield';

type BackendStatus = { kind: 'checking' | 'ok' | 'degraded' | 'error'; message: string };

function useBackendHealth(): BackendStatus {
  const [status, setStatus] = useState<BackendStatus>({ kind: 'checking', message: 'Checking backend' });
  useEffect(() => {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 5000);
    async function check() {
      try {
        const response = await fetch('/api/health', { signal: controller.signal });
        if (!response.ok) throw new Error(`Backend returned HTTP ${response.status}`);
        const data: unknown = await response.json();
        if (typeof data !== 'object' || data === null || !('status' in data) ||
            !('service' in data) || data.service !== 'counter-uas-visualizer' ||
            !('phase' in data) || data.phase !== 1 ||
            (data.status !== 'ok' && data.status !== 'degraded')) {
          throw new Error('Backend returned an unexpected health response');
        }
        if (!active) return;
        setStatus(data.status === 'ok'
          ? { kind: 'ok', message: 'Backend connected / data root configured' }
          : { kind: 'degraded', message: 'Backend connected / research data root not configured' });
      } catch (error) {
        if (!active) return;
        console.error('Backend health check failed', error);
        setStatus({ kind: 'error', message: 'Backend unavailable / start the localhost API on port 8000' });
      } finally {
        clearTimeout(timeout);
      }
    }
    let active = true;
    void check();
    return () => { active = false; clearTimeout(timeout); controller.abort(); };
  }, []);
  return status;
}

const placeholders = [
  ['Defender policy', 'Policy catalog available in Phase 3'],
  ['Attacker policy', 'A0 / A1 selection available in Phase 3'],
  ['Clutter level', 'C0 / C1 / C2 / C3'],
  ['Awareness level', 'B: local / C: map + local'],
] as const;

export function App() {
  const health = useBackendHealth();
  return (
    <main className="app-shell">
      <header className="app-header">
        <div className="brand-mark" aria-hidden="true">CU</div>
        <div className="brand"><span className="eyebrow">LOCAL RESEARCH WORKSPACE</span><h1>Counter-UAS <span>Simulation Explorer</span></h1></div>
        <span className="phase-badge">PHASE 1 / FOUNDATION</span>
      </header>
      <div className="status-bar"><span className={`status-dot ${health.kind}`} /><span role="status">{health.message}</span><span className="localhost-label">LOCALHOST ONLY</span></div>
      <div className="dashboard">
        <aside className="configuration panel">
          <span className="eyebrow">EXPERIMENT SETUP</span><h2>Episode configuration</h2>
          <p className="muted">Planned controls. No simulation can be launched in this phase.</p>
          <fieldset disabled>
            <legend className="sr-only">Episode setup is unavailable until Phase 3</legend>
            {placeholders.map(([label, text]) => <label key={label}>{label}<select aria-label={label}><option>{text}</option></select></label>)}
            <label>Episode count<select aria-label="Episode count" defaultValue="5"><option value="1">1 episode</option><option value="3">3 episodes</option><option value="5">5 episodes (planned default)</option><option value="10">10 episodes</option></select></label>
            <label>Starting simulation seed<input aria-label="Starting simulation seed" placeholder="Consecutive seeds / not yet assigned" /></label>
            <button className="primary-button" type="button">Run episodes / Phase 3</button>
          </fieldset>
          <div className="integrity-note"><strong>Research integrity first</strong><p>Frozen checkpoints are read-only. No training, fabricated frames, or browser-side dynamics.</p></div>
        </aside>
        <div className="scene-column">
          <Battlefield />
          <section className="playback panel" aria-label="Playback controls">
            <div><span className="eyebrow">RECORDED PLAYBACK</span><span className="muted">Available in Phase 4</span></div>
            <fieldset disabled>
              <legend className="sr-only">Playback is unavailable without a real recorded episode</legend>
              <div className="playback-buttons">
                {['Previous episode', 'Step back', 'Play', 'Step forward', 'Replay', 'Next episode'].map(text => <button key={text}>{text}</button>)}
                <select aria-label="Playback speed"><option>1x speed</option></select>
              </div>
              <input type="range" aria-label="Episode timeline" min="0" max="1" value="0" readOnly />
            </fieldset>
            <span className="muted">No frames / no outcome</span>
          </section>
        </div>
        <aside className="diagnostics panel">
          <span className="eyebrow">POLICY INSPECTOR</span><h2>Decision diagnostics</h2><span className="small-badge">PLANNED / PHASE 5</span>
          <p className="muted">Telemetry appears only when captured from a real controller.</p>
          <div className="diagnostic-card"><h3>LR-PPO guidance</h3><p>Lead direction / requested heading / executed movement</p><strong>Canonical correction: 30 deg maximum</strong><p>Signed horizontal adjustment and native 3D angle are distinct.</p></div>
          <div className="diagnostic-card"><h3>Estimation &amp; actions</h3><p>Policy-visible estimates will be separated from privileged ground truth.</p></div>
          <div className="legend"><h3>Planned entity colors</h3><span><i className="defender" />Defender UAV</span><span><i className="attacker" />Attacker UAV</span><span><i className="asset" />Protected mobile asset</span><span><i className="obstacle" />Moving obstacles</span></div>
          <p className="projection-note">Projection is visual only. Native 3D state and collision semantics remain unchanged.</p>
        </aside>
      </div>
      <footer className="app-footer"><span>Real simulation. Recorded evidence. No synthetic results.</span><span>Orbit camera active / follow and top-down modes planned</span></footer>
    </main>
  );
}
