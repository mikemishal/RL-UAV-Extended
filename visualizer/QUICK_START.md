# Simulation Viewer quick start

## First-time setup

From PowerShell at the visualization repository root:

```powershell
.\visualizer\setup-visualizer.ps1
```

Enter the **absolute path** of the existing research Python that imports
Gymnasium, Stable-Baselines3, and PyTorch. The setup script creates the ignored
`visualizer\.env` only if it does not exist; it never installs research
packages. If the API or frontend dependencies are missing, install them once:

```powershell
.\visualizer\setup-visualizer.ps1 -InstallApiDependencies -InstallFrontendDependencies
```

## Launch and run

Start both local servers and open the browser:

```powershell
.\visualizer\start-visualizer.ps1
```

The default workspace is **Simulation Viewer**, configured for Greedy, A0, C0,
and one episode. Click **Run Episode** to queue the actual Python simulator,
validate its replay, load it automatically, and begin playback. For a batch,
choose 3, 5, or 10 episodes; evaluation seeds are consecutive. Each recording
remains separately selectable. Click **Watch Recorded Episode** to open the
newest available validated recording without regenerating it.

## Playback

- Use **Play/Pause**, **Restart**, or the **Timeline** scrubber. Playback speed
  and previous/next episode controls are beside them.
- Rotate with the mouse, zoom with the wheel, and pan with right-drag. Select
  **Free 3D**, **Follow defender**, **Follow attacker**, or **Top-down** in the
  camera presets. **Reset camera** restores the scene framing.
- **Fullscreen Viewer** includes the battlefield, camera presets, timeline,
  transport, and episode status. Press **Exit fullscreen** or Escape to return.
- **Compare Policies** and **Research Cases** remain in the top navigation.
  Switch to **Analysis** for the full diagnostics and research panels.

## Stop

Use the paired shutdown command:

```powershell
.\visualizer\stop-visualizer.ps1
```

It cancels active replay jobs belonging to the owned local server, then stops
only PIDs recorded by the launcher. It refuses a stale or reused PID rather
than terminating an unverified process. If port 8000 or 5173 was already in
use, the launcher reports the owner and leaves it untouched.

## Troubleshooting

- **Missing `.env`:** run the first-time setup and enter the scientific Python
  executable explicitly.
- **Scientific imports missing:** select the existing research environment;
  do not point at the lightweight API venv or install/upgrade research packages.
- **Policy catalog or registry unavailable:** verify the configured research
  root contains `results\defender_adaptation_model_registry.json`. The catalog
  reports unavailable/hash-mismatched checkpoints; it does not substitute one.
- **Port already occupied:** inspect the reported PID and stop that service
  yourself if appropriate; the launcher will not take over its port.
- **Server startup error:** inspect `visualizer\data\local\logs\backend.stderr.log`
  or `frontend.stderr.log`. After a failed launch, the launcher cleans up its
  own server processes.
- **WebGL unavailable:** use a browser with WebGL2 enabled. Existing validated
  recordings can be watched without starting a new simulation when the API
  and research runtime are unavailable.
