# V2 A1 first canonical training campaign — results (`a1_v2_training_v1` / `a1_v2_reward_v1`)

**Verdict: the frozen acceptance gates FAIL (5 of 5). No A1 policy captured the asset in any of the 6,000 A1 evaluation
episodes. The protocol, reward and PPO settings were not changed during the campaign; retraining requires a new reward
and/or protocol version.** No canonical A1 tag is created.

Data: `data/` (CSV/JSON summaries), `figures/` (learning curves, PNG + PDF), `inspection_shortlist.json`.
Models stay in `results/…/seed_*/checkpoints/` (the repository ignores `results/`; they are not in Git). Their SHA-256 values are in
`data/a1_v2_model_registry.json`.

## 1. Campaign record

| Item | Value |
|---|---|
| Research commit during training/evaluation | `b881bddad2cb19f6f6ff0cb1bcc1bf05f31a0fd1` (tag `journal-v2-a1-training-protocol-v1`), frozen files identical to the tag |
| Compute | Python 3.13.14, PyTorch 2.11.0+cpu, SB3 2.7.1, CUDA unavailable, PPO device `cpu`, 20 logical CPUs, 16 `SubprocVecEnv` workers |
| Seeds / budget | 201 and 203: 425,984 steps. 202: **429,376** steps (see incident) |
| Final model SHA-256 | 201 `24762607…ceaa3b`, 202 `32e51d5e…394453`, 203 `bc8ede4f…7ced` (full values in the registry) |

**Incident (seed 202).** At about 238,000 steps the run crashed with `PermissionError` while appending to `episodes.jsonl`
(repository is in a OneDrive folder). It was resumed through the protocol resume mechanism from `resume_latest.zip`
(200,000 steps, `resume_index=1`), with no hyper-parameter change. Because 200,000 is not rollout-aligned, the remaining 200,000
steps rounded up to 7 rollouts, so the final step count is 429,376 instead of 425,984. Episodes logged between 200k and 238k
by the interrupted attempt are discarded from the training report. All three seeds fail identically, so this does not affect the verdict.

## 2. Training health

No NaN or infinite parameters or actions, no corrupted model, no overwritten checkpoint, no guard violation, all terminal rewards exactly ±10, all
reward components within the plausible ranges, and all manifests match the model hashes.

| Seed | Episodes | Capture (first 100k → last 100k) | Exhaustion (last 100k) | Collision (last 100k) | Mean return (last 100k) |
|---|---|---|---|---|---|
| 201 | 5,833 | 0.2% → 0.0% | 98.6% | 1.0% | −9.55 |
| 202 | 5,873 | 0.1% → 0.0% | 98.8% | 1.1% | −9.47 |
| 203 | 5,826 | 0.0% → 0.0% | 97.4% | 2.1% | −9.42 |

Interceptions fall from about 15% to under 1% and collisions from 10% to 1–2%, while exhaustion rises from about 75% to about 98%: PPO learned to avoid
the *early* failures and never learned to capture. Deterministic validation (40 episodes, every 50k) shows 0 captures for all
seeds, with timeouts between 15% and 83% of the validation episodes after 50k steps. The canonical model is `final_model.zip`; `best_validation.zip` is diagnostic only.

## 3. Evaluation (seeds 140000–140099; 4 attackers × 5 defenders × 4 levels × 100 = 8,000 episodes)

Integrity: 8,000 unique episodes, 2,000 matched scenarios, identical initial attacker–asset range for all four attackers (max gap 0 m).

| Attacker | Capture | Interception | Obstacle collision | Endurance exhaustion | Timeout | Median path-excess |
|---|---|---|---|---|---|---|
| A0 | 10.4% | 88.9% | 0.0% | 0.6% | 0.0% | 0.93 |
| A1-201 | 0.0% | 0.5% | 0.0% | 46.0% | 53.5% | 3.28 |
| A1-202 | 0.0% | 2.0% | 1.1% | 50.2% | 46.8% | 3.46 |
| A1-203 | 0.0% | 2.8% | 4.1% | 35.1% | 58.0% | 2.69 |

Paired A1 − A0 capture differences (10,000-replicate paired bootstrap, seed 20260816, the repository convention): overall −10.4 pp for every seed;
by level C0 −5.6, C1 −5.6, C2 −11.6, C3 −18.8 pp; by defender greedy −8.75, kalman_greedy −8.75, PN −9.0, Lead −5.75, RMPC −19.75 pp. All 95% CIs exclude zero. See
`data/paired_a1_minus_a0_capture.csv`.

A1 results are nearly identical across defenders because the attacker barely interacts with the defender; the attacker's own loitering decides the episode.

## 4. Pre-registered gates

| Gate | Result | Evidence |
|---|---|---|
| C0 directness / endurance (exhaustion ≤5%, >2× path-excess ≤5%) | **FAIL** | C0 exhaustion 20.0 / 28.6 / 13.4%, C0 timeouts 79 / 70 / 85% (A0: 0.8%) |
| Mission advantage (≥ +5 pp vs A0 for some defender in ≥2 seeds) | **FAIL** | −5.75 to −19.75 pp for every defender |
| Seed generalization | **FAIL** | spread 0 pp, but every seed is 10.4 pp below A0 (limit 10 pp) |
| Obstacle safety (≤2% per seed and level) | **FAIL** | 202: C3 3.4%; 203: C2 6.2%, C3 10.2% (201: 0%) |
| Path efficiency (given success) | **FAIL** | vacuous: no captures exist to evaluate (undefined = fail by the frozen definition) |

## 5. Diagnosis

* **What the policies do.** The deterministic policies emit a nearly constant command about 145° from the asset direction with a strong upward component, climb
  to the 30 m ceiling (mean altitude 29.5 m in exhaustion episodes), and hover with a mean speed of about 0.7 m/s. They fly 80+ overshoot-like flybys, with zero orbits.
  A parked passive defender does not change this: 0 of 27 passive diagnostic episodes captured.
* **Why the frozen reward permits it.** Under γ = 0.99 and Δt = 0.5 s, a −10 terminal reached after 800–2000 steps is discounted to about zero, so the discounted
  return of timeouts (+1.28) and exhaustions (+0.6 to +0.8) is higher than that of the early interceptions or collisions (A0: −7.5). Capture was
  almost never sampled, so its +10 was not learned. The protocol document assumed the vehicle cannot hover and that endurance bounds the
  discount leak. **That assumption is false for vertical commands against the altitude ceiling**: the path accrues slowly, the budget
  is never exhausted within 2000 steps (53% of episodes end by `timeout`), and the undiscounted shaping leak reaches about +23 per timeout episode.
* **Obstacle collisions (104 re-simulated episodes; see `data/collision_forensics.json`).** All 104 are static-obstacle collisions (202: 22, 203: 82), all obstacles
  were in the attacker's perceived set (`map_local`), 103 followed a tactical command more than 45° off the asset direction (median 102°), 91 occurred with avoidance
  active (left 50, right 36, climb 5). Classification: 103 learned-policy obstacle mistakes, 1 unclear (A1-202 C3 PN seed 140047), 0 perception
  limitations. This confirms the pre-registered risk that the frozen avoidance layer does not sanitize off-axis tactical commands.
* **Endurance exhaustion (2,626 episodes; `data/evaluation_summary.json`).** 2,601 circling/loitering and 25 obstacle-induced detours; no tactical-evasion cases,
  because there is almost no defender contact.

## 6. Recommended next step (not applied here)

Create a new reward/protocol version (for example `a1_v2_reward_v2`) rather than editing this one. Candidate changes, to be pre-registered: remove the
discount leak (potential shaping terminated at episode end or a larger γ), make non-capture terminals independent of survival time, add an early no-progress
termination or an altitude-ceiling guard, and warm-start exploration from A0 so that capture is observed. None of these was tried or used to tune this campaign.
