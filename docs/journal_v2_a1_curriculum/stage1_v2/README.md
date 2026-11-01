# Curriculum Stage 1 V2 — expert initialization of basic pursuit (A0 → A1 by behaviour cloning)

**Result: PASS (3/3 seeds).** These are **curriculum initialization checkpoints, not final intelligent-attacker models** and not canonical A1.

Pure-PPO Stage 1 (`../stage1/`, preserved as a negative curriculum ablation) failed 0/3. Stage 1 V2 instead uses canonical scripted attacker **A0 as an expert teacher for the single basic skill "pursue and capture a stationary asset"**. No `PPO.learn()` is called anywhere in this stage (guarded by a unit test).

## Pre-registered hypothesis

> Basic pursuit can be efficiently transferred from the deterministic A0 expert to A1 through imitation learning, allowing subsequent reinforcement learning to focus on genuinely adversarial behaviour rather than rediscovering elementary goal-directed navigation.

## Clean Stage-1 environment — `curriculum_basic_pursuit_no_defender_interaction`

Curriculum-only wrapper (`experiments/journal_v2_a1_cur1v2.py`); canonical environment behaviour is not changed globally.

| Item | Definition |
|---|---|
| Scenario | C0, canonical V2 semantics (swept capture, endurance k=5), Stage-1-v1 randomized geometry (perimeter spawn 50–75 m, altitude 10–30 m, initial heading ≤ 90° off the line of sight) |
| Asset | `v_s = 0` (stationary) |
| Defender | structurally present, **stationary** at the parked corner `(−48,−48,0)`; **interception disabled** (`intercept_radius = 1e-6`, so defender contact can never terminate an episode); **tactical observation masked as unavailable** (`detection_range = −1` ⇒ observation indices 9–15 are exactly 0 in every state) |
| Observation | `a1_v2_obs_v1` (37-D, all defender/obstacle slots kept, so the network is compatible with later stages) |
| Action | `a1_cur1_action_v1` — continuous `Box(−1,1,(3,))` desired flight direction; canonical constrained dynamics decide acceleration/speed/turn/climb; **no A0 fallback at execution time** (a zero action holds the current heading) |
| Reward | `a1_v2_reward_v2` (unchanged; unused by BC) |

Why: Stage-1-v1's parked defender entered the 25 m information gate in ≈ 7.5% of episodes and intercepted 1.4–3.2% of stochastic training episodes, contaminating the basic pursuit task. Verified by tests: defender slots are zero even with the attacker teleported 0.7 m from the defender; the defender never moves; no episode ends in an interception outcome.

## Expert labels

For every state, the label is the A0 desired-direction command computed from the *same attacker/asset state* (`unit(asset − attacker)`, exactly what `SoldierEnv._move_enemy` computes in C0). It uses no future information, privileged trajectory or shortcut unavailable to A1. A policy that outputs this label through the real override path reproduces A0 **bit-identically** (max position deviation 0.0), which validates the extraction.

## Expert dataset (frozen)

* Namespace: scenario seeds **160000–164999** (5,000 episodes), disjoint from training RNG, validation 146000–146199, final evaluation ranges and all earlier diagnostic ranges (`namespace_overlaps()` is tested).
* **56,134 state–action pairs**, SHA-256 `4de39a6021ea03123d63439a013d25063dc44fce32d64564f319df9bcea56c62` (10 hashed shards, each committed atomically by its sidecar; verified shards are never regenerated).
* Episode-level split (no leakage, checked): 4,000 train episodes / 44,916 pairs and 1,000 validation episodes / 11,218 pairs.
* Coverage (`dataset_manifest.json`): bearing quadrants 1256/1284/1267/1193 episodes; initial heading offset 0.04°–90° (quartiles 1212/1230/1289/1269); initial altitude 10.0–30.0 m; initial range 51.1–76.3 m; 7,144 late-approach pairs (≤ 10 m, 12.7%); 9–16 pairs per episode; label vertical component −1.0…−0.06; all 5,000 A0 episodes captured.

## Behaviour cloning

* Policy: SB3 `MlpPolicy`, `[64,64]`, tanh, 37-D input, 3-D Gaussian action — the exact architecture intended for later PPO (a PPO object is built only to hold the architecture; `learn()` is never called).
* Loss: `MSE(μ(s), a_A0) + (1 − cos(μ(s), a_A0))` on the policy mean; trained parameters are only the actor trunk and action head. **The value network and `log_std` are untouched by BC** (asserted by `value_network_unchanged`); the critic stays randomly initialized for PPO to learn.
* Adam, lr 1e-3, batch 256, 100 epochs, per-epoch atomic restartable checkpoint, selection = lowest validation loss.
* **`log_std = −2.0` after BC** (initial action std = e⁻² = 0.1353), pre-registered and not tuned.
* Independent initializations: seeds **201, 202, 203** on the same frozen dataset; never initialized from the failed PPO checkpoints.

### Round-0 supervised metrics (held-out episodes of the expert dataset)

| Seed | val cos mean | val cos median | val cos min | val median angle | val MSE | model SHA-256 (round 0) |
|---|---|---|---|---|---|---|
| 201 | 0.99993 | 0.99997 | 0.9875 | 0.440° | 8.1e-5 | `a0645632d2074efc3a2b5896c90dad58b448cc1c05354cf11627a002c76996ea` |
| 202 | 0.99995 | 0.99998 | 0.9864 | 0.354° | 6.8e-5 | `76b64c81974706470463172a1c81d60ba7705ecfff44c7518db2e3f43771f9b4` |
| 203 | 0.99994 | 0.99998 | 0.9860 | 0.358° | 7.7e-5 | `9e91b12257954c258f9003710f7cc421a111676e8f3de511473d47585dd155d0` |

## Fixed one-round expert relabeling (DAgger-style, exactly one round)

For each round-0 policy: run it **deterministically** on 1,000 new curriculum scenarios (seeds 170000–170999, disjoint from everything above), record the states **A1 actually visited** (≤ 100 per episode; ≈ 11.1 per episode in practice), label them with A0, append to that seed's dataset (episode-level split again) and fine-tune once (Adam lr 5e-4, 50 epochs). No adaptive repetition.

| Seed | visited states = added expert labels | expert-val cos (pre → post) | relabel-val cos (pre → post) | expert-val MSE (pre → post) |
|---|---|---|---|---|
| 201 | 11,117 | 0.999935 → 0.999966 | 0.999924 → 0.999964 | 8.1e-5 → 4.6e-5 |
| 202 | 11,114 | 0.999949 → 0.999968 | 0.999947 → 0.999964 | 6.8e-5 → 4.1e-5 |
| 203 | 11,114 | 0.999942 → 0.999966 | 0.999945 → 0.999968 | 7.7e-5 → 4.3e-5 |

All 1,000 round-0 relabel rollouts per seed were already captures (`outcomes_round0_policy: soldier_caught = 1000`): the BC policy was already on the expert distribution, so covariate shift was negligible and relabeling gave a modest additional fidelity gain (MSE −40–45%) rather than a behavioural change. Honest reading: the expert-relabeling round is not what makes this stage pass; round-0 policies pass the same held-out gate (below).

## Held-out evaluation — 200 episodes per seed (seeds 146000–146199, never used for BC or relabeling; deterministic policy means)

| | Capture | Exhaustion | Timeout | Median path-excess | >2× | >5× | Orbit | Non-finite | mean/median cosine to A0 | min cosine | mean duration |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **A0 (same 200 scenarios)** | 100% | 0 | 0 | 1.0235 | 0 | 0 | 0 | 0 | — | — | 5.6475 s |
| **A1-BC1-201 (final)** | 100% | 0 | 0 | 1.0235 | 0 | 0 | 0 | 0 | 0.9999 / 1.0000 | 0.9899 | 5.6475 s |
| **A1-BC1-202 (final)** | 100% | 0 | 0 | 1.0236 | 0 | 0 | 0 | 0 | 1.0000 / 1.0000 | 0.9973 | 5.6450 s |
| **A1-BC1-203 (final)** | 100% | 0 | 0 | 1.0235 | 0 | 0 | 0 | 0 | 1.0000 / 1.0000 | 0.9952 | 5.6450 s |

Mastery gate (capture ≥ 98%, exhaustion ≤ 1%, timeout ≤ 1%, median path-excess ≤ 1.20×, >2× ≤ 2%, >5× = 0, orbit 0, non-finite 0): **all three seeds pass**, and so did all three round-0 policies. Mean path length 61.49 m vs A0 61.49 m; one episode per policy (0.5%) exceeds 1.3× path-excess, the same count as A0 (max 1.337× for A0 and 1.335–1.338× for A1). Cosines are measured per executed step against the A0 label at the state the *student* visited. **This is an imitation-fidelity comparison, not a claim that A1 improves on A0.**

Overall Stage-1-v2 verdict: **PASS — 3/3 seeds, no shared structural pathology** (`stage1_v2_decision.json`).

## Comparison with pure PPO from scratch (negative curriculum ablation)

| | Pure PPO from scratch (Stage 1 v1) | A0 expert initialization (this stage) |
|---|---|---|
| Learning signal | reward from 262,144 environment steps (3 seeds × 8 rollouts) | 56,134 A0 labels + ≈ 11.1k relabeled visited states; ≈ 67k simulator steps in total, no reward |
| Held-out capture (200 ep) | 33.5% / 0% / 2.5% | **100% / 100% / 100%** (A0: 100%) |
| Median path-excess | n/a (mostly time-out/exhaustion) | 1.0235× (A0: 1.0235×) |
| Direction vs line of sight | mean cosine ≈ 0.07 / 0 / 0.01 | ≈ 1.0 |
| Action std | ≈ 1.02 (near-random sampled directions) | 0.135 (pre-registered `log_std = −2`) |

## Limitations (read before citing)

1. The task is the easiest stage (stationary asset, no obstacles, no active defender); matching A0 is the ceiling here, and A0's label is a trivial closed-form function of the state, so a near-perfect fit is expected and says nothing about adversarial ability.
2. The three seeds differ only in network initialization and minibatch order (same frozen dataset); this isolates neural variability but not data variability.
3. The relabeling round was essentially unnecessary on this task (round 0 already passes); the DAgger machinery is retained for later, harder stages.
4. Defender interaction is removed by construction; passing says nothing about evasion. The value function is untrained.
5. Checkpoints are initialization-only; they are not registered as canonical A1.

## Restart-safety and provenance

`task_state.json` tracks 12 stages with input SHA, artifacts, SHA-256 and completion time; dataset shards, per-epoch BC checkpoints (hash-verified manifest committed last) and all summaries are written temp → rename under file locks, so an interrupted run resumes from the last verified shard/epoch. Scratch (per-epoch checkpoints) lives outside OneDrive; final models and summaries are under `results/journal_v2_corrected_semantics/a1_curriculum/stage1_v2/` (not tracked by git; hashes are recorded here and in `stage2_handoff.json`).

| Seed | final model (`bc_final_model.zip`) SHA-256 |
|---|---|
| 201 | `4bc446afebbb95f308234399ec22a2a6f84b39338f85a0f109384a338eb2aee4` |
| 202 | `18da296243f10057868939c6b3d3bc07ff2de13e622241b9df10dc55ea30ab65` |
| 203 | `a3e34511d60170ea9ce329ece56f4e778fd01fdc6e22cb918df379675dab42c3` |

## Files

`protocol.json` (frozen protocol + SHA), `dataset_manifest.json` (hash + coverage), `final_evaluation.json` (A0 + round-0 + final, 200 episodes per seed), `stage1_v2_decision.json`, `stage2_handoff.json`, `recordings_shortlist.json` (visualizer shortlist), `manuscript_notes.md`, `ablation_record_v6.md`, `repository_verification.json`, `checkpoint_integrity.json`, `regression_summary.json`, `unit_test_result.json`, `task_state.json`.

## Visualizer

Registered (pin-verified, SHA-checked against `stage2_handoff.json`): `A1-BC1-201/202/203` — C0, V2 semantics, passive defender only. Recordings listed in `recordings_shortlist.json`. Example: `?semantics=V2&clutter=C0&defender=passive&attacker=A1-BC1-203&seed=146009`.

## Next stage (not started)

Stage 2: stationary asset, C0, **active** defender; PPO initialized from these checkpoints to learn evasion while preserving pursuit competence.
