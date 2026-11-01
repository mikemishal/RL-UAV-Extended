# Manuscript notes — curriculum Stage 1 V2 (expert initialization)

## Claim supported
A deterministic scripted expert (A0) transfers the elementary skill *goal-directed pursuit of a stationary asset* to a PPO-compatible neural policy by supervised imitation using ≈ 67k simulator steps, with **no reinforcement learning**. All three independently initialized students (seeds 201/202/203) capture 200/200 held-out scenarios with A0-equivalent path efficiency, whereas pure PPO from scratch with 262,144 environment steps captured 33.5% / 0% / 2.5%.

## Narrative for the dissertation

1. **Why pure PPO Stage 1 failed (preserved negative result).** Three seeds × 8 rollouts: 0/3 mastered. The learned mean direction was essentially uncorrelated with the line of sight (cosine ≈ 0.07 / 0 / 0.01) and the exploration standard deviation stayed ≈ 1.02 on a unit-direction action, so sampled commands were nearly random. 96–98% of training episodes ended by endurance exhaustion after ≈ 73 steps. The parked defender also contaminated the task (visible in ≈ 7.5% of episodes, intercepting 1.4–3.2%).
2. **Motivation.** The failure is a *credit-assignment/exploration* failure on a task whose correct answer is known in closed form. A0 is already a competent pursuit controller, so it should be used as a teacher for that skill and the reinforcement signal reserved for what A0 cannot do: adversarial adaptation.
3. **Decomposition.** A0 teaches mission pursuit; PPO (later stages) learns evasion/adaptation. The Stage-1 environment therefore removes defender interaction (`curriculum_basic_pursuit_no_defender_interaction`) but keeps the 37-D observation and 3-D action unchanged so that weights transfer directly.
4. **BC protocol.** 5,000 A0 episodes (seeds 160000–164999), 56,134 state–action pairs, episode-level 80/20 split; policy `MlpPolicy [64,64] tanh`; loss `MSE + (1 − cos)`; actor trunk/head only, value network and `log_std` untouched; `log_std = −2` (pre-registered, e⁻² = 0.135) after BC; three seeds on one frozen dataset.
5. **Expert relabeling (one fixed DAgger-style round).** Deterministic student rollouts on 1,000 new scenarios (170000–170999), A0 labels at student-visited states (≈ 11.1k per seed), single fine-tuning pass. In this stage every relabel rollout already captured, so the round only sharpened fidelity (MSE −40–45%); the round-0 policies pass the same gate. Report it as a *robustness safeguard*, not as the cause of success.
6. **Quantitative fidelity.** Validation cosine to the expert: mean 0.99997 (post-relabel), median angle 0.27–0.30°. On the 200 held-out closed-loop episodes the executed-direction cosine to A0 at student-visited states has mean 0.9999–1.0000, minimum 0.990–0.997. Capture 100%, median path-excess 1.0235× (identical to A0), no exhaustion/timeouts/orbits.
7. **Sample efficiency.** Pure PPO: 262,144 env steps, reward-driven, failed. BC: ≈ 56k + 11k labeled states (≈ 67k simulator steps incl. relabel rollouts), supervised, 100% capture. Dataset generation took ≈ 4 s per 500-episode shard (10 shards in parallel) and the full BC + relabel + 200-episode evaluation pipeline completed in roughly ten minutes on CPU.
8. **Positioning.** This is imitation-fidelity evidence, not evidence of an improved attacker. The expected ceiling is A0 itself.

## Suggested table row (A1 ablation)

| # | Design | Outcome |
|---|---|---|
| 8 | Pure PPO from scratch, curriculum Stage 1 | **failed** (0/3, capture 33.5% / 0% / 2.5%) |
| 9 | A0 expert initialization (BC + one relabeling round) | **passes 3/3** (capture 100%, path-excess 1.0235× = A0) — initialization checkpoint only |

## Limitations to state explicitly
* Easiest curriculum stage; A0's label is a closed-form function of the state, so near-perfect cloning is expected.
* One dataset, three network initializations: variability of data generation is not sampled.
* The relabeling round was not needed here; its value is expected at later stages where student states diverge from A0's.
* Defender interaction is removed by construction; nothing here speaks to evasion, safety or clutter.
* The value head is randomly initialized; the initial PPO critic is uninformed.
* Checkpoints are curriculum initializations, not canonical A1; the `journal-v2-a1-curriculum-bc-stage1-v1` tag marks reproducibility only.

## Reproduction
`python -m experiments.journal_v2_a1_cur1v2_run all` (restart-safe; frozen protocol SHA in `protocol.json`; dataset SHA in `dataset_manifest.json`; model hashes in `stage2_handoff.json`).
