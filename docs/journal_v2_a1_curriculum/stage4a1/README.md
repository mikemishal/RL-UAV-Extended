# Curriculum Stage 4A-1 — goal-conditioned pursuit via translated-target A0 imitation + Stage3-v2 behaviour retention

**Result: Stage 4A-1 FAILS — 0/3 seeds mastered; all three hit the pre-registered `RETENTION_FAILURE` stop after only 2 epochs; verdict C.** Supervised imitation of canonical A0 on a RANDOMLY
TRANSLATED stationary asset drove the asset-position input-sensitivity ratio from **~2.2% (PRE, every Stage3-v2 parent) to 69–114% (POST)** — far above the ×2 representation-gate requirement —
confirming the Stage 4A hypothesis that the parents simply never learned to use the asset-position observation slots. But the resulting CLOSED-LOOP policy did not reliably capture: held-out
translated capture is only **56.5 / 60.5 / 55.0 %** (far below the 99% mastery gate, though a *huge* improvement over the untrained parent's **4.5 / 6.0 / 4.0 %** on the same scenarios), and all
three seeds caused a catastrophic, immediate regression on the UNCHANGED retention anchors (even Anchor 1, the exact fixed-origin task the parents already solved at ~100%, collapsed to 38–59%
within 2 epochs), tripping the frozen `RETENTION_FAILURE` rule. The mechanism is a textbook behaviour-cloning covariate-shift failure (§5), not a representation-capacity failure.

## 1. Why this stage: isolating the mechanism found in Stage 4A
Stage 4A's diagnostic (`docs/journal_v2_a1_curriculum/stage4a/`) showed the Stage3-v2 parents' first-layer weight columns for the asset-position slots sit at the untrained orthogonal-init value
(because the asset was always at the origin in Stages 1–3), with direction-sensitivity to the asset position only ~2% of the sensitivity to the attacker's own position; 393k steps of PPO only
raised this to ~10%. **Pre-registered hypothesis:** *randomizing the stationary goal location during expert imitation should produce translation-aware pursuit while preserving previously learned
defender-evasion behavior.* This is a representation/curriculum intervention (randomize the training distribution over goal LOCATION, not a new architecture), tested by supervised imitation —
**no `PPO.learn()` is called anywhere in this stage.**

## 2. Environment and translation protocol
* Mechanism: the unchanged Stage-1-v2 `curriculum_basic_pursuit_no_defender_interaction` env (`journal_v2_a1_cur1v2.Stage1V2Env`): defender structurally present, parked, masked (observation slots
  9–15 exactly 0), interception impossible (`intercept_radius = 1e-6`); canonical perimeter spawn, altitude U(10,30), heading offset U(0,90°); C0; `a1_v2_obs_v1` (37-D), `a1_cur1_action_v1`,
  `a1_v2_reward_v2` (unused by BC, kept for evaluation compatibility). The only config differences from canonical are `v_s = 0` (asset stationary *within* an episode — never moved mid-episode,
  per the task constraint), `intercept_radius` and `defender_standby_until_detection`.
* **Translation:** both the attacker's canonical edge-spawn position and the asset's canonical origin are shifted by the SAME horizontal vector δ, drawn from a dedicated stream
  (`0xC4A2`, independent of spawn/soldier/enemy/sensor/obstacle streams) — exactly preserving the relative pursuit geometry and the endurance budget (computed from the initial 3-D distance,
  verified translation-invariant to < 1e-4 m). A safety margin of **5 m** (10% of the 50 m half-arena) keeps both the attacker and the translated asset off the wall. The per-scenario feasible
  box is solved exactly from the edge-spawn point (no clipping; infeasible requests raise `TranslationInfeasible` and a different scenario is tried). Because the asset starts at the origin, its
  translated position **equals δ**. Final asset-position bounds realized: x ∈ [−45.0, 45.0] m, y ∈ [−45.0, 45.0] m (median ≈ (0.3, 1.8), expected for a symmetric design); all four quadrants
  (1,475–1,551 episodes each of 6,000), 1,570 near-diagonal episodes, 3.1% central (within 10 m of the origin) and 54.3% outer (≥ 30 m) — the target coverage (quadrants, central, ±x, ±y,
  near-diagonal) from §5 of the brief was met.

## 3. Datasets
* **Expert (goal) dataset:** 6,000 translated A0 episodes (`172000–177999`), **67,245** state-action pairs (target 60,000–80,000 ✓), 9–20 pairs/episode (mean 11.2); 100% captured (A0 always
  solves its own stationary task); episode-level 80/20 split (4,800/1,200 episodes, 53,835/13,410 pairs, no leakage). Dataset SHA-256 `c5bcf33109969749…` (`dataset_manifest.json`).
* **Stage3-v2 retention (rehearsal) dataset, per seed:** driven by the SEED's OWN accepted Stage3-v2 parent (not A0) on reserved scenarios `191200–192199`; ~999 episodes, **11,003–11,244**
  states per seed (target 10,000–15,000 ✓). State classes (not_visible / newly_visible / active_evasion / post_pass_pursuit), by detection/closest-step, never by future outcome: 31–32% /
  9% / 14% / 45–46%. Labels are the parent's own deterministic mean action (verified bit-identical when recomputed). Episode-level 80/20 split.

## 4. PRE representation baseline (parent, before any Stage4A-1 training)
| | 201 | 202 | 203 |
|---|---:|---:|---:|
| Asset/attacker sensitivity ratio | 2.22% | 2.52% | 2.14% |
| First-layer asset-column norm | ≈ √2 (untrained) | ≈ √2 | ≈ √2 |
| Held-out translated capture | 4.5% | 6.0% | 4.0% |
| Held-out translated exhaustion | 95.5% | 94.0% | 96.0% |
| Held-out expert cosine (closed-loop, mean) | 0.39 | 0.40 | 0.43 |
| Generalization-region capture (min–max) | 0–25% | 0–30% | 0–42.5% |

Matched canonical A0 on the identical 200 translated held-out scenarios: **100% capture**, median path-excess 1.01, no pathology — confirming the task is trivial for a policy that actually uses
the asset position, and the parents' ~5% capture (worse than random in most quadrants) is a genuine representation gap, not a hard task.

## 5. Training, mastery and the covariate-shift failure mechanism
All three seeds ran the combined-loss fine-tuning (`L = L_goal + 0.25·L_retain`, actor-only, `log_std` frozen at −2.0, critic frozen — all verified bit-exact in tests) and hit
`RETENTION_FAILURE` after exactly 2 epochs (two consecutive epochs below an anchor minimum):

| Seed | Epoch | Goal val cosine (open-loop) | Retain val cosine (open-loop) | Translated capture (40-ep check) | Anchor 1 | Anchor 2 | Anchor 3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 201 | 1 | 0.968 | 0.977 | 7.5% | 0.38 | 0.29 | 0.07 |
| 201 | 2 | 0.988 | 0.990 | 52.5% | 0.59 | 0.44 | 0.43 |
| 202 | 1 | 0.970 | 0.977 | 35.0% | 0.14 | 0.13 | 0.14 |
| 202 | 2 | 0.984 | 0.988 | 62.5% | 0.51 | 0.31 | 0.25 |
| 203 | 1 | 0.971 | 0.978 | 45.0% | 0.17 | 0.12 | 0.18 |
| 203 | 2 | 0.985 | 0.988 | 47.5% | 0.41 | 0.23 | 0.38 |

Required minimums: Anchor 1 ≥ 0.98 ⁄ 0.99, Anchor 2 ≥ 0.80, Anchor 3 ≥ 0.75 — **every anchor fails by 40–90 pp at every epoch**, including Anchor 1, the UNCHANGED fixed-origin task the parents
already solve at ~1.00. This is the key diagnostic: **the open-loop imitation metrics (cosine against A0's label at states A0 itself visited) are excellent (0.97–0.99) throughout, while
closed-loop performance (the policy driving itself) collapses — classic behaviour-cloning covariate shift** (Ross & Bagnell): the translated-target task has a vastly larger state space than the
fixed-origin task ever exercised, the training set contains only STATES ALONG A0's OWN near-optimal trajectories (no corrective relabeling / DAgger round, unlike the original Stage-1-v2 BC
protocol, which did include one), and once the learned policy's own small heading errors push it even slightly off that manifold, it has never been taught the right correction — errors compound
over a long rollout. Direct evidence (seed 201, scenario `178002`): the closed-loop trajectory timed out at **2,000 steps** with a **negative** mean cosine to A0 (−0.18), i.e. net average
*not even heading toward the target*, despite the SAME checkpoint scoring near-1.0 cosine on the static held-out dataset.

## 6. Final held-out evaluation (200 translated scenarios, accepted = last completed epoch before the stop)
| | Stage4A-1 201 | 202 | 203 | Stage3-v2 parent 201 | 202 | 203 | A0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Capture | 56.5% | 60.5% | 55.0% | 4.5% | 6.0% | 4.0% | 100% |
| Exhaustion | 41.0% | 39.5% | 45.0% | 95.5% | 94.0% | 96.0% | 0 |
| Timeout | 2.5% | 0 | 0 | 0 | 0 | 0 | 0 |
| Median path-excess | 1.74 | 3.35 | 2.46 | 7.07 | 7.14 | 6.75 | 1.01 |
| >2× / >5× path-excess | 48 / 20.5% | 60 / 27% | 51 / 24% | 99 / 86% | 98 / 85.5% | 97 / 84.5% | 0 / 0 |
| Orbit episodes | 1 | 0 | 1 | 0 | 0 | 0 | 0 |
| Closed-loop expert cosine (mean) | 0.030 | 0.441 | 0.479 | 0.391 | 0.400 | 0.430 | 1.0 (by definition) |
| Anchor 1 / 2 / 3 (accepted epoch) | 0.59 / 0.44 / 0.43 | 0.51 / 0.31 / 0.25 | 0.41 / 0.23 / 0.38 | 1.00 / ~0.9+ / ~0.9+ (unaffected; not re-trained) | | | |

The trained model is a **huge** improvement over its own parent (+50–56.5 pp capture, −8× path-excess, 2× fewer >5× episodes), consistent with it genuinely using the translated asset position
(it visibly orbits the correct, translated target in the trajectory figure — see §8), but still fails both the 99%-capture mastery gate and the retention gates.

## 7. Representation gate: PRE vs POST asset-input sensitivity
| Seed | PRE ratio | POST ratio | Gain | Sensitivity gate (≥ 2×) | Min generalization-region capture | Region gate (≥ 95%) |
|---|---:|---:|---:|:--:|---:|:--:|
| 201 | 2.22% | **114.4%** | 51.6× | pass | 37.5% | fail |
| 202 | 2.52% | **69.3%** | 27.5× | pass | 30.0% | fail |
| 203 | 2.14% | **81.9%** | 38.2× | pass | 35.0% | fail |

The sensitivity gate is passed overwhelmingly (27–52× the required 2× gain; seed 201 even exceeds 100%, i.e. the executed direction is now *more* sensitive to the asset than to the attacker's
own position — an overcorrection consistent with the unstable, oscillatory closed-loop behaviour). The region gate fails everywhere (worst region 30–37.5% capture, far below 95%); capture varies
25–85% across the 8 quadrant×magnitude strata with **no systematic quadrant or inner/outer bias** (fig. 5) — the failure is general closed-loop instability, not a specific blind region.

## 8. Qualitative trajectories (fig. 6, seed 202, four translated targets)
In all four cases the Stage3-v2 parent (orange) either ignores the translated asset entirely (flying off toward an unrelated point) or only coincidentally brushes past it, while the fine-tuned
Stage4A-1 policy (green) clearly turns toward and **orbits** the correct, translated asset location — visually confirming the sensitivity result — but the orbit does not reliably collapse to a
capture. This is the geometric signature of the covariate-shift failure: the policy has learned "head toward the target" from on-trajectory data, but not "null the command out precisely as range
shrinks to zero," because the expert's own near-optimal approach never visited the off-axis, over/under-shoot states a slightly-imperfect learner generates near the target.

## 9. Verdict
Overall rule (≥ 2/3 master, representation gate, retain all three anchors, no shared pathology): **not met — 0/3 mastered** (all three stopped at `RETENTION_FAILURE`). The "needs manual review"
rule (≥ 1 mastered, or ≥ 2 seeds with capture ≥ 0.80 and sensitivity gain ≥ 1.5×) is also not met (best capture 60.5%, short of 0.80). **Verdict C.** Per the stage rule: **STOP, do not proceed to
moving-asset imitation.** No review IDs, `stage4a2_handoff.json`, or tag were created (required only for a pass or verdict B).

## 10. What this result means for the curriculum
The pre-registered hypothesis is **supported**: the Stage-4 failure was indeed caused in large part by the parents' near-total disuse of the asset-position input, and randomizing the goal
location during imitation does teach strong, correctly-signed dependence on it (§7). What failed is the specific *learning method*: one-shot, open-loop behaviour cloning on expert-only states,
without any closed-loop correction, cannot convert that representation change into a reliable closed-loop policy — and the resulting instability is severe enough to wreck the (unrelated)
retention anchors within 2 epochs, triggering the pre-registered stop before the goal task itself could mature. This points to a concrete next learning step (not started here, a new registered
version): **one or more rounds of DAgger-style closed-loop relabeling** (run the current policy, label the states IT visits with A0's action at that state, and fine-tune again) — exactly the
mechanism the original Stage-1-v2 BC protocol used and that Stage 4A-1, as specified, omitted — before any moving-asset imitation is attempted.

## 11. Limitations
1. Training stopped after only 2 epochs per seed (by the pre-registered retention rule); whether more epochs (with the SAME protocol) would recover or worsen the anchors is unknown and was not
   tested (the frozen stop rule was not relaxed).
2. The per-epoch translated-capture check used only 40 of the 200 held-out scenarios for speed; the final 200-episode evaluation is the authoritative number.
3. No closed-loop relabeling / DAgger round was used (unlike the original Stage-1-v2 BC); §10 identifies this as the probable fix, not yet tested.
4. Parents are provisional curriculum checkpoints (no manual-review acceptance of Stage3-v2 recorded).
5. Three seeds; the retention collapse was consistent and severe across all three (not an outlier), which is itself informative.

## 12. Reproduce / verify
```
python -m experiments.journal_v2_a1_cur4a1_run all            # stages 1-8 (restartable): verification, protocol, env checks, dataset, PRE sensitivity, BC fine-tuning x3
python -m experiments.journal_v2_a1_cur4a1_final all           # stages 9-11: held-out eval, retention eval, POST sensitivity + generalization, decision, figures 1-6
pytest tests/test_journal_v2_a1_cur4a1.py -q
```
State: `task_state.json` (15 stages, atomic, hash-verified). Results (untracked): `results/journal_v2_corrected_semantics/a1_curriculum/stage4a1/`. Scratch:
`C:\Users\Public\journal_v2_a1_curriculum\stage4a1\`. Protocol SHA-256: `4a4b91aa92d5bc970e52f7da38227b23c710e85c39be31c084ec3a0127ca9a75` (`protocol.json`).
