# Manuscript notes — hierarchical obstacle-vs-defender CBF priority (Prompt 5)

Material for Methods/Results, documenting a design correction and its evaluation. Companion to `README.md`.

## Methods: the design correction

A shared CBF slack variable allowed tactical defender avoidance to trade against physical obstacle safety under
severe joint conflicts (journal_cbf_attacker, defender_cbf / Prompt 4): when the hard joint QP (obstacle +
defender + velocity bounds, all simultaneously hard) was infeasible, a single soft-slack relaxation applied
EQUALLY to both constraint classes, allowing the optimizer to "spend" relaxation on whichever class made the
overall objective cheapest to satisfy -- including the obstacle-safety constraint. This produced 3/200 (1.5%)
obstacle collisions in a deliberately adversarial 200-case joint-constraint stress set.

We correct this with an explicit priority hierarchy, reflecting the physical reality that an obstacle collision is
an irreversible safety failure while a defender interception is merely a reversible tactical-mission failure:

1. **Priority 1 -- physical static-obstacle safety.** Enforced as a hard constraint at every stage; never relaxed
   to help the defender, under any circumstance.
2. **Priority 2 -- vehicle physical velocity limits.** Hard, unchanged from all prior stages.
3. **Priority 3 -- defender avoidance.** Soft: permitted to relax via a dedicated, non-negative slack variable
   `delta_D`, penalized quadratically (`1e4 * delta_D^2`) in the objective, exactly as the shared slack was
   penalized before.

The resulting QP cascade: (1) is obstacle+bounds-only feasibility achievable at all, with the defender entirely
ignored -- a genuine physical-crisis check independent of any tactical consideration; (2) if yes, is the FULL
hard solution (obstacle+defender+bounds, no slack at all) achievable -- the common, no-conflict case; (3) if not,
solve the hierarchical soft QP, obstacle and bounds held exactly hard, only the defender constraint relaxed. If
even step (1) fails, the defender is dropped entirely and the existing (frozen, pre-defender-CBF) obstacle-only
emergency chain is invoked, with this case separately classified and reported (`obstacle_safety_infeasible`) so it
is never silently conflated with a defender-avoidance trade-off.

## Results

### Correction of the identified design flaw
Reproducing the exact scenario that originally exposed the shared-slack flaw (seed 630069: attacker simultaneously
near a static obstacle and a closing defender, both constraints severely violated), the hierarchical controller
resolves the episode safely (asset capture, zero obstacle collisions), with the defender constraint observed
absorbing the ENTIRE conflict via its own dedicated slack (`delta_D = 13.29` at the critical step) while the
obstacle constraint's own residual remained at machine precision throughout. This directly confirms the priority
restructuring functions exactly as designed for a genuine priority-conflict scenario.

### A newly-discovered, distinct failure mode
Two of the three originally-collision-causing seeds (630123, 630203) were found to STILL collide under the
hierarchical controller, despite the obstacle constraint remaining verifiably hard (residual <=1e-4) at every
solve. Detailed numerical tracing revealed these are NOT instances of the priority-conflict mechanism the
hierarchy was designed to fix: the colliding obstacle hazard already carries a deeply negative barrier value
(`h ~ -2.3`) at the moment it first enters the fixed 20 m active-constraint range (because the attacker's closing
speed when the obstacle activates already exceeds what the stopping-margin formula's one-substep
(`dt_sub=0.1 s`) reaction-time assumption can absorb) -- a state the formula itself implicitly assumes should not
occur for a well-conditioned trajectory, but can arise from the discrete, range-gated nature of hazard activation
for a fast-closing attacker. In this state, the hierarchical QP (constrained only in velocity space, with no
direct awareness of the vehicle's separate acceleration-rate limit) selects a velocity target satisfying the
obstacle inequality exactly but pointed substantially off the critical escape axis -- driven by the SAME quadratic
objective's attempt to also minimize the defender's slack term -- which, after the executor's SEPARATE
acceleration-limited integration (0.6 m/s per 0.1 s substep, independent of the QP), delivers LESS course
correction along the axis that matters than a defender-free solve of the identical obstacle geometry would. We
confirmed this precisely: re-running the identical starting kinematic state with the defender constraint disabled
entirely survives safely, isolating the proximate cause to the mere presence of an additional (even non-binding at
the solution) hard constraint in the QP, not to any relaxation of obstacle safety. A warm-start experiment (seeding
the hard-with-defender solve from the obstacle-only optimum) produced byte-identical results, ruling out an
SLSQP local-optimum artifact as the explanation.

### Monte Carlo confirmation at scale
The full 200-case joint-constraint stress set confirms this is not an isolated artifact: 2/200 (1.0%) obstacle
collisions persist under the hierarchical controller (down from 3/200 under the shared-slack design), and the two
collisions are exactly the two individually-traced seeds above. The acceptance gate's explicit zero-collision
requirement for this stress set is therefore not met.

### Tactical effect is fully preserved
Against the matched 1,000-episode Greedy campaign (common random numbers with Prompt 4), capture (4.9%) and
interception (95.1%) rates are statistically IDENTICAL between the shared-slack and hierarchical controllers,
with zero obstacle collisions in both. Against the four-defender generalization diagnostic (Kalman-Greedy, PN,
Lead, RMPC; 800 additional episodes), every outcome metric is likewise statistically identical between the two
controller versions, including the large, previously-reported interception reductions against PN (-33 percentage
points) and RMPC (-38 percentage points). This directly demonstrates that enforcing obstacle safety as a strict
hard priority neither helps nor harms the defender-avoidance mechanism's tactical effectiveness in the regime
studied -- the two concerns are, in practice, largely orthogonal except in the specific severe-conflict and
acceleration-reachability edge cases documented above.

## Failure cases (honest reporting)

1. 2/200 (1.0%) deliberately-adversarial joint stress scenarios still produce a genuine obstacle collision, traced
   to a gap in the frozen CBF formulation's velocity-space-only representation (no direct acceleration-reachability
   term), exposed -- but not caused -- by the presence of a defender hard-constraint.
2. One additional (otherwise-benign) emergency-brake event appeared in the 1,000-episode primary Greedy campaign
   that did not occur under the shared-slack formulation (a single step, episode still ended in asset capture).

## Limitations and recommended next steps

The root limitation is architectural: the CBF-QP's hard velocity constraints encode the stopping-margin BARRIER
(an algebraic function of distance and closing speed) but never directly encode the vehicle's own acceleration
limit as a constraint on the chosen velocity itself -- that limit is applied entirely separately, downstream, by
the executor. Candidate corrections for a follow-up design review (not authorized or attempted in this stage):
an explicit acceleration-reachability constraint added directly to the QP; a collision-cone CBF reformulation;
or restricting the secondary (post-priority-1) objective to anchor toward the obstacle-only-optimal velocity
rather than the original pursuit-nominal velocity when a conflict is detected, so that axes irrelevant to the
active obstacle hazard are not perturbed by the defender's competing pull. All are explicitly deferred.
