# Manuscript notes — acceleration-reachable CBF velocity-constrained optimization (Prompt 6)

Material for Methods/Results, documenting a controller-model-consistency correction and its evaluation.
Companion to `README.md`.

## Methods: the correction, and why it is NOT parameter tuning

> The initial CBF controller optimized desired velocity and subsequently applied acceleration-rate limiting
> during execution. This created a mismatch between the CBF-certified velocity and the physically executed
> velocity in rare high-closing-speed joint-conflict scenarios. The revised formulation incorporates the UAV's
> one-step acceleration-reachable set directly into the constrained optimization.

This is a model-consistency correction, not a tuning exercise: no CBF gain (`alpha`), safety radius, obstacle
clearance, defender prediction horizon, nominal-pursuit law, or policy was touched. The acceleration limit
(`a_max=6.0 m/s^2`) and substep cadence (`dt_sub=0.1 s`) were ALREADY frozen physical constants governing the
executor since Prompt 1 -- this stage only makes the optimizer AWARE of a limit the executor already enforced
unconditionally, so the certificate the optimizer produces and the motion the vehicle actually flies are the same
velocity, to within numerical tolerance.

### Terminology correction
Because the added reachability constraint is an exact Euclidean-norm (quadratic) inequality, not a linear one,
the solver is more precisely described as a "CBF constrained velocity optimization" over a convex feasible
region at each stage, rather than claiming every stage is a strictly linear-constraint QP. The objective remains
quadratic throughout, and the feasible region remains convex in the intended operating regime.

## Results

### Root cause confirmed, and the exact mechanism by which it manifested
Prompt 5's two residual stress collisions (seeds 630123, 630203) were traced definitively: the colliding obstacle
hazard carried a deeply negative barrier value the instant it entered the fixed 20 m active-constraint range (the
attacker's closing speed already exceeded what the one-substep stopping-margin formula's reaction-time assumption
could absorb). The velocity-space-only QP certified a direction satisfying the obstacle inequality exactly, but
that direction required more instantaneous acceleration than the vehicle's physical 0.6 m/s (one substep)
budget could deliver -- the executor's independent acceleration clip then silently substituted a DIFFERENT,
less-corrective velocity, and the certified safety margin no longer applied to the motion actually flown.

### The fix eliminates all three previously identified collision cases
Reproducing seeds 630123, 630203 (the two still-colliding Prompt-5 cases) and 630069 (the one Prompt-5 already
fixed) under the new controller: all three now show zero obstacle collisions, and `executor_velocity_error`
(the norm difference between the QP's certified velocity and the ACTUAL post-executor velocity) is at machine
precision (<= 1.4e-16 m/s) throughout every substep of every case -- direct, quantitative confirmation that
`v_optimized == v_executed` now holds. Seed 630069's outcome changes from `soldier_caught` to `intercepted`: a
tactical trade-off (the added constraint changes which velocity the solver selects in this specific conflict),
not a safety regression -- the brief's own framing explicitly treats defender interception as an acceptable,
preferable alternative to any obstacle collision. Seed 630123 reaches a notably different kind of resolution: the
attacker brakes to a complete stop within two external steps at a position still formally inside the obstacle's
inflated clearance buffer, but with confirmed-positive physical clearance (never an actual collision); because
escaping that buffer zone in one substep would require more instantaneous velocity than the 0.6 m/s budget
allows, the controller correctly classifies every subsequent substep `physical_safety_infeasible` and holds the
reachable brake (v=0) for the remainder of the fixed 60-step stress horizon rather than inventing an unreachable
escape maneuver -- exactly the brief's own specified emergency-safety behavior, trading tactical resolution for
an unconditional safety guarantee.

### Monte Carlo confirmation at scale, with an honestly-reported behavioral cost
The 200-case joint stress set, the primary 1,000-episode Greedy campaign (both obstacle-only and full variants),
the 500-episode static-clutter (no-defender) regression, and the four-family (Kalman-Greedy/PN/Lead/RMPC)
generalization diagnostic all show ZERO obstacle collisions -- a complete elimination of every previously known
collision pathway. This safety guarantee has a measurable behavioral cost: the frequency of (safe) reachable
emergency-braking rises sharply relative to Prompt 5 in every campaign (e.g. the 1,000-episode obstacle-only
variant's mean emergency-step fraction rises from ~0% to 6.25%, and the adversarially-severe 200-case stress set's
mean emergency-step fraction rises from ~0% to 41.2%), while mission success (asset-capture rate) is UNCHANGED in
every regime that does not deliberately construct an extreme, near-instantaneous joint conflict (static-clutter
capture remains exactly 100%, matching Prompt 5 bit-for-bit). This exposes a real, previously-latent tension
between the frozen Prompt-2 stopping-margin formula (which implicitly assumes continuous, full-authority
reaction from the moment a hazard activates) and a strictly-enforced one-substep reachability requirement --
flagged here for future design review, not resolved or tuned away in this stage (explicitly out of scope per the
brief).

### Tactical effect: Greedy preserved, RMPC's benefit measurably reduced
Greedy capture/interception rates are statistically unchanged from Prompt 4's historical baseline (capture change
<=0.6 percentage points, far under the brief's 5pp catastrophic-loss threshold). PN's large benefit is
essentially preserved (interception reduction 29.5pp vs Prompt 5's 33pp). RMPC's benefit is more substantially
reduced (22pp vs Prompt 5's 38pp) -- still a large, clearly non-trivial effect, not "largely disappeared," but a
real, honestly-reported change in effect size, plausibly because RMPC's own predictive policy can exploit the
attacker's now-bounded one-substep reaction lag. This does not meet the brief's catastrophic-loss bar but is
flagged explicitly for manual design review alongside the already-known modest Greedy effect, consistent with a
verdict-B outcome (physical safety passes; tactical effectiveness against the most capable defenders needs a
final decision).

## Failure cases / limitations (honest reporting)

1. A measurement bug was found and fixed DURING this stage's own verification, before being reported as evidence:
   the first implementation of the `executor_velocity_error` diagnostic incorrectly included substeps where
   `filter_velocity` took its identity shortcut (hazards/defender momentarily absent partway through a substep
   loop) -- in that case `v_safe` carries no acceleration-reachability guarantee at all (it is simply the raw
   nominal velocity, subject to the SAME ordinary acceleration-rate convergence lag that has existed,
   unremarked, since Prompt 1), so comparing it against the post-executor velocity produced a spurious 3.43 m/s
   "violation." The diagnostic now only evaluates the contract when an actual CBF-QP solve occurred for that
   substep; after the fix, the worst observed discrepancy across a 150-episode-pair spot check is 8.0e-5 m/s,
   consistent with ordinary SLSQP constraint-satisfaction precision, not a design flaw.
2. A second test-harness-only bug (not a controller bug) was found and fixed during this stage's own static-clutter
   regression: an initial script left a real, stationary defender in place rather than parking it far away (the
   established no-defender-interaction pattern), which could incidentally intercept the attacker purely via
   flight-path proximity and depressed the measured capture rate to 28.6%. Correcting the script to park the
   defender (matching the established methodology exactly) restored the expected 100% capture rate, confirming
   the controller itself was never at fault.
3. `physical_safety_infeasible` triggers substantially more often than any prior stage (17% of substeps in a
   150-episode-pair spot check) -- this is the direct, now fully transparent cost of never relaxing acceleration
   reachability, and is the primary behavioral change introduced by this stage.

## Limitations and recommended next steps

The remaining open question is whether the elevated emergency-brake frequency and RMPC's reduced benefit are
acceptable for the final benchmark, or whether a further refinement -- e.g. a longer-horizon/earlier-reacting
stopping-margin formula consistent with strict one-substep reachability, or a collision-cone CBF reformulation --
is warranted before freezing the attacker. Per the brief, none of these are authorized or attempted in this
stage; they are explicitly deferred to a future design-review decision.
