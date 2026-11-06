# Manuscript notes — A1-CBF defender-avoidance stage (Prompt 4)

Material for Methods and Results sections. Companion to `README.md` (which is the technical/reproduction record);
this file is organized for direct lifting into a manuscript draft.

## Methods

### Motivation for replacing the PPO attacker (carried over from Prompts 1-3, restated for self-containment)
A learned (PPO/BC) attacker proved unstable to train reliably against a diverse set of defender policies and
required an extensive curriculum that ultimately did not converge to an acceptable benchmark. We replace it with a
fully deterministic, analytically-grounded controller combining classical guidance (proportional pursuit),
quadrotor-consistent constrained dynamics, and control barrier function (CBF) safety filtering -- a
"guidance-and-CBF evasive attacker." This stage adds the final required capability: active avoidance of an
adversarial, independently-moving interceptor (the defender UAV), jointly with static-obstacle avoidance.

### Literature-inspired architecture
The controller follows the standard CBF-QP safety-filter pattern (Ames et al.): a nominal controller proposes a
greedy/goal-directed command, and a convex quadratic program projects it onto the nearest point satisfying all
active safety constraints, expressed as control barrier function inequalities. Static obstacles and the moving
defender are both modeled as CBF constraints with an acceleration-aware stopping-margin term (an "augmented" CBF
accounting for actuator limits), solved JOINTLY in one QP rather than sequentially -- this avoids the classical
failure mode where obstacle avoidance and threat avoidance can produce conflicting, alternating control outputs.

### Moving-target pursuit
Nominal guidance is a proportional law on the relative position to the current route-planner guidance target
(the asset directly, or a locally-planned waypoint around static clutter): `v_nom = kp * r / ||r|| * min(||r||*kp, v_max)`
style saturation (see `CBFAttackerController.nominal_velocity`, unchanged from Prompt 1).

### Quadrotor velocity model
A constrained 3-D point-mass model: velocity is integrated subject to `max_accel=6 m/s^2`, independent horizontal
speed (`12 m/s`) and climb/descent-rate (`5 m/s`) bounds, substepped at `dt_sub=0.1 s` within each external
`dt=0.5 s` step (unchanged from Prompt 1).

### Static clutter planner
A visibility-graph + A* route planner (reused, untouched, from the existing `uav_defend.obstacles.route_planner`)
selects the guidance TARGET only; it has no notion of safety margins. Role separation is strict: the planner
solves GLOBAL/local-minimum deadlock; the CBF-QP provides LOCAL, per-instant safety (Prompt 2/3, unchanged).

### Obstacle CBF
Each active inflated static AABB (within 20 m) contributes a stopping-margin-aware barrier constraint
`n_i^T v_safe >= -alpha h_i`, `h_i = d_i - s_i^2/(2 a_max) - s_i dt_sub` (Prompt 2, unchanged, `alpha=1.0`).

### Defender dynamic CBF (NEW, this stage)
Identical philosophy applied to the defender as a MOVING hazard: `d_D = ||p_A - p_D|| - R_D` (`R_D=3.5 m`),
closing speed `s_D = max(0, -n_D . v_rel)`, stopping margin `d_stop,D = s_D^2/(2 a_max) + s_D dt_sub`,
`h_D = d_D - d_stop,D`, constraint `n_D^T(v_safe - v_D) >= -alpha_D h_D` (`alpha_D=1.0 s^-1`). The defender's
position is gated at the SAME 25 m range the existing PPO/BC attacker-observation contract uses; its velocity is
independently ESTIMATED via finite differences of gated positions only (never ground-truth), with an explicit
validity flag and first-observation/gap-handling semantics. Across one external 0.5 s step, the defender's
position is extrapolated at CONSTANT velocity for each of the 5 substeps, using the single velocity estimate
computed at the step's start (never re-differentiated mid-step; refreshed only at the next external step).

### Observation limitations
The defender-CBF constraint only exists when the defender is within the 25 m gate. The substep extrapolation
assumes constant defender velocity and explicitly does NOT predict defender acceleration -- it is a
receding-horizon approximation, refreshed every external step from the newly observed state, not a guarantee
against unknown future defender maneuvers.

### Joint QP
`min_v 1/2||v-v_nom||^2` subject to ALL active static-obstacle constraints and the defender constraint (when
observable) simultaneously, plus velocity bounds -- hard QP first; on infeasibility, a shared-slack soft QP
(`10^4 delta^2` penalty, one scalar `delta` relaxing BOTH constraint classes equally -- obstacle safety is never
preferentially sacrificed for defender avoidance or vice versa); on soft-QP infeasibility, emergency brake/hover
(zero commanded velocity, never an invented evasive direction).

## Results

### Matched obstacle-only vs. full-CBF experiment (primary, vs. Greedy, N=1,000 matched pairs)
Common-random-number matched comparison isolating the causal effect of the defender-CBF constraint. The full
variant reduced defender-interception rate by 1.3 percentage points (96.4% -> 95.1%), a modest but statistically
significant effect (paired bootstrap 95% CI for the beneficial-direction difference: [0.2%, 2.5%], excluding
zero), with a small, favorable side-effect on asset-capture rate (+1.3 pp, 3.6% -> 4.9%). The pre-registered
10-percentage-point interception-reduction threshold and 1.0 m median-separation-increase threshold were NOT met.
Root cause: Greedy's defender speed (18 m/s) exceeds the attacker's (12 m/s) by 50%, and the Greedy policy always
re-aims at the attacker's instantaneous position -- a fundamental physical limitation of local, reactive,
acceleration-bounded evasion against a faster, continuously-re-aiming pursuer, confirmed (not merely asserted) by
the defender-family generalization result below.

### Defender-family generalization (N=200 matched pairs x 4 additional families)
The SAME controller, with NO parameter re-tuning, was evaluated against Kalman-Greedy, Proportional Navigation
(PN), Lead-intercept, and a receding-horizon model-predictive-control (RMPC) defender. Results were
family-dependent and informative: against the two PREDICTIVE/model-based defenders (PN, RMPC), the defender-CBF's
evasive maneuvering produced LARGE, bootstrap-significant effects (PN: 95% -> 62% interception, -33 pp; RMPC:
75% -> 37%, -38 pp), with correspondingly large asset-capture-rate improvements (PN: 5% -> 38%; RMPC: 25% -> 63%).
Against REACTIVE, always-correctly-aimed defenders (Kalman-Greedy, Lead), effects were small (-1 pp, -2 pp)
consistent with the primary Greedy result. Zero obstacle collisions occurred in any of the 800 additional
generalization episodes. We interpret this as evidence that CBF-triggered evasive maneuvering is most effective
against predictive-targeting interceptors (whose aim is thrown off by sudden, non-linear velocity changes) and
provides only marginal benefit against interceptors with a decisive raw-speed advantage that re-aim every step --
a mechanistically interpretable, honestly reported, and (we believe) novel piece of counter-UAS evidence.

### Joint obstacle+defender stress set (N=200, deterministically constructed)
200 scenarios were deterministically constructed placing the attacker within 10 m of an inflated static obstacle
(moving inward) and within 25 m of a closing defender simultaneously (both hazards verified active by
construction). The joint QP activated both constraint classes simultaneously in 100% of scenarios, with zero
numerical failures. However, 3/200 (1.5%) scenarios produced a genuine physical obstacle collision, traced to the
shared single-slack soft-fallback mechanism accepting an unboundedly large slack value under severe, geometrically
opposed joint-constraint conflict -- a documented, narrow limitation of the frozen shared-slack design that was
NOT observed in 1,000 organically-sampled primary-campaign episodes (where such extreme geometric configurations
are rare), only under deliberately adversarial construction.

## Failure cases (honest reporting, brief requirement)

1. Against a raw-speed-advantaged, reactive pursuer, the attacker is usually still intercepted regardless of the
   defender-CBF constraint (95.1% in the primary full-variant campaign) -- the controller delays and occasionally
   prevents interception, but does not reliably defeat such a defender.
2. 1.5% of deliberately-adversarial joint stress scenarios produced a real obstacle collision via slack
   overcommitment under severe constraint conflict.
3. The median-minimum-separation metric is insensitive when most episodes terminate at the fixed 2.5 m physical
   interception boundary in both variants -- a reporting limitation uncovered during analysis, not a controller
   defect, but worth flagging for future effect-size metric design (e.g., time-weighted separation, or
   separation at a FIXED time horizon rather than the episode minimum).

## Limitations of receding-horizon defender modeling

The defender-CBF's safety guarantee is conditional on the assumption that the defender's velocity remains
approximately constant over the current 0.5 s external step (5 substeps of constant-velocity extrapolation). It
provides NO guarantee against a defender that accelerates sharply within that window in a way not yet reflected in
the attacker's finite-difference velocity estimate. This is an inherent property of reactive, receding-horizon CBF
safety filtering (as opposed to, e.g., a worst-case robust/tube-based formulation), explicitly acknowledged rather
than claimed away.
