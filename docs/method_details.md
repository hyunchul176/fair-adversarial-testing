# Method details

This document holds the method details: policy, shield and refinement (former Sections S1.2–S1.4, with Table S1 and Algorithm S1), data, systems under test, baselines and terms (S2.2–S2.5, Tables S3 and S4) and the CARLA set-up (S4.3). Its text comes verbatim from the supplementary material of the paper "Fair adversarial testing of automated vehicles in highway cut-ins: reference-shielded attacks and a capability-controlled evaluation", which is kept in this repository instead of being published with the paper (supplement draft v13 of 2026-10-12, text as of 2026-11-14). Its last section holds paragraphs that the paper gives only in summary, taken verbatim from an earlier, longer draft of the paper. Section, figure and table numbers without "S" refer to the paper; numbers with "S" refer to sections and tables of these documents, and README.md lists where each is. The figures of the supplementary material are not reproduced.

## S1.2 Policy

Guard G1 (commitment) rests on the observation that the natural highD cut-ins are all monotone, but this holds by the extraction definition: the extractor keeps only completed lane changes with stable labels. Aborted lane changes are therefore outside the scope of this study rather than shown to be absent from traffic.

The policy is trained with PPO [Schulman et al. 2017] (stable-baselines3 [Raffin et al. 2021]) in 15 parallel environments, with an MLP of 2 × 128 units, n_steps 128, a batch size of 256, 8 epochs, γ = 1, λ = 0.95, a learning rate of 3 × 10⁻⁴, an entropy coefficient of 0.01 and a clip range of 0.2.

## S1.3 Shield

When its premises fail, the shield is an approximation. In 2 of 170 replays of the same attacks on two machines, the verdicts differed through floating-point differences (premise vi).  On Autoware the closed-loop decisions sometimes took longer than one 0.1 s tick (maximum 0.64 s), so premise (iii) held only approximately, and every Autoware failure is confirmed by open-loop reruns (Sections 5.5 and 6.7). In CARLA, Proposition 1's premises do not hold, because the shield replays C in a highway-env replica while the verdict replays C with CARLA's physics. With the replica's perception point and bodies matched to CARLA's (Section S4.3), none of 1,488 closed-loop runs of the agent was attacker-caused, against 4 with highway-env's own geometry and 99 without the shield.  The shield needs 15.8 virtual reference replays per attack on average (maximum 84) and no SUT query.

## S1.4 Refinement

Expected improvement [Jones et al. 1998] is computed over 4,096 uniform candidates and 1,024 perturbations of the incumbent (σ = 0.08). The initial design is the boundary plan if it is admissible, otherwise the admissible plan nearest to it among 1,024 local perturbations, followed by uniform draws, each screened, until five plans have run. The boundary plan was admissible in 14 of the 62 development cases, and in 365 of 1,054 searches only uniform draws were admissible. The search stops at the first plan that satisfies the success contract, after 36 SUT queries, after an iteration without an admissible candidate, or after 50,000 C replays.

The objective is robust because optimising the binary fair verdict drives the search to knife-edge cases that C escapes by centimetres. On highD reserved recordings, a fair-verdict objective found 95% fair but only 69% robust failures for the weakest SUT (earlier version).

Screening makes the refinement fair at execution time. The earlier version (v2j) queried the SUT with every candidate and penalised inadmissible ones in the objective: +20 if C collides, otherwise 10·max(0, 0.5 − C margin), 10·[feint], 5·max(0, e_G2 − 1) and 5·max(0, s_joint − 2.77). Only its *reported* failures were fair. Screening removes these queries and raises detection (Section 6.6), at a cost of about 140 C replays per SUT query. These replays are cheap in highway-env but not free. On Autoware, where each query also waits for the stack, the screened refinement took 2.20 and 5.92 instance-hours for the two configurations, against 1.57 and 4.52 for the unscreened one, while using fewer Autoware queries. Replays parallelise across cores.

Table S1 lists when, and in which stage, each condition of the success contract is checked, and Algorithm S1 gives the whole test of one case and SUT, with its budgets and stopping rules, as pseudocode.

Table S1. When and in which stage each success condition is checked

| Success condition | Closed-loop policy and shield (Stage 1) | Screening and refinement (Stage 2) |
|---|---|---|
| Attack-authority contract | Policy actions projected onto the admissible range; validity of the realised attack checked | Plan and executed result checked against the authority contract |
| C's minimum margin of 0.5 m | Held and recover conditions of the shield, and the final verdict; the theoretical guarantee holds only under the stated premises (Proposition 1, Section 4.3) | Checked on C's replay of the whole plan before the SUT query |
| Commitment (G1) | Satisfied by construction: the lane goal cannot be withdrawn once "enter" is chosen; the retreat on C's replay is recorded in the verification record | Checked on C's replay before the SUT query (no retreat of more than 0.2 m toward the departure lane after C's 1.095 m perception boundary) |
| Naturalistic envelope (G2) and joint realism | Checked on C's replay when the completed attack is finally reported (separately from the training reward) | Checked on C's replay before the SUT query |
| SUT collision with the adversary at least 3.5 m ahead at contact | Judged after the SUT run | Judged after the SUT run |
| Optional severity floor | Checked at contact; if missed, the refinement follows | Checked at contact |

Note. G1, commitment; G2, naturalistic envelope (Section 3.3).

Algorithm S1. Reference-guided adversarial testing of one case and SUT

```text
Input: case i, SUT S, reference driver C, trained policy π,
       refinement budgets B_S = 36 SUT queries and B_C = 50,000 C replays,
       initial evaluation target n0 = 5, random-draw cap N_draw = 4,096,
       optional severity floor v_min

    // Stage 0 (boundary plan); no SUT query
1   b ← FindReferenceBoundary(i, C)

    // Stage 1 (policy stage); one SUT query
2   (τS_pol, ξ_obs) ← RunShieldedPolicy(i, S, π, C)
3   if Collision(τS_pol):
4       ξ_pol ← ExtendLastDecision(ξ_obs, first SUT contact)
5       τC_pol ← ReplayReference(i, C, ξ_pol)
        // G1 holds by construction; retreat recorded
6       if FinalSuccess(τS_pol, τC_pol, v_min):
7           RecordVerification(); return SUCCESS

    // Stage 2 (refinement): budgets and observations
8   q ← 0; r ← 0; draws ← 0; evaluated ← ∅

    // Screening helper: one C replay, reused in the final verdict
    Screen(p):
        if r ≥ B_C: return LIMIT
        r ← r + 1
        τC ← ReplayReference(i, C, p)
        if margin(τC) ≥ 0.5 m and G1(τC) and G2(τC) and J(τC) ≤ 2.77: return τC
        return REJECT

    Evaluate(p, stored τC):
        τS ← QuerySUT(i, S, p); q ← q + 1
        if FinalSuccess(τS, τC, v_min):
            RecordVerification(); return SUCCESS
        evaluated ← evaluated ∪ {(p, Objective(τS, τC))}
        return CONTINUE

9   τC_warm ← Screen(b)
10  warm ← b if τC_warm is admissible, otherwise NONE
11  if warm is NONE and r < B_C:
12      L ← 1,024 perturbations of b in the unit box (σ = 0.08)
13      for p in L, nearest to b first:
14          τC ← Screen(p)
15          if τC = LIMIT: return NO_REPORTABLE_FAILURE
16          if τC is admissible:
17              (warm, τC_warm) ← (p, τC); break
        // if no perturbation is admissible, warm stays NONE
        // and the random initialisation follows
18  if warm exists and q < B_S and r ≤ B_C:
19      if Evaluate(warm, τC_warm) = SUCCESS: return SUCCESS

20  while |evaluated| < n0 and draws < N_draw and q < B_S and r < B_C:
21      p ← SampleRandomPlanInUnitBox(); draws ← draws + 1
22      τC ← Screen(p)
23      if τC = LIMIT: break
24      if τC = REJECT: continue
25      if Evaluate(p, τC) = SUCCESS: return SUCCESS
    // no plan executed: zero SUT queries
26  if q = 0: return NO_REPORTABLE_FAILURE

27  while q ≥ 1 and q < B_S and r < B_C:
28      GP ← FitGP(evaluated, Matérn-5/2 + white noise)
29      U ← 4,096 uniform random plans
            ∪ 1,024 perturbations of the current best plan (σ = 0.08)
30      rank U by decreasing expected improvement EI_GP(p)
31      found ← FALSE
32      for p in ranked U:
33          τC ← Screen(p)
34          if τC = LIMIT: break                        // C-replay budget reached
35          if τC is admissible:
36              found ← TRUE; break
        // C-replay budget reached, or no admissible candidate in U
37      if not found: break
38      if Evaluate(p, τC) = SUCCESS: return SUCCESS

39  return NO_REPORTABLE_FAILURE
```

## S2.2 Data

From highD [Krajewski et al. 2018] our extraction yields 266 exposed cut-ins, split into 204 calibration cases (training) and 62 development cases (evaluation). The 62 development cases come from three recordings (36, 16 and 10 cases), which do not overlap the calibration recordings, and three vehicles appear in two cases each (one cutting-in vehicle and two followers). A further 148 cases from reserved recordings were used once in an earlier pre-specified evaluation of a previous version and are reused post hoc here. NGSIM US-101 [FHWA 2016] contributes 47 cases, which were also used once in an earlier pre-specified evaluation and are reused post hoc. TGSIM I-294 L1 and L2 [Ammourah et al. 2025; FHWA 2024a; FHWA 2024b] contributes 80 cases from the pre-specified evaluation of a previous version, reused post hoc for the earlier version v2j and for the final agent.

MiTra [Chaudhari et al. 2025] (Milan A50, recorded in April 2023) contributes 172 cut-ins from its nine flights. They formed the pre-specified evaluation of the earlier version v2j (same policy stage, unscreened refinement), whose protocol was frozen before the data were opened. Later post-hoc analyses reused them (severity, feasibility references, Autoware, the final agent's screened refinement, the human population, the reference readings), each labelled post hoc.

HIGH-SIM [Shi et al. 2021] was recorded by helicopter video on US I-75 in Florida, in one direction (three lanes and an off-ramp) over about 2.44 km at 30 Hz. The paper describes two hours of evening-peak video; the public trajectory file we used [CATS Lab] covers about 38 minutes, and the repository states a CC BY-SA 4.0 licence (its link points to CC BY-NC 4.0). Its 60 cut-ins (a compute cap fixed before the extraction; of 370 eligible cut-ins, 144 were selectable after the initial-state domain and one case per vehicle, Section 5.2) formed the first pre-specified evaluation of the final agent, whose protocol was frozen before the extraction (Sections 5.1 and 6.1).  The dataset differs from highD in institute, country and recording method, but it covers one site and one direction, and the public file we used covers 38 minutes. Cut-ins are extracted with the highD event rule on the three mainline lanes (ramp merges excluded), and passenger cars are identified by their length (< 7 m), because the class column is the same for every vehicle.

exiD [Moers et al. 2022] covers seven motorway entry and exit sites in Germany, recorded by drone; its release v2.1 contains 93 recordings (16.1 hours by the recording metadata). Its 120 cut-ins come from 59 recordings at all seven sites, out of 2,300 eligible mainline lane changes (323 selectable after the initial-state domain, one case per vehicle and at most four per recording, then a compute cap; Section 5.2), and formed the second pre-specified evaluation of the final agent. Its protocol was frozen before the data were opened and before the HIGH-SIM results (Sections 5.1 and 6.1).  exiD comes from the same institute, country and drone method as highD, so it tests new sites and road geometry (on- and off-ramps), not a new data provider. Its cut-ins are extracted with the highD event rule, and the lanes come from the Lanelet2 maps, because exiD has no lane identifiers. The extractor was written and checked on synthetic tracks before any exiD data were seen.  A check of the received maps, before the extraction, found hard-shoulder lanelets without ramp tags, and lane changes into or out of the shoulder were then excluded (1,152 candidates). The 600 eligible on-ramp merges are counted but not simulated.

## S2.3 Systems under test

BC-highD, the behaviour-cloning SUT, was trained on the followers of the 463 exposed natural highD cut-ins, which include the development events, so its development results are not out of sample.

Table 3 of the paper places each SUT by its designed capability relative to C. The following controllers share C's non-emergency law and brake at most 5 m/s², and AEB H0 and H0.5 add emergency braking at 8.0 m/s². Longitudinal RSS treats the cut-in vehicle as relevant once a 0.5 s constant-velocity projection predicts lateral overlap and then brakes at up to 8 m/s² whenever the RSS safe distance is violated, without C's reaction delay; it is intended, not guaranteed, to dominate C, because it can perceive later. Lateral RSS also treats the vehicle as relevant from C's perception point on, so it perceives no later than C, reacts without its 0.75 s delay and brakes harder. It is designed, not guaranteed, to dominate C. In one of the 120 pre-specified exiD cases it had a reported failure. C avoided that attack with its margin, but another reference driver did not. The ADS has C's structure with a 0.3 s latency and 8 m/s² emergency braking and perceives the cut-in once the intruder is displaced by the given threshold from its own lane centre, earlier than C at 0.5 m and later at 2.0 m. BC-highD imitates highD followers, which rarely brake harder than 2 m/s², and has no emergency layer. The learned SUTs are PPO ego policies trained by us on non-adversarial cut-ins (one training seed each); the delayed ones were designated competent after a grid evaluation in which they collided in at most 0.03% of the plans (C: 11.25%). The controls change one factor of C. AW-r157 sets C's braking limits in Autoware's emergency settings, command gates and minimal-risk manoeuvre, with the controller limits as shipped, and the cause-test variants lift the planner's comfort limits and then the controller limits as well (Section S4.1). The SUMO models keep their own gaps, and the CommonRoad planner keeps no headway of its own and avoids only collisions that it predicts within 6 s at constant velocity.

For Autoware, AW-default uses the parameters shipped with the release for the planning simulator. The release does not state the speed range for which they are tuned, and the development cases start at a median SUT speed of 29.5 m/s, so AW-default may not represent a configuration intended for this domain; AW-r157 changes only the emergency braking limits.

For the third-party SUTs (B69), the simulator stays the physics and the judge. The third-party code chooses only the SUT's longitudinal acceleration every 0.1 s; steering is the lane controller shared by all SUTs, and the 9 m/s² braking contract applies. For the SUMO models, SUMO's own leader choice and car-following response give the next speed, capped by its free speed. The planner replans every tick with the adversary as a constant-velocity obstacle and applies the first step of its cheapest collision-free trajectory; when no sampled trajectory is collision-free it brakes at 9 m/s², since it has no fallback above standstill. Configurations are as shipped except for the vehicle size (5 × 2 m) and, for the planner, the 9 m/s² braking limit and lane-keeping lateral samples (premise iv of Proposition 1). Neither was tuned to our cases.

## S2.4 Baselines

The baselines re-implement attack principles, not the original systems; Table S3 lists what each keeps and what it misses. The collision-seeking adversary is trained with and without commitment. The Dynasto-style adversary uses safe-distance validity rules with a 0.2 s reaction, together with their dense and terminal rewards. The CARS-style adversary is rewarded with progress, lateral-proximity, lateral-closing and collision terms, and its collisions are attributed post hoc; it is evaluated also with commitment (G1) enforced at test time, although it was trained without it, and retrained with commitment from the agent's initialisation and from scratch.  The naive responsibility-rewarded learner uses our reward without G1 and G2. The feasibility-referenced learners use our training pipeline with only the reward's reference replaced by an omniscient avoider (FREA/STRIVE-type) or a zero-reaction ideal ADS (ScenePilot-type), and they are evaluated without a shield and with a shield built on their own reference.  Training-free search covers random search, CMA-ES and GP-BO, cold- or boundary-started, including the screened BO alone (no policy), the strongest training-free baseline (Section 6.6).

Table S3. What the baselines re-implement and what they leave out

| Baseline | Original system | What we kept | What is missing |
|---|---|---|---|
| CARS-style reward | CARS [Xiao et al. 2026] | its reward terms and post-hoc attribution | the data-driven diffusion prior (which may suppress feints), its scenes and datasets |
| Dynasto-style rules | Dynasto [Humeniuk et al. 2026] | the validity rules and rewards | its optimiser and environment |
| Feasibility-referenced | FREA [Chen K. et al. 2024], STRIVE [Rempe et al. 2022], ScenePilot [Ruan et al. 2026] | the reference principle (an idealised avoider: optimal without reaction delay, or knowing the adversary's future) | the learned Hamilton–Jacobi feasible region, the traffic prior, the boundary band |

## S2.5 Terms, cost accounting, seeds and intervals

Table S4. Terms and metrics

| Term | Meaning here |
|---|---|
| detection within k | share of cases in which a failure satisfying the success contract is found within k SUT queries (policy query first, then refinement) |
| light contact | a failure closing below 3 m/s at contact |
| reported / confirmed (Autoware) | reported: found in the closed-loop run; confirmed: its adversary trajectory, replayed open-loop four times, collides at least twice |
| consensus-fair | C, \(C_\mathrm{strict}\), FSM and the RSS reference driver all avoid |
| discrimination (P1) | detection of the weakest ladder SUT minus that of longitudinal RSS (both at 37 queries) |
| robust | margin conditions of the verdict (adversary's centre ≥ 3.5 m ahead of the SUT's at contact, a front-position criterion that also admits side contacts near the SUT's front; C margin ≥ 0.5 m); unrelated to reproducibility |
| post hoc | an analysis defined after the data were seen, or a reuse of data already used once |
| implementation check | an outcome guaranteed by construction (e.g., zero attacker-caused policy-stage collisions), reported to verify the implementation |

The query budget k counts SUT search queries only. Reported separately are the SUT verification replays and the Autoware confirmation reruns; the reference replays of the shield, the refinement and the screening; and the training steps and the wall-clock or instance time.

Policy results are seed means over three training seeds, and the seed s.d. are given in the tables. The refinement runs once per case and SUT with one random stream, so these s.d. do not include its variation. Two further streams on the development cases (B67) moved P1 by at most 1.6 points and the weak-ladder mean by at most 1.5, and the s.d. of P1 over the three streams (0.9 points) is a quarter of its case-bootstrap standard error (3.7).  The case bootstrap resamples seed-averaged case values, so it does not include training variation either. Several comparisons rest on one training seed: the naive responsibility learner, the non-committing collision seeker, the guard ablation, the learned SUTs and CARLA.

Intervals are case-bootstrap 95% intervals, with flight-cluster intervals for MiTra. The three development recordings are too few for cluster intervals, and per-recording values are given in Section S6. Development and Autoware results are in-sample, because about sixty design iterations used the same 62 cases.

## S4.3 CARLA

**Set-up.** The CARLA runs use CARLA 0.9.16 [Dosovitskiy et al. 2017] in synchronous mode with 0.1 s steps, rendered off screen, on a straight stretch of Town04 with a same-direction left lane. Its lanes are 3.5 m wide, against 4 m in highway-env. In the main configuration all three vehicles are Ford Mustangs (4.72 × 1.90 m) with a calibrated feed-forward brake map, and a Tesla Model 3 (4.79 × 2.16 m) serves as the wider vehicle. Each run starts with a 3 s pre-roll at the case's SUT speed, and contact is the overlap of the oriented bounding boxes, as in highway-env. The adversary is not driven by CARLA physics. Its motion is computed in highway-env from its decisions and imposed on its CARLA vehicle at every step, with its lateral progress scaled by 3.5/4 between the lane centres (proportional mapping). A sensitivity run instead keeps the lateral distance to the SUT's lane centre in metres (distance-preserving mapping). C perceives the cut-in once the adversary has moved 1.095 m in CARLA's metres. C and our ladder SUTs drive CARLA vehicles through a common interface that turns their acceleration requests into vehicle commands, with a common lane keeper for steering. CARLA's BehaviorAgent, with its cautious, normal and aggressive profiles, and its Traffic Manager, with lane changes disabled, drive with their own controllers, with the cruise speed set to the case's SUT speed. In closed loop the policy observes the adversary's highway-env state and the CARLA SUT's state mapped into highway-env units, and each verdict replays the realised adversary trajectory against C in CARLA.

**Harness correction.** The interface, our harness, sends cruising and mild deceleration through CARLA's constant-velocity mode and stronger braking through the brake map. Before the runs reported in Section 6.7 of the paper and in Section S13.5, we found three defects. First, the longer a vehicle had cruised in constant-velocity mode, the more residual drive force it carried afterwards, a CARLA artefact. C's 0.4 m/s² reaction deceleration then became zero or even an acceleration of 0.3 m/s², and its emergency braking levelled off at 6.7–7.5 m/s² instead of 7.59 m/s². Second, the measured acceleration used by the harness's PI correction was always zero, a defect of our code, so with PI the vehicles braked about 1.5 times as hard as requested. The main configuration had therefore already run without PI. Third, constant-velocity mode ran about 0.5% slower than commanded (0.3% for the Tesla). Cruising SUTs and C therefore reached the cut-in 0.5–1 m behind their highway-env twins, with start-speed errors of −0.05 to −0.09 m/s. The corrected harness sends requests weaker than −1 m/s² exactly through constant-velocity mode, and stronger braking through the brake map minus the disturbance measured over the previous braking step. It drops the PI correction and corrects the commanded cruising speed with the integral of the measured speed error, from the pre-roll on. We checked it in actuation probes and on 20 stored attacks run with C and three SUTs on both vehicles. C's reaction-phase deceleration was −0.40 ± 0.02 m/s² for a 0.4 m/s² request, and its emergency plateau 7.55–7.61 m/s² for 7.59 m/s². The mean braking error was within ±0.03 m/s², against under-braking by 0.6–0.8 m/s² before, and start-speed errors were within ±0.003 m/s. These 20 attacks had made only the Tesla's C collide in an earlier replay; rerun with the earlier harness, the Tesla's C collided in 11 of them, and after the correction in none. Every CARLA experiment was then rerun from scratch. Results obtained with the earlier harness, including the CARLA results of an earlier prototype that used a reference driver calibrated to the old actuation, are not used as evidence. Table S11 records where they informed design decisions.

**Matched twin.** The shield and the screening replay C in highway-env, whose lane width, body sizes and, in highway-env units, perception point differ from CARLA's. In the matched twin, C keeps its R157 reaction and braking, but its perception point and the vehicle bodies are placed where they fall in the execution environment. With the proportional mapping and the Mustang, C perceives at 1.095 × 4/3.5 = 1.251 m, and the bodies are 1.895 × 4/3.5 = 2.166 m wide and 4.718 m long. For the Tesla the values are 1.251, 2.469 and 4.79 m, and for the distance-preserving mapping 1.595, 1.895 and 4.718 m. The default twin keeps highway-env's 1.095 m, 2 m and 5 m. These values follow from the geometry and were not fitted to CARLA outcomes. The body length also enters highway-env's vehicle kinematics and moves the adversary's trajectory for the same plan by at most 1.2 cm. G2 and joint realism are computed on the reference replay in the default twin, as in highway-env.

**Re-checking stored attacks.** A stored attack found in highway-env can be re-checked for another environment by replaying it in the twin matched to that environment, against C and against a model of the SUT. It is kept only if the SUT model still collides and C still avoids the collision with at least 0.5 m between the bodies. For a black-box SUT only C's side can be re-checked. This keeps the attack fair in the new environment but does not ensure that the SUT fails there (Section S13.5).

**Repeatability.** CARLA in synchronous mode is close to, but not fully, deterministic. A second closed-loop run of one policy seed reproduced all 496 case–SUT verdicts. A second replay of the 514 stored development trajectories agreed in 99.0% of the transfer verdicts (SUT 99.2%, C 99.8%), and a repeat of 124 searches inside CARLA agreed on success in 95.2% of them. With the earlier harness these shares had been 99.2, 97.3 and 91.9%, so its artefacts also added variability. The searches inside CARLA and their repeat ran on a second machine and all other CARLA runs on the first, because CARLA physics can differ slightly between machines.

**Top views.** The top views in Figs. 1 and 6 replay stored attacks of development case 207. A recording run executes the episode with CARLA physics and logs both vehicles every 0.1 s, and a rendering run then places the vehicles, with physics off, at poses interpolated to 30 frames per second, with the camera following the follower. Both figures were recorded with the corrected harness. Figs. 1(b) and 6 show the stored policy-stage attack of the agent's second training seed on this case. The first seed's attack is avoided in CARLA by following H0, by 0.05 m, so the attack was chosen between the second and third seeds by a rule fixed before the run: following H0 collides and C avoids it with its 0.5 m margin, with the fewest collisions of the other followers. Both seeds qualified, with no other follower colliding, and the rule's tie-break chose the second. Fig. 6 replays one adversary trajectory open loop against several followers rather than generating a new attack for each SUT, and because CARLA does not repeat runs exactly, minimum gaps can differ slightly between renders of the same attack.  A figure in an earlier draft of the paper posed three further attack shapes from stored states, and Section S13.5 reports their run with CARLA physics. A later version of that figure showed two shapes that hold when run with CARLA physics (additional_results.md, section "Attack shapes of the former Fig. 7, run with CARLA physics"), and it was removed from the paper on 2026-11-16.

## Moved from the paper

The paragraphs below give in full details that the paper summarises. They are taken verbatim from an earlier, longer draft of the paper and grouped by the section of that draft they come from, whose number and title each heading gives. Their cross-references were updated to the paper's numbering on 2026-11-14 and again on 2026-11-16. On 2026-11-16 their term "pre-registered" was also changed to the paper's "pre-specified", and each heading now also names the section of the paper that holds its content.

### From the earlier draft's Section 4.2, Learning a closed-loop adversarial policy (now Section 4.2)

Every 0.5 s the policy \(\pi_\theta(a_t \mid o_t)\) receives a 16-dimensional observation \(o_t\) and chooses the adversary's action \(a_t\). The observation contains the relative position and velocity of the two vehicles and their closing speed, the adversary's own lateral state and last commands, the elapsed time, the lane goal and entry flag, the initial gap and speed difference, and the SUT's last acceleration. The policy receives no SUT identity, reference verdict or future information. Each action combines a lane goal \(\ell_t \in \{\mathrm{stay}, \mathrm{enter}\}\) with a longitudinal acceleration request \(u_t \in \{+1, 0, -1, -2, -3\}\) m/s². The requested action is projected onto the acceleration, jerk, speed and lane-change limits of the attack-authority contract (Section 3.1) before it is executed. Once the policy has chosen to enter the lane, the lane goal cannot be withdrawn, which suppresses feints. The policy stage therefore satisfies commitment (G1) by construction, and the adversary's retreat on C's replay is recorded in the verification record.

We train the policy with PPO [Schulman et al. 2017], implemented in stable-baselines3 [Raffin et al. 2021], in 15 parallel environments (hyperparameters in Section S1.2). Each episode draws, uniformly and independently, one of the 204 highD calibration cut-ins and one of seven rule-based SUTs that differ in reaction timing, braking and other properties. These seven SUTs, the training panel, are the same as the capability ladder on which detection by avoidance capability is later evaluated (Section 5.3). We therefore do not claim generalisation to new systems from the results on these SUTs alone, and we report the pre-specified evaluations on new traffic cases and the results on SUTs outside the training panel separately. A collision-seeking policy with commitment is first trained for 0.75 M steps to serve as the initial policy, and it is then trained for 1.5 M further steps with the reward above. Training thus takes 2.25 M steps per seed and is repeated with three independent seeds. At test time, the trained policy makes the first, closed-loop SUT query, with each of its actions checked by the shield.

### From the earlier draft's Section 4.4, Reference-guided boundary plan and screened refinement (now Section 4.4)

**Open-loop plan space and initialisation.** The refinement searches a five-dimensional plan space: the entry time (0–8 s), the longitudinal acceleration requests before entry, just after entry and from 2 s after entry (each −3 to +1 m/s²), and the lateral duration of the lane change (2–6 s). Each plan fixes the adversary's decisions for the whole episode, so the same complete plan can be applied to C and to the SUT without extending the last decision at the SUT's collision. The search first checks whether the boundary plan passes screening. If it does, it becomes the first initial evaluation point. If it does not, 1,024 local perturbations of the boundary plan, with a standard deviation of 0.08 in the normalised unit plan space, are screened in order of their distance to it, so that the nearest admissible plan is found. Random plans are then screened one at a time, and the admissible ones are run on the SUT, aiming at five initial evaluation points in total; at most 4,096 random plans are drawn. Initialisation may end earlier, through a success or a budget limit. If no plan has been run on the SUT when initialisation ends, the search stops with zero SUT queries. A reportable success found during initialisation also ends the search, and the SUT queries of the initial evaluations count against the refinement's budget.

**Screening before the query and candidate generation.** Before an SUT query, each plan is replayed with C and checked for a minimum distance between the bodies of at least 0.5 m, for G1, for G2 and for \(J \le 2.77\); a plan that fails any of these checks is not run on the SUT.  On this replay, G1 is checked as a retreat test: once the adversary has crossed C's perception boundary, 1.095 m from the centre line of its departure lane, it must not move back towards that lane by more than 0.2 m. After the initial evaluations, each GP-BO iteration generates 4,096 random candidates in the normalised plan space and 1,024 local candidates around the current best plan. The two sets are ranked together by expected improvement (EI) and replayed with C in that order, and **only the first admissible plan** is queried on the SUT. The 1,024 candidates around the initial boundary plan serve the initialisation, whereas the 1,024 candidates around the best plan serve the GP-BO iterations, so the two sets differ in their centre and role. The complete C replay computed for screening is reused for the final verdict. After the SUT run of an admitted candidate, the only conditions left to check are whether a collision actually occurs, the adversary's front position at contact (at least 3.5 m) and, if a severity floor is applied, the closing speed at contact.

**Objective and cost of GP-BO.** The refinement aims to reduce the SUT's minimum gap, computed from the SUT run of an admissible plan, and 5 is subtracted from the objective value of a plan that yields a robust fair failure. The GP takes the five-dimensional plan in the unit box as input and models the SUT's response with a Matérn-5/2 kernel plus a white-noise term [Rasmussen and Williams 2006]. Screening is the limiting case of constrained Bayesian optimisation [Gardner et al. 2014] in which the constraint is checked exactly by a deterministic replay rather than modelled. Admissibility is evaluated directly by C's replay, and the GP reflects only the responses of the plans run on the SUT. As a result, a candidate with a high EI can still lie in an inadmissible region, and many C replays may be needed to filter such candidates out. Screening thus spends no SUT query on inadmissible attacks, but the computational cost of the reference replays remains.

### From the earlier draft's Section 5.2, Scenario cohorts and experimental environments (now Section 5.2)

HIGH-SIM and exiD cut-ins are extracted with the same highD event rule, and only lane changes on the mainline are tested. The public HIGH-SIM trajectories cover one direction of about 2.44 km of US I-75 at 30 Hz, with three mainline lanes and an off-ramp. Because the class column is the same for every vehicle, passenger cars are identified by their length (< 7 m). Of the 370 cut-ins that the extractor accepted, 175 lay outside the simulator's initial-state domain: a follower speed of 15–35 m/s, a cut-in vehicle speed of 15–40 m/s and a duration of 2–8 s for the lateral movement from 10 to 90% of the lane change. A hash-ranked pass that uses each vehicle in at most one case left 144 selectable cut-ins, and the pre-specified evaluation took the first 60, a compute cap fixed before the extraction.  Ramp merges are not included, so the conclusions drawn from these data are not extended to merging in general.

For exiD we use release v2.1, 93 recordings (16.1 hours) from seven motorway entry and exit sites in Germany. The lanes were determined from its Lanelet2 maps, and the case extractor was checked on synthetic tracks before any exiD trajectories were seen. Lane changes into or out of the hard shoulder, identified by a check of the maps before the extraction, were excluded (1,152). Of the 2,300 eligible mainline lane changes, 408 lay outside the same initial-state domain. A hash-ranked pass that uses each vehicle in at most one case and takes at most four cases per recording left 323 selectable cut-ins. The evaluation took the 120 with the smallest hash, from 59 recordings, as a compute cap, and a post-hoc extension later ran the other 203 (Section 6.1).  The 600 eligible on-ramp merges were counted but not simulated. exiD thus tests mainline cut-ins on new road sections with entries and exits; it does not validate on-ramp merging itself. The number of cases is limited by the compute budget.

CARLA 0.9.16 [Dosovitskiy et al. 2017] provides a second execution environment, with vehicle physics. We use a straight motorway section of its Town04 map, whose lanes are 3.5 m wide instead of highway-env's 4 m. In the main configuration all three vehicles are Ford Mustangs (4.72 × 1.90 m), and a Tesla Model 3 (4.79 × 2.16 m) serves as a wider vehicle. The SUT and the reference driver C move under CARLA's vehicle physics. The adversary's motion is computed in highway-env from its decisions and imposed on its CARLA vehicle at every 0.1 s step, with its lateral progress scaled to the narrower lanes. The agent is applied without retraining, and its shield and screening still replay C in highway-env. That replica, however, is matched to the geometry of the execution environment. C's perception point and the vehicle bodies lie where they fall in Town04, while C keeps the reaction and braking of Section 3.2. We call this replica the *matched twin*, and the replica with highway-env's own geometry the *default twin*. The verdict on each CARLA run replays C in CARLA itself. We run the shielded policy in closed loop, search inside CARLA with the screened refinement and replay stored attacks. These are follow-up experiments, each with its design and decision rules fixed before the run (Section S13.5). Closed-loop runs and searches use the development cases, whereas the replays also reuse the other datasets post hoc. Before the runs reported here, we found three defects in how our harness turned acceleration requests into CARLA commands, corrected them and repeated every CARLA experiment (Section S4.3). Repeated closed-loop runs then agreed in all 496 verdicts, repeated replays in 99.0% and repeated searches in 95.2% of their outcomes, so CARLA is close to, but not fully, deterministic. Fig. 1 renders stored highway-env attacks of development case 207 as CARLA scenes, as does the top view in Section 6.3. These renders illustrate verdicts; they are not part of the evidence from CARLA reported in Section 6.7.

### From the earlier draft's Section 5.6, Pre-registered hypotheses and decision rules (now Section 5.6)

**Sample size and power.** The numbers of cases were set by the protocol rules: compute-budget caps of 60 HIGH-SIM and 120 exiD cases, and a minimum analysis size of 20 cases. Before the evaluation data were opened, the expected power of the hypotheses was examined in 10,000 resampling simulations of existing case cohorts. With cases distributed as in MiTra, P1 was predicted to pass in most simulations but K1 and K2 to fail in most; with cases distributed as in the highD development data, all three criteria were predicted to pass almost surely. These are design-stage simulations under the distributions of earlier data; they are neither performance observed on HIGH-SIM or exiD nor a guarantee about their distributions. Section S3.3 gives the sampling and simulation settings.

**Statistical decisions and evidence status.** Policy-stage results are means over the three training seeds, and by default the refinement runs once per case–SUT pair with one random stream. To check the effect of the random stream, two further streams, with the design fixed before the run, were evaluated on the development cases; this check is kept apart from the main estimates. Intervals are 95% case-bootstrap intervals with 10,000 resamples of case-level values that are first averaged over seeds. Cluster-bootstrap intervals by unit of data collection, such as recording or site, are reported separately, with flights as the clusters for MiTra. The development data come from only three recordings, so no recording-cluster interval is computed for them. Each pre-specified hypothesis is judged individually, on its point estimate against the threshold fixed in advance. Judgements based on the lower 95% bound of P1 and the upper 95% bound of P2 are given alongside as auxiliary information. These auxiliary judgements do not replace the pre-specified judgements on point estimates, and no correction for multiple comparisons is applied. Section S9 lists the judged hypotheses, the implementation checks expected by construction, and the post-hoc and descriptive statistics separately.

## References

- [Schulman et al. 2017] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, "Proximal policy optimization algorithms," 2017, arXiv: 1707.06347.
- [Raffin et al. 2021] A. Raffin, A. Hill, A. Gleave, A. Kanervisto, M. Ernestus, and N. Dormann, "Stable-Baselines3: Reliable reinforcement learning implementations," *J. Mach. Learn. Res.*, vol. 22, no. 268, pp. 1–8, 2021.
- [Jones et al. 1998] D. R. Jones, M. Schonlau, and W. J. Welch, "Efficient global optimization of expensive black-box functions," *J. Global Optim.*, vol. 13, no. 4, pp. 455–492, 1998, doi: 10.1023/A:1008306431147.
- [Krajewski et al. 2018] R. Krajewski, J. Bock, L. Kloeker, and L. Eckstein, "The highD dataset: A drone dataset of naturalistic vehicle trajectories on German highways for validation of highly automated driving systems," in *Proc. IEEE ITSC*, 2018, pp. 2118–2125, doi: 10.1109/ITSC.2018.8569552.
- [FHWA 2016] U.S. Department of Transportation, Federal Highway Administration, "Next Generation Simulation (NGSIM) vehicle trajectories and supporting data," [Dataset], ITS DataHub, 2016, doi: 10.21949/1504477.
- [Ammourah et al. 2025] R. Ammourah et al., "Introduction to the third generation simulation dataset: Data collection and trajectory extraction," *Transp. Res. Rec.*, vol. 2679, no. 1, pp. 1768–1784, 2025, doi: 10.1177/03611981241257257.
- [FHWA 2024a] U.S. Department of Transportation, Federal Highway Administration, "Third Generation Simulation Data (TGSIM) I-294 L1 trajectories," [Dataset], ITS DataHub, 2024, doi: 10.21949/1404217.
- [FHWA 2024b] U.S. Department of Transportation, Federal Highway Administration, "Third Generation Simulation Data (TGSIM) I-294 L2 trajectories," [Dataset], ITS DataHub, 2024, doi: 10.21949/1404224.
- [Chaudhari et al. 2025] A. A. Chaudhari, M. Treiber, and O. Okhrin, "MiTra: A drone-based trajectory data for an all-traffic-state inclusive freeway with ramps," *Sci. Data*, vol. 12, Art. no. 1174, 2025, doi: 10.1038/s41597-025-05472-0. Dataset: doi: 10.25532/OPARA-881.
- [Shi et al. 2021] X. Shi, D. Zhao, H. Yao, X. Li, D. K. Hale, and A. Ghiasi, "Video-based trajectory extraction with deep learning for High-Granularity Highway Simulation (HIGH-SIM)," *Commun. Transp. Res.*, vol. 1, art. no. 100014, 2021, doi: 10.1016/j.commtr.2021.100014.
- [CATS Lab] CATS Lab, "Filed-Experiment-Data-HIGH_Sim" (HIGH-SIM full trajectory data), GitHub repository. [Online]. Available: https://github.com/CATS-Lab/Filed-Experiment-Data-HIGH_Sim
- [Moers et al. 2022] T. Moers, L. Vater, R. Krajewski, J. Bock, A. Zlocki, and L. Eckstein, "The exiD dataset: A real-world trajectory dataset of highly interactive highway scenarios in Germany," in *Proc. IEEE IV*, 2022, pp. 958–964, doi: 10.1109/IV51971.2022.9827305.
- [Xiao et al. 2026] Y. Xiao et al., "Learning responsibility-attributed adversarial scenarios for testing autonomous vehicles," 2026, arXiv: 2605.13751.
- [Humeniuk et al. 2026] D. Humeniuk, M. Hamdaqa, H. Ben Braiek, A. Bennaceur, and F. Khomh, "Dynasto: Validity-aware dynamic–static parameter optimization for autonomous driving testing," in *Proc. IEEE Int. Conf. Softw. Testing, Verification Validation (ICST)*, 2026, pp. 28–39, doi: 10.1109/ICST69053.2026.00018.
- [Chen K. et al. 2024] K. Chen, Y. Lei, H. Cheng, H. Wu, W. Sun, and S. Zheng, "FREA: Feasibility-guided generation of safety-critical scenarios with reasonable adversariality," in *Proc. 8th Conf. Robot Learn. (CoRL)*, ser. PMLR, vol. 270, 2024, pp. 566–586.
- [Rempe et al. 2022] D. Rempe, J. Philion, L. J. Guibas, S. Fidler, and O. Litany, "Generating useful accident-prone driving scenarios via a learned traffic prior," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2022, pp. 17284–17294, doi: 10.1109/CVPR52688.2022.01679.
- [Ruan et al. 2026] Q. Ruan, Y. Wang, H. Li, Z. Li, and C.-Z. Xu, "ScenePilot: Controllable boundary-driven critical scenario generation for autonomous driving," in *Proc. 43rd Int. Conf. Mach. Learn. (ICML)*, ser. PMLR, vol. 306, 2026, pp. 105965–105987.
- [Dosovitskiy et al. 2017] A. Dosovitskiy, G. Ros, F. Codevilla, A. Lopez, and V. Koltun, "CARLA: An open urban driving simulator," in *Proc. Conf. Robot Learn. (CoRL)*, PMLR vol. 78, 2017, pp. 1–16.
- [Rasmussen and Williams 2006] C. E. Rasmussen and C. K. I. Williams, *Gaussian Processes for Machine Learning*. Cambridge, MA, USA: MIT Press, 2006.
- [Gardner et al. 2014] J. R. Gardner, M. J. Kusner, Z. Xu, K. Q. Weinberger, and J. P. Cunningham, "Bayesian optimization with inequality constraints," in *Proc. Int. Conf. Mach. Learn. (ICML)*, PMLR vol. 32, no. 2, 2014, pp. 937–945.
