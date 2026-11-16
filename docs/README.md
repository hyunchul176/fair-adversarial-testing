# Documents of the paper

These documents hold the supplementary material of the paper "Fair adversarial testing of automated vehicles in highway cut-ins: reference-shielded attacks and a capability-controlled evaluation", kept in this repository as records instead of being published with the paper, and details that the paper gives only in summary. The text is verbatim from the supplement draft v13 (2026-10-12; text as of 2026-11-14) and from an earlier, longer draft of the paper, without HTML comments and without the figures of the supplement, and with the corrections of 2026-11-14 and 2026-11-16 listed in the last section.

## Contents

- `hypothesis_ledger.md`: The hypothesis ledger (former Section S9, with Tables S12 and S13) and the power analysis of the pre-specified evaluations (former Section S3.3) (5134 words, 0 references).
- `design_decisions.md`: The design-decision ledger (former Section S8, Table S11) (2216 words, 0 references).
- `version_map.md`: The version map (former Section S7, Table S10) (458 words, 0 references).
- `method_details.md`: The method details: policy, shield and refinement (former Sections S1.2–S1.4, with Table S1 and Algorithm S1), data, systems under test, baselines and terms (S2.2–S2.5, Tables S3 and S4) and the CARLA set-up (S4.3) (6347 words, 20 references).
- `additional_results.md`: The full results behind the numbers the paper reports or summarises: baselines, reference readings, MiTra, TGSIM, HIGH-SIM, exiD and attack authority (former Sections S3.1, S3.2 and S3.4–S3.6, Table S5), Autoware and the third-party SUTs (S4.1, S4.2), ablations, controls and cost (S5, Tables S6 and S7), supplementary analyses (S6, Tables S8 and S9), the development results by SUT (S11, Table S14) and the follow-up experiments (S13, Table S15), plus sections added on 2026-11-14 and 15: post-hoc analyses requested by an external review, the attack shapes of the paper's former Fig. 7 run with CARLA physics, and a controlled ablation of the policy stage (19510 words, 0 references).
- `release_notes.md`: Release notes: the records that the authors keep and make available on request, the name of the code package, the protocols of the pre-specified evaluations, earlier pre-registrations, superseded rules and the files derived from highD (656 words, 0 references).
- `tables/reference_readings_tableS5.csv`: Table S5. Reported failures under other readings of the reference driver (12 rows). Note. *Reported failures also avoided*: share of the final agent's reported failures that the reading also avoids. The other columns give detection (%) within 37 queries when a failure counts only if that reading also avoids it, computed post hoc on the fixed stored searches; a search re-run with the other reading in its screening could take other paths, so these values bound neither its detection nor P1. Development in-sample; MiTra post hoc; HIGH-SIM and exiD data of the pre-specified evaluations, read post hoc.
- `tables/budget_detection_tableS6.csv`: Table S6. Detection of the screened search alone and of the agent by SUT-query budget (25 rows). Note. Detection in % of cases. The search alone runs without the policy; the agent makes its policy query first. Last column: the smallest budget at which the search alone reaches the policy's one-query rate. Development in-sample; TGSIM and MiTra post hoc; HIGH-SIM and exiD data of the pre-specified evaluations, read post hoc.
- `tables/cost_tableS7.csv`: Table S7. Cost per detection or per attack (6 rows).
- `tables/development_detection_tableS14.csv`: Table S14. Detection of the agent by SUT on the development cases (15 rows). Note. % of 62 cases, seed mean (in-sample), within 1, 6 and 37 SUT queries, with the case-bootstrap 95% interval and the seed s.d. at 37 queries. The last two columns count only failures closing at 3 m/s or more: *severe only* drops the light contacts the agent reports, and *floor* is the agent with the 3 m/s severity floor in its search (v2s-S).
- `tables/carla_table_full.csv`: Table D1 of additional_results.md, the paper's CARLA table (Table 6) in full: The agent and baselines in CARLA on the development cases (12 rows). Note. % of the 62 development cases (in-sample). Closed-loop rows: one shielded policy query per case and SUT, mean over three policy seeds, 1,488 runs per row. Their robust fair failures are judged by CARLA's reference driver (following H0 in the first row: 84.9% when G2 and joint realism are also applied). Search rows: one search of at most 36 SUT queries per case and SUT; successes meet the success contract with the reference driver's conditions checked in the screening twin. Brackets: case-bootstrap 95% intervals. Matched twin: the highway-env replica with the reference driver's perception point and the bodies placed as in CARLA; default twin: highway-env's own geometry. All vehicles are Ford Mustangs except in the Tesla rows; the width-randomised policy was trained on vehicle widths drawn uniformly from 1.9 to 2.7 m. Last column: runs or search successes in which CARLA's reference driver also collided or, for the baselines, the share of their SUT collisions that it could not avoid; the baselines collided in 81.3–85.4% of the runs.
- `tables/detection_all_suts_figS8.csv`: Values of former Fig. S8: detection of the 17 main SUTs within 37 SUT queries (17 rows).

## Internal codes

These documents keep the internal codes of the project, which the paper does not use. The table maps them to what they are and to the sections of the paper that report them (none: the paper does not report them). The data and version abbreviations of the hypothesis tables, such as MiTra-P, HS-P, exiD-P, hD-behind, exiD-ext, PS and BL, are defined at the top of hypothesis_ledger.md. As in the paper, pre-specified means that the protocol was frozen before the data were opened, an internal pre-registration and not an entry in a public registry. The code package is published as `fairtest` and was renamed for publication.

| Code | What it is | Paper |
|---|---|---|
| v2s | the agent: the policy stage with the screened refinement | Sections 4 and 6 |
| v2s-S | the agent with the 3 m/s severity floor | 4.4, 6.1, 6.5 |
| v2j, v2j-S | the earlier version: the same policy stage with the unscreened refinement, and the same with the severity floor | 5.1, 6.1, 6.5 to 6.7 |
| v1, v2 | earlier prototypes, before the guards in the refinement (v1) and before joint realism (v2) | none |
| Dev, Dev-62 | the 62 highD development cases (in-sample) | 5.1, Table 1 |
| B9-148, hD-R | the 148 reserved highD cases, reused post hoc | 5.2, Table 1 |
| B9, B13, B28 | pre-specified evaluations of earlier designs on the reserved highD, NGSIM and TGSIM cases | 5.2, Table 1 |
| B7, B14, B16, B17, B21, B24 to B26, B29, B32, B33, B36, B39, B39b, B40, B45 to B47, B49 | design experiments on the development cases: reference driver, guards, shield, refinement, reward, joint realism and success contract | design choices only (Sections 3 to 5), see design_decisions.md |
| B23 | baselines, and the guard ablation of the earlier unshielded policy | 6.2, Fig. 3 |
| B41 | pre-specified MiTra evaluation of the earlier version | 6.1, Table 3 |
| B30, B34, B42 to B44 | deceleration and cut-out studies in one dimension (1-D), and their re-execution in CARLA | none |
| B51, B63, B64 | severity floor (earlier version, the agent) and detection without light contacts | 4.4, 6.1, 6.5 |
| B52, B66, B74 | feasibility-referenced and CARS-style baselines, and checks of them | 6.2 |
| B54, B55, B59 | Autoware on the development cases: first run, cause test, screened refinement | 6.6, 6.7 |
| B56 | screened against unscreened refinement | 6.6 |
| B57, B65 | human-like driver population (earlier version, the agent) | 6.4 |
| B58, B60, B62 | Autoware with perception latency, noise or missed detections, and retraining against a driver population | none (removed on 2026-11-16) |
| B61, B81 | controls derived from C (development, exiD) | 6.3 |
| B67 | two further random streams of the refinement | none |
| B68 | reference readings of the CARS code | 6.4, Table 4 |
| B69 | SUMO and CommonRoad SUTs | 6.7 |
| B70, B72 | pre-specified HIGH-SIM evaluation of the agent, and its extension to all 144 cases | 6.1, Table 3, 6.5, Fig. 6 |
| B71, B82 | pre-specified exiD evaluation of the agent, and the 203 further exiD cases | 6.1, Table 3 |
| B73 | contact geometry | 6.3 |
| B75, B76, B78, B79 | searches again under stricter realism, with other margins and severity floors, with braking bounds and from behind the follower | 6.5 |
| B77 | learning versus search, and the reference readings, on the pre-specified data | 6.4, Table 4, 6.6, Fig. 7 |
| B80 | Autoware on exiD | 6.7 |
| B83 | controlled ablation of the policy stage | 6.6, Table 5 |
| CARLA-… | CARLA experiments of the agent | 6.7, Table 6 |

## Where each part of the supplementary material is

| Part | Now | Note |
|---|---|---|
| Overview | left out |  |
| S1.1 What "fair" and "realistic" mean | left out | the paper keeps a shortened form of the sentence that cited it, in Section 7.3 |
| S1.2 Policy | method_details.md |  |
| S1.3 Shield | method_details.md | the paper adds one sentence to its limitations |
| S1.4 Refinement, Table S1, Algorithm S1 | method_details.md |  |
| S2.1 Platforms, Table S2 | left out | Section 5 of the paper describes the platforms |
| S2.2 Data | method_details.md |  |
| S2.3 Systems under test | method_details.md |  |
| S2.4 Baselines, Table S3 | method_details.md | the paper adds one sentence |
| S2.5 Terms, cost accounting, seeds and intervals, Table S4 | method_details.md | the paper adds two or three sentences to Section 5 |
| S3.1 Baselines, reference readings and human-like drivers, Table S5 | additional_results.md | the paper adds a small table in Section 6; Table S5 also as tables/reference_readings_tableS5.csv |
| S3.2 MiTra and TGSIM | additional_results.md |  |
| S3.3 Power of the pre-specified evaluations | hypothesis_ledger.md |  |
| S3.4 HIGH-SIM | additional_results.md |  |
| S3.5 exiD | additional_results.md |  |
| S3.6 Attack authority, Fig. S1 | additional_results.md | the paper adds one sentence; the figure is left out |
| S4.1 Autoware Universe | additional_results.md |  |
| S4.2 Third-party SUTs, Fig. S2 | additional_results.md | the figure is left out |
| S4.3 CARLA (set-up, harness correction, matched twin, re-checks, repeatability, top views), Fig. S3 | method_details.md | the paper adds the definition of the matched twin; Fig. S3 and its paragraph are left out |
| S5 Ablations, controls, cost and extended discussion, Tables S6 and S7, Fig. S4 | additional_results.md | the paper adds two to four cost numbers; Tables S6 and S7 also as CSV; the figure is left out |
| S6 Supplementary analyses, Tables S8 and S9, Figs. S5 and S6 | additional_results.md | the figures are left out |
| S7 Version map, Table S10 | version_map.md |  |
| S8 Design-decision ledger, Table S11 | design_decisions.md |  |
| S9 Hypothesis ledger, Tables S12 and S13 | hypothesis_ledger.md |  |
| S10 Data and code availability | paper | merged into the paper's 'Data and code availability' |
| S11 Detailed development results, Table S14, Figs. S7 and S8 | additional_results.md | the paper's Fig. 5 shows the 17 main SUTs; Table S14 and the values of Fig. S8 also as CSV; the figures are left out |
| S12 Further related work | left out |  |
| S13 Follow-up experiments, Table S15 | additional_results.md | their criteria and verdicts are also rows of the hypothesis ledger |

## Paragraphs from the earlier draft of the paper

Each document lists them in its section "Moved from the paper", grouped by the section of the earlier draft they come from; the paper has since reordered its Section 6. The headings, and the second column below, also give the section of the paper that now holds each paragraph's content.

| Document | Section of the earlier draft, and of the paper now | Paragraph |
|---|---|---|
| method_details.md | 4.2 Learning a closed-loop adversarial policy, now Section 4.2 | Every 0.5 s the policy \(\pi_\theta(a_t \mid o_t)\) receives a 16-dimens… |
| method_details.md | 4.2 Learning a closed-loop adversarial policy, now Section 4.2 | We train the policy with PPO [Schulman et al. 2017], implemented in stab… |
| method_details.md | 4.4 Reference-guided boundary plan and screened refinement, now Section 4.4 | **Open-loop plan space and initialisation.** The refinement searches a f… |
| method_details.md | 4.4 Reference-guided boundary plan and screened refinement, now Section 4.4 | **Screening before the query and candidate generation.** Before an SUT q… |
| method_details.md | 4.4 Reference-guided boundary plan and screened refinement, now Section 4.4 | **Objective and cost of GP-BO.** The refinement aims to reduce the SUT's… |
| method_details.md | 5.2 Scenario cohorts and experimental environments, now Section 5.2 | HIGH-SIM and exiD cut-ins are extracted with the same highD event rule, … |
| method_details.md | 5.2 Scenario cohorts and experimental environments, now Section 5.2 | For exiD we use release v2.1, 93 recordings (16.1 hours) from seven moto… |
| method_details.md | 5.2 Scenario cohorts and experimental environments, now Section 5.2 | CARLA 0.9.16 [Dosovitskiy et al. 2017] provides a second execution envir… |
| method_details.md | 5.6 Pre-registered hypotheses and decision rules, now Section 5.6 | **Sample size and power.** The numbers of cases were set by the protocol… |
| method_details.md | 5.6 Pre-registered hypotheses and decision rules, now Section 5.6 | **Statistical decisions and evidence status.** Policy-stage results are … |
| additional_results.md | 5.4 Baselines and component analyses, now Section 5.4 | **Learning efficiency and further variants.** We compare, by SUT-query b… |
| additional_results.md | 6.3 Detection ordered by designed capability, now Section 6.3 | Fig. 4 shows these differences on a single attack, in development case 2… |
| additional_results.md | 6.3 Detection ordered by designed capability, now Section 6.3 | The same controls kept this order on the 120 exiD cut-ins. This was a po… |
| additional_results.md | 6.3 Detection ordered by designed capability, now Section 6.5, without the margin result | A follow-up experiment that changed the margin C itself must keep reran … |
| additional_results.md | 6.4 Failures of competent SUTs and specificity, now Section 6.3 | The failing AEB cases began at a median natural headway of 0.99–1.20 s, … |
| additional_results.md | 6.5 Sensitivity to the reference, realism and scenario mix, now Section 6.5 | To check this reading further, we extended the evaluation, after seeing … |
| additional_results.md | 6.5 Sensitivity to the reference, realism and scenario mix, now Section 6.5 | To check whether the slower cut-ins are an artefact of the initial place… |
| additional_results.md | 6.6 External software and simulators, now Section 6.7 | **Autoware on the development cases.** To examine whether the agent appl… |
| additional_results.md | 6.6 External software and simulators, now Section 6.7 | To separate first reports from reproduced failures of this non-determini… |
| additional_results.md | 6.6 External software and simulators, now Section 6.7 | A failure that the earlier version's refinement found against AW-r157 on… |
| additional_results.md | 6.6 External software and simulators, no longer in the paper (removed on 2026-11-16) | We also injected faults into the object list of the configuration with b… |
| additional_results.md | 6.6 External software and simulators, now Section 6.7 | **Autoware on exiD (post-hoc reuse).** We applied Autoware to the 120 ex… |
| additional_results.md | 6.6 External software and simulators, now Section 6.7 | Table D1 (that draft's Table 4, CARLA, the paper's Table 6) in full, with its note |
| additional_results.md | 6.6 External software and simulators, now Section 6.7 | **CARLA: a wider vehicle.** A wider SUT lowered detection even when the … |
| additional_results.md | 6.6 External software and simulators, now Section 5.2 | These CARLA results were obtained after correcting the harness (Section … |
| additional_results.md | 6.7 Learning, screening and query cost, now Section 6.6 | At the full budget of 36–37 queries, the agent and the screened GP-BO al… |
| additional_results.md | 6.7 Learning, screening and query cost, now Section 6.6 | **Contributions of screening and the search model.** On the development … |
| additional_results.md | 6.7 Learning, screening and query cost, now Section 6.6 | Screening, which checks admissibility before the query, and the search m… |
| additional_results.md | 6.7 Learning, screening and query cost, now Section 6.6 | Computational cost, too, has to be assessed separately for SUT queries a… |

## Left out of the moved text

- `method_details.md`: figure line "*Fig. S3. Two further attacks on development case 207, seen from above in CARLA*"
- `additional_results.md`: figure line "*Fig. S1. Detection of the weakest SUT by the adversary's speed headroom (earlier version, post hoc)*"
- `additional_results.md`: figure line "*Fig. S2. Four third-party SUTs in their shipped configurations (development cases)*"
- `additional_results.md`: figure line "*Fig. S4. Detection against the SUT-query budget on five datasets*"
- `additional_results.md`: figure line "*Fig. S5. Contact geometry of the agent's reported failures (seed level)*"
- `additional_results.md`: figure line "*Fig. S6. Cut-in style of the agent's reported failures against natural highD cut-ins (development cases)*"
- `additional_results.md`: figure line "*Fig. S7. Detection by SUT for the agent and the earlier version (development and MiTra cases)*"
- `additional_results.md`: figure line "*Fig. S8. Detection of all main systems under test and of the controls derived from the reference driver*"
- `method_details.md`: paragraph "Fig. S3 shows two further attacks on this case. In Fig. S3(a), shown at 6.9 s, when C come…"

## Cross-references to parts that are not carried over

The moved text keeps its references to the former supplement where these documents hold the part referred to. The references below pointed to a figure or a part that is not in these documents; they were reworded on 2026-11-14 to point to a section or table of these documents, to a CSV file in `tables/` or to plain words, and sentences that only described a figure were dropped.

- `hypothesis_ledger.md`: none
- `design_decisions.md`: none
- `version_map.md`: Fig. S1, Fig. S2, Fig. S3, Fig. S4, Fig. S5, Fig. S6, Fig. S7
- `method_details.md`: Fig. S3; Section S4, now Section S4.1
- `additional_results.md`: Fig. S1, Fig. S2, Fig. S4, Fig. S5, Fig. S6, Fig. S7, Fig. S8; Sections S1 and S3, now Sections S1.4 and S3.2

On the same day, the references to the paper's sections, figures and tables were updated to its numbering as of 2026-11-14, in which Section 6 was reordered and the CARLA table became Table 6 (Table 7 from 2026-11-15, when the ablation table was added, and Table 6 again from 2026-11-16), and the references to the paper's earlier Fig. 7, which posed three attack shapes from stored states, were reworded, because Fig. 7 then showed two shapes that hold when run with CARLA physics. Also on that day, the median acceleration before the crossing of −1.34 m/s² in Section S13.4 and in the moved paragraph on cut-ins started behind the follower was attributed to following H0 in the screened GP-BO search, the group for which it was computed; it had read as the median of all 88 fair failures of the rule-based followers.

On 2026-11-16 the paper removed that figure of two attack shapes (then Fig. 7) and the table from SUT query to reported failure (then Table 4), and the references to the paper were updated again. The budget figure is now Fig. 7 and the CARLA table Table 6. The contents of both removed items are kept in additional_results.md, in the sections on the attack shapes of the former Fig. 7 and on the post-hoc analyses requested by an external review.

Later on 2026-11-16 the documents were aligned further with the paper. "Prospective" and "pre-registered" were changed to the paper's "pre-specified" where they describe the paper's evaluations, and hypotheses of follow-up runs are said to be fixed before the run, as in the paper. Quoted titles of the earlier draft keep "pre-registered". The earlier draft's CARLA table in additional_results.md became Table D1, so that it cannot be read as the paper's Table 4. The headings of the moved paragraphs and the table above now name the paper's current sections. The hypothesis ledger gained rows for the prediction of B72 and the six predictions of the controlled ablation (B83). Section S3.4 now relates its 144-case HIGH-SIM values to those of the paper's Section 6.5. The table of internal codes was added and the word counts above were updated.

In a third pass on 2026-11-16, release_notes.md was added. It lists the records that the authors keep and make available on request, the protocols of the pre-specified evaluations and the outcomes of earlier pre-registrations. The hypothesis ledger gained rows for the tiers of B77 on HIGH-SIM and exiD. References to files that are not published here now name the experiment and say that the record is available on request, and one alternative in row 18 of the design-decision ledger that referred to such a record was removed. The code package is named as published, and the word counts above were updated.
