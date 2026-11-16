# Hypothesis ledger

This document holds the hypothesis ledger (former Section S9, with Tables S12 and S13) and the power analysis of the pre-specified evaluations (former Section S3.3). Its text comes verbatim from the supplementary material of the paper "Fair adversarial testing of automated vehicles in highway cut-ins: reference-shielded attacks and a capability-controlled evaluation", which is kept in this repository instead of being published with the paper (supplement draft v13 of 2026-10-12, text as of 2026-11-14). Section, figure and table numbers without "S" refer to the paper; numbers with "S" refer to sections and tables of these documents, and README.md lists where each is. The figures of the supplementary material are not reproduced.

Tables S12 and S13 are compiled from the design and result documents of each experiment (read-only check, 2026-10-04). IDs carry the experiment number because several experiments reuse the same names. In the data column, Dev denotes the highD development cases, in-sample (62 cut-ins; "1-D Dev" denotes the 60-case deceleration and cut-out sets); MiTra-P, HS-P and exiD-P denote the pre-specified MiTra, HIGH-SIM and exiD evaluations; "-PH" or "(PH)" marks a post-hoc reuse of hD-R (highD reserved), NGSIM, TGSIM, MiTra, HIGH-SIM (HS) or exiD; hD-behind denotes the 69 highD cases started with the cut-in vehicle behind the follower (B79); exiD-ext denotes the 203 further exiD cases of B82 (post hoc, predictions fixed before the run); and AW-dev and AW-MiTra-PH denote Autoware on the development cases and, post hoc, on MiTra. In the version column, PS denotes the policy stage (shared by v2, v2j and v2s; the shield differs for v2), v2j-S and v2s-S the versions with the 3 m/s severity floor, width-randomised PS the policy stage of a policy trained on randomised vehicle widths (Section S13.5), BL a baseline, Ref the reference driver only, and 1-D the one-dimensional BO agents of the deceleration and cut-out studies. The tables also abbreviate within 37 SUT queries as k37, following H0 as FH0, longitudinal RSS as LRSS, and the FH0 − LRSS detection at k37 as P1; conf. means confirmed (at least 2 of 4 reruns collide), comp. competent SUTs, and att. attacker-caused.

**S9.1 Judged hypotheses**

Table S12. Judged hypotheses

| ID | Statement | Threshold | Fixed before the run? | Data | Version | Result | Verdict |
|---|---|---|---|---|---|---|---|
| B33-R2 | Baselines' att. collisions that R157 §5.2.5.2 explicitly requires the ALKS to avoid | ≤ 10% | yes (analysis post hoc on stored runs) | Dev + hD-R, NGSIM, TGSIM (PH) | BL | 0% (0 of 19,042); lenient reading 0.1% | supported |
| B33-R3 | C avoids §5.2.5.2-mandated grid cut-ins | ≥ 90% | yes (as above) | Dev, TGSIM, hD-R (PH) | Ref | 99.4% (1,714) | supported |
| B39-F2 | Ideal-ADS shield: share of cases attacker-caused (9 SUTs); majority of comp. collisions att. | > 5%; > 50% | yes | Dev | v2 PS | 0.8% of cases; no comp. collisions | rejected |
| B39-F3 | Omniscient shield: ladder range; share of collisions att. | ≤ 20 pp; ≥ 50% | yes | Dev | v2 PS | 0.8% att.; FH0 / H0.5 / H1 fair 88 / 68 / 9% | rejected |
| B39b-F2 | Refinement with the ideal-ADS reference: found failures C cannot avoid; comp. detection | ≥ 20%; ≥ 20% | yes (written after B39's results) | Dev | v2 refinement | 48% of 383; AEB 47 / 48%, RSS 5% (mean 33%) | supported |
| B39b-F3 | Same with the omniscient reference: comp. detection; ladder range | ≥ 50%; ≤ 30 pp | yes (as above) | Dev | v2 refinement | mean 28%; 16 pp (63% of 331 unavoidable for C) | rejected |
| B40-M1 | Margin-escalating refinement: median reaction-time margin, FH0 and IDM | ≥ 1.25 s | yes | Dev | v2 refinement | 1.125 s · 1.25 s | rejected |
| B40-M2 | Rise in the share with margin ≥ 1.25 s over stage 1 | ≥ +20 pp | yes | Dev | v2 refinement | 62.0 → 72.7% (+10.7 pp) | rejected |
| B41-P1 | Discrimination P1 (authority class fixed in advance: median headroom 7.2 m/s, high) | ≥ 50 pp | yes (frozen) | MiTra-P | v2j | 62.0 pp [54.7, 69.2]; flight cluster [46.5, 75.4] | supported |
| B41-P2 | LRSS, lateral RSS, learned delay 0 / 0.3 / 0.6 s, each | ≤ 5% | yes (frozen) | MiTra-P | v2j | LRSS 2.9% [0.6, 5.8]; others 0% | supported |
| B41-P3 | Collision-seeking collisions with AEB H0 / H0.5 / RSS that are att. | ≥ 80% | yes (frozen) | MiTra-P | BL | 99.8% of 950 | supported |
| B42-K1 | CARLA C avoids the 165 robust fair failures (deceleration) | ≥ 90% | yes | 1-D Dev → CARLA | 1-D | 88.5% (C with the AEB layer 100%); rerun with the corrected harness: 100% | rejected (rerun: supported) |
| B42-K2 | CARLA SUT collides on them | ≥ 80% | yes | same | 1-D | 93.3% (rerun 91.5%) | supported |
| B42-K3 | 1-D versus CARLA verdict agreement, each of 8 followers | ≥ 90% | yes | same | 1-D | 91.5–100% (rerun 95.3–100%) | supported |
| B42-K4 | Collision seeking within human braking range: CARLA C also collides | ≤ 5% | yes | same | BL | 0 / 27 | supported |
| B42-K5 | CARLA fair failures of IDM delay 0 and RSS | ≤ 5% | yes | same | 1-D | 0% | supported |
| B43-K1 | CARLA C avoids the 400 robust fair failures (cut-out) | ≥ 90% | yes | 1-D Dev → CARLA | 1-D | 96.5% (rerun 100%) | supported |
| B43-K2 | CARLA SUT collides on them | ≥ 80% | yes | same | 1-D | 91.0% (rerun 90.0%) | supported |
| B43-K3 | Verdict agreement, each of 10 followers | ≥ 90% | yes | same | 1-D | 9 at 96.5–100%; IDM delay 1 s 85.5% (rerun: all ten at 92.6–100%) | rejected (rerun: supported) |
| B43-K4 | Collision seeking within human range: C also collides | ≤ 5% | yes | same | BL | 2 / 63 (3.2%) | supported |
| B43-K5 | RSS fair failures | ≤ 5% | yes | same | 1-D | 0% | supported |
| B43-K6 | All four CARLA C variants avoid the 197 consensus-agent failures | ≥ 90% | yes | same | 1-D | 95.4% (rerun 100%) | supported |
| B44-U1 | Deceleration: cases in which a plan within human authority makes C with the AEB layer collide | ≤ 5% | yes (after a disclosed probe) | 1-D Dev | Ref | 0 / 60 | supported |
| B44-U2 | Cut-out, C | ≤ 10% | yes (as above) | 1-D Dev | Ref | 28 / 60 (47%) | rejected |
| B44-U2b | Cut-out, C with the AEB layer | ≤ 5% | yes (as above) | 1-D Dev | Ref | 13 / 60 (22%) | rejected |
| B45-R1 | v2 reported failures inside the joint realism support | ≥ 90% | yes | Dev | v2 | first reported 86.4%; 90.6% after correcting the feature source | supported, narrowly (cluster CI 87.0–93.8) |
| B45-R2 | Joint-support share lower without G2 than with it | ≥ 20 pp | yes | Dev | PS versus ablations | 45.1% versus 94.0% (49 pp) | supported |
| B45-R3 | Per-feature G2 versus joint verdict agreement | ≥ 90% | yes | Dev | PS | 93.1% | supported |
| B46-J2 | v2j at k37 ≥ v2 − 5 pp for FH0, H0.5, H1, IDM | each | yes | Dev | v2j versus v2 | max −1.0 pp (unified contract: max −2.7 pp) | supported |
| B46-J3 | v2j at k37 for AEB H0 / H0.5, RSS, lateral RSS, ADS 0.5 m | ≤ 3% each | yes | Dev | v2j | all 0% | supported |
| B51-SV1 | Severity floor raises severe-failure detection, weak rule-based SUTs | ≥ +5 pp | yes | Dev | v2j-S | +1.3 pp [0.4, 2.6] | rejected |
| B51-SV2 | Five comp. SUTs, each | ≤ 3% | yes | Dev | v2j-S | all 0% | supported |
| B51-SV3 | Severe-only P1 not lower | ≥ v2j severe-only | yes | Dev | v2j-S | 88.2 ≥ 87.6 pp | supported |
| B52-F1 | Feasibility-referenced RL (omniscient / ideal ADS): collisions its own reference avoids but C cannot | ≥ 30% each | yes (thresholds set after B39b) | Dev | BL | 81.1% / 80.6% | supported |
| B52-F2 | Own-reference "failures" on AEB H0 / H0.5 / LRSS | ≥ 10% each | yes (as above) | Dev | BL | 82.1% / 57.7% | supported |
| B52-F3 | PS discrimination (88.2 pp) minus each condition's | ≥ 20 pp | yes (as above) | Dev | PS versus BL | 86.6 / 86.6 pp | supported |
| B54-A1 | Policy-stage Autoware collisions that are att., per configuration | ≤ 5% | yes | AW-dev | PS | 0 / 166 · 0 / 139 | supported |
| B54-A2 | Collision-seeking Autoware collisions that are att. | ≥ 80% | yes | AW-dev | BL | 98.8% · 100% | supported |
| B54-A3 | Confirmed detection AW-default > AW-r157 (paired lower bound > 0) | > 0 | yes | AW-dev | v2j | 90.9 versus 66.7%; 24.2 pp [15.6, 33.3] | supported |
| B54-A4 | AW-r157 confirmed detection | ≥ 10% | yes (after pilot runs) | AW-dev | v2j | 66.7% [56.5, 76.3] | supported |
| B54-A5 | Rerun collision ratio | ≥ 0.8 | yes | AW-dev | v2j | 0.993 · 0.903 | supported |
| B55-C1 | Lifting the planner's comfort limits lowers confirmed detection (paired upper bound < 0) | < 0 | yes | AW-dev | v2j | −8.6 pp [−16.7, −1.1] | supported |
| B55-C2 | … to at most half of AW-r157 | ≤ 33.3% | yes | AW-dev | v2j | 58.1% | rejected (written interpretation deviates; disclosed) |
| B55-C4 | Policy-stage att. collisions, both variants | ≤ 5% | yes | AW-dev | PS | 0 / 127 · 0 / 11 | supported |
| B56-S2 | Screening keeps detection on 9 detectable SUTs (v2s − v2j, k37) | mean ≥ −3 pp; each ≥ −10 pp | yes | Dev | v2s versus v2j | mean +17.5 pp; min 0.0 pp | supported |
| B56-S3 | 8 comp. SUTs, each (v2s, k37) | ≤ 3% | yes | Dev | v2s | AEB H0 4.8%, learned delay 0 8.1%, others 0–1.6% (9 / 11 consensus-fair) | rejected |
| B57-H1 | Attentive population: median P(avoid) of v2j reported failures | ≥ 0.5 | yes | Dev | PS + v2j | 0.91 [0.90, 0.93] (884) | supported |
| B57-H2 | Same for baseline collisions C cannot avoid | ≤ 0.1 | yes | Dev | BL | 0.0 [0.0, 0.0] (3,733; 20.6% ≥ 0.5) | supported |
| B57-H3 | AUC between the two groups | ≥ 0.9 | yes | Dev | PS + v2j, BL | 0.989 [0.984, 0.994] | supported |
| B58-L1 | Policy-stage att. collisions at latency 0 / 0.3 / 0.6 s | ≤ 5% | yes | AW-dev | PS | 0 / 15 · 0 / 14 · 0 / 22 | supported |
| B58-L2 | Confirmed detection, 0.6 s − 0 s (lower bound > 0) | > 0 | yes (latency model amended before the run) | AW-dev | PS | +3.8 pp [+1.1, +7.0] | supported |
| B58-L3 | 0 < 0.3 < 0.6 s ordering | point estimate | yes (as above) | AW-dev | PS | 0.0 < 0.5 < 3.8% | supported |
| B59-W2′ | Screened − same-day unscreened confirmed detection (primary) | ≥ −3 pp | yes | AW-dev | v2s versus v2j | +2.15 pp [0.0, 4.8] · +11.29 pp [5.4, 17.7] | supported |
| B59-W2 | Screened − first unscreened run (secondary) | ≥ −3 pp | yes | AW-dev | v2s versus v2j | 0.0 pp [−1.6, 1.6] · +10.75 pp [4.3, 17.7] | supported |
| B60-N1 | Policy-stage att. collisions: control / noise / missed detections | ≤ 5% | yes | AW-dev | PS | 0 / 24 · 0 / 29 · 0 / 24 | supported |
| B60-N2 | Missed detections raise confirmed detection (lower bound > 0) | > 0 | yes | AW-dev | PS | +0.5 pp [0.0, 1.6] | rejected |
| B61-K2 | SUTs stronger than C in one factor | k37 ≤ 3% (v2j, v2s) | yes | Dev | v2j, v2s | 0% | supported |
| B61-K3 | v2s at k37 non-decreasing along each weakened factor; weakest level's lower bound > 0 | all three factors | yes | Dev | v2s | reaction 0 → 86.0 → 91.9 → 93.5%; braking 0 → 41.4 → 83.3%; perception 0 → 90.3 → 91.9% | supported |
| B62-Q1 | Retraining against a driver population raises one-query detection on 5 unseen detectable SUTs | ≥ +10 pp, lower bound > 0 | yes | Dev | retrained versus PS | 22.8 versus 24.8%; −2.0 pp [−4.6, +0.4] | rejected |
| B62-Q2 | Weak rule-based SUTs, one query | ≥ −5 pp | yes | Dev | same | +0.5 pp [−1.9, +3.4] | supported |
| B62-Q3 | 8 comp. SUTs, one query | ≤ 3% each | yes | Dev | same | all 0% | supported |
| B64-SS1 | Severe-only P1 with the severity floor | ≥ 50 pp | yes (motivated by B63 on the same data) | MiTra-PH | v2s-S | 53.1 [45.4, 60.7] (v2s severe-only 43.4) | supported on the point estimate |
| B64-SS2 | Severe detection, weak rule-based SUTs: v2s-S − v2s severe-only | ≥ 0 | yes | Dev | v2s-S | +8.5 pp [5.7, 11.4] | supported |
| B66-BL1 | CARS-style + G1: LRSS robust fair failures | ≤ 5% | yes | Dev | BL | 0.0% | supported |
| B66-BL2 | CARS-style + G1: collisions C cannot avoid (weak 4 + LRSS) | ≥ 50% | yes | Dev | BL | 8.2% (8 / 97) | rejected |
| B66-BL3 | Feasibility RL shielded by its own reference: P1 judged by C | < 50 pp | yes | Dev | BL | 1.6 · 1.6 | supported |
| B68-JR | Share of v2s reported failures also avoided by \(C_\mathrm{CARS}\); tiers ≥ 0.8 robust, 0.5–0.8 partly, < 0.5 reading-dependent | tiers | yes (port fixed and re-run after a contract error) | Dev, MiTra-PH | v2s | 81.6% · 82.7% | robust (at the edge) |
| B68-FR | Same with \(\mathrm{FSM}_\mathrm{CARS}\) | tiers | yes | Dev, MiTra-PH | v2s | 99.3% · 98.1% (our FSM 87.4 · 88.6%) | robust |
| B65-HV1 | Attentive population: median P(avoid) of v2s reported failures | ≥ 0.5 | yes | Dev | v2s | 0.91 [0.90, 0.92] (1,089) | supported |
| B65-HV2 | (a) v2s median; (b) baseline collisions C cannot avoid; (c) AUC — all three must hold | ≥ 0.5; ≤ 0.1; ≥ 0.9 | yes | MiTra-PH | v2s, BL | (a) 0.96 [0.95, 0.97]; (b) 0.11 [0.01, 0.24]; (c) 0.977 [0.968, 0.985] | rejected ((b) fails: CARS-style 0.76, Dynasto-style 0.21) |
| B65-HV3 | v2s-S reported failures (policy ≥ 3 m/s + g4ss): median | ≥ 0.5 each | yes | Dev, MiTra-PH | v2s-S | 0.92 [0.90, 0.93] · 0.96 [0.95, 0.97] | supported |
| B67-ST1–ST2 | Random streams 1 and 2: P1 and weak-ladder detection inside stream 0's case-bootstrap interval | inside the interval | yes | Dev | v2s | P1 90.3 · 91.9 in [82.3, 96.8]; weak 4: 75.5 · 75.8 in [67.5, 80.4] | supported |
| B69-T2 | Third-party SUTs (4, pooled): collision-seeking collisions that C cannot avoid | ≥ 80% (≥ 10 collisions) | yes | Dev | BL | 100% (588 / 588) | supported |
| B69-T3 | Third-party SUTs with a reported failure: cases whose v2s report is not consensus-fair | ≤ 5% each | yes | Dev | v2s | SUMO ACC 0, Krauss 1.6, IDM 0; CommonRoad 9.7 [4.8, 15.1] | supported for SUMO; rejected for CommonRoad |
| B70-P1 | Discrimination P1 (authority class fixed in advance: median headroom 12.0 m/s, high) | ≥ 50 pp | yes (frozen) | HS-P | v2s | 51.7 pp [38.3, 65.0]; block cluster [41.4, 60.9] | supported on the point estimate |
| B70-P2 | LRSS, lateral RSS, learned delay 0 / 0.3 / 0.6 s: reported failures not consensus-fair, each | ≤ 5% | yes (frozen) | HS-P | v2s | 0% each | supported |
| B70-P3 | Collision-seeking collisions with AEB H0 / H0.5 / RSS that are att. | ≥ 80% | yes (frozen) | HS-P | BL | 100% of 232 | supported |
| B70-K1 | Original P2: same five SUTs, all reported failures, each | ≤ 5% | yes (frozen) | HS-P | v2s | LRSS 0, lateral RSS 0, learned delay 3.3 / 0 / 0% | supported |
| B70-K2 | Severe-only P1 (failures closing below 3 m/s not counted) | ≥ 50 pp | yes (frozen) | HS-P | v2s | 48.3 [35.6, 60.6] | rejected |
| B70-K2-S | Same for the severity-floor arm | ≥ 50 pp | yes (frozen) | HS-P | v2s-S | 51.7 [38.3, 65.0] | supported on the point estimate |
| B71-P1 | Discrimination P1 (authority class fixed in advance: median headroom 11.9 m/s, high) | ≥ 50 pp | yes (frozen) | exiD-P | v2s | 77.5 pp [70.0, 85.0]; recording cluster [69.6, 85.2]; site cluster [67.5, 85.9] | supported |
| B71-P2 | LRSS, lateral RSS, learned delay 0 / 0.3 / 0.6 s: reported failures not consensus-fair, each | ≤ 5% | yes (frozen) | exiD-P | v2s | lateral RSS 0.8% [0, 2.5]; others 0% | supported |
| B71-P3 | Collision-seeking collisions with AEB H0 / H0.5 / RSS that are att. | ≥ 80% | yes (frozen) | exiD-P | BL | 100% of 788 | supported |
| B71-K1 | Original P2: same five SUTs, all reported failures, each | ≤ 5% | yes (frozen) | exiD-P | v2s | LRSS 3.3, lateral RSS 0.8, learned delay 3.3 / 0 / 0% | supported |
| B71-K2 | Severe-only P1 (failures closing below 3 m/s not counted) | ≥ 50 pp | yes (frozen) | exiD-P | v2s | 66.7 [58.6, 74.2] | supported |
| B71-K2-S | Same for the severity-floor arm | ≥ 50 pp | yes (frozen) | exiD-P | v2s-S | 75.6 [67.8, 82.8] | supported |
| B72 | HIGH-SIM extension, 84 new cases: FH0 detection for cut-ins that began less than 1.5 s ahead minus that for cut-ins that began 4 s or more ahead | ≥ 40 pp | yes (the extension was decided after the first HIGH-SIM results) | HS, 84 new cases (PH) | v2s | 100% of 25 versus 13.3% of 15, 86.7 pp | supported |
| B73-GA1 | Competent-SUT reported failures (seed level) whose contact persists with both bodies shrunk by 5 cm per side | ≥ 80% | yes | Dev, MiTra-PH, HS-P, exiD-P (stored failures) | v2s | 86.2% of 564 (10 cm: 75.5%) | supported |
| B74-CB1 | CARS-style reward trained with G1 from the agent's initialisation: robust P1 at one query | < 50 pp | yes | Dev | BL | 1.1 [0.0, 3.2] | supported |
| B74-CB1b | Same, trained from scratch | < 50 pp | yes | Dev | BL | 0.5 [0.0, 1.6] | supported |
| B75-RS1 | Agent's P1 searched again with the joint region at its 95th percentile | ≥ 50 pp | yes | Dev | v2s | 82.8 [73.1, 91.4]; filter estimate 11.3 | supported |
| B75-RS2 | Same with half the G2 tolerance (0.5 IQR) | ≥ 50 pp | yes | Dev | v2s | 88.2 [80.1, 95.2]; filter estimate 74.2 | supported |
| B76-FS1 | C margin 1.0 m versus 0.25 m: lower median closing speed of the failures of following H0.5 and H1, BC-highD and ADS 2.0 m | lower | yes | Dev | v2s refinement | 3.23 versus 2.46 m/s | rejected |
| B76-FS2 | Same: lower severe-only detection of these SUTs | lower | yes | Dev | v2s refinement | 28.2 versus 31.5% | supported |
| B77-JR | Share of v2s reported failures that \(C_\mathrm{CARS}\) also avoids on the pre-specified data, in the tiers of B68-JR | tiers | yes (tiers of B68, plan fixed before computing) | HS-PH, exiD-PH | v2s | 76.6% · 83.5% | partial (HIGH-SIM) · robust (exiD) |
| B77-FR | Same with \(\mathrm{FSM}_\mathrm{CARS}\) | tiers | yes (as above) | HS-PH, exiD-PH | v2s | 99.6% · 98.9% | robust |
| B78-NB1 | Agent's P1 searched again with the cut-in vehicle's braking bounded by the natural 1st percentile | ≥ 50 pp | yes | Dev | v2s | 88.7 [80.7, 95.7]; filter estimate 58.1 | supported |
| B78-NB2 | Same with the natural 5th percentile | ≥ 50 pp | yes | Dev | v2s | 74.2 [62.9, 85.5]; filter estimate 5.4 | supported |
| B79-P1 | Started behind: rule-based SUTs' fair failures that brake after the crossing (GP-BO and random pooled) | ≥ 80% | yes | hD-behind | v2s refinement | 69.3% (61 of 88); 95.5% cross slower | rejected |
| B79-P2 | Started behind: BC-highD fair failures (GP-BO) | ≥ 30% | yes | hD-behind | v2s refinement | 81.2% | supported |
| B79-P3 | Longitudinal RSS fair failures in every condition | ≤ 5% | yes | hD-behind, Dev | v2s refinement | 0–1.6% | supported |
| B79-P4 | Weak rule-based SUTs: screened GP-BO above screened random search | higher | yes | hD-behind | v2s refinement | following H0 46.4 versus 8.7; H0.5 17.4 versus 0; IDM 46.4 versus 8.7 | supported |
| CARLA-PC | Three figure episodes with CARLA physics: the SUT collides in (a) and (b), and C avoids all three | all | yes | Dev, hD-behind | v2s refinement | kept in all three (earlier harness); rerun with the corrected harness: the SUT avoided (a) by 0.36 m, C avoided all three | supported (rerun: rejected) |
| CARLA-RP1–RP4 | Stored failures replayed unchanged in CARLA remain fair failures of FH0: development cases (RP1); each other dataset (RP2); each dataset with the distance-preserving lateral mapping (RP3); with the Tesla, within 5 pp of the Mustang (RP4) | ≥ 90%; ≥ 90% each; ≥ 90% each; ± 5 pp | yes | Dev; TGSIM, MiTra, HS, exiD (PH) | v2s | 90.6%; 85.1, 72.4, 80.6, 83.5%; 53.3–66.7%; 59.6 versus 90.6% | RP1 supported; RP2–RP4 rejected |
| CARLA-TW1–TW2 | Matched twin versus CARLA's C on the stored attacks, for the proportional mapping, the distance-preserving mapping and the Tesla: collision-verdict agreement above the default twin's (TW1); attacks it passes that CARLA's C cannot avoid (TW2) | ≥ 95% and above default; ≤ 3% | yes | Dev, TGSIM, MiTra, HS, exiD (PH); Tesla: Dev | v2s | 96.0 vs 91.7, 96.1 vs 42.9, 95.5 vs 96.9%; 0, 0, 0% | TW1 supported for both mappings, rejected for the Tesla; TW2 supported |
| CARLA-TW3–TW4 | Matched twin versus CARLA's SUT: agreement for the four weak SUTs above the default twin's (TW3); FH0 attacks that remain fair failures in the matched twin with C's 0.5 m margin transfer to CARLA, five datasets pooled (TW4) | ≥ 90% and above default; ≥ 90% | yes (designed after the first replays) | Dev, TGSIM, MiTra, HS, exiD (PH) | v2s | 90.6 vs 68.5%; 93.7% of the 782 of 980 kept (84.1% of all) | supported |
| CARLA-CL1–CL5 | Closed loop with the shield in the matched twin: att. collisions (CL1); AEB H0, LRSS, lateral RSS and Traffic Manager (CL2); repeated seed-1 run, verdict agreement (CL3); without the shield, att. collisions (CL4); att. collisions without shield ≥ default twin ≥ matched twin (CL5) | 0; ≤ 3% each; ≥ 95%; ≥ 1; order | yes | Dev | PS | 0 of 1,488; 0% each; 496 of 496; 99; 99 ≥ 4 ≥ 0 | supported (all five) |
| CARLA-SR1–SR3 | Refinement inside CARLA, screened in the matched twin: successes that CARLA's C cannot avoid (SR1); AEB H0, LRSS, lateral RSS and Traffic Manager (SR2); repeated searches of FH0 and BehaviorAgent normal, success agreement and rate difference (SR3) | ≤ 5%; ≤ 3% each; ≥ 85% and ≤ 10 pp | yes | Dev | v2s refinement | 0 of 158; 1.6, 1.6, 0, 0%; 95.2%, −1.6 pp each | supported (all three) |
| CARLA-CS1–CS3 | Collision-only search inside CARLA: cases with an SUT collision, every SUT (CS1); collisions of AEB H0, LRSS, lateral RSS and Traffic Manager that are att. (CS2); of these, collisions meeting the success contract (CS3) | ≥ 70%; ≥ 80%; ≤ 5% | yes | Dev | BL | 82.3–87.1%; 93.3% of 209; 0% | supported (all three) |
| CARLA-TL1–TL3 | Tesla closed loop with the shield in the Tesla-matched twin: att. collisions at most 1% of runs and fewer than with the Mustang-matched twin (TL1); FH0 robust fair failures (TL2); AEB H0, LRSS, lateral RSS and Traffic Manager (TL3) | ≤ 1% and fewer; ≥ 75%; ≤ 3% each | yes | Dev | PS | 0 versus 0; 67.2% [57.5, 75.3]; 0% each | TL1 rejected (both 0); TL2 rejected; TL3 supported |
| CARLA-WR1–WR3 | Policy trained on vehicle widths U(1.9, 2.7) m, three seeds: Tesla FH0 at least 8 pp above the agent's policy with att. at most 1% (WR1); Mustang FH0 not more than 5 pp below, att. at most 1% (WR2); AEB H0, LRSS, lateral RSS and Traffic Manager on both vehicles (WR3) | ≥ +8 pp; ≥ −5 pp; ≤ 3% each | yes (exploratory, after a one-seed probe) | Dev | width-randomised PS | +5.9 pp (73.1 versus 67.2%), 0; −1.7 pp (84.9 versus 86.6%), 0; 0% each | WR1 rejected; WR2, WR3 supported |
| CARLA-DC | Deceleration and cut-out plans rerun with the corrected harness: the two summary values within 5 points of the stored ones | ± 5 pp | yes | 1-D Dev → CARLA | 1-D | SUT collides on the deceleration failures 91.5% (stored 93.3%); C keeps its cut-out avoidances 100% (stored 96.5%) | supported |
| B80-AX1–AX5 | Autoware on exiD: attacker-caused shares, confirmed detection and its difference, reproduction (Section S13.6) | as B54 A1–A5 | yes | exiD-PH | v2s | policy stage 0 of 246 and 0 of 190 attacker-caused; baseline 99.3 and 99.6%; confirmed 73.3 versus 51.1%, difference 22.2 [16.1, 28.6]; reproduction 0.91 and 0.66 | AX1–AX4 supported; AX5 rejected for AW-r157 |
| B81-K1–K3 | Controls derived from C on exiD (Section S13.7) | as B61 K1–K3 | yes | exiD-PH | v2s | C and the stronger variants 0; reaction 75.0 / 80.8 / 81.7, braking 43.1 / 68.1, perception 78.6 / 81.7 (weakest lower bounds 74.2, 59.2, 74.2) | supported (all three) |
| B82-E1 | exiD extension: discrimination P1 | ≥ 50 pp | yes | exiD-ext | v2s | 75.4 [69.5, 81.3]; recording cluster [69.0, 81.6] | supported |
| B82-E2 | Same: LRSS, lateral RSS, learned delay 0 / 0.3 / 0.6 s: reported failures not consensus-fair, each | ≤ 5% | yes | exiD-ext | v2s | LRSS 2.0%, others 0 | supported |
| B82-E3 | Same: collision-seeking collisions with AEB H0 / H0.5 / RSS that are att. | ≥ 80% | yes | exiD-ext | BL | 99.9% of 1,420 | supported |
| B82-E4 | Same: original P2, all reported failures, each of the five SUTs | ≤ 5% | yes | exiD-ext | v2s | LRSS 7.9% (16 cases), learned delay 3.9 / 0 / 0%, lateral RSS 0 | rejected (LRSS) |
| B82-E5 | Same: P1 without failures closing below 3 m/s; with the severity floor | ≥ 50 pp | yes | exiD-ext | v2s, v2s-S | 70.3 [64.2, 76.4]; 75.9 [70.0, 81.3] | supported |
| B83-AB1 | Controlled ablation, collision reward instead of the reward based on C's verdict: one-query P1 below the final agent's | ≥ 50 pp lower | yes (2026-11-14, set after B61) | Dev | collision-reward PS versus PS | 3.8 versus 87.1 pp, difference −83.3 [−91.9, −72.6] | supported |
| B83-AB2 | Same arm: att. policy-stage collisions with the shield · without the shield, collisions with AEB H0 / H0.5 / LRSS that are att. | 0 · ≥ 80% | yes (2026-11-14) | Dev | collision-reward PS | 0 of 550 · 100% (collisions in 86.0 / 86.0 / 85.0% of the cases) | supported |
| B83-AB3 | Shield off when testing: att. policy-stage collisions · change in weak-ladder one-query detection · BC-highD one-query detection | > 0 (with the shield 0) · < 5 pp · lower | yes (2026-11-14, set after B25) | Dev | PS without versus with the shield | 150 of 782 (with the shield 0) · 2.9 pp · 37.1 → 17.2% | supported |
| B83-AB4 | Commitment (G1) off in all phases: LRSS one-query detection counted without G1 · one-query P1 under the full contract | ≥ 50% · below the final agent's | yes (2026-11-14, set after B23 E2, seed 1: 74–76%) | Dev | PS without G1 versus PS | 27.4% · 6.5 versus 87.1 pp | rejected (second part supported) |
| B83-AB5 | G2 off in the reward: fair failures that meet G2 and joint realism · one-query P1 | ≥ 20 pp below the final agent's · below the final agent's | yes (2026-11-14, set after B45-R2) | Dev | PS without G2 versus PS | G2 39.6 versus 97.5%, joint 43.5 versus 94.2% · 27.4 versus 87.1 pp | supported |
| B83-AB6 | Every arm's weak-ladder detection at k37, which joins its policy stage to the stored refinement, at least that of the refinement alone | ≥ refinement alone | yes (2026-11-14) | Dev, HS-PH, exiD-PH | every arm with the v2s refinement | refinement alone 70.6 · 44.2 · 66.0%, no arm below | supported |

Several descriptive companions were not judged: B47 (J2/J3 expectations on hD-R, NGSIM, TGSIM: held, max −2.7 pp, max 2.5%), B52 F1–F3 post hoc on hD-R, TGSIM, MiTra (all above the thresholds), B54 §10 (A1–A5 on MiTra post hoc: 0 / 248 · 0 / 155; 98.8% · 100%; 22.3 pp [16.9, 27.9]; 36.2%; 0.96 · 0.86), B55-C3 (−55.9 pp [−66.7, −45.2]), B56 §8 (MiTra and TGSIM comparisons), B60-N3 (+1.1 pp [0.0, 2.7]), B64-SS3 (competent-SUT detection and consensus), the B65 later-perception bracket (1.5 m: median 0.38 · 0.44, read with a rule fixed before the run) and the B68 P1 values when both readings must avoid a failure, and B74 CB2–CB4 (attacker-caused share, shielded arms, realism).

**S9.2 Implementation checks** (outcomes guaranteed by construction, reported to verify the implementation)

Table S13. Implementation checks

| Check | Experiment | Value |
|---|---|---|
| Policy-stage att. collisions, highway-env | B25, B26 (v2); B39 F1; B41 S1 (v2j, MiTra); B70 S1 (HIGH-SIM); B71 S1 (exiD) | 0 for every SUT, dataset and seed |
| Unshieldable decisions | number check of 2026-09-30 §5-3 | 0 in 17,187 stored episodes (preferred action changed 6,095 times); MiTra 0 in 8,772 |
| Shield on other policies | B61; B62 Q4 | collision seeking + shield 0 / 3,162; known-capability SUTs 0; retrained policy 0 / 3,162 |
| SUT = C is never detected | B61 K1 | 0% at 1 / 6 / 37 queries, v2j and v2s |
| Inadmissible SUT queries, screened refinement | B56 S1; B70 S12; B71 S12 | 0 / 28,410 (Dev); 0 / 73,404 (MiTra-PH); 0 / 35,847 (TGSIM-PH); 0 / 27,599 (HS-P; unscreened 20,511 / 34,103); 0 / 54,125 (exiD-P; unscreened 32,218 / 62,019). Unscreened: 20,007 / 34,077, 39,872 / 87,685, 23,030 / 42,587 |
| Same on Autoware | B59 W1 | 0 / 915 after correcting a rounding of the joint threshold (2.77 for 2.7714) that counted one; unscreened 693 / 1,302 |
| Refinement successes inside the joint support | B46 J1 | 190 / 190 |
| Severity floor respected | B64; B70 | minimum closing speed 3.00 m/s (318 Dev, 874 MiTra successes); 3.02 m/s (186 HS-P successes); 3.00 m/s (613 exiD-P successes) |
| Agent code equals the exact synthesis | B46; B49 C1 | 170 / 170; 3,162 / 3,162 |
| Third-party SUTs (B69 T1) | B69 | natural cut-ins 0 / 62 contacts for each of the four; policy-stage attacker-caused collisions 0; inadmissible queries executed 0 / 9,821 |
| Stored runs reproduced | B56 S5; B40 M0; B51, B64 pre-checks | 1,054 / 1,054 (MiTra 2,924 / 2,924; TGSIM 1,360 / 1,360); 1,054 / 1,054; 51 / 51 each |
| Stored verdicts reproduced by replay | B57; B32; B45; B47; B65 C1–C2; B68 C1 | 6,196 / 6,196; 0 mismatches; 4,752 episodes and 203 / 203; 4,129 / 4,129, 398 / 398, 1,830 / 1,830; 6,450 / 6,450 and 5,082 / 5,082 marks identical to B57; 6,450 / 6,450 |
| Cross-platform replay (premise vi) | B46 §0 | 167 / 170 identical; 2 verdicts flipped |
| Contact-geometry replays | B73 I1–I3 | contact in 1,096 / 1,096 replays; unshrunk replays equal the stored runs and their pre-contact trajectories (1,080 / 1,080 each) |
| Policy stage rerun beside a new baseline | B74 I1 | 3,162 / 3,162 episodes with identical action sequences |
| CARLA harness actuation after the correction | CARLA experiments (Section S13.5): actuation probes; 20 stored attacks × C and three SUTs × two vehicles | reaction-phase deceleration −0.40 ± 0.02 m/s² for a 0.4 m/s² request; emergency plateau 7.55–7.61 m/s² for 7.59; mean braking error within ±0.03 m/s² (earlier harness +0.6 to +0.8); start-speed error within ±0.003 m/s (earlier −0.05 to −0.09) |

The Autoware policy-stage zeros (B54-A1, B55-C4, B58-L1, B60-N1) were judged hypotheses, because Proposition 1's premises hold only approximately on a wall-clock stack; they are listed in Section S9.1, not here.

**S9.3 Deviations and caveats disclosed in the result documents.** For B33 the hypotheses were fixed before the results, but the analysis is post hoc on stored runs and reused data. B39b was written after B39's results, before its own run, and B52's thresholds were set with B39b's results known. B41 had no deviations; the flight-cluster interval of P1 and the upper bound for LRSS (5.8%) cross their thresholds, and post hoc, P1 without contacts closing below 3 m/s is 45.9 [39.0, 52.7]. B44 was designed after a disclosed probe that found the 4.99 m/s² gap, and "full regulatory model" was renamed "C with the chosen AEB implementation". B45 R1 was first reported as rejected (86.4%) because the refinement's features were taken from the SUT run, and it was corrected to 90.6%. The motivation of B46 rested on B45's erroneous value, so J2 and J3 were rechecked under the unified contract. In B51, v2j-S was added to the exiD draft as a descriptive arm although the design allowed it only if SV1 was supported; the v2s draft dropped it, and B64 re-tested the floor for the final agent.

For B54 A4, the threshold was set after pilot runs, and a latency-counter defect was found after the run; counting stalled episodes as misses gives 85.5% / 59.7%, with A3 and A4 unchanged. The first run of B55 was invalid (duplicate chains) and was re-run from scratch; its written interpretation departs from the rule fixed before the run, and excluding one instance with a start-speed anomaly gives 2.4% versus 58.1%. In B58, uncompensated latency was replaced by compensated latency after a smoke test, before the run. In B59, W2′ was made primary before the run, but the unscreened arm ran separately the same evening, and the W1 rounding error was fixed. In B61, policy-stage outputs were read early for B62, after K1–K3 were fixed. B62 was trained on the CPU server instead of the GPU runners. B64 SS1 was judged on reused MiTra cases, motivated by B63's 43.4 on the same data, and its lower bound (45.4) is below 50; five of its cells were re-rounded from exact counts on 2026-10-04 without a verdict change. The first start of B65 stopped at validation because the server archive was incomplete, and it was restarted two minutes later from a complete archive, with no results from the first start. In B73, the first run read the contact geometry from the states that highway-env records after its impact correction, in which the bodies no longer overlap; before any analysis output it was rerun with the pre-impact states (D1), and contacts registered one tick before overlap were kept in the persistence shares and left out of the type statistics (D2). In B77, the composition of the reference readings was aligned with B68 after the first computation, which then reproduced the stored P1 values (45.0 and 78.1), and the Autoware closing speeds were added after the plan. After the review of v9, B73's stored rows were re-read at the interpolated first touch and with only SUT-front contacts counted (post hoc, descriptive), and the cut-in style was characterised (post hoc, before the B78 re-search was designed). In B68, the first run of the port of \(C_\mathrm{CARS}\) braked at 7.5929 m/s², above the simulator's contract of 7.59 m/s², and stopped in every replay that reached full braking; the port was corrected to 7.59 m/s², and only that reading was re-run.

In the CARLA experiments of Section S13.5, a complete first run with the earlier harness was superseded after its analysis revealed the actuation defects of Section S4.3. The hypotheses and decision rules were kept, the harness was corrected and every experiment was rerun from scratch. The third defect was found during that rerun, which was then restarted with all three corrections; its partial results are kept but not used. The searches inside CARLA were moved to a second machine after 42 searches of one arm had run on the first. These were set aside, and both arms ran from scratch on the second machine. Their repeat was started early on another CARLA server of that machine, with settings and random streams unchanged. The SUT-side twin analysis and the re-check of stored attacks were designed after the first replays had been seen, with their criteria fixed before the computation. The width-randomised policy was first probed with one seed and without criteria; criteria were fixed before its seeds 2 and 3 were trained and all three seeds were run. The first Tesla hypothesis could not be met, because the twin matched to the Mustang also produced no attacker-caused collision on the Tesla.

Not compiled here are B30 and B34 (cited only through B42–B44), the pre-specified evaluations of the earlier versions B9, B13 and B28 (only their data are reused), and experiments not cited in the manuscript. B16, B21, B23, B25, B26, B32, B36 and B63 had no thresholds fixed in advance and are descriptive.

## S3.3 Power of the pre-specified evaluations

The power of the pre-specified evaluations was simulated before the data by resampling cases (10,000 draws). With MiTra-like cases, P1 passes with probability 0.75–0.93 for 20–150 cases, while K2 (0.27–0.05) and K1 (0.28–0.00) mostly fail; with highD-like cases all three pass with probability near 1.
