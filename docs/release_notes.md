# Release notes

This document records how the published repository relates to the records of the study: which records the authors keep and make available on request, the name of the code package, the protocols of the pre-specified evaluations, the outcomes of earlier pre-registrations, rules that were later superseded, and the files derived from highD.

## Records available on request

The frozen protocol and design documents, the data-access records, the complete freeze records with their lists of code hashes, and the frozen commits of the pre-specified evaluations are kept by the authors. They are not part of this repository and are available on request.

## Package name

The code package is published as `fairtest`. It was renamed for publication, so the records above refer to it by its earlier name.

## Protocols of the pre-specified evaluations

| Experiment | Cases | What the protocol fixed | Fixed on | Protocol |
|---|---|---|---|---|
| B9 | 148 reserved highD cut-ins | hypotheses H1–H6, the methods, the SUTs and reference drivers, the case selection, the budgets and the bootstrap, with the hashes of the code and of the contracts, before any reserved case was read | 2026-09-25 | available on request |
| B13 | 47 NGSIM US-101 cut-ins | hypotheses H1–H5 and H7–H9, seven SUTs, the continuous search with a budget of 36 queries and the bootstrap, with the code hashes, before any NGSIM event was extracted | 2026-09-25 | available on request |
| B13, addendum A | the same | a general falsification baseline with hypotheses H10, H10b and H11 | 2026-09-25, 17:57 KST | available on request |
| B28 | 80 TGSIM I-294 cut-ins | hypotheses T1–T7 for the agent of that time (v1), its baselines and 17 SUTs, the authority classes, the extractor and the analysis, with data and code hashes | 2026-09-27 | available on request |
| B41 | 172 MiTra cut-ins | hypotheses P1–P3 and the secondary statistics S1–S6 for the earlier version (v2j), at most 20 cases per recording, with data and code hashes | 2026-09-28 | available on request |
| B70 | 60 HIGH-SIM cut-ins | the hypotheses, thresholds and analysis code of the exiD protocol, unchanged, for the agent (v2s), at most 60 cases, with the code commit and the data hashes | 2026-10-05, 15:59 | available on request |
| B71 | 120 exiD cut-ins | hypotheses P1–P3, K1 and K2 and the severity-floor arm for the agent (v2s), at most four cases per recording and 120 in total, with the code commit and the hashes of every exiD file read | 2026-10-06, 05:00 and 05:12 | available on request |

The addendum to B13 was registered after the B13 search had run and before its outputs were opened. The design documents of the follow-up experiments are also available on request. Four of them (B56, B57, B60 and B61) were fixed before their runs but do not record the commit that holds them. The B58 design was revised after a smoke run, before the main run, with its hypotheses, thresholds and interpretation rules unchanged.

## Earlier pre-registrations

The authors' records include the following earlier pre-registrations, which are summarised here. The list is not complete.

| Experiment | Hypotheses | Outcome |
|---|---|---|
| B9, 148 reserved highD cut-ins | H1–H6 | five of six supported (per hypothesis: not recorded) |
| B13, 47 NGSIM US-101 cut-ins | H1–H5, H7–H9 | supported: H1 for the weak SUTs, H3, H4, H7 and H8. Rejected: H1 for the strong SUTs, H2, H5, the second part of H8, and H9 |
| B13, addendum A | H10, H10b, H11 | all supported |
| B28, 80 TGSIM I-294 cut-ins | T1–T7 | all supported |

The records attribute the rejections in B13 to the low speeds of NGSIM, where the adversary's speed floor of 15 m/s left little room to slow down (median headroom 1.8 m/s, against 18.1 m/s on highD). The data of B9, B13 and B28 are reused post hoc in these documents.

## Superseded rules

The B51 and B56 designs (available on request) state that the earlier version (v2j) remains the final agent and that all external evaluations used it. This was superseded on 2026-10-01, when the screened version (v2s) became the final agent (design_decisions.md, row 25), as the paper's Section 5.1 discloses.

## Data derived from highD

`data_shared/protocol_b1.json` and `data_shared/protocol_b2_action.json` each list 16 inputs with the initial gap and speeds of highD cut-ins and their highD event identifiers. They are derived from highD and are shared under the highD terms, for non-commercial use only.
