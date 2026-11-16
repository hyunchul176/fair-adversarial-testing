# fairtest: fair adversarial testing of automated vehicles in highway cut-ins

This repository is the companion code of the paper. It contains the code of the agent, the systems under test (SUTs),
the reference drivers, the shield, the searches, the analyses and the dataset extractors, the trained models, the case
identifiers and derived case files, and the Autoware and CARLA files. The documentation that the paper cites (method
details, the hypothesis and design-decision ledgers, the version map and further results) is in `docs/`, which starts
with `docs/README.md`.

This release is a cleaned copy of the code used for the paper. The package was renamed to `fairtest`, the comments are
in English, and internal notes and private paths were removed. The cleaning does not change results on the checked
cases: twenty highway-env episodes, five MiTra cases from `cases/derived/mitra`, each run with its stored warm-start
plan against four targets (following H0 and H0.5, the reference driver C and the RSS SUT), gave bit-identical episode
records before and after the cleaning.

The frozen protocols of the pre-specified evaluations, the design documents of the follow-up experiments, the
data-access records, the freeze records with their code hashes and the frozen commits are kept by the authors and are
available on request (`docs/release_notes.md`).

## Contents

| Folder | What it holds |
|---|---|
| `code/fairtest/` | All experiment code: `runner.py` (episodes, replay gate), `cc_target.py` (the UN R157 competent and careful reference driver C and its variants), `ref_models.py` (FSM, RSS reference driver, longitudinal-RSS SUT), `sut_targets.py`, `b18_bc.py` (BC-highD), `b17_rl.py` (policy training and evaluation), `b23_search.py` (refinement, with the agent's screened GP-BO `bo_warm_g4s`), `b29_consensus.py` (held-out references), the extractors `cases.py` (highD), `exid.py`, `highsim.py`, `mitra.py`, `tgsim.py` and `ngsim.py`, the Autoware harness `aw_*.py`, the CARLA harness `carla_*.py` (with the actuation correction of `docs/method_details.md`, Section S4.3) and one analysis module per experiment (`bNN_analysis.py`). |
| `code/src/`, `code/scripts/` | The simulation environment and safety metrics that the experiment code imports, and six earlier analysis modules that it still imports. |
| `models/` | `agent_policy/s1–s3.zip`: the agent's PPO policy (three seeds). Learned baselines: collision-seeking with commitment, CARS-style and Dynasto-style rewards (three seeds each). `learned_suts/`: the PPO SUTs (perception delay 0, 0.3 and 0.6 s, lane-assignment perception 1.5, 2.0 and 2.5 m). `bc_highd/bc_human.pkl`: the behaviour-cloning SUT (BC-highD). `agent_policy_width_randomised/s1–s3.zip`: the agent's recipe trained with the vehicle width drawn from 1.9 to 2.7 m (`docs/additional_results.md`, Section S13.5). |
| `data_shared/` | The contracts that the code reads (`base_protocol.json`, `protocol_b1.json`, `protocol_b2_action.json`) and `natural_reference.json`, the realism reference: six summary features of each of the 463 highD natural cut-ins of the calibration and development windows, without trajectories. |
| `cases/identifiers/` | highD (development, calibration and reserved test), exiD and HIGH-SIM cases as identifiers only: recording, location, cut-in vehicle, follower, crossing frame and times. These datasets are obtained from their providers. The extractors regenerate the exiD and HIGH-SIM cases and the highD development and calibration cases. The reserved highD test cases are listed by identifier. |
| `cases/derived/` | Derived case files for MiTra (CC BY 4.0), TGSIM I-294 and NGSIM US-101. |
| `autoware/` | The parameter overrides of AW-default, AW-r157 and the cause-test and perception variants, each with its `CHANGES.json` (the parameters changed against the release), the straight two-lane lanelet2 map, and the launch, restart and ROS-bridge scripts. Autoware Universe release: Docker image `ghcr.io/autowarefoundation/autoware:universe-20260420`. |
| `carla/` | The CARLA runs of the agent (`docs/method_details.md`, Section S4.3, and `docs/additional_results.md`, Section S13.5): run scripts with the working folder `FAIRTEST_WORK`, per-experiment scripts and analyses, the harness change `HARNESS_H2.diff` and the result summaries (`runs/`). |
| `docs/` | The documentation that the paper cites: method details, additional results, the hypothesis and design-decision ledgers, the version map, the release notes and tables as CSV files. |
| `MANIFEST.sha256` | SHA-256 of every file in the repository. |

## Environment

Python 3.11 with `numpy==2.4.4 gymnasium==1.2.3 highway-env==1.10.2 pygame==2.6.1 pandas==3.0.6 scipy==1.17.1
cma==4.0.0 scikit-learn==1.7.2 stable_baselines3==2.8.0 torch==2.5.1 shapely==2.1.2 matplotlib==3.11.2`, and CARLA
0.9.16 (Town04) for the CARLA experiments. NumPy 2 is required: the replay gate checks bit-exact commands, and NumPy
1.x type promotion differs by about 1e-8.

```bash
cd code
export SDL_VIDEODRIVER=dummy OMP_NUM_THREADS=1 PYTHONPATH=$PWD
export FAIRTEST_DATA=/path/to/case_data          # a case_pool.json plus the files of data_shared/
export FAIRTEST_DRL_SUT=$PWD/../models/learned_suts/0.zip FAIRTEST_DRL_SUT_AT="$PWD/../models/learned_suts/{k}.zip"
export FAIRTEST_BC_MODEL=$PWD/../models/bc_highd/bc_human.pkl
python -m fairtest.runner validate-b5            # replay gate (needs the B5 replay states, see below)
```

Quick check without any licensed data: point `FAIRTEST_DATA` at `../cases/derived/mitra` and run the agent's policy
stage on MiTra, for example `FAIRTEST_SHIELD_MARGIN=0.5 python -m fairtest.b17_rl eval ../models/agent_policy/s1.zip
OUT.jsonl --partition all --workers 8 --commit --shield-window --suts following:0,rss_sut:0`.

## Regenerating the cases

```bash
python -m fairtest.mitra MITRA_DIR EVENTS.jsonl            # MiTra (Data_T1.zip .. Data_T9.zip)
python -m fairtest.tgsim TGSIM_DIR EVENTS.jsonl            # TGSIM I-294
python -m fairtest.highsim HIGH_SIM_full.csv EVENTS.jsonl  # HIGH-SIM
python -m fairtest.exid extract DATA_DIR MAPS_DIR EVENTS.jsonl
```

The case-pool step of each evaluation (`python -m fairtest.external_bundle pool`) turns the events into
`case_pool.json`. The run scripts `run_b70.sh` (HIGH-SIM) and `run_b71.sh` (exiD) show the complete pipelines of two
pre-specified evaluations, from extraction to analysis. Compare the regenerated cases with `cases/identifiers/`.

## The agent on a case pool

```bash
python -m fairtest.b17_rl eval ../models/agent_policy/s1.zip POLICY_S1.jsonl --partition development --workers 24
python -m fairtest.b23_search REFINE.jsonl --cont BOUNDARY.jsonl --partition development --workers 24   # bo_warm_g4s
python -m fairtest.b56_analysis OUT.json --jobs JOBS_DIR --stored POLICY_RULE.jsonl POLICY_LEARNED.jsonl --run REFINE.jsonl
```

Each module's docstring gives its full command line. The hypotheses and criteria of the pre-specified evaluations and
of the follow-up experiments are listed in `docs/hypothesis_ledger.md`, with whether each was fixed before the run. The
results are reported in `docs/additional_results.md`.

## Third-party code and data

- The CARS reference-model port (`b68_cars_refs.py`) follows RoboSafe-Lab/CARS-code (commit 6e32a26), Apache License 2.0.
- MiTra derived cases: CC BY 4.0. Please cite the MiTra dataset.
- highD, exiD and HIGH-SIM are not redistributed here. The highD-derived files (`natural_reference.json`, `protocol_b1.json` and `protocol_b2_action.json` in `data_shared/` and their copies in `cases/derived/`, and `models/bc_highd/`) are derivative works under the highD terms, for non-commercial use only. Please cite the highD dataset.
- The B5 replay states used by the replay gate are derived from highD cases and are available to highD licensees on request.
- Autoware Universe and CARLA are used as released.

## License

The code is licensed under the Apache License, Version 2.0 (`LICENSE`). Third-party code and data keep their own
terms, listed in `NOTICE` (MiTra-derived cases: CC BY 4.0, highD-derived files: non-commercial use under the highD
terms).
