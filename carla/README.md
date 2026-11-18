# CARLA runs of the agent

This folder holds the CARLA experiments of the agent itself, run without retraining after the harness correction
described in `docs/method_details.md`, Section S4.3. The design of each experiment and its decision criteria were fixed
before its run. The criteria are listed in `docs/hypothesis_ledger.md` (rows starting with CARLA), and the results are
reported in `docs/additional_results.md`, Section S13.5. The run scripts and analyses are those that were run, with
the package renamed to `fairtest`, the comments in English and the private working folder replaced by
`FAIRTEST_WORK`. The runs used the package's code with the four files below and the package's models (identical
files).

## The corrected harness

The correction changes four files in `code/fairtest/`. `HARNESS_H2.diff` is the change against the earlier harness.

- `carla_replay.py`: actuation `h2`, the default. `FAIRTEST_CARLA_ACT=v1` restores the earlier actuation.
  - Requests above −1 m/s² are tracked exactly in CARLA's constant-velocity mode.
  - Stronger braking uses the brake map minus the disturbance measured over the previous braking step.
  - The PI correction is dropped.
  - The commanded cruising speed is corrected with the integral of the measured speed error, from the pre-roll on.
- `carla_dec.py`, `carla_cutout.py`: the same actuation and pre-roll correction for the CARLA re-executions of the
  deceleration and cut-out scenarios.
- `env.py`: the matched twin.
  - `FAIRTEST_VEH_L` sets the body length and `FAIRTEST_CC_PERCEIVE` sets the perception point of the reference
    driver C, next to the existing `FAIRTEST_VEH_W`.
  - With these variables unset, highway-env behaves as before. The replay gate (`python -m fairtest.runner
    validate-b5`) passes on all 32 conditions with zero state difference.

`common.sh` sets the vehicles and the twins. All runs use `FAIRTEST_CARLA_ACT=h2` and `FAIRTEST_CARLA_PI=0,0`.

| Setting | CARLA vehicle | Brake map | Twin |
|---|---|---|---|
| `mustang`, `geom_m` | `vehicle.ford.mustang` | `1.65,0.75,9.787` | matched: C perceives at 1.251 m, bodies 2.166 × 4.718 m |
| `tesla`, `geom_t` | `vehicle.tesla.model3` | `2.5,0.56,9.97` | matched: 1.251 m, bodies 2.469 × 4.79 m |
| `geom_u` | n/a | n/a | default twin: 1.095 m, bodies 2 × 5 m |

## Running

The scripts work in one folder, given by the environment variable `FAIRTEST_WORK` (default `$HOME/fairtest_work`),
with these contents:

- `code/`: this package's `code/`.
- `models/` and `models_extra/`: the policies.
- `data/`: a case pool with the files of `data_shared/` (see the root README).
- `jobs/`: the stored attacks that the agent's highway-env runs produced.
- `runs/` and `logs/`.

`PY` in `common.sh` sets the Python interpreter. CARLA 0.9.16 is started from `$HOME/carla_0916`, Town04 off screen,
one server per port. The two scripts in `topview_physics/` set their own `PY` and work in
`$FAIRTEST_WORK/topview_physics`, which holds their `logs/`, `phys/` and `bev/` folders and the stored trajectories
that they replay. To run from this package:

```bash
cd carla
export FAIRTEST_WORK=$PWD
ln -s ../code code; ln -s ../models models; mkdir -p logs models_extra
for s in 1 2 3; do ln -s ../../models/agent_policy_width_randomised/s$s.zip models_extra/fair_guarded2_wr_s${s}_1500000.zip; done
# set PY and the CARLA path in common.sh, provide data/ and jobs/, then for example
sh run_B.sh
```

The case pools and stored attacks are not included. They contain states derived from highD, exiD and HIGH-SIM, which
are not redistributed. The per-run logs are left out for the same reason. To regenerate the stored attacks, run the
agent on the case pools (root README, "The agent on a case pool"). Note that `runs/` here holds only the result
summaries.

## Experiments

The results are reported in `docs/additional_results.md`, Section S13.5, and the criteria in
`docs/hypothesis_ledger.md`.

| Experiment | What | Result file |
|---|---|---|
| H2 probes | actuation probes and the 20-attack validation of the corrected harness (`h2_probe.py`, `h21_probe.py`, `h2_validate.py`) | `runs/H21_validate_mustang.json`, `runs/H21_validate_tesla.json` |
| E1, E6, E11 | replays of stored attacks: development cases, the other datasets, repeat | `runs/E1_RESULT.json`, `E6_RESULT.json`, `E11_RESULT.json` |
| E16v2, E16v2-SUT | does the matched twin predict CARLA's verdicts (C, SUT) | `runs/E16v2_RESULT.json`, `E16v2_SUT_RESULT.json` |
| E12, E15 | distance-preserving mapping, Tesla Model 3 | `runs/E12_RESULT.json`, `E15_RESULT.json` |
| E9 | geometry against physics | `runs/E9_RESULT.json` |
| E2, E3 | closed loop with the shield, and its repeat | `runs/E2_RESULT.json` |
| E13 | search for SUT collisions without regard to fairness | `runs/E13_RESULT.json` |
| E5 | without the shield, and the shield in the default twin | `runs/E5_RESULT.json` |
| E10 | closed loop with G2 and joint realism | `runs/E10_RESULT.json` |
| E8 | learned baselines | `runs/E8_RESULT.json` |
| E17, E18 | Tesla closed loop, width-randomised policies | `runs/E17_RESULT.json`, `E18_RESULT.json` |
| E7 | deceleration and cut-out scenarios, rerun | `runs/E7_b42_reanalysis.json`, `E7_b43_reanalysis.json` |
| E4, E14 | the screened refinement inside CARLA, and its repeat | `runs/E4_RESULT.json`, `E14_RESULT.json` |

The experiments ran as chains, one CARLA server per port.

- **First chains:** `run_A.sh`, `run_B.sh` and `run_C.sh`.
- **Split across two machines.** `run_A2.sh` and `run_B2.sh` ran on the first machine. The searches E4 and E14 ran on
  a second machine with the `search` function of `common.sh`. That machine's chain script (`run_T.sh`) is not
  included.
- **Rerun.** All chains were rerun from scratch with the final harness.
- **Width-randomised policies:** `run_E18.sh`, `run_E18b.sh` and `run_E18M_s1.sh`.
- **Analyses:** each analysis script (`e*_analyze.py`) writes the result file of its experiments.

## Width-randomised policies

`models/agent_policy_width_randomised/s1–s3.zip` correspond to `fair_guarded2_wr_s{1,2,3}_1500000.zip` in the run
scripts.

- **Training:** the agent's recipe and starting checkpoints, with the vehicle width drawn uniformly from 1.9 to 2.7 m
  in every episode. The command is `b17_rl train --mode fair_guarded2 --steps 1500000 --envs 15 --checkpoint 250000`
  with `FAIRTEST_VEH_W_RANGE=1.9,2.7`.
- **Seeds:** seed 1 is the variant trained during development. Seeds 2 and 3 were trained for these runs.

## Top views and the physics check

`topview_physics/` re-records, with the corrected harness, the CARLA top views of Figs. 1 and 6 of the paper
(`docs/method_details.md`, Section S4.3) and of a former supplementary figure (S3). It also runs the three attack
shapes of a former Fig. 7 with CARLA physics (`docs/additional_results.md`, Section S13.5).

- `run_select_physics.sh` chooses the attack of Fig. 6 and runs the physics checks.
- `run_render.sh` renders the frames with `fairtest.carla_bev`.
- `make_search_row.py` rebuilds the plan of the former Fig. S3(b) from the search of E4.
- The stored trajectories that these scripts replay are not included, for the reason given above.
