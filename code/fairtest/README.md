# fairtest: fair and naturalistic adversarial testing of cut-in responses

A fair adversary makes a system under test (SUT) fail in a way that a **UN R157 competent and careful (CC) driver
would have avoided**. The same adversarial plan is run against the SUT and against the CC reference driver:

- **Fair failure**: the SUT collides and the CC driver does not.
- **Attacker-at-fault collision**: both collide.

Realism is scored against highD natural cut-ins. The score is the maximum exceedance beyond the natural p1–p99 range,
in IQR units.

Simulator: highway-env 1.10.2, two-vehicle cut-in, 10 Hz, 12 s. The generator authority is inherited from the B2
action contract: |a| ≤ 3 m/s², jerk ≤ 2 m/s³, speed 15–40 m/s, quintic lateral planner, 2 Hz decisions.

## Module map

| Area | Modules |
|---|---|
| Environment and targets | `env.py` (`B7Env`, target registry), `cc_target.py` (CC driver, `cc_r157_strict` variant), `ref_models.py` (FSM, RSS reference drivers, longitudinal-RSS SUT), `sut_targets.py` (following + AEB), `plans.py` (199-plan grid, adapter of the fixed B5 policy) |
| Data | `cases.py` (exposed highD case pool, reserved recordings blocked), `ngsim.py` (NGSIM US-101 extractor and cohort selection), `features.py` (cut-in features, `Realism`) |
| Execution | `runner.py` (episodes, landscape CLI, B5 replay gate) |
| Callable agent | `fair_agent_v2.py` (`FairAdversaryV2(policy).attack(case_id, (sut, H))`: CC boundary plan → shielded guarded policy (1 query) → guarded robust GP-BO (≤ 36) → failure certificate with held-out reference drivers, R157 grade, decision margin, guards and failure-mode features. It reproduces the stage-wise B26/B29 composition exactly, 170/170 development checks) |
| Online agent (B13 era) | `fair_agent.py` (`FairAdversary.attack(case, sut, h)`: CC reference sweep → learned or R157-boundary top-1 → fairness-constrained CMA-ES. It reproduces the offline hybrid, 24/24 exposed cases) |
| Adversaries | `boundary_planner.py` (R157 boundary, zero SUT queries), `guided_search.py`, `b8_selector.py` (learned selector features), `b11_search.py` (continuous CMA-ES, `cont_search`), `b13_falsify.py` (generic falsification baseline), `b14_realism_search.py` (plausibility-constrained falsification, fairness + realism adversary) |
| Analyses of pre-specified evaluations | `b9_analysis.py` (highD reserved recordings), `b13_analysis.py` (NGSIM US-101) |
| Post hoc and sensitivity | `strict_reeval.py`, `strict_analysis.py` (perception-release sensitivity), `b14_analysis.py`, `severity_dial.py` |
| Figures | `figures.py`, `figures_b9.py`, `figures_b13.py`, `figures_case_study.py` |
| Development only | `b8_env.py`, `train_b8.py` (PPO environment, unused), `b10_selector_study.py`, `b12_tuning.py` (weak design, not claimed) |

## Evaluation status

| Study | Data | Status | Protocol |
|---|---|---|---|
| B7 landscape | highD exposed (266 cases) | development | n/a |
| B9 | highD reserved recordings (148 cases) | **pre-specified, used once** | available on request |
| B10–B12 | highD exposed | development | n/a |
| B13 | NGSIM US-101 (47 cases) | **pre-specified, used once** | available on request, with addendum A |
| After B13 | highD exposed, B9 and NGSIM | development and post hoc | n/a |

The later pre-specified evaluations (B28, B41, B70 and B71) are listed with their protocols in `docs/release_notes.md`.
The hypotheses are listed in `docs/hypothesis_ledger.md` and the results are reported in `docs/additional_results.md`.

## Reproduce

Use the environment pinned in the root README and run from `code/`:

```bash
pip install numpy==2.4.4 gymnasium==1.2.3 highway-env==1.10.2 pygame==2.6.1 pandas==3.0.6 scipy==1.17.1 cma==4.0.0 scikit-learn==1.7.2 stable_baselines3==2.8.0 torch==2.5.1 shapely==2.1.2 matplotlib==3.11.2
export SDL_VIDEODRIVER=dummy OMP_NUM_THREADS=1 PYTHONPATH=$PWD FAIRTEST_DATA=/path/to/case_data
```

1. **Replay gate**: every run starts with it. It reproduces the 32 conditions of the B5 run (tolerance 1e-9).
   ```bash
   python -m fairtest.runner validate-b5
   ```
2. **Grid landscape**: 200 policies per case. `--shard k/n` keeps `case_id % n == k`.
   ```bash
   python -m fairtest.runner landscape OUT.jsonl --targets following:0,cc_r157:0 --policies all --workers 30
   ```
3. **Continuous fair adversary**: warm starts come from `warm_starts.json` in the data directory.
   ```bash
   python -m fairtest.b11_search OUT.jsonl --workers 30 --budget 36 --suts following:0,following:1,rss_sut:0
   ```
4. **Generic falsification baseline**:
   ```bash
   python -m fairtest.b13_falsify run OUT.jsonl --workers 30 --b11 CONT.jsonl
   python -m fairtest.b13_falsify analyze OUT.jsonl RESULT.json
   ```
5. **Analysis of the pre-specified NGSIM evaluation**:
   ```bash
   python -m fairtest.b13_analysis ROWS_A.jsonl,ROWS_B.jsonl CONT.jsonl CASE_POOL.json MODELS.pkl OUT.json
   ```
6. **Unit tests**: three of the eleven tests need the highD extraction or the B5 reference extract, which are not
   distributed.
   ```bash
   python -m unittest fairtest.tests.test_b7
   ```

The data directory (`FAIRTEST_DATA`) holds the contracts of `data_shared/`, a case pool (`case_pool.json`), warm starts
(`warm_starts.json`), `natural_reference.json` and, for the replay gate, the B5 reference extract
(`b5_reference_states.json.gz`, available to highD licensees on request). `cases/derived/mitra/` is such a directory
for MiTra, without the B5 extract. Without `FAIRTEST_DATA`, the code reads the local evidence folder of the study from
`FAIRTEST_EVIDENCE` (default `evidence`). That folder is not distributed.

## Known caveats

- **Reference-model exploitation**: continuous optimisers can exploit quirks of the reference driver. Three scenarios exploited the CC perception latch. Judge continuous-search failures with `cc_r157_strict` or with the CC ∩ FSM ∩ RSS consensus.
- **Scope of the RSS SUT**: `rss_sut` implements longitudinal RSS only.
- **Generator authority**: the speed floor of 15 m/s makes low-speed congested traffic (NGSIM) nearly unattackable. Treat it as a scope condition.
