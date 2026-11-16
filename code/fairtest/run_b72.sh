#!/bin/sh
# B72 HIGH-SIM extension (B72 design, fixed before the run, available on request): the frozen
# B70 steps on the B70 events without the 60-case cap (144 cases; the first 60 are the B70 cases). Run from runtime-b72 =
# runtime-b70 (frozen code) + b72_pool.py, b72_analysis.py and this script. Every step
# logs and leaves a .done marker; a rerun resumes after the last completed step.
#   flock -n /tmp/fairtest_b72.lock nice -n 10 taskset -c 0-23 sh run_b72.sh
set -u
C=$HOME/fairtest; R=$C/runtime-b72/code; O=$C/b72; D=$C/b72_data; H=$C/highsim/HIGH_SIM_full.csv
EVSHA=0a7099197a5675336b28f823b14464d35fe2851b16f21df06463d3c81b82c7a7
W=${B72_WORKERS:-24}
mkdir -p $O; cd $R || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 SDL_VIDEODRIVER=dummy PYTHONWARNINGS=ignore
export FAIRTEST_DRL_SUT=$C/b25/drl_models/0.zip FAIRTEST_DRL_SUT_AT="$C/b25/drl_models/{k}.zip"
P=$C/env/bin/python
EV="$P -m fairtest.b17_rl eval"
SUTS=following:0,following:0.5,following:1,reference_idm:0,following_aeb:0,following_aeb:0.5,rss_sut:0,bc_human:0,ads@0.5:0,ads@2.0:0,rss_lat_sut:0,drl_sut:0,drl_sut@3:0,drl_sut@6:0,drl_sut@p1.5:0,drl_sut@p2:0,drl_sut@p2.5:0
OURS1=$C/b17/fair_guarded2_s1/ppo_fair_guarded2_s1_1500000.zip
log() { echo "$(date '+%F %T') $*" >> $O/chain.log; }
has() { [ -e $O/$1.done ]; }
fin() { touch $O/$1.done; log "$1 done"; }
die() { log "$1 FAILED"; exit 1; }
log start
# 0. frozen artefacts: server hashes of the models, learned SUTs, BC human, the HIGH-SIM file and the natural reference
#    (b13_data, copied into the data folder by the pool step; the git archive has no data folder); the frozen judge
#    reproduces its B5 conditions
if ! has hashes; then
  sha256sum $H $OURS1 $C/b23/fair_guarded2_s2/ppo_fair_guarded2_s2_1500000.zip $C/b23/fair_guarded2_s3/ppo_fair_guarded2_s3_1500000.zip \
    $C/b17/collide_commit_s1/ppo_collide_commit_s1_1500000.zip $C/b23/collide_commit_s2/ppo_collide_commit_s2_1500000.zip \
    $C/b23/collide_commit_s3/ppo_collide_commit_s3_1500000.zip $C/b23/cars_s1/ppo_cars_s1_2250000.zip $C/b23/cars_s2/ppo_cars_s2_2250000.zip \
    $C/b23/cars_s3/ppo_cars_s3_2250000.zip $C/b23/dynasto_s1/ppo_dynasto_s1_2250000.zip $C/b23/dynasto_s2/ppo_dynasto_s2_2250000.zip \
    $C/b23/dynasto_s3/ppo_dynasto_s3_2250000.zip $C/b25/drl_models/*.zip $C/b13_data/bc_human.pkl $C/b13_data/natural_reference.json \
    > $O/freeze_server_sha256.txt
  FAIRTEST_DATA=$C/runtime-b40/code/fairtest/data $P -m fairtest.runner validate-b5 > $O/validate_b5.log 2>&1 || die validate
  fin hashes
fi
# 1. the case pool: the stored B70 events (no new extraction) through the frozen selection without the 60-case cap
if ! has extract; then
  $P -m fairtest.b72_pool $C/b70/events.jsonl $D --template $C/b13_data --b70-pool $C/b70/case_pool.json --events-sha $EVSHA \
    > $O/pool.log 2>&1 || die pool
  sha256sum $C/b70/events.jsonl $D/case_pool.json $D/natural_reference.json >> $O/freeze_server_sha256.txt
  cp $D/case_pool.json $O/case_pool.json
  python3 -c "import json; print('cases', len(json.load(open('$D/case_pool.json'))['cases']))" > $O/cohort_size.txt
  fin extract
fi
N=$(cut -d' ' -f2 $O/cohort_size.txt)
if [ "$N" -lt 20 ]; then log "cohort $N < 20: not tested, descriptive statistics only (PROTOCOL section 2); stop"; exit 0; fi
export FAIRTEST_DATA=$D
# 2. CC grid -> warm starts -> continuous CC-boundary search (the refinement's starting plan)
if ! has boundary; then
  $P -m fairtest.runner landscape $O/grid_cc.jsonl --targets cc_r157:0 --policies all --workers $W > $O/grid_cc.log 2>&1 || die grid
  $P -m fairtest.external_bundle warm $O/grid_cc.jsonl $D > $O/warm.log 2>&1 || die warm
  $P -m fairtest.b11_search $O/b11.jsonl --suts following:0 --workers $W > $O/b11.log 2>&1 || die b11
  fin boundary
fi
# 3. policy stage (as B41): the v2s policy (seeds 1-3, window shield, CC margin 0.5 m) and the baselines, 17 SUTs
if ! has policy; then
  for s in 1 2 3; do
    if [ $s = 1 ]; then M=$OURS1; CM=$C/b17/collide_commit_s1/ppo_collide_commit_s1_1500000.zip
    else M=$C/b23/fair_guarded2_s$s/ppo_fair_guarded2_s${s}_1500000.zip; CM=$C/b23/collide_commit_s$s/ppo_collide_commit_s${s}_1500000.zip; fi
    FAIRTEST_SHIELD_MARGIN=0.5 $EV $M $O/final_s$s.jsonl --partition all --workers $W --commit --shield-window --suts $SUTS > $O/final_s$s.log 2>&1 || die final_s$s
    $EV $CM $O/collide_commit_s$s.jsonl --partition all --workers $W --commit --suts $SUTS > $O/collide_commit_s$s.log 2>&1 || die collide_s$s
    $EV $C/b23/cars_s$s/ppo_cars_s${s}_2250000.zip $O/cars_s$s.jsonl --partition all --workers $W --suts $SUTS > $O/cars_s$s.log 2>&1 || die cars_s$s
    $EV $C/b23/dynasto_s$s/ppo_dynasto_s${s}_2250000.zip $O/dynasto_s$s.jsonl --partition all --workers $W --suts $SUTS > $O/dynasto_s$s.log 2>&1 || die dynasto_s$s
  done
  fin policy
fi
# 4. G4 margin of every fair failure
if ! has margin; then
  $P -m fairtest.b23_margin $O/margin.jsonl --evals $O/final_s1.jsonl $O/final_s2.jsonl $O/final_s3.jsonl $O/collide_commit_s1.jsonl \
    $O/collide_commit_s2.jsonl $O/collide_commit_s3.jsonl $O/cars_s1.jsonl $O/cars_s2.jsonl $O/cars_s3.jsonl $O/dynasto_s1.jsonl \
    $O/dynasto_s2.jsonl $O/dynasto_s3.jsonl --workers $W > $O/margin.log 2>&1 || die margin
  fin margin
fi
# 5. C replays of the policy episodes (joint support + G2); the frozen replay's b61 entry reads <jobs>/b61/final_s*.jsonl
if ! has replay; then
  mkdir -p $O/replay_jobs/b61
  for s in 1 2 3; do ln -sf ../../final_s$s.jsonl $O/replay_jobs/b61/final_s$s.jsonl; done
  $P -m fairtest.b45_joint_realism replay $O/policy_replay.jsonl --jobs $O/replay_jobs --dataset b61 --workers $W > $O/policy_replay.log 2>&1 || die replay
  fin replay
fi
# 6. refinement: v2s (bo_warm_g4s) and the v2j ablation (bo_warm_g4j), one job per (case, SUT), query log
if ! has refine; then
  $P -m fairtest.b23_search $O/refine.jsonl --cont $O/b11.jsonl --partition all --methods bo_warm_g4s,bo_warm_g4j --suts $SUTS \
    --workers $W --log-queries --split-suts > $O/refine.log 2>&1 || die refine
  fin refine
fi
# 6b. the severity-floor arm v2s-S (bo_warm_g4ss, 3 m/s) and the closing speed of every policy contact and g4s success
if ! has severity; then
  $P -m fairtest.b64_severity_screened $O/refine_g4ss.jsonl --cont $O/b11.jsonl --partition all --suts $SUTS --workers $W \
    --log-queries > $O/refine_g4ss.log 2>&1 || die g4ss
  $P -m fairtest.b51_severity closing $O/closing.jsonl --policy $O/final_s1.jsonl $O/final_s2.jsonl $O/final_s3.jsonl \
    --refine $O/refine.jsonl --method bo_warm_g4s --workers $W > $O/closing.log 2>&1 || die closing
  fin severity
fi
# 7. held-out consensus of every v2s reported failure on the eight competent SUTs (policy stage and refinement)
if ! has consensus; then
  $P -m fairtest.exid_consensus $O $O/consensus.jsonl --refine $O/refine.jsonl --method bo_warm_g4s --workers $W > $O/consensus.log 2>&1 || die consensus
  fin consensus
fi
# 8. frozen analyses on all cases: P1-P3, K1 (S14), S1-S6, S12-S13; K2 (S7 P1 without light contacts), S8, S10, S11; v2s-S
if ! has analysis; then
  $P -m fairtest.exid_analysis $O $O/b72_result.json --refine $O/refine.jsonl --consensus $O/consensus.jsonl > $O/analysis.log 2>&1 || die analysis
  $P -m fairtest.exid_extra $O $O/b72_extra.json --refine $O/refine.jsonl --method bo_warm_g4s --workers $W > $O/extra.log 2>&1 || die extra
  $P -m fairtest.b64_analysis $O/b72_severity.json --dir $O --policy-closing $O/closing.jsonl --g4s $O/refine.jsonl \
    --g4s-closing $O/closing.jsonl --g4ss $O/refine_g4ss.jsonl --name highsim_ext > $O/severity_analysis.log 2>&1 || die severity_analysis
  fin analysis
fi
# 9. B72 analysis: R1 reproduction of the B70 cases, estimates on the new cases, D1 (case mix out of sample), D2
if ! has b72; then
  $P -m fairtest.b72_analysis $O/b72_analysis.json $O $C/b70 $C/b71/case_pool.json > $O/b72_analysis.log 2>&1 || die b72_analysis
  fin b72
fi
log done
