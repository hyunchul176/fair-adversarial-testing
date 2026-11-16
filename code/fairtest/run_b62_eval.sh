#!/bin/sh
# B62 evaluation (B62 design, available on request), highway-env, development cases:
# the population-trained policy (final 1.5 M model of seeds 1-3, trained on lab GPU runners) with the v2j policy-stage settings
# (window shield, margin 0.5 m, --commit) against the B46 17 SUTs, then margin and joint-support replay.
# Models first: $C/b62/models/ppo_fair_guarded2_s{1,2,3}_1500000.zip (fetched from the job logs, sha256 in models.sha256)
#   flock -n /tmp/fairtest_b62.lock sh run_b62_eval.sh
C=$HOME/fairtest; R=$C/runtime-b62/code; O=$C/b62
mkdir -p $O; cd $R
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 SDL_VIDEODRIVER=dummy PYTHONWARNINGS=ignore
export FAIRTEST_DRL_SUT=$C/b25/drl_models/0.zip FAIRTEST_DRL_SUT_AT="$C/b25/drl_models/{k}.zip"
export FAIRTEST_DATA=$C/runtime-b40/code/fairtest/data
P="nice -n 10 $C/env/bin/python"
S=following:0,following:0.5,following:1,reference_idm:0,following_aeb:0,following_aeb:0.5,rss_sut:0,bc_human:0,ads@0.5:0,ads@2.0:0,rss_lat_sut:0,drl_sut:0,drl_sut@3:0,drl_sut@6:0,drl_sut@p1.5:0,drl_sut@p2:0,drl_sut@p2.5:0
log() { echo "$(date '+%F %T') $*" >> $O/chain.log; }
log start
(cd $O/models && sha256sum -c models.sha256) > $O/models_check.log 2>&1 || { log "model hash rc=$?"; exit 1; }
$P -m fairtest.runner validate-b5 > $O/validate_b5.log 2>&1 || { log "validate rc=$?"; exit 1; }
cp $FAIRTEST_DATA/case_pool.json $O/case_pool.json
: > $O/refine.jsonl  # no refinement in B62 (the stored B56 bo_warm_g4s is composed in b62_analysis)
log policy
for s in 1 2 3; do
  FAIRTEST_SHIELD_MARGIN=0.5 $P -m fairtest.b17_rl eval $O/models/ppo_fair_guarded2_s${s}_1500000.zip $O/final_s$s.jsonl \
    --partition development --workers 30 --commit --shield-window --suts $S > $O/final_s$s.log 2>&1 || log "final_s$s rc=$?"
done
log margin
$P -m fairtest.b23_margin $O/margin.jsonl --evals $O/final_s1.jsonl $O/final_s2.jsonl $O/final_s3.jsonl --workers 30 > $O/margin.log 2>&1 || log "margin rc=$?"
log replay
$P -m fairtest.b45_joint_realism replay $O/policy_replay.jsonl --jobs $C --dataset b62 --workers 30 > $O/policy_replay.log 2>&1 || log "replay rc=$?"
log done
