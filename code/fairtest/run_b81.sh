#!/bin/sh
# B81 (B81 design, fixed before the run, available on request): B61's SUTs of known capability
# (the reference driver C with one factor changed) on the 120 exiD cases of B71 (reuse, labelled post hoc), against the
# final agent: policy stage (seeds 1-3, window shield, C margin 0.5 m) -> C margins -> realism replay -> screened
# refinement bo_warm_g4s (budget 36, B71 boundary plans). Analysis locally: b81_analysis. Code: runtime-b81 (this commit).
# Starts after B80 (Autoware) has released its lock:
#   flock -n /tmp/fairtest_b81.lock nice -n 10 taskset -c 0-23 sh run_b81.sh
C=$HOME/fairtest; R=$C/runtime-b81/code; O=$C/b81
mkdir -p $O; cd $R || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 SDL_VIDEODRIVER=dummy PYTHONWARNINGS=ignore
export FAIRTEST_DRL_SUT=$C/b25/drl_models/0.zip FAIRTEST_DRL_SUT_AT="$C/b25/drl_models/{k}.zip"
export FAIRTEST_HUMAN_POP=$R/fairtest/b61_known_suts.json
P=$C/env/bin/python
W=${B81_WORKERS:-24}
K=cc_r157:0,cc_r157@r0.5:0,cc_r157@r1:0,cc_r157@r1.25:0,cc_r157@r1.75:0,cc_r157@h4:0,cc_r157@h0:0,cc_r157@h1:0,cc_r157@h2:0,cc_r157@h3:0
CONT=$C/b71/b11.jsonl
log() { echo "$(date '+%F %T') $*" >> $O/chain.log; }
log wait-b80
flock /tmp/fairtest_b80.lock true
log start
FAIRTEST_DATA=$C/runtime-b40/code/fairtest/data $P -m fairtest.runner validate-b5 > $O/validate_b5.log 2>&1 || { log "validate rc=$?"; exit 1; }
export FAIRTEST_DATA=$C/b71_data
cp $FAIRTEST_DATA/case_pool.json $O/case_pool.json
log policy
for s in 1 2 3; do
  if [ $s = 1 ]; then M=$C/b17/fair_guarded2_s1/ppo_fair_guarded2_s1_1500000.zip
  else M=$C/b23/fair_guarded2_s$s/ppo_fair_guarded2_s${s}_1500000.zip; fi
  FAIRTEST_SHIELD_MARGIN=0.5 $P -m fairtest.b17_rl eval $M $O/final_s$s.jsonl --partition all --workers $W --commit --shield-window \
    --suts $K > $O/final_s$s.log 2>&1 || log "final_s$s rc=$?"
done
log margin
$P -m fairtest.b23_margin $O/margin.jsonl --evals $O/final_s1.jsonl $O/final_s2.jsonl $O/final_s3.jsonl --workers $W \
  > $O/margin.log 2>&1 || log "margin rc=$?"
log replay
$P -m fairtest.b45_joint_realism replay $O/policy_replay.jsonl --jobs $C --dataset b81 --workers $W > $O/policy_replay.log 2>&1 \
  || log "replay rc=$?"
log refine
$P -m fairtest.b23_search $O/refine.jsonl --cont $CONT --partition all --methods bo_warm_g4s --suts $K --workers $W --log-queries \
  --split-suts > $O/refine.log 2>&1 || log "refine rc=$?"
log done
