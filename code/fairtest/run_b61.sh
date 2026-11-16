#!/bin/sh
# B61 controls (B61 design, available on request), highway-env, no training, development cases:
#  1. the v2j policy stage (seeds 1-3, window shield) and refinement (g4j, g4s) against SUTs of known capability
#  2. the collision-seeking policy (collide_commit s1-3) with the window shield, 17 SUTs
#  3. screened random search (random_g4s), 17 SUTs;  4. attack-free natural cut-ins, 17 SUTs
# Start after B60: flock -n /tmp/fairtest_b61.lock sh run_b61.sh
C=$HOME/fairtest; R=$C/runtime-b61/code; O=$C/b61
mkdir -p $O; cd $R
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 SDL_VIDEODRIVER=dummy PYTHONWARNINGS=ignore
export FAIRTEST_DRL_SUT=$C/b25/drl_models/0.zip FAIRTEST_DRL_SUT_AT="$C/b25/drl_models/{k}.zip"
export FAIRTEST_DATA=$C/runtime-b40/code/fairtest/data FAIRTEST_HUMAN_POP=$R/fairtest/b61_known_suts.json
P=$C/env/bin/python
EV="$P -m fairtest.b17_rl eval"
K=cc_r157:0,cc_r157@r0.5:0,cc_r157@r1:0,cc_r157@r1.25:0,cc_r157@r1.75:0,cc_r157@h4:0,cc_r157@h0:0,cc_r157@h1:0,cc_r157@h2:0,cc_r157@h3:0
S=following:0,following:0.5,following:1,reference_idm:0,following_aeb:0,following_aeb:0.5,rss_sut:0,bc_human:0,ads@0.5:0,ads@2.0:0,rss_lat_sut:0,drl_sut:0,drl_sut@3:0,drl_sut@6:0,drl_sut@p1.5:0,drl_sut@p2:0,drl_sut@p2.5:0
CONT=$C/b23/exposed_cont_search_all7.jsonl
log() { echo "$(date '+%F %T') $*" >> $O/chain.log; }
log start
$P -m fairtest.runner validate-b5 > $O/validate_b5.log 2>&1 || { log "validate rc=$?"; exit 1; }
cp $FAIRTEST_DATA/case_pool.json $O/case_pool.json
log policy
for s in 1 2 3; do
  if [ $s = 1 ]; then M=$C/b17/fair_guarded2_s1/ppo_fair_guarded2_s1_1500000.zip; CM=$C/b17/collide_commit_s1/ppo_collide_commit_s1_1500000.zip
  else M=$C/b23/fair_guarded2_s$s/ppo_fair_guarded2_s${s}_1500000.zip; CM=$C/b23/collide_commit_s$s/ppo_collide_commit_s${s}_1500000.zip; fi
  FAIRTEST_SHIELD_MARGIN=0.5 $EV $M $O/final_s$s.jsonl --partition development --workers 30 --commit --shield-window --suts $K > $O/final_s$s.log 2>&1 || log "final_s$s rc=$?"
  FAIRTEST_SHIELD_MARGIN=0.5 $EV $CM $O/collide_shield_s$s.jsonl --partition development --workers 30 --commit --shield-window --suts $S > $O/collide_shield_s$s.log 2>&1 || log "collide_shield_s$s rc=$?"
done
log margin
$P -m fairtest.b23_margin $O/margin.jsonl --evals $O/final_s1.jsonl $O/final_s2.jsonl $O/final_s3.jsonl $O/collide_shield_s1.jsonl \
  $O/collide_shield_s2.jsonl $O/collide_shield_s3.jsonl --workers 30 > $O/margin.log 2>&1 || log "margin rc=$?"
log replay
$P -m fairtest.b45_joint_realism replay $O/policy_replay.jsonl --jobs $C --dataset b61 --workers 30 > $O/policy_replay.log 2>&1 || log "replay rc=$?"
$P -m fairtest.b45_joint_realism replay $O/policy_replay_c.jsonl --jobs $C --dataset b61c --workers 30 > $O/policy_replay_c.log 2>&1 || log "replay_c rc=$?"
log natural
$P -m fairtest.b61_natural $O/natural.jsonl --partition development --suts $S --workers 30 > $O/natural.log 2>&1 || log "natural rc=$?"
log random
nice -n 10 $P -m fairtest.b23_search $O/random_g4s.jsonl --cont $CONT --partition development --methods random_g4s --suts $S --workers 30 \
  --log-queries --split-suts > $O/random_g4s.log 2>&1 || log "random rc=$?"
log refine
nice -n 10 $P -m fairtest.b23_search $O/refine.jsonl --cont $CONT --partition development --methods bo_warm_g4j,bo_warm_g4s --suts $K \
  --workers 30 --log-queries --split-suts > $O/refine.log 2>&1 || log "refine rc=$?"
log done
