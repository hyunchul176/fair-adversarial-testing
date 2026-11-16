#!/bin/sh
# B69 third-party SUTs (B69 design, fixed before the run, available on request): the final
# agent v2s (policy fair_guarded2 s1-3 with the window shield and CC margin 0.5 m, then bo_warm_g4s from the stored
# CC-boundary plan, 36 queries), the collision-seeking baseline (collide_commit s1-3) and attack-free cut-ins against four
# third-party SUTs in the acc_prod_aeb slot (b69_run; the frozen judge code is unchanged), development cases; then the G4
# margin, C replays of the policy episodes (joint support + G2; the frozen replay's b61 entry reads <jobs>/b61/final_s*.jsonl,
# here links to this SUT's policy files), held-out consensus of the reported failures, closing speeds, and the analysis.
# The CommonRoad planner (first) also gets the severity-floor refinement of v2s-S (bo_warm_g4ss, 3 m/s; B64).
# Code: runtime-b69m = git archive of the design commit (LF). Start after B67:
#   flock /tmp/fairtest_b67.lock true; flock -n /tmp/fairtest_b69.lock nice -n 10 taskset -c 0-23 sh run_b69.sh
# Overrides for the pipeline dry run (not evidence): B69_RUNTIME, B69_OUT, B69_DATA (a data folder with fewer cases),
# B69_WORKERS, B69_SKIP_VALIDATE=1.
C=$HOME/fairtest; R=${B69_RUNTIME:-$C/runtime-b69m}/code; O=${B69_OUT:-$C/b69}
DATA=${B69_DATA:-$C/runtime-b40/code/fairtest/data}; W=${B69_WORKERS:-24}
mkdir -p $O; cd $R || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 SDL_VIDEODRIVER=dummy PYTHONWARNINGS=ignore
export FAIRTEST_DRL_SUT=$C/b25/drl_models/0.zip FAIRTEST_DRL_SUT_AT="$C/b25/drl_models/{k}.zip"
export FAIRTEST_DATA=$DATA FAIRTEST_SUMO_PATH=$C/ext/sumo FAIRTEST_CR_PATH=$C/ext/commonroad
P=$C/env/bin/python
X="$P -m fairtest.b69_run"
CONT=$C/b23/exposed_cont_search_all7.jsonl
log() { echo "$(date '+%F %T') $*" >> $O/chain.log; }
log "start data=$DATA workers=$W"
if [ "${B69_SKIP_VALIDATE:-0}" != 1 ]; then
  $P -m fairtest.runner validate-b5 > $O/validate_b5.log 2>&1 || { log "validate rc=$?"; exit 1; }
fi
for sut in commonroad_rp sumo_acc sumo_krauss sumo_idm; do
  D=$O/$sut
  mkdir -p $D/replay_jobs/b61
  case $sut in
    sumo_*) export FAIRTEST_SUMO=${sut#sumo_}; unset FAIRTEST_CR ;;
    commonroad_rp) export FAIRTEST_CR=rp; unset FAIRTEST_SUMO ;;
  esac
  cp $DATA/case_pool.json $D/case_pool.json
  log "$sut natural"
  $X fairtest.b61_natural $D/natural.jsonl --partition development --suts acc_prod_aeb:0 --workers $W \
    > $D/natural.log 2>&1 || log "$sut natural rc=$?"
  for s in 1 2 3; do
    if [ $s = 1 ]; then M=$C/b17/fair_guarded2_s1/ppo_fair_guarded2_s1_1500000.zip; CM=$C/b17/collide_commit_s1/ppo_collide_commit_s1_1500000.zip
    else M=$C/b23/fair_guarded2_s$s/ppo_fair_guarded2_s${s}_1500000.zip; CM=$C/b23/collide_commit_s$s/ppo_collide_commit_s${s}_1500000.zip; fi
    log "$sut policy s$s"
    FAIRTEST_SHIELD_MARGIN=0.5 $X fairtest.b17_rl eval $M $D/final_s$s.jsonl --partition development --commit --shield-window \
      --suts acc_prod_aeb:0 --workers $W > $D/final_s$s.log 2>&1 || log "$sut final_s$s rc=$?"
    $X fairtest.b17_rl eval $CM $D/collide_commit_s$s.jsonl --partition development --commit --suts acc_prod_aeb:0 \
      --workers $W > $D/collide_commit_s$s.log 2>&1 || log "$sut collide_commit_s$s rc=$?"
    ln -sf ../../final_s$s.jsonl $D/replay_jobs/b61/final_s$s.jsonl
  done
  log "$sut margin"
  $X fairtest.b23_margin $D/margin.jsonl --evals $D/final_s1.jsonl $D/final_s2.jsonl $D/final_s3.jsonl \
    $D/collide_commit_s1.jsonl $D/collide_commit_s2.jsonl $D/collide_commit_s3.jsonl --workers $W > $D/margin.log 2>&1 || log "$sut margin rc=$?"
  log "$sut replay"
  $X fairtest.b45_joint_realism replay $D/policy_replay.jsonl --jobs $D/replay_jobs --dataset b61 --workers $W \
    > $D/policy_replay.log 2>&1 || log "$sut replay rc=$?"
  log "$sut refine"
  $X fairtest.b23_search $D/refine.jsonl --cont $CONT --partition development --methods bo_warm_g4s --suts acc_prod_aeb:0 \
    --workers $W --log-queries --split-suts > $D/refine.log 2>&1 || log "$sut refine rc=$?"
  log "$sut consensus"
  $X fairtest.exid_consensus $D $D/consensus.jsonl --refine $D/refine.jsonl --method bo_warm_g4s --suts acc_prod_aeb_H0 \
    --workers $W > $D/consensus.log 2>&1 || log "$sut consensus rc=$?"
  log "$sut closing"
  $X fairtest.b51_severity closing $D/closing.jsonl --policy $D/final_s1.jsonl $D/final_s2.jsonl $D/final_s3.jsonl \
    --refine $D/refine.jsonl --method bo_warm_g4s --workers $W > $D/closing.log 2>&1 || log "$sut closing rc=$?"
  if [ $sut = commonroad_rp ]; then
    log "$sut refine severity floor"
    $X fairtest.b64_severity_screened $D/refine_g4ss.jsonl --cont $CONT --partition development --suts acc_prod_aeb:0 \
      --workers $W --log-queries > $D/refine_g4ss.log 2>&1 || log "$sut refine_g4ss rc=$?"
  fi
done
log analysis
$P -m fairtest.b69_analysis $O $O/b69_result.json > $O/analysis.log 2>&1 || log "analysis rc=$?"
log done
