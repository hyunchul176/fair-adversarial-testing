#!/bin/sh
# B74 (B74 design, fixed before the run, available on request): the CARS-style
# reward trained with lane-change commitment (G1), seeds 1-3, then the evaluation on the development cases, 17 SUTs, one query.
#   arm A: cars_commit from each seed's collide_commit 750k checkpoint (as the agent's policy), 1.5M steps, 15 envs
#   arm B: cars_commit from scratch, 2.25M steps, 15 envs (the B23 CARS-style recipe, run_b23_laneB.sh, with G1)
#   eval:  carsTA, carsTB (--commit) and carsTAsh, carsTBsh, ours (--commit --shield-window, CC margin 0.5 m as the agent)
# then b23_margin (G4), the CC replays of the contacts (joint realism) and b74_analysis.
# Code: runtime-b74 = git archive (core.autocrlf=false) of the design commit.
#   flock -n /tmp/fairtest_b74.lock taskset -c 0-23 sh run_b74.sh
C=$HOME/fairtest; R=$C/runtime-b74/code; O=$C/b74
mkdir -p $O; cd $R || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 SDL_VIDEODRIVER=dummy PYTHONWARNINGS=ignore
export FAIRTEST_DRL_SUT=$C/b25/drl_models/0.zip FAIRTEST_DRL_SUT_AT="$C/b25/drl_models/{k}.zip"
P=$C/env/bin/python
DD=$C/runtime-b40/code/fairtest/data
S=following:0,following:0.5,following:1,reference_idm:0,following_aeb:0,following_aeb:0.5,rss_sut:0,bc_human:0,ads@0.5:0,ads@2.0:0,rss_lat_sut:0,drl_sut:0,drl_sut@3:0,drl_sut@6:0,drl_sut@p1.5:0,drl_sut@p2:0,drl_sut@p2.5:0
log() { echo "$(date '+%F %T') $*" >> $O/chain.log; }
init() { if [ $1 = 1 ]; then echo $C/b17/collide_commit_s1/ppo_collide_commit_s1_750000.zip; else echo $C/b23/collide_commit_s$1/ppo_collide_commit_s$1_750000.zip; fi; }
ours() { if [ $1 = 1 ]; then echo $C/b17/fair_guarded2_s1/ppo_fair_guarded2_s1_1500000.zip; else echo $C/b23/fair_guarded2_s$1/ppo_fair_guarded2_s$1_1500000.zip; fi; }
log start
for s in 1 2 3; do
  for f in $(init $s) $(ours $s); do [ -f $f ] || { log "missing $f"; exit 1; }; done
done
FAIRTEST_DATA=$DD $P -m fairtest.runner validate-b5 > $O/validate_b5.log 2>&1 || { log "validate rc=$?"; exit 1; }
log validate-ok
TR="nice -n 10 $P -m fairtest.b17_rl train"
for s in 1 2 3; do
  [ -e $O/trainA_s$s/ppo_cars_commit_s${s}_1500000.zip ] || FAIRTEST_DATA=$DD $TR $O/trainA_s$s --mode cars_commit --steps 1500000 \
    --envs 15 --seed $s --checkpoint 250000 --init $(init $s) > $O/trainA_s$s.log 2>&1 &
done
wait
for s in 1 2 3; do [ -e $O/trainA_s$s/ppo_cars_commit_s${s}_1500000.zip ] || { log "trainA_s$s missing final model"; exit 1; }; done
log trainA-done
for s in 1 2 3; do
  [ -e $O/trainB_s$s/ppo_cars_commit_s${s}_2250000.zip ] || FAIRTEST_DATA=$DD $TR $O/trainB_s$s --mode cars_commit --steps 2250000 \
    --envs 15 --seed $s --checkpoint 250000 > $O/trainB_s$s.log 2>&1 &
done
wait
for s in 1 2 3; do [ -e $O/trainB_s$s/ppo_cars_commit_s${s}_2250000.zip ] || { log "trainB_s$s missing final model"; exit 1; }; done
log trainB-done
(cd $O && sha256sum trainA_s*/ppo_cars_commit_s*_1500000.zip trainB_s*/ppo_cars_commit_s*_2250000.zip > models.sha256)
EV="nice -n 10 $P -m fairtest.b17_rl eval"
ev() {  # tag seed model [shield]
  [ -e $O/$1_s$2.jsonl ] && return 0
  if [ -n "$4" ]; then
    FAIRTEST_SHIELD_MARGIN=0.5 FAIRTEST_DATA=$DD $EV $3 $O/$1_s$2.part --partition development --commit --shield-window --suts $S --workers 24 > $O/$1_s$2.log 2>&1
  else
    FAIRTEST_DATA=$DD $EV $3 $O/$1_s$2.part --partition development --commit --suts $S --workers 24 > $O/$1_s$2.log 2>&1
  fi && mv $O/$1_s$2.part $O/$1_s$2.jsonl || log "eval $1_s$2 rc=$?"
}
for s in 1 2 3; do
  A=$O/trainA_s$s/ppo_cars_commit_s${s}_1500000.zip; B=$O/trainB_s$s/ppo_cars_commit_s${s}_2250000.zip
  ev carsTA $s $A; ev carsTAsh $s $A shield; ev carsTB $s $B; ev carsTBsh $s $B shield; ev ours $s $(ours $s) shield
  log "eval seed $s"
done
[ -e $O/margin.jsonl ] || FAIRTEST_DATA=$DD nice -n 10 $P -m fairtest.b23_margin $O/margin.jsonl --evals $O/*_s[0-9].jsonl --workers 24 > $O/margin.log 2>&1
log "margin rc=$?"
[ -e $O/replay.jsonl ] || { FAIRTEST_DATA=$DD nice -n 10 $P -m fairtest.b74_analysis replay $O/replay.part $O --workers 24 > $O/replay.log 2>&1 \
  && mv $O/replay.part $O/replay.jsonl; }
log "replay rc=$?"
FAIRTEST_DATA=$DD $P -m fairtest.b74_analysis analyse $O/B74_ANALYSIS.json $O --b66 $C/b66 > $O/analysis.log 2>&1
log "analysis rc=$?"
log done
