#!/bin/sh
# B66 (B66 design, fixed before the run, available on request): baseline controls
# without retraining on the development cases, 17 SUTs, policy seeds 1-3, one query:
#   carsG1   the B23 CARS-style policy with lane-change commitment (G1) enforced (--commit)
#   cars     the same policy as in B23 (no commitment), re-run on the 17 SUTs as the paired control
#   feasr0sh the B52 zero-reaction-ideal-ADS policy shielded by its own reference (oracle_react0, window shield, 0.5 m)
#   feast0sh the B52 omniscient-reference policy shielded by its own reference (oracle_t0, window shield, 0.5 m)
# then the G4 margins (b23_margin) and b66_analysis. Code: runtime-b66 = git archive (core.autocrlf=false).
#   flock -n /tmp/fairtest_b66.lock taskset -c 0-23 sh run_b66.sh
C=$HOME/fairtest; R=$C/runtime-b66/code; O=$C/b66
mkdir -p $O; cd $R || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 SDL_VIDEODRIVER=dummy PYTHONWARNINGS=ignore
export FAIRTEST_DRL_SUT=$C/b25/drl_models/0.zip FAIRTEST_DRL_SUT_AT="$C/b25/drl_models/{k}.zip"
P=$C/env/bin/python
DD=$C/runtime-b40/code/fairtest/data
S=following:0,following:0.5,following:1,reference_idm:0,following_aeb:0,following_aeb:0.5,rss_sut:0,bc_human:0,ads@0.5:0,ads@2.0:0,rss_lat_sut:0,drl_sut:0,drl_sut@3:0,drl_sut@6:0,drl_sut@p1.5:0,drl_sut@p2:0,drl_sut@p2.5:0
EV="nice -n 10 $P -m fairtest.b17_rl eval"
echo "$(date '+%F %T') start" >> $O/chain.log
FAIRTEST_DATA=$DD $P -m fairtest.runner validate-b5 > $O/validate_b5.log 2>&1 || { echo "$(date '+%F %T') validate rc=$?" >> $O/chain.log; exit 1; }
for s in 1 2 3; do
  M=$C/b23/cars_s$s/ppo_cars_s${s}_2250000.zip
  [ -e $O/carsG1_s$s.jsonl ] || { FAIRTEST_DATA=$DD $EV $M $O/carsG1_s$s.part --partition development --commit --suts $S --workers 24 \
    > $O/carsG1_s$s.log 2>&1 && mv $O/carsG1_s$s.part $O/carsG1_s$s.jsonl; }
  [ -e $O/cars_s$s.jsonl ] || { FAIRTEST_DATA=$DD $EV $M $O/cars_s$s.part --partition development --suts $S --workers 24 \
    > $O/cars_s$s.log 2>&1 && mv $O/cars_s$s.part $O/cars_s$s.jsonl; }
  for f in feasr0:oracle_react0 feast0:oracle_t0; do
    tag=${f%%:*}; ref=${f##*:}
    M=$C/b52/${tag}_s$s/ppo_${tag}_s${s}_1500000.zip
    [ -e $O/${tag}sh_s$s.jsonl ] || { FAIRTEST_SHIELD_REF=$ref FAIRTEST_SHIELD_MARGIN=0.5 FAIRTEST_DATA=$DD $EV $M $O/${tag}sh_s$s.part \
      --partition development --commit --shield-window --suts $S --workers 24 > $O/${tag}sh_s$s.log 2>&1 && mv $O/${tag}sh_s$s.part $O/${tag}sh_s$s.jsonl; }
  done
  echo "$(date '+%F %T') seed $s" >> $O/chain.log
done
[ -e $O/margin.jsonl ] || FAIRTEST_DATA=$DD nice -n 10 $P -m fairtest.b23_margin $O/margin.jsonl --evals $O/*_s[0-9].jsonl --workers 24 > $O/margin.log 2>&1
echo "$(date '+%F %T') margin rc=$?" >> $O/chain.log
$P -m fairtest.b66_analysis $O/B66_ANALYSIS.json $O --feas-noshield $C/b52/eval_dev > $O/analysis.log 2>&1
echo "$(date '+%F %T') analysis rc=$?" >> $O/chain.log
echo "$(date '+%F %T') done" >> $O/chain.log
