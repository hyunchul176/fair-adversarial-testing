#!/bin/sh
# B56 POST HOC (reuse labelled; B56 post hoc plan, available on request):
# bo_warm_g4j (re-run, query log) and bo_warm_g4s (screened) on MiTra (B41's 172 sealed cases) and TGSIM I-294 (B28's 80
# cases), from their stored CC-boundary plans, the 17 SUTs in the B46 order, one job per (case, SUT).
# Start after B59: flock -n /tmp/fairtest_b56x.lock sh run_b56_external.sh
C=$HOME/fairtest; R=$C/runtime-b56/code; O=$C/b56
mkdir -p $O; cd $R
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 SDL_VIDEODRIVER=dummy PYTHONWARNINGS=ignore
export FAIRTEST_DRL_SUT=$C/b25/drl_models/0.zip FAIRTEST_DRL_SUT_AT="$C/b25/drl_models/{k}.zip"
P=$C/env/bin/python
S=following:0,following:0.5,following:1,reference_idm:0,following_aeb:0,following_aeb:0.5,rss_sut:0,bc_human:0,ads@0.5:0,ads@2.0:0,rss_lat_sut:0,drl_sut:0,drl_sut@3:0,drl_sut@6:0,drl_sut@p1.5:0,drl_sut@p2:0,drl_sut@p2.5:0
echo "$(date '+%F %T') start" >> $O/external_chain.log
FAIRTEST_DATA=$C/runtime-b40/code/fairtest/data $P -m fairtest.runner validate-b5 > $O/external_validate_b5.log 2>&1 || { echo "$(date '+%F %T') validate rc=$?" >> $O/external_chain.log; exit 1; }
for ds in mitra tgsim; do
  case $ds in mitra) DD=$C/mitra_data; CONT=$C/b41/b11.jsonl;; tgsim) DD=$C/tgsim_data; CONT=$C/b28/b11.jsonl;; esac
  [ -e $O/b56_$ds.done ] && continue
  echo "$(date '+%F %T') $ds" >> $O/external_chain.log
  FAIRTEST_DATA=$DD nice -n 10 $P -m fairtest.b23_search $O/b56_$ds.jsonl --cont $CONT --partition all --methods bo_warm_g4j,bo_warm_g4s \
    --suts $S --workers 30 --log-queries --split-suts > $O/b56_$ds.log 2>&1 && touch $O/b56_$ds.done
  echo "$(date '+%F %T') $ds rc=$?" >> $O/external_chain.log
done
echo "$(date '+%F %T') done" >> $O/external_chain.log
