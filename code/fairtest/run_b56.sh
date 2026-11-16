#!/bin/sh
# B56 (design fixed before the run): bo_warm_g4j re-run with the query log and bo_warm_g4s (CC-screened)
# on the development cases; the 17 SUTs in the B46 order (streams keep their position); one job per (case, SUT).
# Run under a lock: flock -n /tmp/fairtest_b56.lock sh run_b56.sh
C=$HOME/fairtest; R=$C/runtime-b56/code; O=$C/b56
mkdir -p $O; cd $R
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 SDL_VIDEODRIVER=dummy PYTHONWARNINGS=ignore
export FAIRTEST_DRL_SUT=$C/b25/drl_models/0.zip FAIRTEST_DRL_SUT_AT="$C/b25/drl_models/{k}.zip"
export FAIRTEST_DATA=$C/runtime-b40/code/fairtest/data
S=following:0,following:0.5,following:1,reference_idm:0,following_aeb:0,following_aeb:0.5,rss_sut:0,bc_human:0,ads@0.5:0,ads@2.0:0,rss_lat_sut:0,drl_sut:0,drl_sut@3:0,drl_sut@6:0,drl_sut@p1.5:0,drl_sut@p2:0,drl_sut@p2.5:0
echo "$(date '+%F %T') start" >> $O/chain.log
$C/env/bin/python -m fairtest.runner validate-b5 > $O/validate_b5.log 2>&1 || { echo "$(date '+%F %T') validate rc=$?" >> $O/chain.log; exit 1; }
nice -n 10 $C/env/bin/python -m fairtest.b23_search $O/b56_dev.jsonl --cont $C/b23/exposed_cont_search_all7.jsonl \
  --partition development --methods bo_warm_g4j,bo_warm_g4s --suts $S --workers 30 --log-queries --split-suts > $O/b56_dev.log 2>&1
echo "$(date '+%F %T') search rc=$?" >> $O/chain.log
echo "$(date '+%F %T') done" >> $O/chain.log
