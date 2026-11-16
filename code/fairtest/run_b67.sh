#!/bin/sh
# B67 (B67 design, fixed before the run, available on request): the final agent's
# screened refinement bo_warm_g4s on the 62 development cases with two more random streams (1, 2); stream 0 = stored B56.
# Pre-check on the first 3 cases: stream 0 through the shim must reproduce the stored B56 run (transparency), stream 1 must
# see only its own seeds and differ somewhere. The 17 SUTs in the B46 order (streams keep their position).
# Code: runtime-b67 = git archive (core.autocrlf=false) of the design commit. Starts when B65 releases its lock:
#   nohup setsid sh -c 'flock /tmp/fairtest_b65.lock true; flock -n /tmp/fairtest_b67.lock taskset -c 0-23 sh run_b67.sh' &
C=$HOME/fairtest; R=$C/runtime-b67/code; O=$C/b67
mkdir -p $O; cd $R || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 SDL_VIDEODRIVER=dummy PYTHONWARNINGS=ignore
export FAIRTEST_DRL_SUT=$C/b25/drl_models/0.zip FAIRTEST_DRL_SUT_AT="$C/b25/drl_models/{k}.zip"
P=$C/env/bin/python
DEV=$C/runtime-b40/code/fairtest/data
W=${B67_WORKERS:-24}
CONT=$C/b23/exposed_cont_search_all7.jsonl
S=following:0,following:0.5,following:1,reference_idm:0,following_aeb:0,following_aeb:0.5,rss_sut:0,bc_human:0,ads@0.5:0,ads@2.0:0,rss_lat_sut:0,drl_sut:0,drl_sut@3:0,drl_sut@6:0,drl_sut@p1.5:0,drl_sut@p2:0,drl_sut@p2.5:0
RUN="nice -n 10 $P -m fairtest.b67_streams run"
echo "$(date '+%F %T') start" >> $O/chain.log
FAIRTEST_DATA=$DEV $P -m fairtest.runner validate-b5 > $O/validate_b5.log 2>&1 || { echo "$(date '+%F %T') validate rc=$?" >> $O/chain.log; exit 1; }
if [ ! -e $O/precheck.done ]; then
  FAIRTEST_DATA=$DEV $RUN $O/precheck_s0.jsonl --stream 0 --cont $CONT --suts $S --workers $W --cases 3 > $O/precheck_s0.log 2>&1
  FAIRTEST_DATA=$DEV $P -m fairtest.b67_streams check $O/precheck_s0.jsonl $C/b56/b56_dev.jsonl --stream 0 --expect same \
    > $O/precheck_s0_check.json 2>&1 || { echo "$(date '+%F %T') precheck stream 0 FAILED" >> $O/chain.log; exit 1; }
  FAIRTEST_DATA=$DEV $RUN $O/precheck_s1.jsonl --stream 1 --cont $CONT --suts $S --workers $W --cases 3 > $O/precheck_s1.log 2>&1
  FAIRTEST_DATA=$DEV $P -m fairtest.b67_streams check $O/precheck_s1.jsonl $C/b56/b56_dev.jsonl --stream 1 --expect different \
    > $O/precheck_s1_check.json 2>&1 || { echo "$(date '+%F %T') precheck stream 1 FAILED" >> $O/chain.log; exit 1; }
  touch $O/precheck.done
  echo "$(date '+%F %T') precheck ok" >> $O/chain.log
fi
for s in 1 2; do
  if [ ! -e $O/b67_dev_s$s.jsonl ]; then
    FAIRTEST_DATA=$DEV $RUN $O/b67_dev_s$s.part --stream $s --cont $CONT --suts $S --workers $W > $O/b67_dev_s$s.log 2>&1 \
      && mv $O/b67_dev_s$s.part $O/b67_dev_s$s.jsonl
    echo "$(date '+%F %T') stream $s rc=$?" >> $O/chain.log
  fi
done
$P -m fairtest.b67_analysis $O/B67_ANALYSIS.json --dir $C/exid_prep/dev_standin --stream0 $C/b56/b56_dev.jsonl \
  --streams $O/b67_dev_s1.jsonl,$O/b67_dev_s2.jsonl > $O/analysis.log 2>&1
echo "$(date '+%F %T') analysis rc=$?" >> $O/chain.log
echo "$(date '+%F %T') done" >> $O/chain.log
