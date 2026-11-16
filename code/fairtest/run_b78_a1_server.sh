#!/bin/sh
# B78 a1 on a lab server after the B75/B76 chain (B78 design §5a, available on request: the submission to a lab GPU runner
# was refused on 2026-10-09 because the runner was not ready for team jobs; a5 runs on another lab GPU runner).
# Same frozen code as the fleet jobs (runtime-b78 = design commit c565b47); pre-check as in run_b78.sh. No placeholders here.
#   cd ~/fairtest; (setsid nohup sh -c 'flock /tmp/fairtest_b75.lock true; flock -n /tmp/fairtest_b78.lock taskset -c 0-23 sh b78/run_b78_a1_server.sh' ...)
C=$HOME/fairtest; R=$C/runtime-b78/code; O=$C/b78
mkdir -p $O; cd $R || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 SDL_VIDEODRIVER=dummy PYTHONWARNINGS=ignore
P=$C/env/bin/python; W=${B78_WORKERS:-24}
DEV=$C/runtime-b40/code/fairtest/data
CONT=$C/b23/exposed_cont_search_all7.jsonl
S=following:0,following:0.5,following:1,reference_idm:0,following_aeb:0,following_aeb:0.5,rss_sut:0,bc_human:0,ads@0.5:0,ads@2.0:0,rss_lat_sut:0,drl_sut:0,drl_sut@3:0,drl_sut@6:0,drl_sut@p1.5:0,drl_sut@p2:0,drl_sut@p2.5:0
ONLY=following_H0,following_H0.5,following_H1,reference_idm_H0,rss_sut_H0
X="nice -n 10 $P -m fairtest.b78_variants"
log() { echo "$(date '+%F %T') $*" >> $O/chain.log; }
log start-a1-server
FAIRTEST_DATA=$DEV $P -m fairtest.runner validate-b5 > $O/validate_b5.log 2>&1 || { log "validate rc=$?"; exit 1; }
log validate-ok
if [ ! -e $O/precheck.done ]; then
  FAIRTEST_DATA=$DEV $X $O/precheck.jsonl --cont $CONT --suts $S --only $ONLY --cases 3 --workers $W > $O/precheck.log 2>&1
  $P -m fairtest.b75_analysis precheck $O/precheck.jsonl $C/b56/b56_dev.jsonl > $O/precheck_compare.json 2>&1 \
    || { log "precheck FAILED"; exit 1; }
  touch $O/precheck.done
  log precheck-ok
fi
if [ ! -e $O/a1.jsonl ]; then
  FAIRTEST_DATA=$DEV $X $O/a1.part --cont $CONT --suts $S --only $ONLY --workers $W --log-queries --acc-pct 1 > $O/a1.log 2>&1 \
    && mv $O/a1.part $O/a1.jsonl
  log "B78 a1 rc=$?"
fi
log done
