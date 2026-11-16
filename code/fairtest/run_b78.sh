#!/bin/sh
# B78 (B78 design, fixed before the run, available on request): the final agent's screened
# refinement with a braking bound (b78_variants --acc-pct 1 / 5), development cases, the weak ladder and longitudinal RSS.
#   pre-check  no bound, first 3 cases x 5 SUTs: must reproduce the stored B56 bo_warm_g4s, else a1/a5 are not run
#   a1, a5     braking bound at the natural highD 1st / 5th percentile of acc_pre and acc_post
# Runs after the B75 part of the B75/B76 chain. B76 was deferred by zero-byte placeholders b76/{d025,d10,v5,v7}.jsonl, created
# on 2026-10-09 before B76 started; this script removes them at the end, whatever B78's outcome (only zero-byte files), and
# restarts run_b75.sh, which then skips the finished B75 runs and runs B76.
# Code: runtime-b78 = git archive (core.autocrlf=false) of the design commit. Start:
#   cd ~/fairtest; flock /tmp/fairtest_b75.lock true; flock -n /tmp/fairtest_b78.lock taskset -c 0-23 sh runtime-b78/code/fairtest/run_b78.sh
C=$HOME/fairtest; R=$C/runtime-b78/code; O=$C/b78; O6=$C/b76
mkdir -p $O; cd $R || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 SDL_VIDEODRIVER=dummy PYTHONWARNINGS=ignore
export FAIRTEST_DRL_SUT=$C/b25/drl_models/0.zip FAIRTEST_DRL_SUT_AT="$C/b25/drl_models/{k}.zip"
P=$C/env/bin/python; W=${B78_WORKERS:-24}
DEV=$C/runtime-b40/code/fairtest/data
CONT=$C/b23/exposed_cont_search_all7.jsonl
S=following:0,following:0.5,following:1,reference_idm:0,following_aeb:0,following_aeb:0.5,rss_sut:0,bc_human:0,ads@0.5:0,ads@2.0:0,rss_lat_sut:0,drl_sut:0,drl_sut@3:0,drl_sut@6:0,drl_sut@p1.5:0,drl_sut@p2:0,drl_sut@p2.5:0
ONLY=following_H0,following_H0.5,following_H1,reference_idm_H0,rss_sut_H0
X="nice -n 10 $P -m fairtest.b78_variants"
log() { echo "$(date '+%F %T') $*" >> $O/chain.log; }
log start
run() {  # name args...
  n=$1; shift
  [ -e $O/$n.jsonl ] && return 0
  FAIRTEST_DATA=$DEV $X $O/$n.part --cont $CONT --suts $S --only $ONLY --workers $W --log-queries "$@" > $O/$n.log 2>&1 && mv $O/$n.part $O/$n.jsonl
}
ok=1
FAIRTEST_DATA=$DEV $P -m fairtest.runner validate-b5 > $O/validate_b5.log 2>&1 || { log "validate rc=$?"; ok=0; }
[ $ok = 1 ] && log validate-ok
if [ $ok = 1 ] && [ ! -e $O/precheck.done ]; then
  FAIRTEST_DATA=$DEV $X $O/precheck.jsonl --cont $CONT --suts $S --only $ONLY --cases 3 --workers $W > $O/precheck.log 2>&1
  if $P -m fairtest.b75_analysis precheck $O/precheck.jsonl $C/b56/b56_dev.jsonl > $O/precheck_compare.json 2>&1; then
    touch $O/precheck.done; log precheck-ok
  else
    log "precheck FAILED"; ok=0
  fi
fi
if [ $ok = 1 ]; then
  run a1 --acc-pct 1; log "B78 a1 rc=$?"
  run a5 --acc-pct 5; log "B78 a5 rc=$?"
fi
# restore B76 (zero-byte placeholders only) and restart the B75/B76 chain
for n in d025 d10 v5 v7; do
  if [ -e $O6/$n.jsonl ] && [ ! -s $O6/$n.jsonl ]; then rm -f $O6/$n.jsonl; fi
done
log "B76 placeholders removed; restarting run_b75.sh"
cd $C && (setsid nohup flock -n /tmp/fairtest_b75.lock taskset -c 0-23 sh runtime-b75/code/fairtest/run_b75.sh > $O6/nohup_resume.log 2>&1 < /dev/null &)
log done
