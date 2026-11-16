#!/bin/sh
# B75 then B76 (B75 and B76 designs, fixed before the runs, available on request): the final agent's screened
# refinement re-run with other contract parameters (b75_variants, bo_warm_g4v), development cases.
#   pre-check  contract parameters, first 3 cases x 17 SUTs: must reproduce the stored B56 bo_warm_g4s, else stop
#   B75        q95 (joint support at the 95th percentile), g05 (G2 at 0.5 IQR); 17 SUTs
#   B76        d025, d10 (CC margin 0.25 / 1.0 m), v5, v7 (severity floor 5 / 7 m/s); 9 SUTs (streams keep the 17-SUT positions)
# The analyses run locally (b75_analysis b75 / b76; they need the stored policy-stage rows).
# Code: runtime-b75 = git archive (core.autocrlf=false) of the design commit. Start after B74:
#   flock /tmp/fairtest_b74.lock true; flock -n /tmp/fairtest_b75.lock taskset -c 0-23 sh run_b75.sh
C=$HOME/fairtest; R=$C/runtime-b75/code; O5=$C/b75; O6=$C/b76
mkdir -p $O5 $O6; cd $R || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 SDL_VIDEODRIVER=dummy PYTHONWARNINGS=ignore
export FAIRTEST_DRL_SUT=$C/b25/drl_models/0.zip FAIRTEST_DRL_SUT_AT="$C/b25/drl_models/{k}.zip"
P=$C/env/bin/python; W=${B75_WORKERS:-24}
DEV=$C/runtime-b40/code/fairtest/data
CONT=$C/b23/exposed_cont_search_all7.jsonl
S=following:0,following:0.5,following:1,reference_idm:0,following_aeb:0,following_aeb:0.5,rss_sut:0,bc_human:0,ads@0.5:0,ads@2.0:0,rss_lat_sut:0,drl_sut:0,drl_sut@3:0,drl_sut@6:0,drl_sut@p1.5:0,drl_sut@p2:0,drl_sut@p2.5:0
ONLY=following_H0,following_H0.5,following_H1,reference_idm_H0,rss_sut_H0,bc_human_H0,ads@2.0_H0,following_aeb_H0,following_aeb_H0.5
X="nice -n 10 $P -m fairtest.b75_variants"
log() { echo "$(date '+%F %T') $*" >> $O5/chain.log; }
log start
FAIRTEST_DATA=$DEV $P -m fairtest.runner validate-b5 > $O5/validate_b5.log 2>&1 || { log "validate rc=$?"; exit 1; }
log validate-ok
if [ ! -e $O5/precheck.done ]; then
  FAIRTEST_DATA=$DEV $X $O5/precheck.jsonl --cont $CONT --suts $S --cases 3 --workers $W > $O5/precheck.log 2>&1
  $P -m fairtest.b75_analysis precheck $O5/precheck.jsonl $C/b56/b56_dev.jsonl > $O5/precheck_compare.json 2>&1 \
    || { log "precheck FAILED"; exit 1; }
  touch $O5/precheck.done
  log precheck-ok
fi
run() {  # out_dir name args...
  d=$1; n=$2; shift 2
  [ -e $d/$n.jsonl ] && return 0
  FAIRTEST_DATA=$DEV $X $d/$n.part --cont $CONT --suts $S --workers $W --log-queries "$@" > $d/$n.log 2>&1 && mv $d/$n.part $d/$n.jsonl
}
run $O5 q95 --q 95; log "B75 q95 rc=$?"
run $O5 g05 --g2 0.5; log "B75 g05 rc=$?"
run $O6 d025 --delta 0.25 --only $ONLY; log "B76 d025 rc=$?"
run $O6 d10 --delta 1.0 --only $ONLY; log "B76 d10 rc=$?"
run $O6 v5 --v-min 5 --only $ONLY; log "B76 v5 rc=$?"
run $O6 v7 --v-min 7 --only $ONLY; log "B76 v7 rc=$?"
log done
