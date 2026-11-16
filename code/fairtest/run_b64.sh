#!/bin/sh
# B64 (B64 design, fixed before the run, available on request): the final agent's
# screened refinement with a severity floor (bo_warm_g4ss, v_min 3 m/s). Pre-check (floor off, first 3 development cases
# must reproduce the stored B56 bo_warm_g4s) -> development 62 cases -> MiTra 172 cases (post hoc, reuse labelled).
# The 17 SUTs in the B46 order (streams keep their position), one job per (case, SUT).
# Code: runtime-b64 = git archive (core.autocrlf=false) of the design commit.  flock -n /tmp/fairtest_b64.lock sh run_b64.sh
C=$HOME/fairtest; R=$C/runtime-b64/code; O=$C/b64
mkdir -p $O; cd $R || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 SDL_VIDEODRIVER=dummy PYTHONWARNINGS=ignore
export FAIRTEST_DRL_SUT=$C/b25/drl_models/0.zip FAIRTEST_DRL_SUT_AT="$C/b25/drl_models/{k}.zip"
P=$C/env/bin/python
DEV=$C/runtime-b40/code/fairtest/data
W=${B64_WORKERS:-24}
S=following:0,following:0.5,following:1,reference_idm:0,following_aeb:0,following_aeb:0.5,rss_sut:0,bc_human:0,ads@0.5:0,ads@2.0:0,rss_lat_sut:0,drl_sut:0,drl_sut@3:0,drl_sut@6:0,drl_sut@p1.5:0,drl_sut@p2:0,drl_sut@p2.5:0
echo "$(date '+%F %T') start" >> $O/chain.log
FAIRTEST_DATA=$DEV $P -m fairtest.runner validate-b5 > $O/validate_b5.log 2>&1 || { echo "$(date '+%F %T') validate rc=$?" >> $O/chain.log; exit 1; }
if [ ! -e $O/precheck.done ]; then
  FAIRTEST_DATA=$DEV nice -n 10 $P -m fairtest.b64_severity_screened $O/precheck_vmin_off.jsonl --cont $C/b23/exposed_cont_search_all7.jsonl \
    --partition development --suts $S --workers $W --cases 3 --v-min -1000 --log-queries > $O/precheck.log 2>&1
  $P -m fairtest.b64_precheck $O/precheck_vmin_off.jsonl $C/b56/b56_dev.jsonl > $O/precheck_compare.json 2>&1 \
    || { echo "$(date '+%F %T') precheck FAILED" >> $O/chain.log; exit 1; }
  touch $O/precheck.done
  echo "$(date '+%F %T') precheck ok" >> $O/chain.log
fi
if [ ! -e $O/dev.done ]; then
  FAIRTEST_DATA=$DEV nice -n 10 $P -m fairtest.b64_severity_screened $O/b64_dev.jsonl --cont $C/b23/exposed_cont_search_all7.jsonl \
    --partition development --suts $S --workers $W --log-queries > $O/b64_dev.log 2>&1 && touch $O/dev.done
  echo "$(date '+%F %T') dev rc=$?" >> $O/chain.log
fi
if [ ! -e $O/mitra.done ]; then
  FAIRTEST_DATA=$C/mitra_data nice -n 10 $P -m fairtest.b64_severity_screened $O/b64_mitra.jsonl --cont $C/b41/b11.jsonl \
    --partition all --suts $S --workers $W --log-queries > $O/b64_mitra.log 2>&1 && touch $O/mitra.done
  echo "$(date '+%F %T') mitra rc=$?" >> $O/chain.log
fi
echo "$(date '+%F %T') done" >> $O/chain.log
