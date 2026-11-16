#!/bin/sh
# B73 (B73 design, fixed before the run, available on request): contact geometry of
# the final agent's reported failures (competent SUTs, the weakest SUT as contrast, the CommonRoad planner), plain replay
# plus body-tolerance replays (collision polygon shrunk by 0.05 and 0.10 m per side; 0 = patch check), then the analysis.
# Jobs of dev, mitra, highsim and exid are built locally from the stored rows (b73_geometry jobs) and copied to $O.
# Code: runtime-b73 = git archive (core.autocrlf=false) of the design commit.
#   flock -n /tmp/fairtest_b73.lock taskset -c 24-31 sh run_b73.sh
C=$HOME/fairtest; R=$C/runtime-b73/code; O=$C/b73
mkdir -p $O; cd $R || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 SDL_VIDEODRIVER=dummy PYTHONWARNINGS=ignore
export FAIRTEST_DRL_SUT=$C/b25/drl_models/0.zip FAIRTEST_DRL_SUT_AT="$C/b25/drl_models/{k}.zip"
P=$C/env/bin/python; W=${B73_WORKERS:-6}
DEV=$C/runtime-b40/code/fairtest/data
log() { echo "$(date '+%F %T') $*" >> $O/chain.log; }
log start
for n in dev mitra highsim exid; do [ -f $O/jobs_$n.jsonl ] || { log "missing jobs_$n.jsonl"; exit 1; }; done
FAIRTEST_DATA=$DEV $P -m fairtest.runner validate-b5 > $O/validate_b5.log 2>&1 || { log "validate rc=$?"; exit 1; }
log validate-ok
for ds in dev:$DEV mitra:$C/mitra_data highsim:$C/b70_data exid:$C/b71_data; do
  n=${ds%%:*}; d=${ds#*:}
  [ -e $O/run_$n.jsonl ] || { FAIRTEST_DATA=$d nice -n 10 $P -m fairtest.b73_geometry run $O/run_$n.part $O/jobs_$n.jsonl --workers $W \
    > $O/run_$n.log 2>&1 && mv $O/run_$n.part $O/run_$n.jsonl; }
  log "$n rc=$?"
done
D=$C/b69/commonroad_rp
[ -e $O/jobs_commonroad.jsonl ] || FAIRTEST_DATA=$DEV $P -m fairtest.b73_geometry layout-jobs $O/jobs_commonroad.jsonl $D --dataset commonroad \
  --all-suts > $O/jobs_commonroad.log 2>&1
log "commonroad jobs rc=$?"
[ -e $O/run_commonroad.jsonl ] || { FAIRTEST_DATA=$DEV FAIRTEST_CR=rp FAIRTEST_CR_PATH=$C/ext/commonroad nice -n 10 \
  $P -m fairtest.b69_run fairtest.b73_geometry run $O/run_commonroad.part $O/jobs_commonroad.jsonl --workers $W \
  > $O/run_commonroad.log 2>&1 && mv $O/run_commonroad.part $O/run_commonroad.jsonl; }
log "commonroad rc=$?"
FAIRTEST_DATA=$DEV $P -m fairtest.b73_geometry analyse $O/B73_ANALYSIS.json $O/run_dev.jsonl $O/run_mitra.jsonl $O/run_highsim.jsonl \
  $O/run_exid.jsonl $O/run_commonroad.jsonl > $O/analysis.log 2>&1
log "analysis rc=$?"
log done
