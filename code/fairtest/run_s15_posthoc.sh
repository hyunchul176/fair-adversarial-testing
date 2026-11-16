#!/bin/sh
# S15 post hoc (descriptive, after a frozen external run): s15_posthoc jobs -> b68_cars_refs for cc_r157 (must reproduce
# the stored C verdicts), cc_jrc, ref_fsm and fsm_cars -> s15_posthoc analyse. No SUT runs. Code: runtime-s15 = git archive.
# With S15_WAIT_LOCK set, it starts when that lock (the run it reads) is free.
#   flock -n /tmp/fairtest_s15_NAME.lock nice -n 10 taskset -c 24-31 sh run_s15_posthoc.sh NAME RUN_DIR DATA_DIR
set -u
NAME=$1; RUN=$2; DATA=$3
C=$HOME/fairtest; R=$C/runtime-s15/code; O=$C/s15_$NAME
W=${S15_WORKERS:-6}
mkdir -p $O; cd $R || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 SDL_VIDEODRIVER=dummy PYTHONWARNINGS=ignore
P=$C/env/bin/python
log() { echo "$(date '+%F %T') $*" >> $O/chain.log; }
log start
if [ -n "${S15_WAIT_LOCK:-}" ]; then log "waiting for $S15_WAIT_LOCK"; flock $S15_WAIT_LOCK true; log "lock free"; fi
[ -e $RUN/analysis.done ] || { log "$RUN has no analysis.done: the run did not finish; stop"; exit 1; }
if [ ! -e $O/jobs.done ]; then
  FAIRTEST_DATA=$DATA $P -m fairtest.s15_posthoc jobs $RUN $O $NAME > $O/jobs.log 2>&1 || { log "jobs FAILED"; exit 1; }
  touch $O/jobs.done; log "jobs done"
fi
for v in cc_r157 cc_jrc ref_fsm fsm_cars; do
  [ -e $O/rejudge_$v.done ] && continue
  FAIRTEST_DATA=$DATA $P -m fairtest.b68_cars_refs $O/jobs_$NAME.jsonl $O/rejudge_${v}_$NAME.jsonl --variant $v --workers $W \
    > $O/rejudge_$v.log 2>&1 || { log "$v FAILED"; exit 1; }
  touch $O/rejudge_$v.done; log "$v done"
done
FAIRTEST_DATA=$DATA $P -m fairtest.s15_posthoc analyse $O/S15_$NAME.json $RUN $O $O $NAME > $O/analysis.log 2>&1 || { log "analysis FAILED"; exit 1; }
log done
