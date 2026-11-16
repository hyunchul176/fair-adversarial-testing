#!/bin/sh
# B65 (B65 design, fixed before the run, available on request): the final agent's
# reported failures (development; MiTra post hoc) and baseline collisions re-judged with the B57 human-like reference
# population plus a bracket that perceives later than the CC (b65_spec.json). No SUT runs. Code: runtime-b65 = git archive
# (core.autocrlf=false).
#   flock -n /tmp/fairtest_b65.lock taskset -c 0-23 sh run_b65.sh
C=$HOME/fairtest; R=$C/runtime-b65/code; O=$C/b65
mkdir -p $O; cd $R || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 SDL_VIDEODRIVER=dummy PYTHONWARNINGS=ignore
P=$C/env/bin/python
W=${B65_WORKERS:-24}
DD=$C/runtime-b40/code/fairtest/data
echo "$(date '+%F %T') start" >> $O/chain.log
FAIRTEST_DATA=$DD $P -m fairtest.runner validate-b5 > $O/validate_b5.log 2>&1 || { echo "$(date '+%F %T') validate rc=$?" >> $O/chain.log; exit 1; }
[ -e $O/population.json ] || $P -m fairtest.b57_population fairtest/b65_spec.json $O/population.json > $O/population.log 2>&1 \
  || { echo "$(date '+%F %T') population rc=$?" >> $O/chain.log; exit 1; }
[ -e $O/jobs_mitra.jsonl ] || FAIRTEST_DATA=$DD $P -m fairtest.b65_jobs $O --b57 $C/b57 --b56 $C/b56 --b64 $C/b64 --b51 $C/b51 --b41 $C/b41 \
  --population $O/population.json > $O/jobs.log 2>&1 || { echo "$(date '+%F %T') jobs rc=$?" >> $O/chain.log; exit 1; }
sha256sum $O/population.json $O/jobs_dev.jsonl $O/jobs_mitra.jsonl >> $O/chain.log
FAIRTEST_DATA=$DD FAIRTEST_HUMAN_POP=$O/population.json nice -n 10 $P -m fairtest.b65_rejudge $O/jobs_dev.jsonl $O/rejudge_dev.jsonl \
  --workers $W > $O/rejudge_dev.log 2>&1
echo "$(date '+%F %T') rejudge dev rc=$?" >> $O/chain.log
FAIRTEST_DATA=$C/mitra_data FAIRTEST_HUMAN_POP=$O/population.json nice -n 10 $P -m fairtest.b65_rejudge $O/jobs_mitra.jsonl \
  $O/rejudge_mitra.jsonl --workers $W > $O/rejudge_mitra.log 2>&1
echo "$(date '+%F %T') rejudge mitra rc=$?" >> $O/chain.log
$P -m fairtest.b65_analysis $O/B65_ANALYSIS.json $O --b57-rejudge $C/b57/rejudge_dev.jsonl > $O/analysis.log 2>&1
echo "$(date '+%F %T') analysis rc=$?" >> $O/chain.log
echo "$(date '+%F %T') done" >> $O/chain.log
