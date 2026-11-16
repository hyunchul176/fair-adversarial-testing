#!/bin/sh
# B57 (design fixed before the run): every stored attack of the development job set (b57_jobs.py) replayed
# against the R157 CC and the sampled human-like reference populations (population.json from b57_spec.json).
# Start after B58 (its Autoware runs keep the wall clock): flock -n /tmp/fairtest_b57.lock sh run_b57.sh
C=$HOME/fairtest; R=$C/runtime-b57/code; O=$C/b57
mkdir -p $O; cd $R
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 SDL_VIDEODRIVER=dummy PYTHONWARNINGS=ignore
export FAIRTEST_DATA=$C/runtime-b40/code/fairtest/data FAIRTEST_HUMAN_POP=$O/population.json
echo "$(date '+%F %T') start" >> $O/chain.log
$C/env/bin/python -m fairtest.runner validate-b5 > $O/validate_b5.log 2>&1 || { echo "$(date '+%F %T') validate rc=$?" >> $O/chain.log; exit 1; }
nice -n 10 $C/env/bin/python -m fairtest.b57_rejudge $O/jobs_dev.jsonl $O/rejudge_dev.jsonl --workers 30 > $O/rejudge_dev.log 2>&1
echo "$(date '+%F %T') rejudge rc=$?" >> $O/chain.log
$C/env/bin/python -m fairtest.b57_analysis $O/B57_ANALYSIS.json $O/rejudge_dev.jsonl > $O/analysis.log 2>&1 || echo "analysis rc=$?" >> $O/chain.log
echo "$(date '+%F %T') done" >> $O/chain.log
