#!/bin/sh
# B68 (B68 design, fixed before the run, available on request): the B65 job sets
# (the final agent's reported failures and baseline collisions; development, MiTra post hoc) re-judged by four reference
# drivers: cc_r157 (our C; must reproduce the stored verdicts), cc_jrc and fsm_cars (the CARS-code readings of the R157 CC
# and the FSM, ported in b68_cars_refs) and ref_fsm (ours). No SUT runs. Code: runtime-b68 = git archive.
#   flock -n /tmp/fairtest_b68.lock taskset -c 0-23 sh run_b68.sh
C=$HOME/fairtest; R=$C/runtime-b68/code; O=$C/b68
mkdir -p $O; cd $R || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 SDL_VIDEODRIVER=dummy PYTHONWARNINGS=ignore
P=$C/env/bin/python
W=${B68_WORKERS:-4}
DEV=$C/runtime-b40/code/fairtest/data
echo "$(date '+%F %T') start" >> $O/chain.log
FAIRTEST_DATA=$DEV $P -m fairtest.runner validate-b5 > $O/validate_b5.log 2>&1 || { echo "$(date '+%F %T') validate rc=$?" >> $O/chain.log; exit 1; }
for ds in dev mitra; do
  if [ $ds = dev ]; then DATA=$DEV; else DATA=$C/mitra_data; fi
  for v in cc_r157 cc_jrc ref_fsm fsm_cars; do
    if [ ! -e $O/rejudge_${v}_$ds.done ]; then
      FAIRTEST_DATA=$DATA nice -n 10 $P -m fairtest.b68_cars_refs $C/b65/jobs_$ds.jsonl $O/rejudge_${v}_$ds.jsonl --variant $v \
        --workers $W > $O/rejudge_${v}_$ds.log 2>&1 && touch $O/rejudge_${v}_$ds.done
      echo "$(date '+%F %T') $ds $v rc=$?" >> $O/chain.log
    fi
  done
done
$P -m fairtest.b68_analysis $O/B68_ANALYSIS.json $O --b65 $C/b65 --dev-dir $C/exid_prep/dev_standin --mitra-dir $C/b41 \
  --dev-refine $C/b56/b56_dev.jsonl --mitra-refine $C/b56/b56_mitra.jsonl > $O/analysis.log 2>&1
echo "$(date '+%F %T') analysis rc=$?" >> $O/chain.log
echo "$(date '+%F %T') done" >> $O/chain.log
