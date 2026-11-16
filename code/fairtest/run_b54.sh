#!/bin/bash
# B54 full run (B54 design, available on request): six fresh Autoware instances (three per variant)
# -> v2j policy stage + collision-seeking baseline -> v2j refinement -> reproduction reruns -> analysis.
# usage: bash run_b54.sh [OUT_DIR] [PARTITION]
set -u
OUT=${1:-$HOME/fairtest/aw/b54_dev}
PART=${2:-development}
cd $HOME/fairtest/runtime-b54/code || exit 1
export FAIRTEST_DATA=$HOME/fairtest/runtime-b26/code/fairtest/data
PY=$HOME/fairtest/env/bin/python
SPECS="a1:81:5481:aw_default a2:82:5482:aw_default a3:83:5483:aw_default r1:85:5485:aw_r157 r2:86:5486:aw_r157 r3:87:5487:aw_r157"
I=$(echo $SPECS | tr ' ' ',')
mkdir -p $OUT
date '+%F %T fleet' >> $OUT/chain.log
bash $HOME/fairtest/aw/aw_fleet.sh $SPECS > $OUT/fleet.log 2>&1
for phase in policy refine repro; do
  date "+%F %T $phase" >> $OUT/chain.log
  $PY -m fairtest.aw_run $OUT --phase $phase --instances $I --partition $PART > $OUT/$phase.log 2>&1 || echo "$phase rc=$?" >> $OUT/chain.log
done
date '+%F %T analysis' >> $OUT/chain.log
$PY -m fairtest.aw_analysis $OUT $OUT/B54_SUMMARY.json --partition $PART > $OUT/analysis.log 2>&1 || echo "analysis rc=$?" >> $OUT/chain.log
date '+%F %T done' >> $OUT/chain.log
