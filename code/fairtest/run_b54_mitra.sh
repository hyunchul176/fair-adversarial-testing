#!/bin/bash
# B54 post hoc on MiTra (B41's 172 sealed cases, reuse labelled post hoc; B54 post hoc plan, available on request): the
# chain of run_b54.sh with the MiTra case data and B41's CC-boundary plans.
# usage: bash run_b54_mitra.sh [OUT_DIR]
set -u
OUT=${1:-$HOME/fairtest/aw/b54_mitra}
cd $HOME/fairtest/runtime-b54/code || exit 1
export FAIRTEST_DATA=$HOME/fairtest/mitra_data
PY=$HOME/fairtest/env/bin/python
CONT=$HOME/fairtest/b41/b11.jsonl
SPECS="a1:81:5481:aw_default a2:82:5482:aw_default a3:83:5483:aw_default r1:85:5485:aw_r157 r2:86:5486:aw_r157 r3:87:5487:aw_r157"
I=$(echo $SPECS | tr ' ' ',')
mkdir -p $OUT
date '+%F %T fleet' >> $OUT/chain.log
bash $HOME/fairtest/aw/aw_fleet.sh $SPECS > $OUT/fleet.log 2>&1
for phase in policy refine repro; do
  date "+%F %T $phase" >> $OUT/chain.log
  $PY -m fairtest.aw_run $OUT --phase $phase --instances $I --partition mitra --cont $CONT > $OUT/$phase.log 2>&1 \
    || echo "$phase rc=$?" >> $OUT/chain.log
done
date '+%F %T analysis' >> $OUT/chain.log
$PY -m fairtest.aw_analysis $OUT $OUT/B54_SUMMARY.json --partition mitra > $OUT/analysis.log 2>&1 || echo "analysis rc=$?" >> $OUT/chain.log
$PY -m fairtest.aw_export $OUT $OUT/B54_FIGDATA.json > $OUT/export.log 2>&1 || echo "export rc=$?" >> $OUT/chain.log
date '+%F %T done' >> $OUT/chain.log
