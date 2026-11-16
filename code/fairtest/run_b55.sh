#!/bin/bash
# B55 (B55 design, available on request): the B54 chain for AW-r157n / AW-r157nc, v2j only,
# development 62, then the paired comparison with B54's AW-r157.
# usage: bash run_b55.sh   (start after the B54 MiTra run: it stops that run's containers by name)
set -u
W=$HOME/fairtest/aw
OUT=$W/b55_dev
cd $HOME/fairtest/runtime-b54/code || exit 1
export FAIRTEST_DATA=$HOME/fairtest/runtime-b26/code/fairtest/data
PY=$HOME/fairtest/env/bin/python
mkdir -p $OUT
date '+%F %T start' >> $OUT/chain.log
docker stop -t 5 fairtest_aw_a1 fairtest_aw_a2 fairtest_aw_a3 fairtest_aw_r1 fairtest_aw_r2 fairtest_aw_r3 > /dev/null 2>&1
[ -f $W/aw_up.sh.b55 ] && cp $W/aw_up.sh.b55 $W/aw_up.sh  # the launcher that also mounts the PID controller file
SPECS="n1:81:5481:aw_r157n n2:82:5482:aw_r157n n3:83:5483:aw_r157n c1:85:5485:aw_r157nc c2:86:5486:aw_r157nc c3:87:5487:aw_r157nc"
I=$(echo $SPECS | tr ' ' ',')
date '+%F %T fleet' >> $OUT/chain.log
bash $W/aw_fleet.sh $SPECS > $OUT/fleet.log 2>&1
for phase in policy refine repro; do
  date "+%F %T $phase" >> $OUT/chain.log
  $PY -m fairtest.aw_run $OUT --phase $phase --instances $I --attackers v2j > $OUT/$phase.log 2>&1 || echo "$phase rc=$?" >> $OUT/chain.log
done
date '+%F %T analysis' >> $OUT/chain.log
$PY -m fairtest.aw_analysis $OUT $OUT/B55_SUMMARY.json --variants aw_r157n,aw_r157nc > $OUT/analysis.log 2>&1 || echo "analysis rc=$?" >> $OUT/chain.log
# B54's AW-r157 per case with the same analysis code (its headline numbers must equal B54_SUMMARY.json)
$PY -m fairtest.aw_analysis $W/b54_dev $OUT/B54_SUMMARY_percase.json > $OUT/analysis_b54.log 2>&1 || echo "analysis_b54 rc=$?" >> $OUT/chain.log
$PY -m fairtest.aw_b55_compare $OUT/B54_SUMMARY_percase.json $OUT/B55_SUMMARY.json $OUT/B55_COMPARE.json > $OUT/compare.log 2>&1 || echo "compare rc=$?" >> $OUT/chain.log
$PY -m fairtest.aw_export $OUT $OUT/B55_FIGDATA.json > $OUT/export.log 2>&1 || echo "export rc=$?" >> $OUT/chain.log
docker stop -t 5 fairtest_aw_n1 fairtest_aw_n2 fairtest_aw_n3 fairtest_aw_c1 fairtest_aw_c2 fairtest_aw_c3 > /dev/null 2>&1
date '+%F %T done' >> $OUT/chain.log
