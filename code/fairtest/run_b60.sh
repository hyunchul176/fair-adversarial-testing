#!/bin/bash
# B60 (B60 design, available on request): AW-r157nc with perception noise or missed-detection bursts, and the
# latency-0 control, v2j policy stage (seeds 1-3, no refinement) on the development cases, then the reruns of every reported
# failure and the analysis. Two instances per variant. Start after B59: flock -n /tmp/fairtest_b60.lock bash run_b60.sh
set -u
W=$HOME/fairtest/aw
OUT=$W/b60_dev
cd $HOME/fairtest/runtime-b60/code || exit 1
export FAIRTEST_DATA=$HOME/fairtest/runtime-b26/code/fairtest/data
PY=$HOME/fairtest/env/bin/python
mkdir -p $OUT
date '+%F %T start' >> $OUT/chain.log
SPECS="c0a:91:5491:aw_r157nc_l0 c0b:92:5492:aw_r157nc_l0 nza:93:5493:aw_r157nc_nz nzb:94:5494:aw_r157nc_nz dpa:95:5495:aw_r157nc_dp dpb:96:5496:aw_r157nc_dp"
I=$(echo $SPECS | tr ' ' ',')
date '+%F %T fleet' >> $OUT/chain.log
bash $W/aw_fleet.sh $SPECS > $OUT/fleet.log 2>&1
for phase in policy repro; do
  date "+%F %T $phase" >> $OUT/chain.log
  $PY -m fairtest.aw_run $OUT --phase $phase --instances $I --attackers v2j > $OUT/$phase.log 2>&1 || echo "$phase rc=$?" >> $OUT/chain.log
done
date '+%F %T analysis' >> $OUT/chain.log
$PY -m fairtest.aw_analysis $OUT $OUT/B60_SUMMARY.json --variants aw_r157nc_l0,aw_r157nc_nz,aw_r157nc_dp > $OUT/analysis.log 2>&1 || echo "analysis rc=$?" >> $OUT/chain.log
$PY -m fairtest.aw_b60_compare $OUT/B60_SUMMARY.json $OUT/B60_COMPARE.json > $OUT/compare.log 2>&1 || echo "compare rc=$?" >> $OUT/chain.log
$PY -m fairtest.aw_export $OUT $OUT/B60_FIGDATA.json > $OUT/export.log 2>&1 || echo "export rc=$?" >> $OUT/chain.log
docker stop -t 5 fairtest_aw_c0a fairtest_aw_c0b fairtest_aw_nza fairtest_aw_nzb fairtest_aw_dpa fairtest_aw_dpb > /dev/null 2>&1
date '+%F %T done' >> $OUT/chain.log
