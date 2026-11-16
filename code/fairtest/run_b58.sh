#!/bin/bash
# B58 (B58 design, available on request): AW-r157nc with a perceived-adversary latency of
# 0 / 0.3 / 0.6 s, v2j policy stage (seeds 1-3, no refinement) on the development cases, then the reruns of every reported
# failure and the analysis. Two instances per variant. Start after B56 (CPU): flock -n /tmp/fairtest_b58.lock bash run_b58.sh
set -u
W=$HOME/fairtest/aw
OUT=$W/b58_dev
cd $HOME/fairtest/runtime-b58/code || exit 1
export FAIRTEST_DATA=$HOME/fairtest/runtime-b26/code/fairtest/data
PY=$HOME/fairtest/env/bin/python
mkdir -p $OUT
date '+%F %T start' >> $OUT/chain.log
SPECS="l0a:91:5491:aw_r157nc_l0 l0b:92:5492:aw_r157nc_l0 l3a:93:5493:aw_r157nc_l300 l3b:94:5494:aw_r157nc_l300 l6a:95:5495:aw_r157nc_l600 l6b:96:5496:aw_r157nc_l600"
I=$(echo $SPECS | tr ' ' ',')
date '+%F %T fleet' >> $OUT/chain.log
bash $W/aw_fleet.sh $SPECS > $OUT/fleet.log 2>&1
for phase in policy repro; do
  date "+%F %T $phase" >> $OUT/chain.log
  $PY -m fairtest.aw_run $OUT --phase $phase --instances $I --attackers v2j > $OUT/$phase.log 2>&1 || echo "$phase rc=$?" >> $OUT/chain.log
done
date '+%F %T analysis' >> $OUT/chain.log
$PY -m fairtest.aw_analysis $OUT $OUT/B58_SUMMARY.json --variants aw_r157nc_l0,aw_r157nc_l300,aw_r157nc_l600 > $OUT/analysis.log 2>&1 || echo "analysis rc=$?" >> $OUT/chain.log
$PY -m fairtest.aw_b58_compare $OUT/B58_SUMMARY.json $OUT/B58_COMPARE.json > $OUT/compare.log 2>&1 || echo "compare rc=$?" >> $OUT/chain.log
$PY -m fairtest.aw_export $OUT $OUT/B58_FIGDATA.json > $OUT/export.log 2>&1 || echo "export rc=$?" >> $OUT/chain.log
docker stop -t 5 fairtest_aw_l0a fairtest_aw_l0b fairtest_aw_l3a fairtest_aw_l3b fairtest_aw_l6a fairtest_aw_l6b > /dev/null 2>&1
date '+%F %T done' >> $OUT/chain.log
