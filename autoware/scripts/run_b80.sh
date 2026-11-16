#!/bin/bash
# B80 (B80 design, fixed before the run, available on request): the final agent (v2j policy
# stage + screened refinement, B59) and the collision-seeking baseline against Autoware AW-default / AW-r157 on the 120
# exiD cases of B71 (reuse labelled post hoc), as the B54 MiTra chain with B59's --refine-mode screened. Code: runtime-b59.
# The Autoware containers are kept on cores 0-23 (docker update in a loop; restarts recreate containers).
# usage: flock -n /tmp/fairtest_b80.lock bash run_b80.sh [OUT_DIR]
set -u
OUT=${1:-$HOME/fairtest/aw/b80_exid}
cd $HOME/fairtest/runtime-b59/code || exit 1
export FAIRTEST_DATA=$HOME/fairtest/b71_data
PY=$HOME/fairtest/env/bin/python
CONT=$HOME/fairtest/b71/b11.jsonl
SPECS="xa1:91:5491:aw_default xa2:92:5492:aw_default xa3:93:5493:aw_default xr1:95:5495:aw_r157 xr2:96:5496:aw_r157 xr3:97:5497:aw_r157"
I=$(echo $SPECS | tr ' ' ',')
mkdir -p $OUT
( while :; do
    for c in $(docker ps -q --filter name=fairtest_aw_x); do docker update --cpuset-cpus 0-23 $c > /dev/null 2>&1; done
    sleep 30
  done ) &
PIN=$!
date '+%F %T fleet' >> $OUT/chain.log
bash $HOME/fairtest/aw/aw_fleet.sh $SPECS > $OUT/fleet.log 2>&1
for phase in policy refine repro; do
  date "+%F %T $phase" >> $OUT/chain.log
  EXTRA=""
  [ $phase = refine ] && EXTRA="--refine-mode screened"
  $PY -m fairtest.aw_run $OUT --phase $phase --instances $I --partition exid --cont $CONT $EXTRA > $OUT/$phase.log 2>&1 \
    || echo "$phase rc=$?" >> $OUT/chain.log
done
date '+%F %T analysis' >> $OUT/chain.log
$PY -m fairtest.aw_analysis $OUT $OUT/B80_SUMMARY.json --partition exid > $OUT/analysis.log 2>&1 || echo "analysis rc=$?" >> $OUT/chain.log
$PY -m fairtest.aw_export $OUT $OUT/B80_FIGDATA.json > $OUT/export.log 2>&1 || echo "export rc=$?" >> $OUT/chain.log
kill $PIN 2>/dev/null
docker stop -t 5 fairtest_aw_xa1 fairtest_aw_xa2 fairtest_aw_xa3 fairtest_aw_xr1 fairtest_aw_xr2 fairtest_aw_xr3 > /dev/null 2>&1
date '+%F %T done' >> $OUT/chain.log
