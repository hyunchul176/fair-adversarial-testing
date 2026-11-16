#!/bin/bash
# B59 amendment 1 (8e33215, fixed before the run): the unscreened refinement bo_warm_g4j on the same instances, cases,
# boundary plans and random streams as the screened arm. The main B59 chain (07:02-09:37, 2026-09-30) started from the
# pre-amendment script while the VPN was down, so this arm runs on its own afterwards (disclosed in B59 RESULT §0, §5).
# Refinement and reruns use runtime-b59 (the code of the screened arm); the comparison uses the corrected aw_b59_compare
# (ba00ed9, the search's joint-support threshold) from runtime-b62.
#   flock -n /tmp/fairtest_b59g.lock bash run_b59_g4j.sh
set -u
W=$HOME/fairtest/aw
OUT=$W/b59_dev
OUTJ=$W/b59_dev_g4j
B54=$W/b54_dev
export FAIRTEST_DATA=$HOME/fairtest/runtime-b26/code/fairtest/data
PY=$HOME/fairtest/env/bin/python
[ -e $OUTJ/refine_sa1.jsonl ] && { echo "b59_dev_g4j already has refinement output; not rerunning" >&2; exit 1; }
mkdir -p $OUTJ
cp -n $B54/policy_*.jsonl $OUTJ/
cp -n $OUT/repro_b54policy.jsonl $OUTJ/
SPECS="sa1:71:5471:aw_default sa2:72:5472:aw_default sa3:73:5473:aw_default sr1:74:5474:aw_r157 sr2:75:5475:aw_r157 sr3:76:5476:aw_r157"
I=$(echo $SPECS | tr ' ' ',')
cd $HOME/fairtest/runtime-b59/code || exit 1
date '+%F %T g4j start' >> $OUT/chain.log
bash $W/aw_fleet.sh $SPECS > $OUTJ/fleet.log 2>&1
date '+%F %T refine_g4j' >> $OUT/chain.log
$PY -m fairtest.aw_run $OUTJ --phase refine --instances $I --refine-mode g4j > $OUTJ/refine.log 2>&1 || echo "refine_g4j rc=$?" >> $OUT/chain.log
date '+%F %T repro_g4j' >> $OUT/chain.log
$PY -m fairtest.aw_run $OUTJ --phase repro --instances $I --repro-sources refine > $OUTJ/repro.log 2>&1 || echo "repro_g4j rc=$?" >> $OUT/chain.log
date '+%F %T analysis_g4j' >> $OUT/chain.log
$PY -m fairtest.aw_analysis $OUTJ $OUTJ/B59J_SUMMARY.json > $OUTJ/analysis.log 2>&1 || echo "analysis_g4j rc=$?" >> $OUT/chain.log
cd $HOME/fairtest/runtime-b62/code || exit 1
$PY -m fairtest.aw_b59_compare $OUTJ/B59J_SUMMARY.json $OUT/B59_SUMMARY.json $OUTJ $OUT $OUT/B59_COMPARE_samerun.json > $OUT/compare_samerun.log 2>&1 || echo "compare_samerun rc=$?" >> $OUT/chain.log
docker stop -t 5 fairtest_aw_sa1 fairtest_aw_sa2 fairtest_aw_sa3 fairtest_aw_sr1 fairtest_aw_sr2 fairtest_aw_sr3 > /dev/null 2>&1
date '+%F %T g4j done' >> $OUT/chain.log
