#!/bin/bash
# B59 (B59 design, available on request): the screened refinement (B56 bo_warm_g4s) against
# Autoware AW-default / AW-r157 for the same development cases B54 refined, B54's policy stage and its reruns reused
# (copied read-only), then reruns of the new refinement failures, the frozen analysis and the paired comparison with B54.
# Start after B57: flock -n /tmp/fairtest_b59.lock bash run_b59.sh
set -u
W=$HOME/fairtest/aw
OUT=$W/b59_dev
B54=$W/b54_dev
cd $HOME/fairtest/runtime-b59/code || exit 1
export FAIRTEST_DATA=$HOME/fairtest/runtime-b26/code/fairtest/data
PY=$HOME/fairtest/env/bin/python
mkdir -p $OUT
date '+%F %T start' >> $OUT/chain.log
cp -n $B54/policy_*.jsonl $OUT/
[ -e $OUT/repro_b54policy.jsonl ] || $PY -c "
import json, glob
with open('$OUT/repro_b54policy.jsonl', 'w') as f:
    for p in sorted(glob.glob('$B54/repro_*.jsonl')):
        for line in open(p):
            if line.strip() and json.loads(line)['source'].startswith('policy'):
                f.write(line)
"
SPECS="sa1:71:5471:aw_default sa2:72:5472:aw_default sa3:73:5473:aw_default sr1:74:5474:aw_r157 sr2:75:5475:aw_r157 sr3:76:5476:aw_r157"
I=$(echo $SPECS | tr ' ' ',')
date '+%F %T fleet' >> $OUT/chain.log
bash $W/aw_fleet.sh $SPECS > $OUT/fleet.log 2>&1
date '+%F %T refine' >> $OUT/chain.log
$PY -m fairtest.aw_run $OUT --phase refine --instances $I --refine-mode screened > $OUT/refine.log 2>&1 || echo "refine rc=$?" >> $OUT/chain.log
date '+%F %T repro' >> $OUT/chain.log
$PY -m fairtest.aw_run $OUT --phase repro --instances $I --repro-sources refine > $OUT/repro.log 2>&1 || echo "repro rc=$?" >> $OUT/chain.log
date '+%F %T analysis' >> $OUT/chain.log
$PY -m fairtest.aw_analysis $OUT $OUT/B59_SUMMARY.json > $OUT/analysis.log 2>&1 || echo "analysis rc=$?" >> $OUT/chain.log
$PY -m fairtest.aw_b59_compare $HOME/fairtest/aw/b55_dev/B54_SUMMARY_percase.json $OUT/B59_SUMMARY.json $B54 $OUT $OUT/B59_COMPARE.json > $OUT/compare.log 2>&1 || echo "compare rc=$?" >> $OUT/chain.log
$PY -m fairtest.aw_export $OUT $OUT/B59_FIGDATA.json > $OUT/export.log 2>&1 || echo "export rc=$?" >> $OUT/chain.log
# amendment 1 (before the run, audit A25): the unscreened refinement (bo_warm_g4j) again in the same run, on the same
# instances and cases, so that the primary comparison is within one run (W2'); B54 stays the secondary comparison
OUTJ=$W/b59_dev_g4j
mkdir -p $OUTJ
cp -n $B54/policy_*.jsonl $OUTJ/
cp -n $OUT/repro_b54policy.jsonl $OUTJ/
date '+%F %T refine_g4j' >> $OUT/chain.log
$PY -m fairtest.aw_run $OUTJ --phase refine --instances $I --refine-mode g4j > $OUTJ/refine.log 2>&1 || echo "refine_g4j rc=$?" >> $OUT/chain.log
date '+%F %T repro_g4j' >> $OUT/chain.log
$PY -m fairtest.aw_run $OUTJ --phase repro --instances $I --repro-sources refine > $OUTJ/repro.log 2>&1 || echo "repro_g4j rc=$?" >> $OUT/chain.log
date '+%F %T analysis_g4j' >> $OUT/chain.log
$PY -m fairtest.aw_analysis $OUTJ $OUTJ/B59J_SUMMARY.json > $OUTJ/analysis.log 2>&1 || echo "analysis_g4j rc=$?" >> $OUT/chain.log
$PY -m fairtest.aw_b59_compare $OUTJ/B59J_SUMMARY.json $OUT/B59_SUMMARY.json $OUTJ $OUT $OUT/B59_COMPARE_samerun.json > $OUT/compare_samerun.log 2>&1 || echo "compare_samerun rc=$?" >> $OUT/chain.log
docker stop -t 5 fairtest_aw_sa1 fairtest_aw_sa2 fairtest_aw_sa3 fairtest_aw_sr1 fairtest_aw_sr2 fairtest_aw_sr3 > /dev/null 2>&1
date '+%F %T done' >> $OUT/chain.log
