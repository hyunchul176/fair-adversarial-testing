#!/bin/sh
# B28 TGSIM external evaluation of the final agent (B28 protocol section 7, available on request). Run only after the freeze commit,
# from the frozen code in runtime-b28. Every step writes its own log; nothing is overwritten.
set -e
C=$HOME/fairtest
R=$C/runtime-b28/code
O=$C/b28
D=$C/tgsim_data
mkdir -p $O
[ -e $O/events.jsonl ] && { echo "b28 already extracted; refusing to rerun"; exit 1; }
cd $R
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 SDL_VIDEODRIVER=dummy PYTHONWARNINGS=ignore
P=$C/env/bin/python
OURS1=$C/b17/fair_guarded2_s1/ppo_fair_guarded2_s1_1500000.zip
# 0. server-side hashes of the frozen artefacts, before any extraction
sha256sum $C/tgsim/I294_L1_final.csv $C/tgsim/I294_L2_final.csv $OURS1 $C/b23/fair_guarded2_s2/ppo_fair_guarded2_s2_1500000.zip \
  $C/b23/fair_guarded2_s3/ppo_fair_guarded2_s3_1500000.zip $C/b17/collide_commit_s1/ppo_collide_commit_s1_1500000.zip \
  $C/b23/collide_commit_s2/ppo_collide_commit_s2_1500000.zip $C/b23/collide_commit_s3/ppo_collide_commit_s3_1500000.zip \
  $C/b23/cars_s1/ppo_cars_s1_2250000.zip $C/b23/cars_s2/ppo_cars_s2_2250000.zip $C/b23/cars_s3/ppo_cars_s3_2250000.zip \
  $C/b23/dynasto_s1/ppo_dynasto_s1_2250000.zip $C/b23/dynasto_s2/ppo_dynasto_s2_2250000.zip $C/b23/dynasto_s3/ppo_dynasto_s3_2250000.zip \
  $C/b17/fair_s1/ppo_fair_s1_1500000.zip $C/b25/drl_models/*.zip $C/b13_data/bc_human.pkl fairtest/*.py > $O/freeze_server_sha256.txt
# 1. extraction and sealing
$P -m fairtest.tgsim $C/tgsim $O/events.jsonl > $O/extract.log 2>&1
$P -m fairtest.external_bundle pool $O/events.jsonl $D --template $C/b13_data --source "TGSIM I-294 L1+L2 (data.transportation.gov 7zjf-a4zf, tpsq-zrwa)" > $O/pool.log 2>&1
sha256sum $O/events.jsonl $D/case_pool.json >> $O/freeze_server_sha256.txt
cp $D/case_pool.json $O/case_pool.json
N=$(python3 -c "import json; print(len(json.load(open('$D/case_pool.json'))['cases']))")
echo "cases $N" > $O/cohort_size.txt
[ "$N" -gt 0 ] || { echo "empty cohort"; exit 1; }
# 2. CC grid -> warm starts -> continuous CC-boundary search (the refinement's starting plan)
FAIRTEST_DATA=$D $P -m fairtest.runner landscape $O/grid_cc.jsonl --targets cc_r157:0 --policies all --workers 16 > $O/grid_cc.log 2>&1
FAIRTEST_DATA=$D $P -m fairtest.external_bundle warm $O/grid_cc.jsonl $D > $O/warm.log 2>&1
FAIRTEST_DATA=$D $P -m fairtest.b11_search $O/b11.jsonl --suts following:0 --workers 16 > $O/b11.log 2>&1
# 3. policy stage: final agent, the same policy without the shield, baselines
export FAIRTEST_DRL_SUT=$C/b25/drl_models/0.zip FAIRTEST_DRL_SUT_AT="$C/b25/drl_models/{k}.zip"
LADDER=following:0,following:0.5,following:1,reference_idm:0,following_aeb:0,following_aeb:0.5,rss_sut:0
SUTS=$LADDER,bc_human:0,ads@0.5:0,ads@2.0:0,rss_lat_sut:0,drl_sut:0,drl_sut@3:0,drl_sut@6:0,drl_sut@p1.5:0,drl_sut@p2:0,drl_sut@p2.5:0
EV="$P -m fairtest.b17_rl eval"
for s in 1 2 3; do
  if [ $s = 1 ]; then M=$OURS1; CM=$C/b17/collide_commit_s1/ppo_collide_commit_s1_1500000.zip; else M=$C/b23/fair_guarded2_s$s/ppo_fair_guarded2_s${s}_1500000.zip; CM=$C/b23/collide_commit_s$s/ppo_collide_commit_s${s}_1500000.zip; fi
  FAIRTEST_SHIELD_MARGIN=0.5 FAIRTEST_DATA=$D $EV $M $O/final_s$s.jsonl --partition all --workers 16 --commit --shield-window --suts $SUTS > $O/final_s$s.log 2>&1
  FAIRTEST_DATA=$D $EV $M $O/noshield_s$s.jsonl --partition all --workers 16 --commit --suts $SUTS > $O/noshield_s$s.log 2>&1
  FAIRTEST_DATA=$D $EV $CM $O/collide_commit_s$s.jsonl --partition all --workers 16 --commit --suts $SUTS > $O/collide_commit_s$s.log 2>&1
  FAIRTEST_DATA=$D $EV $C/b23/cars_s$s/ppo_cars_s${s}_2250000.zip $O/cars_s$s.jsonl --partition all --workers 16 --suts $SUTS > $O/cars_s$s.log 2>&1
  FAIRTEST_DATA=$D $EV $C/b23/dynasto_s$s/ppo_dynasto_s${s}_2250000.zip $O/dynasto_s$s.jsonl --partition all --workers 16 --suts $SUTS > $O/dynasto_s$s.log 2>&1
done
FAIRTEST_DATA=$D $EV $C/b17/fair_s1/ppo_fair_s1_1500000.zip $O/naive_fair_s1.jsonl --partition all --workers 16 --suts $SUTS > $O/naive_fair_s1.log 2>&1
# 4. G4 margin of every fair failure
FAIRTEST_DATA=$D $P -m fairtest.b23_margin $O/margin.jsonl --evals $O/*_s[0-9].jsonl --workers 16 > $O/margin.log 2>&1
# 5. refinement (from the CC-boundary plan of step 2)
FAIRTEST_DATA=$D $P -m fairtest.b23_search $O/e3_all.jsonl --cont $O/b11.jsonl --partition all --methods bo_warm_g4 --suts $SUTS --workers 16 > $O/e3_all.log 2>&1
FAIRTEST_DATA=$D $P -m fairtest.b23_search $O/e3_ladder.jsonl --cont $O/b11.jsonl --partition all --methods bo_cold_g4,random_g4 --suts $LADDER --workers 16 > $O/e3_ladder.log 2>&1
# 6. frozen analysis
$P -m fairtest.b28_analysis $O $O/b28_result.json > $O/analysis.log 2>&1
echo done > $C/b28_done.flag
