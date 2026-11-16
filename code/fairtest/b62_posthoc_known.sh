#!/bin/sh
# B62 post hoc diagnostic (after Q1-Q4 were judged): the population-trained policy against B61's SUTs of known capability
# (in-family for B62: C with one factor changed), same policy-stage settings; compared with B61's v2j policy stage.
C=$HOME/fairtest; R=$C/runtime-b62/code; O=$C/b62/posthoc_known
mkdir -p $O; cd $R
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 SDL_VIDEODRIVER=dummy PYTHONWARNINGS=ignore
export FAIRTEST_DATA=$C/runtime-b40/code/fairtest/data FAIRTEST_HUMAN_POP=$R/fairtest/b61_known_suts.json
P="nice -n 10 $C/env/bin/python"
K=cc_r157:0,cc_r157@r0.5:0,cc_r157@r1:0,cc_r157@r1.25:0,cc_r157@r1.75:0,cc_r157@h4:0,cc_r157@h0:0,cc_r157@h1:0,cc_r157@h2:0,cc_r157@h3:0
echo "$(date '+%F %T') start" >> $O/chain.log
for s in 1 2 3; do
  FAIRTEST_SHIELD_MARGIN=0.5 $P -m fairtest.b17_rl eval $C/b62/models/ppo_fair_guarded2_s${s}_1500000.zip $O/final_s$s.jsonl \
    --partition development --workers 8 --commit --shield-window --suts $K > $O/final_s$s.log 2>&1 || echo "final_s$s rc=$?" >> $O/chain.log
done
$P -m fairtest.b23_margin $O/margin.jsonl --evals $O/final_s1.jsonl $O/final_s2.jsonl $O/final_s3.jsonl --workers 8 > $O/margin.log 2>&1 || echo "margin rc=$?" >> $O/chain.log
echo "$(date '+%F %T') done" >> $O/chain.log
