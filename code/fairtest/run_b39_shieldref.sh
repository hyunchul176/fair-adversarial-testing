#!/bin/sh
# B39 (design fixed before the run): the v2 policy stage with the shield replaying another reference; verdicts by the CC
C=$HOME/fairtest; R=$C/runtime-b39/code; O=$C/b39
mkdir -p $O; cd $R
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 SDL_VIDEODRIVER=dummy PYTHONWARNINGS=ignore
S=following:0,following:0.5,following:1,reference_idm:0,following_aeb:0,following_aeb:0.5,rss_sut:0,acc_prod:0,acc_prod_aeb:0
for ref in cc_r157 oracle_react0 oracle_t0; do
  for s in 1 2 3; do
    if [ $s = 1 ]; then M=$C/b17/fair_guarded2_s1/ppo_fair_guarded2_s1_1500000.zip; else M=$C/b23/fair_guarded2_s$s/ppo_fair_guarded2_s${s}_1500000.zip; fi
    FAIRTEST_SHIELD_REF=$ref FAIRTEST_SHIELD_MARGIN=0.5 FAIRTEST_DATA=$R/fairtest/data $C/env/bin/python -m fairtest.b17_rl eval $M \
      $O/final_${ref}_s$s.jsonl --partition development --workers 24 --commit --shield-window --suts $S > $O/final_${ref}_s$s.log 2>&1
  done
done
echo done > $C/b39_done.flag
