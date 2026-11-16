#!/bin/sh
# B51 (B51 design, available on request), development: bo_warm_g4js (severity floor
# 3 m/s) for the 62 development cases x 17 SUTs, and the closing speed of the v2j side (every policy-stage contact of
# seeds 1-3 and every bo_warm_g4j success of B46). Check before the run: with the floor off (v_min -1000) the refinement
# equals the stored B46 bo_warm_g4j on 3 cases x 17 SUTs (51/51, b51_check/). A finished file is not repeated.
C=$HOME/fairtest; R=$C/runtime-b51/code; O=$C/b51
mkdir -p $O; cd $R
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 SDL_VIDEODRIVER=dummy PYTHONWARNINGS=ignore FAIRTEST_DATA=$R/fairtest/data
export FAIRTEST_DRL_SUT=$C/b25/drl_models/0.zip FAIRTEST_DRL_SUT_AT="$C/b25/drl_models/{k}.zip"
P=$C/env/bin/python
S=following:0,following:0.5,following:1,reference_idm:0,following_aeb:0,following_aeb:0.5,rss_sut:0,bc_human:0,ads@0.5:0,ads@2.0:0,rss_lat_sut:0,drl_sut:0,drl_sut@3:0,drl_sut@6:0,drl_sut@p1.5:0,drl_sut@p2:0,drl_sut@p2.5:0
[ -e $O/refine_g4js_dev.jsonl ] || { $P -m fairtest.b51_severity refine $O/refine_g4js_dev.part --cont $C/b23/exposed_cont_search_all7.jsonl \
  --partition development --suts $S --workers 12 > $O/refine_dev.log 2>&1 && mv $O/refine_g4js_dev.part $O/refine_g4js_dev.jsonl; }
[ -e $O/closing_dev.jsonl ] || { $P -m fairtest.b51_severity closing $O/closing_dev.part \
  --policy $C/b26/final/final_s1_dev.jsonl $C/b26/final/final_s2_dev.jsonl $C/b26/final/final_s3_dev.jsonl \
  --refine $C/b46/joint_dev_rule.jsonl $C/b46/joint_dev_learned.jsonl --workers 12 > $O/closing_dev.log 2>&1 \
  && mv $O/closing_dev.part $O/closing_dev.jsonl; }
echo done > $C/b51_dev_done.flag
