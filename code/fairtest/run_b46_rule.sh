#!/bin/sh
# B46: bo_warm_g4j for the ten rule-based SUTs (moved from the local run, same code and streams)
# rule-based SUTs ran locally with the same code. Streams keep their position in the 17-SUT order (--only).
C=$HOME/fairtest; R=$C/runtime-b40/code; O=$C/b46
mkdir -p $O; cd $R
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 SDL_VIDEODRIVER=dummy PYTHONWARNINGS=ignore
export FAIRTEST_DRL_SUT=$C/b25/drl_models/0.zip FAIRTEST_DRL_SUT_AT="$C/b25/drl_models/{k}.zip"
S=following:0,following:0.5,following:1,reference_idm:0,following_aeb:0,following_aeb:0.5,rss_sut:0,bc_human:0,ads@0.5:0,ads@2.0:0,rss_lat_sut:0,drl_sut:0,drl_sut@3:0,drl_sut@6:0,drl_sut@p1.5:0,drl_sut@p2:0,drl_sut@p2.5:0
ONLY=following_H0,following_H0.5,following_H1,reference_idm_H0,following_aeb_H0,following_aeb_H0.5,rss_sut_H0,ads@0.5_H0,ads@2.0_H0,rss_lat_sut_H0
FAIRTEST_DATA=$R/fairtest/data $C/env/bin/python -m fairtest.b23_search $O/joint_dev_rule.jsonl \
  --cont $C/b23/exposed_cont_search_all7.jsonl --partition development --methods bo_warm_g4j --suts $S --only $ONLY \
  --workers 24 > $O/joint_dev_rule.log 2>&1
echo done > $C/b46r_done.flag
