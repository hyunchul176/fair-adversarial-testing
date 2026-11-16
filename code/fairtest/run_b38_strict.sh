#!/bin/sh
# B38 (design fixed before the run): the v2 refinement (bo_warm_g4g) with a slower careful driver as the reference
C=$HOME/fairtest; R=$C/runtime-b38/code; O=$C/b38
mkdir -p $O; cd $R
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 SDL_VIDEODRIVER=dummy PYTHONWARNINGS=ignore
export FAIRTEST_DRL_SUT=$C/b25/drl_models/0.zip FAIRTEST_DRL_SUT_AT="$C/b25/drl_models/{k}.zip"
S=following:0,following:0.5,following:1,reference_idm:0,following_aeb:0,following_aeb:0.5,rss_sut:0,bc_human:0,ads@0.5:0,ads@2.0:0,rss_lat_sut:0,drl_sut:0,drl_sut@3:0,drl_sut@6:0,drl_sut@p1.5:0,drl_sut@p2:0,drl_sut@p2.5:0
for r in 1 1.25 1.5; do
  FAIRTEST_JUDGE_CC=cc_r157@r$r FAIRTEST_DATA=$R/fairtest/data $C/env/bin/python -m fairtest.b23_search $O/guarded_dev_r$r.jsonl \
    --cont $C/b23/exposed_cont_search_all7.jsonl --partition development --methods bo_warm_g4g --suts $S --workers 24 \
    > $O/guarded_dev_r$r.log 2>&1
done
echo done > $C/b38_done.flag
