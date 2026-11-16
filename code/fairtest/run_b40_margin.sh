#!/bin/sh
# B40 (design fixed before the run): margin-escalating refinement on the dev cases (v2 stage 1 reproduced first)
C=$HOME/fairtest; R=$C/runtime-b40/code; O=$C/b40
mkdir -p $O; cd $R
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 SDL_VIDEODRIVER=dummy PYTHONWARNINGS=ignore
export FAIRTEST_DRL_SUT=$C/b25/drl_models/0.zip FAIRTEST_DRL_SUT_AT="$C/b25/drl_models/{k}.zip"
S=following:0,following:0.5,following:1,reference_idm:0,following_aeb:0,following_aeb:0.5,rss_sut:0,bc_human:0,ads@0.5:0,ads@2.0:0,rss_lat_sut:0,drl_sut:0,drl_sut@3:0,drl_sut@6:0,drl_sut@p1.5:0,drl_sut@p2:0,drl_sut@p2.5:0
FAIRTEST_DATA=$R/fairtest/data $C/env/bin/python -m fairtest.b40_margin_search $O/margin_dev.jsonl \
  --cont $C/b23/exposed_cont_search_all7.jsonl --partition development --suts $S --workers 24 > $O/margin_dev.log 2>&1
echo done > $C/b40_done.flag
