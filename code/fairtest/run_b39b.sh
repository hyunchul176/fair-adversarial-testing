#!/bin/sh
# B39b (design fixed before the run): the v2 refinement with a feasibility-style reference, re-judged by the R157 CC
C=$HOME/fairtest; R=$C/runtime-b39/code; O=$C/b39
cd $R
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 SDL_VIDEODRIVER=dummy PYTHONWARNINGS=ignore
export FAIRTEST_DRL_SUT=$C/b25/drl_models/0.zip FAIRTEST_DRL_SUT_AT="$C/b25/drl_models/{k}.zip"
S=following:0,following:0.5,following:1,reference_idm:0,following_aeb:0,following_aeb:0.5,rss_sut:0,bc_human:0,ads@0.5:0,ads@2.0:0,rss_lat_sut:0,drl_sut:0,drl_sut@3:0,drl_sut@6:0,drl_sut@p1.5:0,drl_sut@p2:0,drl_sut@p2.5:0
for ref in oracle_react0 oracle_t0; do
  FAIRTEST_JUDGE_CC=$ref FAIRTEST_DATA=$R/fairtest/data $C/env/bin/python -m fairtest.b23_search $O/guarded_dev_$ref.jsonl \
    --cont $C/b23/exposed_cont_search_all7.jsonl --partition development --methods bo_warm_g4g --suts $S --workers 24 \
    > $O/guarded_dev_$ref.log 2>&1
  FAIRTEST_DATA=$R/fairtest/data $C/env/bin/python -m fairtest.b39b_rejudge $O/rejudge_$ref.jsonl $O/guarded_dev_$ref.jsonl \
    --workers 24 > $O/rejudge_$ref.log 2>&1
done
echo done > $C/b39b_done.flag
