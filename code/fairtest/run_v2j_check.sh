#!/bin/sh
# v2j callable vs the B46 composition (W5 P1-C): fair_agent_v2 --joint, policy seed 1, the first 10 dev cases x 17 SUTs
# (the B29 v2 check was 170/170). Compared locally with b46_analysis's unified composition.
C=$HOME/fairtest; R=$C/runtime-b48/code; O=$C/b47
mkdir -p $O; cd $R
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 SDL_VIDEODRIVER=dummy PYTHONWARNINGS=ignore
export FAIRTEST_DRL_SUT=$C/b25/drl_models/0.zip FAIRTEST_DRL_SUT_AT="$C/b25/drl_models/{k}.zip"
S=following:0,following:0.5,following:1,reference_idm:0,following_aeb:0,following_aeb:0.5,rss_sut:0,bc_human:0,ads@0.5:0,ads@2.0:0,rss_lat_sut:0,drl_sut:0,drl_sut@3:0,drl_sut@6:0,drl_sut@p1.5:0,drl_sut@p2:0,drl_sut@p2.5:0
FAIRTEST_DATA=$R/fairtest/data $C/env/bin/python -m fairtest.fair_agent_v2 batch \
  $C/b17/fair_guarded2_s1/ppo_fair_guarded2_s1_1500000.zip $O/agent_v2j_check_dev.jsonl --partition development --suts $S \
  --cont $C/b23/exposed_cont_search_all7.jsonl --cases 10 --workers 16 --joint > $O/agent_v2j_check_dev.log 2>&1
echo done > $C/v2j_check_done.flag
