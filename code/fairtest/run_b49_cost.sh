#!/bin/sh
# B49 (W5 P2-A, runtime-b49 = runtime-b48 + current code, B49 design available on request): the v2j callable (fair_agent_v2
# --joint) on all 62 development cases x 17 SUTs x policy seeds 1-3, recording its cost counter and wall-clock. The stored
# CC-boundary plans are reused (--cont), as in B46 and the v2j check. A finished seed is not rerun.
C=$HOME/fairtest; R=$C/runtime-b49/code; O=$C/b49
mkdir -p $O; cd $R
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 SDL_VIDEODRIVER=dummy PYTHONWARNINGS=ignore
export FAIRTEST_DRL_SUT=$C/b25/drl_models/0.zip FAIRTEST_DRL_SUT_AT="$C/b25/drl_models/{k}.zip"
S=following:0,following:0.5,following:1,reference_idm:0,following_aeb:0,following_aeb:0.5,rss_sut:0,bc_human:0,ads@0.5:0,ads@2.0:0,rss_lat_sut:0,drl_sut:0,drl_sut@3:0,drl_sut@6:0,drl_sut@p1.5:0,drl_sut@p2:0,drl_sut@p2.5:0
for s in 1 2 3; do
  if [ $s = 1 ]; then M=$C/b17/fair_guarded2_s1/ppo_fair_guarded2_s1_1500000.zip; else M=$C/b23/fair_guarded2_s$s/ppo_fair_guarded2_s${s}_1500000.zip; fi
  [ -e $O/agent_v2j_dev_s$s.jsonl ] && continue
  FAIRTEST_DATA=$R/fairtest/data $C/env/bin/python -m fairtest.fair_agent_v2 batch $M $O/agent_v2j_dev_s$s.part.jsonl \
    --partition development --suts $S --cont $C/b23/exposed_cont_search_all7.jsonl --workers 24 --joint \
    > $O/agent_v2j_dev_s$s.log 2>&1 && mv $O/agent_v2j_dev_s$s.part.jsonl $O/agent_v2j_dev_s$s.jsonl
done
echo done > $C/b49_done.flag
