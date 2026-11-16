#!/bin/sh
# B47 (post hoc): the final agent v2j on B9 / NGSIM / TGSIM -- the bo_warm_g4j refinement (same inputs and SUT order as
# the B29 v2 refinement) and the policy stage's joint realism (CC replays of the stored policy episodes)
C=$HOME/fairtest; R=$C/runtime-b40/code; O=$C/b47
mkdir -p $O; cd $R
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 SDL_VIDEODRIVER=dummy PYTHONWARNINGS=ignore
export FAIRTEST_DRL_SUT=$C/b25/drl_models/0.zip FAIRTEST_DRL_SUT_AT="$C/b25/drl_models/{k}.zip"
P=$C/env/bin/python
S=following:0,following:0.5,following:1,reference_idm:0,following_aeb:0,following_aeb:0.5,rss_sut:0,bc_human:0,ads@0.5:0,ads@2.0:0,rss_lat_sut:0,drl_sut:0,drl_sut@3:0,drl_sut@6:0,drl_sut@p1.5:0,drl_sut@p2:0,drl_sut@p2.5:0
for ds in b9 ngsim tgsim; do
  case $ds in b9) DD=$C/b9_data; CONT=$C/b9_b11_all7.jsonl;; ngsim) DD=$C/b13_data; CONT=$C/b13_b11.jsonl;; tgsim) DD=$C/tgsim_data; CONT=$C/b28/b11.jsonl;; esac
  FAIRTEST_DATA=$DD $P -m fairtest.b23_search $O/joint_$ds.jsonl --cont $CONT --partition all --methods bo_warm_g4j --suts $S \
    --workers 24 > $O/joint_$ds.log 2>&1
  FAIRTEST_DATA=$DD $P -m fairtest.b45_joint_realism replay $O/policy_replay_$ds.jsonl --jobs $C --dataset $ds --workers 24 \
    > $O/policy_replay_$ds.log 2>&1
done
echo done > $C/b47_done.flag
