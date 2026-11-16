#!/bin/sh
# B33 (post hoc): R157 5.2.5.2 explicit cut-in requirement labels (b33_mandate), design fixed before the run
# (B33 design, available on request). B: the B32 job set (reported failures, baseline
# collisions); A: CC grid (dev, TGSIM, B9 post hoc); C: ladder + production-ACC SUT grid (dev).
C=$HOME/fairtest; R=$C/runtime-b33/code; O=$C/b33; E=$C/b23/eval; T=$C/b28; A=$C/b29/acc
mkdir -p $O; cd $R
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 SDL_VIDEODRIVER=dummy PYTHONWARNINGS=ignore
export FAIRTEST_DRL_SUT=$C/b25/drl_models/0.zip FAIRTEST_DRL_SUT_AT="$C/b25/drl_models/{k}.zip"
P=$C/env/bin/python; W=24
X="$P -m fairtest.b33_mandate"
# B. reported failures and baseline collisions
FAIRTEST_DATA=$R/fairtest/data $X jobs $O/jobs_dev.jsonl --final $C/b26/final --dataset dev --refine $C/b29/guarded_dev.jsonl \
  --baselines $E/collide_commit_s1_dev.jsonl $E/collide_commit_s2_dev.jsonl $E/collide_commit_s3_dev.jsonl \
  $E/cars_s1_dev.jsonl $E/cars_s2_dev.jsonl $E/cars_s3_dev.jsonl $E/dynasto_s1_dev.jsonl $E/dynasto_s2_dev.jsonl \
  $E/dynasto_s3_dev.jsonl $E/naive_fair_s1_dev.jsonl --workers $W > $O/jobs_dev.log 2>&1
FAIRTEST_DATA=$C/tgsim_data $X jobs $O/jobs_tgsim.jsonl --final $C/b29/tgsim_final --dataset tgsim --refine $C/b29/guarded_tgsim.jsonl \
  --baselines $T/collide_commit_s1.jsonl $T/collide_commit_s2.jsonl $T/collide_commit_s3.jsonl $T/cars_s1.jsonl \
  $T/cars_s2.jsonl $T/cars_s3.jsonl $T/dynasto_s1.jsonl $T/dynasto_s2.jsonl $T/dynasto_s3.jsonl $T/naive_fair_s1.jsonl \
  --workers $W > $O/jobs_tgsim.log 2>&1
for ds in b9 ngsim; do
  case $ds in b9) DD=$C/b9_data;; ngsim) DD=$C/b13_data;; esac
  FAIRTEST_DATA=$DD $X jobs $O/jobs_$ds.jsonl --final $C/b26/final --dataset $ds --refine $C/b29/guarded_$ds.jsonl \
    --baselines $E/collide_commit_s1_$ds.jsonl $E/collide_commit_s2_$ds.jsonl $E/collide_commit_s3_$ds.jsonl \
    $E/cars_s1_$ds.jsonl $E/cars_s2_$ds.jsonl $E/cars_s3_$ds.jsonl $E/dynasto_s1_$ds.jsonl $E/dynasto_s2_$ds.jsonl \
    $E/dynasto_s3_$ds.jsonl $E/naive_fair_s1_$ds.jsonl --workers $W > $O/jobs_$ds.log 2>&1
done
for ds in dev b9 tgsim; do
  case $ds in dev) DD=$R/fairtest/data;; b9) DD=$C/b9_data;; tgsim) DD=$C/tgsim_data;; esac
  FAIRTEST_DATA=$DD $X jobs $O/jobs_acc_$ds.jsonl --final $A --dataset $ds --refine $A/e3g4g_$ds.jsonl \
    --baselines $A/collide_commit_s1_$ds.jsonl $A/collide_commit_s2_$ds.jsonl $A/collide_commit_s3_$ds.jsonl \
    --workers $W > $O/jobs_acc_$ds.log 2>&1
done
# A. the CC's own grid episodes
FAIRTEST_DATA=$R/fairtest/data $X grid $O/grid_cc_dev.jsonl --targets cc_r157:0 --partition development --workers $W > $O/grid_cc_dev.log 2>&1
FAIRTEST_DATA=$C/tgsim_data $X grid $O/grid_cc_tgsim.jsonl --targets cc_r157:0 --workers $W > $O/grid_cc_tgsim.log 2>&1
FAIRTEST_DATA=$C/b9_data $X grid $O/grid_cc_b9.jsonl --targets cc_r157:0 --workers $W > $O/grid_cc_b9.log 2>&1
# C. ladder + production ACC SUTs on the dev grid
FAIRTEST_DATA=$R/fairtest/data $X grid $O/grid_sut_dev.jsonl --partition development --workers $W \
  --targets following:0,following:0.5,following:1,reference_idm:0,following_aeb:0,following_aeb:0.5,rss_sut:0,acc_prod:0,acc_prod_aeb:0 \
  > $O/grid_sut_dev.log 2>&1
echo done > $C/b33_done.flag
