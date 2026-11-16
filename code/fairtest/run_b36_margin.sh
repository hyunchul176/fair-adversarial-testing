#!/bin/sh
# B36 (post hoc): reaction-time margin of the cut-in verdicts (b36_reaction_margin). Every failure the final agent v2
# reports and every judged baseline collision (the B32 job set) is replayed with the CC at other reaction times.
# dev and TGSIM first, then B9 and NGSIM (post hoc), then the production ACC SUTs.
C=$HOME/fairtest; R=$C/runtime-b36/code; O=$C/b36; E=$C/b23/eval
mkdir -p $O; cd $R
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 SDL_VIDEODRIVER=dummy PYTHONWARNINGS=ignore
P=$C/env/bin/python; W=24
X="$P -m fairtest.b36_reaction_margin"
FAIRTEST_DATA=$R/fairtest/data $X $O/dev.jsonl --final $C/b26/final --dataset dev --refine $C/b29/guarded_dev.jsonl \
  --baselines $E/collide_commit_s1_dev.jsonl $E/collide_commit_s2_dev.jsonl $E/collide_commit_s3_dev.jsonl \
  $E/cars_s1_dev.jsonl $E/cars_s2_dev.jsonl $E/cars_s3_dev.jsonl $E/dynasto_s1_dev.jsonl $E/dynasto_s2_dev.jsonl \
  $E/dynasto_s3_dev.jsonl $E/naive_fair_s1_dev.jsonl --workers $W > $O/dev.log 2>&1
T=$C/b28
FAIRTEST_DATA=$C/tgsim_data $X $O/tgsim.jsonl --final $C/b29/tgsim_final --dataset tgsim --refine $C/b29/guarded_tgsim.jsonl \
  --baselines $T/collide_commit_s1.jsonl $T/collide_commit_s2.jsonl $T/collide_commit_s3.jsonl $T/cars_s1.jsonl \
  $T/cars_s2.jsonl $T/cars_s3.jsonl $T/dynasto_s1.jsonl $T/dynasto_s2.jsonl $T/dynasto_s3.jsonl $T/naive_fair_s1.jsonl \
  --workers $W > $O/tgsim.log 2>&1
for ds in b9 ngsim; do
  case $ds in b9) DD=$C/b9_data;; ngsim) DD=$C/b13_data;; esac
  FAIRTEST_DATA=$DD $X $O/$ds.jsonl --final $C/b26/final --dataset $ds --refine $C/b29/guarded_$ds.jsonl \
    --baselines $E/collide_commit_s1_$ds.jsonl $E/collide_commit_s2_$ds.jsonl $E/collide_commit_s3_$ds.jsonl \
    $E/cars_s1_$ds.jsonl $E/cars_s2_$ds.jsonl $E/cars_s3_$ds.jsonl $E/dynasto_s1_$ds.jsonl $E/dynasto_s2_$ds.jsonl \
    $E/dynasto_s3_$ds.jsonl $E/naive_fair_s1_$ds.jsonl --workers $W > $O/$ds.log 2>&1
done
A=$C/b29/acc
for ds in dev b9 tgsim; do
  case $ds in dev) DD=$R/fairtest/data;; b9) DD=$C/b9_data;; tgsim) DD=$C/tgsim_data;; esac
  FAIRTEST_DATA=$DD $X $O/acc_$ds.jsonl --final $A --dataset $ds --refine $A/e3g4g_$ds.jsonl \
    --baselines $A/collide_commit_s1_$ds.jsonl $A/collide_commit_s2_$ds.jsonl $A/collide_commit_s3_$ds.jsonl \
    --workers $W > $O/acc_$ds.log 2>&1
done
echo done > $C/b36_done.flag
