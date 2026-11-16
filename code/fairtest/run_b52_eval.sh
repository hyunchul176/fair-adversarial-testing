#!/bin/sh
# B52 evaluation: the frozen b17_rl eval --commit (policy stage, one query, no shield) of the six feasibility adversaries
# x 17 SUTs on the development cases (the test fixed before the run), then B9 / TGSIM / MiTra (post hoc); G4 margins per
# dataset. Every collision carries the CC and reference re-judgements (b17_rl REFS). A finished file is not repeated.
C=$HOME/fairtest; R=$C/runtime-b52/code; O=$C/b52
cd $R
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 SDL_VIDEODRIVER=dummy PYTHONWARNINGS=ignore
export FAIRTEST_DRL_SUT=$C/b25/drl_models/0.zip FAIRTEST_DRL_SUT_AT="$C/b25/drl_models/{k}.zip"
P=$C/env/bin/python
S=following:0,following:0.5,following:1,reference_idm:0,following_aeb:0,following_aeb:0.5,rss_sut:0,bc_human:0,ads@0.5:0,ads@2.0:0,rss_lat_sut:0,drl_sut:0,drl_sut@3:0,drl_sut@6:0,drl_sut@p1.5:0,drl_sut@p2:0,drl_sut@p2.5:0
for ds in dev b9 tgsim mitra; do
  case $ds in
    dev) DD=$R/fairtest/data; PART=development;;
    b9) DD=$C/b9_data; PART=all;;
    tgsim) DD=$C/tgsim_data; PART=all;;
    mitra) DD=$C/mitra_data; PART=all;;
  esac
  E=$O/eval_$ds; mkdir -p $E
  for tag in feast0 feasr0; do
    for s in 1 2 3; do
      out=$E/${tag}_s$s.jsonl
      [ -e $out ] && continue
      FAIRTEST_DATA=$DD $P -m fairtest.b17_rl eval $O/${tag}_s$s/ppo_${tag}_s${s}_1500000.zip $E/${tag}_s$s.part --partition $PART \
        --commit --suts $S --workers 24 > $E/${tag}_s$s.log 2>&1 && mv $E/${tag}_s$s.part $out
    done
  done
  [ -e $E/margin.jsonl ] || FAIRTEST_DATA=$DD $P -m fairtest.b23_margin $E/margin.jsonl --evals $E/*_s[0-9].jsonl --workers 24 > $E/margin.log 2>&1
done
echo done > $C/b52_eval_done.flag
