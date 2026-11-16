#!/bin/sh
# B52 (B52 design, available on request): train the fair_guarded2 adversary
# against the feasibility references (oracle_t0, oracle_react0), seeds 1-3, from each seed's collide_commit 750k
# checkpoint (as the v2j policy), 1.5M steps, 15 envs; the two references of one seed run together. A finished run is
# not repeated. Then the evaluation chain (run_b52_eval.sh).
C=$HOME/fairtest; R=$C/runtime-b52/code; O=$C/b52
mkdir -p $O; cd $R
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 SDL_VIDEODRIVER=dummy PYTHONWARNINGS=ignore FAIRTEST_DATA=$R/fairtest/data
P=$C/env/bin/python
for s in 1 2 3; do
  if [ $s = 1 ]; then I=$C/b17/collide_commit_s1/ppo_collide_commit_s1_750000.zip; else I=$C/b23/collide_commit_s$s/ppo_collide_commit_s${s}_750000.zip; fi
  for ref in oracle_t0 oracle_react0; do
    if [ $ref = oracle_t0 ]; then tag=feast0; else tag=feasr0; fi
    [ -e $O/${tag}_s$s/ppo_${tag}_s${s}_1500000.zip ] && continue
    $P -m fairtest.b52_feasible train $O/${tag}_s$s --ref $ref --init $I --steps 1500000 --envs 15 --seed $s > $O/${tag}_s$s.log 2>&1 &
  done
  wait
done
echo done > $C/b52_train_done.flag
sh $C/run_b52_eval.sh
