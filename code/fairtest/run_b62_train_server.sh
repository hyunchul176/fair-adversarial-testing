#!/bin/sh
# B62 training on a lab server (the lab GPU runners were not available on 2026-09-30).
# Same command as the b17train job specs (design a06273a), seeds 1-3 in parallel at nice 10, then the evaluation chain.
#   flock -n /tmp/fairtest_b62.lock sh run_b62_train_server.sh
C=$HOME/fairtest; R=$C/runtime-b62/code; O=$C/b62
mkdir -p $O/models; cd $R
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 SDL_VIDEODRIVER=dummy PYTHONWARNINGS=ignore
export FAIRTEST_DATA=$R/fairtest/data FAIRTEST_HUMAN_POP=$R/fairtest/b62_train_population.json
P="nice -n 10 $C/env/bin/python"
log() { echo "$(date '+%F %T') $*" >> $O/chain.log; }
log train-start
for s in 1 2 3; do
  $P -m fairtest.b17_rl train $O/train_s$s --mode fair_guarded2 --steps 1500000 --envs 15 --seed $s --checkpoint 250000 \
    --init fairtest/b62_init/ppo_collide_commit_s${s}_750000.zip --train-suts fairtest/b62_train_population.json > $O/train_s$s.log 2>&1 &
done
wait
for s in 1 2 3; do
  f=$O/train_s$s/ppo_fair_guarded2_s${s}_1500000.zip
  [ -f $f ] || { log "train_s$s missing final model"; exit 1; }
  cp $f $O/models/
done
(cd $O/models && sha256sum ppo_fair_guarded2_s*_1500000.zip > models.sha256)
log train-done
sh $R/fairtest/run_b62_eval.sh
