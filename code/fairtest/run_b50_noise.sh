#!/bin/sh
# B50 (B50 design, available on request): the v2j callable (policy seed 1) against
# the perception-noise SUT ladder on the development cases, then the reproducibility replays of its reported failures.
C=$HOME/fairtest; R=$C/runtime-b50/code; O=$C/b50
mkdir -p $O; cd $R
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 SDL_VIDEODRIVER=dummy PYTHONWARNINGS=ignore FAIRTEST_DATA=$R/fairtest/data
P=$C/env/bin/python
[ -e $O/attacks.jsonl ] || { $P -m fairtest.b50_noise attack $C/b17/fair_guarded2_s1/ppo_fair_guarded2_s1_1500000.zip \
  $O/attacks.part.jsonl --cont $C/b23/exposed_cont_search_all7.jsonl --workers 24 > $O/attacks.log 2>&1 \
  && mv $O/attacks.part.jsonl $O/attacks.jsonl; }
[ -e $O/repro.jsonl ] || { $P -m fairtest.b50_noise repro $O/attacks.jsonl $O/repro.part.jsonl --reps 20 --workers 24 \
  > $O/repro.log 2>&1 && mv $O/repro.part.jsonl $O/repro.jsonl; }
echo done > $C/b50_done.flag
