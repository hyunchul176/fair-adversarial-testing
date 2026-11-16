#!/bin/sh
# B37 (design fixed before the run, commit 8478a5b): headway-floor intervention on the collision-seeking cut-in attacker
C=$HOME/fairtest; R=$C/runtime-b37/code; O=$C/b37
mkdir -p $O; cd $R
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 SDL_VIDEODRIVER=dummy PYTHONWARNINGS=ignore
FAIRTEST_DATA=$R/fairtest/data $C/env/bin/python -m fairtest.b37_headway $O/headway_dev.jsonl --partition development \
  --hmins 0,0.5,1,1.5 --workers 24 > $O/headway_dev.log 2>&1
echo done > $C/b37_done.flag
