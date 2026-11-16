#!/bin/sh
# B33 addition (after the design, reported as such): dense (c)-boundary probe of constant-speed cut-ins for the CC
C=$HOME/fairtest; R=$C/runtime-b33/code; O=$C/b33
cd $R
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 SDL_VIDEODRIVER=dummy PYTHONWARNINGS=ignore
P=$C/env/bin/python; X="$P -m fairtest.b33_mandate probe"
FAIRTEST_DATA=$R/fairtest/data $X $O/probe_cc_dev.jsonl --partition development --workers 24 > $O/probe_cc_dev.log 2>&1
FAIRTEST_DATA=$C/tgsim_data $X $O/probe_cc_tgsim.jsonl --workers 24 > $O/probe_cc_tgsim.log 2>&1
FAIRTEST_DATA=$C/b9_data $X $O/probe_cc_b9.jsonl --workers 24 > $O/probe_cc_b9.log 2>&1
echo done > $C/b33_probe_done.flag
