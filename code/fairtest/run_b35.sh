#!/bin/sh
# B35 TGSIM I-294 car-following extraction (B35 protocol, available on request). Run only after the freeze commit, from the frozen code in
# runtime-b35. The 1-D simulations run locally on the sealed case files.
set -e
C=$HOME/fairtest; R=$C/runtime-b35/code; O=$C/b35
mkdir -p $O
[ -e $O/dec_cases.json ] && { echo "b35 already extracted; refusing to rerun"; exit 1; }
cd $R
P=$C/env/bin/python
sha256sum $C/tgsim/I294_L1_final.csv $C/tgsim/I294_L2_final.csv fairtest/tgsim.py fairtest/tgsim_follow.py \
  fairtest/tgsim_follow_synthetic_check.py fairtest/dec_sim.py fairtest/cutout_sim.py fairtest/b35_analysis.py > $O/freeze_server_sha256.txt
$P -m fairtest.tgsim_follow_synthetic_check > $O/synthetic.log 2>&1
$P -m fairtest.tgsim_follow $C/tgsim $O/dec_cases.json $O/cutout_cases.json > $O/extract.log 2>&1
sha256sum $O/dec_cases.json $O/cutout_cases.json >> $O/freeze_server_sha256.txt
echo done > $C/b35_extract_done.flag
