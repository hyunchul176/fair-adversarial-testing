#!/bin/sh
# B51 POST HOC on MiTra (B41 reuse, labelled post hoc): the closing speed of the v2j reports (policy contacts of seeds 1-3,
# bo_warm_g4j successes) and bo_warm_g4js for the 172 cases x 17 SUTs from the B41 CC-boundary plans. Reads B41 only.
C=$HOME/fairtest; R=$C/runtime-b51/code; O=$C/b51; B=$C/b41
mkdir -p $O; cd $R
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 SDL_VIDEODRIVER=dummy PYTHONWARNINGS=ignore FAIRTEST_DATA=$C/mitra_data
export FAIRTEST_DRL_SUT=$C/b25/drl_models/0.zip FAIRTEST_DRL_SUT_AT="$C/b25/drl_models/{k}.zip"
P=$C/env/bin/python
S=following:0,following:0.5,following:1,reference_idm:0,following_aeb:0,following_aeb:0.5,rss_sut:0,bc_human:0,ads@0.5:0,ads@2.0:0,rss_lat_sut:0,drl_sut:0,drl_sut@3:0,drl_sut@6:0,drl_sut@p1.5:0,drl_sut@p2:0,drl_sut@p2.5:0
[ -e $O/closing_mitra.jsonl ] || { $P -m fairtest.b51_severity closing $O/closing_mitra.part --policy $B/final_s1.jsonl $B/final_s2.jsonl \
  $B/final_s3.jsonl --refine $B/refine.jsonl --workers 12 > $O/closing_mitra.log 2>&1 && mv $O/closing_mitra.part $O/closing_mitra.jsonl; }
[ -e $O/refine_g4js_mitra.jsonl ] || { $P -m fairtest.b51_severity refine $O/refine_g4js_mitra.part --cont $B/b11.jsonl --partition all \
  --suts $S --workers 12 > $O/refine_mitra.log 2>&1 && mv $O/refine_g4js_mitra.part $O/refine_g4js_mitra.jsonl; }
echo done > $C/b51_mitra_done.flag
