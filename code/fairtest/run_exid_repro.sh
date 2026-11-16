#!/bin/sh
# exiD preparation on the server (not evidence; exiD protocol draft and code status, available on request): run the exiD analysis code on MiTra,
# whose stored outputs it must reproduce, before the exiD data arrive.
#  1. exid_consensus (final agent v2s: policy stage + bo_warm_g4s) on the competent SUTs -> compare with
#     b56/competent_check_mitra.json (b56_competent_check, same b29_consensus.task)
#  2. exid_extra with --method bo_warm_g4j and replays -> compare S7/S8 with b41_posthoc/v2j_extra_mitra.json (v2j_extra.py)
# Inputs are read only: b41/ (B41 outputs), b41/refine.jsonl (g4j), b56/b56_mitra.jsonl (g4s), mitra_data/.
# Code: runtime-exidprep = git archive of the development branch at the commit given as $1 (unpacked by the caller).
#   flock -n /tmp/fairtest_exidprep.lock sh run_exid_repro.sh <commit>
C=$HOME/fairtest; R=$C/runtime-exidprep/code; O=$C/exidprep
mkdir -p $O; cd $R || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 SDL_VIDEODRIVER=dummy PYTHONWARNINGS=ignore
export FAIRTEST_DRL_SUT=$C/b25/drl_models/0.zip FAIRTEST_DRL_SUT_AT="$C/b25/drl_models/{k}.zip"
P=$C/env/bin/python
echo "$(date '+%F %T') start commit=$1" >> $O/chain.log
FAIRTEST_DATA=$C/runtime-b40/code/fairtest/data $P -m fairtest.runner validate-b5 > $O/validate_b5.log 2>&1 || { echo "$(date '+%F %T') validate rc=$?" >> $O/chain.log; exit 1; }
FAIRTEST_DATA=$C/mitra_data nice -n 10 $P -m fairtest.exid_consensus $C/b41 $O/exid_consensus_mitra.jsonl \
  --refine $C/b56/b56_mitra.jsonl --workers 24 > $O/exid_consensus_mitra.log 2>&1
echo "$(date '+%F %T') consensus rc=$?" >> $O/chain.log
FAIRTEST_DATA=$C/mitra_data nice -n 10 $P -m fairtest.exid_extra $C/b41 $O/exid_extra_mitra_g4j.json \
  --refine $C/b41/refine.jsonl --method bo_warm_g4j --workers 24 > $O/exid_extra_mitra_g4j.log 2>&1
echo "$(date '+%F %T') extra rc=$?" >> $O/chain.log
echo "$(date '+%F %T') done" >> $O/chain.log
