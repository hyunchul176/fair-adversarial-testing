#!/bin/sh
# E18 final (addendum v2-6): width-randomised policies s1–s3. Argument: T (Tesla, matched twin T, port 2030) or M (Mustang, matched twin M, port 2040).
COND=$1
if [ "$COND" = T ]; then PORT=2030; else PORT=2040; fi
. ${FAIRTEST_WORK:-$HOME/fairtest_work}/common.sh; P=$W/logs/E18b_progress.log
export FAIRTEST_DATA=$W/data FAIRTEST_SHIELD_MARGIN=0.5 FAIRTEST_SHIELD_CC=cc_r157
if [ "$COND" = T ]; then tesla; geom_t; SEEDS="2 3"; PRE=E18_wr_rl; else mustang; geom_m; SEEDS="1 2 3"; PRE=E18M_wr_rl; fi
for n in $SEEDS; do
  M=$W/models_extra/fair_guarded2_wr_s${n}_1500000.zip
  closed ${PRE}_s$n $M "--commit --shield-window" $P
done
echo "$(date '+%F %T') E18b $COND done" >> $P
stop_server
