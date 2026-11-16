#!/bin/sh
# E18 (addendum v2-4): after E17 ends on server A (port 2030), Tesla closed loop (matched twin T) with the width-randomised policy (wr s1).
PORT=2030; . ${FAIRTEST_WORK:-$HOME/fairtest_work}/common.sh; P=$W/logs/A_progress.log
until grep -q "E17 done" $P 2>/dev/null; do sleep 30; done
sleep 20
tesla; geom_t; export FAIRTEST_DATA=$W/data FAIRTEST_SHIELD_MARGIN=0.5 FAIRTEST_SHIELD_CC=cc_r157
closed E18_wr_rl_s1 $W/models_extra/fair_guarded2_wr_s1_1500000.zip "--commit --shield-window" $P
echo "$(date '+%F %T') E18 done" >> $P
stop_server
