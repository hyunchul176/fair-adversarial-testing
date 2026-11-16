#!/bin/sh
# E18-M s1 (addendum v2-6): Mustang, matched twin M, width-randomised policy s1 (port 2040).
PORT=2040; . ${FAIRTEST_WORK:-$HOME/fairtest_work}/common.sh; P=$W/logs/E18b_progress.log
export FAIRTEST_DATA=$W/data FAIRTEST_SHIELD_MARGIN=0.5 FAIRTEST_SHIELD_CC=cc_r157
mustang; geom_m
closed E18M_wr_rl_s1 $W/models_extra/fair_guarded2_wr_s1_1500000.zip "--commit --shield-window" $P
stop_server
