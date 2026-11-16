#!/bin/sh
# v2 server C (port 2050): E4-S1 (screened with matched twin M). Design v2 §3–4.
PORT=2050; . ${FAIRTEST_WORK:-$HOME/fairtest_work}/common.sh; P=$W/logs/C_progress.log
mustang; geom_m; export FAIRTEST_DATA=$W/data
ss -ltn | grep -q ":$PORT " || start_server
search $W/runs/E4_search_S1.jsonl S1 "" 8 $W/logs/E4_S1.log
echo "$(date '+%F %T') E4-S1 done" >> $P
stop_server
