#!/bin/sh
# v2 server A (port 2030): replay E1 → E11 → E6 → E12 → E15 (Tesla) → E9 → E13 → E14 → E4-S0. Design v2 §3–4.
PORT=2030; . ${FAIRTEST_WORK:-$HOME/fairtest_work}/common.sh; P=$W/logs/A_progress.log
mustang; geom_u; export FAIRTEST_DATA=$W/data
for IN in $W/runs/E1_traj_*.jsonl; do SUT=$(basename $IN .jsonl | sed 's/^E1_traj_//'); replay $IN $W/runs/E1_replay_$SUT.jsonl cc,$(kind_of $SUT) $P; done
echo "$(date '+%F %T') E1 done" >> $P
for IN in $W/runs/E1_traj_*.jsonl; do SUT=$(basename $IN .jsonl | sed 's/^E1_traj_//'); replay $IN $W/runs/E11_replay_$SUT.jsonl cc,$(kind_of $SUT) $P; done
echo "$(date '+%F %T') E11 done" >> $P
for spec in tgsim:data_tgsim highsim:data_highsim mitra:data_mitra exid:data_exid dev_extra:data; do
  TAG=${spec%%:*}; export FAIRTEST_DATA=$W/${spec#*:}
  for IN in $W/runs/E6_${TAG}_traj_*.jsonl; do SUT=$(basename $IN .jsonl | sed "s/^E6_${TAG}_traj_//"); replay $IN $W/runs/E6_${TAG}_replay_$SUT.jsonl cc,$(kind_of $SUT) $P; done
done
echo "$(date '+%F %T') E6 done" >> $P
for IN in $W/runs/E12_*_traj_*.jsonl; do
  B=$(basename $IN .jsonl); TAG=$(echo $B | sed 's/^E12_\([a-z]*\)_traj_.*/\1/'); SUT=$(echo $B | sed 's/^E12_[a-z]*_traj_//')
  case $TAG in dev) export FAIRTEST_DATA=$W/data;; *) export FAIRTEST_DATA=$W/data_$TAG;; esac
  replay $IN $W/runs/E12_${TAG}_replay_$SUT.jsonl cc,$(kind_of $SUT) $P
done
echo "$(date '+%F %T') E12 done" >> $P
tesla; stop_server; export FAIRTEST_DATA=$W/data
for IN in $W/runs/E1_traj_*.jsonl; do SUT=$(basename $IN .jsonl | sed 's/^E1_traj_//'); replay $IN $W/runs/E15a_replay_$SUT.jsonl cc,$(kind_of $SUT) $P; done
echo "$(date '+%F %T') E15 done" >> $P
mustang; stop_server
cd $W && $PY -I e1_analyze.py $W $HOME/fairtest > $W/logs/E1_analyze.log 2>&1; $PY -I e6_analyze.py $W > $W/logs/E6_analyze.log 2>&1
$PY -I e9_select.py $W > $W/logs/E9_select.log 2>&1
for IN in $W/runs/E9_*_traj_*.jsonl; do
  B=$(basename $IN .jsonl); TAG=$(echo $B | sed 's/^E9_\([a-z]*\)_traj_.*/\1/'); SUT=$(echo $B | sed 's/^E9_[a-z]*_traj_//')
  case $TAG in dev) export FAIRTEST_DATA=$W/data;; *) export FAIRTEST_DATA=$W/data_$TAG;; esac
  replay $IN $W/runs/E9_${TAG}_replay_$SUT.jsonl $(kind_of $SUT):hw,cc:hw $P
done
echo "$(date '+%F %T') E9 done" >> $P
export FAIRTEST_DATA=$W/data; ss -ltn | grep -q ":$PORT " || start_server
for attempt in 1 2 3; do
  (cd $W && nice -n 5 $PY -I $W/e13_collide.py $W/runs/E13_collide.jsonl --port $PORT --workers 6 >> $W/logs/E13_collide.log 2>&1)
  [ "$(grep -c -v '"error"' $W/runs/E13_collide.jsonl)" -ge 496 ] && break
done
echo "$(date '+%F %T') E13 done" >> $P
geom_m; search $W/runs/E14_search.jsonl S1 ladder:following:0,ba:normal 6 $W/logs/E14_search.log
echo "$(date '+%F %T') E14 done" >> $P
geom_u; search $W/runs/E4_search_S0.jsonl S0 "" 8 $W/logs/E4_S0.log
echo "$(date '+%F %T') E4-S0 done" >> $P
stop_server
