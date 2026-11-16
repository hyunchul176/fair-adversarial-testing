#!/bin/sh
# B43 (design fixed before the run): the B34 cut-out plans (619) replayed in CARLA against 10 co followers +
# BehaviorAgent + Traffic Manager, E7b main configuration, 3 servers with crash recovery (runtime-b42 code).
# usage: run_b43_carla_cutout.sh [smoke]
C=$HOME/fairtest; R=$C/runtime-b42/code; O=$C/b43; I=$O/inputs
mkdir -p $O $C/carla
export FAIRTEST_CARLA_BP=vehicle.ford.mustang FAIRTEST_CARLA_BRAKE=1.65,0.75,9.787 FAIRTEST_CARLA_PI=0,0

start_server() {
  $C/kill_pattern.sh "carla-rpc-port=$1" > /dev/null 2>&1
  sleep 5
  cd $HOME/carla_0916 && (nohup ./CarlaUE4.sh /Game/Carla/Maps/Town04 -RenderOffScreen -nosound -carla-rpc-port=$1 -quality-level=Low > $C/carla/server_$1.log 2>&1 &)
  for i in $(seq 1 60); do ss -ltn | grep -q ":$1 " && break; sleep 3; done
  sleep 15
}

job() {  # PORT LO HI OUT: one slice of the sorted job list to completion
  PORT=$1; LO=$2; HI=$3; OUT=$4
  for attempt in $(seq 1 20); do
    DONE=$( [ -f $OUT ] && wc -l < $OUT || echo 0 )
    [ "$DONE" -ge $((HI - LO)) ] && return 0
    ss -ltn | grep -q ":$PORT " || start_server $PORT
    cd $R && FAIRTEST_DATA=$R/fairtest/data OMP_NUM_THREADS=1 $C/env/bin/python -m fairtest.carla_cutout $OUT \
      $I/cutout_cases.json $I/cutout_eval_dev.jsonl $I/cutout_eval_dev_consensus.jsonl --slice $LO:$HI --port $PORT \
      --resume >> $OUT.log 2>&1 || start_server $PORT
  done
  return 1
}

if [ "$1" = smoke ]; then  # 2 plans x 12 kinds on one server, separate file (not evidence)
  job 2000 0 2 $O/smoke.jsonl
  $C/kill_pattern.sh "carla-rpc-port=2000" > /dev/null 2>&1
  echo done > $C/b43_smoke_done.flag
  exit 0
fi
job 2000 0 207 $O/carla_cutout_000.jsonl &
job 2010 207 413 $O/carla_cutout_207.jsonl &
job 2020 413 619 $O/carla_cutout_413.jsonl &
wait
for P in 2000 2010 2020; do $C/kill_pattern.sh "carla-rpc-port=$P" > /dev/null 2>&1; done
cd $R && $C/env/bin/python -m fairtest.b43_analysis $O/b43_analysis.json $O/carla_cutout_000.jsonl \
  $O/carla_cutout_207.jsonl $O/carla_cutout_413.jsonl > $O/b43_analysis.txt 2>&1
echo done > $C/b43_done.flag
