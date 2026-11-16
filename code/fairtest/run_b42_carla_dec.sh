#!/bin/sh
# B42 (design fixed before the run): the B30 deceleration plans (363) replayed in CARLA against 8 dec followers +
# BehaviorAgent + Traffic Manager, E7b main configuration (Mustang, feed-forward brake), 3 servers with crash recovery
# (the run_carla_job.sh logic, runtime-b42). usage: run_b42_carla_dec.sh [smoke]
C=$HOME/fairtest; R=$C/runtime-b42/code; O=$C/b42; I=$O/inputs
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
    cd $R && FAIRTEST_DATA=$R/fairtest/data OMP_NUM_THREADS=1 $C/env/bin/python -m fairtest.carla_dec $OUT \
      $I/dec_cases.json $I/dec_eval_dev.jsonl --slice $LO:$HI --port $PORT --resume >> $OUT.log 2>&1 || start_server $PORT
  done
  return 1
}

if [ "$1" = smoke ]; then  # 2 plans x 10 kinds on one server, separate file (not evidence)
  job 2000 0 2 $O/smoke.jsonl
  $C/kill_pattern.sh "carla-rpc-port=2000" > /dev/null 2>&1
  echo done > $C/b42_smoke_done.flag
  exit 0
fi
job 2000 0 121 $O/carla_dec_000.jsonl &
job 2010 121 242 $O/carla_dec_121.jsonl &
job 2020 242 363 $O/carla_dec_242.jsonl &
wait
for P in 2000 2010 2020; do $C/kill_pattern.sh "carla-rpc-port=$P" > /dev/null 2>&1; done
cd $R && $C/env/bin/python -m fairtest.b42_analysis $O/b42_analysis.json $O/carla_dec_000.jsonl $O/carla_dec_121.jsonl \
  $O/carla_dec_242.jsonl > $O/b42_analysis.txt 2>&1
echo done > $C/b42_done.flag
