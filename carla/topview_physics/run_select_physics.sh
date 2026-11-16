#!/bin/sh
# Step 1 (physics only): choose the Fig. 4 attack (seeds 2 and 3), check the search plan of Fig. S3(b), check the three shapes of Fig. 7 with physics.
W=${FAIRTEST_WORK:-$HOME/fairtest_work}; D=$W/topview_physics; PY=$HOME/fairtest/env/bin/python
PORT=2070; L=$D/logs/select_physics.log
export SDL_VIDEODRIVER=dummy OMP_NUM_THREADS=1 PYTHONPATH=$W/code FAIRTEST_CARLA_ACT=h2 FAIRTEST_DATA=$W/data PYTHONWARNINGS=ignore
export FAIRTEST_CARLA_BP=vehicle.ford.mustang FAIRTEST_CARLA_BRAKE=1.65,0.75,9.787 FAIRTEST_CARLA_PI=0,0
unset FAIRTEST_CC_PERCEIVE FAIRTEST_VEH_W FAIRTEST_VEH_L
echo "$(date '+%F %T') start" >> $L
if ! ss -ltn | grep -q ":$PORT "; then
  (cd $HOME/carla_0916 && env -u DISPLAY nohup nice -n 5 ./CarlaUE4.sh /Game/Carla/Maps/Town04 -RenderOffScreen -nosound \
    -carla-rpc-port=$PORT -quality-level=Epic >> $D/logs/carla_server_$PORT.log 2>&1 &)
  for i in $(seq 1 60); do ss -ltn | grep -q ":$PORT " && break; sleep 3; done; sleep 20
fi
echo "$(date '+%F %T') server up" >> $L
cd $W/code
K7=ladder:following:0,ladder:following:0.5,ladder:following:1,ladder:following_aeb:0,ladder:rss_sut:0,ba:normal,cc
for s in s2 s3; do
  nice -n 5 $PY -m fairtest.carla_replay $D/phys/select_$s.jsonl $D/traj_207_policy_$s.jsonl --kinds $K7 --port $PORT --resume >> $L 2>&1
  echo "$(date '+%F %T') select $s rc=$?" >> $L
done
nice -n 5 $PY -m fairtest.carla_replay $D/phys/search_check.jsonl $D/traj_207_carlasearch_final.jsonl --kinds ba:normal,cc,ladder:rss_sut:0 --port $PORT --resume >> $L 2>&1
echo "$(date '+%F %T') search check rc=$?" >> $L
nice -n 5 $PY -m fairtest.carla_replay $D/phys/physcheck_ab.jsonl $D/traj_physics_ab.jsonl --kinds ladder:following:0,cc --port $PORT --resume >> $L 2>&1
echo "$(date '+%F %T') physics ab rc=$?" >> $L
nice -n 5 $PY -m fairtest.carla_replay $D/phys/physcheck_c.jsonl $D/traj_physics_c.jsonl --kinds cc --port $PORT --resume >> $L 2>&1
echo "$(date '+%F %T') physics c rc=$? done" >> $L
