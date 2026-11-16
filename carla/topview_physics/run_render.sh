#!/bin/sh
# Step 2: render (Figs. 1, 4, S3). By the design's rule the Fig. 4 attack is seed 2 (phys/select_s2.jsonl and select_s3.jsonl: in both candidates
# only following H0 collides, a tie, so seed 2). Same view settings as the earlier run_bev.sh, Epic quality, corrected harness (h2).
W=${FAIRTEST_WORK:-$HOME/fairtest_work}; D=$W/topview_physics; PY=$HOME/fairtest/env/bin/python
PORT=2070; L=$D/logs/render.log; B=$D/bev
export SDL_VIDEODRIVER=dummy OMP_NUM_THREADS=1 PYTHONPATH=$W/code FAIRTEST_CARLA_ACT=h2 FAIRTEST_DATA=$W/data PYTHONWARNINGS=ignore
export FAIRTEST_CARLA_BP=vehicle.ford.mustang FAIRTEST_CARLA_BRAKE=1.65,0.75,9.787 FAIRTEST_CARLA_PI=0,0
unset FAIRTEST_CC_PERCEIVE FAIRTEST_VEH_W FAIRTEST_VEH_L
CAM=exposure_compensation=-2
echo "$(date '+%F %T') start" >> $L
ss -ltn | grep -q ":$PORT " || { echo "server down" >> $L; exit 1; }
cd $W/code
nice -n 5 $PY -m fairtest.carla_bev $B $D/traj_207_render.jsonl --port $PORT --width 1800 --height 340 --span 80 --lead 30 --cam $CAM \
  --scenes collide=ladder:following:0,collide=cc,fair=ladder:following:0,fair=cc,competent=ladder:rss_sut:0,competent=cc >> $L 2>&1
echo "$(date '+%F %T') teaser+competent rc=$?" >> $L
nice -n 5 $PY -m fairtest.carla_bev $B $D/traj_207_carlasearch_final.jsonl --port $PORT --width 1800 --height 340 --span 80 --lead 30 --cam $CAM \
  --scenes carlasearch=ba:normal,carlasearch=cc,carlasearch=ladder:rss_sut:0 >> $L 2>&1
echo "$(date '+%F %T') carlasearch rc=$?" >> $L
nice -n 5 $PY -m fairtest.carla_bev $B $D/traj_207_render.jsonl --port $PORT --width 1800 --height 118 --span 100 --lead 38 --cam $CAM \
  --tag-suffix _strip --reuse-physics \
  --scenes fair=ladder:following:0,fair=ladder:following:0.5,fair=ladder:following:1,fair=ladder:following_aeb:0,fair=ladder:rss_sut:0,fair=ba:normal,fair=cc >> $L 2>&1
echo "$(date '+%F %T') strips rc=$? done" >> $L
