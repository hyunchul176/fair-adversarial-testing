# v2 common settings (harness H2). Each run_*.sh sets PORT and then sources this file.
W=${FAIRTEST_WORK:-$HOME/fairtest_work}; PY=$HOME/fairtest/env/bin/python
export SDL_VIDEODRIVER=dummy OMP_NUM_THREADS=1 PYTHONPATH=$W/code FAIRTEST_CARLA_ACT=h2
mustang() { export FAIRTEST_CARLA_BP=vehicle.ford.mustang FAIRTEST_CARLA_BRAKE=1.65,0.75,9.787 FAIRTEST_CARLA_PI=0,0; }
tesla() { export FAIRTEST_CARLA_BP=vehicle.tesla.model3 FAIRTEST_CARLA_BRAKE=2.5,0.56,9.97 FAIRTEST_CARLA_PI=0,0; }
geom_u() { unset FAIRTEST_CC_PERCEIVE FAIRTEST_VEH_W FAIRTEST_VEH_L; }                        # default twin U
geom_m() { export FAIRTEST_CC_PERCEIVE=1.251 FAIRTEST_VEH_W=2.166 FAIRTEST_VEH_L=4.718; }      # matched twin M (Mustang)
geom_t() { export FAIRTEST_CC_PERCEIVE=1.251 FAIRTEST_VEH_W=2.469 FAIRTEST_VEH_L=4.79; }       # matched twin T (Tesla)
KINDS8=ladder:following:0,ladder:following_aeb:0,ladder:rss_sut:0,ladder:rss_lat_sut:0,ba:cautious,ba:normal,ba:aggressive,tm:default
kind_of() { case $1 in following_H0) echo ladder:following:0;; following_H0.5) echo ladder:following:0.5;; following_H1) echo ladder:following:1;;
  reference_idm_H0) echo ladder:reference_idm:0;; following_aeb_H0) echo ladder:following_aeb:0;; following_aeb_H0.5) echo ladder:following_aeb:0.5;;
  rss_sut_H0) echo ladder:rss_sut:0;; rss_lat_sut_H0) echo ladder:rss_lat_sut:0;; esac; }
start_server() {
  for pid in $(pgrep -f "rpc-port=$PORT"); do case "$(cat /proc/$pid/comm 2>/dev/null)" in CarlaUE4*) kill -9 $pid;; esac; done; sleep 5
  (cd $HOME/carla_0916 && env -u DISPLAY nohup nice -n 5 ./CarlaUE4.sh /Game/Carla/Maps/Town04 -RenderOffScreen -nosound \
    -carla-rpc-port=$PORT -quality-level=Low >> $W/logs/carla_server_$PORT.log 2>&1 &)
  for i in $(seq 1 60); do ss -ltn | grep -q ":$PORT " && break; sleep 3; done; sleep 15
}
stop_server() { for pid in $(pgrep -f "rpc-port=$PORT"); do case "$(cat /proc/$pid/comm 2>/dev/null)" in CarlaUE4*) kill -9 $pid;; esac; done; }
replay() {  # IN OUT KINDS PROGRESS_LOG
  N=$(wc -l < $1); [ "$N" -eq 0 ] && return
  for attempt in $(seq 1 10); do
    DONE=$( [ -f $2 ] && wc -l < $2 || echo 0 ); [ "$DONE" -ge "$N" ] && break
    ss -ltn | grep -q ":$PORT " || start_server
    (cd $W/code && nice -n 5 $PY -m fairtest.carla_replay $2 $1 --kinds $3 --port $PORT --resume >> $W/logs/replay_$PORT.log 2>&1) || start_server
  done
  echo "$(date '+%F %T') $(basename $2) $(wc -l < $2)/$N" >> $4
}
closed() {  # NAME MODEL FLAGS PROGRESS_LOG
  OUT=$W/runs/$1.jsonl
  for attempt in $(seq 1 10); do
    DONE=$( [ -f $OUT ] && wc -l < $OUT || echo 0 ); [ "$DONE" -ge 62 ] && break
    ss -ltn | grep -q ":$PORT " || start_server
    (cd $W/code && nice -n 5 $PY -m fairtest.carla_rl $OUT $2 $3 --kinds $KINDS8 --port $PORT --resume >> $W/logs/$1.log 2>&1) || start_server
  done
  echo "$(date '+%F %T') $1 $(wc -l < $OUT)/62" >> $4
}
search() {  # OUT ARMS KINDS WORKERS LOG  (e4_search, settings in environment variables)
  for attempt in 1 2 3; do
    nice -n 5 $PY -I -c "import sys, runpy; sys.path[:0] = ['$W', '$W/code']; sys.argv = ['e4', '$1', '--cont', '$W/jobs/b23_exposed_cont_search_all7.jsonl', '--port', '$PORT', '--arms', '$2', '--workers', '$4'] + (['--kinds', '$3'] if '$3' else []); runpy.run_path('$W/e4_search.py', run_name='__main__')" >> $5 2>&1
    grep -q '"finished"' $5 && break
  done
}
