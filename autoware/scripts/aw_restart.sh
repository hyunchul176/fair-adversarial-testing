#!/bin/bash
# B54: restart one Autoware instance and its bridge; returns when the bridge accepts connections (exit 1 on timeout).
# usage: bash aw_restart.sh NAME ROS_DOMAIN_ID PORT VARIANT
N=$1; D=$2; P=$3; V=$4
W=$HOME/fairtest/aw
docker stop -t 5 fairtest_aw_$N >/dev/null 2>&1
i=0; while docker ps -a --format '{{.Names}}' | grep -qx fairtest_aw_$N; do sleep 1; i=$((i+1)); [ $i -gt 60 ] && break; done
sh $W/aw_up.sh $N $D $V >/dev/null || exit 1
ENV="source /opt/autoware/setup.bash; export ROS_DOMAIN_ID=$D RMW_IMPLEMENTATION=rmw_cyclonedds_cpp CYCLONEDDS_URI=file:///work/aw_cyclonedds.xml"
i=0
while :; do
  n=$(docker exec fairtest_aw_$N bash -c "$ENV; timeout 10 ros2 topic list 2>/dev/null | wc -l" 2>/dev/null)
  [ "${n:-0}" -gt 300 ] && break
  sleep 3; i=$((i+1)); [ $i -gt 60 ] && exit 1
done
sleep 20
sh $W/aw_bridge_up.sh $N $D $P
i=0; until (exec 3<>/dev/tcp/127.0.0.1/$P) 2>/dev/null; do sleep 1; i=$((i+1)); [ $i -gt 60 ] && exit 1; done
exit 0
