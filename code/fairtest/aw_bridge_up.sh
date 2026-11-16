#!/bin/sh
# B54: (re)start the bridge inside container fairtest_aw_NAME on PORT (kills a previous bridge of that port first).
# usage: aw_bridge_up.sh NAME ROS_DOMAIN_ID PORT
N=$1; D=$2; P=$3
docker exec fairtest_aw_$N bash -c "pgrep -f 'aw_bridge.py --port $P' | xargs -r kill -9" 2>/dev/null
docker exec -d fairtest_aw_$N bash -c "source /opt/autoware/setup.bash; export ROS_DOMAIN_ID=$D RMW_IMPLEMENTATION=rmw_cyclonedds_cpp CYCLONEDDS_URI=file:///work/aw_cyclonedds.xml; exec python3 /work/aw_bridge.py --port $P > /work/logs/${N}_bridge.log 2>&1"
