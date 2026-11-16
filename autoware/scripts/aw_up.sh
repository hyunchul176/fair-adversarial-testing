#!/bin/sh
# B54: start one headless Autoware planning-simulator instance (container fairtest_aw_NAME) with a parameter variant.
# usage: aw_up.sh NAME ROS_DOMAIN_ID VARIANT [extra ros2 launch args...]
#   VARIANT = aw_default | aw_r157 | aw_r157n | aw_r157nc (overrides/<VARIANT>/, written by aw_config.py)
# Perception is ground truth: the dummy perception publisher without simulated detection failures, then Autoware's own
# multi-object tracker and map-based prediction. ROS traffic stays on the loopback (CycloneDDS, aw_cyclonedds.xml).
N=$1; D=$2; V=$3; shift 3
W=$HOME/fairtest/aw
O=$W/overrides/$V
S=/opt/autoware/share
IMG=ghcr.io/autowarefoundation/autoware:universe-20260420
[ -d "$O" ] || { echo "no overrides for $V"; exit 2; }
mkdir -p $W/home/$N $W/logs
# B55 variants also carry the longitudinal PID controller parameters (the B54 variants keep the image's file)
PIDMOUNT=""
[ -f "$O/pid.param.yaml" ] && PIDMOUNT="-v $O/pid.param.yaml:$S/autoware_launch/config/control/trajectory_follower/longitudinal/pid.param.yaml:ro"
docker run -d --rm --name fairtest_aw_$N --net host --shm-size=4g --user $(id -u):$(id -g) \
  -e HOME=/work/home/$N -e ROS_DOMAIN_ID=$D \
  -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp -e CYCLONEDDS_URI=file:///work/aw_cyclonedds.xml \
  -v $W:/work \
  -v $O/common.param.yaml:$S/autoware_launch/config/planning/scenario_planning/common/common.param.yaml:ro \
  -v $O/control_command_gate.param.yaml:$S/autoware_launch/config/control/control_command_gate/control_command_gate.param.yaml:ro \
  -v $O/vehicle_cmd_gate.param.yaml:$S/autoware_launch/config/control/vehicle_cmd_gate/vehicle_cmd_gate.param.yaml:ro \
  -v $O/mrm_emergency_stop_operator.param.yaml:$S/autoware_launch/config/system/mrm_emergency_stop_operator/mrm_emergency_stop_operator.param.yaml:ro \
  -v $O/obstacle_cruise.param.yaml:$S/autoware_launch/config/planning/scenario_planning/lane_driving/motion_planning/motion_velocity_planner/obstacle_cruise.param.yaml:ro \
  -v $O/simulator_model.param.yaml:$S/sample_vehicle_description/config/simulator_model.param.yaml:ro \
  -v $O/velocity_smoother.param.yaml:$S/autoware_launch/config/planning/scenario_planning/common/autoware_velocity_smoother/velocity_smoother.param.yaml:ro   -v $O/behavior_path_planner.param.yaml:$S/autoware_launch/config/planning/scenario_planning/lane_driving/behavior_planning/behavior_path_planner/behavior_path_planner.param.yaml:ro   -v $O/operation_mode_transition_manager.param.yaml:$S/autoware_launch/config/control/operation_mode_transition_manager/operation_mode_transition_manager.param.yaml:ro \
  $PIDMOUNT \
  $IMG bash -c "source /opt/autoware/setup.bash; exec ros2 launch autoware_launch planning_simulator.launch.xml map_path:=/work/map vehicle_model:=sample_vehicle sensor_model:=sample_sensor_kit rviz:=false perception/enable_detection_failure:=false $* > /work/logs/$N.log 2>&1"
