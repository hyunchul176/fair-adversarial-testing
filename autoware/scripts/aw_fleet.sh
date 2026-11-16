#!/bin/bash
# B54: (re)start a fleet of Autoware instances in parallel; prints one line per instance with the restart exit code.
# usage: bash aw_fleet.sh NAME:DOMAIN:PORT:VARIANT ...
W=$HOME/fairtest/aw
for spec in "$@"; do
  IFS=: read -r N D P V <<< "$spec"
  ( bash $W/aw_restart.sh $N $D $P $V; echo "$N rc=$?" ) &
  sleep 4  # stagger the launches
done
wait
docker ps --format '{{.Names}} {{.Status}}' | grep fairtest_aw_ | sort
