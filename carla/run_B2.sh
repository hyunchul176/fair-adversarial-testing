#!/bin/sh
# v2 server B (port 2040): closed loop E2 → E3 → E5-A → E5-B → E8 → E7. Design v2 §3–4 (E17 in run_A2.sh).
PORT=2040; . ${FAIRTEST_WORK:-$HOME/fairtest_work}/common.sh; P=$W/logs/B_progress.log
mustang; export FAIRTEST_DATA=$W/data FAIRTEST_SHIELD_MARGIN=0.5 FAIRTEST_SHIELD_CC=cc_r157
M=$W/models
geom_m; for S in s1 s2 s3; do closed E2_rl_$S $M/agent_policy/$S.zip "--commit --shield-window" $P; done
closed E3_rl_s1_rep $M/agent_policy/s1.zip "--commit --shield-window" $P
for S in s1 s2 s3; do closed E5A_noshield_$S $M/agent_policy/$S.zip "--commit" $P; done
geom_u; for S in s1 s2 s3; do closed E5B_hwshield_$S $M/agent_policy/$S.zip "--commit --shield-window" $P; done
geom_m
for job in collide_commit:baseline_collision_seeking_commit:--commit cars:baseline_cars_style: dynasto:baseline_dynasto_style:; do
  B=${job%%:*}; REST=${job#*:}; MD=${REST%%:*}; FLAG=${REST#*:}
  for S in s1 s2 s3; do closed E8_${B}_$S $M/$MD/$S.zip "$FLAG" $P; done
done
for spec in dec:carla_dec:363 cutout:carla_cutout:619; do
  F=${spec%%:*}; REST=${spec#*:}; MOD=${REST%%:*}; N=${REST#*:}; OUT=$W/runs/E7_$F.jsonl
  if [ $F = dec ]; then ARGS="$W/jobs/b42_inputs/dec_cases.json $W/jobs/b42_inputs/dec_eval_dev.jsonl"
  else ARGS="$W/jobs/b43_inputs/cutout_cases.json $W/jobs/b43_inputs/cutout_eval_dev.jsonl $W/jobs/b43_inputs/cutout_eval_dev_consensus.jsonl"; fi
  for attempt in $(seq 1 20); do
    DONE=$( [ -f $OUT ] && wc -l < $OUT || echo 0 ); [ "$DONE" -ge $N ] && break
    ss -ltn | grep -q ":$PORT " || start_server
    (cd $W/code && nice -n 5 $PY -m fairtest.$MOD $OUT $ARGS --slice 0:$N --port $PORT --resume >> $W/logs/E7_$F.log 2>&1) || start_server
  done
  echo "$(date '+%F %T') E7 $F $(wc -l < $OUT)/$N" >> $P
done
# 2026-10-29 change of run location: E17 moved to server A (run_A2.sh).
echo "$(date '+%F %T') B done" >> $P
stop_server
