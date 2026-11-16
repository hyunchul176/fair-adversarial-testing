"""B54: Autoware parameter overrides for the two SUT variants, written from the image's original files.

  AW-default   the released configuration with the speed limits raised for highway cases (the sample configuration is
               tuned for 15 km/h): planning max_vel 4.17 -> 40 m/s, command-gate nominal vel_lim 25 -> 40 m/s, and the
               planning horizon 200-300 -> 800 m (velocity smoother, behavior path planner).
               Everything else, including all deceleration and jerk limits and the AEB, is as released.
  AW-r157      AW-default plus braking authority matched to the R157 careful driver (7.59 m/s^2, 12.65 m/s^3):
               planning limit.min_acc / min_jerk, cruise min accel, command-gate emergency acceleration and longitudinal
               acceleration / jerk limits, MRM emergency stop, vehicle model acceleration limit (7.0 -> 8.0). The logic
               parameters (comfort limits, cut-in detection, RSS assumptions, AEB) stay as released, and so do the
               longitudinal PID controller's own limits (min_acc -5.0, min_jerk -5.0).
  B55 (cause check of the B54 failure mode, variants written only when named on the command line):
  AW-r157n     AW-r157 with the planning comfort ('normal') deceleration limits set to the capability limits
               (normal.min_acc -1.0 -> -7.59, normal.min_jerk -1.0 -> -12.65)
  AW-r157nc    AW-r157n with the longitudinal PID controller's limits matched as well (min_acc / emergency_acc -5.0 ->
               -7.59, min_jerk -5.0 -> -12.65, emergency_jerk -3.0 -> -12.65)
  B58: aw_r157nc_l0 / _l300 / _l600 = AW-r157nc's files; the perception latency is applied by the client (aw_run)
All variants also enable engaging while driving (operation mode transition manager), a test-harness setting: the ego
is brought to the case speed before t = 0. Every changed key is printed and written to OUT_DIR/<variant>/CHANGES.json.

  python3 aw_config.py ORIG_DIR OUT_DIR [VARIANT ...]     (default: aw_default aw_r157)
"""
import json
import os
import re
import sys

A_CC, J_CC = 7.59, 12.65
FILES = {  # file -> path inside the image
    'common.param.yaml': 'autoware_launch/config/planning/scenario_planning/common/common.param.yaml',
    'control_command_gate.param.yaml': 'autoware_launch/config/control/control_command_gate/control_command_gate.param.yaml',
    'vehicle_cmd_gate.param.yaml': 'autoware_launch/config/control/vehicle_cmd_gate/vehicle_cmd_gate.param.yaml',
    'mrm_emergency_stop_operator.param.yaml': 'autoware_launch/config/system/mrm_emergency_stop_operator/mrm_emergency_stop_operator.param.yaml',
    'obstacle_cruise.param.yaml': 'autoware_launch/config/planning/scenario_planning/lane_driving/motion_planning/motion_velocity_planner/obstacle_cruise.param.yaml',
    'simulator_model.param.yaml': 'sample_vehicle_description/config/simulator_model.param.yaml',
    'operation_mode_transition_manager.param.yaml': 'autoware_launch/config/control/operation_mode_transition_manager/operation_mode_transition_manager.param.yaml',
    'velocity_smoother.param.yaml': 'autoware_launch/config/planning/scenario_planning/common/autoware_velocity_smoother/velocity_smoother.param.yaml',
    'behavior_path_planner.param.yaml': 'autoware_launch/config/planning/scenario_planning/lane_driving/behavior_planning/behavior_path_planner/behavior_path_planner.param.yaml',
}
FILES_B55 = {'pid.param.yaml': 'autoware_launch/config/control/trajectory_follower/longitudinal/pid.param.yaml'}
# test harness only (both variants): engage autonomous mode while the ego already drives at the case speed (pre-roll)
HARNESS = [('operation_mode_transition_manager.param.yaml', r'(enable_engage_on_driving:\s*)false', r'\g<1>true')]
SPEED = [('common.param.yaml', r'(max_vel:\s*)4\.17', r'\g<1>40.0'),
         ('control_command_gate.param.yaml', r'(nominal_filter:\n\s*vel_lim:\s*)25\.0', r'\g<1>40.0'),
         ('vehicle_cmd_gate.param.yaml', r'(nominal:\n\s*vel_lim:\s*)25\.0', r'\g<1>40.0'),
         # planning horizon: the smoother ends its trajectory with a comfortable stop (1.0 m/s^2), so a 200 m horizon caps
         # the cruise speed at ~18 m/s; 800 m allows 130 km/h
         ('velocity_smoother.param.yaml', r'(extract_ahead_dist:\s*)200\.0', r'\g<1>800.0'),
         ('velocity_smoother.param.yaml', r'(\n\s*max_trajectory_length:\s*)200\.0', r'\g<1>800.0'),
         ('velocity_smoother.param.yaml', r'(\n\s*min_trajectory_length:\s*)180\.0', r'\g<1>700.0'),
         ('velocity_smoother.param.yaml', r'(post_max_trajectory_length:\s*)300\.0', r'\g<1>800.0'),
         ('behavior_path_planner.param.yaml', r'(\n\s*forward_path_length:\s*)300\.0', r'\g<1>800.0')]
BRAKE = [('common.param.yaml', r'(limit:\n\s*min_acc:\s*)-2\.5', rf'\g<1>-{A_CC}'),
         ('common.param.yaml', r'(limit:\n(?:.*\n){2}\s*min_jerk:\s*)-1\.5', rf'\g<1>-{J_CC}'),
         ('obstacle_cruise.param.yaml', r'(min_accel_during_cruise:\s*)-2\.0', rf'\g<1>-{A_CC}'),
         ('control_command_gate.param.yaml', r'(builtin_emergency_acceleration:\s*)-2\.4', rf'\g<1>-{A_CC}'),
         ('control_command_gate.param.yaml', r'(\n\s*emergency_acceleration:\s*)-2\.4', rf'\g<1>-{A_CC}'),
         ('control_command_gate.param.yaml', r'(nominal_filter:(?:\n.*){4}\n\s*lon_acc_lim_for_lon_vel:\s*)\[5\.0, 5\.0, 5\.0, 4\.0\]', r'\g<1>[8.0, 8.0, 8.0, 8.0]'),
         ('control_command_gate.param.yaml', r'(nominal_filter:(?:\n.*){5}\n\s*lon_jerk_lim_for_lon_acc:\s*)\[80\.0, 5\.0, 5\.0, 4\.0\]', r'\g<1>[80.0, 13.0, 13.0, 13.0]'),
         ('vehicle_cmd_gate.param.yaml', r'(emergency_acceleration:\s*)-2\.4', rf'\g<1>-{A_CC}'),
         ('vehicle_cmd_gate.param.yaml', r'(nominal:(?:\n.*){4}\n\s*lon_acc_lim_for_lon_vel:\s*)\[5\.0, 5\.0, 5\.0, 4\.0\]', r'\g<1>[8.0, 8.0, 8.0, 8.0]'),
         ('vehicle_cmd_gate.param.yaml', r'(nominal:(?:\n.*){5}\n\s*lon_jerk_lim_for_lon_acc:\s*)\[80\.0, 5\.0, 5\.0, 4\.0\]', r'\g<1>[80.0, 13.0, 13.0, 13.0]'),
         ('mrm_emergency_stop_operator.param.yaml', r'(target_acceleration:\s*)-2\.5', rf'\g<1>-{A_CC}'),
         ('mrm_emergency_stop_operator.param.yaml', r'(target_jerk:\s*)-1\.5', rf'\g<1>-{J_CC}'),
         ('simulator_model.param.yaml', r'(vel_rate_lim:\s*)7\.0', r'\g<1>8.0')]
NORMAL = [('common.param.yaml', r'(normal:\n\s*min_acc:\s*)-1\.0', rf'\g<1>-{A_CC}'),
          ('common.param.yaml', r'(normal:\n(?:.*\n){2}\s*min_jerk:\s*)-1\.0', rf'\g<1>-{J_CC}')]
CONTROL = [('pid.param.yaml', r'(\n\s*min_acc:\s*)-5\.0', rf'\g<1>-{A_CC}'),
           ('pid.param.yaml', r'(\n\s*min_jerk:\s*)-5\.0', rf'\g<1>-{J_CC}'),
           ('pid.param.yaml', r'(emergency_acc:\s*)-5\.0', rf'\g<1>-{A_CC}'),
           ('pid.param.yaml', r'(emergency_jerk:\s*)-3\.0', rf'\g<1>-{J_CC}')]
VARIANTS = {'aw_default': HARNESS + SPEED, 'aw_r157': HARNESS + SPEED + BRAKE,
            'aw_r157n': HARNESS + SPEED + BRAKE + NORMAL, 'aw_r157nc': HARNESS + SPEED + BRAKE + NORMAL + CONTROL}
# B58: AW-r157nc with a perceived-adversary latency of 0 / 0.3 / 0.6 s (the latency is the client's, aw_run.LATENCY)
VARIANTS.update({f'aw_r157nc_l{ms}': VARIANTS['aw_r157nc'] for ms in (0, 300, 600)})
# B60: AW-r157nc with perception noise (_nz) or missed-detection bursts (_dp), applied by the client (aw_run.PERCEPTION)
VARIANTS.update({v: VARIANTS['aw_r157nc'] for v in ('aw_r157nc_nz', 'aw_r157nc_dp')})


def write(orig, out, variant, rules):
    d = os.path.join(out, variant)
    os.makedirs(d, exist_ok=True)
    files = dict(FILES, **(FILES_B55 if variant.startswith('aw_r157n') else {}))
    texts = {f: open(os.path.join(orig, f)).read() for f in files}
    changes = []
    for f, pat, rep in rules:
        m = re.search(pat, texts[f])
        if m is None:
            raise SystemExit(f'{variant}: pattern not found in {f}: {pat}')
        if (f, pat) in {(g, q) for g, q, _ in NORMAL + CONTROL} and len(re.findall(pat, texts[f])) != 1:
            raise SystemExit(f'{variant}: pattern not unique in {f}: {pat}')
        before, after = m.group(0).split('\n')[-1].strip(), m.expand(rep).split('\n')[-1].strip()
        texts[f] = texts[f][:m.start()] + m.expand(rep) + texts[f][m.end():]
        changes.append(dict(file=f, before=before, after=after))
    for f, t in texts.items():
        open(os.path.join(d, f), 'w').write(t)
    json.dump(dict(variant=variant, files=files, changes=changes), open(os.path.join(d, 'CHANGES.json'), 'w'), indent=1)
    for c in changes:
        print(variant, c['file'], ':', c['before'], '->', c['after'])


if __name__ == '__main__':
    orig, out = sys.argv[1], sys.argv[2]
    for v in (sys.argv[3:] or ['aw_default', 'aw_r157']):
        write(orig, out, v, VARIANTS[v])
