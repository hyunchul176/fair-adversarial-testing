"""B54 diagnosis (not evidence): grab the motion-planning trajectory and the final trajectory during a drive; print the
lateral offsets, the discrete curvature and the planned speeds (inside the container)."""
import math
import time

import rclpy
from autoware_planning_msgs.msg import Trajectory

rclpy.init()
n = rclpy.create_node('fairtest_b54_traj')
got = {}
for name, topic in (('motion', '/planning/scenario_planning/lane_driving/trajectory'), ('final', '/planning/scenario_planning/trajectory'),
                    ('lat', '/planning/scenario_planning/velocity_smoother/debug/trajectory_lateral_acc_filtered'),
                    ('steer', '/planning/scenario_planning/velocity_smoother/debug/trajectory_steering_rate_limited')):
    n.create_subscription(Trajectory, topic, lambda m, k=name: got.__setitem__(k, m), 1)
time.sleep(10)  # the drive is running by then
t0 = time.time()
while time.time() - t0 < 6 and len(got) < 4:
    rclpy.spin_once(n, timeout_sec=.2)
for k, m in got.items():
    pts = [(p.pose.position.x, p.pose.position.y, p.longitudinal_velocity_mps) for p in m.points]
    ys = [p[1] for p in pts]
    kap = []
    for i in range(1, len(pts) - 1, max(1, len(pts) // 60)):
        (x0, y0, _), (x1, y1, _), (x2, y2, _) = pts[i - 1], pts[i], pts[i + 1]
        a, b, c = math.hypot(x1 - x0, y1 - y0), math.hypot(x2 - x1, y2 - y1), math.hypot(x2 - x0, y2 - y0)
        area = abs((x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)) / 2
        kap.append(4 * area / max(a * b * c, 1e-9))
    print(k, 'points', len(pts), 'x', round(pts[0][0], 1), '->', round(pts[-1][0], 1), 'y range', round(min(ys), 3), round(max(ys), 3),
          'max curvature', round(max(kap) if kap else 0, 5), 'v first/min/max', round(pts[0][2], 2), round(min(p[2] for p in pts), 2),
          round(max(p[2] for p in pts), 2))
