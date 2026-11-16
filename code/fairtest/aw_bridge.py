"""B54 bridge: runs INSIDE the Autoware container (Python 3.10 + rclpy) and drives one Autoware planning-simulator instance
for a client in our Python environment (fairtest.aw_sut) over a local TCP socket (one JSON object per line). The client
never imports ROS.

Commands (request {"cmd": NAME, ...} -> reply {"ok": bool, "result" | "error"}):
  info   {}                                  vehicle geometry: length, width, centre offset ahead of base_link
  reset  {ego: [x, y, yaw, v], goal: [x, y, yaw], vmax}
         delete objects, stop mode, clear the route, initialise the ego pose (localization API), set the route and the
         velocity limit, engage autonomous mode, then set the ego speed (simple planning simulator initial twist)
         and wait until Autoware drives autonomously. Returns the timings and the ego state.
  step   {obj: [x, y, yaw, vx, vy, length, width] | null}
         publish the adversary as a ground-truth dummy-perception object (ADD first, MODIFY afterwards), wait to the
         next 0.1 s wall-clock boundary and return the ego state {t, x, y, yaw, v, a, acc_cmd, mode}
  clear  {}                                  delete all dummy objects
  state  {}                                  ego state now
Autoware runs on wall-clock time here; the client keeps the 10 Hz pace (fairtest.aw_sut) and records the lateness.

  python3 aw_bridge.py --port 5454
"""
from __future__ import annotations

import argparse
import json
import math
import socketserver
import threading
import time
import uuid

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time as RosTime

DT = .1


def yaw_q(yaw):
    return math.sin(yaw / 2), math.cos(yaw / 2)


class Bridge(Node):
    def __init__(self):
        import os
        super().__init__(f'fairtest_b54_bridge_{os.getpid()}')  # unique: a restarted bridge must not duplicate a node
        from autoware_adapi_v1_msgs.msg import OperationModeState, RouteState
        from autoware_adapi_v1_msgs.srv import ChangeOperationMode, ClearRoute, InitializeLocalization, SetRoutePoints
        from autoware_control_msgs.msg import Control
        from autoware_internal_planning_msgs.msg import VelocityLimit
        from autoware_planning_msgs.msg import Trajectory
        from geometry_msgs.msg import AccelWithCovarianceStamped, TwistStamped
        from nav_msgs.msg import Odometry
        from tier4_simulation_msgs.msg import DummyObject
        self.M = dict(VelocityLimit=VelocityLimit, TwistStamped=TwistStamped, DummyObject=DummyObject)
        self.S = dict(ChangeOperationMode=ChangeOperationMode, ClearRoute=ClearRoute, InitializeLocalization=InitializeLocalization,
                      SetRoutePoints=SetRoutePoints)
        self.lock = threading.Lock()
        self.odom = self.accel = self.cmd = self.mode = self.route = None
        self.traj_wall = None  # wall time of the last planned trajectory
        tl = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
        self.pub_twist = self.create_publisher(TwistStamped, '/simulation/input/initialtwist', 1)
        self.pub_vlim = self.create_publisher(VelocityLimit, '/planning/scenario_planning/max_velocity_default', tl)
        self.pub_obj = self.create_publisher(DummyObject, '/simulation/dummy_perception_publisher/object_info', 10)
        self.create_subscription(Odometry, '/localization/kinematic_state', lambda m: self._set('odom', m), 10)
        self.create_subscription(AccelWithCovarianceStamped, '/localization/acceleration', lambda m: self._set('accel', m), 10)
        self.create_subscription(Control, '/control/command/control_cmd', lambda m: self._set('cmd', m), 10)
        self.create_subscription(OperationModeState, '/api/operation_mode/state', lambda m: self._set('mode', m), tl)
        self.create_subscription(RouteState, '/api/routing/state', lambda m: self._set('route', m), tl)
        self.traj_v0 = None  # planned speed at the first trajectory point (the engage stability check compares it)
        self.create_subscription(Trajectory, '/planning/scenario_planning/trajectory', self._traj, 1)
        self.mrm = None  # the fail-safe state (AD API), recorded per tick: an MRM is Autoware's behaviour, a stall is not
        try:
            from autoware_adapi_v1_msgs.msg import MrmState
            self.create_subscription(MrmState, '/api/fail_safe/mrm_state', lambda m: self._set('mrm', m), 10)
        except ImportError:
            pass
        self.cli = {name: self.create_client(srv, path) for name, srv, path in (
            ('stop', ChangeOperationMode, '/api/operation_mode/change_to_stop'),
            ('auto', ChangeOperationMode, '/api/operation_mode/change_to_autonomous'),
            ('clear_route', ClearRoute, '/api/routing/clear_route'),
            ('set_route', SetRoutePoints, '/api/routing/set_route_points'),
            ('init_pose', InitializeLocalization, '/api/localization/initialize'))}
        self.obj_id, self.obj_added = None, False
        self.next_tick = None

    def _set(self, name, m):
        with self.lock:
            setattr(self, name, m)

    def _traj(self, m):
        with self.lock:
            self.traj_wall = time.time()
            self.traj_v0 = float(m.points[0].longitudinal_velocity_mps) if len(m.points) else None

    def call(self, name, req, timeout=10.):
        c = self.cli[name]
        if not c.wait_for_service(timeout_sec=timeout):
            raise RuntimeError(f'service {name} not available')
        fut = c.call_async(req)
        t0 = time.time()
        while not fut.done():
            if time.time() - t0 > timeout:
                raise RuntimeError(f'service {name} timed out')
            time.sleep(.01)
        res = fut.result()
        st = getattr(res, 'status', None)
        return dict(success=None if st is None else bool(st.success), code=None if st is None else int(st.code),
                    message=None if st is None else str(st.message))

    def wait(self, cond, timeout, what):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if cond():
                return round(time.time() - t0, 2)
            time.sleep(.02)
        raise RuntimeError(f'timeout waiting for {what}')

    # ---------------------------------------------------------------------------------------------- state
    def ego(self):
        with self.lock:
            m, a, c, mode, tw, mrm = self.odom, self.accel, self.cmd, self.mode, self.traj_wall, self.mrm
        if m is None:
            return None
        p, q, v = m.pose.pose.position, m.pose.pose.orientation, m.twist.twist.linear
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        now = time.time()
        return dict(t=m.header.stamp.sec + m.header.stamp.nanosec * 1e-9, wall=now, x=p.x, y=p.y, yaw=yaw, v=v.x,
                    a=None if a is None else a.accel.accel.linear.x,
                    acc_cmd=None if c is None else c.longitudinal.acceleration,
                    mode=None if mode is None else int(mode.mode),
                    traj_age=None if tw is None else round(now - tw, 3),
                    mrm=None if mrm is None else [int(mrm.state), int(mrm.behavior)])

    def cmd_info(self):
        # sample_vehicle: wheel_base 2.79, front_overhang 1.0, rear_overhang 1.1, wheel_tread 1.64, side overhangs 0.128
        length, width = 1.0 + 2.79 + 1.1, 1.64 + 2 * .128
        return dict(length=length, width=width, centre_ahead_of_base_link=(2.79 + 1.0) - length / 2.)

    def cmd_state(self):
        return self.ego()

    def cmd_end(self):
        """After an episode: objects off, stop mode, ego at rest -- Autoware must not keep driving towards the goal
        between episodes (its behaviour planner crashed near the goal after long idle drives)."""
        from autoware_adapi_v1_msgs.msg import OperationModeState
        self.cmd_clear()
        if self.mode is not None and self.mode.mode != OperationModeState.STOP:
            self.call('stop', self.S['ChangeOperationMode'].Request())
        self.set_twist(0.)
        return self.ego()

    def cmd_status(self):
        with self.lock:
            r, m = self.route, self.mode
        return dict(route=None if r is None else int(r.state),
                    mode=None if m is None else dict(mode=int(m.mode), in_transition=bool(m.is_in_transition),
                                                     autonomous_available=bool(m.is_autonomous_mode_available),
                                                     control_enabled=bool(m.is_autoware_control_enabled)),
                    ego=self.ego())

    # ---------------------------------------------------------------------------------------------- objects
    def object_msg(self, obj, action):
        x, y, yaw, vx, vy, length, width = obj
        m = self.M['DummyObject']()
        m.header.frame_id = 'map'
        m.header.stamp = self.get_clock().now().to_msg()
        m.id.uuid = list(self.obj_id)
        pose = m.initial_state.pose_covariance.pose
        pose.position.x, pose.position.y, pose.position.z = float(x), float(y), 0.
        pose.orientation.z, pose.orientation.w = yaw_q(yaw)
        m.initial_state.twist_covariance.twist.linear.x = float(math.hypot(vx, vy))  # along the heading
        m.classification.label = 1  # CAR
        m.classification.probability = 1.
        m.shape.type = 0  # BOUNDING_BOX
        m.shape.dimensions.x, m.shape.dimensions.y, m.shape.dimensions.z = float(length), float(width), 1.5
        m.max_velocity, m.min_velocity = 60., 0.
        m.action = action
        return m

    def cmd_clear(self):
        m = self.M['DummyObject']()
        m.header.frame_id = 'map'
        m.header.stamp = self.get_clock().now().to_msg()
        m.action = self.M['DummyObject'].DELETEALL
        self.pub_obj.publish(m)
        self.obj_id, self.obj_added = None, False
        return {}

    def cmd_step(self, obj=None):
        now = time.time()
        # the true lateness of this request (B54 recorded 0 when it restarted the pace, hiding stalls > 0.1 s)
        late = 0. if self.next_tick is None else max(0., now - self.next_tick)
        if self.next_tick is None or self.next_tick < now - DT:
            self.next_tick = now  # first step or the client fell behind by more than one tick: restart the pace
        if obj is not None:
            if self.obj_id is None:
                self.obj_id = uuid.uuid4().bytes
            D = self.M['DummyObject']
            m = self.object_msg(obj, D.MODIFY if self.obj_added else D.ADD)
            # stamp the state with its scheduled time, so that a late step does not shift the object back (the dummy
            # publisher extrapolates from the stamp)
            m.header.stamp = RosTime(nanoseconds=int(self.next_tick * 1e9)).to_msg()
            self.pub_obj.publish(m)
            self.obj_added = True
        self.next_tick += DT
        time.sleep(max(0., self.next_tick - time.time()))
        e = self.ego()
        e['late'] = round(late, 4)
        return e

    # ---------------------------------------------------------------------------------------------- reset
    def set_twist(self, v):
        tw = self.M['TwistStamped']()
        tw.header.frame_id = 'base_link'
        tw.header.stamp = self.get_clock().now().to_msg()
        tw.twist.linear.x = float(v)
        self.pub_twist.publish(tw)

    def cmd_reset(self, ego, goal, vmax, settle=1.0):
        """A failed reset leaves no engaged ego behind (an engaged ego drives on to the goal, where the behaviour planner
        crashed after long idle drives): objects off, stop mode, ego at rest, then the error is raised."""
        try:
            return self._reset(ego, goal, vmax, settle)
        except Exception:
            try:
                self.cmd_end()
            except Exception:
                pass
            raise

    def _reset(self, ego, goal, vmax, settle):
        from autoware_adapi_v1_msgs.msg import OperationModeState, RouteState
        from geometry_msgs.msg import Pose, PoseWithCovarianceStamped
        x, y, yaw, v = ego
        timings = {}
        self.cmd_clear()
        self.next_tick = None
        if self.mode is not None and self.mode.mode != OperationModeState.STOP:
            timings['stop'] = self.call('stop', self.S['ChangeOperationMode'].Request())
            timings['wait_stop'] = self.wait(lambda: self.mode.mode == OperationModeState.STOP and not self.mode.is_in_transition,
                                             20., 'stop mode')
        if self.route is not None and self.route.state != RouteState.UNSET:
            timings['clear_route'] = self.call('clear_route', self.S['ClearRoute'].Request())
            timings['wait_unset'] = self.wait(lambda: self.route.state == RouteState.UNSET, 20., 'route unset')
        # the ego pose (localization API in simulation mode)
        req = self.S['InitializeLocalization'].Request()
        p = PoseWithCovarianceStamped()
        p.header.frame_id = 'map'
        p.header.stamp = self.get_clock().now().to_msg()
        p.pose.pose.position.x, p.pose.pose.position.y = float(x), float(y)
        p.pose.pose.orientation.z, p.pose.pose.orientation.w = yaw_q(yaw)
        req.pose = [p]
        timings['init_pose'] = self.call('init_pose', req)
        self.wait(lambda: (lambda e: e is not None and math.hypot(e['x'] - x, e['y'] - y) < 1.)(self.ego()), 10., 'ego at the initial pose')
        # the simulator keeps the previous speed across a pose reset: stop the ego first (the planning validator rejects
        # a trajectory whose speed is far from the ego's, which blocks autonomous mode)
        self.set_twist(0.)
        timings['wait_rest'] = self.wait(lambda: (lambda e: e is not None and abs(e['v']) < .05)(self.ego()), 10., 'ego at rest')
        # route and velocity limit
        req = self.S['SetRoutePoints'].Request()
        req.header.frame_id = 'map'
        req.header.stamp = self.get_clock().now().to_msg()
        req.option.allow_goal_modification = False
        gp = Pose()
        gp.position.x, gp.position.y = float(goal[0]), float(goal[1])
        gp.orientation.z, gp.orientation.w = yaw_q(goal[2])
        req.goal = gp
        timings['route_before_set'] = None if self.route is None else int(self.route.state)
        for attempt in range(3):
            timings['set_route'] = self.call('set_route', req)
            if timings['set_route']['success']:
                break
            timings[f'set_route_fail_{attempt}'] = dict(timings['set_route'], route=None if self.route is None else int(self.route.state))
            try:  # e.g. "The route is already set.": clear and retry
                self.call('clear_route', self.S['ClearRoute'].Request())
                self.wait(lambda: self.route is not None and self.route.state == RouteState.UNSET, 10., 'route unset')
            except RuntimeError:
                time.sleep(1.)
        if not timings['set_route']['success']:
            raise RuntimeError(f"route not set: {json.dumps(timings)}")
        timings['wait_set'] = self.wait(lambda: self.route is not None and self.route.state == RouteState.SET, 20., 'route set')
        t_set = time.time()
        timings['wait_planning'] = self.wait(lambda: self.traj_wall is not None and self.traj_wall > t_set, 10.,
                                             'planning output (planning dead?)')
        vl = self.M['VelocityLimit']()
        vl.stamp = self.get_clock().now().to_msg()
        vl.max_velocity = float(vmax)
        vl.use_constraints = False
        vl.sender = 'fairtest_b54'
        self.pub_vlim.publish(vl)
        # engage, then the case speed. The transition manager's stability check compares the ego with the planned speed:
        # wait until the plan starts at rest; a transition that does not complete is aborted (stop) and tried once more
        try:
            timings['wait_plan_rest'] = self.wait(lambda: self.traj_v0 is not None and abs(self.traj_v0) < 1., 10.,
                                                  'planned speed at rest')
        except RuntimeError:
            timings['plan_v0'] = self.traj_v0
        for attempt in range(2):
            timings['wait_available'] = self.wait(lambda: self.mode is not None and self.mode.is_autonomous_mode_available, 20.,
                                                  'autonomous mode available')
            timings['auto'] = self.call('auto', self.S['ChangeOperationMode'].Request())
            try:
                timings['wait_autonomous'] = self.wait(lambda: self.mode is not None and self.mode.mode == OperationModeState.AUTONOMOUS
                                                       and not self.mode.is_in_transition, 12., 'autonomous mode')
                break
            except RuntimeError:
                m = self.mode
                timings[f'engage_fail_{attempt}'] = dict(
                    plan_v0=self.traj_v0, mode=None if m is None else int(m.mode),
                    in_transition=None if m is None else bool(m.is_in_transition),
                    available=None if m is None else bool(m.is_autonomous_mode_available))
                if attempt == 1:
                    raise RuntimeError(f'engage failed: {json.dumps(timings)}')
                self.call('stop', self.S['ChangeOperationMode'].Request())
                self.wait(lambda: self.mode.mode == OperationModeState.STOP and not self.mode.is_in_transition, 20., 'stop mode')
        for attempt in range(3):  # the case speed; the twist is re-sent if the simulator did not take it
            self.set_twist(v)
            try:
                timings['wait_speed'] = self.wait(lambda: (lambda e: e is not None and abs(e['v'] - v) < .5)(self.ego()), 2.,
                                                  'ego speed')
                break
            except RuntimeError:
                timings[f'speed_retry_{attempt}'] = (lambda e: None if e is None else round(e['v'], 3))(self.ego())
                if attempt == 2:
                    raise
        time.sleep(settle)
        return dict(timings=timings, ego=self.ego())


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        node = self.server.node
        for line in self.rfile:
            req = json.loads(line)
            try:
                fn = getattr(node, 'cmd_' + req.pop('cmd'))
                res = dict(ok=True, result=fn(**req))
            except Exception as e:  # reported to the client; the bridge keeps running
                res = dict(ok=False, error=f'{type(e).__name__}: {e}')
            self.wfile.write((json.dumps(res) + '\n').encode())
            self.wfile.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=5454)
    a = ap.parse_args()
    rclpy.init()
    node = Bridge()
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    server = socketserver.ThreadingTCPServer(('127.0.0.1', a.port), Handler)
    server.node = node
    print(json.dumps(dict(listening=a.port)), flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
