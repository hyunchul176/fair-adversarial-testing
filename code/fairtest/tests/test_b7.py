"""B7 development tests (unittest). Run from code/ with the pinned venv:
    python -m unittest fairtest.tests.test_b7 -v
"""
import json
import math
import unittest

import numpy as np

from fairtest import devboot  # noqa: F401
from fairtest import cc_target
from fairtest.cases import RESERVED, build_pool, load_events
from fairtest.features import Realism, simulated_features
from fairtest.plans import Plan, grid


class FakeVehicle:
    LENGTH, WIDTH = 5., 2.

    def __init__(self, x, y, speed, heading=0.):
        self.position = np.array([x, y], float)
        self.heading = heading
        self.speed = speed
        self.action = None

    @property
    def velocity(self):
        return self.speed * np.array([math.cos(self.heading), math.sin(self.heading)])

    def act(self, action):
        self.action = action


class FakeInner:
    def __init__(self, generator, target):
        self.vehicle = generator
        self.road = type('Road', (), {'vehicles': [generator, target]})()


class FakeSim:
    def __init__(self, generator, target):
        self.inner = FakeInner(generator, target)
        self.target = target
        self.cc_clock = 0.


def step_cc(driver, sim, previous, cruise=30.):
    applied, telemetry = driver(sim, 'cc_r157', None, cruise, previous)
    return applied, telemetry


class CCDriverTests(unittest.TestCase):
    def test_parameters_match_r157_summary(self):
        self.assertAlmostEqual(cc_target.WANDER_M + cc_target.PERCEPTION_M, 1.095)
        self.assertEqual(cc_target.REACTION_S, .75)
        self.assertEqual(cc_target.EMERGENCY_MPS2, -7.59)
        self.assertEqual(cc_target.JERK_MPS3, 12.65)
        self.assertEqual(cc_target.TTC_EMERGENCY_S, 2.)

    def test_no_reaction_inside_perception_zone(self):
        g, t = FakeVehicle(30., 1.0, 25.), FakeVehicle(0., 4., 30.)
        a, tel = step_cc(cc_target.CCDriver(), FakeSim(g, t), 0.)
        self.assertEqual(tel['recognition']['cc_phase'], 'free')
        self.assertGreaterEqual(a, 0.)

    def test_reaction_then_emergency_ramp(self):
        g, t = FakeVehicle(12., 1.2, 20.), FakeVehicle(0., 4., 30.)
        sim, driver, prev = FakeSim(g, t), cc_target.CCDriver(), 0.
        phases, accs = [], []
        for k in range(20):
            sim.cc_clock = k * .1
            prev, tel = step_cc(driver, sim, prev)
            phases.append(tel['recognition']['cc_phase'])
            accs.append(prev)
        self.assertEqual(phases[:8], ['reaction'] * 8)  # 0.0 .. 0.7 s
        self.assertTrue(all(abs(a - cc_target.REACTION_MPS2) < 1e-9 for a in accs[:8]))
        self.assertIn('emergency', phases[8:])
        jerks = np.abs(np.diff(accs)) / .1
        self.assertLessEqual(jerks.max(), cc_target.JERK_MPS3 + 1e-9)
        self.assertAlmostEqual(min(accs), cc_target.EMERGENCY_MPS2)

    def test_no_emergency_when_ttc_large(self):
        g, t = FakeVehicle(80., 1.2, 29.), FakeVehicle(0., 4., 30.)
        sim, driver, prev = FakeSim(g, t), cc_target.CCDriver(), 0.
        for k in range(12):
            sim.cc_clock = k * .1
            prev, tel = step_cc(driver, sim, prev)
        self.assertFalse(tel['recognition']['emergency'])
        self.assertGreaterEqual(prev, cc_target.COMFORT_MPS2 - 1e-9)


class PoolTests(unittest.TestCase):
    def test_reserved_recordings_absent(self):
        self.assertFalse(any(e['recording'] in RESERVED for e in load_events()))

    def test_pool_and_archived_dev16(self):
        cases, ledger = build_pool()
        self.assertEqual(len(ledger), 463)
        self.assertEqual(len(cases), 266)
        dev16 = [c for c in cases if c['dev16']]
        self.assertEqual(len(dev16), 16)
        self.assertTrue(all(c['initial_state_source'] == 'archived_b5_server' for c in dev16))
        self.assertEqual(len({c['event']['event_id'] for c in cases}), 266)


class PlanFeatureTests(unittest.TestCase):
    def test_grid_size_and_nominal(self):
        plans = grid()
        self.assertEqual(len(plans), 199)
        self.assertTrue(math.isinf(plans[0].entry_s))
        self.assertEqual(len({p.name for p in plans}), 199)

    def test_plan_action_schedule(self):
        p = Plan(2., -3., 0.)
        env = type('E', (), {'tick': 0})()
        self.assertEqual(p.act(env), (0., -3.))
        env.tick = 20
        self.assertEqual(p.act(env), (4., 0.))

    def test_features_on_synthetic_cut_in(self):
        ticks = 60
        init = [dict(x=2000., y=.4, vx=30., vy=0., heading=0., length=5., width=2.),
                dict(x=1980., y=4., vx=32., vy=0., heading=0., length=5., width=2.)]
        trace = []
        for k in range(1, ticks + 1):
            y = min(4., .4 + 3.6 * k / 40)
            trace.append(dict(generator=dict(x=2000. + 30. * .1 * k, y=y, vx=30.),
                              target=dict(x=1980. + 32. * .1 * k, y=4., vx=32.)))
        f = simulated_features(dict(trace=trace, physical_initial_state=init))
        self.assertTrue(f['cut_in'])
        self.assertAlmostEqual(f['closing_p50'], 2.)
        self.assertAlmostEqual(f['acc_pre'], 0.)
        self.assertGreater(f['gap_p50'], 0.)
        self.assertIsNotNone(f['duration_10_90'])

    def test_realism_inside_and_outside(self):
        r = Realism()
        inside = r.score(dict(gap_p50=37., thw_p50=1.1, closing_p50=-1.5, acc_pre=0., acc_post=0., duration_10_90=5.))
        outside = r.score(dict(gap_p50=3., thw_p50=.1, closing_p50=12., acc_pre=-3., acc_post=-3., duration_10_90=2.))
        self.assertEqual(inside['max_exceedance'], 0.)
        self.assertGreater(outside['max_exceedance'], 1.)


class ReplayTests(unittest.TestCase):
    def test_frozen_b5_replay_matches_server(self):
        from fairtest.runner import validate_b5
        self.assertLess(validate_b5(), 1e-12)


if __name__ == '__main__':
    unittest.main()
