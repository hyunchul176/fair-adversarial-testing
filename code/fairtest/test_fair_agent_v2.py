"""Contract checks of fair_agent_v2.attack without simulation (W5 P1-C): stages and certificates are stubbed.

  python -m fairtest.test_fair_agent_v2
"""
from __future__ import annotations

import unittest
from unittest import mock

from fairtest import b29_consensus, fair_agent_v2 as F

GOOD = dict(cc_contact=False, cc_clearance=1.2)


def agent(joint):
    a = F.FairAdversaryV2.__new__(F.FairAdversaryV2)  # no policy file needed
    a.policy_path, a.budget, a.joint = 'stub', 36, joint
    return a


def run(a, policy, refine, certs, realism=None, front=True):
    certs = iter(certs)
    with mock.patch.object(F.FairAdversaryV2, 'policy_stage', lambda self, c, s: policy), \
            mock.patch.object(F.FairAdversaryV2, 'boundary_plan', staticmethod(lambda c: (None, 36))), \
            mock.patch.object(F.FairAdversaryV2, 'refine_stage', lambda self, c, s, b, i=0: refine), \
            mock.patch.object(b29_consensus, 'task', lambda job: next(certs)), \
            mock.patch.object(F, '_front_contact', lambda *x: front), \
            mock.patch.object(F, '_cc_realism', lambda *x: realism), \
            mock.patch('fairtest.runner._init', lambda: None):
        return a.attack(204, ('following', 0.))


NO_POLICY = dict(valid=True, contact=False, actions=[], cc_rollouts=3)
HIT = dict(valid=True, contact=True, actions=[[0, -1]], cc_rollouts=5)
REF_OK = dict(found=True, queries=7, theta=[0.] * 5)
REF_NONE = dict(found=False, queries=36, theta=None)


class Contract(unittest.TestCase):
    def test_refine_certificate_error_is_not_reported(self):
        for joint in (False, True):
            r = run(agent(joint), NO_POLICY, REF_OK, [dict(error='injected')])
            self.assertFalse(r['found'])
            self.assertFalse(r['certified'])
            self.assertEqual(r['status'], 'uncertified')

    def test_refine_success_is_certified(self):
        r = run(agent(True), NO_POLICY, REF_OK, [GOOD])
        self.assertTrue(r['found'] and r['certified'])
        self.assertEqual((r['stage'], r['queries']), ('refine', 8))
        self.assertEqual(r['cost']['sut_search_episodes'], 8)

    def test_policy_certificate_error_falls_through(self):
        r = run(agent(False), HIT, REF_NONE, [dict(error='injected')])
        self.assertFalse(r['found'])
        self.assertEqual(r['queries'], 37)

    def test_v2j_policy_needs_realism(self):
        outside = dict(exceedance=.4, joint_score=3.5, g2=True, joint=False)
        r = run(agent(True), HIT, REF_NONE, [GOOD], realism=outside)
        self.assertFalse(r['found'])  # robust but outside the joint support -> stage 2 (which finds nothing)
        inside = dict(exceedance=.4, joint_score=1.1, g2=True, joint=True)
        r = run(agent(True), HIT, REF_NONE, [GOOD], realism=inside)
        self.assertEqual((r['found'], r['stage'], r['queries']), (True, 'policy', 1))
        over = dict(exceedance=1.01, joint_score=1.1, g2=False, joint=True)
        r = run(agent(True), HIT, REF_NONE, [GOOD], realism=over)
        self.assertFalse(r['found'])  # the per-feature G2 is part of the v2j contract

    def test_v2_policy_unchanged(self):
        r = run(agent(False), HIT, REF_NONE, [GOOD], realism=None)
        self.assertEqual((r['found'], r['stage'], r['queries']), (True, 'policy', 1))


if __name__ == '__main__':
    unittest.main()
