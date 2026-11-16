"""E17 (design addendum 12): run the carla_rl closed loop, but match only the C of the shield twin to the vehicle in the run (perception threshold, reaction-phase deceleration,
emergency level). Only env.CCDriver (the twin C) is swapped, so CARLA's C (fairness verdict, carla_replay.ShimTarget) stays the R157 C.
The twin width is given by FAIRTEST_VEH_W and the shield's C target by FAIRTEST_SHIELD_CC (as in carla_rl).

  FAIRTEST_E17_TWIN='1.251,0.0,-7.35' python e17_closed.py OUT.jsonl MODEL.zip [the carla_rl arguments as they are]
  (without FAIRTEST_E17_TWIN nothing is changed: E17-B)
"""
import os
import runpy
import sys
from pathlib import Path

W = Path(__file__).resolve().parent
sys.path.insert(0, str(W / 'code'))

spec = os.environ.get('FAIRTEST_E17_TWIN')
if spec:
    import fairtest.cc_target as CT
    import fairtest.env as E
    perceive, react, emergency = (float(x) for x in spec.split(','))
    shield = os.environ.get('FAIRTEST_SHIELD_CC', 'cc_r157')

    class TwinCC(CT.CCDriver):
        """Twin C: the R157 C with only the perception threshold and the reaction-phase deceleration changed (the deceleration is changed only during the call, so it does not leak into CARLA's C)."""

        def __init__(self, *a, perceive_m=perceive, **k):
            super().__init__(*a, perceive_m=perceive_m, **k)

        def __call__(self, *a, **k):
            saved, CT.REACTION_MPS2 = CT.REACTION_MPS2, react
            try:
                return super().__call__(*a, **k)
            finally:
                CT.REACTION_MPS2 = saved

    E.CCDriver = TwinCC
    E.CC_EMERGENCY = dict(E.CC_EMERGENCY, **{shield: emergency})

sys.argv = ['carla_rl'] + sys.argv[1:]
runpy.run_module('fairtest.carla_rl', run_name='__main__', alter_sys=True)
