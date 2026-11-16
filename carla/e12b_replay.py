"""E12b (addendum 7, post hoc): replay with a C whose perception threshold alone is changed. The harness (carla_replay) is left as is and only the kind `ccp:<m>` is added.

  ccp:<m>   as the R157 C (`cc_target.CCDriver`), but with the perception threshold at m from the centre of the adversary vehicle's own lane (default 1.095 m)

  python e12b_replay.py OUT.jsonl TRAJECTORIES.jsonl --kinds ccp:0.595 --port 2030 --resume
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'code'))
from fairtest import carla_replay as R  # noqa: E402

_init = R.ShimTarget.__init__


def _shim_init(self, kind, cruise, base):
    if not kind.startswith('ccp:'):
        return _init(self, kind, cruise, base)
    from fairtest import cc_target
    self.kind, self.cruise, self.prev_a, self.integ = kind, cruise, 0., 0.
    self.v_cmd, self.braking, self.hw = None, False, False
    self.ctrl, self.target_id = cc_target.CCDriver(perceive_m=float(kind.split(':')[1])), 'cc_r157'


R.ShimTarget.__init__ = _shim_init

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out', nargs='?')
    ap.add_argument('trajectories', nargs='?')
    ap.add_argument('--kinds', default='ccp:0.595')
    ap.add_argument('--port', type=int, default=2000)
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--resume', action='store_true')
    R.main(ap.parse_args())
