"""B69: run any fairtest module with a third-party SUT in the `acc_prod_aeb` slot (frozen code unchanged).

The SUT slot `acc_prod_aeb` (9 m/s^2 contract) is filled by a third-party SUT by replacing sut_acc.ProductionACC before
the module runs; the module's worker pools fork after the replacement and inherit it. `acc_prod` (no AEB) keeps the B29
controller. Outputs label the SUT `acc_prod_aeb_H0`; the analysis renames. Exactly one of:
  FAIRTEST_SUMO=acc|krauss|idm  a SUMO car-following model (sut_sumo); libsumo from FAIRTEST_SUMO_PATH (pip --target
                                 folder, prepended to sys.path)
  FAIRTEST_CR=rp                the CommonRoad reactive planner (sut_commonroad); packages from FAIRTEST_CR_PATH (pip
                                 --target folder, appended to sys.path so that the environment's own packages win)

  FAIRTEST_SUMO=acc FAIRTEST_SUMO_PATH=$HOME/fairtest/ext/sumo python -m fairtest.b69_run fairtest.b17_rl eval ...
  FAIRTEST_CR=rp FAIRTEST_CR_PATH=$HOME/fairtest/ext/commonroad python -m fairtest.b69_run fairtest.b61_natural ...
"""
from __future__ import annotations

import os
import runpy
import sys


def install():
    sumo, cr = os.environ.get('FAIRTEST_SUMO'), os.environ.get('FAIRTEST_CR')
    if bool(sumo) == bool(cr):
        raise RuntimeError('set exactly one of FAIRTEST_SUMO (SUMO variant) and FAIRTEST_CR=rp (CommonRoad planner)')
    from fairtest import sut_acc
    if sumo:
        path = os.environ.get('FAIRTEST_SUMO_PATH')
        if path and path not in sys.path:
            sys.path.insert(0, path)
        from fairtest.sut_sumo import SumoSUT as SUT, _variant
        _variant()  # fail early on a missing or unknown variant
    else:
        if cr != 'rp':
            raise RuntimeError('FAIRTEST_CR must be rp')
        path = os.environ.get('FAIRTEST_CR_PATH')
        if path and path not in sys.path:
            sys.path.append(path)
        from fairtest.sut_commonroad import CommonRoadSUT as SUT
    original = sut_acc.ProductionACC

    def factory(aeb=False):
        return SUT() if aeb else original(aeb=False)
    sut_acc.ProductionACC = factory


if __name__ == '__main__':
    install()
    module = sys.argv[1]
    sys.argv = [module] + sys.argv[2:]
    runpy.run_module(module, run_name='__main__', alter_sys=True)
