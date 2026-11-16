"""For Fig. S3(b): rebuild the highway-env trajectory of the adversary vehicle for the plan (success on the first query) that the final in-CARLA
search (E4, screened with the matched twin, S1) found on case 207 against BehaviorAgent normal, with the same method as the search (carla_search.adversary_row) and the same twin geometry (M).
    FAIRTEST_DATA=.../data PYTHONPATH=.../code FAIRTEST_CC_PERCEIVE=1.251 FAIRTEST_VEH_W=2.166 FAIRTEST_VEH_L=4.718 \
        python make_search_row.py OUT.jsonl
"""
import json
import os
import sys

from fairtest import runner
from fairtest.carla_search import adversary_row

SRC = os.path.join(os.environ.get('FAIRTEST_WORK', os.path.expanduser('~/fairtest_work')), 'runs', 'E4_search_S1.jsonl')
rec = next(r for r in map(json.loads, open(SRC)) if r['case_id'] == 207 and r['kind'] == 'ba:normal' and r['arm'] == 'S1')
runner._init()
valid, row = adversary_row(runner._STATE['cases'][207], rec['best']['theta'])
row.update(tag='carlasearch', source='E4 S1 ba:normal case 207 first query (final agent, matched-twin screening)', case_id=207,
           sut='ba:normal', valid=valid, hwenv={}, search=dict(theta=rec['best']['theta'], carla_cc=rec.get('carla_cc'),
                                                               cc_clearance_twin=rec['best'].get('cc_clearance'),
                                                               contact_dx=rec['best'].get('contact_dx')))
open(sys.argv[1], 'w').write(json.dumps(row) + '\n')
print('valid', valid, 'carla_cc', rec.get('carla_cc'), 'twin C clearance', rec['best'].get('cc_clearance'))
