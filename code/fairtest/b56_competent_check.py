"""B56 S3 follow-up (rule fixed before the run): judge each bo_warm_g4s success on a competent SUT with the held-out
reference drivers (b29_consensus.task: consensus-fair, R157 grade, SUT failure-mode features) and add the CC's own
features (speed, gap and headway when the cut-in becomes visible) for comparison with the SUT's."""
import json, sys
from fairtest import devboot  # noqa
from fairtest import runner
from fairtest.b29_consensus import task, features, _run
runner._init()
rows = json.load(open(sys.argv[1], encoding='utf-8'))
out = []
for o in rows:
    r = task(('b56_dev.jsonl', 'refine_g4s', o['case_id'], o['sut'], 'theta', o['theta']))
    case = runner._STATE['cases'][o['case_id']]
    r['cc_features'] = features(case, _run(case, 'theta', o['theta'], 'cc_r157'))
    out.append(r)
    f, c = r.get('features', {}), r['cc_features']
    print(json.dumps(dict(case=o['case_id'], sut=o['sut'], consensus_fair=r.get('consensus_fair'), grade=r.get('grade'),
                          refs_contact=r.get('refs_contact'), sut_closing_cutin=f.get('closing_cutin'), cc_closing_cutin=c.get('closing_cutin'),
                          sut_gap_cutin=f.get('gap_cutin_m'), cc_gap_cutin=c.get('gap_cutin_m'), sut_brake_delay=f.get('sut_brake_delay_s'),
                          cc_brake_delay=c.get('sut_brake_delay_s'), sut_min_acc=f.get('sut_min_acc'), cc_min_acc=c.get('sut_min_acc'),
                          error=r.get('error'))))
json.dump(out, open(sys.argv[2], 'w', encoding='utf-8'), indent=1)
