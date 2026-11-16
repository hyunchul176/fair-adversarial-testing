"""B13: retrain learned selectors on ALL highD cases (exposed 266 + B9 reserved 148) and freeze.

B9 reserved cases were used once (B9, frozen 5c3db5c); for the external NGSIM/exiD test they are
ordinary training data. B9 case ids are offset by 10000 to keep them distinct.

  python -m fairtest.b13_freeze_selectors OUTDIR
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier

from fairtest.analyze_landscape import load
from fairtest.b8_selector import candidates
from fairtest.boundary_planner import SUTS, index
from fairtest.features import Realism

EV = Path(os.environ.get('FAIRTEST_EVIDENCE', 'evidence'))  # local evidence folder of the study (not distributed)
B9_ROWS = [EV / 'fairtest_jobs' / s / 'landscape.jsonl' for s in ('b9-s0', 'b9-s1', 'b9-s2b', 'b9-s3b')]
B9_POOL = EV / 'b9_design' / 'reserved_cohort' / 'case_pool.json'
OFFSET = 10000
PARAMS = dict(max_iter=300, learning_rate=.05, max_leaf_nodes=31, l2_regularization=1., random_state=0)


def all_highd():
    from fairtest.report_b7 import rows_all
    from fairtest.runner import _STATE, _init
    _init()
    rows, _ = rows_all()
    cases = dict(_STATE['cases'])
    for p in B9_ROWS:
        r, e = load(p)
        if e:
            raise RuntimeError('error rows in ' + str(p))
        for x in r:
            x = dict(x, case_id=x['case_id'] + OFFSET)
            rows.append(x)
    for c in json.loads(B9_POOL.read_text(encoding='utf-8'))['cases']:
        cases[c['case_id'] + OFFSET] = dict(c, case_id=c['case_id'] + OFFSET)
    return rows, cases


def main(outdir):
    rows, cases = all_highd()
    by = index(rows)
    realism = Realism()
    models, summary = {}, {}
    for sut in SUTS:
        name = f'{sut[0]}_H{sut[1]:g}'
        cand = candidates(by, cases, realism, sut, math.inf)
        tr = [r for c in cand for r in cand[c]]
        pos = int(sum(r[2] for r in tr))
        if pos < 5:
            summary[name] = dict(model=None, positives=pos, fallback='r157_boundary_ranking')
            continue
        X = np.array([r[1] for r in tr], float)
        y = np.array([r[2] for r in tr], int)
        models[name] = HistGradientBoostingClassifier(**PARAMS).fit(X, y)
        summary[name] = dict(model='HistGradientBoostingClassifier', positives=pos, candidates=len(tr), cases=len(cand))
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    blob = pickle.dumps(models, protocol=5)
    (outdir / 'models.pkl').write_bytes(blob)
    manifest = dict(models_sha256=hashlib.sha256(blob).hexdigest(), sklearn=sklearn.__version__, params=PARAMS,
                    training_cases=len({r['case_id'] for r in rows}), suts=summary,
                    training_sources=['highD exposed calibration+development (266)', 'highD B9 reserved (148, used once)'])
    (outdir / 'MANIFEST.json').write_text(json.dumps(manifest, indent=1), encoding='utf-8')
    print(json.dumps(manifest, indent=1))


if __name__ == '__main__':
    main(sys.argv[1])
