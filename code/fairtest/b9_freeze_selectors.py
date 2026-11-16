"""Train and freeze the B9 learned plan selectors on ALL exposed cases (calibration+development).

One HistGradientBoosting classifier per SUT (same features/hyper-parameters as B8a). SUTs with
fewer than 5 positive training candidates get no model; B9 then falls back to the R157-boundary
ranking for that SUT (declared in the manifest). Output: models.pkl + MANIFEST.json (sha256).

  python -m fairtest.b9_freeze_selectors OUTDIR
"""
from __future__ import annotations

import hashlib
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier

from fairtest.b8_selector import candidates
from fairtest.boundary_planner import SUTS, index
from fairtest.features import Realism

PARAMS = dict(max_iter=300, learning_rate=.05, max_leaf_nodes=31, l2_regularization=1., random_state=0)


def main(outdir):
    from fairtest.report_b7 import rows_all
    from fairtest.runner import _STATE, _init
    _init()
    rows, check = rows_all()
    by = index(rows)
    realism = Realism()
    models, summary = {}, {}
    for sut in SUTS:
        name = f'{sut[0]}_H{sut[1]:g}'
        cand = candidates(by, _STATE['cases'], realism, sut, math.inf)
        tr = [r for c in cand for r in cand[c]]
        pos = int(sum(r[2] for r in tr))
        if pos < 5:
            summary[name] = dict(model=None, positives=pos, fallback='r157_boundary_ranking')
            continue
        X = np.array([r[1] for r in tr], float)
        y = np.array([r[2] for r in tr], int)
        m = HistGradientBoostingClassifier(**PARAMS).fit(X, y)
        models[name] = m
        summary[name] = dict(model='HistGradientBoostingClassifier', positives=pos, candidates=len(tr),
                             cases=len(cand), train_accuracy=float((m.predict(X) == y).mean()))
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    blob = pickle.dumps(models, protocol=5)
    (outdir / 'models.pkl').write_bytes(blob)
    manifest = dict(models_sha256=hashlib.sha256(blob).hexdigest(), sklearn=sklearn.__version__,
                    numpy=np.__version__, params=PARAMS, training_rows=check, suts=summary,
                    training_partitions=['calibration', 'development'], reserved_rows_used=0)
    (outdir / 'MANIFEST.json').write_text(json.dumps(manifest, indent=1), encoding='utf-8')
    print(json.dumps(manifest, indent=1))


if __name__ == '__main__':
    main(sys.argv[1])
