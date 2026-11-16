"""Metadata-only eligibility for a fixed perturbation outside its selection cases."""
import math


def classify(case, source_ids, selection_ids, gap_delta, speed_delta):
    i=case['case_id'];p=case['event']['metrics']['p10']
    if i not in source_ids:return 'prior_source_ineligible'
    if i in selection_ids:return 'selection_case'
    if not 15<=p['target_speed']<=35:return 'baseline_speed_outside'
    if not 15<=p['target_speed']+speed_delta<=35:return 'perturbed_speed_outside'
    if not math.isfinite(p['gap']+gap_delta) or p['gap']+gap_delta<=0:return 'nonpositive_gap'
    return 'eligible'
