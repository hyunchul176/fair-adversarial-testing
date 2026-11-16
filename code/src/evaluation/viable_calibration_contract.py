"""Pure paired-outcome contract; simulator/model imports deliberately absent."""

def paired_outcome(scenario, seed, attack, nominal, witness):
    reasons=[]
    rows=[nominal,attack]+([witness] if witness is not None else [])
    if not all(r.get('valid_reset',False) for r in rows):reasons.append('invalid_initial_scene')
    if len({r.get('initial_hash') for r in rows})!=1:reasons.append('paired_world_mismatch')
    for name,row in (('nominal',nominal),('attack',attack)):
        if not row.get('full_horizon',False):reasons.append(name+'_not_full_horizon')
        if any(row.get(k) for k in ('pair_contact','background_contact','unattributed_crash')):reasons.append(name+'_contact')
        if row.get('offroad') or row.get('wrong_way'):reasons.append(name+'_physical_failure')
    if scenario=='cut_in' and not attack.get('merge_complete',False):reasons.append('attack_merge_incomplete')
    if witness is None:reasons.append('missing_recoverability_witness')
    elif not witness.get('full_horizon',False) or witness.get('contacts') or witness.get('offroad') or witness.get('wrong_way') or witness.get('target_progress_m',0)<80:
        reasons.append('witness_not_safe_and_progressing')
    delta=float(attack.get('risk',0)-nominal.get('risk',0))
    if delta < -1e-9:reasons.append('negative_added_risk')
    return dict(seed=int(seed),admissible=not reasons,rejection_reasons=sorted(set(reasons)),paired_risk_delta=delta)


def legacy_view(row):
    """No old output field may be omitted when checking nominal equivalence."""
    return {k:v for k,v in row.items() if k!='adapter'}


def same_paid_result(a,b):
    return all(a[k]==b[k] for k in ('admissible','rejection_reasons','paired_risk_delta')) and all(
        a[p]==b[p] for p in ('attack','nominal','witness'))
