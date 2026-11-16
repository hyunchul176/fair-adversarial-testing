"""Pure fixed-tape reference diagnostics; original admissibility stays unchanged."""
import copy

def physical_view(row):
    # Everything except explicit control-stage/replay metadata must match for
    # a source target replaying its own complete attack tape.
    return {k:v for k,v in row.items() if k not in ('adapter','intervention_controls','anticipation_controls','replayed_tape')}

def following_view(row):
    out=copy.deepcopy(row)
    out['adapter']['version']='viable-calibration-integration-v1'
    return out

def reference_safe(row):
    return bool(row.get('valid_reset') and row.get('full_horizon') and
                not any(row.get(k) for k in ('contacts','offroad','wrong_way','pair_contact','background_contact','unattributed_crash')) and row.get('target_progress_m',0)>=80)

def eligible(attack):
    return bool(attack.get('full_horizon') and len(attack.get('generator_controls',[]))==120)
