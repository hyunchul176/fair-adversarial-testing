"""Pure lossless record storage and scoring; never executes a model or simulator."""
import hashlib
import json

PHASES=('attack','nominal','witness')

def digest(row):
    return hashlib.sha256(json.dumps(row,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def retain(store,row):
    key=digest(row)
    if key in store:
        assert store[key]==row
    else:
        store[key]=row
    return key

def pack(store,result):
    return {**{k:v for k,v in result.items() if k not in PHASES},
            'phase_refs':{k:retain(store,result[k]) if result[k] is not None else None for k in PHASES}}

def unpack(store,record):
    return {**{k:v for k,v in record.items() if k!='phase_refs'},
            **{k:store[v] if v is not None else None for k,v in record['phase_refs'].items()}}

def predictions(outcomes,selection,config):
    envelope=max([0.]+[o['paired_risk_delta'] for o in outcomes if o['admissible']])
    responsive=envelope>=config['responsiveness_threshold']
    rows=[]
    for level,index in zip(config['risk_levels'],selection['candidate_indices']):
        outcome=outcomes[index] if index is not None else None
        error=abs(outcome['paired_risk_delta']/envelope-level) if responsive and outcome and outcome['admissible'] else None
        minimum=min((abs(o['paired_risk_delta']/envelope-level) for o in outcomes if o['admissible']),default=None) if responsive else None
        rows.append(dict(level=level,candidate_index=index,responsive=responsive,execute=selection['execute'],error=error,
            within_tolerance=error is not None and error<=config['tolerance'],oracle_reachable=minimum is not None and minimum<=config['tolerance'],
            inadmissible_execution=bool(outcome and not outcome['admissible']),nonresponse_abstain=not responsive and not selection['execute']))
    return dict(envelope=envelope,responsive=responsive,rows=rows)
