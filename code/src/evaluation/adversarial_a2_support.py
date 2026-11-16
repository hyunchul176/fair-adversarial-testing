"""A2 bookkeeping and selection. No simulator imports or execution."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import numpy as np


def json_value(value):
    if isinstance(value, np.ndarray):return value.tolist()
    if isinstance(value, np.generic):return value.item()
    raise TypeError(type(value).__name__)


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),default=json_value,allow_nan=False).encode()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,default=json_value,allow_nan=False)+'\n',encoding='utf-8')


class Jsonl:
    def __init__(self,path):self.stream=Path(path).open('x',encoding='utf-8')
    def append(self,row):
        self.stream.write(json.dumps(row,default=json_value,allow_nan=False)+'\n');self.stream.flush()
    def close(self):self.stream.close()


class InputCycle:
    """Isolated RNG; each consecutive block visits every registered input."""
    def __init__(self,ids,seed):
        self.ids=list(ids)
        if len(set(self.ids))!=len(self.ids) or not self.ids:raise ValueError('Unique nonempty input pool required')
        self.rng=np.random.Generator(np.random.PCG64(seed));self.pending=[];self.draws=[]
    def next(self):
        if not self.pending:self.pending=self.rng.permutation(self.ids).tolist()
        value=self.pending.pop(0);self.draws.append(value);return value
    def state(self):return deepcopy(dict(rng=self.rng.bit_generator.state,pending=self.pending,draws=self.draws))


class BudgetLedger:
    """Reserve before stepping; charge actual ticks, or the upper bound on failure."""
    LIMITS={'screen':dict(episodes=240,actors=5760,ticks=28800),
            'wrapper':dict(episodes=8,actors=192,ticks=960),
            'training':dict(episodes=None,actors=8192,ticks=40960),
            'evaluation':dict(episodes=32,actors=768,ticks=3840)}
    def __init__(self,path=None):
        self.rows={k:dict(starts=0,completed=0,actors=0,ticks=0,measured_ticks=0,uncertain_failed_calls=0) for k in self.LIMITS}
        self.path=Path(path) if path else None
    def start(self,phase):
        row=self.rows[phase];cap=self.LIMITS[phase]['episodes']
        if cap is not None and row['starts']>=cap:raise RuntimeError('Episode budget exhausted')
        row['starts']+=1;self.save()
    def reserve_action(self,phase):
        row=self.rows[phase];cap=self.LIMITS[phase]
        if row['actors']>=cap['actors'] or row['ticks']+5>cap['ticks']:raise RuntimeError('Action/physics budget exhausted')
        if sum(x['ticks'] for x in self.rows.values())+5>74560:raise RuntimeError('Combined physics budget exhausted')
    def charge(self,phase,measured,failed=False):
        if not 0<=measured<=5 or (not failed and measured<1):raise RuntimeError('Invalid tick accounting')
        row=self.rows[phase];row['actors']+=1;row['ticks']+=5 if failed else measured
        row['measured_ticks']+=measured;row['uncertain_failed_calls']+=int(failed);self.save()
    def complete(self,phase):self.rows[phase]['completed']+=1;self.save()
    def save(self):
        if self.path:
            temporary=self.path.with_suffix('.tmp');write(temporary,self.rows);temporary.replace(self.path)


def equivalent(a,b):
    """Canonical schedules may differ only in ineffective request labels."""
    if any(a[k]!=b[k] for k in ('initial_hash','steps','pair_contact','min_pair_ttc','agent_return','agent_outcome','maneuver_valid_precontact')):return False
    if len(a['trace'])!=len(b['trace']) or len(a['decisions'])!=len(b['decisions']):return False
    fields=('generator','target','acceleration','jerk','pair_ttc','applied_action','target_control','pulse','agent_reward','agent_outcome','maneuver_checks','consumed_impacts')
    if any(any(x[k]!=y[k] for k in fields) for x,y in zip(a['trace'],b['trace'])):return False
    return all(x['tick']==y['tick'] and x['observation']==y['observation'] for x,y in zip(a['decisions'],b['decisions']))


def screen_summary(rows):
    if len(rows)!=256 or len({(r['input_id'],r['plan_id']) for r in rows})!=256:raise ValueError('Complete 16 by 16 matrix required')
    ids=sorted({r['input_id'] for r in rows})
    if ids!=list(range(16)) or {r['plan_id'] for r in rows}!=set(range(16)):raise ValueError('Wrong matrix domain')
    maximum={r['input_id']:r for r in rows if r['plan_id']==1}
    baseline_cases={r['case_id'] for r in maximum.values() if r['valid_risk']}
    winners=[];signals=[]
    for input_id in ids:
        choices=[r for r in rows if r['input_id']==input_id]
        value=max(r['agent_return'] for r in choices)
        winner=min((r for r in choices if r['agent_return']>=value-1e-9),key=lambda r:r['plan_id'])
        winners.append(dict(input_id=input_id,plan_id=winner['plan_id']))
        for r in choices:
            why=[]
            if r['valid_risk'] and r['case_id'] not in baseline_cases:why.append('new_valid_risk_case')
            if r['valid_pair_contact'] and not maximum[input_id]['valid_pair_contact']:why.append('contact_upgrade')
            if r['maneuver_valid_precontact'] and r['agent_return']-maximum[input_id]['agent_return']>=.02:why.append('return_gain_ge_0.02')
            if why:signals.append(dict(input_id=input_id,case_id=r['case_id'],plan_id=r['plan_id'],reasons=why))
    table=[]
    for plan_id in range(16):
        rr=[r for r in rows if r['plan_id']==plan_id]
        table.append(dict(plan_id=plan_id,runs=len(rr),valid=sum(r['maneuver_valid_precontact'] for r in rr),
            risk_conditions=sum(r['valid_risk'] for r in rr),risk_cases=sorted({r['case_id'] for r in rr if r['valid_risk']}),
            contacts=sum(r['valid_pair_contact'] for r in rr),mean_return=sum(r['agent_return'] for r in rr)/16))
    return dict(signal=bool(signals),signals=signals,winners=winners,plans=table,
                inference='Retrospective finite-library scheduling diagnostic, not a deployable feedback comparator')
