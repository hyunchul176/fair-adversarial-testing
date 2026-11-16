"""Read-only independent audit of saved online-agent pilot trajectories."""
import argparse
from collections import Counter
from copy import deepcopy
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.audit_consumed_impact import swept, consumed_impact


def read(p):return json.loads(p.read_text())
def lines(p):
    with p.open() as f:
        for s in f:yield json.loads(s)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def native(x,scale):
    return float(np.float32(-scale)+(np.float32(x)+np.float32(1))*np.float32(2*scale)/np.float32(2))
def risk(ttc):return 0. if ttc is None else max(0.,min(1.,1-ttc/3))
def close(x,y,tol=1e-8):
    if x is None or y is None:assert x is y,(x,y)
    else:assert abs(x-y)<=tol,(x,y,tol)
def onroad(s):
    radius=(s['length']*abs(math.sin(s['heading']))+s['width']*abs(math.cos(s['heading'])))/2
    return s['y']-radius>=-2-1e-9 and s['y']+radius<=10+1e-9


def audit_episode(row,window,case,divisors):
    trace=row['trace'];initial=row['physical_initial_state'];tol=1e-5
    initial_ttc=swept(*initial)['ttc'];maximum=risk(initial_ttc);total=0.;previous_a=[0.,0.]
    previous_steer=0.;previous_state=initial;ys=[initial[0]['y']];valid=True
    observed_finite=[];max_before=[maximum];last_pulse=None
    for index,t in enumerate(trace):
        gc=t['generator_control'];states=[t['generator'],t['target']]
        ttc=swept(*states)['ttc'];close(ttc,t['pair_ttc'],1e-10)
        if ttc is not None:observed_finite.append(ttc)
        unimpacted=[]
        for actor,(old,s,pending) in enumerate(zip(previous_state,states,t['consumed_impacts'])):
            u=dict(s)
            if pending is not None:u['x']-=pending[0];u['y']-=pending[1]
            unimpacted.append(u)
            assert u==t['unimpacted_generator' if actor==0 else 'unimpacted_target']
            old_speed=math.hypot(old['vx'],old['vy']);speed=math.hypot(s['vx'],s['vy'])
            acceleration=(speed-old_speed)/.1
            close(acceleration,t['acceleration'][actor],1e-10)
            applied=native(t['applied_action'][0],5) if actor==0 else t['target_control']['applied_mps2']
            close(acceleration,applied,1e-5)
            steer=native(t['applied_action'][1],.3) if actor==0 else t['target_control']['steering']
            beta=math.atan(.5*math.tan(steer))
            close(u['x'],old['x']+old_speed*math.cos(old['heading']+beta)*.1)
            close(u['y'],old['y']+old_speed*math.sin(old['heading']+beta)*.1)
            close(s['heading'],old['heading']+old_speed*math.sin(beta)/(s['length']/2)*.1)
            if actor==1:
                assert abs(acceleration)<=5+tol and abs((acceleration-previous_a[1])/.1)<=2.5+tol
                assert onroad(u) and u['vx']>=-tol
        g=unimpacted[0];a=t['acceleration'][0];steer=native(t['applied_action'][1],.3)
        ys.append(g['y'])
        ay=(ys[-1]-2*ys[-2]+ys[-3])/.01 if len(ys)>=3 else 0.
        jy=(ys[-1]-3*ys[-2]+3*ys[-3]-ys[-4])/.001 if len(ys)>=4 else 0.
        checks=dict(acceleration=abs(a)<=3+tol,jerk=abs((a-previous_a[0])/.1)<=2+tol,
                    speed=15-tol<=math.hypot(g['vx'],g['vy'])<=40+tol,
                    realization=abs(a-native(t['applied_action'][0],5))<=tol,
                    steering=abs(steer)<=.3+tol,steering_slew=abs((steer-previous_steer)/.1)<=.15+tol,
                    path=abs(g['y']-gc['y_reference'])<=1e-5,speed_tracking=abs(g['vx']-gc['vx_reference'])<=.5,
                    lateral_acceleration=abs(ay)<=2+tol,lateral_jerk=abs(jy)<=4+tol,road=onroad(g),forward=g['vx']>=-tol)
        assert checks==t['maneuver_checks']
        valid &= all(checks.values())
        outcome=t['agent_outcome'];q=risk(ttc)
        if outcome=='maneuver_invalid':expected=-1-maximum
        elif outcome=='intended_contact':expected=2-maximum
        else:
            updated=max(maximum,q);expected=updated-maximum;maximum=updated
        close(t['agent_reward'],expected,1e-12);total+=expected;max_before.append(maximum)
        pulse=t['pulse']
        if last_pulse is None:last_pulse={'deficit':0.,'rate':0.,'phase':'armed'}
        close(pulse['deficit']-last_pulse['deficit'],.1*pulse['rate'],1e-10)
        assert 0<=pulse['deficit']<=2 and abs(pulse['rate'])<=1+1e-10 and abs(pulse['rate']-last_pulse['rate'])<=.1+1e-10
        if last_pulse['phase'] in ('release','done'):assert pulse['phase'] in ('release','done')
        last_pulse=pulse;previous_a=t['acceleration'];previous_steer=steer;previous_state=states
    assert valid==row['maneuver_valid_precontact'];close(total,row['agent_return'],1e-12)
    close(min(observed_finite,default=None),row['min_pair_ttc'],1e-10)
    expected_final=(-1 if not valid else 2 if row['pair_contact'] else maximum)-risk(initial_ttc)
    close(total,expected_final,1e-12)
    assert row['initial_risk_event']==(initial_ttc is not None and initial_ttc<1.5)
    contacts=[]
    if row['pair_contact']:
        assert row['agent_outcome']=='intended_contact' if valid else row['agent_outcome']=='maneuver_invalid'
        if all(e['kind']=='scheduled_impact' for e in row['confirmed_contacts']):
            diagnostic=dict(row,input_id=row.get('input_id'),family=row['policy'],horizon_s=0.,gap_delta_m=None,speed_delta_mps=None)
            contacts=consumed_impact(diagnostic)['impacts']
        else:
            for e in row['confirmed_contacts']:
                if e['kind']=='overlap':assert swept(*e['states'])['current_overlap']
                else:assert e['step']<row['steps']-1
    elif valid:assert row['steps']==120 and row['agent_outcome']=='horizon'
    # Reconstruct every decision input from already-observed states. No policy
    # input may use the target command selected for its upcoming transition.
    for d in row['decisions']:
        tick=d['tick'];g,t=initial if tick==0 else [trace[tick-1]['generator'],trace[tick-1]['target']]
        pulse={'deficit':0.,'rate':0.,'goal':0.,'phase':'armed'} if tick==0 else trace[tick-1]['pulse']
        a=[0.,0.] if tick==0 else trace[tick-1]['acceleration']
        steer=0. if tick==0 else native(trace[tick-1]['applied_action'][1],.3)
        ts=min(window['times'][0]+tick*.1,case['continuation_end_time'])
        nominal=float(np.interp(ts,window['generator']['time'],window['generator']['vx']))*window['sign']
        # Current path reference is documented, own exogenous input; use the
        # archived previous reference, and reset y at the initial boundary.
        lateral=initial[0]['y'] if tick==0 else trace[tick-1]['generator_control']['y_reference']
        ttc=swept(g,t)['ttc']
        values=[g['x']-t['x'],g['y']-t['y'],g['vx'],g['vy'],t['vx'],t['vy'],
                math.sin(g['heading']),math.cos(g['heading']),math.sin(t['heading']),math.cos(t['heading']),
                *a,float(tick>0),tick*.1,lateral,nominal,pulse['deficit'],pulse['rate'],pulse['goal'],steer,
                risk(ttc),float(ttc is not None),max_before[tick],
                *[float(pulse['phase']==phase) for phase in ('armed','active','release','done')]]
        expected=np.clip(np.asarray(values)/np.asarray(divisors),-1,1).astype(np.float32)
        assert np.array_equal(expected,np.asarray(d['observation'],dtype=np.float32)),(row['run_id'],tick,'observation')
    return dict(run_id=row['run_id'],ticks=len(trace),decision_count=len(row['decisions']),contacts=contacts)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=a.root/'result';data=a.root/'data';c=read(result/'complete.json');spec=read(data/'execution_protocol.json')
    for name,h in c['hashes'].items():assert sha(result/name)==h
    for name,h in spec['input_hashes'].items():assert sha(data/name)==h
    cases={r['case_id']:r for r in read(data/'selected_cases.json')};windows=read(data/'windows.json')
    divisors=[f['divisor'] for f in read(data/'contract.json')['observation']['features']]
    summaries=[];audit=[];main=[];repeats={};sentinels=read(data/'sentinels.json')
    for filename in ('sentinels.jsonl','episodes.jsonl','repeats.jsonl'):
        for row in lines(result/filename):
            audit.append(audit_episode(row,windows[row['event_id']],cases[row['case_id']],divisors))
            if row['group']=='sentinel':
                ref=sentinels[row['sentinel_id']]['reference']
                assert row['initial_hash']==ref['initial_hash'] and row['steps']==ref['steps']
                for new,old in zip(row['trace'],ref['trace']):
                    for k in ('generator','target','acceleration','jerk','pair_ttc','following_drac','target_control','applied_action'):assert new[k]==old[k]
            elif row['group']=='main':
                main.append(row);repeats[row['input_id'],row['target_id'],row['policy']]=row['trace']
            else:assert row['trace']==repeats[row['input_id'],row['target_id'],row['policy']]
    assert len(audit)==320 and sum(r['ticks'] for r in audit)==c['ticks']
    zero={(r['input_id'],r['target_id']):r for r in main if r['policy']=='zero'}
    for policy in spec['policies']:
        rr=[r for r in main if r['policy']==policy]
        added=[r for r in rr if r['valid_risk'] and not zero[r['input_id'],r['target_id']]['valid_risk']]
        summaries.append(dict(policy=policy,runs=len(rr),valid=sum(r['maneuver_valid_precontact'] for r in rr),
            contacts=sum(r['valid_pair_contact'] for r in rr),risk_conditions=sum(r['valid_risk'] for r in rr),
            risk_case_ids=sorted({r['case_id'] for r in rr if r['valid_risk']}),
            additional_conditions_vs_zero=len(added),additional_case_ids_vs_zero=sorted({r['case_id'] for r in added}),
            new_case_ids_vs_zero=sorted({r['case_id'] for r in rr if r['valid_risk']}-{r['case_id'] for r in zero.values() if r['valid_risk']}),
            mean_return=sum(r['agent_return'] for r in rr)/len(rr),min_ttc=min((r['min_pair_ttc'] for r in rr if r['min_pair_ttc'] is not None),default=None),
            physical_ticks=sum(r['steps'] for r in rr)))
    comparisons=[]
    by_key={(r['input_id'],r['target_id'],r['policy']):r for r in main}
    for input_id,target in zero:
        maximum=by_key[input_id,target,'maximum'];feedback=by_key[input_id,target,'feedback_rule']
        prefix=min(len(maximum['trace']),len(feedback['trace']))
        comparisons.append(dict(input_id=input_id,target=target,
            different_applied_commands=any(x['applied_action']!=y['applied_action'] for x,y in zip(maximum['trace'][:prefix],feedback['trace'][:prefix])),
            feedback_minus_maximum_return=feedback['agent_return']-maximum['agent_return'],
            feedback_minus_maximum_min_ttc=(feedback['min_pair_ttc']-maximum['min_pair_ttc']) if feedback['min_pair_ttc'] is not None and maximum['min_pair_ttc'] is not None else None,
            feedback_adds_event=feedback['valid_risk'] and not maximum['valid_risk']))
    target_adaptation=[]
    for input_id in sorted({r['input_id'] for r in main}):
        f=by_key[input_id,'following','feedback_rule'];d=by_key[input_id,'reference_idm','feedback_rule']
        common=min(len(f['decisions']),len(d['decisions']))
        target_adaptation.append(dict(input_id=input_id,different_request_prefix=any(x['action']!=y['action'] for x,y in zip(f['decisions'][:common],d['decisions'][:common]))))
    target_pairs=[]
    for input_id in sorted({r['input_id'] for r in main}):
        for policy in spec['policies']:
            f=by_key[input_id,'following',policy];d=by_key[input_id,'reference_idm',policy]
            paired=list(zip(f['trace'],d['trace']))
            target_pairs.append(dict(input_id=input_id,policy=policy,
                identical_dynamics=len(f['trace'])==len(d['trace']) and all(x['generator']==y['generator'] and x['target']==y['target'] for x,y in paired),
                request_difference_ticks=sum(x['target_control']['requested_mps2']!=y['target_control']['requested_mps2'] for x,y in paired),
                applied_difference_ticks=sum(x['target_control']['applied_mps2']!=y['target_control']['applied_mps2'] for x,y in paired),
                gate_difference_ticks=sum(x['target_control']['recognition']['native_gate']!=y['target_control']['recognition']['native_gate'] for x,y in paired)))
    negative_controls=[]
    for kind in ('reward','observation','applied_action','consumed_impact'):
        row=deepcopy(next(r for r in main if r['pair_contact']) if kind=='consumed_impact' else main[0])
        if kind=='reward':row['trace'][0]['agent_reward']+=.1
        elif kind=='observation':row['decisions'][0]['observation'][0]+=.1
        elif kind=='applied_action':row['trace'][0]['applied_action'][0]+=.1
        else:
            impact=next(t for t in row['trace'] if t['consumed_impacts'][0] is not None)
            impact['consumed_impacts'][0][0]+=.1
        try:audit_episode(row,windows[row['event_id']],cases[row['case_id']],divisors)
        except AssertionError:negative_controls.append(dict(mutation=kind,rejected=True))
        else:raise AssertionError('Audit accepted corrupted '+kind)
    a.output.mkdir(exist_ok=False,parents=True)
    summary=dict(audit_passed=True,registered_runs=320,main_runs=288,unique_events=8,
        physics_ticks=c['ticks'],audited_decisions=sum(r['decision_count'] for r in audit),
        sentinels_exact=16,repeats_exact=16,policies=summaries,
        feedback_vs_maximum=comparisons,target_feedback_adaptation=target_adaptation,
        target_pair_diagnostics=target_pairs,audit_negative_controls=negative_controls,
        feedback_comparison_counts=dict(applied_commands_differ=sum(r['different_applied_commands'] for r in comparisons),
            return_higher=sum(r['feedback_minus_maximum_return']>1e-12 for r in comparisons),
            return_lower=sum(r['feedback_minus_maximum_return']<-1e-12 for r in comparisons),
            return_equal=sum(abs(r['feedback_minus_maximum_return'])<=1e-12 for r in comparisons)),
        extra_legacy_regression_smoke=dict(episodes=1,physics_ticks=1,reason='Broad unittest discovery also selected existing hierarchical-env regression; retained separately, not a new agent sample'),
        total_executed_including_regression=dict(episodes=321,physics_ticks=c['ticks']+1),
        training_steps=0,inference='Outcome-informed eight-event diagnostic, no learned-agent superiority or population estimate')
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (a.output/'trajectory_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    with (a.output/'policy_summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('feedback_vs_maximum','target_feedback_adaptation','target_pair_diagnostics')},indent=2))


if __name__=='__main__':main()
