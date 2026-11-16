"""Stream and audit the registered B1 campaign without executing a simulator."""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from src.evaluation.adversarial_b1_audit import (audit_episode, score_episode, condition_key,
    summarize_method, scientific_gates, feedback_pair, equivalent_trajectory)


def read(path):
    with Path(path).open(encoding='utf-8-sig') as f:
        return json.load(f)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def lines(path):
    if not path.exists():
        return
    with path.open(encoding='utf-8-sig') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def load_episode(results, index):
    if 'trace' in index:
        return index
    path = (results/index['episode_path']).resolve()
    assert path.is_relative_to(results.resolve()), 'Episode path outside results'
    if index.get('episode_sha256'):
        assert sha(path) == index['episode_sha256']
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as f:
        row = json.load(f)
    for key in ('run_id','phase','group','stage','method','seed','checkpoint','input_id','H','target_h','plan_id','status'):
        if key in index:
            if key in row:
                assert row[key] == index[key], ('Index/trace metadata mismatch', key)
            else:
                row[key] = index[key]
    return row


def phase(row):
    return row.get('phase') or row.get('group')


def selected_plans(scores):
    """Per-seed, per-input mean over both hidden H; deterministic ID tie break."""
    cells = defaultdict(list)
    for row in scores:
        if phase(row) == 'search':
            cells[row['seed'],row['input_id'],row['plan_id']].append(row)
    selected = {}
    for seed in (42,123):
        for input_id in range(16):
            candidates = []
            for plan in range(21):
                pair = cells[seed,input_id,plan]
                if len(pair) != 2 or {r['H'] for r in pair} != {0.,.5}:
                    continue
                candidates.append((float(np.mean([r['total_return'] for r in pair])),plan))
            if len(candidates) == 21:
                selected[seed,input_id] = min(candidates,key=lambda t:(-t[0],t[1]))[1]
    return selected


def aggregate(scores, feedback):
    zero_rows = [r for r in scores if phase(r)=='search' and r['seed']==42 and r['plan_id']==0]
    zero = {condition_key(r):r for r in zero_rows}
    if len(zero)!=32:
        return dict(campaign_complete=False,scientific_gates_evaluated=False,
                    reason='Missing one or more first-seed matched Family-B zero conditions',
                    G1_functional_generator=False,G2_learned_advantage=False,G3_online_evidence=False), [], {}
    anchors=[r for r in scores if phase(r)=='search' and r['seed']==42 and r['plan_id'] in range(5)]
    integration_pass=(all(r['valid'] for r in zero_rows) and len(anchors)==160 and sum(r['valid'] for r in anchors)>=144)
    selections = selected_plans(scores)
    groups = defaultdict(list)
    for r in scores:
        if phase(r)=='evaluation' and r.get('stage') != 'confirmation' and r['method'] in ('reactive','initial_context'):
            checkpoint = 'initial' if r['checkpoint'] == 0 else 'final' if r['checkpoint']==16128 else str(r['checkpoint'])
            groups[r['method'],r['seed'],checkpoint].append(r)
        elif phase(r)=='search' and r['plan_id'] in range(5):
            groups['fixed_'+str(r['plan_id']),r['seed'],'fixed'].append(r)
    for seed in (42,123):
        chosen = [r for r in scores if phase(r)=='search' and r['seed']==seed and
                  selections.get((seed,r['input_id'])) == r['plan_id']]
        groups['open_loop',seed,'selected'] = chosen
    summaries = [dict(method=k[0],seed=k[1],checkpoint=k[2],**summarize_method(rows,zero))
                 for k,rows in sorted(groups.items())]
    comparators = {name:{} for name in ('initial_reactive','initial_context','selected_open_loop','strongest_fixed')}
    chosen_fixed = {}
    for seed in (42,123):
        fixed = []
        for plan in range(5):
            rows = groups['fixed_'+str(plan),seed,'fixed']
            if len(rows)==32:
                s = summarize_method(rows,zero)
                fixed.append((s['generated_conditions'],len(s['generated_case_ids']),s['mean_return'],plan))
        if len(fixed)==5:
            best = min(fixed,key=lambda x:(-x[0],-x[1],-x[2],x[3]))[3]
            chosen_fixed[seed]=best
            comparators['strongest_fixed'][seed]=groups['fixed_'+str(best),seed,'fixed']
        comparators['initial_reactive'][seed]=groups['reactive',seed,'initial']
        comparators['initial_context'][seed]=groups['initial_context',seed,'final']
        comparators['selected_open_loop'][seed]=groups['open_loop',seed,'selected']
    reactive = groups['reactive',42,'final']+groups['reactive',123,'final']
    gates = scientific_gates(reactive,comparators,zero,feedback)
    expected_evals = all(len(groups[method,seed,checkpoint])==32 for method in ('reactive','initial_context')
                         for seed in (42,123) for checkpoint in ('initial','final'))
    complete_search = sum(phase(r)=='search' for r in scores)==1344 and len(selections)==32
    summary = dict(campaign_complete=bool(expected_evals and complete_search),
                   integration_gate_passed=integration_pass,
                   scientific_gates_evaluated=True,**gates,
                   strongest_fixed_plan_by_seed=chosen_fixed,matched_zero_conditions=32,
                   selected_plans=[dict(seed=k[0],input_id=k[1],plan_id=v) for k,v in sorted(selections.items())],
                   interpretation='Outcome-informed same8 pilot; ticks, H variants and perturbations are not independent original cases; no held-out or population claim.')
    return summary,summaries,selections


def discovery_curves(scores):
    """Exploration discoveries versus realized cost, not deployed-policy scores."""
    zero={condition_key(r):r for r in scores if phase(r)=='search' and r['seed']==42 and r['plan_id']==0}
    curves=defaultdict(list);totals=Counter();seen=defaultdict(set);case_sets=defaultdict(set)
    for row in scores:
        if phase(row) not in ('training','search'):continue
        key=(phase(row),row['method'],row['seed']);totals[key]+=row['ticks']
        ck=condition_key(row)
        generated=bool(row['complete'] and row['valid'] and row['event'] and not row['initial_event'] and
                       ck in zero and not zero[ck]['event'])
        if generated:
            seen[key].add(ck);case_sets[key].add(row['case_id'])
        curves[key].append(dict(run_id=row['run_id'],cumulative_native_ticks=totals[key],
                               cumulative_discovered_conditions=len(seen[key]),
                               cumulative_discovered_original_cases=len(case_sets[key]),new_valid_risk=generated))
    common=min(totals.values()) if len(totals)==6 else None
    reports=[]
    for key,points in sorted(curves.items()):
        eligible=[r for r in points if common is not None and r['cumulative_native_ticks']<=common]
        reports.append(dict(phase=key[0],method=key[1],seed=key[2],realized_native_ticks=totals[key],
            first_valid_discovery_native_ticks=next((r['cumulative_native_ticks'] for r in points if r['new_valid_risk']),None),
            common_native_tick_prefix=common,common_prefix_completed_episode_discoveries=eligible[-1] if eligible else None,
            points=points))
    return dict(interpretation='Search/training exploratory episode discoveries only; these curves do not establish equal-cost final greedy-policy superiority.',
                common_realized_native_tick_prefix=common,runs=reports)


def audit_buffer(path, observed, start, final_observation=None):
    with np.load(path) as b:
        n=len(b['observations']); part=observed[start:start+n]
        assert len(part)==n
        np.testing.assert_array_equal(b['observations'][:,0,:],np.asarray([r['observation'] for r in part],dtype=np.float32))
        np.testing.assert_array_equal(b['actions'].reshape(-1),[r['action'] for r in part])
        rewards=np.asarray([r['reward'] for r in part],dtype=np.float32)
        np.testing.assert_allclose(b['rewards'].reshape(-1),rewards,rtol=0,atol=2e-7)
        np.testing.assert_array_equal(b['episode_starts'].reshape(-1),[r['start'] for r in part])
        assert bool(b['dones'][0])==part[-1]['done']
        expected=np.empty(n,dtype=np.float64); tail=float(b['last_values'][0])
        for i in reversed(range(n)):
            tail=float(rewards[i])+(0. if part[i]['done'] else tail); expected[i]=tail
        np.testing.assert_allclose(b['returns'].reshape(-1),expected,rtol=1e-6,atol=2e-6)
        if 'bootstrap_observations' in b:
            # VecEnv resets a terminal episode before publishing the bootstrap
            # observation; its terminal observation is not the next actor input.
            next_obs=(observed[start+n]['observation'] if start+n<len(observed) else final_observation)
            assert next_obs is not None
            np.testing.assert_array_equal(b['bootstrap_observations'][0],np.asarray(next_obs,dtype=np.float32))
    return n


def checkpoint_parameter_hash(path):
    """Hash archived tensor state directly, without loading a policy or env."""
    import torch
    with zipfile.ZipFile(path) as archive:
        with archive.open('policy.pth') as stream:
            tensors=torch.load(stream,map_location='cpu',weights_only=True)
    h=hashlib.sha256()
    for name,tensor in sorted(tensors.items()):
        assert torch.isfinite(tensor).all()
        h.update(name.encode());h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def training_transitions(row):
    snaps={s['tick']:s for s in row['observation_snapshots']}
    out=[]
    for i,d in enumerate(row['decisions']):
        ticks=[t for t in row['trace'] if t['decision_index']==i]
        assert ticks and 1<=len(ticks)<=5
        next_tick=d['tick']+len(ticks)
        out.append(dict(observation=d['observation'],action=d['action'],
                        reward=sum(t['agent_reward'] for t in ticks),start=i==0,
                        done=ticks[-1]['agent_outcome']!='ongoing',next_observation=snaps[next_tick]['observation']))
    return out


def analyze(results,data_root,output,protocol_path=None):
    results=Path(results);data_root=Path(data_root);output=Path(output)
    protocol=read(protocol_path or data_root/'protocol.json')
    for name,expected in protocol.get('data',{}).get('hashes',{}).items():
        assert sha(data_root/name)==expected, ('Frozen input hash mismatch',name)
    base_path=data_root/'base_protocol.json'
    if not base_path.exists(): base_path=data_root/'a1'/'data'/'base_protocol.json'
    base=read(base_path) if base_path.exists() else None
    case_path=data_root/'selected_cases.json'
    if not case_path.exists(): case_path=data_root/'a1'/'data'/'selected_cases.json'
    cases={c['case_id']:c for c in read(case_path)} if case_path.exists() else {}
    from src.evaluation.adversarial_b1_continuation import verify_receipt
    recovery=verify_receipt(results)
    indexes=list(lines(results/'episodes.jsonl'))
    partial_indexes=list(lines(results/'partials.jsonl'))
    if recovery:
        duplicate=recovery['duplicate_partial_index']
        assert sum(row==duplicate for row in partial_indexes)==1
        assert sum(row==recovery['completed_index'] for row in indexes)==1
        partial_indexes=[row for row in partial_indexes if row!=duplicate]
    indexes+=partial_indexes
    scores=[]; audits=[];refs={}; reactive={}; repeated_zero={};confirmations=[]
    physical_resets={}
    training=defaultdict(list);training_tails={};cost=defaultdict(Counter);deviations=[];action_stats=[]
    for index in indexes:
        row=load_episode(results,index);p=phase(row)
        cost[p]['starts']+=1;cost[p]['physics_ticks']+=len(row['trace']);cost[p]['actor_calls']+=len(row['decisions'])
        cost[p]['complete']+=row.get('status','complete')=='complete'
        if p=='sentinel':
            continue
        audit=audit_episode(row,protocol,base,cases.get(row['case_id']))
        if row['input_id'] in physical_resets:
            assert row['physical_initial_state']==physical_resets[row['input_id']], ('Physical reset changed across method/H',row['run_id'])
        else:physical_resets[row['input_id']]=row['physical_initial_state']
        audits.append(audit);score=score_episode(row);scores.append(score)
        if p=='search':
            refs[row['seed'],row['input_id'],float(row['target_h']),row['plan_id']]=index
            if row['plan_id']==0:
                repeated_zero[row['seed'],row['input_id'],float(row['target_h'])]=index
        if p=='evaluation' and row.get('stage')=='confirmation':
            confirmations.append(index)
        if p=='evaluation' and row['method']=='reactive' and row['checkpoint']==16128:
            reactive[row['seed'],row['input_id'],float(row['target_h'])]=index
        if p=='training':
            training[row['method'],row['seed']].extend(training_transitions(row))
            if row.get('status')=='incomplete':
                training_tails[row['method'],row['seed']]=row['current_observation']
        counts=Counter(d['action'] for d in row['decisions'])
        frequencies=np.asarray(list(counts.values()),dtype=float)/max(1,len(row['decisions']))
        action_stats.append(dict(run_id=row['run_id'],phase=p,method=row['method'],seed=row['seed'],
            action_counts=dict(counts),request_entropy_nats=float(-np.sum(frequencies*np.log(frequencies))),
            unique_commands=len({tuple(t['applied_action']) for t in row['trace']}),
            projection_interventions=sum(t['generator_control']['projection']['intervened'] for t in row['trace']),
            physical_ticks=len(row['trace'])))
    feedback=[]
    for seed in (42,123):
        for input_id in range(16):
            if all((seed,input_id,h) in reactive for h in (0.,.5)):
                feedback.append(feedback_pair(*[load_episode(results,reactive[seed,input_id,h]) for h in (0.,.5)]))
    summary,tables,selections=aggregate(scores,feedback)
    exact=[]
    for input_id in range(16):
        for h in (0.,.5):
            if all((seed,input_id,h) in repeated_zero for seed in (42,123)):
                a,b=[load_episode(results,repeated_zero[seed,input_id,h]) for seed in (42,123)]
                assert equivalent_trajectory(a,b), ('Repeated zero mismatch',input_id,h)
                exact.append(dict(kind='zero_repetition',input_id=input_id,H=h,passed=True))
    for index in confirmations:
        row=load_episode(results,index);key=(row['seed'],row['input_id'])
        assert selections[key]==row['plan_id']
        old=load_episode(results,refs[*key,float(row['target_h']),row['plan_id']])
        assert equivalent_trajectory(row,old),('Confirmation mismatch',row['run_id'])
        exact.append(dict(kind='selected_confirmation',seed=key[0],input_id=key[1],H=row['target_h'],passed=True))
    # Shared live ticks only; never pad a terminated nominal or policy tail.
    for index in indexes:
        if phase(index)!='evaluation' or index.get('stage')=='confirmation':continue
        row=load_episode(results,index)
        nominal_index=repeated_zero.get((42,row['input_id'],float(row['target_h'])))
        if nominal_index is None:continue
        nominal=load_episode(results,nominal_index)
        pair=list(zip(row['trace'],nominal['trace']))
        speed=[a['generator_scalar_speed']-b['generator_scalar_speed'] for a,b in pair]
        displacement=[a['generator']['x']-b['generator']['x'] for a,b in pair]
        deviations.append(dict(run_id=row['run_id'],method=row['method'],seed=row['seed'],checkpoint=row['checkpoint'],
            input_id=row['input_id'],H=row['target_h'],shared_ticks=len(pair),policy_ticks=len(row['trace']),zero_ticks=len(nominal['trace']),
            policy_unpaired_tail=len(row['trace'])-len(pair),zero_unpaired_tail=len(nominal['trace'])-len(pair),
            max_absolute_scalar_speed_deviation=max(map(abs,speed),default=None),
            final_shared_displacement_deviation=displacement[-1] if displacement else None,
            max_absolute_displacement_deviation=max(map(abs,displacement),default=None)))
    buffer_reports=[]
    for (method,seed),observed in sorted(training.items()):
        directory=results/'learning'/f'{method}_{seed}'/'buffers'
        files=sorted(directory.glob('*.npz'))
        if not files:
            directory=results/'buffers'/f'{method}_{seed}';files=sorted(directory.glob('*.npz'))
        count=0
        for path in files:
            count+=audit_buffer(path,observed,count,training_tails.get((method,seed)))
        buffer_reports.append(dict(method=method,seed=seed,transitions=len(observed),audited_transitions=count,
                                   buffers=len(files),complete=len(observed)==16128 and count==16128 and len(files)==63))
    complete_path=results/'complete.json'
    completion=read(complete_path) if complete_path.exists() else {}
    initializations=[]
    for seed in (42,123):
        manifests=[]
        for method in ('reactive','initial_context'):
            path=results/'learning'/f'{method}_{seed}'/'summary.json'
            if path.exists():manifests.append(read(path))
        if len(manifests)==2:
            # The runner archives separately instantiated actor/critic state
            # dictionaries; equal hashes establish matching initialization.
            hashes=[m.get('initial_parameter_sha256') for m in manifests]
            matched=hashes[0] is not None and hashes[0]==hashes[1]
            assert matched, ('Mismatched policy initialization',seed,hashes)
            checkpoints=[]
            for method,manifest in zip(('reactive','initial_context'),manifests):
                run_dir=results/'learning'/f'{method}_{seed}'
                for checkpoint in ('initial','final'):
                    observed_hash=checkpoint_parameter_hash(run_dir/f'{checkpoint}_model.zip')
                    assert observed_hash==manifest[f'{checkpoint}_parameter_sha256']
                    checkpoints.append(dict(method=method,checkpoint=checkpoint,parameter_sha256=observed_hash))
                assert manifest['actor_transitions']==16128 and manifest['buffer_batches']==63 and manifest['optimizer_update_calls']==63
                assert manifest['initial_parameter_sha256']!=manifest['final_parameter_sha256']
                sampler=read(run_dir/'sampler.json')
                resets=[r for r in lines(results/'resets.jsonl') if phase(r)=='training' and r['method']==method and r['seed']==seed]
                assert sampler['draws']==[r['condition_id'] for r in resets]
                for start in range(0,len(sampler['draws'])-31,32):
                    assert set(sampler['draws'][start:start+32])==set(range(32))
                updates=[r for r in lines(results/'updates.jsonl') if r['method']==method and r['seed']==seed]
                assert [r['actor_transitions'] for r in updates]==list(range(256,16129,256))
                assert all(np.isfinite(v) for u in updates for v in u['metrics'].values())
            initializations.append(dict(seed=seed,matched=True,parameter_hash=hashes[0],checkpoints=checkpoints))
    costs={k:dict(v) for k,v in cost.items()}
    measured=dict(physics_ticks=sum(c['physics_ticks'] for c in costs.values()),
                  actor_calls=sum(c['actor_calls'] for c in costs.values()),
                  completed=sum(c['complete'] for c in costs.values()),episode_starts=sum(c['starts'] for c in costs.values()))
    budget=protocol['budget']
    assert measured['physics_ticks']<=budget['total_native_tick_cap']
    assert measured['actor_calls']<=budget['total_actor_call_or_slot_cap']
    assert measured['episode_starts']<=budget['maximum_episode_starts']
    ledger=read(results/'ledger.json') if (results/'ledger.json').exists() else None
    cost_audit=[]
    if ledger:
        for name,measured_phase in costs.items():
            entry=ledger['phases'][name]
            assert entry['measured_ticks']==measured_phase['physics_ticks']
            assert entry['measured_actors']==measured_phase['actor_calls']
            assert entry['starts']==measured_phase['starts']
            reconciled=int(bool(recovery) and name=='search')
            assert entry['completed']+reconciled==measured_phase['complete']
            assert entry['uncertain_failed_calls']==reconciled
            assert entry['ticks']==entry['measured_ticks'] and entry['actors']==entry['measured_actors']
            cost_audit.append(dict(phase=name,passed=True))
        assert ledger['attempt_counts']['active']==0 and ledger['attempt_counts']['failed']==int(bool(recovery))
    integrity_audit=False
    if completion:
        assert completion['execution_complete'] and completion['integrity_passed']
        assert completion['protocol_sha256']==sha(protocol_path or data_root/'protocol.json')
        for name,expected in completion['hashes'].items():
            assert sha(results/name)==expected, ('Result hash mismatch',name)
        integrity_audit=True
    campaign_complete=(summary['campaign_complete'] and len(confirmations)==64 and len(exact)==96 and
                       len(buffer_reports)==4 and all(b['complete'] for b in buffer_reports) and len(initializations)==2 and
                       integrity_audit and len(cost_audit)==4 and summary.get('integration_gate_passed',False))
    summary.update(campaign_complete=campaign_complete,trace_audit_passed=True,
                   saved_episode_reconciliation=recovery,
                   completed=sum(r['complete'] for r in scores),partial=sum(not r['complete'] for r in scores),
                   measured_cost=measured,original_cases=8,recordings=6,
                   runtime_reporting_complete=bool(completion.get('resources',{}).get('memory_reporting_complete',False)))
    if not campaign_complete:
        for key in ('G1_functional_generator','G2_learned_advantage','G3_online_evidence'):summary[key]=False
        summary['incomplete_campaign_forces_gates_false']=True
    output.mkdir(parents=True,exist_ok=False)
    write(output/'summary.json',summary);write(output/'condition_scores.json',scores)
    write(output/'method_summary.json',tables);write(output/'feedback_pairs.json',feedback)
    write(output/'audit.json',dict(passed=True,details=audits,exact_replays=exact,buffers=buffer_reports,initializations=initializations,
                                   cost_reconciliation=cost_audit,result_file_hashes_verified=integrity_audit,
                                   sentinels='Root runner independently audits old-contract sentinels; B auditor excludes them.'))
    write(output/'cost_report.json',dict(by_phase=costs,measured=measured,completion_report=completion))
    write(output/'deviations.json',deviations);write(output/'actions.json',action_stats)
    write(output/'discovery_cost_curves.json',discovery_curves(scores))
    if scores:
        with (output/'condition_scores.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(scores[0]));writer.writeheader();writer.writerows(scores)
    print(json.dumps(summary,indent=2))
    return summary


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--results','--result',dest='results',type=Path,required=True)
    p.add_argument('--data-root','--data',dest='data_root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--protocol',type=Path)
    a=p.parse_args();analyze(a.results,a.data_root,a.output,a.protocol)


if __name__=='__main__':main()
