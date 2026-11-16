"""Independent saved-state audits for the frozen Family B1 campaign.

No environment, controller, learner or production observation implementation is
imported. Cached rows are evidence; they are never sent through a simulator.
"""
from collections import Counter, defaultdict
import math

import numpy as np

from scripts.audit_consumed_impact import swept, minimum_translation_candidates


DT = .1
REQUESTS = (0., -3., 3.)
DIVISORS = np.array([100, 4, 40, 10, 40, 10, 1, 1, 1, 1, 3, 5, 1,
                     12, 4, .3, 1, 1, 1, 40, 40, 3, 5, 2, 5, 2])


def close(a, b, tol=1e-8):
    if a is None or b is None:
        assert a is b, (a, b)
    else:
        assert math.isfinite(a) and math.isfinite(b) and abs(a-b) <= tol, (a, b, tol)


def native(command, scale):
    return float(-np.float32(scale) + (np.float32(command)+np.float32(1)) *
                 np.float32(2*scale)/np.float32(2))


def risk(ttc):
    if ttc is None:
        return 0.
    assert math.isfinite(ttc) and ttc >= 0
    return max(0., min(1., 1-ttc/3))


def onroad(s):
    radius = (s['length']*abs(math.sin(s['heading'])) +
              s['width']*abs(math.cos(s['heading'])))/2
    return s['y']-radius >= -2-1e-9 and s['y']+radius <= 10+1e-9


def reserve_acceleration_cap(distance):
    """Analytic piecewise inversion of discrete 1.99 jerk stopping reserve."""
    budget=max(0.,distance-1e-6);delta=.199;ceiling=3.-1e-6
    for n in range(1,17):
        upper=min(ceiling,n*delta)
        consumed=.1*(n*upper-delta*n*(n-1)/2)
        if consumed>=budget:
            return min(ceiling,max(0.,(budget/.1+delta*n*(n-1)/2)/n))
    return ceiling


def lateral_reference(case, seconds):
    """Independently evaluate frozen observed-midpoint continue polynomial."""
    lo, hi = 0., .5
    for _ in range(60):
        u = (lo+hi)/2
        if 10*u**3-15*u**4+6*u**5 < .1:
            lo = u
        else:
            hi = u
    tenth = (lo+hi)/2
    duration = case['event']['metrics']['duration_10_90']
    total = duration/(1-2*tenth)
    phase = min(1., max(0., tenth+max(0., seconds)/total))
    value = .4 if seconds <= 0 else 4*phase**3*(10+phase*(-15+6*phase))
    start, middle, end = case['event']['times']
    length = end-start
    if 0 < seconds < length:
        fraction = (middle-start)/length
        p = tenth+(1-2*tenth)*fraction
        scale = (2-4*(10*p**3-15*p**4+6*p**5))/(fraction**3*(1-fraction)**3)
        u = seconds/length
        value += scale*u**3*(1-u)**3
    return value


def frame(row, tick, maximum):
    initial = row['physical_initial_state']
    g, t = initial if tick == 0 else [row['trace'][tick-1][k] for k in ('generator', 'target')]
    a = [0., 0.] if tick == 0 else row['trace'][tick-1]['acceleration']
    steering = 0. if tick == 0 else native(row['trace'][tick-1]['applied_action'][1], .3)
    lateral = initial[0]['y'] if tick == 0 else row['trace'][tick-1]['generator_control']['y_reference']
    previous_request = 0. if tick == 0 else REQUESTS[row['trace'][tick-1]['agent_action']]
    ttc = swept(g, t)['ttc']
    return [g['x']-t['x'], g['y']-t['y'], g['vx'], g['vy'], t['vx'], t['vy'],
            math.sin(g['heading']), math.cos(g['heading']), math.sin(t['heading']), math.cos(t['heading']),
            *a, float(tick > 0), tick*.1, lateral, steering, risk(ttc), float(ttc is not None),
            maximum, math.hypot(initial[0]['vx'], initial[0]['vy']), math.hypot(g['vx'], g['vy']),
            previous_request, g['length'], g['width'], t['length'], t['width']]


def reconstruct_snapshots(row, maxima):
    """Recreate each stored frame at its own historical time before stacking."""
    snapshots = row.get('observation_snapshots', row['decisions'])
    by_tick = {}
    for snap in snapshots:
        tick = snap['tick']
        assert tick not in by_tick, ('duplicate observation snapshot', tick)
        by_tick[tick] = snap
    for d in row['decisions']:
        assert d['tick'] in by_tick
    history = []
    initial = frame(row, 0, maxima[0])
    total_clips = np.zeros(26, dtype=np.int64)
    total_raw_clips = np.zeros(26, dtype=np.int64)
    for tick in sorted(by_tick):
        snap = by_tick[tick]
        raw = frame(row, tick, maxima[tick])
        np.testing.assert_allclose(snap['raw_frame'], raw, rtol=0, atol=1e-8)
        effective = raw[:]
        if row.get('observation_mode', row.get('method')) == 'initial_context':
            effective = initial[:]
            effective[13], effective[21] = raw[13], raw[21]
        np.testing.assert_allclose(snap['effective_frame'], effective, rtol=0, atol=1e-8)
        normalized = np.asarray(effective)/DIVISORS
        raw_clips = (np.abs(np.asarray(raw)/DIVISORS) > 1).astype(int)
        if 'raw_frame_clip_counts' in snap:
            assert snap['raw_frame_clip_counts'] == raw_clips.tolist()
        total_raw_clips += raw_clips
        clipped = np.clip(normalized, -1, 1).astype(np.float32)
        clips = (np.abs(normalized) > 1).astype(int)
        if 'frame_clip_counts' in snap:
            assert snap['frame_clip_counts'] == clips.tolist()
        total_clips += clips
        history.append((tick, clipped))
        available = history[-4:]
        masks = [0]*(4-len(available))+[1]*len(available)
        ticks = [None]*(4-len(available))+[t for t, _ in available]
        values = [np.zeros(26, dtype=np.float32)]*(4-len(available))+[v for _, v in available]
        expected = np.concatenate(values+[np.asarray(masks, dtype=np.float32)])
        assert len(expected) == 108
        np.testing.assert_array_equal(np.asarray(snap['observation'], dtype=np.float32), expected)
        assert snap['frame_masks'] == masks
        assert snap['frame_ticks'] == ticks
    for d in row['decisions']:
        snap = by_tick[d['tick']]
        for key in ('observation', 'raw_frame', 'effective_frame', 'frame_ticks', 'frame_masks'):
            assert d[key] == snap[key], (key, d['tick'])
        assert d['action'] in range(3)
        if 'requested_acceleration_mps2' in d:
            close(d['requested_acceleration_mps2'], REQUESTS[d['action']], 0)
    if 'observation_clip_counts' in row:
        assert row['observation_clip_counts'] == total_clips.tolist()
    if 'raw_observation_clip_counts' in row:
        assert row['raw_observation_clip_counts'] == total_raw_clips.tolist()
    return len(by_tick)


def audit_contacts(row):
    """Recompute confirmation/consumption; pending-only collisions get no credit."""
    trace = row['trace']
    contacts = row.get('confirmed_contacts', [])
    for event in contacts:
        assert set(event['actors']) == {0, 1}
        step = event['step']
        assert 0 <= step < len(trace)
        states = [trace[step]['generator'], trace[step]['target']]
        expected_states = states if event['actors'] == [0, 1] else list(reversed(states))
        assert event['states'] == expected_states
        geom = swept(*event['states'])
        if event['kind'] == 'overlap':
            assert geom['current_overlap']
        else:
            assert event['kind'] == 'scheduled_impact'
            assert step < len(trace)-1 and geom['scheduled']
            assert not geom['current_overlap'] and geom['ttc'] is not None and 0 < geom['ttc'] <= .1+1e-10
            subsequent = trace[step+1]
            transitions = minimum_translation_candidates(*event['states'])
            # Both actors must consume the SAME independently reconstructed
            # minimum-axis candidate; near-parallel SAT ties are not unique.
            assert any(all(impact is not None and all(
                math.isfinite(got) and abs(got-change*(.5 if pos==0 else -.5))<=1e-8
                for got,change in zip(impact,transition))
                for pos,actor in enumerate(event['actors'])
                for impact in [subsequent['consumed_impacts'][actor]]) for transition in transitions)
            for pos, actor in enumerate(event['actors']):
                impact = subsequent['consumed_impacts'][actor]
                assert impact is not None
    assert bool(contacts) == bool(row.get('pair_contact', False))
    for k, step in enumerate(trace):
        for actor, impact in enumerate(step['consumed_impacts']):
            if impact is not None:
                assert any(e['kind'] == 'scheduled_impact' and e['step'] == k-1 and
                           actor in e['actors'] for e in contacts)
    if contacts:
        assert set(row.get('crashed_ids', [])) == {0, 1}
        assert row.get('termination_reason') == 'collision'
    return len(contacts)


def score_episode(row):
    states = [row['physical_initial_state']]+[[t['generator'], t['target']] for t in row['trace']]
    ttcs = [swept(*s)['ttc'] for s in states]
    valid = all(all(t['maneuver_checks'].values()) for t in row['trace'])
    complete = row.get('status', 'complete') == 'complete'
    contact = bool(row.get('pair_contact', False)) and valid and complete
    initial_event = ttcs[0] is not None and ttcs[0] < 1.5
    event = bool(complete and valid and (contact or any(t is not None and t < 1.5 for t in ttcs[1:])))
    q = max(map(risk, ttcs))
    result = {k: row.get(k) for k in ('run_id', 'phase', 'group', 'stage', 'method', 'seed',
               'checkpoint', 'input_id', 'case_id', 'event_id', 'plan_id', 'observation_mode')}
    result.update(H=float(row.get('H', row.get('target_h', 0))), valid=valid,
                  complete=complete, contact=contact, event=event, initial_event=initial_event,
                  new_event=event and not initial_event, Q=q, initial_Q=risk(ttcs[0]),
                  total_return=row.get('agent_return', row.get('accumulated_return')),
                  ticks=len(row['trace']), decisions=len(row['decisions']))
    return result


def audit_episode(row, protocol=None, base=None, case=None):
    trace = row['trace']; initial = row['physical_initial_state']
    assert len(initial) == 2 and all(onroad(s) for s in initial)
    assert swept(*initial)['ttc'] != 0
    assert 15-1e-5 <= math.hypot(initial[0]['vx'], initial[0]['vy']) <= 40+1e-5
    assert 15 <= initial[1]['vx'] <= 35
    if protocol is not None:
        inputs={r['input_id']:r for r in protocol['inputs']}
        item=inputs[row['input_id']]
        assert row['case_id']==item['case_id'] and row['event_id']==item['event_id']
        close(initial[1]['vx'],item['target_initial_and_cruise_mps'],1e-9)
        close(initial[0]['x']-initial[1]['x']-5,item['initial_gap_m'],1e-9)
    assert len(trace) <= 120
    complete = row.get('status', 'complete') == 'complete'
    decisions = {d['tick']: d for d in row['decisions']}
    assert sorted(decisions) == list(range(0, len(trace), 5))
    maximum = risk(swept(*initial)['ttc']); maxima = [maximum]
    prior = initial; previous_a = [0., 0.]; previous_steer = 0.; ys = [initial[0]['y']]
    valid = True; total = 0.
    contact_count = audit_contacts(row)
    for tick, t in enumerate(trace):
        assert t['tick'] == tick
        d = decisions[5*(tick//5)]
        assert t['agent_action'] == d['action'] and t['decision_index'] == tick//5
        gc = t['generator_control']; tc = t['target_control']
        close(gc['requested_mps2'], REQUESTS[d['action']], 0)
        close(gc['projection']['requested_mps2'], REQUESTS[d['action']], 0)
        close(gc['native_acceleration'], native(t['applied_action'][0], 5), 0)
        close(gc['native_steering'], native(t['applied_action'][1], .3), 0)
        close(gc['projection']['applied_mps2'], gc['native_acceleration'], 0)
        assert gc['projection']['lower_mps2'] <= gc['native_acceleration'] <= gc['projection']['upper_mps2']
        assert gc['projection']['intervened'] == (abs(gc['native_acceleration']-REQUESTS[d['action']]) > 1e-5)
        previous_speed=math.hypot(prior[0]['vx'],prior[0]['vy'])
        lower=max(-3+1e-6,previous_a[0]-.2,-reserve_acceleration_cap(previous_speed-15))
        upper=min(3-1e-6,previous_a[0]+.2,reserve_acceleration_cap(40-previous_speed))
        close(gc['projection']['lower_mps2'],lower,1e-9)
        close(gc['projection']['upper_mps2'],upper,1e-9)
        projected=min(upper,max(lower,REQUESTS[d['action']]))
        assert abs(gc['native_acceleration']-projected)<=2e-5
        states = [t['generator'], t['target']]; reconstructed = []
        for actor, (old, state, impact) in enumerate(zip(prior, states, t['consumed_impacts'])):
            unimpacted = state.copy()
            if impact is not None:
                unimpacted['x'] -= impact[0]; unimpacted['y'] -= impact[1]
            assert unimpacted == t['unimpacted_generator' if actor == 0 else 'unimpacted_target']
            reconstructed.append(unimpacted)
            speed = math.hypot(old['vx'], old['vy'])
            next_speed = math.hypot(state['vx'], state['vy'])
            acceleration = (next_speed-speed)/DT
            close(acceleration, t['acceleration'][actor], 1e-9)
            applied = gc['native_acceleration'] if actor == 0 else tc['applied_mps2']
            close(acceleration, applied, 1e-5)
            steering = gc['native_steering'] if actor == 0 else tc['steering']
            beta = math.atan(.5*math.tan(steering))
            close(unimpacted['x'], old['x']+speed*math.cos(old['heading']+beta)*DT)
            close(unimpacted['y'], old['y']+speed*math.sin(old['heading']+beta)*DT)
            close(state['heading'], old['heading']+speed*math.sin(beta)/(state['length']/2)*DT)
            close(state['vx'], next_speed*math.cos(state['heading']), 1e-8)
            close(state['vy'], next_speed*math.sin(state['heading']), 1e-8)
        g, target = reconstructed
        a, ta = t['acceleration']; steering = gc['native_steering']
        ys.append(g['y'])
        ay = (ys[-1]-2*ys[-2]+ys[-3])/.01 if len(ys) >= 3 else 0.
        jy = (ys[-1]-3*ys[-2]+3*ys[-3]-ys[-4])/.001 if len(ys) >= 4 else 0.
        close(ay, t['lateral_acceleration']); close(jy, t['lateral_jerk'])
        if case is not None:
            close(gc['y_reference'], lateral_reference(case, (tick+1)*DT), 1e-9)
        checks = dict(acceleration=abs(a)<=3+1e-5, jerk=abs((a-previous_a[0])/DT)<=2+1e-5,
                      speed=15-1e-5<=math.hypot(g['vx'],g['vy'])<=40+1e-5,
                      realization=abs(a-gc['native_acceleration'])<=1e-5,
                      steering=abs(steering)<=.3+1e-5, steering_slew=abs((steering-previous_steer)/DT)<=.15+1e-5,
                      path=abs(g['y']-gc['y_reference'])<=1e-5, lateral_acceleration=abs(ay)<=2+1e-5,
                      lateral_jerk=abs(jy)<=4+1e-5, road=onroad(g), forward=g['vx']>=-1e-5)
        target_checks = dict(acceleration=abs(ta)<=5+1e-5, jerk=abs((ta-previous_a[1])/DT)<=2.5+1e-5,
                             realization=abs(ta-tc['applied_mps2'])<=1e-5, road=onroad(target), forward=target['vx']>=-1e-5)
        assert checks == t['maneuver_checks']
        assert target_checks == t['target_checks'] and all(target_checks.values())
        valid &= all(checks.values())
        ttc = swept(*states)['ttc']; close(ttc, t['pair_ttc'], 1e-10)
        contact_now = any(e['kind'] == 'overlap' and e['step'] <= tick or
                          e['kind'] == 'scheduled_impact' and e['step'] < tick
                          for e in row.get('confirmed_contacts', []))
        expected_outcome = ('maneuver_invalid' if not valid else 'intended_contact' if contact_now
                            else 'horizon' if tick == 119 else 'ongoing')
        assert t['agent_outcome'] == expected_outcome
        if expected_outcome == 'maneuver_invalid':
            reward = -1-maximum
        elif expected_outcome == 'intended_contact':
            reward = 2-maximum
        else:
            updated = max(maximum, risk(ttc)); reward = updated-maximum; maximum = updated
        close(reward, t['agent_reward'], 1e-12); total += reward; maxima.append(maximum)
        if expected_outcome != 'ongoing':
            assert tick == len(trace)-1 and complete
        prior = states; previous_a = t['acceleration']; previous_steer = steering
    close(total, row.get('agent_return', row.get('accumulated_return')), 1e-10)
    if complete:
        assert trace and trace[-1]['agent_outcome'] != 'ongoing'
        assert row['agent_outcome'] == trace[-1]['agent_outcome']
        assert row['maneuver_valid_precontact'] == valid
        final = (-1 if not valid else 2 if contact_count else maximum)
        close(total, final-maxima[0], 1e-10)
        finite = [t['pair_ttc'] for t in trace if t['pair_ttc'] is not None]
        close(row['min_pair_ttc'], min(finite, default=None), 1e-10)
    else:
        assert not trace or trace[-1]['agent_outcome'] == 'ongoing'
    snapshots = reconstruct_snapshots(row, maxima)
    score = score_episode(row)
    if 'maximum_q' in row:
        close(score['Q'], row['maximum_q'], 1e-10)
    if complete:
        assert row['valid_risk'] == score['event']
        assert row['valid_pair_contact'] == score['contact']
        assert row['initial_risk_event'] == score['initial_event']
    return dict(score, passed=True, snapshots=snapshots, contacts_audited=contact_count,
                lateral_reference_independently_audited=case is not None)


def condition_key(row):
    return int(row['input_id']), float(row.get('H', row.get('target_h', 0.)))


def generated_map(rows, zero):
    result = {}
    for row in rows:
        key = condition_key(row)
        assert key not in result, ('duplicate condition', key)
        assert key in zero
        result[key] = bool(row['complete'] and row['valid'] and row['event'] and
                           not row['initial_event'] and not zero[key]['event'])
    return result


def summarize_method(rows, zero):
    generated = generated_map(rows, zero)
    return dict(conditions=len(rows), valid=sum(r['valid'] for r in rows),
                contact=sum(r['contact'] for r in rows), noncontact_risk=sum(r['event'] and not r['contact'] for r in rows),
                generated_conditions=sum(generated.values()),
                generated_case_ids=sorted({r['case_id'] for r in rows if generated[condition_key(r)]}),
                generated_recordings=sorted({r['event_id'].split(':')[0] for r in rows if generated[condition_key(r)]}),
                mean_return=float(np.mean([r['total_return'] for r in rows])) if rows else None,
                mean_Q=float(np.mean([r['Q'] for r in rows])) if rows else None,
                mean_delta_Q_vs_zero=float(np.mean([r['Q']-zero[condition_key(r)]['Q'] for r in rows])) if rows else None,
                physical_ticks=sum(r['ticks'] for r in rows), actor_decisions=sum(r['decisions'] for r in rows),
                per_H={str(h):dict(conditions=sum(r['H']==h for r in rows),
                    generated=sum(generated[condition_key(r)] for r in rows if r['H']==h),
                    contact=sum(r['contact'] for r in rows if r['H']==h)) for h in sorted({r['H'] for r in rows})})


def paired_advantage(reactive, comparator, zero):
    a = generated_map(reactive, zero); b = generated_map(comparator, zero)
    assert set(a) == set(b)
    new = sorted(k for k in a if a[k] and not b[k])
    lost = sorted(k for k in a if b[k] and not a[k])
    return dict(net=len(new)-len(lost), new_conditions=new, lost_conditions=lost,
                reactive_generated=sum(a.values()), comparator_generated=sum(b.values()))


def target_difference(a, b):
    return (any(abs(a[k]-b[k]) > 1e-6 for k in ('x','y','vx','vy','acceleration')) or
            abs(a['heading']-b['heading']) > 1e-8)


def target_at(row, tick):
    target = dict(row['physical_initial_state'][1] if tick == 0 else row['trace'][tick-1]['target'])
    target['acceleration'] = 0. if tick == 0 else row['trace'][tick-1]['acceleration'][1]
    return target


def positive_endpoint(row):
    """Earliest full-episode positive maximum or valid event, no local bumps."""
    if not row['maneuver_valid_precontact']:
        return dict(event_time=None, peak_time=None, Q=max([risk(swept(*row['physical_initial_state'])['ttc'])]+
                    [risk(t['pair_ttc']) for t in row['trace']]))
    qs = [risk(swept(*row['physical_initial_state'])['ttc'])]+[risk(t['pair_ttc']) for t in row['trace']]
    maximum = max(qs)
    event = next(((i+1)*DT for i,t in enumerate(row['trace']) if
                  t['agent_outcome']=='intended_contact' or t['pair_ttc'] is not None and t['pair_ttc']<1.5), None)
    peak = next((i*DT for i,q in enumerate(qs) if q == maximum), None) if maximum > 0 else None
    return dict(event_time=event, peak_time=peak, Q=maximum)


def feedback_pair(a, b):
    assert a['input_id'] == b['input_id'] and a['seed'] == b['seed']
    assert a['physical_initial_state'] == b['physical_initial_state']
    da, db = a['decisions'], b['decisions']
    assert da and db and da[0]['observation'] == db[0]['observation']
    shared = list(zip(da, db)); first_obs = first_target = first_request = None
    causality = True
    for x, y in shared:
        assert x['tick'] == y['tick']
        tick = x['tick']
        different = not np.array_equal(np.asarray(x['observation'], dtype=np.float32),
                                       np.asarray(y['observation'], dtype=np.float32))
        if different and first_obs is None:
            first_obs = tick*DT
        if target_difference(target_at(a, tick), target_at(b, tick)) and first_target is None:
            first_target = tick*DT
        if x['action'] != y['action']:
            if first_request is None:
                first_request = tick*DT
            if not different:
                causality = False
    first_applied = next((k*DT for k,(x,y) in enumerate(zip(a['trace'],b['trace'])) if
                          abs(x['generator_control']['native_acceleration']-
                              y['generator_control']['native_acceleration']) > 1e-5), None)
    endpoints = [positive_endpoint(r) for r in (a,b)]
    later = first_applied is not None and any(t is not None and t > first_applied for
                 endpoint in endpoints for key,t in endpoint.items() if key in ('event_time','peak_time'))
    supports = bool(causality and first_target is not None and first_request is not None and
                    first_obs is not None and first_target <= first_request and first_obs <= first_request and
                    first_applied is not None and first_applied >= first_request and later)
    return dict(input_id=a['input_id'], case_id=a['case_id'], seed=a['seed'],
                initial_observations_identical=True, causality_passed=causality,
                first_actor_observation_difference_s=first_obs, first_observed_meaningful_target_difference_s=first_target,
                first_requested_action_difference_s=first_request, first_applied_acceleration_difference_s=first_applied,
                endpoints=endpoints, shared_decisions=len(shared), shared_ticks=min(len(a['trace']),len(b['trace'])),
                supports_G3=supports)


def scientific_gates(reactive_final, comparators, zero, pairs, expected_conditions=32):
    """Fail incomplete campaigns; all seed-wise thresholds are conjunctive."""
    seed_reports = []
    for seed in (42,123):
        rr = [r for r in reactive_final if r['seed']==seed]
        summary = summarize_method(rr,zero)
        coverage = len(rr)==expected_conditions and len({condition_key(r) for r in rr})==expected_conditions
        g1 = coverage and summary['valid']>=31 and len(summary['generated_case_ids'])>=2
        comparisons = {}
        for name, by_seed in comparators.items():
            bb = by_seed.get(seed, [])
            if len(bb)==expected_conditions and {condition_key(r) for r in bb}=={condition_key(r) for r in rr}:
                comparisons[name] = paired_advantage(rr,bb,zero)
            else:
                comparisons[name] = dict(net=None, incomplete=True)
        g2 = g1 and len(comparisons)==4 and all(r.get('net') is not None and r['net']>=2 for r in comparisons.values())
        pp = [p for p in pairs if p['seed']==seed]
        mechanism_cases = sorted({p['case_id'] for p in pp if p['supports_G3']})
        causal = len(pp)==16 and all(p['causality_passed'] for p in pp)
        g3 = g2 and causal and len(mechanism_cases)>=2
        seed_reports.append(dict(seed=seed,summary=summary,comparisons=comparisons,
                                 G1=bool(g1),G2=bool(g2),G3=bool(g3),
                                 mechanism_case_ids=mechanism_cases,causality_audit_passed=causal))
    return dict(G1_functional_generator=all(r['G1'] for r in seed_reports),
                G2_learned_advantage=all(r['G2'] for r in seed_reports),
                G3_online_evidence=all(r['G3'] for r in seed_reports), per_seed=seed_reports)


def equivalent_trajectory(a,b):
    fields = ('physical_initial_state','trace','agent_return','agent_outcome','maneuver_valid_precontact',
              'pair_contact','valid_pair_contact','valid_risk','initial_risk_event','min_pair_ttc')
    return all(a[k]==b[k] for k in fields) and [d['action'] for d in a['decisions']]==[d['action'] for d in b['decisions']]
