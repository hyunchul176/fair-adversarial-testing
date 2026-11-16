"""Frozen, event-level cut-in descriptors; no simulation or policy inference."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def lane_centers(meta):
    upper = [float(x) for x in meta['upperLaneMarkings'].split(';')]
    lower = [float(x) for x in meta['lowerLaneMarkings'].split(';')]
    lanes = {}
    for marks, offset, direction in ((upper, 2, 1), (lower, len(upper)+2, 2)):
        for i in range(len(marks)-1):
            lanes[offset+i] = ((marks[i]+marks[i+1])/2, direction)
    return lanes


def crossings(times, progress, value):
    indexes = np.flatnonzero((progress[:-1] < value) & (progress[1:] >= value))
    return [float(times[i]+(times[i+1]-times[i])*(value-progress[i]) /
                  (progress[i+1]-progress[i])) for i in indexes]


def anchors(times, progress, reference, tolerance):
    middle = crossings(times, progress, .5)
    if not middle:
        return None
    middle = min(middle, key=lambda x: abs(x-reference))
    if abs(middle-reference) > tolerance:
        return None
    starts = [x for x in crossings(times, progress, .1) if x < middle]
    ends = [x for x in crossings(times, progress, .9) if x > middle]
    return [max(starts), middle, min(ends)] if starts and ends else None


def interp(track, times, key):
    return np.interp(times, track['time'], track[key])


def descriptors(generator, target, event_times, sign, sample_s):
    start, middle, end = event_times
    result = {'duration_10_90': float(end-start)}
    for name, t in (('p10', start), ('p50', middle)):
        gv, tv = float(interp(generator, t, 'vx'))*sign, float(interp(target, t, 'vx'))*sign
        gap = sign*(float(interp(generator, t, 'cx'))-float(interp(target, t, 'cx'))) - (
            float(interp(generator, t, 'length'))+float(interp(target, t, 'length')))/2
        result[name] = dict(generator_speed=gv, target_speed=tv, gap=gap,
                            headway=gap/tv if tv>0 else None, closing_speed=tv-gv)
    grid = np.arange(start, end+1e-10, sample_s)
    if len(grid) < 4:
        raise ValueError('insufficient_motion_samples')
    for axis, key in (('x', 'cx'), ('y', 'cy')):
        pos = interp(generator, grid, key)
        acc = np.diff(pos, n=2)/sample_s**2
        jerk = np.diff(pos, n=3)/sample_s**3
        result['peak_a'+axis] = float(np.max(np.abs(acc)))
        result['peak_j'+axis] = float(np.max(np.abs(jerk)))
    result['motion_samples'] = len(grid)
    return result


def standard_track(frame, fps):
    return dict(time=frame.frame.to_numpy()/fps, cx=(frame.x+frame.width/2).to_numpy(),
                cy=(frame.y+frame.height/2).to_numpy(), vx=frame.xVelocity.to_numpy(),
                length=frame.width.to_numpy(), lane=frame.laneId.to_numpy())


def reject(row, reason):
    return dict(row, eligible=False, exclusion=reason), None


def extract_event(rec, partition, vehicle, track, k, tracks, static, lanes, fps, cfg, by_frame):
    old, new = int(track.laneId.iloc[k-1]), int(track.laneId.iloc[k])
    flip = int(track.frame.iloc[k])
    row = dict(event_id=f'{rec:02d}:{vehicle}:{flip}', partition=partition,
               recording=rec, vehicle=int(vehicle), flip_frame=flip, old_lane=old, new_lane=new)
    if old not in lanes or new not in lanes or abs(new-old)!=1 or lanes[old][1]!=lanes[new][1]:
        return reject(row, 'nonadjacent_or_nondriving_lane')
    direction = lanes[old][1]
    if int(static.loc[vehicle, 'drivingDirection']) != direction:
        return reject(row, 'direction_metadata_mismatch')
    stability = int(round(cfg['stable_lane_s']*fps))
    if k < stability or k+stability > len(track):
        return reject(row, 'truncated_stable_lane_window')
    stable = track.iloc[k-stability:k+stability]
    if not np.all(np.diff(stable.frame)==1):
        return reject(row, 'discontinuous_stable_lane_window')
    if not (np.all(track.laneId.iloc[k-stability:k]==old) and np.all(track.laneId.iloc[k:k+stability]==new)):
        return reject(row, 'unstable_lane_labels')
    changes = np.flatnonzero(np.diff(track.laneId.to_numpy())!=0)+1
    previous = changes[changes<k]
    following = changes[changes>k]
    low = max(int(previous[-1]) if len(previous) else 0, k-int(cfg['search_radius_s']*fps))
    high = min(int(following[0]) if len(following) else len(track), k+int(cfg['search_radius_s']*fps)+1)
    window = track.iloc[low:high]
    if not np.all(np.diff(window.frame)==1):
        return reject(row, 'discontinuous_generator_window')
    center_old, center_new = lanes[old][0], lanes[new][0]
    progress = (window.y.to_numpy()+window.height.to_numpy()/2-center_old)/(center_new-center_old)
    times = window.frame.to_numpy()/fps
    event_times = anchors(times, progress, flip/fps, cfg['lane_flip_center_cross_tolerance_s'])
    if event_times is None:
        return reject(row, 'unobserved_or_inconsistent_10_50_90_crossing')
    row.update(times=event_times, lane_center_distance=abs(center_new-center_old),
               lateral_backtrack_fraction=float(max(0., -np.min(np.diff(progress)))))
    generator = standard_track(window, fps)
    middle_frame = int(np.ceil(event_times[1]*fps-1e-8))
    frame = by_frame.get_group(middle_frame)
    sign = -1 if direction==1 else 1
    gx = float(interp(generator, middle_frame/fps, 'cx'))
    candidates = frame[(frame.laneId==new) & (frame.id!=vehicle)].copy()
    candidates = candidates[candidates.id.map(static.drivingDirection)==direction]
    candidates['behind'] = sign*(gx-candidates.x-candidates.width/2)
    candidates = candidates[candidates.behind>0].sort_values(['behind', 'id'])
    if candidates.empty:
        return reject(row, 'no_observed_destination_follower')
    follower = int(candidates.iloc[0].id)
    row['follower'] = follower
    if static.loc[vehicle, 'class']!='Car' or static.loc[follower, 'class']!='Car':
        return reject(row, 'non_car_pair')
    rear = tracks[follower]
    lo, hi = int(np.floor(event_times[0]*fps)), int(np.ceil(event_times[2]*fps))
    rear = rear[(rear.frame>=lo) & (rear.frame<=hi)]
    if len(rear)!=hi-lo+1 or not np.array_equal(rear.frame.to_numpy(), np.arange(lo, hi+1)):
        return reject(row, 'truncated_or_discontinuous_follower')
    if not np.all(rear.laneId==new):
        return reject(row, 'follower_changes_lane')
    target = standard_track(rear, fps)
    try:
        values = descriptors(generator, target, event_times, sign, cfg['common_position_sample_s'])
    except ValueError as error:
        return reject(row, str(error))
    if any(values[a]['generator_speed']<cfg['minimum_speed_mps'] or
           values[a]['target_speed']<cfg['minimum_speed_mps'] for a in ('p10','p50')):
        return reject(row, 'below_speed_population')
    all_times = rear.frame.to_numpy()/fps
    gaps = sign*(interp(generator, all_times, 'cx')-interp(target, all_times, 'cx'))-(
        interp(generator, all_times, 'length')+interp(target, all_times, 'length'))/2
    if np.min(gaps)<=0:
        return reject(row, 'nonpositive_longitudinal_gap')
    row.update(eligible=True, exclusion=None, metrics=values)
    saved = dict(event_id=row['event_id'], sign=sign, times=event_times,
                 generator={key: value.tolist() for key,value in generator.items()},
                 target={key: value.tolist() for key,value in target.items()})
    return row, saved


def synthetic_queries(sources, replays, cfg):
    queries = []
    for kind, path in (('source', sources), ('replay', replays)):
        with Path(path).open() as f:
            for line in f:
                row = json.loads(line)
                if row['geometry']!='adjacent' or row['maneuver']=='nominal':
                    continue
                states = [[r[actor] for r in row['trace']] for actor in ('generator','target')]
                initial = row['physical_initial_state']
                objects = []
                for actor in range(2):
                    sequence = [initial[actor]]+states[actor]
                    objects.append(dict(time=np.arange(len(sequence))*.1,
                        cx=np.array([s['x'] for s in sequence]), cy=np.array([s['y'] for s in sequence]),
                        vx=np.array([s['vx'] for s in sequence]), length=np.array([s['length'] for s in sequence])))
                generator, target = objects
                progress=(generator['cy']-initial[0]['y'])/(initial[1]['y']-initial[0]['y'])
                event_times=anchors(generator['time'],progress,2.,cfg['lane_flip_center_cross_tolerance_s'])
                q = dict(query_id=f"{kind}:{row['source_id']}:{row.get('target_id','idm')}",
                         kind=kind, source_id=row['source_id'], target_id=row.get('target_id','idm'),
                         initial_headway=row['initial_headway'], initial_closing_speed=row['closing_speed'],
                         maneuver=row['maneuver'], pair_contact=row['pair_contact'],
                         exclusive_ttc_event=row['exclusive_ttc_event'])
                if event_times is None:
                    q.update(eligible=False, exclusion='truncated_lateral_query')
                else:
                    q.update(eligible=True, times=event_times,
                             metrics=descriptors(generator,target,event_times,1,cfg['common_position_sample_s']))
                queries.append(q)
    return queries


def weights(events):
    recordings=Counter(e['recording'] for e in events)
    vehicles=Counter((e['recording'],e['vehicle']) for e in events)
    per_record=Counter(rec for rec, vehicle in vehicles)
    return np.array([1/len(recordings)/per_record[e['recording']]/vehicles[e['recording'],e['vehicle']]
                     for e in events])


def vector(metrics, anchor, keys):
    return [metrics[anchor][key] if key in metrics[anchor] else metrics[key] for key in keys]


def support(events, queries, cfg):
    result=[]
    for partition in cfg['partitions']:
        eligible=[e for e in events if e['eligible'] and e['partition']==partition]
        weight=weights(eligible)
        for anchor in ('p10','p50'):
            for dimension, scales in (('state_duration',cfg['geometry_tolerances']),
                    ('state_duration_motion',dict(cfg['geometry_tolerances'],**cfg['motion_tolerances']))):
                data=np.array([vector(e['metrics'],anchor,scales) for e in eligible])
                scale=np.array(list(scales.values()))
                for query in queries:
                    if not query['eligible']:
                        continue
                    item=dict(query_id=query['query_id'],partition=partition,anchor=anchor,dimensions=dimension,
                              eligible_events=len(eligible),nonempty_recordings=len({e['recording'] for e in eligible}))
                    distances=np.max(np.abs(data-np.array(vector(query['metrics'],anchor,scales)))/scale,axis=1) if eligible else np.array([])
                    order=np.argsort(distances)[:3]
                    item['nearest']=[dict(event_id=eligible[i]['event_id'],distance=float(distances[i])) for i in order]
                    for radius in cfg['neighborhood_radii']:
                        mask=distances<=radius
                        selected=[e for e,m in zip(eligible,mask) if m]
                        item[f'radius_{radius:g}']=dict(events=len(selected),
                            vehicles=len({(e['recording'],e['vehicle']) for e in selected}),
                            recordings=len({e['recording'] for e in selected}),weighted_mass=float(weight[mask].sum()))
                    result.append(item)
    return result


def main():
    ap=argparse.ArgumentParser()
    for name in ('config','data','calibration','sources','replays','output'):
        ap.add_argument('--'+name,type=Path,required=True)
    args=ap.parse_args()
    cfg=json.loads(args.config.read_text())
    assert cfg['partitions']==['calibration','development']
    assert digest(args.calibration/'complete.json')==cfg['calibration_complete_sha256']
    assert digest(args.calibration/'protocol.json')==cfg['calibration_protocol_sha256']
    assert digest(args.sources)==cfg['source_sha256']
    assert digest(args.replays)==cfg['replay_sha256']
    calibration=json.loads((args.calibration/'complete.json').read_text())
    split=calibration['protocol']['split']
    allowed=split['calibration_recordings']+split['development_recordings']
    assert len(set(allowed))==len(allowed) and not set(allowed)&set(split['test_recordings'])
    args.output.mkdir(parents=True,exist_ok=False)
    write(args.output/'locked_protocol.json',dict(config=cfg,split=split,config_sha256=digest(args.config),
        script_sha256=digest(__file__),source_sha256=digest(args.sources),replay_sha256=digest(args.replays),
        python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__))
    expected={r['recording']:r['sha256'] for r in calibration['records']}
    events=[]; records=[]; accessed=[]; started=time.time()
    try:
        with (args.output/'events.jsonl').open('w') as ef, (args.output/'windows.jsonl').open('w') as wf:
            for partition in cfg['partitions']:
                for recording in split[partition+'_recordings']:
                    assert recording not in split['test_recordings']
                    paths={suffix:args.data/f'{recording:02d}_{suffix}.csv' for suffix in ('tracks','tracksMeta','recordingMeta')}
                    hashes={suffix:digest(path) for suffix,path in paths.items()}
                    if partition=='calibration':
                        assert hashes['tracks']==expected[recording]
                    accessed.append(recording)
                    meta=pd.read_csv(paths['recordingMeta']).iloc[0].to_dict()
                    assert int(meta['locationId']) in split[partition+'_locations']
                    fps=float(meta['frameRate']); lanes=lane_centers(meta)
                    static=pd.read_csv(paths['tracksMeta']).set_index('id')
                    df=pd.read_csv(paths['tracks'],usecols=['frame','id','x','y','width','height','xVelocity','laneId'])
                    assert not df.duplicated(['id','frame']).any()
                    assert np.isfinite(df.to_numpy()).all()
                    tracks={int(i):g.sort_values('frame').reset_index(drop=True) for i,g in df.groupby('id',sort=True)}
                    by_frame=df.groupby('frame',sort=False)
                    start=len(events)
                    for vehicle,track in tracks.items():
                        for k in np.flatnonzero(np.diff(track.laneId.to_numpy())!=0)+1:
                            row,window=extract_event(recording,partition,vehicle,track,int(k),tracks,static,lanes,fps,cfg,by_frame)
                            events.append(row);ef.write(json.dumps(row,allow_nan=False)+'\n')
                            if window is not None: wf.write(json.dumps(window,allow_nan=False)+'\n')
                    local=events[start:]
                    item=dict(recording=recording,partition=partition,location=int(meta['locationId']),
                              hashes=hashes,frames=len(df),vehicles=len(tracks),candidates=len(local),
                              eligible=sum(e['eligible'] for e in local),
                              exclusions=dict(Counter(e['exclusion'] for e in local if not e['eligible'])))
                    records.append(item);print(json.dumps(item),flush=True)
        queries=synthetic_queries(args.sources,args.replays,cfg)
        write(args.output/'queries.json',queries)
        write(args.output/'support.json',support(events,queries,cfg))
        counts={p:dict(candidates=sum(e['partition']==p for e in events),
                      eligible=sum(e['partition']==p and e['eligible'] for e in events),
                      exclusions=dict(Counter(e['exclusion'] for e in events if e['partition']==p and not e['eligible']))) for p in cfg['partitions']}
        assert sorted(accessed)==sorted(allowed)
        complete=dict(version=cfg['version'],partitions=counts,records=records,
                      trajectory_recordings_accessed=accessed,forbidden_recordings_accessed=[],
                      test_partition_remains_unopened=True,queries=len(queries),eligible_queries=sum(q['eligible'] for q in queries),
                      seconds=time.time()-started,artifacts={p.name:digest(p) for p in args.output.iterdir() if p.is_file()})
        write(args.output/'complete.json',complete)
        print(json.dumps({k:v for k,v in complete.items() if k not in ('records','artifacts')},indent=2),flush=True)
    except Exception as error:
        write(args.output/'failure.json',dict(error=repr(error),trajectory_recordings_accessed=accessed))
        raise


if __name__=='__main__':
    main()
