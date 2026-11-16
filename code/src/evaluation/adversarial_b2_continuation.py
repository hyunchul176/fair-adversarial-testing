"""Zero-native B2 saved-episode reconciliation with the original budget intact."""
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import shutil

from src.evaluation.adversarial_b2_budget import B2Budget
from src.evaluation.adversarial_b1_budget import BudgetError
from src.evaluation.adversarial_a2_support import read, sha, equivalent

SOURCE='0705eb38384adb2f1a112af7eb03b1af459f4726a3b90c005eba3f46c5c1833a'
JOURNAL='125061bc7c69425135eb1eb53f30d982de9f87f9b660560b7f859954c4e2a9d2'
SNAPSHOT='13f407795bb8ac92b608eac9f2281169fa2ece9ea1506535293f24c607886a0c'
BYTES=1087453
SEQUENCE=7801
ATTEMPT='search_open_loop_b2_42_train_000297'

def lines(p):return [json.loads(s) for s in Path(p).read_text(encoding='utf-8').splitlines()]
def episode(p):return json.loads(gzip.decompress(Path(p).read_bytes()))

class ContinuedB2Budget(B2Budget):
    def __init__(self,*args,**kwargs):
        self.reconciliation=None
        super().__init__(*args,**kwargs)

    def _apply(self,event,persist=True):
        if event['op']=='reconcile_b2_saved_v1':
            receipt=event['receipt']
            if self.reconciliation is not None or self.sequence!=SEQUENCE or self._journal_bytes!=BYTES:
                raise BudgetError('Unexpected B2 reconciliation boundary')
            if hashlib.sha256(self.journal_path.read_bytes()[:BYTES]).hexdigest()!=JOURNAL:
                raise BudgetError('Original B2 journal prefix changed')
            a=self.attempts.get(ATTEMPT,{})
            expected=dict(status='failed',measured_ticks=44,measured_actors=9,ticks=120,actors=24,uncertain_failed_calls=1)
            if any(a.get(k)!=v for k,v in expected.items()) or receipt['attempt_id']!=ATTEMPT or receipt['original_journal_sha256']!=JOURNAL or receipt['old_source_sha256']!=SOURCE:
                raise BudgetError('B2 completed-failure evidence differs')
            self._hash(receipt['execution_source_sha256'])
            if persist:self._append(event)
            self.reconciliation=deepcopy(receipt)
            return
        if event['op']=='start' and self.reconciliation is not None:
            if event['condition'].get('execution_source_sha256')!=self.reconciliation['execution_source_sha256']:
                raise BudgetError('B2 continuation source differs')
        return super()._apply(event,persist)

    def reconcile(self,receipt):
        if self.reconciliation is not None:
            if self.reconciliation!=receipt:raise BudgetError('Conflicting B2 reconciliation')
            return
        self._apply(dict(op='reconcile_b2_saved_v1',receipt=receipt))

    def summary(self):
        result=super().summary()
        if self.reconciliation is not None:result['saved_episode_reconciliation']=deepcopy(self.reconciliation)
        return result

def restore(prior,output,factory,manifest):
    """Reaudit all313 saved episodes, then append a receipt; never refund costs."""
    from src.evaluation.adversarial_b2_audit import audit_episode
    from scripts.analyze_adversarial_agent_a1 import audit_episode as audit_old
    prior,output=Path(prior),Path(output)
    if sha(prior/'ledger.json')!=SNAPSHOT or sha(prior/'ledger.json.events.jsonl')!=JOURNAL:
        raise ValueError('B2 original accounting hashes differ')
    ledger=B2Budget(prior/'ledger.json',SOURCE,sha(factory.data/'protocol.json'))
    if (ledger.summary()['global_totals']['starts'],ledger.summary()['global_totals']['ticks'],ledger.summary()['global_totals']['actors'])!=(313,35907,7189):
        raise ValueError('B2 original costs differ')
    indices,partials=lines(prior/'episodes.jsonl'),lines(prior/'partials.jsonl')
    if len(indices)!=313 or len(partials)!=1:raise ValueError('B2 original coverage differs')
    refs=episode(factory.data/'sentinels/references.json.gz')
    windows=read(factory.data/'sentinels/windows.json');contract=read(factory.data/'contract.json')
    sentinels=[];seen=set();accepted=None
    for index in indices:
        if sha(prior/index['episode_path'])!=index['episode_sha256']:raise ValueError('B2 saved row drift')
        row=episode(prior/index['episode_path'])
        if row['phase']=='sentinel':
            case=factory.case(row['input_id'])
            audit_old(row,windows[case['event']['event_id']],case,[x['divisor'] for x in contract['observation']['features']])
            if not equivalent(row,next(r for r in refs if r['input_id']==row['input_id'])):raise ValueError('Sentinel differs')
            sentinels.append(row['input_id'])
        else:
            key=row['plan_id'],row['condition_id']
            if key in seen:raise ValueError('Duplicate B2 completed cell')
            seen.add(key);audit=audit_episode(row,factory.protocol)
            if row['run_id']==ATTEMPT:
                if key!=(9,8) or row['status']!='complete' or len(row['trace'])!=44 or len(row['decisions'])!=9:
                    raise ValueError('Unexpected B2 saved failure')
                accepted=index,row,audit
        if row['run_id']!=ATTEMPT and ledger.attempts[row['run_id']]['status']!='completed':raise ValueError('Unresolved attempt')
    if sorted(sentinels)!=list(range(16)) or len(seen)!=297 or accepted is None:raise ValueError('Coverage differs')
    index,row,audit=accepted;part=partials[0]
    if part['run_id']!=ATTEMPT or sha(prior/part['episode_path'])!=part['episode_sha256']:raise ValueError('Partial differs')
    failed=episode(prior/part['episode_path'])
    if failed['status']!='failed' or failed['reason']!='poststep_accounting_or_audit_exception' or failed['trace']!=row['trace'] or failed['decisions']!=row['decisions']:
        raise ValueError('Failure does not describe the same completed execution')
    if list(output.iterdir()):raise ValueError('Continuation output must be empty')
    shutil.copytree(prior,output,dirs_exist_ok=True)
    (output/'history').mkdir()
    for name in ('ledger.json','ledger.json.events.jsonl'):
        shutil.copy2(prior/name,output/'history'/('original-'+name))
    receipt=dict(version=1,attempt_id=ATTEMPT,old_source_sha256=SOURCE,execution_source_sha256=sha(manifest),
        original_journal_sha256=JOURNAL,original_snapshot_sha256=SNAPSHOT,original_journal_bytes=BYTES,
        completed_index=index,duplicate_partial_index=part,offline_audit=audit,
        additional_native_ticks=0,additional_actor_calls=0,operational_failed_count_retained=1,
        conservative_extra_ticks_retained=76,conservative_extra_actor_slots_retained=15,
        completed_cells_skipped=297,sentinels_skipped=16)
    result=ContinuedB2Budget(output/'ledger.json',SOURCE,sha(factory.data/'protocol.json'))
    result.reconcile(receipt);result.save()
    (output/'continuation.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    shutil.copy2(manifest,output/'continuation_runtime_manifest.json')
    with (output/'audits.jsonl').open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(dict(audit,recovered_after_audit_failure=True,run_id=ATTEMPT))+'\n')
    return result

def verify_receipt(results,protocol_sha):
    results=Path(results)
    if not (results/'continuation.json').exists():return None
    receipt=read(results/'continuation.json')
    if sha(results/'history/original-ledger.json')!=SNAPSHOT or sha(results/'history/original-ledger.json.events.jsonl')!=JOURNAL or sha(results/'continuation_runtime_manifest.json')!=receipt['execution_source_sha256']:
        raise ValueError('B2 continuation lineage changed')
    ledger=ContinuedB2Budget(results/'ledger.json',SOURCE,protocol_sha)
    if ledger.reconciliation!=receipt or read(results/'ledger.json')!=dict(ledger.summary(),attempts=ledger.attempts):raise ValueError('B2 journal/snapshot disagree')
    a,b=receipt['completed_index'],receipt['duplicate_partial_index']
    for item in (a,b):
        if sha(results/item['episode_path'])!=item['episode_sha256']:raise ValueError('B2 failure evidence changed')
    full,partial=episode(results/a['episode_path']),episode(results/b['episode_path'])
    if full['trace']!=partial['trace'] or full['decisions']!=partial['decisions']:raise ValueError('B2 duplicate execution differs')
    return receipt
