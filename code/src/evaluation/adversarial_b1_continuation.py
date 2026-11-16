"""Exact saved-episode reconciliation; keeps the failed journal prefix intact."""
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import shutil

from src.evaluation.adversarial_b1_budget import NativeBudget, BudgetError
from src.evaluation.adversarial_a2_support import read, sha, equivalent, json_value

ORIGINAL_SOURCE = 'dcabb5f7e885f05cc2f6f94d1de908f20dc1db883176b33919f8cbcc49a779e9'
ORIGINAL_JOURNAL = '1ef722757128ff7b433f059903b28bf98202410ac09be363a2a2b2ae05a7f599'
ORIGINAL_SNAPSHOT = '903f860f1986dfcda87ab02dcde76a1d2755db304aad228a644701f4695356ab'
ORIGINAL_BYTES = 44725
ORIGINAL_SEQUENCE = 409
ATTEMPT = 'search_open_loop_42_train_000001'

def lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()]

def episode(path):
    with gzip.open(path,'rt',encoding='utf-8') as stream: return json.load(stream)

class AppendJsonl:
    def __init__(self,path): self.stream=Path(path).open('a',encoding='utf-8')
    def append(self,row):
        self.stream.write(json.dumps(row,default=json_value,allow_nan=False)+'\n');self.stream.flush()
    def close(self): self.stream.close()

class ContinuedBudget(NativeBudget):
    """Original counters stay historical; reconciliation is a separate record."""
    def __init__(self,*args,**kwargs):
        self.reconciliation=None
        super().__init__(*args,**kwargs)

    def _apply(self,event,persist=True):
        if event['op']=='reconcile_saved_episode_v1':
            receipt=event['receipt']
            if self.reconciliation is not None: raise BudgetError('Reconciliation already recorded')
            if self.sequence!=ORIGINAL_SEQUENCE or self._journal_bytes!=ORIGINAL_BYTES:
                raise BudgetError('Unexpected original journal boundary')
            prefix=self.journal_path.read_bytes()[:ORIGINAL_BYTES]
            if hashlib.sha256(prefix).hexdigest()!=ORIGINAL_JOURNAL:
                raise BudgetError('Original journal prefix changed')
            attempt=self.attempts.get(ATTEMPT,{})
            if (attempt.get('status')!='failed' or attempt.get('measured_ticks')!=120
                    or attempt.get('measured_actors')!=24 or attempt.get('uncertain_failed_calls')!=1
                    or receipt['attempt_id']!=ATTEMPT or receipt['original_journal_sha256']!=ORIGINAL_JOURNAL
                    or receipt['old_source_sha256']!=ORIGINAL_SOURCE):
                raise BudgetError('Saved failure evidence differs')
            self._hash(receipt['execution_source_sha256'])
            if persist: self._append(event)
            self.reconciliation=deepcopy(receipt)
            return
        if event['op']=='start' and self.reconciliation is not None:
            if event['condition'].get('execution_source_sha256')!=self.reconciliation['execution_source_sha256']:
                raise BudgetError('Continuation attempt source identity differs')
        return super()._apply(event,persist)

    def reconcile(self,receipt):
        if self.reconciliation is not None:
            if self.reconciliation!=receipt: raise BudgetError('Conflicting reconciliation')
            return
        self._apply(dict(op='reconcile_saved_episode_v1',receipt=receipt))

    def summary(self):
        result=super().summary()
        if self.reconciliation is not None: result['saved_episode_reconciliation']=deepcopy(self.reconciliation)
        return result

def restore(root,output,factory,new_manifest):
    """Validate then copy original records; append one zero-native reconciliation."""
    from scripts.analyze_adversarial_b1 import audit_episode
    from scripts.analyze_adversarial_agent_a1 import audit_episode as audit_old
    root=Path(root);output=Path(output);source=root/'campaign'
    transport=read(root/'RECOVERY_RECEIPT.json')
    for name,digest in transport['files'].items():
        path=root/name
        if path.is_symlink() or sha(path)!=digest: raise ValueError('Recovered file drift: '+name)
    if sha(source/'ledger.json')!=ORIGINAL_SNAPSHOT or sha(source/'ledger.json.events.jsonl')!=ORIGINAL_JOURNAL:
        raise ValueError('Original ledger identity mismatch')
    protocol_sha=sha(factory.data/'protocol.json')
    old=NativeBudget(source/'ledger.json',ORIGINAL_SOURCE,protocol_sha)
    if old.summary()['global_totals']['ticks']!=1865 or old.summary()['global_totals']['actors']!=374:
        raise ValueError('Original budget differs')
    indexes=lines(source/'episodes.jsonl');partials=lines(source/'partials.jsonl')
    if len(indexes)!=17 or len(partials)!=1: raise ValueError('Unexpected recovered coverage')
    refs=episode(factory.data/'sentinels/references.json.gz')
    windows=read(factory.data/'sentinels/windows.json');contract=read(factory.data/'contract.json')
    sentinels=[];accepted=[]
    for index in indexes:
        if sha(source/index['episode_path'])!=index['episode_sha256']: raise ValueError('Episode hash differs')
        row=episode(source/index['episode_path'])
        if row['phase']=='sentinel':
            case=factory.case(row['input_id'])
            audit_old(row,windows[case['event']['event_id']],case,[x['divisor'] for x in contract['observation']['features']])
            if not equivalent(row,next(r for r in refs if r['input_id']==row['input_id'])):
                raise ValueError('Saved sentinel differs')
            if old.attempts[row['run_id']]['status']!='completed': raise ValueError('Sentinel not complete')
            sentinels.append(row['run_id'])
        else:
            audit=audit_episode(row,factory.protocol,factory.base,factory.case(row['input_id']))
            if (row['run_id']!=ATTEMPT or row['seed']!=42 or row['plan_id']!=0 or row['condition_id']!=0
                    or row['status']!='complete' or row['steps']!=120): raise ValueError('Unexpected saved B1 cell')
            accepted.append((index,row,audit))
    if len(set(sentinels))!=16 or len(accepted)!=1: raise ValueError('Recovery coverage differs')
    index,row,audit=accepted[0];part_index=partials[0]
    if part_index['run_id']!=ATTEMPT or sha(source/part_index['episode_path'])!=part_index['episode_sha256']:
        raise ValueError('Failed snapshot identity differs')
    partial=episode(source/part_index['episode_path'])
    if (partial['status']!='failed' or partial['reason']!='poststep_accounting_or_audit_exception'
            or partial['trace']!=row['trace'] or partial['decisions']!=row['decisions']):
        raise ValueError('Failure is not the same completed execution')
    if list(output.iterdir()): raise ValueError('Continuation output must start empty')
    shutil.copytree(source,output,dirs_exist_ok=True)
    (output/'history').mkdir()
    for name in ('ledger.json','ledger.json.events.jsonl','failure.json'):
        shutil.copy2(source/name,output/'history'/('original-'+name))
    shutil.copy2(new_manifest,output/'continuation_runtime_manifest.json')
    receipt=dict(version=1,attempt_id=ATTEMPT,old_source_sha256=ORIGINAL_SOURCE,
        execution_source_sha256=sha(new_manifest),original_journal_sha256=ORIGINAL_JOURNAL,
        original_journal_bytes=ORIGINAL_BYTES,original_snapshot_sha256=ORIGINAL_SNAPSHOT,
        completed_index=index,duplicate_partial_index=part_index,offline_audit=audit,
        sentinels_skipped=sentinels,additional_native_ticks=0,additional_actor_calls=0,
        operational_failed_count_retained=1,uncertainty_resolved_by_saved_complete_trace=True)
    ledger=ContinuedBudget(output/'ledger.json',ORIGINAL_SOURCE,protocol_sha)
    ledger.reconcile(receipt);ledger.save()
    with (output/'continuation.json').open('x',encoding='utf-8') as stream: json.dump(receipt,stream,indent=2)
    with (output/'audits.jsonl').open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(dict(audit,recovered_after_recording_failure=True))+'\n')
    return ledger,{(42,0,0):index}

def verify_receipt(results):
    """Analysis may de-duplicate only the one evidence-bound failed snapshot."""
    results=Path(results)
    if not (results/'continuation.json').exists(): return None
    receipt=read(results/'continuation.json')
    if (sha(results/'history/original-ledger.json')!=ORIGINAL_SNAPSHOT
            or sha(results/'history/original-ledger.json.events.jsonl')!=ORIGINAL_JOURNAL
            or sha(results/'continuation_runtime_manifest.json')!=receipt['execution_source_sha256']):
        raise ValueError('Continuation lineage hash mismatch')
    ledger=ContinuedBudget(results/'ledger.json',ORIGINAL_SOURCE,sha(results/'protocol.json'))
    if ledger.reconciliation!=receipt or read(results/'ledger.json')!=dict(ledger.summary(),attempts=ledger.attempts):
        raise ValueError('Journal/snapshot/reconciliation disagree')
    full=receipt['completed_index'];part=receipt['duplicate_partial_index']
    for item in (full,part):
        if sha(results/item['episode_path'])!=item['episode_sha256']: raise ValueError('Recovery trace hash differs')
    a,b=[episode(results/item['episode_path']) for item in (full,part)]
    if a['trace']!=b['trace'] or a['decisions']!=b['decisions']: raise ValueError('Duplicate execution differs')
    return receipt
