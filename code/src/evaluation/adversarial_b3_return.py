"""Lossless, bounded B3 result return through JSON/NPZ-only artifacts.

This module only packages saved files. It never imports a simulator or learner.
The local tar.gz is authoritative when an artifact capacity limit is exceeded;
four empty NPZ files then make the failure explicit without returning a prefix.
"""
import argparse
import base64
import gzip
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import tarfile
import zlib

import numpy as np


PART_COUNT = 4
PART_SIZE_BYTES = 128*1024*1024
MAX_ARTIFACT_FILE_BYTES = 512*1024*1024
MAX_ARTIFACT_TOTAL_BYTES = 2*1024*1024*1024
MAX_STDOUT_BYTES = 2*1024*1024
MAX_SUMMARY_SOURCE_BYTES = 16*1024*1024
ARCHIVE_NAME = 'b3-results.tar.gz'
SUMMARY_NAME = 'return-summary.json'
BEGIN = 'B3_SUMMARY_BEGIN '
CHUNK = 'B3_SUMMARY_CHUNK '
END = 'B3_SUMMARY_END '


class ReturnIntegrityError(ValueError):
    pass


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(1024*1024),b''):
            h.update(data)
    return h.hexdigest()


def _json(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf-8')


def _write_json(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False)
        stream.write('\n')


def _reject_links(path):
    path=Path(path).absolute()
    for component in (path,*path.parents):
        if component.is_symlink() or (hasattr(component,'is_junction') and component.is_junction()):
            raise ReturnIntegrityError('Symlink/junction paths are not accepted: '+str(component))


def _inventory(root):
    _reject_links(root)
    if not root.is_dir():
        raise ReturnIntegrityError('Results root must be an existing directory')
    resolved=root.resolve();files=[];directories=[]
    # os.walk never follows symlink directories; detect and reject rather than
    # silently dropping them from what is advertised as a complete package.
    for directory,dirs,names in os.walk(root,followlinks=False):
        for name in sorted(dirs):
            path=Path(directory)/name;_reject_links(path)
            if not path.resolve().is_relative_to(resolved) or not path.is_dir():
                raise ReturnIntegrityError('Unsafe directory in result tree')
            directories.append(path)
        for name in sorted(names):
            path=Path(directory)/name;_reject_links(path)
            metadata=path.lstat()
            if not path.resolve().is_relative_to(resolved) or not stat.S_ISREG(metadata.st_mode):
                raise ReturnIntegrityError('Only ordinary files are accepted: '+str(path))
            files.append((path,metadata))
    return sorted(files,key=lambda item:item[0].relative_to(root).as_posix()),sorted(directories)


class _HashingReader:
    def __init__(self,stream):
        self.stream=stream;self.hash=hashlib.sha256();self.size=0
    def read(self,size=-1):
        value=self.stream.read(size);self.hash.update(value);self.size+=len(value)
        return value


def _archive(root,output):
    inventory,directories=_inventory(root);records=[]
    with output.open('xb') as raw:
        with gzip.GzipFile(filename='',fileobj=raw,mode='wb',mtime=0,compresslevel=6) as compressed:
            with tarfile.open(fileobj=compressed,mode='w|',format=tarfile.PAX_FORMAT) as archive:
                for directory in directories:
                    info=tarfile.TarInfo(directory.relative_to(root).as_posix()+'/')
                    info.type=tarfile.DIRTYPE;info.mode=0o755;archive.addfile(info)
                for path,before in inventory:
                    _reject_links(path)
                    flags=os.O_RDONLY|getattr(os,'O_BINARY',0)|getattr(os,'O_NOFOLLOW',0)
                    descriptor=os.open(path,flags)
                    with os.fdopen(descriptor,'rb') as stream:
                        opened=os.fstat(stream.fileno())
                        if (not stat.S_ISREG(opened.st_mode) or (opened.st_dev,opened.st_ino)!=(before.st_dev,before.st_ino)
                                or opened.st_size!=before.st_size):
                            raise ReturnIntegrityError('Result file changed while opening: '+str(path))
                        reader=_HashingReader(stream)
                        info=tarfile.TarInfo(path.relative_to(root).as_posix())
                        info.size=opened.st_size;info.mode=0o644
                        archive.addfile(info,reader)
                        after=os.fstat(stream.fileno())
                        if (reader.size!=opened.st_size or stream.read(1) or
                            (after.st_size,after.st_mtime_ns)!=(opened.st_size,opened.st_mtime_ns)):
                            raise ReturnIntegrityError('Result file changed during packaging: '+str(path))
                        records.append(dict(path=info.name,size_bytes=reader.size,sha256=reader.hash.hexdigest()))
    # A new file or replaced inode after inventory must not silently disappear.
    after,_=_inventory(root)
    signatures=lambda rows:[(p.relative_to(root).as_posix(),s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns) for p,s in rows]
    if signatures(after)!=signatures(inventory):
        raise ReturnIntegrityError('Result tree changed during packaging')
    return records


def _part(output,index,payload):
    path=output/f'part{index:02d}.npz'
    with path.open('xb') as stream:
        np.savez(stream,payload=np.frombuffer(payload,dtype=np.uint8))
    if path.stat().st_size>MAX_ARTIFACT_FILE_BYTES:
        raise ReturnIntegrityError('NPZ artifact exceeds the per-file return limit')
    return dict(index=index,file=path.name,size_bytes=path.stat().st_size,sha256=sha256(path),
                payload_bytes=len(payload),payload_sha256=hashlib.sha256(payload).hexdigest())


def package_results(results,output,*,part_size_bytes=PART_SIZE_BYTES):
    """Save all results and return four lossless NPZ chunks plus a JSON manifest.

    ``part_size_bytes`` only permits smaller limits, for bounded unit fixtures.
    Capacity failure returns complete=False (CLI exit 2), retains the complete
    local archive, and publishes four EMPTY parts. No truncated prefix is valid.
    Input/IO/integrity errors raise and never claim a complete package.
    """
    results=Path(results).absolute();output=Path(output).absolute()
    if isinstance(part_size_bytes,bool) or not isinstance(part_size_bytes,int) or not 1<=part_size_bytes<=PART_SIZE_BYTES:
        raise ValueError('Part size must be a positive integer no larger than 128 MiB')
    _reject_links(results);_reject_links(output)
    if output.resolve().is_relative_to(results.resolve()):
        raise ReturnIntegrityError('Return directory must be outside the results tree')
    output.mkdir(parents=True,exist_ok=True)
    names=[ARCHIVE_NAME,SUMMARY_NAME]+[f'part{i:02d}.npz' for i in range(PART_COUNT)]
    if any((output/name).exists() for name in names):
        raise FileExistsError('Refusing to overwrite an existing B3 return package')
    archive=output/ARCHIVE_NAME
    records=_archive(results,archive)
    size=archive.stat().st_size;capacity=PART_COUNT*part_size_bytes
    complete=size<=capacity
    parts=[]
    with archive.open('rb') as stream:
        for index in range(PART_COUNT):
            parts.append(_part(output,index,stream.read(part_size_bytes) if complete else b''))
        if complete and stream.read(1):
            raise ReturnIntegrityError('Archive grew beyond the declared chunk capacity')
    manifest=dict(version='adversarial-B3-return-v1',complete=complete,
        status='complete' if complete else 'capacity_exceeded',
        error=None if complete else 'Complete archive exceeds four parts; all returned payloads are empty. Local full archive retained.',
        archive=dict(file=ARCHIVE_NAME,size_bytes=size,sha256=sha256(archive),locally_preserved=True),
        part_size_bytes=part_size_bytes,part_count=PART_COUNT,capacity_bytes=capacity,parts=parts,
        source_files=records,source_file_count=len(records),source_bytes=sum(r['size_bytes'] for r in records),
        artifact_return_files=[p['file'] for p in parts]+[SUMMARY_NAME],
        allowed_suffixes=['.npz','.json'],artifact_file_limit_bytes=MAX_ARTIFACT_FILE_BYTES,
        artifact_total_limit_bytes=MAX_ARTIFACT_TOTAL_BYTES,artifact_file_count_limit=64,
        authority='The full local tar.gz and per-source-file hashes are retained; incomplete parts must never be reassembled as results.')
    _write_json(output/SUMMARY_NAME,manifest)
    total=sum((output/name).stat().st_size for name in manifest['artifact_return_files'])
    if total>MAX_ARTIFACT_TOTAL_BYTES or (output/SUMMARY_NAME).stat().st_size>MAX_ARTIFACT_FILE_BYTES:
        raise ReturnIntegrityError('Manifest/artifacts exceed the registered return limits')
    return manifest


def restore_archive(return_root,archive_path):
    """Verify four returned NPZ files and atomically restore the original tar.gz."""
    root=Path(return_root);destination=Path(archive_path)
    _reject_links(root);_reject_links(destination)
    with (root/SUMMARY_NAME).open(encoding='utf-8') as stream:manifest=json.load(stream)
    if not manifest.get('complete') or manifest.get('part_count')!=PART_COUNT:
        raise ReturnIntegrityError('Incomplete return package cannot be restored')
    parts=manifest['parts']
    if len(parts)!=PART_COUNT or [p['file'] for p in parts]!=[f'part{i:02d}.npz' for i in range(PART_COUNT)]:
        raise ReturnIntegrityError('Unexpected or unsafe part list')
    if destination.exists():raise FileExistsError('Refusing to overwrite restored archive')
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=destination.with_name(destination.name+'.partial')
    h=hashlib.sha256();size=0;created=False
    try:
        with temporary.open('xb') as out:
            created=True
            for record in parts:
                path=root/record['file'];_reject_links(path)
                if path.stat().st_size!=record['size_bytes'] or sha256(path)!=record['sha256']:
                    raise ReturnIntegrityError('NPZ hash/size mismatch: '+path.name)
                with np.load(path,allow_pickle=False) as data:
                    if data.files!=['payload']:
                        raise ReturnIntegrityError('Unexpected NPZ members')
                    payload=data['payload']
                    if payload.dtype!=np.uint8 or payload.ndim!=1 or payload.size>PART_SIZE_BYTES:
                        raise ReturnIntegrityError('NPZ payload must be a bounded uint8 vector')
                    raw=payload.tobytes()
                if len(raw)!=record['payload_bytes'] or hashlib.sha256(raw).hexdigest()!=record['payload_sha256']:
                    raise ReturnIntegrityError('NPZ payload hash/size mismatch')
                out.write(raw);h.update(raw);size+=len(raw)
        if size!=manifest['archive']['size_bytes'] or h.hexdigest()!=manifest['archive']['sha256']:
            raise ReturnIntegrityError('Reassembled archive hash/size mismatch')
        temporary.replace(destination)
    except BaseException:
        if created and temporary.exists():temporary.unlink()
        raise
    return dict(path=str(destination),size_bytes=size,sha256=h.hexdigest(),verified=True)


def _compact_document(value):
    if isinstance(value,list):return [_compact_document(item) for item in value]
    if not isinstance(value,dict):return value
    output={}
    for key,item in value.items():
        if key in ('attempts','hashes','details','initial','final','points','trace','trajectories',
                   'decisions','tape','observations','actions','returns','buffers','rollout_buffers',
                   'model_state','state_dict','parameters','weights'):
            output[key+'_omitted_count']=len(item) if hasattr(item,'__len__') else 1
        elif key=='traceback' and isinstance(item,str):
            output[key]=item[:8192]
            if len(item)>8192:output['traceback_truncated']=True
        else:output[key]=_compact_document(item)
    return output


def _summary_document(results,bootstrap_status):
    candidates=['analysis/summary.json','analysis/method_summary.json','analysis/cost_report.json','analysis/audit.json',
                'analysis/paired_comparison.json','analysis/slot_registry.json','analysis/resource_report.json',
                'campaign/sentinel_gate.json','campaign/actor_preflight.json',
                'campaign/technical_gate.json','campaign/cost_report.json',
                'campaign/complete.json','campaign/failure.json','campaign/resources.json','campaign/ledger.json',
                'campaign/integration_gate.json','complete.json','failure.json','resources.json','ledger.json',
                'integration_gate.json','bootstrap_status.json','meta/bootstrap_status.json']
    documents=[]
    for relative in candidates:
        path=results/relative
        if not path.exists():continue
        _reject_links(path)
        if not path.is_file() or not path.resolve().is_relative_to(results.resolve()):
            raise ReturnIntegrityError('Unsafe summary document path')
        size=path.stat().st_size
        entry=dict(path=relative,size_bytes=size,sha256=sha256(path))
        if size>MAX_SUMMARY_SOURCE_BYTES:
            entry.update(omitted=True,reason='Source exceeds bounded summary reader limit')
        else:
            try:
                with path.open(encoding='utf-8-sig') as stream:entry['data']=_compact_document(json.load(stream))
            except (ValueError,UnicodeError) as exc:
                entry.update(omitted=True,reason='Invalid JSON: '+str(exc)[:300])
        documents.append(entry)
    return dict(version='adversarial-B3-stdout-summary-v1',bootstrap_status=_compact_document(bootstrap_status),
                scope='Selected compact JSON summaries only; no raw trajectories, model tensors, rollout buffers or raw logs.',
                documents=documents,summary_contents_complete=all(not r.get('omitted') for r in documents))


def _encoded_lines(document):
    raw=_json(document);compressed=zlib.compress(raw,9);encoded=base64.b64encode(compressed).decode('ascii')
    chunks=[encoded[i:i+3072] for i in range(0,len(encoded),3072)]
    header=dict(version=1,codec='zlib+base64',raw_size_bytes=len(raw),compressed_size_bytes=len(compressed),
                raw_sha256=hashlib.sha256(raw).hexdigest(),compressed_sha256=hashlib.sha256(compressed).hexdigest(),
                chunks=len(chunks),base64_bytes=len(encoded),summary_contents_complete=document['summary_contents_complete'])
    lines=[BEGIN+_json(header).decode('utf-8')]+[
        CHUNK+f'{i:04d}/{len(chunks):04d} '+chunk for i,chunk in enumerate(chunks)]+[END+header['raw_sha256']]
    return header,'\n'.join(lines)+'\n'


def export_summary(results,bootstrap_status=None,stream=None,*,max_stdout_bytes=MAX_STDOUT_BYTES):
    """Emit <=2 MiB of checksummed summary chunks suitable for CI log recovery."""
    results=Path(results).absolute();_reject_links(results)
    if isinstance(max_stdout_bytes,bool) or not isinstance(max_stdout_bytes,int) or not 1024<=max_stdout_bytes<=MAX_STDOUT_BYTES:
        raise ValueError('Summary stdout bound must be between 1024 and 2097152 bytes')
    document=_summary_document(results,bootstrap_status)
    header,payload=_encoded_lines(document)
    # Drop largest data objects, retaining their path/hash, until strictly bound.
    for entry in sorted(document['documents'],key=lambda d:len(_json(d.get('data'))),reverse=True):
        if len(payload.encode('utf-8'))<=max_stdout_bytes:break
        if 'data' in entry:
            del entry['data'];entry.update(omitted=True,reason='Excluded to honor stdout byte bound')
            document['summary_contents_complete']=False
            header,payload=_encoded_lines(document)
    if len(payload.encode('utf-8'))>max_stdout_bytes:
        document['bootstrap_status']=dict(omitted=True,reason='Bootstrap status exceeds stdout bound',
                                           sha256=hashlib.sha256(_json(bootstrap_status)).hexdigest())
        document['summary_contents_complete']=False
        header,payload=_encoded_lines(document)
    if len(payload.encode('utf-8'))>max_stdout_bytes:
        raise ReturnIntegrityError('Summary metadata itself exceeds stdout bound; nothing emitted')
    (stream or sys.stdout).write(payload)
    if hasattr(stream or sys.stdout,'flush'):(stream or sys.stdout).flush()
    return dict(header,stdout_size_bytes=len(payload.encode('utf-8')))


def decode_summary_export(text):
    """Recover and verify exactly one summary envelope from ordinary CI logs."""
    rows=[]
    for line in text.splitlines():
        positions=[p for prefix in (BEGIN,CHUNK,END) if (p:=line.find(prefix))>=0]
        if positions:rows.append(line[min(positions):])
    starts=[line for line in rows if line.startswith(BEGIN)]
    ends=[line for line in rows if line.startswith(END)]
    if len(starts)!=1 or len(ends)!=1:raise ReturnIntegrityError('Expected one complete summary envelope')
    header=json.loads(starts[0][len(BEGIN):]);chunks={}
    if header.get('codec')!='zlib+base64' or not 0<header.get('raw_size_bytes',0)<=MAX_SUMMARY_SOURCE_BYTES*16:
        raise ReturnIntegrityError('Unsupported summary codec or unbounded document')
    for line in rows:
        if line.startswith(CHUNK):
            index,data=line[len(CHUNK):].split(' ',1);number,total=map(int,index.split('/'))
            if total!=header['chunks'] or number in chunks:raise ReturnIntegrityError('Duplicate or inconsistent chunk')
            chunks[number]=data
    if sorted(chunks)!=list(range(header['chunks'])):raise ReturnIntegrityError('Missing summary chunks')
    encoded=''.join(chunks[i] for i in range(header['chunks']))
    if len(encoded)!=header['base64_bytes'] or len(encoded)>MAX_STDOUT_BYTES:raise ReturnIntegrityError('Encoded size mismatch')
    try:compressed=base64.b64decode(encoded,validate=True)
    except ValueError as exc:raise ReturnIntegrityError('Invalid base64') from exc
    if len(compressed)!=header['compressed_size_bytes'] or hashlib.sha256(compressed).hexdigest()!=header['compressed_sha256']:
        raise ReturnIntegrityError('Compressed summary hash/size mismatch')
    decompressor=zlib.decompressobj()
    raw=decompressor.decompress(compressed,header['raw_size_bytes']+1)
    if len(raw)!=header['raw_size_bytes'] or not decompressor.eof or decompressor.unused_data:
        raise ReturnIntegrityError('Decoded summary size/stream mismatch')
    if hashlib.sha256(raw).hexdigest()!=header['raw_sha256'] or ends[0][len(END):]!=header['raw_sha256']:
        raise ReturnIntegrityError('Decoded summary hash mismatch')
    return json.loads(raw)


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    pack=sub.add_parser('package');pack.add_argument('--results',type=Path,required=True);pack.add_argument('--output',type=Path,required=True)
    export=sub.add_parser('export');export.add_argument('--results',type=Path,required=True);export.add_argument('--bootstrap-status',type=Path)
    restore=sub.add_parser('restore');restore.add_argument('--return-root',type=Path,required=True);restore.add_argument('--archive',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='package':
        result=package_results(args.results,args.output)
        print(json.dumps({key:result[key] for key in ('status','complete','archive','source_file_count')},sort_keys=True))
        return 0 if result['complete'] else 2
    if args.command=='export':
        status=None
        if args.bootstrap_status:
            _reject_links(args.bootstrap_status)
            if args.bootstrap_status.stat().st_size>MAX_SUMMARY_SOURCE_BYTES:raise ReturnIntegrityError('Bootstrap status is too large')
            with args.bootstrap_status.open(encoding='utf-8') as stream:status=json.load(stream)
        export_summary(args.results,status);return 0
    print(json.dumps(restore_archive(args.return_root,args.archive),sort_keys=True));return 0


if __name__=='__main__':raise SystemExit(main())
