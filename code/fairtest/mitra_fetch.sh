#!/bin/bash
# B41 data acquisition (2026-09-28 13:49): MiTra, TU Dresden OPARA,
# DOI 10.25532/OPARA-881, CC BY 4.0 -- the trajectory archives Data_T1..T9 and the Readme ONLY (no tracking logs or
# videos, no Teleporting_jumps.csv). At most 3 files at a time (the repository limits each connection to ~100-200 kB/s);
# partial files resume. Each file is checked against OPARA's size and MD5; sha256 goes into the manifest.
# Before the B41 pre-registration is frozen, nothing beyond sizes, hashes, archive listings, CSV header lines, row
# counts and the Readme may be looked at (the B28 rule).
D=$HOME/fairtest/mitra; mkdir -p $D; cd $D
export API=https://opara.zih.tu-dresden.de/server/api/core/bitstreams
fetch() {  # UUID:NAME
  u=${1%%:*}; f=${1#*:}
  [ -s "meta_$f.json" ] || curl -sf --max-time 60 "$API/$u" > "meta_$f.json" || { echo "meta failed $f"; return; }
  want=$(python3 -c "import json, sys; print(json.load(open(sys.argv[1]))['sizeBytes'])" "meta_$f.json")
  have=$( [ -f "$f" ] && stat -c %s "$f" || echo 0 )
  [ "$have" = "$want" ] && { echo "complete $f"; return; }
  curl -sS -fL --retry 5 --retry-delay 15 -C - -o "$f" "$API/$u/content" && echo "done $f" || echo "download failed $f"
}
export -f fetch
printf '%s\n' \
  48e0a468-c779-4f37-80aa-68bbeb1a8fd2:Readme_OPARA_MiTra.pdf \
  83c9ec15-7438-4d45-8b5d-f0d57b3f49a4:Data_T1.zip \
  148d57aa-5004-4052-a37e-a84b39733ed4:Data_T2.zip \
  76e56d99-3c09-4086-bad1-7f23e914410f:Data_T3.zip \
  e2481143-cba1-42f7-97a1-750ec30e510e:Data_T4.zip \
  01c03292-26d4-4815-bbff-a6ec8f7a9635:Data_T5.zip \
  5f252cf5-6d10-4dd9-984e-b6a5657a60fe:Data_T6.zip \
  a73a313c-42a5-4659-8f7b-eb2e2b5d528d:Data_T7.zip \
  d13a5f47-145e-49a9-80c9-d256ce598f3d:Data_T8.zip \
  c25589d5-fa2d-4216-94f4-a097b4bf5dcc:Data_T9.zip | xargs -P 3 -I{} bash -c 'fetch "$1"' _ {}
python3 - <<'EOF'
import hashlib, json, os
out, ok = [], True
for fn in sorted(os.listdir('.')):
    if not fn.startswith('meta_'):
        continue
    name = fn[5:-5]
    meta = json.load(open(fn))
    size = os.path.getsize(name) if os.path.exists(name) else None
    md5, sha = hashlib.md5(), hashlib.sha256()
    if size is not None:
        with open(name, 'rb') as f:
            for chunk in iter(lambda: f.read(1 << 20), b''):
                md5.update(chunk); sha.update(chunk)
    want = (meta.get('checkSum') or {}).get('value')
    row = dict(name=name, bitstream=meta.get('uuid'), size=size, size_opara=meta.get('sizeBytes'), md5=md5.hexdigest(),
               md5_opara=want, md5_algorithm=(meta.get('checkSum') or {}).get('checkSumAlgorithm'), sha256=sha.hexdigest())
    row['verified'] = size == meta.get('sizeBytes') and (want is None or want == row['md5'])
    ok &= row['verified']
    out.append(row)
json.dump(dict(source='OPARA DOI 10.25532/OPARA-881 (CC BY 4.0)', files=out, all_verified=ok), open('manifest.json', 'w'), indent=1)
print(json.dumps(dict(files=len(out), all_verified=ok)))
EOF
echo done > $HOME/fairtest/mitra_done.flag
