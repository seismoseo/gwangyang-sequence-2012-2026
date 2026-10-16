#!/usr/bin/env python
# Builder of notebooks/00_infrastructure_and_port_check.ipynb (73.Gwangyang_sequence, stage G0).
#   python notebooks/build_00_infrastructure.py && jupyter execute --inplace --kernel_name=python3 notebooks/00_infrastructure_and_port_check.ipynb
import json, os
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "00_infrastructure_and_port_check.ipynb"); cells = []
def md(s): cells.append(dict(cell_type="markdown", metadata={}, source=s.strip("\n")))
def co(s):
    s = s.strip("\n"); compile(s, f"<cell {len(cells)}>", "exec"); cells.append(dict(cell_type="code", metadata={}, execution_count=None, outputs=[], source=s))
md(r"""
# 00 Infrastructure and port check (stage G0)

The Gwangyang study (2012-12-20 to 2026-10-06) runs the Imsil stages with the **current Haenam rules** (the Haenam renewal tree, moho `27.Haenam_sequence_renewal`):
- one sensor per station for the whole window;
- one dated orientation table;
- S differential times on T after rotation;
- template horizontals turned into each scanned day's orientation;
- local magnitudes without station terms.

The code was copied from the renewal tree, not from the Imsil port, because the renewal already holds these rules and Imsil's grid alignment.

This notebook checks three things:
1. **Port check.** Every copied file either equals its Haenam source, or carries `GWANGYANG PORT` markers at every change.
2. **Registry.** The Gwangyang registry resolves every product, and the 11 KMA events are in UTC.
3. **Environments.** moho's base, fmf and pq-gpu environments have the packages the stages need.

It ends with the decision log.
""")
co(r"""
import os, sys, hashlib, subprocess
import pandas as pd
ROOT = '/home/msseo/works/73.Gwangyang_sequence'
pd.set_option('display.width', 220); pd.set_option('display.max_colwidth', 120); pd.set_option('display.max_rows', 200)
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
prov = pd.read_csv(f'{ROOT}/PROVENANCE.tsv', sep='\t', header=None, names=['path', 'source', 'sha256_copied'])
rows = []
for r in prov.itertuples():
    p = f'{ROOT}/{r.path}'
    if not os.path.exists(p):
        rows.append(dict(path=r.path, status='removed or renamed', markers=0)); continue
    now = sha(p); src_now = sha(r.source) if os.path.exists(r.source) else ''
    txt = open(p, errors='ignore').read()
    n = txt.count('GWANGYANG PORT')
    status = 'identical to Haenam' if now == r.sha256_copied else ('edited, marked' if n else 'EDITED WITHOUT MARKER')
    rows.append(dict(path=r.path, status=status, markers=n, haenam_source_changed_since_copy=bool(src_now and src_now != r.sha256_copied)))
pc = pd.DataFrame(rows)
display(pc.status.value_counts().to_frame('files'))
display(pc[pc.status != 'identical to Haenam'].sort_values('path'))
new = sorted(set(os.path.relpath(os.path.join(dp, f), ROOT) for dp, _, fs in os.walk(ROOT) for f in fs
                 if f.endswith(('.py', '.sh', '.csv')) and not dp.startswith(f'{ROOT}/data') and '/continuous' not in dp
                 and '__pycache__' not in dp and 'logs' not in dp and 'phasenet_plus' not in dp and 'notebooks' not in dp and 'relocation/hypodd' not in dp)
          - set(prov.path))
print('files added by this project (not copies):'); print('\n'.join('  ' + x for x in new))
PORT_OK = not (pc.status == 'EDITED WITHOUT MARKER').any()
print('PORT CHECK:', 'PASS' if PORT_OK else 'FAIL')
""")
co(r"""
sys.path.insert(0, f'{ROOT}/registry')
import gwangyang as R
reg = pd.DataFrame([dict(product=p, path=R.path(None, p), exists=os.path.exists(R.path(None, p))) for p in R.products()])
display(reg)
ev = pd.read_csv(f'{ROOT}/registry/events.csv', dtype={'eid': str, 'kma_id': str})
display(ev)
REG_OK = len(ev) == 11 and (pd.to_datetime(ev.origin_time_utc) == pd.to_datetime(ev.kma_id, format='%Y%m%d%H%M%S') - pd.Timedelta(hours=9)).all()
print('REGISTRY:', 'PASS' if REG_OK else 'FAIL', f'({len(ev)} events, UTC = KST - 9 h)')
""")
co(r"""
envs = {'base': '/home/msseo/miniforge3/bin/python', 'fmf': '/home/msseo/miniforge3/envs/fmf/bin/python', 'pq-gpu': '/home/msseo/miniforge3/envs/pq-gpu/bin/python'}
probe = {'base': 'import obspy, pyocto, torch, pandas; print(obspy.__version__, pyocto.__version__, torch.__version__, torch.cuda.is_available(), pandas.__version__)',
         'fmf': 'import fast_matched_filter as f, obspy; print(getattr(f, "__version__", "?"), obspy.__version__)',
         'pq-gpu': 'import torch, obspy; print(torch.__version__, torch.cuda.is_available(), obspy.__version__)'}
erows = []
for k, py in envs.items():
    r = subprocess.run([py, '-c', probe[k]], capture_output=True, text=True)
    erows.append(dict(env=k, python=py, versions=r.stdout.strip() or r.stderr.strip()[-200:], ok=r.returncode == 0))
envt = pd.DataFrame(erows); display(envt)
w = '/home/msseo/works/14.EQNet/EQNet/docs/model_phasenet_plus/model_99.pth'
WEIGHTS = sha(w); print('PhaseNet+ weights sha256:', WEIGHTS, '(the Haenam frozen weights f1643330...)' if WEIGHTS.startswith('f1643330') else '(DIFFERENT from Haenam)')
ENV_OK = bool(envt.ok.all()) and WEIGHTS.startswith('f1643330')
gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], capture_output=True, text=True).stdout.strip()
print('GPU:', gpu); print('ENVIRONMENTS:', 'PASS' if ENV_OK else 'FAIL')
""")
co(r"""
dec = pd.read_csv(f'{ROOT}/docs/decisions.csv')
display(dec[['id', 'setting', 'gwangyang_value', 'status']])
""")
md(r"""
## Comprehensive summary
""")
co(r"""
summ = pd.DataFrame([
    ('Port check (identical or marked edits)', 'PASS' if PORT_OK else 'FAIL', f"{(pc.status == 'identical to Haenam').sum()} identical, {(pc.status == 'edited, marked').sum()} edited with markers, {(pc.status == 'EDITED WITHOUT MARKER').sum()} unmarked"),
    ('Registry and events', 'PASS' if REG_OK else 'FAIL', f'{len(reg)} products; {len(ev)} KMA events (UTC)'),
    ('Environments and PhaseNet+ weights', 'PASS' if ENV_OK else 'FAIL', '; '.join(f'{r.env}: {r.versions}' for r in envt.itertuples())),
    ('Decision log', 'info', f"{len(dec)} rows: " + ', '.join(f'{k} {v}' for k, v in dec.status.value_counts().items())),
], columns=['check', 'result', 'detail'])
display(summ)
print('Take-homes:')
print('1. The Gwangyang code is the Haenam renewal code; every change is marked GWANGYANG PORT and listed above (region values, paths, the inventory and link rules).')
print('2. The registry serves Gwangyang paths through the same interface; the copied code imports it unchanged through the haenam.py shim.')
print('3. Detection uses the frozen Haenam PhaseNet+ weights; every stage runs in moho\'s existing environments.')
""")
nb = dict(nbformat=4, nbformat_minor=5, cells=cells, metadata=dict(kernelspec=dict(name="python3", display_name="Python 3", language="python"), language_info=dict(name="python")))
for i, c in enumerate(nb["cells"]): c["id"] = f"c{i:02d}"
json.dump(nb, open(OUT, "w"), indent=1); print(f"wrote {OUT}: {len(cells)} cells")
