"""Repeater gathers for nb40 from the FINAL catalogue (full_period_5_20H, frozen 2026-10-06).

The nb40 gathers of 2026-09-29 were drawn while the scan was still running (2023 and part of 2020 in the catalogue), so
"every detection of the template" was incomplete. This redraws the same four gathers over the whole period from the frozen
catalogue with fmf/station_gather.py, unchanged. Nothing else is written. Run with ONLY FMF_RUN set:

    cd template_matching && FMF_RUN=full_period_5_20H /home/msseo/miniforge3/envs/fmf/bin/python nb40_final_gathers.py

Output: figs_similarity_full_period_5_20H/gathers_final/GATHER_<cuspid>_<STA>.png
"""
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
if os.environ.get('FMF_RUN') != 'full_period_5_20H':
    raise SystemExit('run with FMF_RUN=full_period_5_20H (and nothing else set)')
import config as C
from station_gather import station_gather

OUT = os.path.join(HERE, 'figs_similarity_full_period_5_20H', 'gathers_final')
GATHERS = [(400091, 'KS.JDO2'), (400091, 'KG.AJD'), (400177, 'KS.JDO2'), (400177, 'KG.AJD')]
T0, T1 = '2020-01-01', '2027-01-01'

if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    print('catalogue:', C.CATALOG_CSV)
    for cusp, sta in GATHERS:
        png = os.path.join(OUT, f'GATHER_{cusp}_{sta.split(".")[1]}.png')
        station_gather(cusp, sta, T0, T1, png, title=f'Template {cusp}: every detection in the final catalog at {sta.split(".")[1]}, aligned on the template’s P pick',
                       same_template=True)
        print('->', png)
