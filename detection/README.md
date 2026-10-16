# Haenam DL detection project (2020 → 2026)

Deep-learning detection → association → absolute location → template matching → fold-back into the
canonical relocation, for the Haenam earthquake sequences (2020-04/05 sequence, the 2020-06 → 2026-08
quiescence with one relocated event on 2022-06-11, and the 2026-08/09 sequence with the three
200 Hz JNU stations). **Nothing is re-invented**: this is the validated Buan chain
(`20.2024_Buan_EQ_DL_detection` + the precursor variant `21.2024_Buan_precursor`) and the Buan FMF
prototype (`/home/msseo/works/FMF`) with region values changed, plus one new glue step that hands
new events to PocketQuake's `--augment` so they join the existing 195-event relocation frame.

Everything is driven by `run_haenam.sh` (dry run by default; append `run`). You run it.

## Layout

```
detection/
  run_haenam.sh            driver: link | stations | detect | locate | catalog | vet | fmf-templates |
                                   fmf-scan | fmf-catalog | stage-augment      (dry run unless `run`)
  pipeline/                Buan pipeline, byte-identical except config.py (Haenam), HYP_DIS/HYP_ZTR
                           read from config, and sumio.py vendored (see "What changed vs Buan")
  station_table/           stations.csv (54 KS + 6 KG within 100 km, 7 JU within 100 km) + builder
  velocity_model/kim1983.csv, HypoInv/{kim1983,kim2011}/*.crh     copied from Buan (Kim 1983 crust)
  ../data/continuous/sds_links/   SYMLINK TREE in the Buan layout over the real archives (tools/link_continuous.py)
  ../catalogs/templates/Haenam_templates_195ev.csv   the 195 relocated events (JNU-augmented Haenam_joint) = FMF templates
  haenam_catalog.py      DL catalog with QC tiers + KMA/template cross-match
  tools/vet_candidate.py       record section of one candidate (move-out check)
  ../template_matching/    FMF port: config, templates, shelly (two-pass), scan, detect, segmad, build_catalog
  ../relocation/code/stage_for_augment.py      SAC + catalog for PocketQuake --augment (fold-back into the canonical frame)
  tools/link_continuous.py, tools/extract_jnu_all.sh
  models/, ../template_matching/catalogs/, ../template_matching/templates_cache/    outputs (git-ignored)
```

## Data (2026-09-08)

| source | stations | span on disk | layout (real files) |
|---|---|---|---|
| NECIS cron archive (`Claude/data/necis/continuous/2026`) | all 404 KS (54 within 100 km, HG/HH/EL 100 Hz) | 2026 DOY 1–3, 130–244, grows nightly | `YYYY/STA/KS.STA.CHA.YYYY.DOY.00.00.00` |
| JNU tars (`data/jnu/JN0{1,8,9}.tar`) | JN01 (9 km), JN08 (17 km), JN09 (4 km), HH 200 Hz | 2025-12-04 → 2026-08-31 04:09; **only DOY 226–243 extracted** | `STA/STA.JU.00.CHA.YYYY.DOY` |
| Earthworm (`~/EW/EW_JNU/data/mseed/JU`) | all 30 JU (7 within 100 km) | 2026 DOY 244 → today | same as tars |
| 2020–2025 continuous | **none on disk** (KG never) | — | — |

`tools/link_continuous.py --apply` exposes all of these as
`data/continuous/sds_links/NET/STA/CHA.D/NET.STA..CHA.D.YYYY.DOY` (25,062 links for 2026). Re-run it whenever an
archive grows; add a future bulk archive (the KMA hard-disk transfer of 2020–2025) as one more
`SOURCES` entry. `tools/extract_jnu_all.sh run` pulls the remaining JNU days (≈62 GB) for the
pre-sequence scan.

## Stage 1 — 2026 sequence (start here)

```bash
./run_haenam.sh link run                       # refresh the symlink tree
./run_haenam.sh detect 2026 213-251 run        # PhaseNet+ picks, GPU, ~3–5 min per day, resumable
./run_haenam.sh locate 2026 run                # PyOcto (relaxed-strict) -> augment -> PHS -> HYPOINVERSE
./run_haenam.sh catalog run                    # ../catalogs/Haenam_DL_catalog_2020-2026.csv
./run_haenam.sh vet "2026-08-14 05:34:13" x.png run    # eyeball any candidate
```
Then the pre-sequence quiescence of 2026 (`detect 2026 130-225`, needs the JNU extraction for
JNU coverage; KS alone works) and the FMF pass:
```bash
./run_haenam.sh fmf-templates run              # 118 buildable 2026 templates (2020 ones need 2020 data)
./run_haenam.sh fmf-scan 2026 130-251 run      # Shelly two-pass GPU scan (MAD_SEGMENTED=1 optional)
./run_haenam.sh fmf-catalog run                # 12xMAD + cc>=0.20 + >=4 stations, declustered, flagged
```
Fold-back of the new Haenam events into the canonical relocation (one frame, the validated chain):
```bash
./run_haenam.sh stage-augment                  # dry run: which tier-A/B in-box events would be staged
./run_haenam.sh stage-augment run              # STP-layout SAC + catalogs/haenam_augment_catalog.csv
cd /home/msseo/works/15.PocketQuake
./pocketquake.sh <the augment catalogue of stage_for_augment.py> Haenam_joint --augment --dry-run
./pocketquake.sh <the augment catalogue of stage_for_augment.py> Haenam_joint --augment --skip-download
/home/msseo/works/27.Haenam_sequence/relocation/jnu_augmentation/run_jnu_augment.sh run      # re-add JN01/08/09
```

### Standalone HypoDD relocation of the DL catalog (`dl-reloc`)

Besides the fold-back, `./run_haenam.sh dl-reloc run` relocates the whole QC-passing in-box DL catalog
on its own, exactly as the Buan project did: `../relocation/code/run_dl_hypodd.py` builds the catalog from the
`.sum` (gap<180°, N≥8, RMS<1 s, errors<5 km, inside `HAENAM_BOX`), cuts per-event SAC from the
continuous archive with the PyOcto picks (`../relocation/code/event_sac_export.py`, KS 100 Hz and JU 200 Hz at
native rate), and enters korea-cluster-relocation at its `hypoinverse` stage (hyp1.40 on the
subset → ph2dt → dt.ct → re-reference → GPU cross-correlation → dt.cc with adaptive damping →
1,000-replica bootstrap). `../relocation/code/haenam_dl_cluster_config.py` mirrors the canonical Haenam_joint
settings (kim1983, STD_HYP, OBSCC 4, LSQR, 1000 Hz / 5–20 Hz / ±0.5 s / CC ≥ 0.7, onset window
±0.05 s 1–40 Hz for the twelve 2026 M ≥ 2 events, mainshock = Event 15). Outputs:
the run root's `2.HypoDD/02.dt.cc/final_locations.csv` (PocketQuake schema; the 2026-only run, registry/runs.csv
hypodd_dl_2026only_481ev, is kept in `../relocation/provenance/hypodd_dl_2026only_481ev`) and
`../catalogs/supplementary/Haenam_relocated_2026only_481ev.csv` (+ template match, template ML, HYPOINVERSE quality columns).
Testing knobs: `--after 2026-08-14 --max-events 25 --n-boot 50 --run-root <scratch>`.
**Pair order matters on this disk.** The framework enumerates all pairs lexicographically, so each row's
working set is every remaining event; with the 20,000-trace in-process cache (about 109 events) the
cache thrashes and most of the 145 GB interpolation cache is re-read per row: measured 1,400 pairs per
hour on the 2026-only event set (62 h). The driver therefore orders pairs block-wise (`--blocked-pairs 45`, a
driver-local monkeypatch of the pair generator; identical pair set) and writes per-pair files every
`--xcorr-chunk 4000` pairs, which also gives visible progress and pair-level resume (`--xcorr-resume`).
**dt.cc schedule: open cc distance cutoff (`--open-wdcc`).** The canonical 7-set schedule tightens the cc
inter-event distance cutoff to 4/2/1/0.5 km in the cc-dominant sets. On the 2026-only event set that stranded
49 events (two Aug 14 foreshocks among them) with `nccp = nccs = 0`: HypoDD starts every event at the
centroid (ISTART 1), the catalog-dominated sets 1–3 move an event with one bad far-station DL pick 1–3 km
off the core, and the cutoff (plus the residual cut) then drops its cc data. With the cutoff removed
(WDCC −9 in the four cc sets, everything else unchanged) the well-linked events move 1 m (median) / 3 m
(95th percentile) and all 49 regain cc links, so the delivered solution uses `--open-wdcc`; the canonical-
schedule solution of the 2026-only run was compared in notebook 05 (it was not kept when that run was reduced
to its provenance files, `../relocation/provenance/hypodd_dl_2026only_481ev`). Removing the residual cut instead (WRCC −9 in the first cc sets) was tested and rejected:
it perturbs the well-linked events by 17 m / 250 m and still leaves the two foreshocks S-only.
`--bootstrap-only` reruns just the bootstrap + catalog writing on the existing dt.cc solution. `--years 2020-2026`
relocates the merged multi-year catalog jointly (one `.sum` per year, per-year station/PyOcto tables resolved by the
exporter; onset windows for the 17 canonical M ≥ 2 events of 2020 and 2026); the final joint run is
hypodd_dl_2020-2026 (registry/runs.csv, `../relocation/hypodd_dl_2020-2026`), whose catalogue is
`../catalogs/Haenam_relocated_2020-2026_802ev.csv`.
**Independent position check: JN01 S–P (`../relocation/code/sp_consistency.py`).** For every event with P and S
picks at JN01 (427 of 481) the observed change of S–P relative to Event 05 (the first KMA event) is compared with the change
predicted by the relocated hypocenter (0.114 s per km at 20 km depth in kim1983). A residual beyond 100 ms
(about 0.9 km) means the relocation contradicts the picks at the best near station. The script adds
`sp_obs_s`, `dsp_obs_ms`, `dsp_pred_ms`, `sp_resid_ms`, `sp_consistent` to `../catalogs/supplementary/Haenam_relocated_2026only_481ev.csv`
(run it after `dl-reloc`). Result on the delivered solution: 44 events relocated 1–8 km from the core have
S–P times that put them AT the core (catalog-pick artefacts that the catalog-dominated HypoDD sets locked
in); the bootstrap flags 43 of them as under-constrained. Notebook 05 therefore uses a *trusted* gate
(≥ 20 cc links, not under-constrained, S–P consistent where checkable): 392 events, robust SVD plane
96/66 with 7 m plane-normal rms, matching the canonical 2026 relocation (97/67, 6 m) to 3 m medians.
**cc-first weighting (tested, not adopted).** Because the mislocations come from the catalog-dominated
sets 1–3, two variants were solved on the same dt files in the scratchpad (88 s each): equal cc/catalog
weights in sets 1–3, and cc-dominant from set 1. Both halve the S–P-inconsistent events (44 → 28/29),
bring the two lopsided-link foreshocks to within 200 m of Event 05 with thousands of S links, and move
the near-core events by 8 m (median) / 120–150 m (95th percentile). Against the independent canonical
relocation they are slightly worse (E/N/Z median residuals 3.1/2.0/6.3 m, 90th percentile 21 m, versus
2.8/1.8/3.2 m and 12 m for the delivered catalog-first schedule), so the delivered catalog keeps the
SOTA-matching schedule and carries the `sp_consistent` flag instead. To switch, edit the three
catalog-dominated rows' cc weights in `../relocation/code/haenam_dl_cluster_config.py` and re-run
`dl-reloc --stage-from dtcc --open-wdcc` (22 min + 1.7 h bootstrap).
**Notebooks on the relocation.** `05_relocation_results` (diagnostics: schedule check, S–P checks, errors),
`06_relocation_figures` (constraint classes resolved/unresolved/weak; canonical-style figures on the frozen
195-event plane about Event 03), `07_dl_sequence_analyses` (the magnitude-free analyses of
`figures` and `analysis/source_sequence` — nb15 across-strike thickness, F07/nb09 diffusion front, nb09
centroid path and Theil–Sen drift, nb08 on-fault density versus the slip models, nb11 snapshots — repeated
with the DL catalog next to the canonical 2026 events). `../relocation/code/dl_reloc_frame.py` writes
`../catalogs/supplementary/Haenam_relocated_2026only_canonical_frame.csv`, the DL catalog in the canonical frame with the suite's
`load_reloc()` columns (`load_reloc_dl()`); 277 of the 392 resolved events have no magnitude, so the
moment-based suite analyses (nb09 Mc/b/ETAS/stress, nb17, nb23, F08, F09) are not repeated.

## Stage 2 — trace back 2020 → 2026 (full-period run from MARA)

Decided 2026-09-17 (plan in `../docs/history/detection/full_period_launch_plan.md`): the continuous data for 2020–2026 come from the
MARA SDS archive (`<user>@<mara-host>:<mara-sds-root>/`), mirrored by
`tools/mirror_mara.py` into `/home/msseo/works/27.Haenam_sequence/data/continuous/KS_KG_mara_sds/` with the pipeline's own channel
policy applied at fetch time (one sensor per station-day; KS 2020–2026, KG 2020–2024 — KG has no 2025/2026 data
anywhere). `tools/link_continuous.py` links the mirror first (then NECIS, JNU tars, Earthworm). The 2026 run is
re-detected from the mirror so that the whole period uses one station rule (KS/KG within 100 km whenever they have
data, JU whenever present); this also fills the KS hole DOY 004–129 of 2026, during which only three JU stations
recorded and no event (e.g. 2026-04-29 20:12 UTC) could pass the relocation gate. Driver stages: `mirror list|plan|
pull|verify|status`, `epochs`, `years Y0-Y1` (gate → link → parameter hash → detect → locate → status), `status`.
Old text of this section follows.


Blocked only by data: no continuous waveforms before 2026 exist on this machine. Options: the KMA
hard-disk transfer (Sangwoo, ~2 years), `Claude/archive_necis.sh START END` for a date range through
NECIS (≈13 h of quota per calendar day for all KS; restrict with `STATION_CSV` to the 18 stations
within 50 km), or the MARA server. Once a year is linked: `detect YEAR 1-366`, `locate YEAR`,
`catalog`, `fmf-templates` (now including the 2020 templates), `fmf-scan YEAR 1-366`, `fmf-catalog`.
For the quiescence years, the FMF pass is the sensitive tool (templates from both sequences).

## What changed vs Buan (and why)

* `config.py`: paths; `REGION_NAME=HAENAM`; `CHANNEL_POLICY` KS/KG `HH>EL>HG`, JU `HH`; PyOcto box
  33.95–35.35 N / 125.60–127.20 E; `--strict` = the Buan **precursor** relaxed-strict set (3 P + 3 S,
  both phases at ≥2 stations, `pick_match_tolerance=1.0 s`) because Haenam's near field is 3 JNU +
  HAWA + 4 KS at 16–22 km, not a dense array; the dense 8P+8S gate is kept as `REGION_STRICT_DENSE`.
  QC tiers A/B as in the precursor project; `HAENAM_BOX` for the fold-back selection.
* `core.py`: only the HYPOINVERSE `DIS`/`ZTR` lines now come from config —
  `DIS 4 50 1 3` (the korea-cluster-relocation `STD_HYP` used for the canonical Haenam absolute
  locations; Buan used 25 km for its 10-km arrays) and `ZTR 10 F`.
* `sumio.py`: `read_sum`/`apply_qc` vendored verbatim from the Ulsan `uf_cluster` — Buan's import
  path (`02.Ulsan_Fault_detection/KS_KG/HypoInv`) no longer exists.
* `core.py` (second, data-driven change): near-integer sampling rates are rounded before `merge`
  in both readers. NECIS KS day files mix 99.99999237 Hz and 100.0 Hz segments (sampled scan of the
  2026 tree: 82 of 3,581 files; JDO2 on about a third of its days, MJDB, SIJA, NLDA, YEGA) and obspy
  otherwise refuses to merge them, silently losing the station-day. The same guard is in the FMF
  readers and the augment stager.
* `pick_augmentation.py`: `apply_safeguards` now returns the (accepted, diagnostics) pair when there
  are no orphan-pick candidates; the Buan original returned a lone DataFrame there and crashed the
  augment stage (never triggered on Buan, triggered on the Haenam smoke day). Worth back-porting to
  `20.2024_Buan_EQ_DL_detection` and `21.2024_Buan_precursor`.
* FMF: templates come from ONE csv (`haenam_templates.csv`) and TWO SAC roots (KS/KG 100 Hz +
  JU 200 Hz bundles); templates whose origin day has no continuous data are dropped loudly; the scan
  takes an explicit YEAR; each detection carries `n_present`; the CC floor (0.20) and the ≥4-station
  gate that the Buan notebooks applied live in `../template_matching/build_catalog.py`.

## Verification (scratch copy, not in this tree)

One-day smoke test of the whole ported chain on 2026-08-14 (DOY 226) with the 17 stations within
50 km (14 KS + 3 JU): 5,029 picks → 68 located events → **all 18 relocated events of that day
recovered** (median offset from the relocation 0.12 km horizontal, 0.3 km depth) plus 45 further
in-box events (most tier A/B at ~20 km depth, the sequence's small events; KMA lists none of them).
`haenam_catalog.py` and `../relocation/code/stage_for_augment.py` (dry run) were exercised on that output.
FMF: pick extraction from the SAC bundles (170 picks / 57 stations / 3 JU for Event 05), continuous
band prefixes, one JN09 day processed (200 → 100 Hz), and the GPU library load were checked in the
`fmf` env; no scan was run. The mixed-sampling-rate quirk above was found on KS.SIJA during this
test and fixed in all readers; `logs/mixed_rate_scan_2026.txt` holds the sampled archive scan.

## Results and history

The results of this pipeline are delivered by the notebooks in `../analysis/seismicity/` (registry/notebooks.csv;
`../analysis/00_index.ipynb`), with every count computed there. The dated working notes written after each stage
(2026-09-15 to 2026-10-06) are kept verbatim in `../docs/history/detection/README_journal_20260915-20261006.md`.
