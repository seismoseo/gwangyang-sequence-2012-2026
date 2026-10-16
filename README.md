# Gwangyang sequence, 2012-12-20 to 2026-10-06

The Gwangyang earthquakes of 2021 (6 KMA events, ML 1.1-2.2) and 2026 (5 KMA events) re-studied from the start of KS.GWYB
broadband data, with the stages of the Imsil study and the current Haenam orientation rules: one sensor per station, one
dated orientation table, S differential times on T, template N/E turned into each scanned day's orientation, ML without
station terms. Code is copied from the Haenam renewal tree (`27.Haenam_sequence_renewal`); `PROVENANCE.tsv` lists every
copied file with its source and checksum, and every edit is marked `GWANGYANG PORT`.

Settings and their evidence: `docs/decisions.csv` (one row per decision; `G-D*` are this study's).

## Layout

| Folder | What |
|---|---|
| `registry/` | `gwangyang.py` (paths by product name, `layout.csv`), `haenam.py` (shim for the copied code), `orientation.py` (dated orientation table), `events.csv` (11 KMA events, UTC), `streams.csv` |
| `data/` | catalogs, MARA listings, `continuous/` (MARA mirror, KIGAM fill, NECIS re-fetches, the link tree `sds_links`), `stations/` (responses, orientation measurements, TauP model) |
| `docs/` | `station_inventory/` (sensor choice, periods, response epochs), `response_inventory/` (merged StationXML), `decisions.csv` |
| `detection/` | PhaseNet+ -> PyOcto -> HYPOINVERSE (frozen Haenam settings), station table, tools |
| `relocation/` | HypoDD with dt.ct + dt.cc, S on T, 1000 bootstrap replicas; 25 m adoption |
| `magnitudes/` | Sheen (2018) ML, merged responses, no station terms |
| `template_matching/` | FMF GPU scan with turned templates |
| `catalogs/` | adopted products (25 m reference, templates) |
| `notebooks/` | builder scripts and executed notebooks, one per stage |
| `pilot/` | 2021 + 2026 trial runs while the pull ran; NOT used by any production stage |

## Run order (each script is a dry run unless given `run` / `--apply`)

1. **G1 data.** `detection/tools/mirror_mara_streams.py pull --apply --jobs 8` (MARA, about 6 TB);
   `detection/tools/fetch_kigam_fdsn.py plan --apply` then `pull --apply` (KG after 2024-10-31);
   `detection/tools/audit_components.py --years ... --apply` then `detection/tools/necis_refetch_episodes.sh run`
   (components missing on MARA in the episodes); `detection/tools/link_continuous.py --apply --refresh`, then `--check`.
2. **G2 orientation.** `detection/tools/r1c_orientation.py cut --group gwangyang --workers 8`, `measure --workers 8`,
   `periods`; then `docs/station_inventory/build_station_inventory.py`, `detection/station_table/build_stations_renewal.py --apply`,
   `detection/station_table/add_link_until.py --apply`, link again.
3. **G4 detection.** `detection/run_renewal_detection.sh freeze`, then `years`; `detection/haenam_catalog.py`.
4. **G5 relocation** (env pq-gpu; never with detection or template matching on the GPU):
   `relocation/code/run_dl_hypodd.py --years 2012-2026 --radius-region --min-depth 10 --open-wdcc --s-on-t --fast-xcorr --n-boot 1000 --run-root relocation/hypodd_dl_2012-2026`;
   `relocation/code/adopt_gwangyang.py --run-root relocation/hypodd_dl_2012-2026 --apply`.
5. **G6 magnitudes.** `docs/response_inventory/build_response_inventory.py`;
   `magnitudes/local_ml/full_period/full_magnitudes.py --out <dir>`.
6. **G7 template matching** (env fmf): `catalogs/templates/build_template_catalog.py --run-root relocation/hypodd_dl_2012-2026 --apply`;
   `template_matching/make_station_list.py --apply`; `template_matching/run_G7.sh templates`, `check YEAR DOY`, `batch`, `catalog`.
7. **G8** joint relocation of the detections; **G9** analysis and report.

## Traps

- **Smoke-test picks in the production folder.** G3 wrote 2021.239-240, 2026.001 and 2013.200 into
  `detection/phasenet_plus_2012-2026`. Detection skips days whose picks exist, so move them aside before the G4 run.
- **KG after 2024-10-31** is not on MARA; it comes from KIGAM FDSN through an SSH tunnel to PAGO (the key is registered
  for PAGO's address, not moho's).
- **Missing components on MARA.** KMA's source archive lacks Z on some days (2021-08-27/28 at several stations); NECIS
  sometimes has them (`data/continuous/necis_refetch`, the link tool's first source). NECIS had none for BOSB and GMNA on
  2021-08-27 or for MALA on 2021-07-10. HWJA 2026-01-03..05-09 is a station outage.
- **20-sps SL channels** (YSU before 2015 and others) carry no response and cannot hold the 5-20 Hz template band; unused (G-D12).
- **Haenam `adopt_r4.py`** (2026-10-15) has `--apply` inside a comment; this tree uses `relocation/code/adopt_gwangyang.py`.
- **Registry names kept from Haenam.** Some products keep Haenam names so the copied code runs unchanged
  (`tm_templates_195` = the Gwangyang template table; `wf_dl_2026only_481ev` = the G5 exports, replaced at run time).
- **moho is shared.** Use `nice`, at most one GPU stage at a time, and `prlimit` on memory-heavy steps.

## Status (2026-10-16)

- G0, G1, G2, G3 done; notebooks 00, 01, 02. Detection parameters frozen (hash d724acfddcd608dd, 12:51); smoke-test outputs in detection/smoke_G3_20261015.
- G4 production detection waits for mohos GPU: the Haenam renewal R5 template-matching scan holds it until about 2026-10-19. Never start a GPU stage beside another projects GPU job.
- pilot/ (2021 + 2026 trial) is stopped after both episodes were picked; its location is done; its relocation also waits for the GPU.

## Repository

This repository holds the code, the executed notebooks (with their figures), the decision log, the station, orientation and response tables, and the reports. It does not hold the continuous waveforms (about 7 TB), the third-party catalogs and metadata, the run outputs, the logs or the pilot outputs. The `.gitignore` lists what is left out.

Server addresses and the MARA archive path are replaced by placeholders (`<moho-host>`, `<mara-host>`, `<mara-sds-root>`, ...) in the committed files by a git clean filter. On the server, the working files keep the real values. To work in a clone:

    git config filter.redact.clean "sed -E -f <path>/gwangyang-redact.sed"
    git config filter.redact.smudge cat
