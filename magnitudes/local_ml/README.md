# Haenam ML derived with the current Buan method

The notebook that documented this product (methodology, reference frames, first-event
origins and provisional location screening) is kept as a record in
`../../docs/history/external_pre_cleanup/Claude/haenam_ml/11_local_magnitudes_and_origins.ipynb`
(written before the October 2026 restructure; its paths name the folders of that time).

Primary catalog: `haenam_local_magnitudes.csv`, one row per relocated event.
`ml` uses corrected theoretical-P noise timing; `ml_buan_exact` reproduces the
original Buan executable including its double pre-roll in that fallback path.
The unchanged source snapshot is `buan_sheen2018_snapshot.py`. No KMA magnitude
enters amplitude measurement. Existing Haenam catalogs are not overwritten.

Measurement policy follows the executable Buan engine: calibrated KS/KG traces,
Sheen coefficients, station terms zero, peak SNR >=2, horizontal preferred with
vertical fallback, median station ML, >=3 measured stations with picked P.
`ml_h_strict` is a separate horizontal-only result requiring >=3 picked H stations.
The KS/KG restriction is deliberate for this reproduction. Haenam JU metadata exist
but are not used in this calculation. Station codes, not independent physical
sites, are counted, matching Buan; nearby/co-located instruments can share errors.
The station scatter is not a standard error or full calibration uncertainty.

Every relocated event remains in the CSV. NaN indicates the method cannot report
an ML with its available calibrated, picked measurements; it is not ML=0 or proof
of a false detection. The primary result covers 112/481 events, including 111/392
currently resolved August events. 94 have an existing KMA match and measured ML.

`haenam_ml_per_station.csv` has accepted station measurements for both timing modes.
`haenam_ml_per_component.csv` includes response availability, window times, SNRs and
rejected components. Original SAC files are read in place; no waveforms are copied.
`responses_buan_subset.xml` is the selected existing Buan response inventory.
`provenance_ml.json` records code/metadata/catalog hashes and the definition of the
waveform file-stat signature. `validation_buan_exact.csv` contains direct original
engine comparisons. No response epochs are extended or station gains refitted.

For a complete remeasurement, in the base Python environment with ObsPy:

    python haenam_magnitudes.py --out . --workers 6

The notebook uses saved measurements by default; set RECOMPUTE_ML=True to remeasure.
The verified run used six workers with BLAS/OpenMP threads limited to one and a
lower scheduling priority. No detection, cross-correlation or relocation is run.
Clear `event_cache` or increment the processing VERSION before changing processing
settings. The exact Buan model/response originals remain read-only dependencies.
