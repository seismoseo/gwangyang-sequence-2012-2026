# The Gwangyang earthquake sequence, 2012–2026: study plan

First issued 15 October 2026; updated 16 October 2026 with the final data and orientation results and the pilot results.

**`report.pdf`** covers:
- the questions, the sequence and earlier work;
- the approach (Imsil stages with the current Haenam orientation rules);
- the stations and data, and the completeness of the 2021 episode data;
- sensor orientation;
- the ten processing stages with settings and checks;
- **the pilot results on the episode days** (section 7): locations with HYPOINVERSE errors and error ellipses, waveforms, the S−P and waveform tests, relative magnitudes, timing;
- the independent checks, the status and schedule, risks and deliverables;
- appendix A, the decision log, and appendix B, the 33 pilot events.

The status numbers are a snapshot taken when notebook 90 ran.

**`figures/`** holds the figures as `Figure_NN_<topic>.pdf` and `.png` (300 dpi):
- Figures 1–6 are drawn by `notebooks/90_study_plan_report.ipynb`; `figures/sources.json` lists their input files with checksums.
- Figures 7–12 (`*_pilot_*`) are drawn by `notebooks/91_pilot_episode_events.ipynb`.

## How every number gets into the text

No number in `report.tex` is typed by hand.
- Notebook 90 writes `facts.json` and `facts.tex`, one macro `\F{key}` per number. It also writes `tab_decisions.tex` from `docs/decisions.csv`.
- Notebook 91 writes `facts_pilot.json` and `facts_pilot.tex` (keys `pilot.*`) and `tab_pilot_events.tex`.
- A fact is computed from the project tables, read from the frozen settings files (with assertions), or quoted from the executed output of notebooks 00 and 02 with a pattern that must match exactly one value.
- An undefined key stops the LaTeX run.
- The schedule estimates in section 9 (dates and durations) are plain text.

## Rebuild

Run notebook 90 first, because it defines the `\F` macro:

    /home/msseo/miniforge3/bin/python notebooks/build_90_study_plan_report.py
    jupyter execute --inplace --kernel_name=python3 notebooks/90_study_plan_report.ipynb    # ~1 min
    /home/msseo/miniforge3/bin/python notebooks/build_91_pilot_episode_events.py
    jupyter execute --inplace --kernel_name=python3 notebooks/91_pilot_episode_events.ipynb  # ~3 min
    /home/msseo/miniforge3/envs/tex/bin/tectonic -X compile report.tex                    # in this folder

The "report.bbl changed / stopping at 6 passes" message from tectonic is harmless.

`refs.bib`: eight entries copied from the Imsil summary report, which copied them from the Haenam summary report. Seo et al. (2022) is taken from its Crossref record.
