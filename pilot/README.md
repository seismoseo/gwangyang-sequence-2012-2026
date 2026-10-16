# Pilot (2026-10-15): NOT a production product

Detection, relocation and template matching on the two complete episode years (2021, 2026) while the MARA pull of the
other years runs. Purpose: find problems in G4-G7 early. The station table was not frozen yet (seven stations wait for the
G2 orientation measurements), so nothing here is used by the production stages; they re-run every year from scratch.

A sandbox: pilot/registry is a copy of registry/, so every registry path resolves inside pilot/ (ROOT = this folder);
data and docs are links to the production folders (read only here). Code copies: detection (MODEL_DIRS ->
detection/phasenet_plus_pilot_2021_2026, the only edit), relocation/code, template_matching, magnitudes, catalogs/templates.
