# GWANGYANG PORT: compatibility shim. The code copied from the Haenam renewal tree imports `haenam` and finds this registry folder by
# looking upward for registry/haenam.py; in this project that name serves the Gwangyang registry (registry/gwangyang.py).
# The original Haenam module is recorded in PROVENANCE.tsv (registry/haenam_haenam_original.py keeps a copy for reference).
from gwangyang import *  # noqa: F401,F403
from gwangyang import ROOT, WORKS, path, gpath, key, eid, keys, reload  # noqa: F401
