"""Gwangyang project registry (73.Gwangyang_sequence): one place for the project root, product paths and the KMA-listed events.

The same interface as the Haenam renewal registry (27.Haenam_sequence_renewal/registry/haenam.py) for global products:
    gwangyang.ROOT, gwangyang.WORKS
    gwangyang.path(None, "dl_continuous")   -> a global product (layout.csv)
    gwangyang.keys(), key(x), eid(k)        -> the KMA-listed sequence events (events.csv; UTC)
There are no per-event products here; path(k, ...) with an event raises KeyError.
"""
import csv
import os
import re
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("GWANGYANG_ROOT", str(HERE.parent)))
WORKS = Path(os.environ.get("GWANGYANG_WORKS", "/home/msseo/works"))
_TOKEN = re.compile(r"\{([A-Za-z0-9_]+)\}")


def _read(name):
    with open(HERE / name, newline="") as fh:
        return list(csv.DictReader(fh))


@lru_cache(maxsize=None)
def _layout():
    return {r["product"]: r for r in _read("layout.csv")}


@lru_cache(maxsize=None)
def _events():
    return {r["key"]: r for r in _read("events.csv")}


def reload():
    for f in (_layout, _events):
        f.cache_clear()


def keys():
    return list(_events())


def key(x):
    x = str(x).strip()
    for k, r in _events().items():
        if x in (k, r["eid"], r["kma_id"]):
            return k
    raise KeyError(f"unknown Gwangyang event id or key: {x!r}")


def eid(k):
    return _events()[key(k)]["eid"]


def products():
    return list(_layout())


def _expand(template, depth=0):
    if depth > 8:
        raise KeyError("layout template recursion")

    def tok(m):
        t = m.group(1)
        if t == "ROOT":
            return str(ROOT)
        if t == "WORKS":
            return str(WORKS)
        if t in _layout():
            return _expand(_layout()[t]["template"], depth + 1)
        raise KeyError(f"no value for {{{t}}}")
    return _TOKEN.sub(tok, template)


def path(k, product, variant=None):
    """Path (str) of a global product (k must be None)."""
    if k is not None:
        raise KeyError(f"{product!r}: the Gwangyang registry has no per-event products")
    r = _layout().get(product)
    if r is None:
        raise KeyError(f"unknown product {product!r}")
    return _expand(r["template"])


def gpath(product):
    return path(None, product)
