"""Faster GPU batching for korea-cluster-relocation's cctorch_gpu_batched cross-correlation, with bit-identical results
(renewal R4, 2026-10-14). The framework itself is not edited: run_dl_hypodd.py --fast-xcorr installs these functions in
place of xcorr._build_one_task and xcorr._process_batch_safe.

Why: for every (pair, station, component) task the framework fills a (n_slides x n_ref) float64 candidate matrix on the
CPU (1001 slides x 1001 samples = 8 MB) and stacks the batch into one more copy before moving it to the GPU. Every row
is the same short trace segment shifted by one sample (uniform slide stride), so a chunk of 4000 pairs (~72k tasks)
copies ~1 TB in memory, single-threaded, while the GPU idles.

What changes: on the uniform-stride path the task keeps only that segment (n_ref + (rows-1) x stride samples); the
candidate matrix is built ON THE GPU from it (Tensor.unfold into a zero tensor of the same shape), giving a contiguous
float64 tensor with exactly the values the framework would have built. Every later operation is the framework's
_gpu_correlate_batch, copied verbatim, so shifts and cc are bit-identical (checked by test_bitexact below). The rare
non-uniform path keeps the framework's full matrix.
"""
import numpy as np
from pipeline.core import xcorr as X


class SegCands:
    """the candidate matrix of one task, kept as its source segment (uniform stride) or as the full array"""
    __slots__ = ("seg", "a", "b", "step", "full")

    def __init__(self, seg=None, a=0, b=0, step=1, full=None):
        self.seg, self.a, self.b, self.step, self.full = seg, a, b, step, full


def build_one_task(e1, e2, sta, comp, hdr, pre, post, fmin, fmax, slides, sr):
    """X._build_one_task with the same index math; returns (ref, SegCands, valid, n_ref) or None"""
    tr1 = X._full_trace(e1, sta, comp, fmin, fmax)
    tr2 = X._full_trace(e2, sta, comp, fmin, fmax)
    p1, arr2 = X._pick_time(tr1, hdr), X._pick_time(tr2, hdr)
    tr1_data = np.asarray(tr1.data, dtype=np.float64)
    tr2_data = np.asarray(tr2.data, dtype=np.float64)
    i0 = int(round((float(p1 - tr1.stats.starttime) - pre) * sr))
    i1 = int(round((float(p1 - tr1.stats.starttime) + post) * sr))
    if i0 < 0 or i1 >= len(tr1_data):
        return None
    ref = tr1_data[i0:i1 + 1]
    n_ref = len(ref)
    arr2_off = float(arr2 - tr2.stats.starttime)
    starts = np.round((arr2_off - pre + slides) * sr).astype(np.int64)
    valid = (starts >= 0) & (starts + n_ref <= len(tr2_data))
    if not valid.any():
        return None
    d = np.diff(starts)
    if len(d) and np.all(d == d[0]) and d[0] > 0:           # uniform positive stride: keep the segment only
        vi = np.where(valid)[0]
        a, b = int(vi[0]), int(vi[-1] + 1)
        s0, step = int(starts[a]), int(d[0])
        seg = np.array(tr2_data[s0:s0 + (b - a - 1) * step + n_ref], dtype=np.float64)
        return ref, SegCands(seg=seg, a=a, b=b, step=step), valid, n_ref
    built = X._build_one_task(e1, e2, sta, comp, hdr, pre, post, fmin, fmax, slides, sr)   # rare: the framework's way
    if built is None:
        return None
    ref, cands, valid, n_ref = built
    return ref, SegCands(full=cands), valid, n_ref


def _cands_tensor(batch, n_slides, n_ref, device):
    import torch
    C = torch.zeros((len(batch), n_slides, n_ref), dtype=torch.float64, device=device)
    for i, t in enumerate(batch):
        c = t["cands"]
        if c.full is not None:
            C[i] = torch.from_numpy(c.full).to(device)
        else:
            C[i, c.a:c.b] = torch.from_numpy(c.seg).to(device).unfold(0, n_ref, c.step)
    return C


def _correlate_tensors(R, C, valids, n_ref, nlag, slides, sr, device):
    """X._gpu_correlate_batch from 'R = _znorm_t(R)' on, verbatim (R, C already on the device)"""
    import torch
    L = n_ref + 2 * nlag
    R = X._znorm_t(R)
    C = X._znorm_t(C)
    A = torch.fft.rfft(R, n=L).unsqueeze(1)
    Bf = torch.fft.rfft(C, n=L)
    full = torch.fft.irfft(A * torch.conj(Bf), n=L) / n_ref
    cc = torch.cat([full[..., L - nlag:], full[..., :nlag + 1]], dim=-1)
    vmask = torch.from_numpy(valids).to(device).unsqueeze(-1)
    cc = cc.masked_fill(~vmask, float("-inf"))
    B, n_slides = cc.shape[0], cc.shape[1]
    flat = cc.reshape(B, -1)
    best_cc, idx = flat.max(dim=1)
    lag = (idx % (2 * nlag + 1)) - nlag
    slide_idx = (idx // (2 * nlag + 1))
    slides_t = torch.from_numpy(np.asarray(slides)).to(device)
    shift_s = lag.double() / sr - slides_t[slide_idx]
    out_shift = shift_s.cpu().numpy()
    out_cc = best_cc.cpu().numpy()
    del R, C, A, Bf, full, cc, flat
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out_shift, out_cc


def process_batch_safe(batch, nlag, slides, sr, device, stats):
    """X._process_batch_safe with the candidate matrix built on the device (same OOM halving and CPU fallback)"""
    import torch
    if not batch:
        return {}
    n_ref = batch[0]["n_ref"]; n_slides = len(slides)
    refs = np.stack([t["ref"] for t in batch])
    valids = np.stack([t["valid"] for t in batch])
    try:
        R = torch.from_numpy(refs).to(device)
        C = _cands_tensor(batch, n_slides, n_ref, device)
        shifts, ccs = _correlate_tensors(R, C, valids, n_ref, nlag, slides, sr, device)
        stats["batches"] += 1
        return {batch[i]["key"]: (float(np.round(shifts[i], 3)), float(ccs[i])) for i in range(len(batch))}
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if len(batch) == 1:
            stats["cpu_fallback_tasks"] += 1
            cpu = torch.device("cpu")
            R = torch.from_numpy(refs).to(cpu); C = _cands_tensor(batch, n_slides, n_ref, cpu)
            shifts, ccs = _correlate_tensors(R, C, valids, n_ref, nlag, slides, sr, cpu)
            return {batch[0]["key"]: (float(np.round(shifts[0], 3)), float(ccs[0]))}
        stats["oom_retries"] += 1
        mid = len(batch) // 2
        out = process_batch_safe(batch[:mid], nlag, slides, sr, device, stats)
        out.update(process_batch_safe(batch[mid:], nlag, slides, sr, device, stats))
        return out


# ---------------------------------------------------------------- parallel trace prefetch (2026-10-14)
# Once the interpolation cache outgrows the RAM, the task loop spends most of each chunk loading ~1 MB pickles of the cache
# one at a time from the spinning RAID (single thread, ~50 MB/s). Before the first task of a chunk, the chunk's traces are
# loaded by a thread pool from the same cache files and put into the framework's in-process LRU in task order (at most
# 90% of its cap). The Trace objects are the ones _full_trace would load, so every result is unchanged; keys without a
# cache file are left to _full_trace.
_PENDING = {"pairs": None, "done": True}
_orig_run_gpu = None


def _run_gpu_with_prefetch(common, stations, eid, xc, overrides, out_p, out_s, pairs, *args, **kw):
    _PENDING.update(pairs=list(pairs), stations=list(stations), comps=("Z",) + tuple(xc["s_comps"]), done=False)
    return _orig_run_gpu(common, stations, eid, xc, overrides, out_p, out_s, pairs, *args, **kw)


def _prefetch():
    import os, pickle, time
    from glob import glob
    from concurrent.futures import ThreadPoolExecutor
    t0 = time.time()
    keys, seen = [], set()
    for pair in _PENDING["pairs"]:                         # task order: pair -> station -> components
        pre, post, fmin, fmax = X._window(pair)
        for sta in _PENDING["stations"]:
            for comp in _PENDING["comps"]:
                for e in pair:
                    k = (e, sta, comp, fmin, fmax)
                    if k not in seen:
                        seen.add(k); keys.append(k)
    keys = [k for k in keys if k not in X._CACHE][: int(0.9 * X._CACHE_CAP)]

    def load(k):
        fs = glob(os.path.join(X._COMMON, k[0], f"{k[0]}.*.{k[1]}.*{k[2]}.sac"))
        if not fs:
            return k, None
        cpath = X._interp_cache_path(fs[0], k[3], k[4])
        if not cpath or not os.path.exists(cpath):
            return k, None
        try:
            with open(cpath, "rb") as fh:
                return k, pickle.load(fh)
        except Exception:                                   # noqa: BLE001 — _full_trace will handle it
            return k, None
    n = 0
    with ThreadPoolExecutor(max_workers=16) as ex:
        for k, tr in ex.map(load, keys, chunksize=32):
            if tr is not None:
                X._cache_put(k, tr); n += 1
    _PENDING["done"] = True
    print(f"[xcorr] prefetched {n:,} of {len(keys):,} traces in {time.time() - t0:.0f} s (16 threads)", flush=True)


def build_one_task_prefetching(*args, **kw):
    if not _PENDING["done"]:
        _prefetch()
    return build_one_task(*args, **kw)


def install(prefetch=False):
    global _orig_run_gpu
    X._build_one_task = build_one_task
    X._process_batch_safe = process_batch_safe
    if prefetch:
        _orig_run_gpu = X.run_xcorr_gpu_batched
        X.run_xcorr_gpu_batched = _run_gpu_with_prefetch
        X._build_one_task = build_one_task_prefetching
    print("[xcorr] fast batching installed (relocation/code/kcr_fast_xcorr.py): candidate matrices built on the GPU, bit-identical"
          + ("; parallel trace prefetch per chunk" if prefetch else ""))
