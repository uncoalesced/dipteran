"""FlyWire v783 connectome -> GPU sparse adjacency matrix, and cluster rows.

Phase 1 tasks 9-10.

  fetch_flywire_data  -- pull the static v783 archive (never committed)
  aggregate_pairs     -- one weighted entry per connected neuron pair
  build_adjacency     -- torch.sparse_csr_tensor on the configured device
  resolve_clusters    -- named sensory/motor clusters -> real root IDs -> rows

Orientation matters: the LIF step in PLAN.md computes `W @ v`, so W[i, j] is
the weight from presynaptic neuron j onto postsynaptic neuron i. Rows are
postsynaptic, columns presynaptic.

Run `python -m dipteran.connectome` to fetch, build, and verify.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import shutil
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dipteran.config import Config, load_config


@dataclass(frozen=True)
class Source:
    url: str
    size_bytes: int | None
    description: str
    mirrors: tuple[str, ...] = ()


# FlyWire v783 static archive. Both of these are openly downloadable -- no
# account gates the bulk data -- but the use terms still apply: see
# DATA_LICENSE.md. Nothing fetched here may be committed to this repo.
FLYWIRE_SOURCES: dict[str, Source] = {
    "connectome_edges": Source(
        url="https://zenodo.org/api/records/10676866/files/"
            "proofread_connections_783.feather/content",
        size_bytes=852_022_274,
        description=(
            "FlyWire v783 proofread connections, already aggregated per "
            "(neuron pair, neuropil). Zenodo record 10676866. Using this "
            "instead of flywire_synapses_783.feather avoids downloading and "
            "grouping 9.5 GB of individual synapses for the same result."
        ),
    ),
    "cell_annotations": Source(
        url="https://raw.githubusercontent.com/flyconnectome/flywire_annotations/"
            "main/supplemental_files/Supplemental_file1_neuron_annotations.tsv",
        size_bytes=None,
        description=(
            "FlyWire v783 neuron annotations (root_id, cell_type, "
            "hemibrain_type, super_class, cell_class) from "
            "flyconnectome/flywire_annotations."
        ),
        mirrors=(
            "https://raw.githubusercontent.com/flyconnectome/flywire_annotations/"
            "master/supplemental_files/Supplemental_file1_neuron_annotations.tsv",
        ),
    ),
}

# Column names differ between FlyWire releases and export paths, so detect
# rather than hardcode.
PRE_COLUMNS = ("pre_pt_root_id", "pre_root_id", "pre_id", "pre")
POST_COLUMNS = ("post_pt_root_id", "post_root_id", "post_id", "post")
COUNT_COLUMNS = ("syn_count", "n_syn", "synapses", "count", "weight")
NT_NAMES = ("ach", "gaba", "glut", "oct", "ser", "da")

# Drosophila fast synaptic transmission: acetylcholine is excitatory, GABA is
# inhibitory, and glutamate is treated as inhibitory because the dominant
# postsynaptic receptor in the fly CNS is the glutamate-gated chloride channel
# GluCl-alpha. The monoamines (octopamine, serotonin, dopamine) are
# neuromodulatory rather than fast-ionotropic; they are left excitatory-signed
# here, which is a placeholder -- PAM/PPL1 dopamine is supposed to act on
# synaptic *weights* in Phase 2, not as a direct current.
NT_SIGN = {"ach": 1.0, "gaba": -1.0, "glut": -1.0,
           "oct": 1.0, "ser": 1.0, "da": 1.0}

TORCH_DTYPES = {"float16": "float16", "float32": "float32", "bfloat16": "bfloat16"}


# --------------------------------------------------------------------------
# Named clusters
# --------------------------------------------------------------------------
#
# Each cluster is a list of regex groups tried in priority order against the
# annotation table's type columns. The first group that matches at least one
# neuron wins, and which one matched is recorded in the output -- so a
# fallback can never masquerade as the primary mapping.

@dataclass(frozen=True)
class Cluster:
    role: str
    patterns: tuple[str, ...]
    fallbacks: tuple[tuple[str, str], ...] = ()   # (pattern, why)
    note: str = ""


CLUSTERS: dict[str, Cluster] = {
    # -- sensory -----------------------------------------------------------
    "odor_dm1": Cluster(
        "sensory", (r"^DM1[_ ]",),
        note="DM1 glomerulus olfactory PNs; the odour-beacon input.",
    ),
    "odor_dm4": Cluster(
        "sensory", (r"^DM4[_ ]",),
        note="DM4 glomerulus olfactory PNs.",
    ),
    "compass_epg": Cluster(
        "sensory", (r"^EPGt?$",),
        note="Ellipsoid body EPG ring; the heading bump.",
    ),
    "optic_flow_t4": Cluster(
        "sensory", (r"^T4[a-d]$",),
        note="T4 ON-pathway motion detectors.",
    ),
    "optic_flow_t5": Cluster(
        "sensory", (r"^T5[a-d]$",),
        note="T5 OFF-pathway motion detectors.",
    ),
    "looming_lc4": Cluster(
        "sensory", (r"^LC4$",),
        note="LC4 looming detector; drives DNp01.",
    ),
    "looming_lplc2": Cluster(
        "sensory", (r"^LPLC2$",),
        note="LPLC2 looming detector; drives DNp01.",
    ),
    "acoustic_apn": Cluster(
        "sensory", (r"^aPN\d*$",),
        fallbacks=((r"^JO-[A-F]$",
                    "auditory PNs absent under that name; falling back to the "
                    "Johnston's organ afferents themselves"),),
        note="Johnston's organ auditory pathway; the horn-blast input.",
    ),
    # -- motor -------------------------------------------------------------
    "steer_l_dna01": Cluster(
        "motor", (r"^DNa01$",),
        note="Descending neuron for left turns.",
    ),
    "steer_r_dna02": Cluster(
        "motor", (r"^DNa02$",),
        note="Descending neuron for right turns.",
    ),
    "panic_dnp01": Cluster(
        "motor", (r"^DNp01$",),
        note="Giant Fiber; emergency brake / panic swerve.",
    ),
    "throttle": Cluster(
        "motor", (r"^(b1|b2|MN ?b1|MN ?b2)$",),
        fallbacks=((r"^DNg02$",
                    "b1/b2 are basalar wing motor neurons in the ventral nerve "
                    "cord, and FAFB/FlyWire is a brain-only volume, so they "
                    "cannot be in this dataset. DNg02 is the brain-side "
                    "descending population that sets wingbeat amplitude -- the "
                    "closest thing to a throttle that exists in the brain."),),
        note="Throttle. See the fallback note: PLAN.md's B1/B2 are not in FAFB.",
    ),
    "reward_pam": Cluster(
        "motor", (r"^PAM",),
        note="PAM dopaminergic cluster; sugar reward.",
    ),
    "punish_ppl1": Cluster(
        "motor", (r"^PPL1",),
        note="PPL1 dopaminergic cluster; pain / synaptic depression.",
    ),
}

TYPE_COLUMNS = ("cell_type", "hemibrain_type", "cell_class", "super_class",
                "cell_sub_class", "type")


# --------------------------------------------------------------------------
# Fetch
# --------------------------------------------------------------------------

def fetch_flywire_data(cfg: Config, force: bool = False) -> dict[str, Path]:
    """Download the v783 archive to the configured (gitignored) data paths."""
    out: dict[str, Path] = {}
    for key, source in FLYWIRE_SOURCES.items():
        target: Path = getattr(cfg.paths, key)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and not force:
            size = target.stat().st_size
            if source.size_bytes is None or size == source.size_bytes:
                print(f"[flywire] have {target.name} ({size / 1e6:.1f} MB)",
                      flush=True)
                out[key] = target
                continue
            print(f"[flywire] {target.name} is {size} bytes, expected "
                  f"{source.size_bytes}; re-downloading", flush=True)
        out[key] = _download(source, target)
    return out


def _download(source: Source, target: Path) -> Path:
    tmp = target.with_suffix(target.suffix + ".part")
    last_error: Exception | None = None
    for url in (source.url, *source.mirrors):
        expect = (f" (~{source.size_bytes / 1e6:.0f} MB)"
                  if source.size_bytes else "")
        print(f"[flywire] downloading {target.name}{expect} from {url}",
              flush=True)
        try:
            started = time.monotonic()
            request = urllib.request.Request(
                url, headers={"User-Agent": "dipteran/0.1"})
            with urllib.request.urlopen(request, timeout=120) as response, \
                    open(tmp, "wb") as fh:
                shutil.copyfileobj(response, fh, length=1 << 22)
            size = tmp.stat().st_size
            if source.size_bytes is not None and size != source.size_bytes:
                raise RuntimeError(
                    f"got {size} bytes, expected {source.size_bytes}")
            if size < 1000:
                raise RuntimeError(f"suspiciously small response: {size} bytes")
            tmp.replace(target)
            print(f"[flywire] wrote {target} ({size / 1e6:.1f} MB) in "
                  f"{time.monotonic() - started:.0f}s", flush=True)
            return target
        except Exception as exc:
            last_error = exc
            tmp.unlink(missing_ok=True)
            print(f"[flywire] {url} failed: {exc}", flush=True)
    raise RuntimeError(
        f"could not fetch {target.name}: {last_error}\n"
        f"Source: {source.description}\n"
        f"Download it manually to {target} if the network is blocked."
    )


# --------------------------------------------------------------------------
# Edge list -> one weighted entry per pair
# --------------------------------------------------------------------------

def _pick(columns, candidates, what: str) -> str:
    lowered = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    raise KeyError(
        f"no {what} column in the edge list; looked for {candidates}, "
        f"found {sorted(columns)[:25]}"
    )


def _nt_columns(columns) -> dict[str, str]:
    """Map neurotransmitter name -> its probability column, if present."""
    lowered = {c.lower(): c for c in columns}
    found: dict[str, str] = {}
    for nt in NT_NAMES:
        for suffix in ("_avg", "", "_prob", "_probability"):
            key = f"{nt}{suffix}"
            if key in lowered:
                found[nt] = lowered[key]
                break
    return found


def aggregate_pairs(cfg: Config) -> dict:
    """Collapse the edge list to one signed weight per connected pair.

    The v783 release splits each connected pair across neuropils; summing
    those gives the aggregate synapse count, which is the single number this
    project wants per pair. There is deliberately no per-synapse matrix here:
    54.5M synapses collapse to a few million connected pairs, and it is the
    pairs that define the adjacency structure.
    """
    import pyarrow.feather as feather

    path = cfg.paths.connectome_edges
    schema = feather.read_table(path, memory_map=True).schema
    columns = list(schema.names)
    pre_col = _pick(columns, PRE_COLUMNS, "presynaptic root id")
    post_col = _pick(columns, POST_COLUMNS, "postsynaptic root id")
    count_col = _pick(columns, COUNT_COLUMNS, "synapse count")
    nt_cols = _nt_columns(columns)
    print(f"[conn] {path.name}: pre={pre_col} post={post_col} "
          f"count={count_col} nt={sorted(nt_cols)}", flush=True)

    wanted = [pre_col, post_col, count_col, *nt_cols.values()]
    started = time.monotonic()
    frame = feather.read_table(path, columns=wanted, memory_map=True).to_pandas()
    print(f"[conn] read {len(frame):,} (pair, neuropil) rows in "
          f"{time.monotonic() - started:.0f}s", flush=True)

    # Synapse counts sum over neuropils; neurotransmitter probabilities are
    # averaged weighted by those counts, so a pair's sign comes from where its
    # synapses actually are.
    counts = frame[count_col].astype(np.float64)
    for col in nt_cols.values():
        frame[col] = frame[col].astype(np.float64) * counts
    frame[count_col] = counts
    grouped = frame.groupby([pre_col, post_col], sort=True, as_index=False).sum()
    del frame

    pre = grouped[pre_col].to_numpy(dtype=np.int64)
    post = grouped[post_col].to_numpy(dtype=np.int64)
    total = grouped[count_col].to_numpy(dtype=np.float64)

    if nt_cols:
        order = sorted(nt_cols)
        probs = np.column_stack([
            grouped[nt_cols[nt]].to_numpy(dtype=np.float64) / np.maximum(total, 1.0)
            for nt in order
        ])
        top = np.asarray(order, dtype=object)[probs.argmax(axis=1)]
        signs = np.asarray([NT_SIGN.get(nt, 1.0) for nt in top])
        nt_mix = {nt: int((top == nt).sum()) for nt in order}
    else:
        print("[conn] no neurotransmitter columns; leaving all weights "
              "excitatory (Phase 2 must revisit)", flush=True)
        top = np.full(len(total), "unknown", dtype=object)
        signs = np.ones(len(total))
        nt_mix = {}

    del grouped
    print(f"[conn] {len(total):,} connected neuron pairs, "
          f"{total.sum():,.0f} synapses total", flush=True)
    print(f"[conn] predicted transmitter mix: {nt_mix}", flush=True)
    return {
        "pre": pre, "post": post, "count": total,
        "sign": signs, "nt": top, "nt_mix": nt_mix,
    }


def build_adjacency(cfg: Config, pairs: dict | None = None,
                    signed: bool = True) -> dict:
    """Build the GPU-resident sparse CSR adjacency matrix W."""
    import torch
    from scipy.sparse import coo_matrix

    pairs = pairs if pairs is not None else aggregate_pairs(cfg)

    neurons = np.union1d(pairs["pre"], pairs["post"])
    n = len(neurons)
    rows = np.searchsorted(neurons, pairs["post"])   # postsynaptic -> row
    cols = np.searchsorted(neurons, pairs["pre"])    # presynaptic  -> column
    values = pairs["count"] * (pairs["sign"] if signed else 1.0)

    cfg.paths.neuron_index.parent.mkdir(parents=True, exist_ok=True)
    np.save(cfg.paths.neuron_index, neurons)

    # scipy gives us a canonical, row-sorted CSR with int32 indices for free;
    # torch then wraps those buffers without another sort.
    csr = coo_matrix((values, (rows, cols)), shape=(n, n)).tocsr()
    csr.sort_indices()
    nnz = csr.nnz

    device = torch.device(cfg.compute.device if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print(f"[gpu] CUDA unavailable; falling back to {device}", flush=True)
    dtype = getattr(torch, TORCH_DTYPES[cfg.compute.dtype])

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        before = torch.cuda.memory_allocated(device)
        free_before, total_vram = torch.cuda.mem_get_info(device)
    else:
        before = free_before = total_vram = 0

    w = torch.sparse_csr_tensor(
        torch.from_numpy(csr.indptr.astype(np.int32)).to(device),
        torch.from_numpy(csr.indices.astype(np.int32)).to(device),
        torch.from_numpy(csr.data).to(device=device, dtype=dtype),
        size=(n, n), device=device,
    )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        after = torch.cuda.memory_allocated(device)
        peak = torch.cuda.max_memory_allocated(device)
        free_after, _ = torch.cuda.mem_get_info(device)
    else:
        after = peak = free_after = 0

    # What the matrix *should* cost: values at dtype width, column indices and
    # the row pointer at int32.
    predicted = nnz * dtype.itemsize + nnz * 4 + (n + 1) * 4
    stats = {
        "neurons": int(n),
        "nnz": int(nnz),
        "density": nnz / float(n) ** 2,
        "dtype": cfg.compute.dtype,
        "device": str(device),
        "matrix_bytes_predicted": int(predicted),
        "matrix_bytes_allocated": int(after - before),
        "peak_allocated_bytes": int(peak),
        "vram_total_bytes": int(total_vram),
        "vram_used_by_build_bytes": int(free_before - free_after),
    }
    print(f"[gpu] W: {n:,} x {n:,}, {nnz:,} non-zeros "
          f"(density {stats['density']:.2e})", flush=True)
    print(f"[gpu] {cfg.compute.dtype} CSR on {device}: "
          f"{stats['matrix_bytes_allocated'] / 2**20:.1f} MiB allocated "
          f"(predicted {predicted / 2**20:.1f} MiB), peak "
          f"{peak / 2**20:.1f} MiB", flush=True)
    if total_vram:
        print(f"[gpu] VRAM: {stats['vram_used_by_build_bytes'] / 2**20:.1f} MiB "
              f"consumed of {total_vram / 2**30:.2f} GiB total", flush=True)
    return {"W": w, "neurons": neurons, "stats": stats}


# --------------------------------------------------------------------------
# Cell-ID resolution
# --------------------------------------------------------------------------

def load_annotations(cfg: Config):
    import pandas as pd

    path = cfg.paths.cell_annotations
    frame = pd.read_csv(path, sep="\t", dtype=str, low_memory=False)
    if "root_id" not in frame.columns:
        raise KeyError(
            f"{path.name} has no root_id column; found {list(frame.columns)[:20]}"
        )
    present = [c for c in TYPE_COLUMNS if c in frame.columns]
    if not present:
        raise KeyError(
            f"{path.name} has none of the expected type columns {TYPE_COLUMNS}; "
            f"found {list(frame.columns)}"
        )
    print(f"[cells] {path.name}: {len(frame):,} annotated neurons, "
          f"type columns {present}", flush=True)
    return frame, present


def resolve_clusters(cfg: Config, neurons: np.ndarray | None = None) -> dict:
    """Resolve every named cluster to real root IDs and matrix row indices."""
    frame, type_columns = load_annotations(cfg)
    if neurons is None:
        neurons = np.load(cfg.paths.neuron_index)

    root_ids = frame["root_id"].fillna("").to_numpy()
    haystacks = [frame[c].fillna("").to_numpy() for c in type_columns]

    resolved: dict[str, dict] = {}
    for name, cluster in CLUSTERS.items():
        attempts = [(p, "", "primary") for p in cluster.patterns]
        attempts += [(p, why, "fallback") for p, why in cluster.fallbacks]

        hit: dict | None = None
        tried: list[str] = []
        for pattern, why, kind in attempts:
            tried.append(pattern)
            regex = re.compile(pattern)
            mask = np.zeros(len(frame), dtype=bool)
            for column in haystacks:
                mask |= np.fromiter(
                    (bool(regex.match(str(value))) for value in column),
                    dtype=bool, count=len(column),
                )
            if not mask.any():
                continue
            matched = np.unique(root_ids[mask])
            ids = np.asarray([int(v) for v in matched if str(v).isdigit()],
                             dtype=np.int64)
            if not len(ids):
                continue
            # A root ID only has a matrix row if it actually has connections
            # in the edge list.
            positions = np.clip(np.searchsorted(neurons, ids), 0, len(neurons) - 1)
            in_matrix = neurons[positions] == ids
            hit = {
                "role": cluster.role,
                "pattern": pattern,
                "match_kind": kind,
                "why_fallback": why or None,
                "annotated": int(len(ids)),
                "root_ids": [str(v) for v in ids[in_matrix]],
                "rows": [int(p) for p in positions[in_matrix]],
                "missing_from_matrix": int((~in_matrix).sum()),
                "note": cluster.note,
            }
            break

        if hit is None:
            raise AssertionError(
                f"cluster {name!r} resolved to nothing. Patterns tried: "
                f"{tried}. Available type columns: {type_columns}. Fix the "
                "patterns rather than inventing indices."
            )
        resolved[name] = hit
        flag = "" if hit["match_kind"] == "primary" else "  <-- FALLBACK"
        print(f"[cells] {name:<16} {hit['pattern']:<24} "
              f"{len(hit['rows']):>6} rows{flag}", flush=True)

    payload = {
        "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "neuron_count": int(len(neurons)),
        "annotation_source": cfg.paths.cell_annotations.name,
        "type_columns": type_columns,
        "clusters": resolved,
    }
    cfg.paths.cluster_ids.parent.mkdir(parents=True, exist_ok=True)
    cfg.paths.cluster_ids.write_text(json.dumps(payload, indent=2),
                                     encoding="utf-8")

    sensory = sum(1 for v in resolved.values() if v["role"] == "sensory")
    motor = sum(1 for v in resolved.values() if v["role"] == "motor")
    print(f"[cells] {sensory} sensory and {motor} motor clusters resolved, "
          f"wrote {cfg.paths.cluster_ids}", flush=True)
    return payload


def sensory_motor_ids(cluster_payload: dict) -> tuple[dict, dict]:
    """The `sensory_ids` / `motor_ids` shape FlyBangaloreNavigator expects."""
    sensory = {k: v["rows"] for k, v in cluster_payload["clusters"].items()
               if v["role"] == "sensory"}
    motor = {k: v["rows"] for k, v in cluster_payload["clusters"].items()
             if v["role"] == "motor"}
    return sensory, motor


def main() -> None:
    import torch

    cfg = load_config()
    fetch_flywire_data(cfg)
    built = build_adjacency(cfg)
    clusters = resolve_clusters(cfg, neurons=built["neurons"])

    w = built["W"]
    n = built["stats"]["neurons"]
    assert w.shape == (n, n), w.shape
    assert w.layout is torch.sparse_csr, w.layout

    # One real sparse matmul, to prove the matrix is usable on this device and
    # not merely resident on it.
    v = torch.zeros(n, 1, dtype=w.dtype, device=w.device)
    for rows in sensory_motor_ids(clusters)[0].values():
        if rows:
            v[rows] = 1.0
    out = torch.sparse.mm(w, v)
    if w.device.type == "cuda":
        torch.cuda.synchronize()
    driven = int((out.squeeze(1) != 0).sum())
    print(f"[gpu] one sparse matmul from the sensory clusters drives "
          f"{driven:,} downstream neurons", flush=True)
    assert driven > 0, "sensory injection reached no downstream neuron"

    for name, cluster in clusters["clusters"].items():
        assert cluster["rows"], f"{name} has no matrix rows"
        assert max(cluster["rows"]) < n, f"{name} has an out-of-range row"

    print("\nconnectome self-check OK")


if __name__ == "__main__":
    main()
