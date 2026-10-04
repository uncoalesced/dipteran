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

Engineered by uncoalesced.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import shutil
import subprocess
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

# Drosophila fast synaptic transmission: acetylcholine is excitatory, GABA is
# inhibitory, and glutamate is treated as inhibitory because the dominant
# postsynaptic receptor in the fly CNS is the glutamate-gated chloride channel
# GluCl-alpha. The monoamines (octopamine, serotonin, dopamine) are
# neuromodulatory rather than fast-ionotropic; they are left excitatory-signed
# here, which is a placeholder -- PAM/PPL1 dopamine is supposed to act on
# synaptic *weights* in Phase 2, not as a direct current.
#
# Signs are taken per *presynaptic neuron* from the annotation table's
# consensus `top_nt`, not per synapse. That follows Dale's principle -- one
# neuron releases one fast transmitter -- and the per-neuron consensus is far
# less noisy than the per-synapse prediction. It is also much cheaper: the
# edge list's six per-transmitter probability columns over 16.8M rows do not
# need to be read or reduced at all.
NT_SIGN = {
    "acetylcholine": 1.0,
    "gaba": -1.0,
    "glutamate": -1.0,
    "octopamine": 1.0,
    "serotonin": 1.0,
    "dopamine": 1.0,
}
UNKNOWN_NT_SIGN = 1.0

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
    # Restrict to one soma hemisphere via the annotation table's `side`
    # column. Only meaningful for populations that are genuinely bilateral
    # pairs -- EPG is 26 left / 25 right but its wedges tile the whole ring
    # across both hemispheres, so splitting it by `side` would mean nothing.
    side: str | None = None
    # Restrict matching to specific type columns instead of all of
    # TYPE_COLUMNS. `^DNa01$` against every column pools two unrelated cell
    # pairs (see steer_trim_l/steer_trim_r), which is how that cluster ended
    # up with four neurons instead of two.
    columns: tuple[str, ...] | None = None


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
    "mechano_bristle": Cluster(
        "sensory", (r"^(campaniform|chordotonal)",),
        fallbacks=((r"^(head|eye) bristle$",
                    "campaniform sensilla and chordotonal organs sit in the "
                    "legs and wings, so their afferents are ventral nerve "
                    "cord neurons and cannot be in a brain-only volume -- the "
                    "same reason PLAN.md's b1/b2 throttle had to become "
                    "DNg02. The brain-side mechanosensory afferents v783 does "
                    "curate are the bristle populations, which is where a "
                    "mechanical shock actually enters this dataset."),),
        note=(
            "Pothole impact. PLAN.md section 3 routes potholes to 'chordotonal "
            "organ & campaniform sensilla'; see the fallback for why those "
            "resolve to bristle mechanosensory afferents here instead."
        ),
    ),
    "acoustic_jo": Cluster(
        "sensory", (r"^auditory$",),
        fallbacks=((r"^JO-[AB]",
                    "cell_sub_class 'auditory' missing; falling back to the "
                    "JO-A/JO-B cell types by name"),),
        note=(
            "Johnston's organ auditory pathway; the horn-blast input. PLAN.md "
            "calls this 'acoustic projection neurons (aPNs)', but no aPN label "
            "exists in v783. What the release does curate is "
            "cell_sub_class='auditory': 390 neurons, every one a JO-A*/JO-B* "
            "type with super_class='sensory' -- the sound-sensitive Johnston's "
            "organ subgroups, as opposed to the JO-C/D/E wind-and-gravity and "
            "grooming subgroups in the same organ. These are afferents rather "
            "than projection neurons, which makes them the better injection "
            "site anyway: it is where sound actually enters the brain."
        ),
    ),
    # -- steering command input --------------------------------------------
    #
    # PFL3 is where a goal-relative heading error becomes a turn command. It
    # is the only site measured to have real authority over the steering
    # readout: driving side-split DM1/DM4 (PLAN.md's odour channel, and the
    # site an earlier Phase 3 draft locked) moves DNa02 by 2.2 Hz in the wrong
    # direction, because six olfactory PNs cannot shift a descending neuron
    # whose in-degree is 742. Driving PFL3 moves it 40.3 Hz, monotonically.
    # See `scripts/probe_diagnostics.py` Tests E and F.
    #
    # THE PROJECTION IS CONTRALATERAL, and this is measured from W directly,
    # not assumed: DNa02-left receives 238 synapses from all 12 PFL3-right cells and zero
    # from PFL3-left; DNa02-right is the mirror. So driving PFL3-LEFT turns
    # the vehicle RIGHT. That matches Nature 2024's "PFL3R cells are recruited
    # when the fly is oriented to the left of its goal, and their activity
    # drives rightward turning" -- the connectome agrees with the paper
    # without having been told about it. `World.sense()` relies on this
    # crossing; `tests/test_units.py` asserts the sign.
    "steer_drive_pfl3_l": Cluster(
        "sensory", (r"^PFL3$",), side="left",
        note=("PFL3 left. Projects contralaterally onto DNa02-right, so "
              "driving this population steers the vehicle RIGHT."),
    ),
    "steer_drive_pfl3_r": Cluster(
        "sensory", (r"^PFL3$",), side="right",
        note=("PFL3 right. Projects contralaterally onto DNa02-left, so "
              "driving this population steers the vehicle LEFT."),
    ),
    # -- motor -------------------------------------------------------------
    #
    # The steering readout is the right-minus-left difference *within* DNa02,
    # which is what the literature relates to rotational velocity. The old
    # `rate(DNa02) - rate(DNa01)` was a category error: two functionally
    # distinct, largely non-overlapping circuits, each pooled across both
    # hemispheres. Measured, it returns a near-constant +0.063 at rest with no
    # directional input at all -- which is exactly the `steer_abs_median`
    # 0.075 that made Phase 2's vehicle orbit. Its sign is not even structural:
    # it flips to -0.066 at `min_syn_count=5`.
    "steer_left": Cluster(
        "motor", (r"^DNa02$",), side="left",
        note="DNa02 left. High-gain transient steering; drives left turns.",
    ),
    "steer_right": Cluster(
        "motor", (r"^DNa02$",), side="right",
        note="DNa02 right. High-gain transient steering; drives right turns.",
    ),
    # DNa01 is resolved but deliberately NOT blended into the steering
    # difference -- it is a separate circuit (low-gain, sustained, tied to
    # forward velocity) with largely non-overlapping inputs from DNa02, so
    # combining them reintroduces the same category error one level down.
    # Held as an optional trim channel.
    #
    # Matched on `hemibrain_type` alone, which is the disambiguation: against
    # all five type columns `^DNa01$` also catches a pair whose `cell_type` is
    # DNa01 but whose `hemibrain_type` is VES006 -- a different cell entirely.
    # That collision is why the old cluster held four neurons and the
    # annotation table's `cell_type` query only ever found two.
    "steer_trim_l": Cluster(
        "motor", (r"^DNa01$",), side="left", columns=("hemibrain_type",),
        note="Genuine DNa01 left (cell_type DNae001). Low-gain trim, unused.",
    ),
    "steer_trim_r": Cluster(
        "motor", (r"^DNa01$",), side="right", columns=("hemibrain_type",),
        note="Genuine DNa01 right (cell_type DNae001). Low-gain trim, unused.",
    ),
    "panic_dnp01": Cluster(
        "motor", (r"^DNp01$",),
        note="Giant Fiber; emergency brake / panic swerve.",
    ),
    "throttle": Cluster(
        "motor", (r"^(b1|b2|MN ?b1|MN ?b2)$",),
        fallbacks=((r"^DNg02",
                    "b1/b2 are basalar wing motor neurons in the ventral nerve "
                    "cord, and FAFB/FlyWire is a brain-only volume, so they "
                    "cannot be in this dataset at all. DNg02 is the brain-side "
                    "descending population that sets wingbeat amplitude -- the "
                    "closest thing to a throttle that exists in the brain. "
                    "Matched as a prefix because v783 splits it into the "
                    "subtypes DNg02_a through DNg02_h."),),
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
            _fetch(url, tmp)
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
            print(f"[flywire] {url} failed: {exc}", flush=True)
            # tmp is deliberately kept, not deleted -- on a resumable
            # download it holds real progress. It's only ever removed once
            # every URL has been exhausted (below).
    tmp.unlink(missing_ok=True)
    raise RuntimeError(
        f"could not fetch {target.name}: {last_error}\n"
        f"Source: {source.description}\n"
        f"Download it manually to {target} if the network is blocked."
    )


def _fetch(url: str, tmp: Path) -> None:
    """Resumable, retrying download via curl, falling back to a single
    urllib request if curl isn't on PATH.

    A large (multi-hundred-MB) file over a connection that stalls partway
    through is exactly what this needs to survive: curl -C - resumes from
    tmp's existing size instead of restarting from byte 0 on every retry,
    and --speed-limit/--speed-time aborts a transfer that's crawled to a
    near-stop so --retry can reconnect, rather than sitting on one dead
    connection until a long fixed timeout expires. A naive
    urlopen-then-copy loses all progress on every failure, which is why an
    earlier version of this function needed three from-scratch attempts
    (up to a 600s timeout each) to fail the exact same way each time.
    """
    cmd = [
        "curl", "-fSL",
        "-C", "-",                                       # resume tmp
        "--retry", "20", "--retry-delay", "3", "--retry-all-errors",
        "--speed-limit", "2000", "--speed-time", "30",    # reconnect if the
                                                            # transfer drops
                                                            # under ~2KB/s
                                                            # for 30s
        "-A", "dipteran/0.1",
        "-o", str(tmp),
        url,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return
    except FileNotFoundError:
        pass   # no curl on PATH -- fall through to urllib below
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"curl failed: {exc.stderr.strip() or exc}") from exc

    request = urllib.request.Request(url, headers={"User-Agent": "dipteran/0.1"})
    with urllib.request.urlopen(request, timeout=120) as response, \
            open(tmp, "wb") as fh:
        shutil.copyfileobj(response, fh, length=1 << 22)


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


def neuron_signs(cfg: Config) -> dict[int, float]:
    """Presynaptic root ID -> synaptic sign, from the annotation consensus."""
    frame, _ = load_annotations(cfg)
    if "top_nt" not in frame.columns:
        raise KeyError(
            f"{cfg.paths.cell_annotations.name} has no top_nt column; cannot "
            "sign the adjacency matrix"
        )
    ids = frame["root_id"].fillna("")
    nts = frame["top_nt"].fillna("").str.strip().str.lower()
    mix = dict(nts.value_counts())
    unknown = sorted(set(nts) - set(NT_SIGN) - {""})
    if unknown:
        print(f"[conn] transmitters with no sign rule, treated as "
              f"{UNKNOWN_NT_SIGN:+.0f}: {unknown}", flush=True)
    print(f"[conn] annotated transmitter mix: "
          f"{ {k: int(v) for k, v in mix.items() if k} }", flush=True)
    return {
        int(root_id): NT_SIGN.get(nt, UNKNOWN_NT_SIGN)
        for root_id, nt in zip(ids, nts)
        if str(root_id).isdigit()
    }


def aggregate_pairs(cfg: Config, signed: bool = True,
                    min_syn_count: int | None = None) -> dict:
    """Collapse the edge list to one signed weight per connected pair.

    The v783 release splits each connected pair across neuropils; summing
    those gives the aggregate synapse count, which is the single number this
    project wants per pair. There is deliberately no per-synapse matrix here:
    54.5M synapses collapse to 15.1M connected pairs, and it is the pairs that
    define the adjacency structure.

    Only three columns are read. Signs come from the presynaptic neuron's
    annotated transmitter (see NT_SIGN), so the six per-transmitter
    probability columns never have to be loaded or reduced over 16.8M rows.

    `min_syn_count` drops weak pairs. The median pair in v783 has just *two*
    synapses, and the FlyWire papers conventionally treat >=5 as the threshold
    for a reliable connection -- below that, a "connection" is within the
    noise of segmentation and synapse detection. The default comes from
    config.yaml and is 0, keeping everything, because PLAN.md's standing
    decision is maximum fidelity over a curated subset. Raise it if the weak
    tail turns out to dominate Phase 2's dynamics.
    """
    import pyarrow.feather as feather

    path = cfg.paths.connectome_edges
    columns = list(feather.read_table(path, memory_map=True).schema.names)
    pre_col = _pick(columns, PRE_COLUMNS, "presynaptic root id")
    post_col = _pick(columns, POST_COLUMNS, "postsynaptic root id")
    count_col = _pick(columns, COUNT_COLUMNS, "synapse count")
    print(f"[conn] {path.name}: pre={pre_col} post={post_col} "
          f"count={count_col}", flush=True)

    started = time.monotonic()
    frame = feather.read_table(
        path, columns=[pre_col, post_col, count_col], memory_map=True,
    ).to_pandas()
    print(f"[conn] read {len(frame):,} (pair, neuropil) rows in "
          f"{time.monotonic() - started:.0f}s", flush=True)

    grouped = frame.groupby([pre_col, post_col], sort=True, as_index=False,
                            observed=True).sum()
    del frame

    pre = grouped[pre_col].to_numpy(dtype=np.int64)
    post = grouped[post_col].to_numpy(dtype=np.int64)
    total = grouped[count_col].to_numpy(dtype=np.float64)
    del grouped

    threshold = cfg.compute.min_syn_count if min_syn_count is None else min_syn_count
    if threshold > 1:
        keep = total >= threshold
        print(f"[conn] min_syn_count={threshold}: keeping "
              f"{int(keep.sum()):,} of {len(total):,} pairs "
              f"({keep.mean():.1%})", flush=True)
        pre, post, total = pre[keep], post[keep], total[keep]

    if signed:
        signs_by_id = neuron_signs(cfg)
        signs = np.fromiter(
            (signs_by_id.get(int(p), UNKNOWN_NT_SIGN) for p in pre),
            dtype=np.float64, count=len(pre),
        )
        unannotated = sum(1 for p in pre if int(p) not in signs_by_id)
        n_inhibitory = int((signs < 0).sum())
        print(f"[conn] {n_inhibitory:,} inhibitory / "
              f"{len(signs) - n_inhibitory:,} excitatory pairs; "
              f"{unannotated:,} pairs had an unannotated presynaptic neuron",
              flush=True)
    else:
        signs = np.ones(len(total))
        n_inhibitory = 0
        unannotated = 0

    print(f"[conn] {len(total):,} connected neuron pairs, "
          f"{total.sum():,.0f} synapses total", flush=True)
    return {
        "pre": pre, "post": post, "count": total, "sign": signs,
        "inhibitory_pairs": n_inhibitory,
        "unannotated_presynaptic": unannotated,
        "synapses_total": float(total.sum()),
        "min_syn_count": int(threshold),
    }


def build_adjacency(cfg: Config, pairs: dict | None = None,
                    signed: bool = True) -> dict:
    """Build the GPU-resident sparse CSR adjacency matrix W."""
    import torch
    from scipy.sparse import coo_matrix

    pairs = pairs if pairs is not None else aggregate_pairs(cfg, signed=signed)

    neurons = np.union1d(pairs["pre"], pairs["post"])
    n = len(neurons)
    rows = np.searchsorted(neurons, pairs["post"])   # postsynaptic -> row
    cols = np.searchsorted(neurons, pairs["pre"])    # presynaptic  -> column
    values = pairs["count"] * pairs["sign"]

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
        "inhibitory_pairs": int(pairs["inhibitory_pairs"]),
        "unannotated_presynaptic": int(pairs["unannotated_presynaptic"]),
        "synapses_total": int(pairs["synapses_total"]),
        "synapses_per_pair": round(pairs["synapses_total"] / max(nnz, 1), 3),
        "min_syn_count": int(pairs["min_syn_count"]),
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
    haystacks = {c: frame[c].fillna("").to_numpy() for c in type_columns}
    # `side` is what makes a left/right steering readout possible at all. It
    # is populated cleanly in v783 -- 69,956 left / 69,088 right / 174 center
    # / 30 na -- but a cluster asking for a hemisphere the table cannot supply
    # should fail loudly rather than resolve to everything.
    sides = (frame["side"].fillna("na").to_numpy() if "side" in frame.columns
             else None)

    resolved: dict[str, dict] = {}
    for name, cluster in CLUSTERS.items():
        if cluster.side and sides is None:
            raise AssertionError(
                f"cluster {name!r} asks for side={cluster.side!r} but "
                f"{cfg.paths.cell_annotations.name} has no `side` column"
            )
        columns = cluster.columns or tuple(type_columns)
        missing = [c for c in columns if c not in haystacks]
        if missing:
            raise AssertionError(
                f"cluster {name!r} restricts matching to {columns}, but "
                f"{missing} are not in {cfg.paths.cell_annotations.name}"
            )

        attempts = [(p, "", "primary") for p in cluster.patterns]
        attempts += [(p, why, "fallback") for p, why in cluster.fallbacks]

        hit: dict | None = None
        tried: list[str] = []
        for pattern, why, kind in attempts:
            tried.append(pattern)
            regex = re.compile(pattern)
            mask = np.zeros(len(frame), dtype=bool)
            for column in columns:
                mask |= np.fromiter(
                    (bool(regex.match(str(value)))
                     for value in haystacks[column]),
                    dtype=bool, count=len(frame),
                )
            if cluster.side:
                mask &= sides == cluster.side
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
                "side": cluster.side,
                "columns": list(columns),
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
        side = f" side={cluster.side}" if cluster.side else ""
        print(f"[cells] {name:<20} {hit['pattern']:<24}{side:<12} "
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
    # Weights are raw synapse counts, so scale the drive down to keep an fp16
    # product inside range. Phase 2's LIF calibration has the same problem.
    out = torch.sparse.mm(w, v / max(float(w.abs().values().max()), 1.0))
    if w.device.type == "cuda":
        torch.cuda.synchronize()
    assert bool(torch.isfinite(out).all()), (
        f"sparse matmul overflowed in {cfg.compute.dtype}"
    )
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
