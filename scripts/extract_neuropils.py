"""Build the HUD's brain point cloud from the real FlyWire annotation table.

One-time ETL: 139,248 annotated neurons in, a downsampled tagged point cloud
out, for the Three.js panel in index.html. Independent of the SNN loop, which
is why it is a script and not part of the runtime.

Two corrections to the design this was based on, both found by reading the
actual annotation file rather than assuming its shape:

1.  There are no `x`/`y`/`z` columns. v783 has `pos_x`/`pos_y`/`pos_z`
    (139,248 rows, complete) and `soma_x`/`soma_y`/`soma_z` (118,104 rows,
    85%). A `usecols=[..., "x", "y", "z"]` raises KeyError on the first line
    that touches the file. `pos_*` is used here for the coverage.

2.  Those coordinates are FAFB voxel indices, not microns, and the voxels are
    not cubes: 4 nm in x and y, 40 nm in z. Plotting them raw squashes the
    brain by 10x along z. They are scaled to microns here.

Region tagging composes the two schemes that exist rather than picking one.
The 15 named clusters resolved in Phase 1 are real FlyWire root IDs and drive
the HUD's glow, so they win wherever a neuron belongs to one. Everything else
-- about 120k neurons that exist only as visual context -- falls back to a
crude `super_class`/`cell_class` heuristic, which is all a background scaffold
needs.

    python scripts/extract_neuropils.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# FAFB voxel size. x and y are 4 nm, z is 40 nm -- see the module docstring.
VOXEL_NM = (4.0, 4.0, 40.0)

# Okabe-Ito, which is colourblind-safe, with its black swapped for a light
# grey so the scaffold reads on a dark HUD instead of vanishing into it.
REGIONS = [
    ("optic",    "Optic lobes",          "#56B4E9"),
    ("looming",  "Looming",              "#D55E00"),
    ("antennal", "Antennal lobes",       "#009E73"),
    ("central",  "Central complex",      "#F0E442"),
    ("auditory", "Johnston's organ",     "#CC79A7"),
    ("mechano",  "Mechanosensory",       "#E69F00"),
    ("steer",    "Descending steer",     "#0072B2"),
    ("throttle", "Descending throttle",  "#8FD3C7"),
    ("giant",    "Giant fiber",          "#FF4B3E"),
    ("dopamine", "Dopaminergic",         "#B07AA1"),
    ("scaffold", "Unassigned",           "#5A6068"),
]
REGION_INDEX = {rid: i for i, (rid, _, _) in enumerate(REGIONS)}

# Floor on the per-region sample quota. Small clusters are the whole point of
# the precise tier -- DNa02 and DNp01 have two neurons each -- so no region is
# ever sampled below this, and any region at or under it is kept entire.
MIN_PER_REGION = 128

# Which HUD region each Phase 1 cluster belongs to. These are the precise
# ones: real root IDs, verified in Phase 1, and what the SNN actually reads
# and writes.
CLUSTER_REGION = {
    "optic_flow_t4": "optic",
    "optic_flow_t5": "optic",
    "looming_lc4": "looming",
    "looming_lplc2": "looming",
    "odor_dm1": "antennal",
    "odor_dm4": "antennal",
    "compass_epg": "central",
    "acoustic_jo": "auditory",
    "mechano_bristle": "mechano",
    "steer_l_dna01": "steer",
    "steer_r_dna02": "steer",
    "throttle": "throttle",
    "panic_dnp01": "giant",
    "reward_pam": "dopamine",
    "punish_ppl1": "dopamine",
}


def classify_background(super_class: str, cell_class: str,
                        cell_sub_class: str) -> str:
    """Crude fallback for the ~120k neurons in no named cluster.

    Deliberately coarse. These points exist so the cloud looks like a brain
    rather than a handful of lonely dots; nothing reacts to them, so a
    string-prefix guess is the right amount of effort.
    """
    sup = (super_class or "").lower()
    cls = (cell_class or "").lower()
    sub = (cell_sub_class or "").lower()

    if sup in {"optic", "visual_projection", "visual_centrifugal"}:
        return "optic"
    if cls == "visual":
        return "optic"
    if cls == "olfactory":
        return "antennal"
    if sub == "auditory":
        return "auditory"
    if cls == "mechanosensory":
        return "mechano"
    if sup == "descending":
        return "throttle"
    return "scaffold"


def build(cfg, budget: int, seed: int) -> dict:
    import pandas as pd

    from dipteran.connectome import resolve_clusters

    path = cfg.paths.cell_annotations
    frame = pd.read_csv(
        path, sep="\t", dtype=str, low_memory=False,
        usecols=["root_id", "pos_x", "pos_y", "pos_z",
                 "super_class", "cell_class", "cell_sub_class"],
    )
    print(f"[etl] {path.name}: {len(frame):,} annotated neurons", flush=True)

    root = pd.to_numeric(frame["root_id"], errors="coerce")
    xyz = np.column_stack([
        pd.to_numeric(frame[c], errors="coerce").to_numpy(dtype=np.float64)
        for c in ("pos_x", "pos_y", "pos_z")
    ])
    keep = root.notna().to_numpy() & np.isfinite(xyz).all(axis=1)
    frame, xyz = frame[keep].reset_index(drop=True), xyz[keep]
    ids = root[keep].to_numpy(dtype=np.int64)
    print(f"[etl] {len(ids):,} with usable coordinates", flush=True)

    # Voxels -> microns, then centre the cloud on its own bounding box so the
    # front end can drop it into a scene without re-centring.
    microns = xyz * np.asarray(VOXEL_NM) / 1000.0
    microns -= (microns.min(axis=0) + microns.max(axis=0)) / 2.0

    # -- precise tier: the Phase 1 clusters win wherever they apply --------
    region = np.full(len(ids), REGION_INDEX["scaffold"], dtype=np.int16)
    precise = np.zeros(len(ids), dtype=bool)

    payload = resolve_clusters(cfg)
    order = np.argsort(ids)
    sorted_ids = ids[order]
    for name, entry in payload["clusters"].items():
        target = CLUSTER_REGION.get(name)
        if target is None:
            print(f"[etl] WARNING: cluster {name!r} has no HUD region",
                  flush=True)
            continue
        # resolve_clusters stores root_ids as strings.
        want = np.asarray([int(v) for v in entry["root_ids"]], dtype=np.int64)
        if not len(want):
            continue
        pos = np.clip(np.searchsorted(sorted_ids, want), 0, len(sorted_ids) - 1)
        hit = order[pos[sorted_ids[pos] == want]]
        region[hit] = REGION_INDEX[target]
        precise[hit] = True

    print(f"[etl] {int(precise.sum()):,} neurons tagged from resolved clusters",
          flush=True)

    # -- background tier: everything else ---------------------------------
    rest = np.flatnonzero(~precise)
    sup = frame["super_class"].fillna("").to_numpy()
    cls = frame["cell_class"].fillna("").to_numpy()
    sub = frame["cell_sub_class"].fillna("").to_numpy()
    for i in rest:
        region[i] = REGION_INDEX[classify_background(sup[i], cls[i], sub[i])]

    # -- downsample -------------------------------------------------------
    # Stratified per region, with a quota rather than "keep every precise
    # neuron". Keeping them all does not work: the precise tier is 14,777
    # neurons, mostly the 12,246 T4/T5 optic-flow cells, which on its own
    # overruns a 4,000-point budget and leaves no room for the scaffold --
    # clusters floating in empty space instead of inside a brain.
    #
    # The quota has a floor so the opposite failure cannot happen either.
    # DNa02 is two neurons out of 139,248 and DNp01 is two more; a
    # proportional sample drops both, and the HUD has a legend entry for each.
    # Any group at or under the quota is kept whole.
    rng = np.random.default_rng(seed)
    present = np.unique(region)
    quota = max(MIN_PER_REGION, budget // max(1, len(present)))

    def sample(pool: np.ndarray) -> np.ndarray:
        groups = [pool[region[pool] == r] for r in np.unique(region[pool])]
        picked = [g if len(g) <= quota
                  else rng.choice(g, size=quota, replace=False) for g in groups]
        return np.concatenate(picked) if picked else np.zeros(0, dtype=np.int64)

    precise_index = sample(np.flatnonzero(precise))
    scaffold_index = sample(rest)

    index = np.concatenate([precise_index, scaffold_index])
    rng.shuffle(index)          # so draw order does not stack one region on top
    print(f"[etl] quota {quota}/region -> sampled {len(index):,} points "
          f"({len(precise_index):,} precise + {len(scaffold_index):,} scaffold)",
          flush=True)

    points = [[round(float(x), 2), round(float(y), 2), round(float(z), 2),
               int(region[i])]
              for i, (x, y, z) in zip(index, microns[index])]

    counts = {rid: int((region == REGION_INDEX[rid]).sum())
              for rid, _, _ in REGIONS}

    return {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(
            timespec="seconds"),
        "source": path.name,
        "neurons_total": int(len(ids)),
        "neurons_precise": int(precise.sum()),
        "voxel_nm": list(VOXEL_NM),
        "regions": [{"id": r, "label": lbl, "color": c}
                    for r, lbl, c in REGIONS],
        "cluster_region": CLUSTER_REGION,
        "counts_all_neurons": counts,
        "points": points,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--budget", type=int, default=4000,
                    help="target point count for the HUD cloud")
    ap.add_argument("--seed", type=int, default=20260914)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    from dipteran.config import load_config

    cfg = load_config()
    payload = build(cfg, args.budget, args.seed)

    out = args.out or (cfg.paths.cache_dir / "fly_neuropils.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"[etl] wrote {out} ({out.stat().st_size / 1024:.0f} KiB)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
