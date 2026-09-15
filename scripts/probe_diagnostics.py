"""Steering/reward diagnostics against the *untouched* Phase 2 baseline.

`PHASE3_HANDOFF.md` Part 0 forbids changing sensory or motor code before these
numbers exist, because three separate "obvious" fixes proposed while diagnosing
the orbit turned out to be wrong once checked against the real annotation table.
So nothing here imports a fix or touches production code -- it only measures
what the shipped network already does.

Four tests, written to `cache/steering_bias_diagnostic.json`:

  A  The pre-Phase-3 steering readout, `rate(DNa02) - rate(DNa01)` with both
     pooled across hemispheres, under symmetric drive. This is the number that
     drove the vehicle for 25,727 frames, and Phase 2 measured its effect as a
     constant turn rate (`steer_abs_median` 0.075 with a persistent sign) --
     an orbit. Kept after the fix as the comparison point; `_decode` no longer
     computes it.

  B  The same run, split by the annotation table's `side` column without
     touching `CLUSTERS`: per-hemisphere rates for DNa01 and DNa02, plus the
     corrected `DNa02_right - DNa02_left` the literature actually relates to
     rotational velocity. If that corrected difference is ~0 at rest, the orbit
     is entirely the missing directional *input* rather than a structural
     asymmetry, and that changes which fix matters most.

     B also splits DNa01 by which column matched. `resolve_clusters` ORs its
     regex across five type columns, so `^DNa01$` pools two unrelated cell
     pairs -- `hemibrain_type=DNa01` (cell_type `DNae001`, the genuine DNa01)
     and `cell_type=DNa01` (hemibrain_type `VES006`, a different cell). That
     is the source of the "DNa01 has 4 neurons" comment in `navigator.py`
     that the handoff flagged as unexplained.

  C  A and B again against `min_syn_count=5`, the threshold the FlyWire papers
     treat as a reliable connection. Structural contrast only; the checked-in
     default stays 0.

  D  The PAM/PPL1 current -> firing-rate transfer curve. `World.rewards()`
     does not return dopamine -- it returns a *current* that gets injected on
     the PAM rows, and `_decode` reads back the rate those rows actually fire
     at. With `decay=0.9` a sustained current `c` settles at `10c` against a
     threshold of 1.0, so the shape of this curve decides whether a
     proximity-scaled reward (0.02 at the far end of a 15 km route) produces
     any dopamine at all or silently rounds to zero.

    DIPTERAN_DATA_ROOT=/opt/dipteran python scripts/probe_diagnostics.py

Engineered by uncoalesced.
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dipteran.config import load_config                        # noqa: E402
from dipteran.connectome import (                              # noqa: E402
    aggregate_pairs,
    build_adjacency,
    load_annotations,
    resolve_clusters,
)
from dipteran.navigator import (  # noqa: E402
    FlyBangaloreNavigator,
    MotorOutput,
    monoamine_rows,
)

# Matches `calibrate_gain`'s harness: a drive deliberately below `v_thresh`,
# because a sensory current at threshold makes those rows fire every sub-tick,
# which is a short circuit rather than a stimulus.
DRIVE = 0.4
TICKS = 200                     # 200 ticks x 10 sub-ticks = 2,000 sub-ticks
PAM_SWEEP = (0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0)

# The clusters that carry navigation. Excludes the two looming detectors, the
# horn and the pothole channel -- see `tests_ab` for why driving those at 0.4
# puts the network in a permanent escape state no real frame of the route run
# was ever in.
NAVIGATION_SENSORY = ("odor_dm1", "odor_dm4", "compass_epg",
                      "optic_flow_t4", "optic_flow_t5")


# --------------------------------------------------------------------------
# harness
# --------------------------------------------------------------------------

def probe(nav, currents: dict[str, float], *, ticks: int = TICKS,
          row_currents: list[tuple[list, float]] | None = None):
    """Run `ticks` ticks under `currents`; return mean per-neuron rates.

    Plasticity is frozen for the duration. `step()` calls `_reinforce` with
    whatever dopamine it just decoded, so a 200-tick probe with PAM firing
    would quietly retrain the very weights being measured -- the diagnostic
    would then report a moving target. `learning_rate`/`homeostasis` go to
    zero and are restored, leaving `p_val` bit-identical across the probe.

    The rates tensor is captured from `_decode` rather than recomputed, so
    what is reported is exactly what the vehicle's readout sees, not a
    re-derivation that could drift from it.
    """
    learning_rate, homeostasis = nav.learning_rate, nav.homeostasis
    nav.learning_rate = nav.homeostasis = 0.0

    captured: dict = {}
    original = nav._decode

    def spy(rates):
        captured["rates"] = rates
        return original(rates)

    nav._decode = spy
    nav.reset()
    nav.set_sensory(currents)
    # Per-row injection, applied after set_sensory because set_sensory zeroes
    # i_ext first and can only address whole clusters. This is how a bilateral
    # asymmetry gets tested before `CLUSTERS` is split -- no production change.
    for rows, value in (row_currents or []):
        if len(rows):
            nav.i_ext[list(rows)] = float(value) * nav.sensory_gain

    cut = ticks // 3            # discard the transient, as calibrate_gain does
    total, kept, outs = None, 0, []
    for i in range(ticks):
        out = nav.step()
        if i < cut:
            continue
        outs.append(out)
        rates = captured["rates"]
        total = rates.clone() if total is None else total + rates
        kept += 1

    nav._decode = original
    nav.learning_rate, nav.homeostasis = learning_rate, homeostasis
    # Averaged, not the last tick: with one neuron per steering side a
    # single tick quantises `steer` to multiples of 1/subticks, which is
    # the very artefact `MotorOutput.average` exists to remove.
    return total / max(kept, 1), MotorOutput.average(outs)


def mean_rate(rates, rows) -> float:
    """Population mean over `rows`, in Hz. A sub-tick is 2 ms."""
    if rows is None or not len(rows):
        return 0.0
    return round(float(rates[list(rows)].mean()) * 500.0, 3)


# --------------------------------------------------------------------------
# side resolution -- read straight from the annotation table, no CLUSTERS edit
# --------------------------------------------------------------------------

def annotation_index(cfg) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    """(root_id -> side, root_id -> {cell_type, hemibrain_type})."""
    frame, _ = load_annotations(cfg)
    if "side" not in frame.columns:
        raise KeyError(
            f"{cfg.paths.cell_annotations.name} has no `side` column; the "
            "whole per-hemisphere diagnosis depends on it"
        )
    root = frame["root_id"].fillna("").astype(str)
    side = frame["side"].fillna("na").astype(str)
    types = {
        rid: {"cell_type": ct, "hemibrain_type": ht}
        for rid, ct, ht in zip(root,
                               frame["cell_type"].fillna("").astype(str),
                               frame["hemibrain_type"].fillna("").astype(str))
    }
    return dict(zip(root, side)), types


def merge_clusters(payload: dict, *names: str) -> dict:
    """Recombine side-split clusters into one pooled pseudo-cluster.

    `CLUSTERS` now resolves DNa02 and DNa01 per hemisphere, which is the
    fix. The diagnostics still want the pooled populations -- to report
    the pre-fix readout for comparison, and to re-measure `snn.steer_bias`
    -- so they are stitched back together here rather than by keeping
    redundant entries in `CLUSTERS`.
    """
    root_ids: list = []
    rows: list = []
    for name in names:
        root_ids += payload[name]["root_ids"]
        rows += payload[name]["rows"]
    return {"root_ids": root_ids, "rows": rows}


def split_by_side(cluster: dict, side_of: dict[str, str]) -> dict[str, list]:
    """A resolved cluster's rows, grouped by soma hemisphere.

    `root_ids` and `rows` are built as parallel arrays by `resolve_clusters`
    (both filtered by the same `in_matrix` mask), so zipping them is safe.
    """
    out: dict[str, list] = {"left": [], "right": [], "other": []}
    for root_id, row in zip(cluster["root_ids"], cluster["rows"]):
        side = side_of.get(str(root_id), "na")
        out[side if side in ("left", "right") else "other"].append(row)
    return out


def split_dna01_collision(cluster: dict, types: dict, side_of: dict) -> dict:
    """DNa01's four rows are two different cell types. Separate them.

    `hemibrain_type == DNa01` is the genuine DNa01 (its `cell_type` is the
    systematic name `DNae001`). `cell_type == DNa01` is a different cell whose
    `hemibrain_type` is `VES006`. Pooling them is what makes the cluster four
    neurons instead of two.
    """
    groups: dict[str, dict] = {}
    for root_id, row in zip(cluster["root_ids"], cluster["rows"]):
        info = types.get(str(root_id), {})
        if info.get("hemibrain_type") == "DNa01":
            key = "genuine_dna01"        # cell_type DNae001
        elif info.get("cell_type") == "DNa01":
            key = "ves006_collision"     # a different cell entirely
        else:
            key = "unclassified"
        entry = groups.setdefault(key, {"rows": [], "left": [], "right": [],
                                        "members": []})
        entry["rows"].append(row)
        side = side_of.get(str(root_id), "na")
        if side in ("left", "right"):
            entry[side].append(row)
        entry["members"].append({
            "root_id": str(root_id), "side": side,
            "cell_type": info.get("cell_type", ""),
            "hemibrain_type": info.get("hemibrain_type", ""),
        })
    return groups


# --------------------------------------------------------------------------
# the tests
# --------------------------------------------------------------------------

def tests_ab(nav, clusters, side_of, types, *, label: str,
             sensory: list[str] | None = None) -> dict:
    """Test A (current readout) and Test B (side-aware), from one probe.

    `sensory` defaults to every sensory cluster, matching `calibrate_gain`.
    That is the right harness for an excitability sweep and the wrong one for
    a steering-bias reading: driving `looming_lc4`/`looming_lplc2` at 0.4
    saturates the LC4/LPLC2 -> DNp01 escape pathway, which zeroes throttle in
    `_decode` and is nothing like the live regime (the Phase 2 route run spent
    24,518 of 25,727 frames in IDLE DRIFT and only 15 in PANIC SWERVE). Pass
    NAVIGATION_SENSORY to read the bias at an operating point the vehicle
    actually occupies.
    """
    if sensory is None:
        sensory = [n for n, role in nav.roles.items() if role == "sensory"]
    rates, out = probe(nav, {name: DRIVE for name in sensory})

    payload = clusters["clusters"]
    dna01 = merge_clusters(payload, "steer_trim_l", "steer_trim_r")
    dna02 = merge_clusters(payload, "steer_left", "steer_right")

    a_left = mean_rate(rates, dna01["rows"])
    a_right = mean_rate(rates, dna02["rows"])

    s01 = split_by_side(dna01, side_of)
    s02 = split_by_side(dna02, side_of)
    collision = split_dna01_collision(dna01, types, side_of)

    corrected = mean_rate(rates, s02["right"]) - mean_rate(rates, s02["left"])
    genuine = collision.get("genuine_dna01", {})
    trim = (mean_rate(rates, genuine.get("right")) -
            mean_rate(rates, genuine.get("left"))) if genuine else None

    return {
        "label": label,
        "drive": DRIVE,
        "ticks": TICKS,
        "gain": nav.gain,
        "test_a_current_readout": {
            "formula": "rate(DNa02, both sides) - rate(DNa01, both sides)",
            "dna02_hz": a_right,
            "dna01_hz": a_left,
            "steer": round(a_right - a_left, 4),
            "steer_normalised": round((a_right - a_left) / 500.0, 6),
            "note": "what _decode returns today, before any fix",
        },
        "test_b_side_aware": {
            "dna02": {
                "left_hz": mean_rate(rates, s02["left"]),
                "right_hz": mean_rate(rates, s02["right"]),
                "n_left": len(s02["left"]), "n_right": len(s02["right"]),
                "corrected_steer_hz": round(corrected, 4),
                "corrected_steer_normalised": round(corrected / 500.0, 6),
            },
            "dna01_pooled": {
                "left_hz": mean_rate(rates, s01["left"]),
                "right_hz": mean_rate(rates, s01["right"]),
                "n_left": len(s01["left"]), "n_right": len(s01["right"]),
            },
            "dna01_collision": {
                key: {
                    "n": len(g["rows"]),
                    "left_hz": mean_rate(rates, g["left"]),
                    "right_hz": mean_rate(rates, g["right"]),
                    "members": g["members"],
                }
                for key, g in collision.items()
            },
            "dna01_genuine_trim_hz": None if trim is None else round(trim, 4),
        },
        "population": {
            "spike_rate_hz": round(out.spike_rate * 500.0, 3),
            "throttle_hz": round(out.throttle * 500.0, 3),
            "panic_hz": round(out.panic * 500.0, 3),
        },
        "wiring": wiring_report(nav, {"dna01": dna01, "dna02": dna02},
                                side_of, types),
    }


def wiring_report(nav, clusters: dict, side_of: dict, types: dict) -> dict:
    """Per-neuron in-degree and total input weight for the steering cells.

    A silent neuron has two very different explanations, and they call for
    different fixes: it is inhibited by a network that genuinely wires it that
    way, or it has almost no reconstructed input in this volume at all. The
    second is a completeness artefact of the EM segmentation, not biology, and
    a left/right readout built on it would be measuring the dataset rather
    than the fly. `_prepare` already computes both numbers; this just reports
    them per row.
    """
    out: dict[str, list] = {}
    for name, cluster in clusters.items():
        rows = []
        for root_id, row in zip(cluster["root_ids"], cluster["rows"]):
            info = types.get(str(root_id), {})
            rows.append({
                "root_id": str(root_id),
                "side": side_of.get(str(root_id), "na"),
                "cell_type": info.get("cell_type", ""),
                "hemibrain_type": info.get("hemibrain_type", ""),
                "in_degree": int(nav.in_degree[row]),
                "input_weight": round(float(nav.input_weight[row]), 1),
            })
        out[name] = sorted(rows, key=lambda r: r["side"])
    return out


def test_e(nav, clusters, side_of) -> dict:
    """Does a bilateral odour asymmetry reach DNa02 at all, and with what sign?

    This is the experiment the whole Step 3 plan rests on and nobody has run.
    The plan injects a signed steering drive into side-split DM1/DM4 and reads
    it back as `rate(DNa02_right) - rate(DNa02_left)`. Between those two points
    sit several synaptic hops through a 139k-neuron recurrent network, and
    nothing guarantees the signal survives them, keeps its sign, or arrives
    large enough to matter against the resting L/R gap measured in Test B.

    So: drive the left and right odour PNs antisymmetrically, sweep the
    contrast, and report what DNa02 actually does. Done by row index, with
    `CLUSTERS` untouched, exactly as Part 0 requires.
    """
    sensory = [n for n in NAVIGATION_SENSORY if n in nav.roles]
    base = {name: DRIVE for name in sensory if not name.startswith("odor_")}

    payload = clusters["clusters"]
    dm = {"left": [], "right": []}
    for name in ("odor_dm1", "odor_dm4"):
        split = split_by_side(payload[name], side_of)
        dm["left"] += split["left"]
        dm["right"] += split["right"]

    d02 = split_by_side(merge_clusters(payload, "steer_left",
                                       "steer_right"), side_of)
    d01 = split_by_side(merge_clusters(payload, "steer_trim_l",
                                       "steer_trim_r"), side_of)

    sweep = []
    for right_drive, left_drive in ((0.4, 0.0), (0.4, 0.1), (0.4, 0.2),
                                    (0.2, 0.2), (0.2, 0.4), (0.1, 0.4),
                                    (0.0, 0.4)):
        rates, _ = probe(nav, base, row_currents=[(dm["right"], right_drive),
                                                  (dm["left"], left_drive)])
        r = mean_rate(rates, d02["right"])
        left = mean_rate(rates, d02["left"])
        sweep.append({
            "dm_right_current": right_drive,
            "dm_left_current": left_drive,
            "contrast": round(right_drive - left_drive, 3),
            "dna02_right_hz": r,
            "dna02_left_hz": left,
            "dna02_r_minus_l_hz": round(r - left, 3),
            "dna01_right_hz": mean_rate(rates, d01["right"]),
            "dna01_left_hz": mean_rate(rates, d01["left"]),
        })

    spread = ([p["dna02_r_minus_l_hz"] for p in sweep])
    balanced = [p for p in sweep if p["contrast"] == 0.0]
    return {
        "dm_left_rows": len(dm["left"]), "dm_right_rows": len(dm["right"]),
        "sweep": sweep,
        "r_minus_l_range_hz": round(max(spread) - min(spread), 3),
        "r_minus_l_at_balance_hz": balanced[0]["dna02_r_minus_l_hz"] if balanced else None,
        "sign_follows_contrast": (
            spread[0] > spread[-1] if len(spread) > 1 else None),
        "note": (
            "r_minus_l_range_hz is how much authority a full bilateral odour "
            "swing has over the steering readout. Compare it against the "
            "resting R-L gap in Test B: if the range is far smaller, the "
            "directional signal cannot overcome the bias and the readout "
            "needs the bias subtracted, not just the sides split."
        ),
    }


def resolve_rows(cfg, neurons, pattern: str, side_of: dict) -> dict[str, list]:
    """Row indices for a `cell_type` regex, grouped by side. Read-only."""
    import re

    import numpy as np

    frame, _ = load_annotations(cfg)
    regex = re.compile(pattern)
    mask = frame["cell_type"].fillna("").map(lambda v: bool(regex.match(v)))
    ids = np.asarray([int(v) for v in frame.loc[mask, "root_id"]
                      if str(v).isdigit()], dtype=np.int64)
    ids = np.unique(ids)
    positions = np.clip(np.searchsorted(neurons, ids), 0, len(neurons) - 1)
    present = neurons[positions] == ids
    out: dict[str, list] = {"left": [], "right": [], "other": []}
    for root_id, row in zip(ids[present], positions[present]):
        side = side_of.get(str(root_id), "na")
        out[side if side in ("left", "right") else "other"].append(int(row))
    return out


def test_f(nav, cfg, neurons, clusters, side_of) -> dict:
    """The same bilateral question, asked one synapse from the readout.

    Test E shows DM1/DM4 has no authority over DNa02 -- six olfactory PNs
    cannot move a descending neuron whose in-degree is 742. PFL3 is the site
    the handoff's own cited source names: "PFL3 synapses directly onto DNa02",
    and "PFL3R cells are recruited when the fly is oriented to the left of its
    goal, and their activity drives rightward turning" (Nature 2024). v783
    curates 24 PFL3 neurons, split exactly 12 left / 12 right.

    Reports both the direct PFL3 -> DNa02 synaptic weights (is the wire even
    there, and with what sign) and what a bilateral PFL3 drive does to the
    steering readout.
    """
    pfl3 = resolve_rows(cfg, neurons, r"^PFL3$", side_of)
    d02 = split_by_side(merge_clusters(clusters["clusters"],
                                       "steer_left", "steer_right"),
                        side_of)

    # Direct connectivity, straight out of the CSR: W[post, pre].
    wiring = {}
    indptr = nav.w.crow_indices()
    cols, vals = nav.w.col_indices(), nav.w.values()
    for post_side in ("left", "right"):
        for row in d02[post_side]:
            lo, hi = int(indptr[row]), int(indptr[row + 1])
            row_cols = cols[lo:hi].tolist()
            row_vals = vals[lo:hi].tolist()
            index = {c: v for c, v in zip(row_cols, row_vals)}
            for pre_side in ("left", "right"):
                hits = [index[c] for c in pfl3[pre_side] if c in index]
                wiring[f"dna02_{post_side}<-pfl3_{pre_side}"] = {
                    "synapses": len(hits),
                    "summed_normalised_weight": round(sum(hits), 6),
                }

    sensory = [n for n in NAVIGATION_SENSORY if n in nav.roles]
    base = {name: DRIVE for name in sensory}
    sweep = []
    for right_drive, left_drive in ((0.4, 0.0), (0.4, 0.2), (0.2, 0.2),
                                    (0.2, 0.4), (0.0, 0.4)):
        rates, _ = probe(nav, base, row_currents=[(pfl3["right"], right_drive),
                                                 (pfl3["left"], left_drive)])
        r = mean_rate(rates, d02["right"])
        left = mean_rate(rates, d02["left"])
        sweep.append({
            "pfl3_right_current": right_drive,
            "pfl3_left_current": left_drive,
            "contrast": round(right_drive - left_drive, 3),
            "dna02_right_hz": r,
            "dna02_left_hz": left,
            "dna02_r_minus_l_hz": round(r - left, 3),
        })

    spread = [p["dna02_r_minus_l_hz"] for p in sweep]
    return {
        "pfl3_left_rows": len(pfl3["left"]),
        "pfl3_right_rows": len(pfl3["right"]),
        "direct_wiring": wiring,
        "sweep": sweep,
        "r_minus_l_range_hz": round(max(spread) - min(spread), 3),
        "sign_follows_contrast": spread[0] > spread[-1] if len(spread) > 1 else None,
    }


def calibrate_steer_bias(nav, clusters, side_of) -> dict:
    """Measure `snn.steer_bias`: the readout's resting value, dead ahead.

    `config.yaml` says this number is reproduced here, so it is produced here
    rather than transcribed. The operating point is the one the live loop
    actually sits at when the fly is pointed straight at the goal: navigation
    sensory at DRIVE, and both PFL3 populations driven equally at
    `world.SENSORY_CEILING` -- which is what `steering_drive(0)` yields once
    `sense()` clamps its two 0.707 lobes.

    Whatever `steer` reads under that balanced condition is, by definition,
    the offset that should decode as "no turn".
    """
    from dipteran.world import SENSORY_CEILING

    sensory = [n for n in NAVIGATION_SENSORY if n in nav.roles]
    pfl3_l = clusters["clusters"]["steer_drive_pfl3_l"]["rows"]
    pfl3_r = clusters["clusters"]["steer_drive_pfl3_r"]["rows"]
    rows_l = clusters["clusters"]["steer_left"]["rows"]
    rows_r = clusters["clusters"]["steer_right"]["rows"]

    # Dead ahead, `steering_drive(0)` gives both lobes cos(45) = 0.707,
    # which `sense()` scales by SENSORY_CEILING. That is the balanced
    # operating point, not a flat SENSORY_CEILING on both sides.
    balanced = math.cos(math.radians(45.0)) * SENSORY_CEILING
    rates, _ = probe(nav, {name: DRIVE for name in sensory},
                     row_currents=[(pfl3_l, balanced),
                                   (pfl3_r, balanced)])
    right, left = mean_rate(rates, rows_r), mean_rate(rates, rows_l)
    return {
        "balanced_pfl3_drive": round(balanced, 6),
        "dna02_right_hz": right,
        "dna02_left_hz": left,
        "resting_r_minus_l_hz": round(right - left, 3),
        "recommended_steer_bias": round((right - left) / 500.0, 4),
        "note": "copy recommended_steer_bias into config.yaml snn.steer_bias",
    }


def test_g_directional_response(nav, clusters, cfg) -> dict:
    """Step 4(i): does `steer` track *signed* heading error, end to end?

    "Does it move" is not the check. The Phase 2 vehicle moved 6,776 m and
    closed 103.6 m, because its steering was a constant. What has to be true
    is that a goal on the right decodes to a positive steer and a goal on the
    left to a negative one, through the whole path -- `steering_drive` ->
    scaled injection -> 139k-neuron network -> `_decode` minus `steer_bias`.

    Runs the real `world.sense()` arithmetic for the steering channel rather
    than a reimplementation, so a change to either stays in step. No SUMO: the
    only input that matters here is the heading error.
    """
    from dipteran.world import SENSORY_CEILING, steering_drive

    sensory = [n for n in NAVIGATION_SENSORY if n in nav.roles]
    pfl3_l = clusters["clusters"]["steer_drive_pfl3_l"]["rows"]
    pfl3_r = clusters["clusters"]["steer_drive_pfl3_r"]["rows"]

    curve = []
    for error in (-180, -135, -90, -45, -20, 0, 20, 45, 90, 135, 180):
        left, right = steering_drive(float(error))
        _, out = probe(nav, {name: DRIVE for name in sensory},
                       row_currents=[(pfl3_l, left * SENSORY_CEILING),
                                     (pfl3_r, right * SENSORY_CEILING)])
        command = max(-1.0, min(1.0, out.steer * cfg.vehicle.steer_scale))
        curve.append({
            "heading_error_deg": error,
            "pfl3_left_current": round(left * SENSORY_CEILING, 4),
            "pfl3_right_current": round(right * SENSORY_CEILING, 4),
            "steer": round(out.steer, 6),
            "steer_command": round(command, 6),
            "turn_deg_s": round(command * cfg.vehicle.max_turn_deg_s, 3),
        })

    def at(error):
        return next(p["steer"] for p in curve if p["heading_error_deg"] == error)

    return {
        "steer_bias": nav.steer_bias,
        "curve": curve,
        # The whole point of Phase 3, stated as a pass/fail.
        "right_goal_turns_right": at(45) > 0.0,
        "left_goal_turns_left": at(-45) < 0.0,
        "straight_is_neutral": abs(at(0)) < 0.01,
        "monotonic_between_lobes": all(
            at(a) <= at(b) for a, b in zip((-90, -45, -20, 0, 20, 45),
                                           (-45, -20, 0, 20, 45, 90))),
    }


def test_d(nav, clusters) -> dict:
    """PAM/PPL1 current -> firing rate. Decides the dopamine current mapping."""
    sensory = [n for n, role in nav.roles.items() if role == "sensory"]
    base = {name: DRIVE for name in sensory}
    pam_rows = clusters["clusters"]["reward_pam"]["rows"]
    ppl1_rows = clusters["clusters"]["punish_ppl1"]["rows"]

    curve = []
    for current in PAM_SWEEP:
        rates, out = probe(nav, {**base, "reward_pam": current})
        curve.append({
            "injected_current": current,
            "reward_pam_hz": mean_rate(rates, pam_rows),
            "punish_ppl1_hz": mean_rate(rates, ppl1_rows),
            "decoded_reward": round(out.reward, 6),
            "decoded_punishment": round(out.punishment, 6),
            "decoded_dopamine": round(out.dopamine, 6),
        })

    ppl1_curve = []
    for current in PAM_SWEEP:
        rates, out = probe(nav, {**base, "punish_ppl1": current})
        ppl1_curve.append({
            "injected_current": current,
            "punish_ppl1_hz": mean_rate(rates, ppl1_rows),
            "decoded_punishment": round(out.punishment, 6),
            "decoded_dopamine": round(out.dopamine, 6),
        })

    firing = [p for p in curve if p["reward_pam_hz"] > 0.0]
    baseline = curve[0]["reward_pam_hz"]
    above = [p for p in curve
             if p["injected_current"] > 0 and p["reward_pam_hz"] > baseline * 1.05]
    return {
        "sweep": list(PAM_SWEEP),
        "pam_curve": curve,
        "ppl1_curve": ppl1_curve,
        "pam_rows": len(pam_rows),
        "ppl1_rows": len(ppl1_rows),
        "baseline_pam_hz": baseline,
        "onset_current": above[0]["injected_current"] if above else None,
        "graded": len({round(p["reward_pam_hz"], 1) for p in firing}) > 2,
        "note": (
            "onset_current is the smallest injected current that lifts PAM "
            "more than 5% above its unstimulated rate. A proximity-scaled "
            "reward must land above it or it produces no dopamine at all."
        ),
    }


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def build(cfg, *, min_syn_count: int | None = None):
    pairs = aggregate_pairs(cfg, min_syn_count=min_syn_count)
    built = build_adjacency(cfg, pairs=pairs)
    clusters = resolve_clusters(cfg, neurons=built["neurons"])
    mono = monoamine_rows(cfg, built["neurons"])
    nav = FlyBangaloreNavigator.from_config(cfg, built["W"], built["neurons"],
                                            clusters, monoamine=mono)
    return built, clusters, nav


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-min-syn", action="store_true",
                        help="skip Test C (saves a full re-aggregation)")
    args = parser.parse_args()

    cfg = load_config()
    side_of, types = annotation_index(cfg)
    print(f"[probe] annotation side column: "
          f"{sum(s == 'left' for s in side_of.values()):,} left / "
          f"{sum(s == 'right' for s in side_of.values()):,} right", flush=True)

    result = {
        "generated_at": dt.datetime.now(dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "baseline_route_run": "cache/route_run.json",
        "config_gain": cfg.snn.gain,
        "config_min_syn_count": cfg.compute.min_syn_count,
    }

    built, clusters, nav = build(cfg)
    result["baseline"] = tests_ab(
        nav, clusters, side_of, types,
        label=f"min_syn_count={cfg.compute.min_syn_count}, all sensory")
    result["baseline_navigation"] = tests_ab(
        nav, clusters, side_of, types,
        label=f"min_syn_count={cfg.compute.min_syn_count}, navigation only",
        sensory=list(NAVIGATION_SENSORY))
    result["test_g_directional"] = test_g_directional_response(
        nav, clusters, cfg)
    result["steer_bias_calibration"] = calibrate_steer_bias(
        nav, clusters, side_of)
    result["test_e_bilateral_authority"] = test_e(nav, clusters, side_of)
    result["test_f_pfl3_authority"] = test_f(nav, cfg, built["neurons"],
                                             clusters, side_of)
    result["test_d_pam_transfer"] = test_d(nav, clusters)
    result["baseline"]["neurons"] = int(built["stats"]["neurons"])
    result["baseline"]["nnz"] = int(built["stats"]["nnz"])

    if not args.skip_min_syn:
        # build_adjacency and resolve_clusters both write cached artefacts, and
        # a min_syn_count=5 graph has a *different* neuron set -- letting that
        # overwrite the checked-in cache would silently corrupt every later run.
        backups = []
        for path in (cfg.paths.neuron_index, cfg.paths.cluster_ids):
            if path.exists():
                spare = path.with_suffix(path.suffix + ".probe-backup")
                shutil.copy2(path, spare)
                backups.append((path, spare))

        import torch
        del nav, built
        torch.cuda.empty_cache()

        try:
            built5, clusters5, nav5 = build(cfg, min_syn_count=5)
            result["min_syn_5"] = tests_ab(
                nav5, clusters5, side_of, types,
                label="min_syn_count=5, navigation only",
                sensory=list(NAVIGATION_SENSORY))
            result["min_syn_5"]["neurons"] = int(built5["stats"]["neurons"])
            result["min_syn_5"]["nnz"] = int(built5["stats"]["nnz"])
        finally:
            for path, spare in backups:
                shutil.copy2(spare, path)
                spare.unlink()
            print(f"[probe] restored {len(backups)} cached artefact(s)",
                  flush=True)

    out = cfg.paths.cache_dir / "steering_bias_diagnostic.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    for key in ("baseline", "baseline_navigation", "min_syn_5"):
        if key not in result:
            continue
        base = result[key]
        a = base["test_a_current_readout"]
        b = base["test_b_side_aware"]["dna02"]
        print(f"\n--- {base['label']} ---")
        print(f"  A  rate(DNa02) - rate(DNa01) = {a['steer']:+9.3f} Hz "
              f"(normalised {a['steer_normalised']:+.6f})   <- drives the vehicle today")
        print(f"  B  DNa02  left {b['left_hz']:9.3f} Hz   right {b['right_hz']:9.3f} Hz")
        print(f"     corrected R-L         = {b['corrected_steer_hz']:+9.3f} Hz "
              f"(normalised {b['corrected_steer_normalised']:+.6f})")
        for name, g in base["test_b_side_aware"]["dna01_collision"].items():
            print(f"     DNa01/{name:<16} n={g['n']}  "
                  f"left {g['left_hz']:9.3f} Hz   right {g['right_hz']:9.3f} Hz")
        print(f"     panic {base['population']['panic_hz']:.1f} Hz   "
              f"throttle {base['population']['throttle_hz']:.1f} Hz")
        for name, rows in base["wiring"].items():
            for r in rows:
                print(f"     wiring {name:<14} {r['side']:<5} "
                      f"in_degree {r['in_degree']:>6}  "
                      f"input_weight {r['input_weight']:>10.1f}  "
                      f"({r['cell_type']}/{r['hemibrain_type']})")

    e = result["test_e_bilateral_authority"]
    print("\n--- Test E: can a bilateral odour asymmetry steer DNa02? ---")
    for p in e["sweep"]:
        print(f"  DM R={p['dm_right_current']:<4} L={p['dm_left_current']:<4} "
              f"(contrast {p['contrast']:+.1f}) -> "
              f"DNa02 R {p['dna02_right_hz']:8.3f}  L {p['dna02_left_hz']:8.3f}  "
              f"R-L {p['dna02_r_minus_l_hz']:+9.3f} Hz")
    print(f"  authority over R-L: {e['r_minus_l_range_hz']:.3f} Hz across the "
          f"full swing; sign follows contrast: {e['sign_follows_contrast']}")

    print("\n--- Test D: PAM current -> firing rate ---")
    for point in result["test_d_pam_transfer"]["pam_curve"]:
        print(f"  i_ext {point['injected_current']:<5} -> "
              f"PAM {point['reward_pam_hz']:8.3f} Hz   "
              f"dopamine {point['decoded_dopamine']:+.6f}")
    print(f"  onset current: {result['test_d_pam_transfer']['onset_current']}, "
          f"graded: {result['test_d_pam_transfer']['graded']}")
    print(f"\n[probe] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
