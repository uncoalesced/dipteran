#!/usr/bin/env python3
"""Re-run a fixed roster of routes and diff the figures against what is on
record.

Joel's framing, Handoff_22-4pm item 6: "we have to do continuous testing to
make sure it is stable and working," with routes getting harder over time.

This deliberately does **not** invent a pass/fail bar. PLAN.md section 9
already rejected "must arrive at Point B" as one, on the grounds that tuning
the fly until it beelines to the goal turns it into an artificial
potential-field pathfinder wearing SNN clothing. Circling, wandering and
getting stuck are the project, not bugs in it. So this flags drift for a
human to look at and always exits 0 -- it is a smoke alarm, not a gate.

Everything it reads is already written by `run_route.py`. No new
instrumentation, no new telemetry: this is a diff on top of existing output.

    python scripts/regression_check.py --compare-only   # diff what is on disk
    python scripts/regression_check.py                  # re-run, then diff
    python scripts/regression_check.py --only oneturn_500m

Engineered by uncoalesced.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BASELINES = Path(__file__).resolve().parent / "baselines.json"

# How far a figure may move before it is worth a human's attention.
#
# "abs" compares raw difference, "rel" a fraction of the baseline, "order" a
# factor. These are a first guess off two or three runs each -- they are not
# measured tolerances and should tighten once the roster has more history.
# ponytail: hand-picked thresholds, replace with per-metric spread once there
# are enough runs on record to compute one.
TOLERANCES = {
    "episodes_arrived":          ("abs", 2.0),
    "frames_to_arrival_mean":    ("rel", 0.15),
    "closed_route_m_mean":       ("rel", 0.25),
    "heading_error_abs_median":  ("rel", 0.20),
    "steer_sign_agreement":      ("abs", 0.05),
    "no_route_frames_mean":      ("rel", 0.50),
    "travelled_m":               ("rel", 0.30),
    "plastic_drift_relative":    ("order", 10.0),
}


def _mean(values) -> float | None:
    kept = [v for v in values if v is not None]
    return round(statistics.mean(kept), 3) if kept else None


def metrics(run: dict) -> dict:
    """The figures worth watching, pulled out of a `run_route.py` result."""
    episodes = run.get("episodes", [])
    arrived = [e for e in episodes if e.get("arrived")]
    return {
        "episodes_arrived": run.get("episodes_arrived"),
        "episode_count": run.get("episode_count"),
        # Only arriving episodes have a meaningful frame count: a timeout
        # reports the episode cap, which says nothing about how fast the fly
        # got there.
        "frames_to_arrival_mean": _mean(e.get("frames") for e in arrived),
        "closed_route_m_mean": _mean(e.get("closed_route_m") for e in episodes),
        "heading_error_abs_median": run.get("heading_error_abs_median"),
        "steer_sign_agreement": run.get("steer_sign_agreement"),
        "no_route_frames_mean": _mean(e.get("no_route_frames")
                                      for e in episodes),
        "internal_edge_frames_mean": _mean(e.get("internal_edge_frames")
                                           for e in episodes),
        "travelled_m": run.get("travelled_m"),
        "plastic_drift_relative": run.get("plastic_drift_relative"),
    }


def drifted(field: str, was, now) -> str | None:
    """A one-line description of how far `now` has moved, or None."""
    if was is None or now is None:
        if was == now:
            return None
        return f"{field}: {was!r} -> {now!r} (one side was never recorded)"
    kind, limit = TOLERANCES.get(field, (None, None))
    if kind is None:
        return None
    if kind == "abs":
        if abs(now - was) > limit:
            return (f"{field}: {was} -> {now}  "
                    f"(moved {now - was:+.3f}, limit +/-{limit})")
    elif kind == "rel":
        if was == 0:
            return None if now == 0 else f"{field}: 0 -> {now}"
        change = (now - was) / abs(was)
        if abs(change) > limit:
            return (f"{field}: {was} -> {now}  "
                    f"({change:+.0%}, limit +/-{limit:.0%})")
    elif kind == "order":
        if was == 0 or now == 0:
            return None
        factor = now / was
        if factor > limit or factor < 1.0 / limit:
            return f"{field}: {was} -> {now}  ({factor:.1f}x, limit {limit:.0f}x)"
    return None


def invariants(run: dict) -> list[str]:
    """The checks that are about arithmetic, not taste.

    `closed_route_m <= travelled_m` is the one that exposed the 1,862 m unit
    switch on 2026-09-19: you cannot close more route than you drove.
    `run_route.py` asserts it at the episode boundary, but a result file
    written before that assertion existed has never been checked.
    """
    broken = []
    for e in run.get("episodes", []):
        closed, travelled = e.get("closed_route_m"), e.get("travelled_m")
        if closed is None or travelled is None:
            continue
        if closed > travelled + 1.0:
            broken.append(
                f"episode {e.get('episode')}: closed {closed} m of route on "
                f"{travelled} m travelled -- impossible")
    return broken


def rerun(name: str, spec: dict, out: Path) -> bool:
    cmd = [sys.executable, str(REPO / "scripts" / "run_route.py"),
           "--origin", spec["origin"], "--dest", spec["dest"],
           "--episodes", str(spec["episodes"]),
           "--episode-minutes", str(spec["episode_minutes"]),
           "--max-minutes", str(spec["max_minutes"]),
           "--out", str(out)]
    print(f"[regression] re-running {name}: {' '.join(cmd[2:])}", flush=True)
    # One SUMO process at a time: two concurrent ones have taken the whole
    # WSL2 distro down (Wsl/Service/E_UNEXPECTED). Sequential by construction.
    return subprocess.run(cmd, cwd=REPO, check=False).returncode == 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--compare-only", action="store_true",
                    help="diff the result files already on disk, run nothing")
    ap.add_argument("--only", metavar="NAME",
                    help="one roster entry instead of all of them")
    ap.add_argument("--baselines", type=Path, default=BASELINES)
    args = ap.parse_args()

    book = json.loads(args.baselines.read_text(encoding="utf-8"))
    roster = book["routes"]
    if args.only:
        if args.only not in roster:
            print(f"[regression] no roster entry {args.only!r}; have: "
                  f"{', '.join(roster)}")
            return 0
        roster = {args.only: roster[args.only]}

    findings: list[str] = []
    for name, spec in roster.items():
        print(f"\n=== {name} -- {spec['description']}", flush=True)
        result = REPO / spec["source"]
        if not args.compare_only:
            fresh = REPO / "runs" / f"{name}_latest.json"
            if rerun(name, spec, fresh):
                result = fresh
            else:
                print(f"[regression] {name} did not complete; "
                      f"falling back to {spec['source']}")

        if not result.exists():
            print(f"    no result file at {result} -- nothing to compare")
            findings.append(f"{name}: missing {result}")
            continue

        run = json.loads(result.read_text(encoding="utf-8"))
        now = metrics(run)
        was = spec["expect"]
        print(f"    against {spec['source']}, recorded {spec['recorded']}")

        for problem in invariants(run):
            print(f"    !! {problem}")
            findings.append(f"{name}: {problem}")

        for field in TOLERANCES:
            note = drifted(field, was.get(field), now.get(field))
            print(f"    {'DRIFT' if note else 'ok   '} "
                  f"{note or f'{field}: {now.get(field)}'}")
            if note:
                findings.append(f"{name}: {note}")

        for caveat in spec.get("caveats", []):
            print(f"    note: {caveat}")

    print("\n" + "=" * 70)
    if findings:
        print(f"{len(findings)} thing(s) moved -- worth a look, not a failure:")
        for f in findings:
            print(f"  - {f}")
    else:
        print("nothing moved past its tolerance")
    print("=" * 70)
    # Always 0. See the module docstring: this surfaces drift, it does not
    # gate anything.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
