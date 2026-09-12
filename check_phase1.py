#!/usr/bin/env python
"""Phase 1 exit criteria, asserted rather than eyeballed.

    python check_phase1.py                 # every check
    python check_phase1.py --only id_map   # one check
    python check_phase1.py --list          # what the checks are

Each check either produces evidence or raises. Nothing here reports success
from the mere existence of a file. Exits non-zero if any check fails, and
writes cache/phase1_evidence.json with whatever every check actually saw.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from dipteran.config import load_config  # noqa: E402

# Parsing an 862 MB .net.xml and a 154 MB .graphml takes minutes each, and
# four of the checks below want them. Build once per process.
_CACHE: dict[str, object] = {}


def _graph():
    if "graph" not in _CACHE:
        from dipteran.roads import build_osmnx_graph
        _CACHE["graph"] = build_osmnx_graph(load_config())
    return _CACHE["graph"]


def _net():
    if "net" not in _CACHE:
        from dipteran.roads import SumoNet, build_sumo_net
        cfg = load_config()
        _CACHE["net"] = SumoNet(build_sumo_net(cfg))
    return _CACHE["net"]

# Each check is (name, the exit criterion it answers, its function).
CRITERIA: list[tuple[str, str, str]] = [
    ("config", "config.yaml validates; no lat/lon constants in code",
     "check_config"),
    ("scaffold", "python main.py serves the mock WebSocket + MapLibre frontend",
     "check_scaffold"),
    ("osmnx", "OSMnx graph loads cleanly for the configured route",
     "check_osmnx"),
    ("sumo_net", "netconvert produces a valid, routable .net.xml",
     "check_sumo_net"),
    ("id_map", "deterministic OSM-edge<->SUMO-edge lookup matching on geometry",
     "check_id_map"),
    ("connectome", "connectome loaded as a GPU-resident sparse CSR, VRAM logged",
     "check_connectome"),
    ("cell_ids", "every sensory/motor cluster resolves to a valid row index",
     "check_cell_ids"),
]


# --------------------------------------------------------------------------

def check_config() -> dict:
    cfg = load_config()
    r = cfg.route
    assert r.bbox.contains(r.origin), "origin outside bbox"
    assert r.bbox.contains(r.destination), "destination outside bbox"
    assert r.dist_m >= r.separation_m, "extract radius cannot reach destination"

    # The frontend's hardcoded bounds must be the configured ones, or the map
    # will refuse to show the agent where the backend puts it. Compare the
    # numbers, not the source text -- 77.40 and 77.4 are the same bound.
    html = (REPO_ROOT / "index.html").read_text(encoding="utf-8")
    match = re.search(r"BANGALORE_BOUNDS\s*=\s*(\[[^;]*?\])\s*;", html, re.S)
    assert match, "index.html does not define BANGALORE_BOUNDS"
    found = [[float(v) for v in pair]
             for pair in json.loads(match.group(1).replace("'", '"'))]
    expected = r.bbox.as_maplibre()
    assert found == expected, (
        f"index.html BANGALORE_BOUNDS is {found}, route.bbox is {expected}"
    )
    return {
        "route": r.name,
        "origin": [r.origin.lat, r.origin.lon],
        "destination": [r.destination.lat, r.destination.lon],
        "separation_km": round(r.separation_m / 1000, 2),
        "device": cfg.compute.device,
        "dtype": cfg.compute.dtype,
        "frontend_bounds_match": True,
    }


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def check_scaffold(timeout: float = 45.0) -> dict:
    """Start main.py for real and read the mock stream off the socket."""
    try:
        from websockets.sync.client import connect
    except ImportError as exc:
        raise AssertionError(
            "websockets missing; install uvicorn[standard]"
        ) from exc

    port = _free_port()
    env = {**os.environ, "DIPTERAN_PORT": str(port), "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(
        [sys.executable, "main.py"], cwd=REPO_ROOT, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        base = f"http://127.0.0.1:{port}"
        deadline = time.monotonic() + timeout
        html = None
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise AssertionError(
                    f"main.py exited early ({proc.returncode}):\n"
                    f"{proc.stdout.read()[-2000:]}"
                )
            try:
                with urllib.request.urlopen(base + "/", timeout=2) as response:
                    html = response.read().decode("utf-8", "replace")
                break
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                time.sleep(0.3)
        assert html is not None, f"main.py never answered on {base} in {timeout}s"

        for needle in ("maplibregl.Map", "BANGALORE_BOUNDS", 'id="terminal"'):
            assert needle in html, f"served page is missing {needle!r}"

        cfg = load_config()
        frames = []
        with connect(f"ws://127.0.0.1:{port}/ws", open_timeout=10) as ws:
            for _ in range(5):
                frames.append(json.loads(ws.recv(timeout=5)))

        required = {"lat", "lon", "heading", "sugar_reward", "pain_level",
                    "giant_fiber", "event"}
        for frame in frames:
            missing = required - set(frame)
            assert not missing, f"payload missing {missing}"
            assert cfg.route.bbox.south <= frame["lat"] <= cfg.route.bbox.north, \
                f"streamed lat {frame['lat']} outside the configured bbox"
            assert cfg.route.bbox.west <= frame["lon"] <= cfg.route.bbox.east, \
                f"streamed lon {frame['lon']} outside the configured bbox"
            assert 0.0 <= frame["heading"] < 360.0, frame["heading"]

        moved = any(f["lat"] != frames[0]["lat"] or f["lon"] != frames[0]["lon"]
                    for f in frames[1:])
        assert moved, "the mock streamer never moved the agent"
        return {
            "port": port,
            "html_bytes": len(html),
            "frames_read": len(frames),
            "payload_keys": sorted(required),
            "agent_moved": moved,
        }
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def check_osmnx() -> dict:
    import networkx as nx

    from dipteran.roads import route_endpoints

    cfg = load_config()
    graph = _graph()
    assert graph.number_of_nodes() > 0, "empty graph"
    assert graph.number_of_edges() > 0, "graph has no edges"

    orig, dest = route_endpoints(cfg, graph)
    assert orig in graph, orig
    assert dest in graph, dest
    assert orig != dest, "origin and destination snapped to the same node"

    # "Loads cleanly for the configured route" has to mean the route exists.
    assert nx.has_path(graph, orig, dest), (
        f"no directed path from node {orig} to {dest} in the OSMnx graph"
    )
    length_m = nx.shortest_path_length(graph, orig, dest, weight="length")

    # Every edge needs the attributes the SNN will read each tick.
    sample = list(graph.edges(keys=True, data=True))[:1000]
    missing_length = [e[:3] for e in sample if "length" not in e[3]]
    assert not missing_length, f"edges without length: {missing_length[:5]}"

    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "strongly_connected_components":
            nx.number_strongly_connected_components(graph),
        "origin_node": orig,
        "destination_node": dest,
        "shortest_path_km": round(length_m / 1000, 2),
        "straight_line_km": round(cfg.route.separation_m / 1000, 2),
    }


def check_sumo_net() -> dict:
    from dipteran.roads import build_sumo_net, verify_routable

    cfg = load_config()
    net_xml = build_sumo_net(cfg)
    assert net_xml.exists() and net_xml.stat().st_size > 0, net_xml

    net = _net()
    assert net.edge_ids, ".net.xml contains no non-internal edges"
    assert net.proj_params, ".net.xml has no projParameter"
    assert not net.skipped_no_shape, (
        f"{net.skipped_no_shape} edges have no lane geometry"
    )

    routing = verify_routable(cfg, net)
    assert routing["route_edges"] > 1, routing
    return {
        "net_xml_mb": round(net_xml.stat().st_size / 1e6, 1),
        "edges": len(net.edge_ids),
        "shape_points": int(len(net.lonlat)),
        "proj": net.proj_params,
        **routing,
    }


def check_id_map() -> dict:
    from dipteran.roads import build_id_map, verify_determinism, verify_id_map

    cfg = load_config()
    graph = _graph()
    net = _net()
    id_map = build_id_map(cfg, graph=graph, net=net, force=True)

    counts = id_map["counts"]
    coverage = counts["mapped"] / max(counts["osmnx_edges"], 1)
    assert coverage >= 0.95, (
        f"only {coverage:.1%} of OSMnx edges mapped to any SUMO edge"
    )
    assert id_map["strategy"] == "origId", (
        f"join fell back to {id_map['strategy']!r}; netconvert should be run "
        "with --output.original-names true"
    )

    geometry = verify_id_map(cfg, id_map, graph, net)
    determinism = verify_determinism(cfg, graph, net)
    return {
        "strategy": id_map["strategy"],
        "coverage": round(coverage, 4),
        **counts,
        "refinement": id_map["refinement"],
        "geometry": {k: v for k, v in geometry.items()
                     if k != "examples_where_geometry_disagrees"},
        "geometry_disagreements":
            geometry["examples_where_geometry_disagrees"],
        **determinism,
    }


def check_connectome() -> dict:
    import torch

    from dipteran.connectome import build_adjacency, fetch_flywire_data

    cfg = load_config()
    fetch_flywire_data(cfg)
    built = build_adjacency(cfg)
    w, stats = built["W"], built["stats"]

    assert w.layout is torch.sparse_csr, f"W is {w.layout}, not sparse CSR"
    assert w.device.type == "cuda", (
        f"W is on {w.device}; the Phase 1 criterion is GPU-resident"
    )
    assert stats["neurons"] > 100_000, stats["neurons"]
    assert stats["nnz"] > 1_000_000, stats["nnz"]
    # The sizing decision this checks: an aggregate-per-pair matrix, not a
    # per-synapse one. v783 has 54.5M synapses over 15.1M connected pairs, so
    # "did aggregation happen" is the right question, not a raw nnz bound --
    # a per-synapse matrix would have exactly one synapse per entry.
    assert stats["synapses_per_pair"] > 1.0, (
        f"{stats['synapses_per_pair']:.2f} synapses per non-zero -- this is a "
        "per-synapse matrix, not one entry per connected neuron pair"
    )
    assert stats["nnz"] < 20_000_000, (
        f"{stats['nnz']:,} non-zeros exceeds the number of connected pairs "
        "in v783"
    )
    assert stats["matrix_bytes_allocated"] < stats["vram_total_bytes"] // 4, (
        "the matrix is using more than a quarter of VRAM; the sizing "
        "assumption in the Phase 1 brief was wrong"
    )

    # Ten sparse matmuls, the same shape of work one 20ms SNN tick does.
    # Weights are raw aggregate synapse counts (up to 2405) and in-degree runs
    # to hundreds, so an unnormalised fp16 product saturates to inf within a
    # couple of steps and every subsequent check would be measuring nothing.
    # Rescaling to unit max each step keeps the kernel exercised on real
    # values -- and is a standing note for Phase 2, whose LIF calibration has
    # to scale these weights for the same reason.
    started = time.monotonic()
    v = torch.zeros(stats["neurons"], 1, dtype=w.dtype, device=w.device)
    v[::1000] = 1.0
    finite = True
    for _ in range(10):
        v = torch.sparse.mm(w, v)
        finite = finite and bool(torch.isfinite(v).all())
        scale = v.abs().max()
        if scale > 0:
            v = v / scale
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    assert finite, (
        "sparse matmul produced inf/nan even with per-step rescaling; the "
        f"{cfg.compute.dtype} matrix cannot be driven at this weight scale"
    )
    assert int((v != 0).sum()) > 0, "ten matmuls drove no neuron at all"

    return {
        "matmul_stayed_finite": finite,
        "driven_neurons_after_10_steps": int((v != 0).sum()),
        **stats,
        "matrix_mib": round(stats["matrix_bytes_allocated"] / 2**20, 2),
        "predicted_mib": round(stats["matrix_bytes_predicted"] / 2**20, 2),
        "vram_total_gib": round(stats["vram_total_bytes"] / 2**30, 2),
        "ten_sparse_mm_seconds": round(elapsed, 4),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "capability": "sm_%d%d" % torch.cuda.get_device_capability(0),
    }


def check_cell_ids() -> dict:
    import numpy as np

    from dipteran.connectome import CLUSTERS, resolve_clusters, sensory_motor_ids

    cfg = load_config()
    neurons = np.load(cfg.paths.neuron_index)
    payload = resolve_clusters(cfg, neurons=neurons)
    clusters = payload["clusters"]

    assert set(clusters) == set(CLUSTERS), (
        f"resolved {sorted(clusters)} but CLUSTERS declares {sorted(CLUSTERS)}"
    )
    for name, entry in clusters.items():
        assert entry["rows"], f"{name} resolved to no matrix row"
        assert entry["root_ids"], f"{name} resolved to no FlyWire root ID"
        assert len(entry["rows"]) == len(entry["root_ids"]), name
        assert max(entry["rows"]) < len(neurons), f"{name} row out of range"
        assert min(entry["rows"]) >= 0, f"{name} negative row"
        # Every row must round-trip back to the root ID it claims to be.
        for root_id, row in zip(entry["root_ids"], entry["rows"]):
            assert int(neurons[row]) == int(root_id), (
                f"{name}: row {row} is neuron {neurons[row]}, not {root_id}"
            )

    sensory, motor = sensory_motor_ids(payload)
    assert sensory and motor, "sensory or motor side is empty"

    fallbacks = {n: e["why_fallback"] for n, e in clusters.items()
                 if e["match_kind"] == "fallback"}
    return {
        "neurons_in_matrix": int(len(neurons)),
        "annotation_source": payload["annotation_source"],
        "type_columns": payload["type_columns"],
        "sensory": {k: len(v) for k, v in sensory.items()},
        "motor": {k: len(v) for k, v in motor.items()},
        "patterns": {n: e["pattern"] for n, e in clusters.items()},
        "fallbacks_used": fallbacks,
        "example_root_ids": {n: e["root_ids"][:3] for n, e in clusters.items()},
    }


# --------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", action="append", default=None,
                        help="run only these checks (repeatable)")
    parser.add_argument("--skip", action="append", default=[],
                        help="skip these checks (repeatable)")
    parser.add_argument("--list", action="store_true",
                        help="list the checks and exit")
    args = parser.parse_args()

    if args.list:
        for name, criterion, _ in CRITERIA:
            print(f"{name:<12} {criterion}")
        return 0

    selected = [c for c in CRITERIA
                if (args.only is None or c[0] in args.only)
                and c[0] not in args.skip]
    if not selected:
        print("no checks selected", file=sys.stderr)
        return 2

    results = []
    for name, criterion, func_name in selected:
        print(f"\n{'=' * 72}\n{name}: {criterion}\n{'=' * 72}", flush=True)
        started = time.monotonic()
        try:
            evidence = globals()[func_name]()
            status, error = "PASS", None
        except Exception as exc:
            evidence, status, error = {}, "FAIL", f"{type(exc).__name__}: {exc}"
            print(f"\n!! {error}", flush=True)
        results.append({
            "name": name, "criterion": criterion, "status": status,
            "seconds": round(time.monotonic() - started, 2),
            "evidence": evidence, "error": error,
        })

    cfg = load_config()
    out = cfg.paths.cache_dir / "phase1_evidence.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "generated_at": dt.datetime.now(dt.timezone.utc)
                          .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "results": results,
    }, indent=2, default=str), encoding="utf-8")

    print(f"\n{'=' * 72}\nPHASE 1 EXIT CRITERIA\n{'=' * 72}")
    for entry in results:
        print(f"[{entry['status']}] {entry['name']:<12} "
              f"{entry['seconds']:>7.1f}s  {entry['criterion']}")
        if entry["error"]:
            print(f"         {entry['error']}")
    failed = [e["name"] for e in results if e["status"] == "FAIL"]
    print(f"\nevidence written to {out}")
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        return 1
    print("all selected criteria pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
