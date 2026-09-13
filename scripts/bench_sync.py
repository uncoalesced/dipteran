"""Phase 2 pre-implementation bench: does the locked SUMO/SNN sync actually hold?

PLAN.md locks the cadence -- SUMO at --step-length 0.1 (10 Hz), the SNN at
50 Hz (20 ms tick), one batched traci sample+control on every 5th SNN tick --
but PHASE1_REPORT.md is explicit that it has never been run against live
traci with traffic demand. It also frames the budget wrongly: it asks whether
SUMO "fits in the remaining ~10 ms" of a 20 ms tick, when SUMO only steps
once per *five* SNN ticks. The real frame is a 100 ms sim-time window:

    5 SNN ticks + 1 SUMO step + 1 traci sample/control  <=  100 ms wall

This measures each term, and the thing Phase 1 found actually binds -- host
RAM, with SUMO's 862 MB network resident *alongside* torch and W.

Stages run independently because Phase 1 learned the hard way that two heavy
stages in one process get OOM-killed:

    python scripts/bench_sync.py --stage snn      # SNN half only
    python scripts/bench_sync.py --stage sumo     # SUMO half only
    python scripts/bench_sync.py --stage combined # both resident, real cadence
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

SUMO_TOOLS = Path(os.environ.get("SUMO_HOME", "/usr/share/sumo")) / "tools"
if SUMO_TOOLS.is_dir():
    sys.path.insert(0, str(SUMO_TOOLS))

STEP_LENGTH = 0.1        # SUMO, from config.yaml
SNN_HZ = 50              # 20 ms tick
TICKS_PER_STEP = 5       # control on the 100 ms boundary
LIF_SUBTICKS = 10        # sub-ticks per SNN tick, as benched in Phase 1
FRAME_BUDGET_MS = 1000.0 / SNN_HZ * TICKS_PER_STEP   # 100 ms of sim per frame


# --------------------------------------------------------------------------
# memory probes
# --------------------------------------------------------------------------

def rss_mb(pid: int | None = None) -> float:
    """Current RSS in MiB. Linux only -- this bench is WSL2-only by design."""
    pid = pid if pid is not None else os.getpid()
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    except (FileNotFoundError, ProcessLookupError):
        return 0.0
    return 0.0


def peak_rss_mb(pid: int | None = None) -> float:
    """High-water RSS in MiB -- VmHWM, so a freed spike still shows up."""
    pid = pid if pid is not None else os.getpid()
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024
    except (FileNotFoundError, ProcessLookupError):
        return 0.0
    return 0.0


def host_total_mb() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemTotal:"):
            return int(line.split()[1]) / 1024
    return 0.0


# --------------------------------------------------------------------------
# SNN half
# --------------------------------------------------------------------------

def build_snn(cfg):
    """Load the real W onto the GPU and return a tick() closure."""
    import torch

    from dipteran.connectome import build_adjacency

    started = time.monotonic()
    payload = build_adjacency(cfg)
    build_s = time.monotonic() - started

    w = payload["W"]
    n = payload["stats"]["neurons"]
    device = w.device
    dtype = w.dtype

    # LIF state. Thresholds are NOT calibrated here -- PHASE1_REPORT.md flags
    # that v_thresh=1.0 against a weight of 2405 will not behave. This bench
    # only needs the arithmetic to have the real shape and cost, so the scale
    # comes off the matrix itself rather than pretending it is tuned.
    scale = float(torch.abs(w.values()).mean())
    v = torch.zeros(n, device=device, dtype=dtype)
    spikes = torch.zeros(n, device=device, dtype=dtype)
    i_ext = torch.zeros(n, device=device, dtype=dtype)
    decay = 0.9
    v_thresh = scale * 8.0

    def tick() -> float:
        """One 20 ms SNN tick = LIF_SUBTICKS sparse mv + LIF update."""
        nonlocal v, spikes
        for _ in range(LIF_SUBTICKS):
            current = torch.mv(w, spikes) + i_ext
            v = v * decay + current
            spikes = (v >= v_thresh).to(dtype)
            v = v * (1.0 - spikes)          # reset where it fired
        return float(spikes.sum())

    return {
        "tick": tick,
        "stats": payload["stats"],
        "build_s": build_s,
        "torch": torch,
        "device": device,
        "sensory": i_ext,
        "n": n,
    }


def bench_snn(cfg, ticks: int = 200) -> dict:
    snn = build_snn(cfg)
    torch = snn["torch"]
    tick = snn["tick"]

    for _ in range(10):                      # warm up kernels/autotune
        tick()
    torch.cuda.synchronize()

    samples = []
    for _ in range(ticks):
        t0 = time.perf_counter()
        tick()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1000)

    vram_mb = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0
    return {
        "tick_ms_median": round(statistics.median(samples), 3),
        "tick_ms_p95": round(sorted(samples)[int(len(samples) * 0.95)], 3),
        "tick_ms_max": round(max(samples), 3),
        "ticks": ticks,
        "lif_subticks": LIF_SUBTICKS,
        "build_s": round(snn["build_s"], 1),
        "neurons": snn["stats"]["neurons"],
        "nnz": snn["stats"]["nnz"],
        "vram_peak_mb": round(vram_mb, 1),
        "host_rss_mb": round(rss_mb(), 1),
        "host_peak_rss_mb": round(peak_rss_mb(), 1),
    }


# --------------------------------------------------------------------------
# SUMO half
# --------------------------------------------------------------------------

def start_sumo(cfg, port: int = 8813):
    """Launch sumo ourselves so we own the pid and can read its VmHWM."""
    import traci

    binary = os.environ.get("SUMO_BINARY", "sumo")
    cmd = [
        binary,
        "-n", str(cfg.paths.sumo_net),
        "--step-length", str(STEP_LENGTH),
        "--remote-port", str(port),
        "--no-step-log", "true",
        "--no-warnings", "true",
        "--ignore-route-errors", "true",
        "--time-to-teleport", "300",
        "--collision.action", "warn",
    ]
    started = time.monotonic()
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    # netconvert's output is 862 MB; sumo takes a while to parse it, so the
    # connect has to retry rather than assume the port is up.
    deadline = time.monotonic() + 900
    while True:
        if proc.poll() is not None:
            raise RuntimeError(f"sumo exited {proc.returncode}: "
                               f"{proc.stderr.read().decode()[:2000]}")
        try:
            traci.init(port)
            break
        except Exception:
            if time.monotonic() > deadline:
                proc.kill()
                raise RuntimeError("sumo did not accept a traci connection in 900 s")
            time.sleep(1.0)

    return proc, traci, time.monotonic() - started


def seed_traffic(traci, cfg, count: int, radius_m: float = 1200.0) -> dict:
    """Background traffic clustered on the configured origin.

    Deliberately does *not* go through duarouter: Phase 1 measured that at a
    3.85-4.4 GB peak on this network, which is most of the budget this bench
    exists to protect. Trips are seeded by asking SUMO itself to map lon/lat
    to an edge (convertRoad) and letting its internal router fill in the path
    at insertion (changeTarget) -- the same call the fly's own rerouting will
    make in Phase 2, so its cost is worth measuring here too.
    """
    lat0 = cfg.route.origin.lat
    lon0 = cfg.route.origin.lon
    dlat = radius_m / 111_320.0
    dlon = radius_m / (111_320.0 * abs(math.cos(math.radians(lat0))))

    rng = random.Random(20260914)
    edges: list[str] = []
    attempts = 0
    while len(edges) < count * 2 and attempts < count * 20:
        attempts += 1
        lat = lat0 + rng.uniform(-dlat, dlat)
        lon = lon0 + rng.uniform(-dlon, dlon)
        try:
            edge, _pos, _lane = traci.simulation.convertRoad(
                lon, lat, isGeo=True, vClass="passenger")
        except Exception:
            continue
        if edge and not edge.startswith(":"):
            edges.append(edge)

    added = 0
    reroute_ms: list[float] = []
    for i in range(min(count, len(edges) // 2)):
        src, dst = edges[2 * i], edges[2 * i + 1]
        if src == dst:
            continue
        rid, vid = f"bg_r{i}", f"bg_{i}"
        try:
            traci.route.add(rid, [src])
            traci.vehicle.add(vid, rid, departLane="best", departSpeed="max")
            t0 = time.perf_counter()
            traci.vehicle.changeTarget(vid, dst)          # SUMO routes it here
            reroute_ms.append((time.perf_counter() - t0) * 1000)
            added += 1
        except Exception:
            continue

    return {
        "requested": count,
        "edges_found": len(edges),
        "convertRoad_attempts": attempts,
        "added": added,
        "reroute_ms_median": round(statistics.median(reroute_ms), 3) if reroute_ms else None,
        "reroute_ms_max": round(max(reroute_ms), 3) if reroute_ms else None,
    }


def bench_sumo(cfg, vehicles: int = 300, warmup_steps: int = 100,
               measure_steps: int = 200) -> dict:
    proc, traci, load_s = start_sumo(cfg)
    try:
        boot_peak = peak_rss_mb(proc.pid)

        empty = []
        for _ in range(50):
            t0 = time.perf_counter()
            traci.simulationStep()
            empty.append((time.perf_counter() - t0) * 1000)

        seeded = seed_traffic(traci, cfg, vehicles)

        for _ in range(warmup_steps):
            traci.simulationStep()

        loaded = []
        counts = []
        for _ in range(measure_steps):
            t0 = time.perf_counter()
            traci.simulationStep()
            loaded.append((time.perf_counter() - t0) * 1000)
            counts.append(traci.vehicle.getIDCount())

        return {
            "net_load_s": round(load_s, 1),
            "sumo_peak_rss_mb": round(max(boot_peak, peak_rss_mb(proc.pid)), 1),
            "sumo_rss_mb": round(rss_mb(proc.pid), 1),
            "step_ms_empty_median": round(statistics.median(empty), 3),
            "step_ms_loaded_median": round(statistics.median(loaded), 3),
            "step_ms_loaded_p95": round(sorted(loaded)[int(len(loaded) * 0.95)], 3),
            "step_ms_loaded_max": round(max(loaded), 3),
            "vehicles_median": int(statistics.median(counts)),
            "vehicles_max": int(max(counts)),
            "seeded": seeded,
            "python_peak_rss_mb": round(peak_rss_mb(), 1),
            "host_total_mb": round(host_total_mb(), 1),
        }
    finally:
        try:
            traci.close()
        except Exception:
            pass
        proc.wait(timeout=60)


# --------------------------------------------------------------------------
# both halves, at the locked cadence
# --------------------------------------------------------------------------

def _subscribe(traci, ego):
    traci.vehicle.subscribeContext(
        ego, traci.constants.CMD_GET_VEHICLE_VARIABLE, 50.0,
        [traci.constants.VAR_POSITION, traci.constants.VAR_SPEED,
         traci.constants.VAR_ANGLE])


def bench_combined(cfg, vehicles: int = 300, frames: int = 100) -> dict:
    """The real question: 5 SNN ticks + 1 SUMO step + control, inside 100 ms."""
    snn = build_snn(cfg)
    torch = snn["torch"]
    tick = snn["tick"]
    for _ in range(10):
        tick()
    torch.cuda.synchronize()
    snn_only_rss = rss_mb()

    proc, traci, load_s = start_sumo(cfg)
    try:
        seeded = seed_traffic(traci, cfg, vehicles)
        for _ in range(100):
            traci.simulationStep()

        ego = next(iter(traci.vehicle.getIDList()), None)
        if ego is not None:
            _subscribe(traci, ego)          # what the navigator asks for each frame

        frame_ms, snn_ms, sumo_ms, ctl_ms, nearby = [], [], [], [], []
        # Summing the two VmHWM figures overstates the requirement badly: the
        # Python side peaks reading the 852 MB feather, which is finished and
        # freed before sumo is even launched. What has to fit is what is
        # resident *at the same time*, so sample both live each frame.
        concurrent_mb = []
        for _ in range(frames):
            concurrent_mb.append(rss_mb() + rss_mb(proc.pid))
            f0 = time.perf_counter()

            t0 = time.perf_counter()
            for _ in range(TICKS_PER_STEP):
                tick()
            torch.cuda.synchronize()
            snn_ms.append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            traci.simulationStep()
            sumo_ms.append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            live = traci.vehicle.getIDList()
            if ego is not None and ego in live:
                ctx = traci.vehicle.getContextSubscriptionResults(ego) or {}
                nearby.append(len(ctx))
                traci.vehicle.setSpeed(ego, 8.0)
            else:
                ego = next(iter(live), None)
                if ego is not None:
                    _subscribe(traci, ego)
            ctl_ms.append((time.perf_counter() - t0) * 1000)

            frame_ms.append((time.perf_counter() - f0) * 1000)

        med = statistics.median(frame_ms)
        p95 = sorted(frame_ms)[int(len(frame_ms) * 0.95)]
        return {
            "frames": frames,
            "frame_budget_ms": FRAME_BUDGET_MS,
            "frame_ms_median": round(med, 2),
            "frame_ms_p95": round(p95, 2),
            "frame_ms_max": round(max(frame_ms), 2),
            "realtime_factor_median": round(FRAME_BUDGET_MS / med, 2),
            "realtime_factor_p95": round(FRAME_BUDGET_MS / p95, 2),
            "snn_ms_median": round(statistics.median(snn_ms), 2),
            "sumo_ms_median": round(statistics.median(sumo_ms), 2),
            "control_ms_median": round(statistics.median(ctl_ms), 3),
            "nearby_agents_median": int(statistics.median(nearby)) if nearby else 0,
            "vehicles": traci.vehicle.getIDCount(),
            "seeded": seeded,
            "net_load_s": round(load_s, 1),
            "snn_only_rss_mb": round(snn_only_rss, 1),
            "python_peak_rss_mb": round(peak_rss_mb(), 1),
            "sumo_peak_rss_mb": round(peak_rss_mb(proc.pid), 1),
            "concurrent_rss_mb_max": round(max(concurrent_mb), 1),
            "concurrent_rss_mb_median": round(statistics.median(concurrent_mb), 1),
            "host_total_mb": round(host_total_mb(), 1),
            "headroom_mb": round(host_total_mb() - max(concurrent_mb), 1),
            "vram_peak_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1),
            "meets_realtime": bool(p95 <= FRAME_BUDGET_MS),
            "fits_in_ram": bool(max(concurrent_mb) < host_total_mb()),
        }
    finally:
        try:
            traci.close()
        except Exception:
            pass
        proc.wait(timeout=60)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["snn", "sumo", "combined"], required=True)
    ap.add_argument("--vehicles", type=int, default=300)
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    from dipteran.config import load_config
    cfg = load_config()

    if args.stage == "snn":
        result = bench_snn(cfg)
    elif args.stage == "sumo":
        result = bench_sumo(cfg, vehicles=args.vehicles)
    else:
        result = bench_combined(cfg, vehicles=args.vehicles, frames=args.frames)

    result["stage"] = args.stage
    print(json.dumps(result, indent=2), flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))

    if args.stage == "combined" and not result["meets_realtime"]:
        print("FAIL: p95 frame exceeds the 100 ms budget", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
