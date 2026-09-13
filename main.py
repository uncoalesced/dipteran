"""Dipteran -- the server behind the map and the HUD.

Two modes, because they have very different startup costs:

  mock (default)   Random-walk streamer, starts instantly. This is what
                   `check_phase1.py`'s scaffold criterion exercises, and it
                   has a 45 s timeout, so the default cannot be the live
                   backend -- building W takes ~28 s and SUMO needs another
                   75-130 s to parse the 862 MB network.

  live             DIPTERAN_LIVE=1. The real FlyBangaloreNavigator over the
                   FlyWire connectome, driving a free-body vehicle through
                   SUMO traffic.

Build order in live mode is not arbitrary. `scripts/bench_sync.py` measures
the connectome build peaking at 4.3 GB of host RAM and the resident `sumo`
process at 7.5 GB, against a 10.9 GB ceiling. Those two peaks do not fit
together, but they do not have to: the connectome peak is transient. W is
built and the pandas intermediates freed *before* sumo is launched, which
lands at 8.5 GB concurrent with 2.4 GB to spare. Reverse the order and it
OOMs.
"""

import asyncio
import json
import math
import os
import random
import threading
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

app = FastAPI()

REPO = Path(__file__).parent
HTML_CONTENT = (REPO / "index.html").read_text(encoding="utf-8")

LIVE = os.environ.get("DIPTERAN_LIVE", "").strip().lower() in {"1", "true", "yes"}
# One payload per 100 ms control frame; the page interpolates to 60 fps.
STREAM_HZ = 10.0


@app.get("/")
def get_ui():
    return HTMLResponse(HTML_CONTENT)


@app.get("/neuropils.json")
def get_neuropils():
    """The point cloud from scripts/extract_neuropils.py.

    Not committed -- it is derived from FlyWire data, which DATA_LICENSE.md
    keeps out of the repo. The HUD degrades to map-only without it.
    """
    try:
        from dipteran.config import load_config

        path = load_config().paths.cache_dir / "fly_neuropils.json"
    except Exception:
        path = REPO / "cache" / "fly_neuropils.json"

    if not path.is_file():
        return JSONResponse(
            {"error": "not generated",
             "hint": "python scripts/extract_neuropils.py"},
            status_code=404,
        )
    return JSONResponse(json.loads(path.read_text()))


# --------------------------------------------------------------------------
# live simulation
# --------------------------------------------------------------------------

class Simulation:
    """The real loop, on its own thread.

    One SUMO process and one GPU-resident connectome exist per server, so
    there is exactly one simulation and every connected client watches the
    same fly. The thread publishes the newest payload into `latest`; sockets
    read it. No queue, because a client that falls behind should see the
    current state, not a backlog of stale frames.
    """

    def __init__(self):
        self.latest: dict | None = None
        self.error: str | None = None
        self.started = False
        self._stop = threading.Event()

    def start(self) -> None:
        if self.started:
            return
        self.started = True
        threading.Thread(target=self._run, name="dipteran-sim",
                         daemon=True).start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        try:
            self._loop()
        except Exception as exc:                      # surfaced to the client
            import traceback

            self.error = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()

    def _loop(self) -> None:
        from dipteran.config import load_config
        from dipteran.connectome import build_adjacency, resolve_clusters
        from dipteran.navigator import (FlyBangaloreNavigator, monoamine_rows,
                                        telemetry)
        from dipteran.world import World

        cfg = load_config()

        # 1. Connectome first: this is the transient 4.3 GB peak.
        print("[sim] building the connectome", flush=True)
        built = build_adjacency(cfg)
        clusters = resolve_clusters(cfg, neurons=built["neurons"])
        mono = monoamine_rows(cfg, built["neurons"])
        navigator = FlyBangaloreNavigator.from_config(
            cfg, built["W"], built["neurons"], clusters, monoamine=mono)
        del built
        print(f"[sim] navigator ready, gain {cfg.snn.gain}", flush=True)

        # 2. Only now start SUMO, with the peak behind us.
        print("[sim] starting SUMO (this parses an 862 MB network)", flush=True)
        ticks_per_frame = 5                            # 5 x 20 ms = 100 ms
        frame_dt = ticks_per_frame / 50.0

        with World(cfg, step_length=cfg.sumo.step_length) as world:
            print(f"[sim] {world.background} background vehicles", flush=True)
            while not self._stop.is_set():
                frame_started = time.perf_counter()

                sensory = world.sense(frame_dt)
                pam, ppl1 = world.rewards()
                # PAM and PPL1 are dopaminergic, so they are driven as
                # external current here rather than through W -- their own
                # synapses were dropped from the fast path on purpose.
                sensory["reward_pam"] = pam
                sensory["punish_ppl1"] = ppl1
                navigator.set_sensory(sensory)

                motor = None
                for _ in range(ticks_per_frame):
                    motor = navigator.step()

                world.apply(motor, frame_dt)
                world.step_sumo()

                s = world.state
                self.latest = telemetry(
                    motor, lat=s.lat, lon=s.lon, heading=s.heading,
                    speed_ms=s.speed, throttle_cmd=s.throttle_cmd, edge=s.edge,
                    nearby_agents=s.nearby[:24],
                    pothole=sensory["mechano_bristle"] > 0.0,
                    horn=sensory["acoustic_jo"],
                    event=self._event(motor, sensory, s),
                )

                # Hold real time if we are ahead of it; the bench measures
                # ~61 ms of work against a 100 ms frame, so we usually are.
                slack = frame_dt - (time.perf_counter() - frame_started)
                if slack > 0:
                    self._stop.wait(slack)

    @staticmethod
    def _event(motor, sensory, state) -> str | None:
        """One line for the event log, or nothing."""
        if motor.panic > 0.0:
            return "DNp01 GIANT FIBER: evasive escape"
        if sensory["mechano_bristle"] > 0.0:
            return f"POTHOLE: mechanosensory shock on {state.edge}"
        if sensory["looming_lc4"] > 0.5:
            return "LC4/LPLC2 LOOM: near-collision closing"
        if sensory["acoustic_jo"] > 0.6:
            return "JOHNSTON'S ORGAN: horn pressure, compass destabilised"
        if state.stalled_s > 10.0:
            return f"ZERO OPTIC FLOW: stalled {state.stalled_s:.0f}s"
        return None


SIM = Simulation()


# --------------------------------------------------------------------------
# mock scaffold
# --------------------------------------------------------------------------

def mock_frames():
    """The Phase 1 random walk.

    Kept so the page, and check_phase1's scaffold criterion, run with no GPU,
    no SUMO and no FlyWire download.
    """
    lat, lon = 12.9177, 77.6238           # Silk Board
    heading = 45.0                        # toward Tin Factory
    events = [
        "LC4 SPIKE: Auto cut-in detected",
        "POTHOLE HIT: Campaniform sensilla shock",
        "BMTC BUS LOOM: Giant Fiber evasion triggered",
        "ZERO OPTIC FLOW: Stagnant gridlock panic",
        "JOHNSTON'S ORGAN: 110dB Pressure horn blast",
    ]
    regions = ["optic", "looming", "antennal", "central", "auditory",
               "mechano", "steer", "throttle", "giant", "dopamine"]

    while True:
        heading = (heading + (random.random() - 0.5) * 20) % 360
        rad = math.radians(heading)
        lat += math.cos(rad) * 0.00005
        lon += math.sin(rad) * 0.00005

        pain = random.uniform(0.1, 0.9)
        panic = random.uniform(0.0, 1.0)
        throttle = random.uniform(0.0, 1.0)
        state = ("PANIC SWERVE" if panic > 0.85 else
                 "FULL THROTTLE" if throttle > 0.5 else "IDLE DRIFT")

        yield {
            "lat": lat,
            "lon": lon,
            "heading": heading,
            "sugar_reward": round(1.0 - pain, 2),
            "pain_level": round(pain, 2),
            "giant_fiber": round(panic, 2),
            "event": random.choice(events) if random.random() < 0.15 else None,
            "state": state,
            "caption": "Mock scaffold. Set DIPTERAN_LIVE=1 for the real brain.",
            "steer": round(random.uniform(-1, 1), 2),
            "throttle": round(throttle, 2),
            "dopamine": round(random.uniform(-1, 1), 2),
            "speed_ms": round(throttle * 16.0, 2),
            "edge": "",
            "spike_rate": round(random.uniform(0.0, 0.1), 4),
            "regions": {r: round(random.random(), 3) for r in regions},
            "nearby_agents": [],
            "mock": True,
        }


@app.websocket("/ws")
async def simulation_feed(ws: WebSocket):
    await ws.accept()

    if not LIVE:
        stream = mock_frames()
        try:
            while True:
                await ws.send_json(next(stream))
                await asyncio.sleep(0.05)             # 20 Hz
        except WebSocketDisconnect:
            return
        return

    SIM.start()
    try:
        while True:
            if SIM.error:
                await ws.send_json({"error": SIM.error})
                return
            if SIM.latest is not None:
                await ws.send_json(SIM.latest)
            await asyncio.sleep(1.0 / STREAM_HZ)
    except WebSocketDisconnect:
        return


if __name__ == "__main__":
    print(f"[dipteran] mode: {'live' if LIVE else 'mock scaffold'}", flush=True)
    uvicorn.run(
        app,
        host=os.environ.get("DIPTERAN_HOST", "127.0.0.1"),
        port=int(os.environ.get("DIPTERAN_PORT", "8000")),
    )
