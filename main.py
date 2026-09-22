"""Dipteran -- the server behind the map and the HUD.

Two modes, because they have very different startup costs:

  mock (default)   Random-walk streamer, starts instantly. Lets the page,
                   the route picker and the HUD be exercised with no GPU, no
                   SUMO and no FlyWire download. It is the default because
                   the live backend costs ~28 s to build W plus 75-130 s for
                   SUMO to parse the 862 MB network, which is not what you
                   want from `python main.py` with no arguments.

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

Engineered by uncoalesced.
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
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

app = FastAPI()

REPO = Path(__file__).parent
HTML_CONTENT = (REPO / "index.html").read_text(encoding="utf-8")
NOT_FOUND_HTML = (REPO / "404.html").read_text(encoding="utf-8")

# Nothing ever told the loop to stop steering once it arrived. `rewards()`
# only fires PAM on a *new* best distance (see world.py), so once the fly is
# this close the progress signal it needs to hold a heading dries up and it
# free-runs on whatever residual motor drive is left -- which looks like
# circling the goal forever. 20 m: bigger than snn.progress_threshold_m
# (12 m, so it doesn't fight the reward gate) and inside reward_near_m
# (500 m), so it only ever triggers once actually there.
ARRIVAL_RADIUS_M = 20.0

# Joel's report: "it says there are 200 vehicles but I can see only 4 or 3
# dots at once". The dots were capped here at 24 regardless of how many
# `world.sense()` actually found nearby -- and until this session's
# world.py change, it rarely found more than a handful anyway, because the
# context subscription itself only reached 60 m (world.VISUAL_RADIUS_M is
# 350 m now). 200 comfortably covers the whole background fleet at once;
# kept as a cap rather than removed outright so a much bigger `background`
# count later can't blow up the payload size unbounded.
NEARBY_AGENTS_CAP = 200

LIVE = os.environ.get("DIPTERAN_LIVE", "").strip().lower() in {"1", "true", "yes"}
# One payload per 100 ms control frame; the page interpolates to 60 fps.
STREAM_HZ = 10.0


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """A 404 on an unmatched route gets the HUD-styled page instead of
    FastAPI's bare JSON default. Every other status keeps FastAPI's own
    {"detail": ...} shape -- this only replaces what a browser hits by
    typo or by an unregistered path, not the deliberate 404s the two
    asset routes below already return with their own body.
    """
    if exc.status_code == 404:
        return HTMLResponse(NOT_FOUND_HTML, status_code=404)
    return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)


@app.get("/")
def get_ui():
    return HTMLResponse(HTML_CONTENT)


@app.get("/fly-dorsal.png")
def get_fly_sprite():
    """The map marker, rendered from the flybody MuJoCo model.

    Unlike the neuropil cloud this one *is* committed -- it derives from
    TuragaLab/flybody, which is Apache-2.0 rather than CC BY-NC, so the asset
    can live in the repo as long as the attribution does too (DATA_LICENSE.md).
    Regenerate with `python scripts/render_fly_sprite.py`.
    """
    path = REPO / "assets" / "sprites" / "fly-dorsal.png"
    if not path.is_file():
        return JSONResponse(
            {"error": "not generated",
             "hint": "python scripts/render_fly_sprite.py"},
            status_code=404,
        )
    return FileResponse(path, media_type="image/png")


class Endpoint(BaseModel):
    lat: float
    lon: float


class RouteRequest(BaseModel):
    origin: Endpoint
    destination: Endpoint


def _route_config():
    from dipteran.config import load_config

    return load_config().route


@app.get("/route")
def get_route():
    """What the picker needs: the default pair, and where picking is legal.

    `bbox` is the OSM extract's own bounds. Outside it there is no road
    network for SUMO to snap a vehicle to, so the page uses it both as the
    map's `maxBounds` and as the validity test on a click.
    """
    r = _route_config()
    return {
        "name": r.name,
        "origin": {"lat": r.origin.lat, "lon": r.origin.lon,
                   "label": r.origin.label},
        "destination": {"lat": r.destination.lat, "lon": r.destination.lon,
                        "label": r.destination.label},
        "bbox": {"west": r.bbox.west, "south": r.bbox.south,
                 "east": r.bbox.east, "north": r.bbox.north},
        "live": LIVE,
        "started": SIM.started,
    }


@app.post("/route")
def post_route(req: RouteRequest):
    """Accept the pair the user picked and release the simulation.

    In live mode this is what actually boots the thing -- the connectome
    build and SUMO's 75-130 s network parse are both downstream of it, which
    is why the endpoint returns immediately and progress arrives over `/ws`
    as status frames instead. Restarting is deliberately not supported: one
    SUMO process and one GPU-resident `W` exist per server.
    """
    r = _route_config()
    for name, pt in (("origin", req.origin), ("destination", req.destination)):
        if not (r.bbox.south <= pt.lat <= r.bbox.north
                and r.bbox.west <= pt.lon <= r.bbox.east):
            raise HTTPException(
                422, f"{name} ({pt.lat:.5f}, {pt.lon:.5f}) is outside the "
                     f"road network extract")
    if SIM.started:
        raise HTTPException(
            409, "a simulation is already running; restart the server to "
                 "choose a different route")

    SIM.set_route((req.origin.lat, req.origin.lon),
                  (req.destination.lat, req.destination.lon))
    if LIVE:
        SIM.start()
    return {"ok": True, "live": LIVE, "status": SIM.status}


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
        # Bumped whenever the route changes, so the mock streamer -- which
        # runs before anyone has picked anything -- knows to re-seed itself
        # rather than carrying on from wherever its random walk had got to.
        self.route_version = 0
        self.origin: tuple[float, float] | None = None
        self.goal: tuple[float, float] | None = None
        self.status = "waiting for a route"
        self._stop = threading.Event()

    def set_route(self, origin: tuple[float, float],
                  goal: tuple[float, float]) -> None:
        self.origin, self.goal = origin, goal
        self.route_version += 1

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
            self.status = "failed"
            traceback.print_exc()

    def _loop(self) -> None:
        from dipteran.config import load_config
        from dipteran.connectome import build_adjacency, resolve_clusters
        from dipteran.navigator import (FlyBangaloreNavigator, MotorOutput,
                                        monoamine_rows, telemetry)
        from dipteran.world import LOOM_FIELD_DEG, LOOM_RADIUS_M, World

        cfg = load_config()

        # 1. Connectome first: this is the transient 4.3 GB peak.
        self.status = "building the connectome (~28 s)"
        print("[sim] building the connectome", flush=True)
        built = build_adjacency(cfg)
        clusters = resolve_clusters(cfg, neurons=built["neurons"])
        mono = monoamine_rows(cfg, built["neurons"])
        navigator = FlyBangaloreNavigator.from_config(
            cfg, built["W"], built["neurons"], clusters, monoamine=mono)
        del built
        print(f"[sim] navigator ready, gain {cfg.snn.gain}", flush=True)

        # 2. Only now start SUMO, with the peak behind us.
        self.status = "starting SUMO (parsing an 862 MB network, 75-130 s)"
        print("[sim] starting SUMO (this parses an 862 MB network)", flush=True)
        ticks_per_frame = 5                            # 5 x 20 ms = 100 ms
        frame_dt = ticks_per_frame / 50.0

        with World(cfg, step_length=cfg.sumo.step_length,
                   origin=self.origin, goal=self.goal) as world:
            self.status = "running"
            print(f"[sim] {world.background} background vehicles", flush=True)
            while not self._stop.is_set():
                frame_started = time.perf_counter()

                sensory = world.sense(frame_dt)

                if world.state.distance_to_goal_m <= ARRIVAL_RADIUS_M:
                    s = world.state
                    self.latest = telemetry(
                        MotorOutput(), lat=s.lat, lon=s.lon,
                        heading=s.heading, speed_ms=0.0, throttle_cmd=0.0,
                        edge=s.edge, nearby_agents=s.nearby[:NEARBY_AGENTS_CAP],
                        pothole=False, horn=0.0, event="ARRIVED",
                        fov_deg=LOOM_FIELD_DEG * 2.0, fov_radius_m=LOOM_RADIUS_M,
                    )
                    self.status = "arrived"
                    print(f"[sim] arrived, {s.distance_to_goal_m:.1f} m "
                          f"from goal", flush=True)
                    break

                pam, ppl1 = world.rewards()
                # PAM and PPL1 are dopaminergic, so they are driven as
                # external current here rather than through W -- their own
                # synapses were dropped from the fast path on purpose.
                sensory["reward_pam"] = pam
                sensory["punish_ppl1"] = ppl1
                navigator.set_sensory(sensory)

                # Averaged, not sampled: DNa02 is one neuron per side, so
                # a single tick quantises `steer` to multiples of 0.1.
                motor = MotorOutput.average(
                    [navigator.step() for _ in range(ticks_per_frame)])

                world.apply(motor, frame_dt)
                world.step_sumo()

                s = world.state
                self.latest = telemetry(
                    motor, lat=s.lat, lon=s.lon, heading=s.heading,
                    speed_ms=s.speed, throttle_cmd=s.throttle_cmd, edge=s.edge,
                    nearby_agents=s.nearby[:NEARBY_AGENTS_CAP],
                    pothole=sensory["mechano_bristle"] > 0.0,
                    horn=sensory["acoustic_jo"],
                    event=self._event(motor, sensory, s),
                    fov_deg=LOOM_FIELD_DEG * 2.0, fov_radius_m=LOOM_RADIUS_M,
                    heading_error_deg=s.heading_error_deg,
                    wrong_way_ms=s.wrong_way_ms,
                    speed_cap_ms=world.speed_cap_ms,
                    steer_filtered=s.steer_filtered,
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

def mock_frames(origin=None, goal=None):
    """The Phase 1 random walk.

    Kept so the page runs with no GPU, no SUMO and no FlyWire download. It
    takes the picked route so the walk starts where the user pointed and sets
    off toward the destination, rather than always leaving Silk Board --
    otherwise the picker would look broken to anyone trying it without the
    live backend.
    """
    from dipteran.world import bearing_deg   # stdlib-only at import time

    r = _route_config()
    lat, lon = origin or (r.origin.lat, r.origin.lon)
    heading = bearing_deg(lat, lon,
                          *(goal or (r.destination.lat, r.destination.lon)))
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
            "status": "running (mock)",
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
        # Same gate as live mode: nothing streams until a route is posted.
        # The walk re-seeds if the route changes, which it can in mock mode
        # because there is no SUMO process to be already committed to one.
        version, stream = None, None
        try:
            while True:
                if SIM.origin is None:
                    await ws.send_json({"status": SIM.status, "booting": False})
                    await asyncio.sleep(0.2)
                    continue
                if SIM.route_version != version:
                    version = SIM.route_version
                    stream = mock_frames(SIM.origin, SIM.goal)
                await ws.send_json(next(stream))
                await asyncio.sleep(0.05)             # 20 Hz
        except WebSocketDisconnect:
            return
        return

    try:
        while True:
            if SIM.error:
                await ws.send_json({"error": SIM.error})
                return
            if SIM.latest is not None:
                await ws.send_json(SIM.latest)
            else:
                # Booting, or nobody has posted a route yet. Either way the
                # page needs something to show for the next two minutes.
                await ws.send_json({"status": SIM.status,
                                    "booting": SIM.started})
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
