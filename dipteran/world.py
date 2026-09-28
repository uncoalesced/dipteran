"""The environment half: SUMO traffic, sensory encoding, hybrid motion.

One measured decision shapes this whole module. PHASE1_REPORT.md's notes tell
Phase 2 to use `SumoNet.nearest_edge` to answer "which edge is the vehicle
on". That advice predates the memory numbers: `scripts/bench_sync.py` measures
the `sumo` process alone at 7,453 MB resident on the 862 MB network, against a
10,952 MB WSL2 ceiling, leaving about 2.4 GB once torch and W are loaded. A
resident `SumoNet` is a second full copy of that same network inside our own
process and does not fit.

So nothing here parses the network. Every geometric question is asked of the
already-running SUMO process over traci -- `convertRoad` for "which edge is
this point on", `convert2D`/`convertGeo` for frames -- which costs no extra
resident memory at all. `SumoNet` stays what Phase 1 used it for: offline
construction and verification of the ID map, one stage at a time.

The vehicle itself is not a normal SUMO vehicle. PLAN.md's hybrid motion model
wants continuous heading and position, soft-pulled toward the nearest road
rather than snapped onto a lane, so the fly can wander down wrong turns. That
is `moveToXY(..., keepRoute=2)`: SUMO stops route-constraining it but still
simulates it, so background traffic reacts to a fly drifting into their lane.
It also sidesteps the stranding risk in PLAN.md section 11 -- with 239
strongly-connected components a route-constrained vehicle can end up in a dead
component, but a free body only ever needs to be near *some* road.

Engineered by uncoalesced.
"""

from __future__ import annotations

import collections
import math
import os
import subprocess
import sys
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path

SUMO_TOOLS = Path(os.environ.get("SUMO_HOME", "/usr/share/sumo")) / "tools"
if SUMO_TOOLS.is_dir() and str(SUMO_TOOLS) not in sys.path:
    sys.path.insert(0, str(SUMO_TOOLS))

EARTH_R = 6_371_000.0
EGO_ID = "fly"
EGO_ROUTE = "fly_route"

# Perception constants. The vehicle envelope and the motor transfer live in
# config.yaml's `vehicle:` block instead -- they are tuning knobs, and the
# project rule is that tunables are fields, not constants.
#
# The two scales there exist because `MotorOutput.throttle` is a DNg02
# population firing rate, measured at 0.05-0.24, while the motion model wants
# a command in [0, 1]. Descending neurons sit in balanced excitation and
# inhibition (6.07M inhibitory pairs against 9.02M excitatory) and fire
# sparsely, which is what a real brain does; treating the rate as the command
# pins the vehicle at 5% throttle. Raising `snn.sensory_gain` does not help
# (measured: throttle holds at 0.04-0.05 from 1x to 32x, and past 8x the
# giant fiber fires on 45-70% of ticks) because the ceiling is the recurrent
# balance, not the input.
LOOM_FIELD_DEG = 60.0        # forward half-field the looming detectors watch
HORN_RADIUS_M = 15.0
VEHICLE_WIDTH_M = 1.8

# `LOOM_RADIUS_M` is what the SNN's looming/horn tuning was actually measured
# against -- it is the old context-subscription radius, kept exactly as it
# was so bug #3's panic-reflex fix (this session) is not disturbed by the
# change below. `VISUAL_RADIUS_M` is a second, much wider number purely for
# what the HUD draws: Joel's report was "it says 200 vehicles but I only see
# 3 or 4 dots", and the reason was that the context subscription itself only
# ever asked SUMO for vehicles within 60 m, so nothing further out existed to
# draw regardless of how many were simulated. Sensing radius and draw radius
# are deliberately two different numbers now; `_perceive` gates the SNN
# channels on the small one and the telemetry list on the large one.
LOOM_RADIUS_M = 60.0
VISUAL_RADIUS_M = 350.0

# How many frames to wait before asking SUMO again for a route it has
# already refused from this edge. At 10 Hz this is five seconds. A refusal
# means Dijkstra searched the whole reachable component and found nothing,
# which is the expensive direction of that call, so it is not worth
# retrying every frame -- but it has to be retried, because the fly can sit
# on such an edge indefinitely.
ROUTE_RETRY_FRAMES = 50

def balanced_pfl3(half_angle_deg: float) -> float:
    """What both lobes read with the goal dead ahead, for a given lobe width.

    `steering_drive(0)` puts cos(half_angle) on each side, so this is the
    balanced operating point: the drive that asks for no turn at all. The
    calibration blink in `World.apply` injects it on both lobes, and
    `scripts/probe_diagnostics.py` measures `snn.steer_bias` at the same
    point. It moves with the lobe width, which is why it is computed rather
    than written down -- at the configured 70 degrees it is 0.342, and at a
    90-degree width it would be zero, which is what makes 90 unusable.
    """
    return math.cos(math.radians(half_angle_deg))

# Cosmetic-only classification for the map's traffic dots -- every background
# vehicle keeps the same SUMO vClass and routing regardless of its assigned
# type, so this can never change what the simulation actually does, only
# what colour dot the HUD draws for it. The weights are an unmeasured guess
# at a Bangalore-ish mix (heavy two-wheeler and auto-rickshaw share, few
# trucks) -- there is no dataset behind these numbers, they only exist to
# beat "every vehicle on the map is the same grey dot".
BG_VEHICLE_TYPES = ("bike", "auto", "car_blue", "car_red", "truck")
BG_VEHICLE_WEIGHTS = (0.32, 0.28, 0.15, 0.15, 0.10)

# Ceiling on any single injected sensory current. `calibrate_gain` was run at
# drive=0.4 and its docstring explains why the value is deliberately below
# `v_thresh`: at threshold a clamped row fires every sub-tick, which is a
# short circuit rather than a stimulus. Every channel here is individually
# bounded to [0, 1] already, so this only bites on the steering lobes, which
# do reach 1.0 by construction.
SENSORY_CEILING = 0.4


# --------------------------------------------------------------------------
# pure geometry -- no SUMO, no traci, unit-testable
# --------------------------------------------------------------------------

def wrap180(deg: float) -> float:
    """Fold an angle into [-180, 180).

    Half-open at the bottom: exactly 180 comes back as -180. Same direction
    either way, and every caller here consumes the magnitude.
    """
    return (deg + 180.0) % 360.0 - 180.0


def _junction_of(internal_edge: str) -> str:
    """Junction id of an internal edge id, `:<junction>_<index>`."""
    return internal_edge[1:].rsplit("_", 1)[0]


def _project(shape, x: float, y: float):
    """Nearest point on a polyline: (distance, position along it, (x, y)),
    or None for a missing or degenerate shape."""
    if not shape or len(shape) < 2:
        return None
    best, along = None, 0.0
    for (x0, y0), (x1, y1) in zip(shape, shape[1:]):
        dx, dy = x1 - x0, y1 - y0
        seg = math.hypot(dx, dy)
        if seg <= 0.0:
            continue
        t = max(0.0, min(1.0, ((x - x0) * dx + (y - y0) * dy) / (seg * seg)))
        px, py = x0 + dx * t, y0 + dy * t
        d = math.hypot(x - px, y - py)
        if best is None or d < best[0]:
            best = (d, along + seg * t, (px, py))
        along += seg
    return best


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = (math.sin(dp / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2)
    return 2 * EARTH_R * math.asin(min(1.0, math.sqrt(a)))


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial great-circle bearing, degrees clockwise from north."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return math.degrees(math.atan2(y, x)) % 360.0


def looming_rate(prev_dist: float, dist: float, dt: float,
                 width: float = VEHICLE_WIDTH_M) -> float:
    """Angular expansion rate (rad/s) of an object of `width` at `dist`.

    This is what an LC4 cell responds to -- not proximity, but how fast the
    thing is growing on the retina. An auto-rickshaw holding station two
    metres away is not a threat; one closing at 8 m/s is.
    """
    if dt <= 0 or prev_dist <= 0 or dist <= 0:
        return 0.0
    theta_now = 2.0 * math.atan(width / (2.0 * dist))
    theta_prev = 2.0 * math.atan(width / (2.0 * prev_dist))
    return max(0.0, (theta_now - theta_prev) / dt)


def steering_drive(error_deg: float, half_angle_deg: float = 70.0
                   ) -> tuple[float, float]:
    """Signed heading error -> (PFL3-left drive, PFL3-right drive).

    `error` is `bearing_to_goal - heading`, so a positive error means the goal
    lies clockwise of where the fly is pointing -- to its RIGHT -- and
    `World.apply` increases heading for positive `steer`, so positive steer
    turns right. The existing `compass_epg` current, `max(0, cos(error))`, is
    even in `error` and so cannot express any of this: a goal 90 degrees left
    and one 90 degrees right produce the identical scalar. It says how aligned
    the fly is, never which way to fix it, which is why nothing in Phase 2 ever
    corrected a heading.

    Two overlapping cosine lobes offset by +/- `half_angle_deg` restore the
    sign, the way a pair of antennae or a pair of PFL3 populations do.

    THE RETURN IS CROSSED, and that is not a typo. PFL3 projects
    contralaterally onto DNa02 -- measured directly from W, 12 synapses from
    PFL3-right onto DNa02-left and zero onto DNa02-right, mirrored on the
    other side. So a goal to the RIGHT must drive PFL3-LEFT in order to raise
    DNa02-right and turn right. This matches Nature 2024's PFL3R/PFL3L
    asymmetry. `tests/test_units.py` asserts the sign rather than trusting it:
    an earlier draft of this formula shipped with its worked example
    contradicting its own arithmetic, twice.

    At the configured 70-degree width:

        error = +70 (goal to the right) -> left lobe 1.0, right lobe 0.0
        error =   0 (dead ahead)        -> both 0.342, no net turn
        error = 180 (behind)            -> both 0.0, the dead zone

    That dead zone is why `half_angle_deg` is configurable and why
    `snn.steer_half_angle_deg` defaults to 70 rather than 45. At 45 a fly
    pointing more than 135 degrees off its route gets no steering drive at
    all and the readout falls back to a constant: measured live, a median
    heading error of 122 degrees with a positive command on 95% of frames,
    which is a fly facing away from the road turning slow circles.

    The width cannot simply go to 90. The lobes must *overlap* dead ahead,
    because DNa02-left rests silent while DNa02-right runs at ~101 Hz (see
    `navigator._decode`), so a left turn is expressed by pushing DNa02-right
    below its balanced baseline -- and at 90 degrees the off lobe receives
    zero drive, leaving no baseline to push below. Measured at 90: the
    command was positive on 100% of frames, a fly that could physically only
    turn right. 70 keeps a real balanced drive dead ahead (cos 70 = 0.34)
    while pushing the dead zone out past 160 degrees.
    """
    right_of_nose = max(0.0, math.cos(math.radians(error_deg - half_angle_deg)))
    left_of_nose = max(0.0, math.cos(math.radians(error_deg + half_angle_deg)))
    # goal on the right -> drive PFL3-left; goal on the left -> PFL3-right.
    return right_of_nose, left_of_nose


def proximity_scale(distance_m: float, near_m: float = 500.0,
                    far_m: float = 15000.0, far_scale: float = 0.02) -> float:
    """How much a unit of progress is worth, by remaining distance.

    Full strength within `near_m` of the goal, falling to `far_scale` at
    `far_m` and held there beyond it. Inverse-power through both anchors --
    the same shape as `sense()`'s odour law, which already uses a 500 m
    constant.

    This scales the *magnitude* of a progress reward; it never pays out for
    mere proximity. `rewards()` still gates on having actually closed
    distance, so a fly parked near the goal earns nothing, which is the
    contract `rewards()` has always had.
    """
    exponent = math.log(far_scale) / math.log(near_m / far_m)
    return max(far_scale,
               min(1.0, (near_m / max(distance_m, 1.0)) ** exponent))


def is_pothole(edge_id: str, density: float = 0.04) -> bool:
    """Synthetic hazard, deterministic per edge.

    Hashed rather than randomised so a given road is potholed on every run --
    the fly has to be able to learn that this junction hurts, which it cannot
    do if the hazard moves between episodes.
    """
    if not edge_id or density <= 0.0:
        return False
    return (zlib.crc32(edge_id.encode()) % 10_000) < density * 10_000


# --------------------------------------------------------------------------
# vehicle state
# --------------------------------------------------------------------------

@dataclass
class VehicleState:
    lat: float
    lon: float
    heading: float = 0.0         # degrees clockwise from north
    speed: float = 0.0           # m/s
    # Ground actually covered last frame, after `_pull_to_road`, per second.
    # `speed` is what the motor asked for; the road pull can cancel most of
    # it -- a fly pressed against a kerb reads 4 m/s here and moves 0.4. The
    # senses that are about *achieved* motion read this one.
    realised_speed_ms: float = 0.0
    edge: str = ""
    off_road_m: float = 0.0
    distance_to_goal_m: float = 0.0
    # Position along `edge`, metres from its start -- what `_edge_at` already
    # returns and previously threw away. `_update_route` needs it to know how
    # far into the current edge the fly already is, so the route-remaining
    # estimate doesn't jump by a whole edge length the instant it enters one.
    edge_pos: float = 0.0
    # Remaining distance to the goal *along the road network* SUMO knows
    # about, not the crow-flies line `distance_to_goal_m` is. Always in route
    # metres once a route has resolved: while routing is failing it holds the
    # last known route distance rather than switching to crow-flies, which is
    # what it used to do and which produced a 4.8x unit jump mid-episode (see
    # `_route_distance_now`). `World.route_valid` says which you are reading.
    # This is what `rewards()` now gates progress on -- see the comment there
    # for why the crow-flies distance could not do that job on its own.
    route_distance_to_goal_m: float = 0.0
    travelled_m: float = 0.0
    stalled_s: float = 0.0
    # Signed degrees from the fly's heading to the route target, as `sense`
    # last computed it. Recorded rather than recomputed because it is the
    # one number that separates "the compass is wrong" from "the compass is
    # right and the brain is not turning": a run with a large standing
    # error and no matching steer is a motor problem, the same run with a
    # small error is not.
    heading_error_deg: float = 0.0
    # Smoothed rate at which the remaining route distance is *growing*,
    # m/s. Positive means driving away from the goal; `rewards()` punishes
    # it, and the HUD shows it.
    wrong_way_ms: float = 0.0
    # The time-averaged steering readout the vehicle actually acts on, as
    # opposed to the single-frame `MotorOutput.steer` it is filtered from.
    steer_filtered: float = 0.0
    # The *scaled* throttle command, [0, 1] -- not MotorOutput.throttle,
    # which is a firing rate in roughly [0.05, 0.24]. Keeping the command
    # here rather than passing it around is what stops the two being
    # confused: comparing a 0.24-max firing rate against a 0.3 threshold
    # silently disabled gridlock detection entirely.
    throttle_cmd: float = 0.0
    nearby: list = field(default_factory=list)


# --------------------------------------------------------------------------
# the world
# --------------------------------------------------------------------------

class World:
    """SUMO microsimulation plus the fly-vehicle's free-body motion."""

    def __init__(self, cfg, *, port: int = 8813, background: int = 300,
                 pothole_density: float | None = None,
                 step_length: float | None = None, seed: int = 20260914,
                 origin: tuple[float, float] | None = None,
                 goal: tuple[float, float] | None = None):
        import traci

        self.cfg = cfg
        self.traci = traci
        self.v = cfg.vehicle
        self.step_length = step_length or cfg.sumo.step_length
        self.pothole_density = (cfg.vehicle.pothole_density
                                if pothole_density is None else pothole_density)
        self.seed = seed

        # config.yaml holds the default route; the UI's pick-on-map flow
        # passes its own pair instead. Both are validated against route.bbox
        # by the caller -- outside the extract there is simply no road network
        # to snap to, and `_spawn_ego` would raise on an unresolvable origin.
        self.origin = origin or (cfg.route.origin.lat, cfg.route.origin.lon)
        self.goal = goal or (cfg.route.destination.lat,
                             cfg.route.destination.lon)

        self.proc = self._launch(port)
        self.state = VehicleState(lat=self.origin[0], lon=self.origin[1])
        self.state.heading = bearing_deg(*self.origin, *self.goal)
        self.state.distance_to_goal_m = haversine_m(*self.origin, *self.goal)
        self.state.route_distance_to_goal_m = self.state.distance_to_goal_m
        self.best_distance_m = self.state.distance_to_goal_m

        self._prev_dist: dict[str, float] = {}
        self._ego_xy: tuple[float, float] | None = None
        self._bg_category: dict[str, str] = {}

        # Road-network routing state -- see `_update_route` and
        # `_route_target`. `_goal_edge` is resolved once here: the goal never
        # moves during a run, so there is no reason to ask SUMO the same
        # nearest-edge question about it every frame the way `_edge_at` does
        # for the fly's own constantly-changing position.
        # Same junction problem as the spawn: a goal that snaps to an
        # internal edge is not a routable endpoint, so `findRoute` finds
        # nothing and every frame falls back to the bee-line.
        self._goal_edge, _pos, _lane = self._road_edge_near(*self.goal)
        self._route_edge: str | None = None
        self._route_edges: list[str] = []
        self._route_length_m: float | None = None
        # Which edge `_route_length_m` was measured from. `edge_pos` is a
        # position *along one edge*, so subtracting it from a length measured
        # from a different edge is comparing two unrelated rulers -- see
        # `_route_distance_m`.
        self._route_anchor_edge: str | None = None
        self._route_retry: int = 0
        self._lane_shapes: dict[str, list | None] = {}
        # Last live route distance, held across routing failures so the
        # published value never silently changes units -- see
        # `_route_distance_now` for the measurement that made this necessary.
        self._last_route_m: float | None = None
        # Wrong-way detection state -- see the punishment branch in
        # `rewards()`. Seeded from the starting route distance so the first
        # frame reads zero movement rather than the whole route's length.
        self._prev_route_m: float = self.state.route_distance_to_goal_m
        self._away_ema: float = 0.0
        self._wrong_way_pulse: float = 0.0
        self._wrong_way_cooldown: float = 0.0

        # Fast and slow running averages of the steering readout -- see
        # `apply`. The fast one is the command; the slow one is the standing
        # offset subtracted from it.
        self._steer_filtered: float = 0.0
        self._steer_rest: float = 0.0
        # Calibration blink state -- see `apply`.
        self._balanced_pfl3: float = balanced_pfl3(
            self.cfg.snn.steer_half_angle_deg)
        self._calibrating: int = int(self.v.steer_calibration_s
                                     / max(self.step_length, 1e-3))
        self._calibrate_sum: float = 0.0
        self._calibrate_n: int = 0

        self._spawn_background(background)
        self._spawn_ego()

        # Now that the ego has a resolved starting edge, an initial route can
        # be found, and `best_distance_m` -- the ratchet `rewards()` compares
        # against -- should start from the same quantity `rewards()` will
        # compare against later, not from the crow-flies distance that quantity
        # is deliberately no longer always equal to.
        self._update_route()
        route_m = self._route_distance_m()
        if route_m is not None:
            self.state.route_distance_to_goal_m = route_m
            self.best_distance_m = route_m
            self._last_route_m = route_m
        # Seeded here rather than above, because the route distance found
        # above is usually much larger than the crow-flies distance the
        # state carried before it: reading the difference as a frame's worth
        # of movement would punish the fly for existing.
        self._prev_route_m = self.state.route_distance_to_goal_m
        # All three seeds are provisional until the fly is on the edge it
        # actually drives -- see `_settle_route_seed`.
        self.start_route_m = self.state.route_distance_to_goal_m
        self.route_seeded = False
        # The one line that tells "routing works" apart from "routing
        # silently fell back to the bee-line" -- from outside they look
        # identical, because both just steer at something.
        print(f"[route] goal edge {self._goal_edge or 'UNRESOLVED'}, "
              f"start edge {self.state.edge or 'UNRESOLVED'}, "
              f"{len(self._route_edges)} edges, "
              f"{'no route' if route_m is None else format(route_m, '.0f') + ' m'}"
              f" along route (crow-flies "
              f"{self.state.distance_to_goal_m:.0f} m)", flush=True)

    # -- lifecycle -------------------------------------------------------

    def _launch(self, port: int):
        traci = self.traci
        cmd = [
            os.environ.get("SUMO_BINARY", "sumo"),
            "-n", str(self.cfg.paths.sumo_net),
            "--step-length", str(self.step_length),
            "--remote-port", str(port),
            "--seed", str(self.seed),
            "--no-step-log", "true",
            "--no-warnings", "true",
            "--ignore-route-errors", "true",
            "--time-to-teleport", "300",
            "--collision.action", "warn",
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                stderr=subprocess.PIPE)
        # The network is 862 MB of XML and takes 75-130 s to parse, measured.
        deadline = time.monotonic() + 900
        while True:
            if proc.poll() is not None:
                raise RuntimeError(
                    f"sumo exited {proc.returncode}: "
                    f"{proc.stderr.read().decode()[:2000]}")
            try:
                traci.init(port)
                return proc
            except Exception:
                if time.monotonic() > deadline:
                    proc.kill()
                    raise RuntimeError("sumo never accepted a traci connection")
                time.sleep(1.0)

    def close(self) -> None:
        try:
            self.traci.close()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=60)
        except Exception:
            self.proc.kill()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    # -- setup -----------------------------------------------------------

    def _edge_at(self, lat: float, lon: float) -> tuple[str, float, int]:
        """(edge, position along it, lane) nearest a coordinate, via SUMO."""
        try:
            return self.traci.simulation.convertRoad(
                lon, lat, isGeo=True, vClass="passenger")
        except Exception:
            return "", 0.0, 0

    def _spawn_background(self, count: int) -> None:
        """Background traffic around the origin.

        Not duarouter: Phase 1 measured that at a 3.85-4.4 GB peak on this
        network, and the budget here has 2.4 GB. Trips are seeded through
        SUMO's own incremental router at insertion instead.
        """
        import random

        rng = random.Random(self.seed)
        lat0, lon0 = self.origin
        dlat = 1200.0 / 111_320.0
        dlon = dlat / max(0.2, math.cos(math.radians(lat0)))

        edges: list[str] = []
        for _ in range(count * 20):
            if len(edges) >= count * 2:
                break
            edge, _pos, _lane = self._edge_at(lat0 + rng.uniform(-dlat, dlat),
                                              lon0 + rng.uniform(-dlon, dlon))
            if edge and not edge.startswith(":"):
                edges.append(edge)

        self.background = 0
        for i in range(len(edges) // 2):
            src, dst = edges[2 * i], edges[2 * i + 1]
            if src == dst:
                continue
            try:
                self.traci.route.add(f"bg_r{i}", [src])
                self.traci.vehicle.add(f"bg_{i}", f"bg_r{i}",
                                       departLane="best", departSpeed="max")
                self.traci.vehicle.changeTarget(f"bg_{i}", dst)
                # Cosmetic type, not a SUMO vType -- see BG_VEHICLE_WEIGHTS.
                # Drawn from `rng`, the same seeded stream everything else in
                # this method uses, so the mix is deterministic per seed like
                # the rest of the spawn.
                self._bg_category[f"bg_{i}"] = rng.choices(
                    BG_VEHICLE_TYPES, weights=BG_VEHICLE_WEIGHTS)[0]
                self.background += 1
            except Exception:
                continue

    def _road_edge_near(self, lat: float, lon: float
                        ) -> tuple[str, float, int]:
        """Nearest *normal* edge, searching outward if the point lands in a
        junction.

        `_edge_at` answers with whatever edge is nearest, and inside a
        junction that is an internal edge (`:cluster_...`). Internal edges
        are not places a vehicle can be inserted: SUMO does not reject the
        attempt, it aborts the process --

            sumo: ./src/microsim/MSVehicle.cpp:6002:
            void MSVehicle::updateBestLanes(bool, const MSLane*):
            Assertion `false' failed.

        -- and takes the whole simulation with it. Since the origin and the
        goal both come from a click on a map, a click that lands on any of
        this network's junctions would otherwise kill the run, which is why
        this searches instead of trusting the first answer.
        """
        edge, pos, lane = self._edge_at(lat, lon)
        if edge and not edge.startswith(":"):
            return edge, pos, lane
        lat_per_m = 1.0 / 111_320.0
        lon_per_m = lat_per_m / max(0.2, math.cos(math.radians(lat)))
        for radius in (10.0, 20.0, 35.0, 60.0):
            for bearing in range(0, 360, 30):
                rad = math.radians(bearing)
                edge, pos, lane = self._edge_at(
                    lat + radius * math.cos(rad) * lat_per_m,
                    lon + radius * math.sin(rad) * lon_per_m)
                if edge and not edge.startswith(":"):
                    return edge, pos, lane
        return "", 0.0, 0

    def _spawn_ego(self) -> None:
        traci = self.traci
        edge, pos, _lane = self._road_edge_near(*self.origin)
        if not edge:
            raise RuntimeError("origin does not resolve to any SUMO edge")
        self.state.edge = edge
        self.state.edge_pos = pos
        traci.route.add(EGO_ROUTE, [edge])
        traci.vehicle.add(EGO_ID, EGO_ROUTE, departLane="best")
        traci.vehicle.setSpeedMode(EGO_ID, 0)      # the SNN drives, not Krauss
        traci.simulationStep()
        try:
            self._ego_xy = traci.vehicle.getPosition(EGO_ID)
        except Exception:
            self._ego_xy = None
        # VISUAL_RADIUS_M, not LOOM_RADIUS_M: this one subscription result
        # feeds both the SNN's sensing (`_perceive` re-filters down to
        # LOOM_RADIUS_M for that) and the HUD's traffic dots, and the dots
        # are the one that needed to see further than the old 60 m.
        traci.vehicle.subscribeContext(
            EGO_ID, traci.constants.CMD_GET_VEHICLE_VARIABLE, VISUAL_RADIUS_M,
            [traci.constants.VAR_POSITION, traci.constants.VAR_SPEED,
             traci.constants.VAR_ANGLE])

    # -- sensing ---------------------------------------------------------

    def sense(self, dt: float) -> dict:
        """Sensory currents keyed by cluster name, per PLAN.md section 3."""
        s = self.state
        s.distance_to_goal_m = haversine_m(s.lat, s.lon, *self.goal)

        # Bug report (Joel, this session): close to the goal, the fly would
        # just try to cross straight through whatever wall or building sat
        # between it and the destination. The cause was that `error` below
        # used to be measured against the goal's raw lat/lon -- a compass
        # bearing has no notion that roads exist, so it pointed the same way
        # whether or not anything was drivable in that direction. The fly was
        # never "confused"; the sensory current genuinely carried no
        # information about the road network at all, so there was nothing for
        # the SNN's existing trial-and-error learning to act on except "keep
        # pushing this way".
        #
        # `_update_route`/`_route_target` fix that by asking SUMO -- which
        # already has the whole road graph loaded for its own routing -- for
        # the shortest path from the fly's current edge to the goal edge, and
        # steering the compass at the far end of the next edge on that path
        # instead of at the goal itself. That bearing can only ever point down
        # a real road. The fly still has to *learn* to follow it through the
        # same reward-driven trial and error as everything else here -- this
        # only fixes what the compass is allowed to say, not who acts on it.
        self._update_route()
        s.route_distance_to_goal_m = self._route_distance_now()
        self._settle_route_seed()
        target_lat, target_lon = self._route_target()

        # Odour beacon: PLAN.md's inverse gradient, normalised so it saturates
        # within smelling distance of the goal rather than blowing up to
        # infinity on arrival. Left on the true crow-flies distance -- a real
        # plume is carried by air, not roads, so it is fine (correct, even)
        # for this channel to still smell the goal directly through a wall.
        # It is only ever used as a magnitude here, never a direction.
        odour = min(1.0, 500.0 / max(s.distance_to_goal_m, 1.0))

        # Sun compass: EPG drive peaks when the fly faces the *route target*
        # now, not the goal directly -- see the routing comment above. Still
        # even in `error` and so still carries no turn direction on its own;
        # the signed half is the PFL3 pair below, this scalar stays the "how
        # aligned am I with where I should be heading" channel.
        error = wrap180(bearing_deg(s.lat, s.lon, target_lat, target_lon)
                        - s.heading)
        s.heading_error_deg = error
        compass = max(0.0, math.cos(math.radians(error)))
        if self._calibrating > 0:
            # Calibration blink -- see `apply`. Both lobes get the balanced
            # drive `steering_drive(0)` would produce, so the readout during
            # these frames is the resting asymmetry and nothing else.
            pfl3_left = pfl3_right = self._balanced_pfl3
        else:
            pfl3_left, pfl3_right = steering_drive(
                error, self.cfg.snn.steer_half_angle_deg)

        # Optic flow: what the fly is *actually* achieving, not what it asked
        # for. Zero flow with the throttle open is the gridlock signal.
        # Realised motion, not `s.speed`: the two only differ when the road
        # pull is cancelling the motor, and that is exactly the case this
        # channel exists for. Measured on the Silk Board trace, 2026-09-23:
        # pinned against a roadside for 5,600+ frames at `s.speed` ~4 m/s
        # and 0.4-0.5 m/s realised, T5 read no jam and the stall timer
        # below never started -- 105 of a 120-minute run spent like that.
        flow = min(1.0, s.realised_speed_ms / self.v.max_speed_ms)

        nearby, loom, horn = self._perceive(dt)
        s.nearby = nearby

        pothole = is_pothole(s.edge, self.pothole_density) and s.speed > 1.0

        net = self._net_speed_ms()
        if s.throttle_cmd > 0.3 and (s.realised_speed_ms < 0.5
                                     or (net is not None and net < 0.5)):
            s.stalled_s += dt
        else:
            s.stalled_s = 0.0

        return {
            "odor_dm1": odour,
            "odor_dm4": odour * 0.6,
            "compass_epg": compass,
            # The signed steering command, scaled into the calibrated input
            # range. SENSORY_CEILING exists because `calibrate_gain`'s
            # docstring is explicit that a sensory current at `v_thresh` makes
            # those rows fire every sub-tick -- "not a stimulus, it is a short
            # circuit" -- and the sweep that produced `snn.gain` was run at a
            # drive of 0.4. A lobe peaks at exactly 1.0, so the raw value
            # cannot be injected as-is.
            #
            # Scaled, not clipped, and the difference is the whole signal.
            # `min(0.4, lobe)` looks equivalent and destroys the channel --
            # shown at a 45-degree lobe width, the geometry where it bites:
            # dead ahead the lobes are 0.707/0.707 and both clip to 0.4/0.4,
            # while a goal hard right gives 1.0/0.0 which clips to 0.4/0.0 --
            # the left lobe reads 0.4 in both cases, so half the steering
            # range collapses to a constant. Multiplying keeps the ratio
            # intact: 0.283/0.283 dead ahead, 0.4/0.0 hard right. At the
            # configured 70 degrees the dead-ahead lobes are 0.342, scaling
            # to 0.137/0.137. `test_steering_scaling_preserves_contrast`
            # asserts this at the 45-degree width for the same reason.
            "steer_drive_pfl3_l": pfl3_left * SENSORY_CEILING,
            "steer_drive_pfl3_r": pfl3_right * SENSORY_CEILING,
            "optic_flow_t4": flow,
            # A jam reads as *absent* flow, so this drive is the shortfall:
            # still air on a fly that is trying to move is what panics it.
            "optic_flow_t5": max(0.0, s.throttle_cmd - flow),
            "looming_lc4": loom,
            "looming_lplc2": loom * 0.8,
            "acoustic_jo": horn,
            "mechano_bristle": 1.0 if pothole else 0.0,
        }

    def _update_route(self) -> None:
        """Recompute the road-network path to the goal, when the fly's
        current edge has changed since the last time this ran.

        SUMO already holds the whole graph in the running process for its
        own routing, so asking it for a path is one more cheap traci call --
        `findRoute` runs Dijkstra over the graph SUMO has anyway, and does
        not touch the 862 MB network XML this module's whole design goes out
        of its way to avoid parsing a second time (see the module docstring).
        """
        edge = self.state.edge
        if not edge or edge.startswith(":"):
            # No resolved edge yet, or mid-junction on an internal edge --
            # those aren't routable endpoints.
            return
        if edge == self._route_edge:
            # Nothing has changed since the last attempt, so normally there
            # is no question to ask again. The exception is an attempt that
            # *failed*: the fly is then parked on an edge SUMO can't route
            # from, and since it is barely moving (see below) the edge won't
            # change on its own either. Asking again on a timer is the only
            # way back out.
            if self._route_length_m is not None:
                return
            self._route_retry -= 1
            if self._route_retry > 0:
                return
        self._route_retry = ROUTE_RETRY_FRAMES
        self._route_edge = edge
        if not self._goal_edge:
            return
        try:
            stage = self.traci.simulation.findRoute(edge, self._goal_edge)
            if stage.edges:
                self._route_edges = list(stage.edges)
                self._route_length_m = stage.length
                self._route_anchor_edge = edge
                self._no_route_frames = 0
            else:
                self._route_failed(edge, "findRoute returned no path")
        except Exception as exc:
            # A disconnected component, or SUMO briefly not ready.
            self._route_failed(edge, f"{type(exc).__name__}: {exc}")

    def _route_failed(self, edge: str, why: str) -> None:
        """No path from here -- keep the last road target, drop the length.

        The obvious fallback, and the one this had first, is to steer at the
        goal again. That turned out to be the bug: a bee-line target sits on
        the far side of whatever building is in the way, the heading locks
        onto it, and `_pull_to_road` then cancels most of each step, so the
        fly hovers in place staring at the goal. Joel watched it do exactly
        that -- 0.8 m/s of realised motion against 4.3 m/s of commanded
        speed, on an edge whose `findRoute` had just returned nothing.

        Keeping the stale `_route_edges` instead aims it at the last road it
        could route from, which is behind it and reachable, so it drives
        back onto the connected network and the next `findRoute` succeeds.

        `_route_length_m` is dropped either way: a length measured from an
        edge the fly has left is not a distance `rewards()` can gate
        progress on, so that one falls back to crow-flies, as before.
        """
        self._route_length_m = None
        self._warn_no_route(edge, why)

    def _warn_no_route(self, edge: str, why: str) -> None:
        """Print the first frame of each routing outage, then stay quiet.

        Falling back is by design and can happen on a lot of consecutive
        frames; a print per frame at 10 Hz would bury the log. One line per
        outage answers the question worth asking -- did the road route fail,
        and where.

        One line *per outage*, not per run: `_update_route` zeroes the
        counter on a successful route, so this re-arms and the next failure
        prints again. Reading these as "one line means one failure for the
        whole run" undercounts badly -- the 2026-09-22 twisty run printed
        six before its first episode ended, from two different edges.
        """
        self._no_route_frames = getattr(self, "_no_route_frames", 0) + 1
        if self._no_route_frames == 1:
            print(f"[route] no road route from {edge} to "
                  f"{self._goal_edge}: {why} -- steering bee-line until it "
                  f"resolves", flush=True)

    def _route_distance_m(self) -> float | None:
        """Remaining distance to the goal *along the road network*.

        None means "no live route right now". It does NOT mean "use the
        crow-flies distance instead" -- see `_route_distance_now`, which is
        what callers should read. Substituting one for the other is what
        produced this project's most misleading number to date.

        `findRoute`'s length runs from the *start* of the edge it was asked
        from -- SUMO sums every edge's full length, the first one included
        -- so what is left is that length minus how far along the edge the
        fly now stands. `_route_point_ahead` has always measured the route
        this way. This used to subtract only the distance moved since the
        route was found, which overstated every route found partway along an
        edge by the position it was found at: every spawn and every
        `reset_episode`, since an origin is wherever the click landed. On
        the 500 m route that was ~118 m, dropped in one frame at the first
        edge change -- a new best, so progress dopamine for ground never
        driven, and a `closed_route_m` 118 m over the truth.

        The length is only meaningful against the edge it was measured
        from. `edge_pos` is a position along *one* edge, so subtracting it
        from a length measured from another subtracts two unrelated rulers.
        That is reachable every time the fly crosses a junction:
        `_pull_to_road` sets `state.edge` from raw `_edge_at`, which answers
        with an internal `:cluster_...` edge inside a junction (the same
        behaviour `_road_edge_near` exists to work around), while
        `_update_route` early-returns on those and leaves the length
        measured from the edge before the junction.

        Measured on the synthetic repro: route 2,000 m, fly 400 m along edge
        A (1,600 m remaining), then one frame on a junction edge at
        `edge_pos` 2.0 -- published distance jumped to 1,998 m. A 398 m
        backward step in one 0.1 s frame reads as 3,980 m/s of driving away
        from the goal, and `_punishments` fires PPL1 on anything over
        `snn.wrong_way_threshold_ms` (0.5), saturating a detector whose full
        scale is 4.0 m/s. `rewards()` is unaffected -- its gate is a *new
        best*, and a backward jump never sets one.

        Refusing to answer is the right failure mode here for the same
        reason it is in `_route_failed`: `_route_distance_now` holds the
        last live value, and `route_valid` reports that it did.
        """
        if (self._route_length_m is None
                or self.state.edge != self._route_anchor_edge):
            return None
        return max(0.0, self._route_length_m - self.state.edge_pos)

    def _route_distance_now(self) -> float:
        """The route distance to publish this frame, in route metres.

        On a routing failure this holds the last known *route* distance
        rather than swapping in the crow-flies one, and that distinction is
        the whole point of this method existing.

        What it fixes, measured: on the 2026-09-19 twisty run the route was
        2,398 m along roads to cover 502 m direct -- a 4.8x ratio, because
        the route is a one-way loop system. Mid-episode `findRoute` failed
        once, `_route_length_m` went None, and this value silently became
        the crow-flies 536 m. Everything downstream read that as 1,862 m of
        progress in a single frame:

          - `run_route.py` reported "episode 1 ... closed 1862 m of route"
            for an episode that travelled 790 m in total. Closing more route
            than you drove is not possible, and that impossibility is what
            exposed this.
          - `rewards()` gates progress dopamine on this number, so the drop
            fired a full-strength reward and ratcheted `best_distance_m`
            down to 536, where nothing could earn reward again for the rest
            of the episode.
          - The wrong-way detector saw the reverse jump on recovery as
            driving away from the goal at enormous speed.

        Going stale is the correct failure mode: a frozen route distance
        makes `rewards()` decline to fire rather than fire wrongly, and the
        crow-flies distance is still published separately as
        `distance_to_goal_m` for anything that genuinely wants it.

        Before any route has ever resolved there is nothing to hold, so the
        crow-flies distance seeds it -- that is a starting value, not a
        substitution, and `route_valid` reports which of the two you got.
        """
        route_m = self._route_distance_m()
        if route_m is not None:
            self._last_route_m = route_m
        return (self._last_route_m if self._last_route_m is not None
                else self.state.distance_to_goal_m)

    def _settle_route_seed(self) -> None:
        """Re-seed the episode's route distance from the edge the fly drives.

        The spawn's edge is a lookup, not an observation. Both test routes'
        origins sit inside junctions, where `_road_edge_near` searches
        outward -- clockwise from north, first normal edge wins -- and routes
        run one way, so the edge it lands on can be much further from the
        goal than the one the fly takes. Traced on the 500 m route: the
        lookup gives `38042287#0`, 658.4 m to the goal around a loop of four
        more edges; the fly, pointed at the goal, is on `1150874214` after
        0.7 m of travel, 535.6 m out. That 122.8 m drop on 0.7 m driven was a
        new best, so `rewards()` paid it as progress at the start of every
        episode, the wrong-way average went deeply negative, and run_route's
        closure ran ~120 m ahead of the distance driven until its assertion
        caught it: "episode 1 closed 383.3 m of route on 268.6 m travelled".

        So the seed is provisional until the first live route measured after
        the fly has moved -- `travelled_m` only grows after `_pull_to_road`
        has assigned an edge from where the fly really is -- and that one
        measurement re-seeds the episode start, the reward ratchet and the
        wrong-way baseline together. Anything driven inside the junction
        before then is not counted as closure: a few metres, and
        under-claiming is the safe direction.
        """
        s = self.state
        if self.route_seeded or s.travelled_m <= 0.0 or not self.route_valid:
            return
        self.route_seeded = True
        self.start_route_m = s.route_distance_to_goal_m
        self.best_distance_m = s.route_distance_to_goal_m
        self._prev_route_m = s.route_distance_to_goal_m

    @property
    def route_valid(self) -> bool:
        """Whether `route_distance_to_goal_m` is a live route measurement.

        False while routing is failing (the value is the last known one,
        held), before the first route ever resolved (it is crow-flies), and
        on the junction frames where the anchor belongs to a different edge
        than the fly is standing on. Recorded per episode so a run whose
        numbers were taken during an outage is identifiable afterwards
        instead of silently averaged in.

        Asks `_route_distance_m` rather than testing `_route_length_m`
        directly, so the two cannot drift apart: there is exactly one
        definition of "is this a live measurement", and it is the one the
        published value is computed from. The call is arithmetic on fields
        already in hand, not a traci round trip.
        """
        return self._route_distance_m() is not None

    def _route_target(self) -> tuple[float, float]:
        """Where to steer: a point further down the road route to the goal,
        not the goal's raw coordinates. See the routing comment in `sense`.

        Aims at the far end of the next edge on the route rather than
        anywhere on the current one, so the target is always a real point
        further along the path and not the spot the fly is already standing
        on (which would give a near-zero, noisy bearing). `lane.getShape`
        returns that edge's own polyline in SUMO's x/y frame -- a per-lane
        runtime query, not network topology this module would otherwise have
        to parse -- and its last point is the edge's far end.
        """
        if not self._route_edges:
            return self.goal
        try:
            point = self._route_point_ahead(self.v.route_lookahead_m)
        except Exception:
            point = None
        if point is None:
            return self.goal
        try:
            lon, lat = self.traci.simulation.convertGeo(*point)
            return lat, lon
        except Exception:
            return self.goal

    def _route_point_ahead(self, lookahead_m: float
                           ) -> tuple[float, float] | None:
        """The point `lookahead_m` along the route from where the fly is.

        Walks the route's lane polylines, skipping what is already behind
        the fly on its current edge, and interpolates inside whichever
        segment the look-ahead lands in. Returns SUMO x/y, or None if the
        route runs out first -- the caller then steers at the goal, which is
        correct on the last stretch.

        The earlier version aimed at the far end of the next edge instead,
        which is fine on long arterials and bad everywhere else: on a
        49-edge route through a one-way grid the target jumped to a new edge
        every few seconds and the bearing jumped with it, so the steering
        spent its time chasing the discontinuities. A fixed distance ahead
        moves smoothly and continuously, which is the whole idea behind
        pure pursuit.
        """
        remaining = max(1.0, lookahead_m)
        for index, edge in enumerate(self._route_edges):
            if index == 0 and self.state.edge.startswith(":"):
                # Inside a junction the fly has already left the route's
                # first edge -- `_update_route` does not re-route from
                # internal edges, so the route still starts with the edge
                # behind it. Measured from that edge's start, the lookahead
                # pointed back the way the fly came: 31-56% of junction
                # frames had the target over 120 degrees off in three live
                # arrival recordings, against 0-15% on normal edges.
                continue
            shape = self._lane_shape(edge)
            if shape is None or len(shape) < 2:
                continue
            # Only the current edge has anything behind the fly to skip.
            skip = (self.state.edge_pos
                    if index == 0 and edge == self.state.edge else 0.0)
            for (x0, y0), (x1, y1) in zip(shape, shape[1:]):
                seg = math.hypot(x1 - x0, y1 - y0)
                if seg <= 0.0:
                    continue
                if skip >= seg:
                    skip -= seg
                    continue
                usable = seg - skip
                if remaining <= usable:
                    frac = (skip + remaining) / seg
                    return x0 + (x1 - x0) * frac, y0 + (y1 - y0) * frac
                remaining -= usable
                skip = 0.0
        return None

    def _lane_shape(self, edge: str):
        """Lane polyline for an edge, cached for the life of the run.

        Route geometry does not change, and `_route_target` now walks
        several edges per frame rather than reading one -- without the cache
        that is a fresh traci round trip per edge per frame.
        """
        if edge not in self._lane_shapes:
            try:
                self._lane_shapes[edge] = self.traci.lane.getShape(
                    f"{edge}_0")
            except Exception:
                self._lane_shapes[edge] = None
        return self._lane_shapes[edge]

    def _perceive(self, dt: float) -> tuple[list, float, float]:
        """Nearby agents, peak looming rate, horn pressure.

        Worked entirely in SUMO's own network frame, which is already metres.
        The obvious implementation converts each neighbour to lon/lat with
        `traci.simulation.convertGeo` and measures with haversine, and that is
        what this did first -- at a cost of one synchronous traci round trip
        *per vehicle per frame*. Measured: the full loop ran at 2.65 frames/s
        (0.27x real time) against the 16 frames/s the bench predicted, because
        the bench only ever performed a single context read and so never saw
        the per-neighbour traffic. Cartesian distance on the coordinates the
        subscription already returned is exact, and free.
        """
        traci = self.traci
        s = self.state
        results = traci.vehicle.getContextSubscriptionResults(EGO_ID) or {}
        ego = self._ego_xy
        if ego is None or not results:
            self._prev_dist = {}
            return [], 0.0, 0.0

        # The HUD draws neighbours on a map, so telemetry still needs lon/lat.
        # Derived locally from the ego's own fix plus the metre offset rather
        # than asked of SUMO per vehicle: over a VISUAL_RADIUS_M context
        # radius the flat-earth error is still under a metre, and it costs no
        # round trip.
        lat_per_m = 1.0 / 111_320.0
        lon_per_m = lat_per_m / max(0.2, math.cos(math.radians(s.lat)))

        nearby, loom, close = [], 0.0, 0
        seen: dict[str, float] = {}
        for vid, data in results.items():
            if vid == EGO_ID:
                continue
            x, y = data.get(traci.constants.VAR_POSITION, (0.0, 0.0))
            dx, dy = x - ego[0], y - ego[1]
            dist = math.hypot(dx, dy)
            if dist <= 0.0:
                continue
            seen[vid] = dist
            if dist <= HORN_RADIUS_M:
                close += 1

            # SUMO's y axis points north, x east, so a compass bearing is
            # atan2(east, north) -- the arguments are the other way round
            # from the usual atan2(y, x).
            relative = wrap180(math.degrees(math.atan2(dx, dy)) - s.heading)
            # Gated on LOOM_RADIUS_M explicitly now that the subscription
            # itself reaches out to VISUAL_RADIUS_M for the HUD's sake --
            # this keeps the SNN seeing exactly the same 60 m/forward-cone
            # window bug #3's panic-reflex fix was tuned against, unaffected
            # by how far the map now draws dots.
            if dist <= LOOM_RADIUS_M and abs(relative) <= LOOM_FIELD_DEG:
                loom = max(loom, looming_rate(self._prev_dist.get(vid, dist),
                                              dist, dt))
            nearby.append({
                "id": vid,
                # Cosmetic vehicle class for the HUD's colour coding -- see
                # BG_VEHICLE_TYPES. Defaults to "car_blue" for anything not
                # in the map (there shouldn't be any; background vehicles are
                # all categorised at spawn).
                "type": self._bg_category.get(vid, "car_blue"),
                "lat": round(s.lat + dy * lat_per_m, 6),
                "lon": round(s.lon + dx * lon_per_m, 6),
                "speed": round(float(data.get(traci.constants.VAR_SPEED, 0.0)), 2),
                "bearing": round(relative, 1),
                "dist": round(dist, 1),
            })

        self._prev_dist = seen
        # Angular expansion saturates the detector at roughly 1 rad/s.
        return nearby, min(1.0, loom), min(1.0, close / 5.0)

    # -- acting ----------------------------------------------------------

    def apply(self, motor, dt: float) -> None:
        """Hybrid motion: free heading and position, soft-pulled to the road."""
        traci = self.traci
        s = self.state

        if self._calibrating > 0:
            # Calibration blink: a couple of seconds, parked, both PFL3
            # lobes balanced (see `sense`), averaging what the readout says
            # when no turn is being asked for. That average *is* the resting
            # offset, and it has to be measured in the live loop rather than
            # read from config, because `snn.steer_bias` is calibrated with
            # only the navigation channels driven while the loop also
            # injects odour, optic flow, looming, horn and PPL1.
            #
            # Without it the run orbits: measured on the attempt before this
            # one, the command came out positive on 98.4% of frames and the
            # fly turned circles, 81 m of path in 150 s.
            #
            # Runs before the filters below so neither the command average
            # nor the offset estimate is seeded with balanced-drive frames.
            self._calibrate_sum += motor.steer
            self._calibrate_n += 1
            self._calibrating -= 1
            if self._calibrating == 0 and self._calibrate_n:
                self._steer_rest = self._calibrate_sum / self._calibrate_n
                print(f"[steer] resting offset {self._steer_rest:+.4f} "
                      f"measured over {self._calibrate_n} frames", flush=True)
            s.steer_filtered = 0.0
            s.throttle_cmd = 0.0
            return

        # The steering readout is filtered over time before it is used, and
        # that is not a smoothing nicety -- it is the difference between the
        # fly homing and the fly wandering.
        #
        # DNa02 is one neuron per side. `scripts/probe_diagnostics.py` Test G
        # shows the decode is cleanly directional (-15 deg/s at a 45-degree
        # error to the left, +21 to the right), but it measures that over 200
        # ticks. The live loop averages 5 ticks per frame, and five ticks of
        # a sparsely firing single cell is mostly noise: measured over a real
        # run, the sign of `steer` matched the sign of the heading error on
        # 48.5% of 7,886 frames -- a coin flip -- and correlation stayed at
        # 0.05 across every lag. Averaging across frames instead of within
        # one recovers the same resolution the probe gets from tick count,
        # at the cost of `steer_smoothing_s` of lag.
        tau = self.v.steer_smoothing_s
        if tau > 0.0:
            self._steer_filtered += ((motor.steer - self._steer_filtered)
                                     * min(1.0, dt / tau))
            raw_steer = self._steer_filtered
        else:
            raw_steer = motor.steer

        # Then the standing offset comes off. `snn.steer_bias` is measured
        # by probe_diagnostics under navigation sensory alone; the live loop
        # also injects odour, optic flow, looming, horn and -- now loudly --
        # PPL1, and those shift DNa02's resting asymmetry away from the
        # value that was calibrated. Measured on the run before this
        # existed: the route target sat to the left on 95% of frames while
        # the filtered steer was positive on 57% of them, correlation with
        # the error a healthy 0.57 riding on an offset that outvoted it.
        #
        # A much slower average of the same readout *is* that offset, so
        # subtracting it leaves the modulation and drops the bias --
        # recalibration in the loop rather than a number in a file that was
        # right under different conditions. The time constant has to sit far
        # above `steer_smoothing_s` so a real sustained turn survives it.
        # Estimated only while the route target is dead ahead, which is the
        # one condition where the right answer is known: pointing the way
        # the road goes, the correct command is zero, so whatever the
        # readout says is offset by definition.
        #
        # Updating it on every frame instead -- the first version of this --
        # cancels real steering along with the bias. Measured: a window
        # where the road went left on 100% of frames, and a plain slow
        # average quietly subtracted the sustained left the fly needed,
        # leaving a zero-mean command against a one-sided road.
        bias_tau = self.v.steer_bias_tau_s
        gain_left = gain_right = 1.0
        if bias_tau > 0.0:
            fit = None
            if motor.panic <= 0.0:
                fit = self._steer_fit(raw_steer, s.heading_error_deg,
                                      s.speed, dt)
            if fit is not None:
                self._steer_rest, gain_left, gain_right = fit
            elif abs(s.heading_error_deg) <= self.v.steer_bias_window_deg:
                self._steer_rest += ((motor.steer - self._steer_rest)
                                     * min(1.0, dt / bias_tau))
            raw_steer -= self._steer_rest
        s.steer_filtered = raw_steer
        # Per-side gain. The right side started with a fixed gain taken from
        # the static probe (Test F: +14.2 Hz of authority to the right against
        # -26.1 Hz to the left). Measured live near Tin Factory, 2026-09-28,
        # it was the other way round: +0.030 at +50 deg against -0.015 at
        # -50 deg, both from the true zero-error readout. Each side's gain is
        # therefore taken from the live fit (`_steer_fit`), and
        # `steer_right_gain` stays only as a manual multiplier, 1.0 by default.
        if raw_steer > 0.0:
            raw_steer *= gain_right * self.v.steer_right_gain
        else:
            raw_steer *= gain_left
        steer = max(-1.0, min(1.0, raw_steer * self.v.steer_scale))
        throttle = max(0.0, min(1.0, motor.throttle * self.v.throttle_scale))

        s.throttle_cmd = throttle
        s.heading = (s.heading + steer * self.v.max_turn_deg_s * dt) % 360.0

        panic_onset = motor.panic > 0.0 and not getattr(
            self, "_panic_prev", False)
        self._panic_prev = motor.panic > 0.0
        if motor.panic > 0.0:
            # Giant fiber: a startle burst -- hard acceleration plus a wide
            # swerve. This used to brake toward 0 m/s in the same frame it
            # swerved, which meant it rotated in place without covering any
            # ground: _pull_to_road just eased it straight back onto the same
            # hazard edge every time, an endless panic loop with zero net
            # progress. A real escape needs to *displace*, not just turn.
            #
            # Three times the normal acceleration, against the same envelope
            # the cruise branch uses: an escape from something looming is
            # exactly the moment to out-accelerate surrounding traffic, the
            # way a startled two-wheeler does.
            s.speed = min(self.v.max_speed_ms,
                         s.speed + self.v.max_accel_ms2 * 3.0 * dt)
            # The swerve is one jink per burst, on its first frame. It used
            # to be 45 degrees on *every* frame DNp01 fired, so a four-frame
            # burst was a 180-degree flip: that is how the Silk Board trace
            # (2026-09-23) put the fly onto the reverse carriageway, where it
            # then pinned for the rest of the run.
            if panic_onset:
                s.heading = (s.heading
                             + math.copysign(45.0, steer or 1.0)) % 360.0
        else:
            target = throttle * self.v.max_speed_ms
            delta = target - s.speed
            rate = self.v.max_accel_ms2 if delta > 0 else self.v.brake_ms2
            s.speed = max(0.0, s.speed + math.copysign(
                min(abs(delta), rate * dt), delta))

        step = s.speed * dt
        rad = math.radians(s.heading)
        lat = s.lat + (step * math.cos(rad)) / 111_320.0
        lon = s.lon + (step * math.sin(rad)) / (
            111_320.0 * max(0.2, math.cos(math.radians(s.lat))))

        lat, lon = self._pull_to_road(lat, lon, dt)
        moved = haversine_m(s.lat, s.lon, lat, lon)
        s.travelled_m += moved
        s.realised_speed_ms = moved / max(dt, 1e-6)
        s.lat, s.lon = lat, lon
        self._track_position(lat, lon, dt)
        self._kerb_steer(dt)

        try:
            x, y = traci.simulation.convertGeo(lon, lat, fromGeo=True)
            # Cached so _perceive can measure neighbours in this same frame
            # without a round trip per vehicle.
            self._ego_xy = (x, y)
            # keepRoute=2 is what makes this a free body: SUMO places it where
            # we say and stops enforcing a route.
            traci.vehicle.moveToXY(EGO_ID, "", -1, x, y,
                                   angle=s.heading, keepRoute=2)
            traci.vehicle.setSpeed(EGO_ID, s.speed)
        except Exception:
            pass

    def _steer_fit(self, readout: float, error_deg: float, speed: float,
                   dt: float) -> tuple[float, float, float] | None:
        """Live fit of the steering readout against heading error.

        Returns (resting offset, left gain, right gain), or None while there
        is not enough data to trust a fit.

        The offset used to be estimated only while |error| <= 10 degrees, and
        that trapped the fly. Traced near Tin Factory, 2026-09-28: the
        offset came from the parked calibration (-0.013) and the zero-error
        readout while driving was +0.001. Subtracting the stale value put
        the fly's equilibrium at a standing -35 deg error, so the window
        almost never opened (14 of 2,847 frames) and the offset never
        corrected. The fly crabbed 25-35 deg off the road and drifted right
        at junctions, onto branches off its route.

        This fits `readout = rest + k_left * min(err, 0) + k_right *
        max(err, 0)` over exponentially weighted frames (time constant
        `steer_bias_tau_s`, errors inside `steer_fit_window_deg`) and takes
        the intercept as the offset. The intercept extrapolates to zero
        error from wherever the fly actually is, so no window has to open.
        While both slopes have the correct sign, each side's gain is
        normalised toward their mean, clamped to [0.5, 2].
        """
        stats = getattr(self, "_fit_stats", None)
        if stats is None:
            stats = self._fit_stats = [0.0] * 8
        window = self.v.steer_fit_window_deg
        if speed < 1.0 or abs(error_deg) > window:
            return None
        decay = 1.0 - min(1.0, dt / self.v.steer_bias_tau_s)
        u, v = min(error_deg, 0.0), max(error_deg, 0.0)
        for i, value in enumerate((1.0, u, v, readout, u * u, v * v,
                                   u * readout, v * readout)):
            stats[i] = stats[i] * decay + value
        n, su, sv, sy, suu, svv, suy, svy = stats
        # About 10 s of frames before any fit, and 25 deg^2 of spread
        # (a 5 deg standard deviation) on a side before that side counts.
        if n < 10.0 / dt:
            return None
        left_ok = suu / n - (su / n) ** 2 > 25.0
        right_ok = svv / n - (sv / n) ** 2 > 25.0
        den, num = n, sy
        if left_ok:
            den -= su * su / suu
            num -= su * suy / suu
        if right_ok:
            den -= sv * sv / svv
            num -= sv * svy / svv
        if not (left_ok or right_ok) or den < 0.05 * n:
            return None
        rest = num / den
        gain_left = gain_right = 1.0
        if left_ok and right_ok:
            k_left = (suy - su * rest) / suu
            k_right = (svy - sv * rest) / svv
            if k_left > 0.0 and k_right > 0.0:
                k_mean = 0.5 * (k_left + k_right)
                gain_left = max(0.5, min(2.0, k_mean / k_left))
                gain_right = max(0.5, min(2.0, k_mean / k_right))
        return rest, gain_left, gain_right

    def _track_position(self, lat: float, lon: float, dt: float) -> None:
        """Keep the last `stall_window_s` of positions for `_net_speed_ms`."""
        window = getattr(self, "_positions", None)
        if window is None:
            window = self._positions = collections.deque()
        window.append((lat, lon))
        keep = max(2, int(round(self.v.stall_window_s / max(dt, 1e-6))) + 1)
        while len(window) > keep:
            window.popleft()
        self._positions_dt = dt

    def _net_speed_ms(self) -> float | None:
        """Net displacement over the stall window, per second.

        None until the window is full. A pinned fly jitters: on the 93358a2
        Silk Board trace realised speed swung 0.1-1.9 m/s frame to frame
        while the fly sat within 10 m of one spot for 7,000 frames, so the
        instantaneous reading kept resetting the stall timer. Net progress
        over a few seconds does not jitter.
        """
        window = getattr(self, "_positions", None)
        if not window:
            return None
        dt = getattr(self, "_positions_dt", 0.0)
        span = (len(window) - 1) * dt
        if span < self.v.stall_window_s - 1e-6:
            return None
        (lat0, lon0), (lat1, lon1) = window[0], window[-1]
        return haversine_m(lat0, lon0, lat1, lon1) / span

    def _lane_bearing(self, edge: str, pos: float) -> float | None:
        """Compass bearing of `edge`'s lane at `pos` metres along it."""
        if not edge:
            return None
        shape = self._lane_shape(edge)
        if shape is None or len(shape) < 2:
            return None
        remaining = max(0.0, pos)
        segments = list(zip(shape, shape[1:]))
        for index, ((x0, y0), (x1, y1)) in enumerate(segments):
            seg = math.hypot(x1 - x0, y1 - y0)
            if seg <= 0.0:
                continue
            if remaining <= seg or index == len(segments) - 1:
                return math.degrees(math.atan2(x1 - x0, y1 - y0)) % 360.0
            remaining -= seg
        return None

    def _kerb_steer(self, dt: float) -> None:
        """A fly held back by the road slides its heading along the lane.

        `_pull_to_road` fixes the position but used to leave the heading
        alone, so a fly facing off the road stayed facing off it and every
        step was cancelled again. Traced on Silk Board (93358a2, 2026-09-23):
        at the end of `38098070#1` the fly faced 108 degrees, the lane and
        its route target lay 46 degrees to its right, the SNN's steer sat in
        the right-side decode dead zone (+/-0.01), and it realised 0.1-1.9
        m/s of a commanded 4 m/s for 7,000 frames -- the whole pin.

        A car pressed against a kerb is turned by it; this is that contact
        force. It only acts while the road is cancelling at least half the
        commanded motion, so a fly driving freely steers on the connectome
        alone, exactly as before.
        """
        s = self.state
        rate = self.v.kerb_turn_deg_s
        if rate <= 0.0 or s.speed < 1.0:
            return
        if s.realised_speed_ms >= 0.5 * s.speed:
            return
        tangent = self._lane_bearing(s.edge, s.edge_pos)
        if tangent is None:
            return
        diff = wrap180(tangent - s.heading)
        turn = math.copysign(min(abs(diff), rate * dt), diff)
        s.heading = (s.heading + turn) % 360.0

    def _reachable(self, edge: str) -> tuple[set, set, set]:
        """(normal successor edges, junction ids, junction edges) reachable
        from `edge`.

        Read from SUMO's own connection table (`lane.getLinks`) and cached:
        topology does not change during a run.
        """
        cache = getattr(self, "_reach_cache", None)
        if cache is None:
            cache = self._reach_cache = {}
        if edge not in cache:
            succ, junctions, vias = set(), set(), set()
            try:
                for i in range(self.traci.edge.getLaneNumber(edge)):
                    for link in self.traci.lane.getLinks(f"{edge}_{i}"):
                        succ.add(link[0].rsplit("_", 1)[0])
                        via = link[4] if len(link) > 4 else ""
                        if via:
                            vias.add(via.rsplit("_", 1)[0])
                            junctions.add(_junction_of(via.rsplit("_", 1)[0]))
            except Exception:
                pass
            cache[edge] = (succ, junctions, vias)
        return cache[edge]

    def _connected_snap(self, lat: float, lon: float, nearest: str):
        """Refuse a snap onto a road the fly's lane does not connect to.

        `_edge_at` answers with the nearest lane anywhere in the network. In
        2D, flyovers, the roads beneath them and the opposite carriageway
        all lie within a few metres of each other, so a fly 1-2 m off its
        lane could be snapped onto one of them and teleported onto another
        road. Traced near Tin Factory, 2026-09-28: from `1356191564#0`
        (position 72) onto `1089715166` (position 317), which is not a
        successor, and from `1370245925` onto `-1504434604`, the reverse
        carriageway of its next route edge. Each jump added ~400-1,200 m of
        route.

        The fly may be snapped onto its last normal edge, that edge's SUMO
        successors, or the junction lanes that connect to them. Every legal
        turn stays open, so which branch it takes is still up to the fly.
        Returns None when `nearest` is already legal, or when no legal lane
        lies within `vehicle.lane_keep_m` (the fly is genuinely elsewhere),
        otherwise (edge, pos, (x, y)) on the nearest legal lane.
        """
        last = getattr(self, "_last_normal_edge", "")
        if not last or nearest == last:
            return None
        succ, junctions, vias = self._reachable(last)
        if not succ and not junctions:
            return None
        if nearest.startswith(":"):
            if _junction_of(nearest) in junctions:
                return None
        elif nearest in succ:
            return None
        try:
            x, y = self.traci.simulation.convertGeo(lon, lat, fromGeo=True)
        except Exception:
            return None
        best = None
        # The junction lanes matter: past the end of `last` they are the only
        # legal road until the successor starts, and without them the fly was
        # pulled back to the edge end every frame, pinned.
        for candidate in (last, *succ, *vias):
            hit = _project(self._lane_shape(candidate), x, y)
            if hit is not None and (best is None or hit[0] < best[0]):
                best = (hit[0], candidate, hit[1], hit[2])
        if best is None or best[0] > self.v.lane_keep_m:
            return None
        return best[1], best[2], best[3]

    def _pull_to_road(self, lat: float, lon: float,
                      dt: float) -> tuple[float, float]:
        """Close part of the gap back to the nearest *connected* road, and
        record it."""
        traci = self.traci
        edge, pos, lane = self._edge_at(lat, lon)
        if not edge:
            self.state.off_road_m = float("inf")
            return lat, lon

        snapped = self._connected_snap(lat, lon, edge)
        if snapped is not None:
            edge, pos, (x, y) = snapped
        self.state.edge = edge
        self.state.edge_pos = pos
        if not edge.startswith(":"):
            self._last_normal_edge = edge
        try:
            if snapped is None:
                x, y = traci.simulation.convert2D(edge, pos, lane)
            road_lon, road_lat = traci.simulation.convertGeo(x, y)
        except Exception:
            return lat, lon

        self.state.off_road_m = haversine_m(lat, lon, road_lat, road_lon)
        k = min(1.0, self.v.edge_pull_per_s * dt)
        return lat + (road_lat - lat) * k, lon + (road_lon - lon) * k

    # -- reinforcement triggers -----------------------------------------

    def rewards(self) -> tuple[float, float]:
        """(PAM current, PPL1 current), per PLAN.md section 4.

        Reward is still *progress*, not proximity -- a fly parked 100 m from
        the goal is not rewarded forever for standing still, because the gate
        is having closed distance since the last best. What proximity now
        changes is how much a unit of progress is worth: the closer the goal,
        the larger the dopamine, `proximity_scale` above.

        The gate reads `route_distance_to_goal_m`, not the crow-flies
        `distance_to_goal_m`, as of this session. That distinction is the
        other half of the road-following fix in `sense()`: if progress were
        still measured as a straight line to the goal, then any detour the
        road network actually requires -- going around a block to get past a
        building sitting on the direct line -- would read as moving *away*
        from the goal and earn nothing, punishing the fly for the only path
        that was ever available. Gating on distance-along-the-route instead
        means a legal detour that shortens the real remaining drive is
        rewarded exactly like any other progress, which is what lets the
        existing trial-and-error learning here actually discover working
        routes instead of being pushed to keep failing the same straight
        line. `route_distance_to_goal_m` itself falls back to the crow-flies
        distance whenever no route is known, so this degrades to the old
        behaviour rather than breaking when routing can't resolve.

        The progress threshold moved from 100 m to `snn.progress_threshold_m`
        (12 m). At 100 m the channel fired at most twice in a 25,727-frame run
        -- `closed_m` for that entire run was 103.6 m -- so `_reinforce` ran
        homeostasis-only for essentially the whole route and no learning could
        occur. At the measured median speed of 3.82 m/s, 12 m is about three
        seconds of clean running.

        Punishment is the sustained stall, the pothole, and wandering off the
        network, unchanged.
        """
        s = self.state
        r = self.cfg.snn
        goal_m = s.route_distance_to_goal_m
        pam = 0.0
        if goal_m < self.best_distance_m - r.progress_threshold_m:
            self.best_distance_m = goal_m
            pam = 1.0
        elif s.speed > 8.33:                       # sustained clean cruising
            pam = 0.3

        if pam:
            # Scale by how close the goal is, then map onto the PAM cells'
            # measured dynamic range.
            #
            # Both halves matter. The scale is what makes arriving worth more
            # than setting off -- full strength inside `reward_near_m`, two
            # percent at `reward_far_m`. The mapping is what makes that
            # scaling survive contact with the neurons: PAM's current ->
            # firing response is superlinear at the bottom (measured, see
            # `scripts/probe_diagnostics.py` Test D), so injecting the raw
            # fraction as a current delivers 0.5% of peak dopamine where 2%
            # was asked for. `pam_current_min` is read off that curve as the
            # current that actually produces `reward_far_scale` of the
            # dopamine `pam_current_max` produces.
            scale = proximity_scale(goal_m, r.reward_near_m,
                                    r.reward_far_m, r.reward_far_scale)
            span = (scale - r.reward_far_scale) / (1.0 - r.reward_far_scale)
            pam *= r.pam_current_min + span * (r.pam_current_max
                                               - r.pam_current_min)

        # Going the wrong way, measured along the route rather than as a
        # bearing: a legal detour still shortens the drive, so only distance
        # growing counts as wrong. Smoothed, because a single frame's
        # difference is mostly the edge-snapping jitter of `edge_pos`, and
        # punishing that would punish nothing the fly did.
        away_ms = (goal_m - self._prev_route_m) / max(self.step_length, 1e-3)
        self._prev_route_m = goal_m
        self._away_ema += (away_ms - self._away_ema) * r.wrong_way_smoothing
        s.wrong_way_ms = self._away_ema

        ppl1 = 0.0
        if s.stalled_s > 45.0:
            ppl1 = 1.0
        elif is_pothole(s.edge, self.pothole_density) and s.speed > 1.0:
            ppl1 = 0.7
        elif math.isinf(s.off_road_m):
            ppl1 = 0.5
        # Phasic, not tonic, and that distinction is load-bearing.
        # `_reinforce`'s own docstring says it: a sustained punishment
        # depresses the motor pathway the fly needs in order to escape. Held
        # continuously high, this channel did exactly that -- mean pain 0.43
        # across a run, and the steering signal that had just been recovered
        # went back to zero-mean while the road asked for a turn on 99% of
        # frames. So the punishment fires as a pulse when the fly starts
        # getting it wrong, decays, and can only re-fire after a refractory
        # gap, which is also how PPL1 behaves in the animal.
        self._wrong_way_pulse *= math.exp(
            -self.step_length / max(r.wrong_way_pulse_s, 1e-3))
        self._wrong_way_cooldown = max(
            0.0, self._wrong_way_cooldown - self.step_length)
        if self._away_ema > r.wrong_way_threshold_ms:
            # Joel, watching a run: "PPL1 needs to be pretty high when going
            # the wrong way, especially when it's far away from the goal."
            # Before this the only punishments were a 45-second stall, a
            # pothole and leaving the network entirely -- driving away from
            # the goal at speed, the one thing a navigating fly most needs
            # to stop doing, cost it nothing at all.
            #
            # Distance scales it because a wrong turn 50 m out is a wrong
            # turn; the same heading 5 km out is the whole trip going wrong.
            # The ceiling is deliberate: enough to trigger the escape
            # response, not so much that the fly thrashes.
            # Graded on two axes, not one. Distance alone made the channel
            # effectively binary on a short route -- anything past
            # `wrong_way_full_m` pinned it at the ceiling, so Joel watched
            # the pain rise once and then sit there. How *fast* the fly is
            # getting it wrong is the half that keeps moving: drifting away
            # at 1 m/s and barrelling away at 5 are not the same mistake.
            far = min(1.0, goal_m / max(r.wrong_way_full_m, 1.0))
            hard = min(1.0, self._away_ema / max(r.wrong_way_full_ms, 1e-3))
            severity = 0.5 * far + 0.5 * hard
            target = (r.wrong_way_min
                      + (r.wrong_way_max - r.wrong_way_min) * severity)
            if self._wrong_way_cooldown <= 0.0:
                self._wrong_way_pulse = max(self._wrong_way_pulse, target)
                self._wrong_way_cooldown = r.wrong_way_refractory_s
        ppl1 = max(ppl1, self._wrong_way_pulse)
        return pam, ppl1

    def step_sumo(self) -> None:
        self.traci.simulationStep()

    # -- episodes --------------------------------------------------------

    def reset_episode(self, navigator=None) -> None:
        """Teleport back to the origin for another attempt, in-process.

        Episodic training needs many runs at the same route, and the obvious
        way to get them -- restart the process -- costs 75-130 s of SUMO
        network parse plus ~28 s of connectome build every time. Nothing about
        the road network changes between episodes, so nothing needs rebuilding:
        `moveToXY` puts the vehicle back and the simulation carries on with its
        background traffic still running.

        `navigator.reset()` clears membrane voltages, spikes and the
        eligibility trace. It deliberately does NOT touch `p_val`, the plastic
        overlay -- that is the only thing the fly carries between episodes, and
        zeroing it would throw away the entire point of running more than one.
        """
        s = self.state
        s.lat, s.lon = self.origin
        s.heading = bearing_deg(*self.origin, *self.goal)
        s.speed = 0.0
        s.realised_speed_ms = 0.0
        s.throttle_cmd = 0.0
        s.travelled_m = 0.0
        s.stalled_s = 0.0
        self._positions = collections.deque()
        self._panic_prev = False
        self._fit_stats = None
        self._last_normal_edge = ""
        s.off_road_m = 0.0
        s.nearby = []
        s.distance_to_goal_m = haversine_m(*self.origin, *self.goal)
        s.route_distance_to_goal_m = s.distance_to_goal_m
        self._prev_dist = {}

        # The route resets too: whatever path the fly had worked out from
        # wherever it ended the last episode does not start anywhere near the
        # origin. Cleared here and recomputed below, once `s.edge` is current.
        self._route_edge = None
        self._route_edges = []
        self._route_length_m = None
        self._route_anchor_edge = None
        # The held route distance goes with it. Holding the *previous*
        # episode's last route distance across a teleport back to the origin
        # would be exactly the stale-value bug `_route_distance_now` exists
        # to prevent, one level up.
        self._last_route_m = None

        try:
            x, y = self.traci.simulation.convertGeo(s.lon, s.lat, fromGeo=True)
            self._ego_xy = (x, y)
            self.traci.vehicle.moveToXY(EGO_ID, "", -1, x, y,
                                        angle=s.heading, keepRoute=2)
            self.traci.vehicle.setSpeed(EGO_ID, 0.0)
        except Exception:
            pass

        # `_road_edge_near`, not raw `_edge_at`: the same lookup `_spawn_ego`
        # uses for episode 1. At the test routes' origins raw `_edge_at`
        # answers with an internal junction edge, `_update_route` refuses to
        # route from one, and the seed below fell back to crow-flies. Measured
        # on every multi-episode run on record: episode 1 started its ratchet
        # in route metres and all 20 later episodes at exactly the crow-flies
        # distance (426.0 m on the 500 m route, 502.2 m on the twisty one) --
        # so `rewards()` was 4.8x too tight on the twisty route in every
        # episode after the first, which is the failure the comment below
        # describes as fixed. It was fixed for the fallback, not for the edge
        # lookup that feeds it.
        edge, pos, _lane = self._road_edge_near(s.lat, s.lon)
        s.edge = edge
        s.edge_pos = pos

        self._update_route()
        s.route_distance_to_goal_m = self._route_distance_now()
        # Reset the ratchet too: `best_distance_m` is monotonic within an
        # episode, so carrying it across would mean the next run starts
        # already "past" its own progress and can never earn reward again.
        # Matches whichever distance `rewards()` now gates on -- and this
        # goes through `_route_distance_now` for the same reason the
        # constructor guards on `route_m is not None`: seeding the ratchet
        # with crow-flies metres while `rewards()` later compares route
        # metres sets it 4.8x too tight on the twisty route, so no progress
        # could ever be earned. The two paths disagreed until 2026-09-22.
        self.best_distance_m = s.route_distance_to_goal_m
        # Same reasoning for the wrong-way detector: a fresh episode has not
        # moved anywhere yet, and the jump from wherever the last one ended
        # back to the origin is a teleport, not a wrong turn.
        self._prev_route_m = s.route_distance_to_goal_m
        # Provisional, as at spawn: the lookup above picks *an* edge near the
        # origin, not necessarily the one the fly drives off on.
        self.start_route_m = s.route_distance_to_goal_m
        self.route_seeded = False
        self._away_ema = 0.0
        self._wrong_way_pulse = 0.0
        self._wrong_way_cooldown = 0.0
        s.wrong_way_ms = 0.0
        # Re-measure the steering offset for the new episode: the resting
        # asymmetry moves with the channel mix, and a fresh episode has a
        # fresh one.
        self._calibrating = int(self.v.steer_calibration_s
                                / max(self.step_length, 1e-3))
        self._calibrate_sum = 0.0
        self._calibrate_n = 0
        self._steer_filtered = 0.0

        if navigator is not None:
            navigator.reset()
