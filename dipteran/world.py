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


def steering_drive(error_deg: float, half_angle_deg: float = 45.0
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

        error = +45 (goal to the right) -> left lobe 1.0, right lobe 0.0
        error =   0 (dead ahead)        -> both 0.707, no net turn
        error = 180 (behind)            -> both 0.0, the dead zone
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
    edge: str = ""
    off_road_m: float = 0.0
    distance_to_goal_m: float = 0.0
    travelled_m: float = 0.0
    stalled_s: float = 0.0
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
        self.best_distance_m = self.state.distance_to_goal_m

        self._prev_dist: dict[str, float] = {}
        self._ego_xy: tuple[float, float] | None = None
        self._spawn_background(background)
        self._spawn_ego()

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
                self.background += 1
            except Exception:
                continue

    def _spawn_ego(self) -> None:
        traci = self.traci
        edge, _pos, _lane = self._edge_at(*self.origin)
        if not edge:
            raise RuntimeError("origin does not resolve to any SUMO edge")
        self.state.edge = edge
        traci.route.add(EGO_ROUTE, [edge])
        traci.vehicle.add(EGO_ID, EGO_ROUTE, departLane="best")
        traci.vehicle.setSpeedMode(EGO_ID, 0)      # the SNN drives, not Krauss
        traci.simulationStep()
        try:
            self._ego_xy = traci.vehicle.getPosition(EGO_ID)
        except Exception:
            self._ego_xy = None
        traci.vehicle.subscribeContext(
            EGO_ID, traci.constants.CMD_GET_VEHICLE_VARIABLE, 60.0,
            [traci.constants.VAR_POSITION, traci.constants.VAR_SPEED,
             traci.constants.VAR_ANGLE])

    # -- sensing ---------------------------------------------------------

    def sense(self, dt: float) -> dict:
        """Sensory currents keyed by cluster name, per PLAN.md section 3."""
        s = self.state
        s.distance_to_goal_m = haversine_m(s.lat, s.lon, *self.goal)

        # Odour beacon: PLAN.md's inverse gradient, normalised so it saturates
        # within smelling distance of the goal rather than blowing up to
        # infinity on arrival.
        odour = min(1.0, 500.0 / max(s.distance_to_goal_m, 1.0))

        # Sun compass: EPG drive peaks when the fly faces the goal. Kept
        # unchanged -- it is a real alignment signal and Phase 2 validated the
        # behaviour built on it -- but it is even in `error` and so carries no
        # turn direction at all. The signed half of the compass is the PFL3
        # pair below; this scalar stays as the "how aligned am I" channel.
        error = wrap180(bearing_deg(s.lat, s.lon, *self.goal) - s.heading)
        compass = max(0.0, math.cos(math.radians(error)))
        pfl3_left, pfl3_right = steering_drive(error)

        # Optic flow: what the fly is *actually* achieving, not what it asked
        # for. Zero flow with the throttle open is the gridlock signal.
        flow = min(1.0, s.speed / self.v.max_speed_ms)

        nearby, loom, horn = self._perceive(dt)
        s.nearby = nearby

        pothole = is_pothole(s.edge, self.pothole_density) and s.speed > 1.0

        if s.throttle_cmd > 0.3 and s.speed < 0.5:
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
            # `min(0.4, lobe)` looks equivalent and destroys the channel: dead
            # ahead the lobes are 0.707/0.707 and both clip to 0.4/0.4, while
            # a goal hard right gives 1.0/0.0 which clips to 0.4/0.0 -- the
            # left lobe reads 0.4 in both cases, so half the steering range
            # collapses to a constant. Multiplying keeps the ratio intact:
            # 0.283/0.283 dead ahead, 0.4/0.0 hard right.
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
        # than asked of SUMO per vehicle: over a 60 m context radius the
        # flat-earth error is millimetres, and it costs no round trip.
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
            if abs(relative) <= LOOM_FIELD_DEG:
                loom = max(loom, looming_rate(self._prev_dist.get(vid, dist),
                                              dist, dt))
            nearby.append({
                "id": vid,
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

        steer = max(-1.0, min(1.0, motor.steer * self.v.steer_scale))
        throttle = max(0.0, min(1.0, motor.throttle * self.v.throttle_scale))

        s.throttle_cmd = throttle
        s.heading = (s.heading + steer * self.v.max_turn_deg_s * dt) % 360.0

        if motor.panic > 0.0:
            # Giant fiber: full braking plus an uncontrolled swerve, which is
            # what DNp01 actually produces -- an escape, not a safe stop.
            s.speed = max(0.0, s.speed - self.v.brake_ms2 * dt)
            s.heading = (s.heading + math.copysign(25.0, steer or 1.0)) % 360.0
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
        s.travelled_m += haversine_m(s.lat, s.lon, lat, lon)
        s.lat, s.lon = lat, lon

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

    def _pull_to_road(self, lat: float, lon: float,
                      dt: float) -> tuple[float, float]:
        """Close part of the gap back to the nearest road, and record it."""
        traci = self.traci
        edge, pos, lane = self._edge_at(lat, lon)
        if not edge:
            self.state.off_road_m = float("inf")
            return lat, lon

        self.state.edge = edge
        try:
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
        pam = 0.0
        if s.distance_to_goal_m < self.best_distance_m - r.progress_threshold_m:
            self.best_distance_m = s.distance_to_goal_m
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
            scale = proximity_scale(s.distance_to_goal_m, r.reward_near_m,
                                    r.reward_far_m, r.reward_far_scale)
            span = (scale - r.reward_far_scale) / (1.0 - r.reward_far_scale)
            pam *= r.pam_current_min + span * (r.pam_current_max
                                               - r.pam_current_min)

        ppl1 = 0.0
        if s.stalled_s > 45.0:
            ppl1 = 1.0
        elif is_pothole(s.edge, self.pothole_density) and s.speed > 1.0:
            ppl1 = 0.7
        elif math.isinf(s.off_road_m):
            ppl1 = 0.5
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
        s.throttle_cmd = 0.0
        s.travelled_m = 0.0
        s.stalled_s = 0.0
        s.off_road_m = 0.0
        s.nearby = []
        s.distance_to_goal_m = haversine_m(*self.origin, *self.goal)
        # Reset the ratchet too: `best_distance_m` is monotonic within an
        # episode, so carrying it across would mean the next run starts
        # already "past" its own progress and can never earn reward again.
        self.best_distance_m = s.distance_to_goal_m
        self._prev_dist = {}

        try:
            x, y = self.traci.simulation.convertGeo(s.lon, s.lat, fromGeo=True)
            self._ego_xy = (x, y)
            self.traci.vehicle.moveToXY(EGO_ID, "", -1, x, y,
                                        angle=s.heading, keepRoute=2)
            self.traci.vehicle.setSpeed(EGO_ID, 0.0)
        except Exception:
            pass

        edge, _pos, _lane = self._edge_at(s.lat, s.lon)
        s.edge = edge

        if navigator is not None:
            navigator.reset()
