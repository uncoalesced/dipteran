"""Pydantic-validated configuration.

Everything route-specific lives in config.yaml -- no hardcoded lat/lon in code.
Relative paths in the `paths` block resolve against the repo root so scripts
work from any working directory.

Run `python -m dipteran.config` for a self-check.

Engineered by uncoalesced.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, model_validator

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "config.yaml"

# Relative paths in config.yaml resolve against the repo by default. Set
# DIPTERAN_DATA_ROOT to put the multi-gigabyte artefacts somewhere else --
# specifically, off a WSL 9p mount. Reading the 862 MB .net.xml and rewriting
# a 64 MB JSON across /mnt/<drive> is both slow and, on this machine,
# destabilising: sustained 9p I/O took the WSL VM down repeatedly, at the same
# point each run, with plenty of free RAM.
DATA_ROOT = Path(os.environ.get("DIPTERAN_DATA_ROOT") or REPO_ROOT)

EARTH_RADIUS_M = 6_371_008.8


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


class Point(BaseModel):
    label: str
    lat: float = Field(ge=-90.0, le=90.0)
    lon: float = Field(ge=-180.0, le=180.0)


class BBox(BaseModel):
    west: float = Field(ge=-180.0, le=180.0)
    south: float = Field(ge=-90.0, le=90.0)
    east: float = Field(ge=-180.0, le=180.0)
    north: float = Field(ge=-90.0, le=90.0)

    @model_validator(mode="after")
    def _ordered(self) -> "BBox":
        if self.west >= self.east:
            raise ValueError(f"bbox.west ({self.west}) must be < bbox.east ({self.east})")
        if self.south >= self.north:
            raise ValueError(f"bbox.south ({self.south}) must be < bbox.north ({self.north})")
        return self

    def contains(self, point: Point) -> bool:
        return (
            self.west <= point.lon <= self.east
            and self.south <= point.lat <= self.north
        )

    def as_maplibre(self) -> list[list[float]]:
        """[[SW lon, SW lat], [NE lon, NE lat]] -- the frontend's maxBounds form."""
        return [[self.west, self.south], [self.east, self.north]]


class Route(BaseModel):
    name: str
    origin: Point
    destination: Point
    dist_m: float = Field(gt=0.0)
    network_type: Literal[
        "all", "all_public", "bike", "drive", "drive_service", "walk"
    ]
    bbox: BBox

    @model_validator(mode="after")
    def _endpoints_reachable(self) -> "Route":
        for name, pt in (("origin", self.origin), ("destination", self.destination)):
            if not self.bbox.contains(pt):
                raise ValueError(f"route.{name} ({pt.lat}, {pt.lon}) falls outside route.bbox")
        # ox.graph_from_point(origin, dist=dist_m) yields a square-ish extract
        # centred on origin. If dist_m is smaller than the A->B separation the
        # destination is simply not in the graph, and every downstream step
        # silently works on a route that cannot be completed.
        span = self.separation_m
        if self.dist_m < span:
            raise ValueError(
                f"route.dist_m ({self.dist_m:.0f} m) is smaller than the "
                f"origin->destination distance ({span:.0f} m); the OSM extract "
                "would not contain the destination"
            )
        return self

    @property
    def separation_m(self) -> float:
        return haversine_m(
            self.origin.lat, self.origin.lon,
            self.destination.lat, self.destination.lon,
        )


class Compute(BaseModel):
    device: str = "cuda:0"
    dtype: Literal["bfloat16", "float16", "float32"] = "float16"
    # Drop connected pairs below this many aggregate synapses. 0 keeps
    # everything (PLAN.md's maximum-fidelity decision); the FlyWire papers
    # conventionally treat >=5 as a reliable connection.
    min_syn_count: int = Field(default=0, ge=0)


class Paths(BaseModel):
    data_dir: Path
    cache_dir: Path
    connectome_edges: Path
    cell_annotations: Path
    neuron_index: Path
    cluster_ids: Path
    osm_extract: Path
    sumo_net: Path
    graphml: Path
    id_map: Path

    @model_validator(mode="after")
    def _absolutise(self) -> "Paths":
        for field in type(self).model_fields:
            value: Path = getattr(self, field)
            if not value.is_absolute():
                object.__setattr__(self, field, (DATA_ROOT / value).resolve())
        return self


class Sumo(BaseModel):
    step_length: float = Field(gt=0.0)
    binary_dir: Optional[Path] = None
    # Left-hand traffic, as India drives. netconvert builds a right-hand
    # network by default no matter where the OSM extract is from, so this
    # has to be asked for explicitly -- and changing it means rebuilding
    # `cache/bangalore.net.xml`, since the side of the road is baked into
    # the lane geometry, not applied at runtime.
    left_hand: bool = True


class Vehicle(BaseModel):
    """Fly-vehicle envelope and the motor transfer.

    `throttle_scale` and `steer_scale` are the boundary between two different
    quantities: `MotorOutput.throttle` is a DNg02 *population firing rate*,
    measured in roughly [0.05, 0.24], while the motion model wants a command
    in [0, 1]. Treating the rate as if it were the command pins the vehicle
    at 5% throttle -- an accurate brain wired to a mis-scaled servo.
    """

    throttle_scale: float = Field(default=4.0, gt=0.0)
    steer_scale: float = Field(default=2.5, gt=0.0)
    max_speed_ms: float = Field(default=16.0, gt=0.0)
    max_accel_ms2: float = Field(default=2.5, gt=0.0)
    brake_ms2: float = Field(default=8.0, gt=0.0)
    max_turn_deg_s: float = Field(default=45.0, gt=0.0)
    # Fraction of the off-road offset closed per second by the soft pull.
    edge_pull_per_s: float = Field(default=2.0, ge=0.0)
    pothole_density: float = Field(default=0.05, ge=0.0, le=1.0)

    # Time constant of the steering low-pass, seconds. DNa02 is one neuron
    # per side and a frame is five ticks, so the per-frame readout is mostly
    # noise -- see the comment in `World.apply`. 0 disables the filter and
    # restores the raw per-frame command.
    steer_smoothing_s: float = Field(default=1.2, ge=0.0)
    # Time constant of the standing-offset estimate subtracted from the
    # steering command, seconds. Well above `steer_smoothing_s` so a real
    # sustained turn survives it, well below a route so a drifting resting
    # asymmetry does not. 0 disables it and leaves `snn.steer_bias` alone.
    steer_bias_tau_s: float = Field(default=30.0, ge=0.0)
    # The offset is only re-estimated while the heading error is inside
    # this many degrees, because that is the condition where zero is known
    # to be the right command. Wider makes the estimate faster and dirtier.
    steer_bias_window_deg: float = Field(default=10.0, gt=0.0)
    # Seconds spent parked at the start of a run measuring that offset
    # directly, with both PFL3 lobes balanced so the readout carries
    # nothing but the resting asymmetry. 0 skips the blink and trusts
    # `snn.steer_bias` alone.
    steer_calibration_s: float = Field(default=2.0, ge=0.0)
    # How far along the route the steering target sits, metres. Too short
    # and the bearing is noisy and twitchy; too long and the fly cuts
    # corners across whatever sits between it and the target.
    route_lookahead_m: float = Field(default=45.0, gt=0.0)
    # Degrees per second the lane turns a fly whose motion the road pull is
    # cancelling -- see `World._kerb_steer`. 0 disables it.
    kerb_turn_deg_s: float = Field(default=60.0, ge=0.0)
    # Window for the net-progress stall measure -- see `World._net_speed_ms`.
    stall_window_s: float = Field(default=5.0, gt=0.0)
    # Extra gain on right turns, from the per-side authority Test F
    # measures -- see `World.apply`. 1.0 is symmetric.
    steer_right_gain: float = Field(default=1.0, gt=0.0)
    # Heading errors beyond this are left out of the live steering fit --
    # see `World._steer_fit`.
    steer_fit_window_deg: float = Field(default=60.0, gt=0.0)
    # How far a fly may be from a lane it is connected to before a snap onto
    # an unconnected road is allowed -- see `World._connected_snap`.
    lane_keep_m: float = Field(default=8.0, gt=0.0)
    # Seconds of no net progress, motor open, before lane-keeping lets go
    # for 5 s -- see `World._connected_snap`.
    lane_keep_release_s: float = Field(default=10.0, gt=0.0)


class Snn(BaseModel):
    """LIF and reinforcement parameters.

    `gain` is measured rather than chosen -- see `navigator.calibrate_gain`.
    """

    gain: float = Field(default=16.0, gt=0.0)
    # Scales external sensory current. Separate from `gain`, which scales the
    # recurrent drive: the world supplies small, uneven currents (an inverse
    # odour law is 0.047 at the start of a 10.6 km route) and they need their
    # own scale to reach threshold.
    sensory_gain: float = Field(default=1.0, gt=0.0)
    decay: float = Field(default=0.9, ge=0.0, lt=1.0)
    v_thresh: float = Field(default=1.0, gt=0.0)
    v_reset: float = -0.2
    subticks: int = Field(default=10, ge=1)
    # Relative, not absolute: the update is scaled by each synapse's own
    # anatomical weight (see navigator._reinforce). With the trace saturating
    # at 1/(1 - trace_decay) = 5, this caps drift at 0.05% of baseline per
    # tick, so meaningful adaptation needs ~1000 ticks (20 s) of sustained
    # dopamine. An absolute 0.01 here moved weights 500x their own magnitude
    # per tick and flatlined the motors inside 70 s.
    learning_rate: float = Field(default=1e-4, ge=0.0)
    trace_decay: float = Field(default=0.8, ge=0.0, lt=1.0)
    # Slow pull back toward the anatomy, applied every tick regardless of
    # dopamine, so a punished episode is survivable. Without it a long stall
    # depresses the very motor pathway needed to escape it.
    homeostasis: float = Field(default=0.001, ge=0.0, lt=1.0)
    # Learned weights stay within +/- this fraction of |anatomical weight|.
    plastic_span: float = Field(default=0.5, gt=0.0)

    # Resting value of rate(DNa02_right) - rate(DNa02_left), subtracted in
    # `navigator._decode` so the steering readout is zero-centred. Measured,
    # not chosen: DNa02-left is silent and DNa02-right runs at ~101 Hz under
    # information-free drive, and both cells are equally well reconstructed
    # (in-degree 742 vs 751), so the gap is the network's own E/I balance
    # rather than a data artefact. Left in, `steer` is positive on every tick
    # and the vehicle can only turn right. Reproduce with
    # `python scripts/probe_diagnostics.py`.
    #
    # Moves with `steer_half_angle_deg`, because it is measured at the
    # dead-ahead operating point and that point is cos(half_angle). 0.209 is
    # the 70-degree figure; the 0.191 this defaulted to until 2026-09-22 was
    # measured at 45. Kept in step with config.yaml deliberately -- a stale
    # default here stays invisible until someone runs without a config file.
    steer_bias: float = 0.209

    # -- reward shaping --------------------------------------------------
    # Metres of progress that earn a dopamine pulse. Was 100 m, which fired
    # at most twice in a 25,727-frame route run (total distance closed: 103.6
    # m), leaving plasticity effectively off for the whole route.
    progress_threshold_m: float = Field(default=12.0, gt=0.0)
    # Distance at which a progress reward is worth full strength, and the
    # distance at which it decays to `reward_far_scale`. The fly should care
    # more about the last 500 m than about the first kilometre of fifteen.
    reward_near_m: float = Field(default=500.0, gt=0.0)
    reward_far_m: float = Field(default=15000.0, gt=0.0)
    reward_far_scale: float = Field(default=0.02, gt=0.0, le=1.0)
    # The PAM cells' usable current range. `pam_current_min` is not
    # `reward_far_scale` expressed as a current: PAM's current -> firing
    # transfer is superlinear near zero, so a current of 0.02 yields 0.5% of
    # peak dopamine rather than 2%. 0.05 is where the measured curve actually
    # delivers 2%.
    pam_current_min: float = Field(default=0.05, gt=0.0, le=1.0)
    pam_current_max: float = Field(default=1.0, gt=0.0, le=1.0)

    # Wrong-way punishment. Driving away from the goal used to cost the fly
    # nothing -- PPL1 only fired on a 45-second stall, a pothole, or leaving
    # the road network -- so the one behaviour a navigator most needs to
    # unlearn carried no teaching signal at all.
    #
    # `wrong_way_threshold_ms` is the rate of route-distance growth that
    # counts as going the wrong way rather than as edge-snapping jitter;
    # `wrong_way_smoothing` is the per-frame weight of the EMA that measures
    # it. The punishment runs from `wrong_way_min` at the goal to
    # `wrong_way_max` at `wrong_way_full_m` away and no higher -- high
    # enough to trigger the escape response, bounded so it cannot drown out
    # every other channel.
    wrong_way_threshold_ms: float = Field(default=0.5, ge=0.0)
    wrong_way_smoothing: float = Field(default=0.05, gt=0.0, le=1.0)
    wrong_way_min: float = Field(default=0.5, ge=0.0, le=1.0)
    wrong_way_max: float = Field(default=0.8, ge=0.0, le=1.0)
    wrong_way_full_m: float = Field(default=800.0, gt=0.0)
    # The other half of the grading: the rate of wrong-way travel that
    # counts as getting it maximally wrong. Without this the punishment is
    # binary on any route shorter than `wrong_way_full_m`.
    wrong_way_full_ms: float = Field(default=4.0, gt=0.0)
    # The punishment is a decaying pulse, not a level. Held continuously
    # high it depresses the motor pathway it is meant to correct -- see
    # `_reinforce` -- so it decays with this time constant and cannot
    # re-fire until `wrong_way_refractory_s` has passed.
    # Width of each PFL3 cosine lobe, degrees. 45 was the original pair and
    # leaves a dead zone past +/-135, where a fly facing away from its route
    # gets no directional drive and turns circles. 90 removes the dead zone
    # but also removes the overlap dead ahead, and the overlap is what makes
    # a left turn expressible at all -- DNa02-left rests silent, so left is
    # "DNa02-right below its baseline", and without overlap there is no
    # baseline. Measured at 90: a positive command on 100% of frames. 70
    # keeps the overlap and pushes the dead zone past 160 degrees.
    steer_half_angle_deg: float = Field(default=70.0, gt=0.0, lt=90.0)

    wrong_way_pulse_s: float = Field(default=2.5, gt=0.0)
    wrong_way_refractory_s: float = Field(default=10.0, gt=0.0)

    @model_validator(mode="after")
    def _reward_band_is_ordered(self) -> "Snn":
        if self.reward_near_m >= self.reward_far_m:
            raise ValueError(
                f"reward_near_m ({self.reward_near_m}) must sit below "
                f"reward_far_m ({self.reward_far_m}); the scale runs from full "
                "strength near the goal down to reward_far_scale far away"
            )
        if self.pam_current_min >= self.pam_current_max:
            raise ValueError(
                f"pam_current_min ({self.pam_current_min}) must sit below "
                f"pam_current_max ({self.pam_current_max})"
            )
        if self.wrong_way_min > self.wrong_way_max:
            raise ValueError(
                f"wrong_way_min ({self.wrong_way_min}) must sit at or below "
                f"wrong_way_max ({self.wrong_way_max}); the punishment grows "
                "with distance from the goal, it does not shrink"
            )
        return self

    @model_validator(mode="after")
    def _reset_below_threshold(self) -> "Snn":
        if self.v_reset >= self.v_thresh:
            raise ValueError(
                f"v_reset ({self.v_reset}) must sit below v_thresh "
                f"({self.v_thresh}); a neuron that resets at or above its own "
                "threshold fires every sub-tick forever"
            )
        return self


class Config(BaseModel):
    route: Route
    compute: Compute
    paths: Paths
    sumo: Sumo
    snn: Snn = Snn()
    vehicle: Vehicle = Vehicle()


def load_config(path: Path | str | None = None) -> Config:
    """Load and validate config.yaml."""
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return Config.model_validate(raw)


def demo() -> None:
    cfg = load_config()
    r = cfg.route
    print(f"route          : {r.name}")
    print(f"  origin       : {r.origin.label} ({r.origin.lat}, {r.origin.lon})")
    print(f"  destination  : {r.destination.label} ({r.destination.lat}, {r.destination.lon})")
    print(f"  separation   : {r.separation_m / 1000:.2f} km  (extract radius {r.dist_m / 1000:.1f} km)")
    print(f"  network_type : {r.network_type}")
    print(f"  maxBounds    : {r.bbox.as_maplibre()}")
    print(f"device         : {cfg.compute.device} / {cfg.compute.dtype}")
    print(f"cache_dir      : {cfg.paths.cache_dir}")

    assert cfg.paths.sumo_net.is_absolute(), "paths must resolve to absolute"
    assert r.bbox.contains(r.origin) and r.bbox.contains(r.destination)

    # An extract radius that cannot reach the destination must be rejected.
    bad = cfg.model_dump()
    bad["route"]["dist_m"] = 100.0
    try:
        Config.model_validate(bad)
    except Exception as exc:
        assert "dist_m" in str(exc), exc
    else:
        raise AssertionError("dist_m shorter than A->B separation was accepted")

    # A destination outside the bbox must be rejected.
    bad = cfg.model_dump()
    bad["route"]["destination"]["lat"] = 28.61  # Delhi
    try:
        Config.model_validate(bad)
    except Exception as exc:
        assert "bbox" in str(exc), exc
    else:
        raise AssertionError("destination outside bbox was accepted")

    print("\nself-check OK")


if __name__ == "__main__":
    demo()
