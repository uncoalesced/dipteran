"""Pydantic-validated configuration.

Everything route-specific lives in config.yaml -- no hardcoded lat/lon in code.
Relative paths in the `paths` block resolve against the repo root so scripts
work from any working directory.

Run `python -m dipteran.config` for a self-check.
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
