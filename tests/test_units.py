"""Unit tests for the pure logic Phase 1 shipped without any.

GATES.md G1/G2 record this gap: the `.net.xml` parser, the point-to-polyline
geometry the whole OSM<->SUMO join rests on, the `osmid` normaliser, the
way-index strategy selection and the BFS all went to production untested.
Phase 2 stacks the navigator on top of them, so they get tests first.

Nothing here touches the 1 GB downloaded corpus -- the SUMO parser is
exercised against a synthetic `.net.xml` written to a temp dir, which is the
point: these have to run on a fresh clone with no data fetched.

    python tests/test_units.py

Engineered by uncoalesced.
"""

from __future__ import annotations

import math
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from dipteran.navigator import (  # noqa: E402
    MotorOutput,
    classify_state,
    telemetry,
)
from dipteran.roads import (  # noqa: E402
    DEG_LAT_M,
    SumoNet,
    _osmids,
    _way_index,
    point_to_polyline_m,
)
from dipteran.world import (  # noqa: E402
    SENSORY_CEILING,
    bearing_deg,
    haversine_m,
    is_pothole,
    looming_rate,
    proximity_scale,
    steering_drive,
    wrap180,
)

# --------------------------------------------------------------------------
# synthetic network
# --------------------------------------------------------------------------

# Bangalore sits in UTM zone 43N. netOffset is what netconvert subtracted from
# the projected coordinates, so network (0,0) here maps back to UTM
# (785000, 1429000) -- inside the config.yaml bbox, which the projection test
# asserts rather than hard-coding a lon/lat.
PROJ = "+proj=utm +zone=43 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
NET_OFFSET = (-785_000.0, -1_429_000.0)

# A 300 m chain 100#0 -> 200#0 -> 300#0, plus an island 900#0 reachable from
# nothing, plus the three edges the parser is supposed to reject.
NET_XML = f"""<?xml version="1.0" encoding="UTF-8"?>
<net version="1.20">
    <location netOffset="{NET_OFFSET[0]:.2f},{NET_OFFSET[1]:.2f}"
              convBoundary="0.00,0.00,400.00,100.00"
              origBoundary="77.0,12.0,78.0,13.0"
              projParameter="{PROJ}"/>

    <edge id="100#0" from="n0" to="n1" priority="5">
        <lane id="100#0_0" index="0" speed="13.89" length="100.00"
              shape="0.00,0.00 100.00,0.00">
            <param key="origId" value="100"/>
        </lane>
        <lane id="100#0_1" index="1" speed="13.89" length="104.00"
              shape="0.00,3.20 100.00,3.20">
            <param key="origId" value="100"/>
        </lane>
    </edge>

    <edge id="200#0" from="n1" to="n2" priority="5">
        <lane id="200#0_0" index="0" speed="13.89" length="100.00"
              shape="100.00,0.00 200.00,0.00">
            <param key="origId" value="200 201"/>
        </lane>
    </edge>

    <edge id="300#0" from="n2" to="n3" priority="5">
        <lane id="300#0_0" index="0" speed="13.89" length="100.00"
              shape="200.00,0.00 300.00,0.00">
            <param key="origId" value="300"/>
        </lane>
    </edge>

    <edge id="900#0" from="n8" to="n9" priority="5">
        <lane id="900#0_0" index="0" speed="13.89" length="50.00"
              shape="0.00,500.00 50.00,500.00">
            <param key="origId" value="900"/>
        </lane>
    </edge>

    <!-- internal junction connector: must be skipped -->
    <edge id=":n1_0" function="internal">
        <lane id=":n1_0_0" index="0" speed="13.89" length="5.00"
              shape="100.00,0.00 100.50,0.00"/>
    </edge>

    <!-- named like a normal edge but flagged internal: also skipped -->
    <edge id="777#0" function="internal">
        <lane id="777#0_0" index="0" speed="13.89" length="5.00"
              shape="1.00,1.00 2.00,2.00"/>
    </edge>

    <!-- no lane shape at all: skipped, but counted -->
    <edge id="888#0" from="n5" to="n6" priority="5">
        <lane id="888#0_0" index="0" speed="13.89" length="10.00"/>
    </edge>

    <connection from="100#0" to="200#0" fromLane="0" toLane="0" dir="s"/>
    <connection from="200#0" to="300#0" fromLane="0" toLane="0" dir="s"/>
    <connection from=":n1_0" to="200#0" fromLane="0" toLane="0" dir="s"/>
</net>
"""

# No origId params anywhere: forces _way_index onto its edge-id fallback.
NET_XML_NO_ORIGID = f"""<?xml version="1.0" encoding="UTF-8"?>
<net version="1.20">
    <location netOffset="{NET_OFFSET[0]:.2f},{NET_OFFSET[1]:.2f}"
              convBoundary="0.00,0.00,200.00,100.00"
              origBoundary="77.0,12.0,78.0,13.0"
              projParameter="{PROJ}"/>
    <edge id="100#0" from="n0" to="n1" priority="5">
        <lane id="100#0_0" index="0" speed="13.89" length="100.00"
              shape="0.00,0.00 100.00,0.00"/>
    </edge>
    <edge id="-100#1" from="n1" to="n0" priority="5">
        <lane id="-100#1_0" index="0" speed="13.89" length="100.00"
              shape="100.00,0.00 0.00,0.00"/>
    </edge>
</net>
"""

# Edge IDs SUMO_EDGE_ID cannot parse either: no way index is possible.
NET_XML_UNPARSEABLE = f"""<?xml version="1.0" encoding="UTF-8"?>
<net version="1.20">
    <location netOffset="{NET_OFFSET[0]:.2f},{NET_OFFSET[1]:.2f}"
              convBoundary="0.00,0.00,200.00,100.00"
              origBoundary="77.0,12.0,78.0,13.0"
              projParameter="{PROJ}"/>
    <edge id="gneE0" from="n0" to="n1" priority="5">
        <lane id="gneE0_0" index="0" speed="13.89" length="100.00"
              shape="0.00,0.00 100.00,0.00"/>
    </edge>
</net>
"""


# --------------------------------------------------------------------------
# point_to_polyline_m
# --------------------------------------------------------------------------

def test_polyline_single_point():
    """A one-point polyline degrades to plain point-to-point distance."""
    d = point_to_polyline_m(np.array([3.0, 4.0]), np.array([[0.0, 0.0]]))
    assert abs(d - 5.0) < 1e-9, d


def test_polyline_point_on_line():
    poly = np.array([[0.0, 0.0], [100.0, 0.0]])
    assert point_to_polyline_m(np.array([50.0, 0.0]), poly) < 1e-9


def test_polyline_perpendicular():
    poly = np.array([[0.0, 0.0], [100.0, 0.0]])
    d = point_to_polyline_m(np.array([50.0, 7.0]), poly)
    assert abs(d - 7.0) < 1e-9, d


def test_polyline_clamps_past_the_end():
    """Beyond the segment the answer is the endpoint, not the infinite line.

    This is the whole reason t is clipped to [0, 1]: without it a point far
    off the end of a road would report a tiny perpendicular distance and the
    join would match it to the wrong edge.
    """
    poly = np.array([[0.0, 0.0], [100.0, 0.0]])
    d = point_to_polyline_m(np.array([130.0, 40.0]), poly)
    assert abs(d - 50.0) < 1e-9, d          # hypot(30, 40)


def test_polyline_zero_length_segment():
    """Duplicate consecutive points must not divide by zero."""
    poly = np.array([[0.0, 0.0], [0.0, 0.0], [10.0, 0.0]])
    d = point_to_polyline_m(np.array([0.0, 6.0]), poly)
    assert abs(d - 6.0) < 1e-9, d


def test_polyline_picks_nearest_segment():
    poly = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 100.0]])
    d = point_to_polyline_m(np.array([97.0, 60.0]), poly)
    assert abs(d - 3.0) < 1e-9, d


# --------------------------------------------------------------------------
# _osmids
# --------------------------------------------------------------------------

def test_osmids_missing():
    assert _osmids({}) == []
    assert _osmids({"osmid": None}) == []


def test_osmids_int():
    assert _osmids({"osmid": 12345}) == [12345]


def test_osmids_list():
    assert _osmids({"osmid": [1, 2, 3]}) == [1, 2, 3]


def test_osmids_stringified_list():
    """OSMnx round-trips through GraphML as a string, not a list."""
    assert _osmids({"osmid": "[123, 456]"}) == [123, 456]


def test_osmids_nested_and_mixed():
    assert _osmids({"osmid": [1, "[2, 3]", [4]]}) == [1, 2, 3, 4]


def test_osmids_bare_string():
    assert _osmids({"osmid": "98765"}) == [98765]


# --------------------------------------------------------------------------
# SumoNet parsing
# --------------------------------------------------------------------------

def _write(tmp: Path, name: str, body: str) -> Path:
    path = tmp / name
    path.write_text(body, encoding="utf-8")
    return path


def test_parser_skips_internal_and_shapeless(tmp: Path):
    net = SumoNet(_write(tmp, "a.net.xml", NET_XML))
    assert net.edge_ids == ["100#0", "200#0", "300#0", "900#0"], net.edge_ids
    assert ":n1_0" not in net.index
    assert "777#0" not in net.index          # function="internal", not ':'-named
    assert "888#0" not in net.index          # no lane shape
    assert net.skipped_no_shape == 1, net.skipped_no_shape


def test_parser_collects_origids(tmp: Path):
    """origId lives on the lane, and one value can carry several way IDs."""
    net = SumoNet(_write(tmp, "b.net.xml", NET_XML))
    assert net.origids[net.index["100#0"]] == {"100"}
    assert net.origids[net.index["200#0"]] == {"200", "201"}


def test_parser_length_is_max_over_lanes(tmp: Path):
    net = SumoNet(_write(tmp, "c.net.xml", NET_XML))
    assert abs(net.lengths[net.index["100#0"]] - 104.0) < 1e-9


def test_parser_shape_is_first_lane_with_geometry(tmp: Path):
    net = SumoNet(_write(tmp, "d.net.xml", NET_XML))
    local = net.edge_local("100#0")
    assert len(local) == 2, local
    # first lane runs y=0.00, the second y=3.20; the first one wins
    assert abs(float(np.hypot(*(local[1] - local[0]))) - 100.0) < 0.5


def test_projection_lands_in_the_configured_bbox(tmp: Path):
    """UTM -> lon/lat has to put this network in Bangalore, not the ocean."""
    from dipteran.config import load_config

    net = SumoNet(_write(tmp, "e.net.xml", NET_XML))
    bbox = load_config().route.bbox
    for lon, lat in net.lonlat:
        assert bbox.west <= lon <= bbox.east, (lon, lat)
        assert bbox.south <= lat <= bbox.north, (lon, lat)


def test_local_frame_preserves_distance(tmp: Path):
    """Projected then re-localised, a 100 m edge is still 100 m."""
    net = SumoNet(_write(tmp, "f.net.xml", NET_XML))
    for edge_id in ("100#0", "200#0", "300#0"):
        local = net.edge_local(edge_id)
        span = float(np.hypot(*(local[-1] - local[0])))
        assert abs(span - 100.0) < 0.5, (edge_id, span)


def test_local_frame_scale_constants(tmp: Path):
    """to_local is a flat-earth scaling; check both axes against the constants."""
    net = SumoNet(_write(tmp, "g.net.xml", NET_XML))
    base = np.array([[net.lon0, net.lat0]])
    assert np.allclose(net.to_local(base)[0], [0.0, 0.0], atol=1e-6)

    north = net.to_local(np.array([[net.lon0, net.lat0 + 1.0]]))[0]
    assert abs(north[1] - DEG_LAT_M) < 1e-6, north

    east = net.to_local(np.array([[net.lon0 + 1.0, net.lat0]]))[0]
    expected = 111_320.0 * math.cos(math.radians(net.lat0))
    assert abs(east[0] - expected) < 1.0, (east, expected)


def test_nearest_edge(tmp: Path):
    net = SumoNet(_write(tmp, "h.net.xml", NET_XML))
    # midpoint of 300#0, pushed 2 m sideways
    mid = net.edge_local("300#0").mean(axis=0) + np.array([0.0, 2.0])
    lon = mid[0] / net._kx + net.lon0
    lat = mid[1] / DEG_LAT_M + net.lat0
    edge_id, gap = net.nearest_edge(lon, lat)
    assert edge_id == "300#0", edge_id
    assert gap < 3.0, gap


def test_nearest_edge_finds_the_island(tmp: Path):
    """The KD-tree must not be biased toward the dense chain."""
    net = SumoNet(_write(tmp, "i.net.xml", NET_XML))
    mid = net.edge_local("900#0").mean(axis=0)
    lon = mid[0] / net._kx + net.lon0
    lat = mid[1] / DEG_LAT_M + net.lat0
    edge_id, gap = net.nearest_edge(lon, lat)
    assert edge_id == "900#0", edge_id
    assert gap < 1.0, gap


# --------------------------------------------------------------------------
# _way_index strategy selection
# --------------------------------------------------------------------------

def test_way_index_prefers_origid(tmp: Path):
    net = SumoNet(_write(tmp, "j.net.xml", NET_XML))
    index, strategy = _way_index(net)
    assert strategy == "origId", strategy
    assert index["200"] == ["200#0"], index["200"]
    assert index["201"] == ["200#0"], index["201"]     # one lane, two source ways


def test_way_index_falls_back_to_edge_id(tmp: Path):
    net = SumoNet(_write(tmp, "k.net.xml", NET_XML_NO_ORIGID))
    index, strategy = _way_index(net)
    assert strategy == "edge_id_parse", strategy
    # the forward and reversed halves of way 100 both land under "100"
    assert sorted(index["100"]) == ["-100#1", "100#0"], index["100"]


def test_way_index_gives_up(tmp: Path):
    net = SumoNet(_write(tmp, "l.net.xml", NET_XML_UNPARSEABLE))
    index, strategy = _way_index(net)
    assert strategy == "none", strategy
    assert index == {}, index


# --------------------------------------------------------------------------
# reachability BFS
# --------------------------------------------------------------------------

def test_reachable_requires_connections(tmp: Path):
    net = SumoNet(_write(tmp, "m.net.xml", NET_XML))
    try:
        net.reachable("100#0", "300#0")
    except RuntimeError as exc:
        assert "with_connections" in str(exc), exc
    else:
        raise AssertionError("reachable() worked without connections loaded")


def test_reachable_follows_the_chain(tmp: Path):
    net = SumoNet(_write(tmp, "n.net.xml", NET_XML), with_connections=True)
    ok, explored = net.reachable("100#0", "300#0")
    assert ok
    assert explored >= 3, explored


def test_reachable_is_directional(tmp: Path):
    """Connections are one-way; the BFS must not walk them backwards."""
    net = SumoNet(_write(tmp, "o.net.xml", NET_XML), with_connections=True)
    ok, _ = net.reachable("300#0", "100#0")
    assert not ok


def test_unreachable_island(tmp: Path):
    """The 239-component finding means this case is real, not hypothetical."""
    net = SumoNet(_write(tmp, "p.net.xml", NET_XML), with_connections=True)
    ok, explored = net.reachable("100#0", "900#0")
    assert not ok
    assert explored >= 1, explored


def test_reachable_self(tmp: Path):
    net = SumoNet(_write(tmp, "q.net.xml", NET_XML), with_connections=True)
    ok, _ = net.reachable("900#0", "900#0")
    assert ok


def test_connections_drop_internal_edges(tmp: Path):
    """`:n1_0 -> 200#0` refers to an edge that was never indexed."""
    net = SumoNet(_write(tmp, "r.net.xml", NET_XML), with_connections=True)
    assert int(net._conn_offsets[-1]) == 2, net._conn_offsets[-1]
    succ = net.successors(net.index["100#0"])
    assert [net.edge_ids[i] for i in succ] == ["200#0"]


# --------------------------------------------------------------------------
# world geometry
# --------------------------------------------------------------------------

def test_wrap180():
    """Half-open at the bottom: 180 folds to -180, same direction."""
    assert wrap180(0.0) == 0.0
    assert wrap180(180.0) == -180.0
    assert wrap180(181.0) == -179.0
    assert wrap180(-181.0) == 179.0
    assert wrap180(360.0) == 0.0
    assert abs(wrap180(719.0) - (-1.0)) < 1e-9
    for deg in (-1000.0, -181.0, -0.5, 0.0, 45.0, 180.0, 359.0, 1000.0):
        assert -180.0 <= wrap180(deg) < 180.0, deg


def test_haversine_matches_the_measured_route():
    """PHASE1_REPORT.md measured the A->B straight line at 10.59 km."""
    from dipteran.config import load_config

    r = load_config().route
    d = haversine_m(r.origin.lat, r.origin.lon,
                    r.destination.lat, r.destination.lon)
    assert abs(d - 10_590.0) < 120.0, d


def test_haversine_zero_and_symmetric():
    assert haversine_m(12.9, 77.6, 12.9, 77.6) == 0.0
    a = haversine_m(12.9, 77.6, 13.0, 77.7)
    b = haversine_m(13.0, 77.7, 12.9, 77.6)
    assert abs(a - b) < 1e-6


def test_bearing_cardinals():
    assert abs(wrap180(bearing_deg(0.0, 0.0, 1.0, 0.0) - 0.0)) < 1e-6    # N
    assert abs(bearing_deg(0.0, 0.0, 0.0, 1.0) - 90.0) < 1e-6            # E
    assert abs(bearing_deg(0.0, 0.0, -1.0, 0.0) - 180.0) < 1e-6          # S
    assert abs(bearing_deg(0.0, 0.0, 0.0, -1.0) - 270.0) < 1e-6          # W


def test_bearing_to_destination_is_north_east():
    """Tin Factory is north and east of Silk Board, so the bearing is in Q1."""
    from dipteran.config import load_config

    r = load_config().route
    b = bearing_deg(r.origin.lat, r.origin.lon,
                    r.destination.lat, r.destination.lon)
    assert 0.0 < b < 90.0, b


def test_looming_grows_as_it_closes():
    """Closing is positive expansion; receding reads as zero, not negative."""
    closing = looming_rate(20.0, 10.0, dt=0.1)
    assert closing > 0.0
    assert looming_rate(10.0, 20.0, dt=0.1) == 0.0


def test_looming_is_about_rate_not_proximity():
    """A car sitting still nearby must not loom; one closing fast must."""
    parked = looming_rate(3.0, 3.0, dt=0.1)
    rushing = looming_rate(30.0, 22.0, dt=0.1)
    assert parked == 0.0
    assert rushing > parked


def test_looming_accelerates_when_nearer():
    """Same closing distance, nearer in: bigger angular expansion."""
    far = looming_rate(50.0, 45.0, dt=0.1)
    near = looming_rate(10.0, 5.0, dt=0.1)
    assert near > far


def test_looming_guards_degenerate_input():
    assert looming_rate(10.0, 5.0, dt=0.0) == 0.0
    assert looming_rate(0.0, 5.0, dt=0.1) == 0.0
    assert looming_rate(10.0, 0.0, dt=0.1) == 0.0


def test_pothole_is_deterministic():
    """The fly cannot learn a hazard that moves between runs."""
    sample = [f"edge_{i}#0" for i in range(2000)]
    first = [is_pothole(e) for e in sample]
    assert first == [is_pothole(e) for e in sample]


def test_pothole_density_is_roughly_honoured():
    sample = [f"edge_{i}#0" for i in range(5000)]
    hit = sum(is_pothole(e, 0.04) for e in sample)
    assert 0.02 < hit / len(sample) < 0.06, hit


def test_pothole_edges():
    assert is_pothole("", 0.5) is False        # no edge, no hazard
    assert is_pothole("any#0", 0.0) is False   # density 0 disables it
    assert all(is_pothole(f"e{i}", 1.0) for i in range(50))


# --------------------------------------------------------------------------
# state classification and telemetry
# --------------------------------------------------------------------------

def test_panic_overrides_everything():
    out = MotorOutput(panic=0.4, throttle=1.0, reward=1.0)
    title, _ = classify_state(out, speed_ms=0.0, throttle_cmd=1.0,
                              pothole=True, horn=1.0)
    assert title == "PANIC SWERVE", title


def test_gridlock_needs_throttle_and_no_motion():
    out = MotorOutput(throttle=0.9)
    stuck, _ = classify_state(out, speed_ms=0.0, throttle_cmd=0.9)
    moving, _ = classify_state(out, speed_ms=9.0, throttle_cmd=0.9)
    assert stuck == "GRIDLOCK STAGNATION", stuck
    assert moving != "GRIDLOCK STAGNATION", moving


def test_pothole_and_horn_rank_under_panic_but_over_gridlock():
    out = MotorOutput(throttle=0.9)
    pot, _ = classify_state(out, speed_ms=0.0, throttle_cmd=0.9, pothole=True)
    horn, _ = classify_state(out, speed_ms=0.0, throttle_cmd=0.9, horn=0.9)
    assert pot == "POTHOLE IMPACT", pot
    assert horn == "HORN OVERDRIVE", horn


def test_idle_drift_is_the_fallback():
    title, caption = classify_state(MotorOutput(), speed_ms=0.0,
                                    throttle_cmd=0.0)
    assert title == "IDLE DRIFT", title
    assert caption


def test_every_state_has_a_caption_and_no_emoji():
    """Emoji-free HUD copy was settled twice over; keep it enforced."""
    from dipteran.navigator import STATES

    for title, caption in STATES:
        assert title and caption
        assert all(ord(ch) < 0x2190 for ch in title + caption), title


def test_telemetry_keeps_the_mock_contract():
    """index.html's existing bindings must not break."""
    out = MotorOutput(steer=0.2, throttle=0.5, reward=0.3, punishment=0.1)
    payload = telemetry(out, lat=12.9, lon=77.6, heading=45.0, speed_ms=4.0,
                        throttle_cmd=0.5)
    for key in ("lat", "lon", "heading", "sugar_reward", "pain_level",
                "giant_fiber", "event"):
        assert key in payload, key
    for key in ("state", "caption", "steer", "throttle", "dopamine",
                "regions", "nearby_agents"):
        assert key in payload, key


def test_telemetry_is_json_serialisable():
    """It goes down a WebSocket, so numpy/torch scalars would break it."""
    import json

    out = MotorOutput(steer=0.2, regions={"compass_epg": 0.3})
    payload = telemetry(out, lat=12.9, lon=77.6, heading=45.0, speed_ms=4.0,
                        throttle_cmd=0.5, nearby_agents=[{"id": "bg_1"}])
    assert json.loads(json.dumps(payload))["regions"]["compass_epg"] == 0.3


def test_panic_zeroes_throttle_in_the_decode_contract():
    """MotorOutput is built by _decode, but the override is worth pinning."""
    assert MotorOutput(panic=0.5, throttle=0.0).throttle == 0.0


# --------------------------------------------------------------------------
# the firing-rate / command boundary
# --------------------------------------------------------------------------
#
# Both of the bugs a long route run exposed live here. `MotorOutput.throttle`
# is a DNg02 population *firing rate*, measured at 0.05-0.24; the motion model
# and every threshold in the world want a *command* in [0, 1]. Mixing them
# silently disabled gridlock detection (a 0.24-max rate can never clear a 0.3
# threshold) and pinned the vehicle at 5% throttle. These lock the contract.

def test_scaled_command_reaches_the_gridlock_threshold():
    """The whole point of the scale: 0.3 has to be reachable."""
    from dipteran.config import load_config

    v = load_config().vehicle
    cruise_rate = 0.24          # measured top of the DNg02 band
    assert cruise_rate <= 0.3, "raw rate should NOT clear the threshold"
    assert min(1.0, cruise_rate * v.throttle_scale) > 0.3, (
        "scaled command must clear the gridlock threshold, or stagnation "
        "detection is dead code again"
    )


def test_command_scales_stay_inside_the_unit_interval():
    """Scaling must saturate, never overflow the [0, 1] contract."""
    from dipteran.config import load_config

    v = load_config().vehicle
    for rate in (0.0, 0.05, 0.24, 1.0, 50.0):
        assert 0.0 <= min(1.0, max(0.0, rate * v.throttle_scale)) <= 1.0
    for rate in (-50.0, -0.4, 0.0, 0.4, 50.0):
        assert -1.0 <= max(-1.0, min(1.0, rate * v.steer_scale)) <= 1.0


def test_vehicle_state_carries_the_command_not_the_rate():
    """The field exists so the two quantities cannot be confused again."""
    from dipteran.world import VehicleState

    s = VehicleState(lat=12.9, lon=77.6)
    assert hasattr(s, "throttle_cmd")
    assert s.throttle_cmd == 0.0


def test_sense_takes_no_throttle_argument():
    """Regression guard: the command is read off state, never passed in.

    Passing it is what allowed a firing rate to be handed to a threshold
    written for a command.
    """
    import inspect

    from dipteran.world import World

    params = list(inspect.signature(World.sense).parameters)
    assert params == ["self", "dt"], params


def test_reinforcement_is_scaled_by_the_anatomy():
    """An absolute update is 500x the weight it modifies; a relative one is not.

    Reproduces the arithmetic that flatlined a 55-minute route run, without
    needing a GPU or the corpus: a row-normalised synapse is ~1e-4, the
    eligibility trace saturates at 1/(1 - trace_decay), and the absolute form
    swamps the anatomy on the very first tick.
    """
    from dipteran.config import load_config

    snn = load_config().snn
    weight = 1e-4                                   # typical row-normalised
    trace_max = 1.0 / (1.0 - snn.trace_decay)       # 5.0 at decay 0.8

    # The shipped bug, as a literal: an *absolute* update at the old rate.
    # Kept as 0.01 rather than read from config precisely because config no
    # longer holds it -- this line is the regression it must never return to.
    absolute = 0.01 * 1.0 * trace_max
    relative = snn.learning_rate * 1.0 * trace_max * weight

    assert absolute > weight * 100, "the absolute form really was that big"
    assert relative < weight * 0.01, (
        f"one update moves the synapse by {relative / weight:.1%} of itself; "
        "it must stay small enough that adaptation takes hundreds of ticks"
    )
    # Saturating the permitted band should take hundreds of ticks, not ten.
    ticks = snn.plastic_span / (relative / weight)
    assert ticks > 500, f"band saturates in {ticks:.0f} ticks"


def test_every_cluster_has_a_hud_region():
    """A cluster absent from CLUSTER_REGION is silently dropped by the HUD.

    Telemetry is keyed by cluster name and the page groups those into
    anatomical regions via `cluster_region`. An unmapped cluster produces no
    error anywhere -- the ETL prints a warning nobody reads, and the region
    simply never lights up. Adding `mechano_bristle` in Phase 2 took the count
    from 14 to 15, so this is a live failure mode, not a hypothetical.
    """
    import sys as _sys

    _sys.path.insert(0, str(REPO / "scripts"))
    from extract_neuropils import CLUSTER_REGION, REGIONS

    from dipteran.connectome import CLUSTERS

    assert set(CLUSTERS) == set(CLUSTER_REGION), (
        f"unmapped: {sorted(set(CLUSTERS) - set(CLUSTER_REGION))}, "
        f"stale: {sorted(set(CLUSTER_REGION) - set(CLUSTERS))}"
    )
    region_ids = {r[0] for r in REGIONS}
    assert not set(CLUSTER_REGION.values()) - region_ids


def test_plastic_bounds_survive_inhibitory_weights():
    """6.07M of the 15.1M pairs are inhibitory, so the band must be signed.

    A `clamp(min=w*0.2, max=w*2.5)` form inverts for negative w -- min ends up
    above max and torch silently returns the max. The `w +/- |w| * span` form
    brackets correctly for either sign.
    """
    from dipteran.config import load_config

    span = load_config().snn.plastic_span
    for w in (1e-4, -1e-4, 0.0):
        lo, hi = w - abs(w) * span, w + abs(w) * span
        assert lo <= hi, (w, lo, hi)
        assert lo <= w <= hi
        # never flips sign
        if w:
            assert lo * w >= 0 and hi * w >= 0, (w, lo, hi)


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# Phase 3: the signed steering drive, and reward shaping
# --------------------------------------------------------------------------

def test_steering_drive_sign_goal_to_the_right():
    """Goal to the right must drive PFL3-LEFT. The crossing is the point.

    This exact formula shipped backwards twice during design review -- once
    with a worked example that contradicted its own arithmetic -- so the sign
    is asserted rather than inspected.

    `error = bearing_to_goal - heading`, positive means the goal is clockwise
    of the nose, i.e. to the right. PFL3 projects *contralaterally* onto
    DNa02 (measured: 12 synapses PFL3-right -> DNa02-left, zero
    PFL3-right -> DNa02-right), so raising DNa02-right -- which is what
    `steer = right - left` needs for a positive, rightward steer -- requires
    driving PFL3-left.
    """
    # cos(90 degrees) is 6.1e-17 rather than 0 in binary floating point, so
    # the off lobe is asserted against a tolerance. The sign is what matters
    # and it is exact: one lobe at full drive, the other at zero.
    left, right = steering_drive(45.0)
    assert math.isclose(left, 1.0, rel_tol=1e-9), left
    assert math.isclose(right, 0.0, abs_tol=1e-12), right
    # ... and the mirror, so a sign flip in one lobe alone cannot pass.
    left, right = steering_drive(-45.0)
    assert math.isclose(left, 0.0, abs_tol=1e-12), left
    assert math.isclose(right, 1.0, rel_tol=1e-9), right


def test_steering_drive_is_balanced_dead_ahead():
    """Facing the goal produces equal lobes and therefore no net turn."""
    left, right = steering_drive(0.0)
    assert math.isclose(left, right), (left, right)
    assert math.isclose(left, math.cos(math.radians(45.0)), rel_tol=1e-9)


def test_steering_drive_is_antisymmetric():
    """Swapping the sign of the error swaps the two lobes, at every angle."""
    for error in range(-180, 181, 5):
        left, right = steering_drive(float(error))
        mirror_left, mirror_right = steering_drive(float(-error))
        assert math.isclose(left, mirror_right, abs_tol=1e-12), error
        assert math.isclose(right, mirror_left, abs_tol=1e-12), error


def test_steering_drive_dead_zone_behind():
    """Both lobes clamp to zero past +/-135 degrees, and nothing goes negative.

    Documented consequence of two cosine lobes at +/-45: a goal directly
    behind the fly produces no steering drive at all. `max(0, ...)` is what
    keeps the dead zone a *zero* rather than a negative current, which would
    otherwise be injected as though it were a stimulus.
    """
    for error in (136.0, 150.0, 180.0, -180.0, -150.0, -136.0):
        left, right = steering_drive(error)
        assert left == 0.0 and right == 0.0, (error, left, right)
    for error in range(-180, 181):
        left, right = steering_drive(float(error))
        assert left >= 0.0 and right >= 0.0, error


def test_steering_drive_scaled_stays_below_threshold():
    """The injected value never reaches v_thresh, at any heading error.

    A lobe peaks at exactly 1.0, and `calibrate_gain`'s docstring is explicit
    that a sensory current at threshold makes those rows fire every sub-tick
    -- "not a stimulus, it is a short circuit".
    """
    assert 0.0 < SENSORY_CEILING < 1.0, SENSORY_CEILING
    for error in range(-180, 181):
        left, right = steering_drive(float(error))
        assert 0.0 <= left * SENSORY_CEILING <= SENSORY_CEILING, error
        assert 0.0 <= right * SENSORY_CEILING <= SENSORY_CEILING, error


def test_steering_scaling_preserves_contrast():
    """Scaling must not flatten the channel the way clipping does.

    `min(SENSORY_CEILING, lobe)` is the obvious way to respect the ceiling and
    it silently destroys half the steering range: dead ahead the lobes are
    0.707/0.707 and both clip to 0.4, while a goal hard right is 1.0/0.0 and
    clips to 0.4/0.0 -- so the left lobe reads exactly 0.4 in both cases and
    can no longer distinguish them. This asserts the injected contrast really
    does change between those two headings.
    """
    ahead = [v * SENSORY_CEILING for v in steering_drive(0.0)]
    hard_right = [v * SENSORY_CEILING for v in steering_drive(45.0)]
    assert ahead[0] != hard_right[0], (ahead, hard_right)
    assert math.isclose(ahead[0] - ahead[1], 0.0, abs_tol=1e-12)
    assert hard_right[0] - hard_right[1] > 0.3, hard_right

    clipped_ahead = min(SENSORY_CEILING, steering_drive(0.0)[0])
    clipped_right = min(SENSORY_CEILING, steering_drive(45.0)[0])
    assert clipped_ahead == clipped_right, (
        "if this ever differs, clipping became safe and this test is stale")


def test_proximity_scale_hits_both_anchors():
    """500 m -> 100%, 15 km -> 2%. The two numbers actually asked for."""
    assert math.isclose(proximity_scale(500.0), 1.0, rel_tol=1e-9)
    assert math.isclose(proximity_scale(15000.0), 0.02, rel_tol=1e-6)


def test_proximity_scale_clamps_outside_the_anchors():
    """Arriving earns no more than 100%, and 40 km earns no less than 2%."""
    for near in (0.0, 1.0, 100.0, 499.0):
        assert proximity_scale(near) == 1.0, near
    for far in (15001.0, 40000.0, 1e9):
        assert proximity_scale(far) == 0.02, far


def test_proximity_scale_is_monotonic():
    """Closer is always worth at least as much -- never a local dip."""
    previous = 0.0
    for distance in range(20000, 0, -250):
        value = proximity_scale(float(distance))
        assert value >= previous - 1e-12, distance
        previous = value


def test_proximity_scale_is_well_below_full_at_route_start():
    """The 10.6 km start of the real route must not read as 'nearly arrived'.

    Guards the failure this scaling exists to prevent: a curve flat enough to
    pay out near-full dopamine at the start teaches nothing about approaching
    the goal.
    """
    at_start = proximity_scale(10587.5)      # cache/route_run.json
    assert 0.02 < at_start < 0.10, at_start
    assert at_start < proximity_scale(2000.0) < proximity_scale(600.0)


def _toy_navigator():
    """A 6-neuron navigator, so the loop's logic is testable without the corpus.

    Same spirit as the synthetic `.net.xml` the SUMO cases use: these tests
    have to run on a fresh clone with nothing downloaded, and the behaviour
    under test -- what `reset` clears, how a frame's ticks combine -- does not
    depend on having 139k real neurons.
    """
    import numpy as _np
    import torch

    from dipteran.navigator import FlyBangaloreNavigator

    n = 6
    dense = torch.zeros((n, n))
    for i in range(n):
        dense[i, (i + 1) % n] = 3.0
        dense[i, (i + 2) % n] = -1.0
    w = dense.to_sparse_csr()
    neurons = _np.arange(n, dtype=_np.int64)
    clusters = {"clusters": {
        "sense_a": {"role": "sensory", "rows": [0, 1]},
        "steer_left": {"role": "motor", "rows": [2]},
        "steer_right": {"role": "motor", "rows": [3]},
        "throttle": {"role": "motor", "rows": [4]},
    }}
    return FlyBangaloreNavigator(w, neurons, clusters, gain=4.0,
                                 learning_rate=0.1, steer_bias=0.0)


def test_reset_clears_state_but_keeps_learned_weights():
    """`reset()` must not touch `p_val`. It is the only thing episodes carry.

    `World.reset_episode` calls this between attempts at the route. Membrane
    voltage, spikes and the eligibility trace are all within-episode state and
    have to go. The plastic overlay is the entire point of running more than
    one episode, so zeroing it would silently make training a no-op -- and it
    would look like "learning just doesn't work" rather than like a bug.
    """
    nav = _toy_navigator()
    nav.set_sensory({"sense_a": 0.9})
    for _ in range(5):
        nav.step()
    nav._reinforce(1.0)          # move p_val away from the anatomy

    drifted = nav.p_val.clone()
    assert not bool((drifted == nav.p_val0).all()), "test needs p_val to move"

    nav.reset()
    assert float(nav.v.abs().sum()) == 0.0
    assert float(nav.spikes.abs().sum()) == 0.0
    assert float(nav.trace.abs().sum()) == 0.0
    assert bool((nav.p_val == drifted).all()), "reset() clobbered learning"


def test_motor_output_average_recovers_resolution():
    """Averaging a frame's ticks is what makes a 1-neuron readout usable.

    DNa02 is a single bilateral pair, so one tick's rate quantises to
    multiples of 1/subticks = 0.1. Subtract a resting bias of 0.191 and no
    reachable value is near zero -- the vehicle cannot hold a straight line.
    Averaging the frame's ticks samples subticks*ticks sub-steps instead.
    """
    from dipteran.navigator import MotorOutput

    ticks = [MotorOutput(steer=s, throttle=0.5, dopamine=0.2)
             for s in (0.1, 0.0, 0.1, 0.0, 0.1)]
    averaged = MotorOutput.average(ticks)
    assert math.isclose(averaged.steer, 0.06), averaged.steer
    assert averaged.steer not in {t.steer for t in ticks}, (
        "averaging must reach values no single tick can express")
    assert math.isclose(averaged.throttle, 0.5)


def test_motor_output_average_keeps_panic_as_a_maximum():
    """One giant-fiber spike in a frame is an escape, not a fifth of one."""
    from dipteran.navigator import MotorOutput

    ticks = [MotorOutput(panic=0.0) for _ in range(4)] + [MotorOutput(panic=0.8)]
    assert MotorOutput.average(ticks).panic == 0.8
    assert MotorOutput.average([]).panic == 0.0


# --------------------------------------------------------------------------
# the route picker
#
# index.html's entry flow is two map clicks and a button, and the button is
# a POST /route. These cover the parts of it that can go quietly wrong: a
# point outside the OSM extract has no road to snap to and must be refused
# before SUMO is asked to spawn a vehicle there, and the mock streamer has
# to honour the picked pair or the picker looks broken to anyone trying it
# without the live backend.
# --------------------------------------------------------------------------

def test_route_rejects_a_point_outside_the_extract():
    from fastapi import HTTPException

    import main as server

    r = server._route_config()
    outside = server.Endpoint(lat=r.bbox.north + 1.0, lon=r.origin.lon)
    inside = server.Endpoint(lat=r.destination.lat, lon=r.destination.lon)

    for bad in (server.RouteRequest(origin=outside, destination=inside),
                server.RouteRequest(origin=inside, destination=outside)):
        try:
            server.post_route(bad)
        except HTTPException as exc:
            assert exc.status_code == 422, exc.status_code
        else:
            raise AssertionError("a point outside route.bbox was accepted")


def test_route_accepts_the_configured_default():
    import main as server

    r = server._route_config()
    server.SIM.started = False                  # mock mode never sets it
    reply = server.post_route(server.RouteRequest(
        origin=server.Endpoint(lat=r.origin.lat, lon=r.origin.lon),
        destination=server.Endpoint(lat=r.destination.lat,
                                    lon=r.destination.lon)))
    assert reply["ok"] is True
    assert server.SIM.origin == (r.origin.lat, r.origin.lon)
    assert server.SIM.goal == (r.destination.lat, r.destination.lon)


def test_mock_stream_starts_where_the_user_pointed():
    """And faces the goal, so the first frame already reads as a departure."""
    import main as server
    from dipteran.world import bearing_deg, wrap180

    origin, goal = (12.95, 77.60), (13.00, 77.66)
    frame = next(server.mock_frames(origin, goal))

    # One 20 Hz step of drift has already been applied, so this is "near",
    # not "equal" -- the point is that it is not Silk Board.
    assert abs(frame["lat"] - origin[0]) < 1e-3, frame["lat"]
    assert abs(frame["lon"] - origin[1]) < 1e-3, frame["lon"]
    assert abs(wrap180(frame["heading"] - bearing_deg(*origin, *goal))) < 20.0


def test_world_takes_an_origin_and_goal_override():
    """The picker's pair has to reach World without going through config."""
    import inspect

    from dipteran.world import World

    params = inspect.signature(World.__init__).parameters
    for name in ("origin", "goal"):
        assert name in params, f"World.__init__ lost its {name} override"
        assert params[name].default is None, (
            f"World.__init__ {name} must default to None so config.yaml "
            f"stays the fallback")


def main() -> int:
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]

    passed, failed = 0, []
    with tempfile.TemporaryDirectory(prefix="dipteran-tests-") as raw:
        tmp = Path(raw)
        for name, fn in tests:
            try:
                if fn.__code__.co_argcount:
                    fn(tmp)
                else:
                    fn()
            except Exception:
                failed.append(name)
                print(f"FAIL {name}", flush=True)
                traceback.print_exc()
            else:
                passed += 1

    print(f"\n{passed}/{len(tests)} passed", flush=True)
    if failed:
        print("failed: " + ", ".join(failed), flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
