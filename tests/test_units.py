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
    DNa02 (measured: 238 synapses from the 12 PFL3-right cells -> DNa02-left, zero
    PFL3-right -> DNa02-right), so raising DNa02-right -- which is what
    `steer = right - left` needs for a positive, rightward steer -- requires
    driving PFL3-left.
    """
    # cos(90 degrees) is 6.1e-17 rather than 0 in binary floating point, so
    # the off lobe is asserted against a tolerance. The sign is what matters
    # and it is exact: one lobe at full drive, the other at zero.
    # Asserted at the lobe width this formula was designed around, so the
    # numbers stay the ones the docstring works through, and again at the
    # configured default, so the crossing is checked at both widths.
    left, right = steering_drive(45.0, 45.0)
    assert math.isclose(left, 1.0, rel_tol=1e-9), left
    assert math.isclose(right, 0.0, abs_tol=1e-12), right
    # ... and the mirror, so a sign flip in one lobe alone cannot pass.
    left, right = steering_drive(-45.0, 45.0)
    assert math.isclose(left, 0.0, abs_tol=1e-12), left
    assert math.isclose(right, 1.0, rel_tol=1e-9), right

    wide_left, wide_right = steering_drive(45.0)
    assert wide_left > wide_right == 0.0, (wide_left, wide_right)
    wide_left, wide_right = steering_drive(-45.0)
    assert wide_right > wide_left == 0.0, (wide_left, wide_right)


def test_steering_drive_is_balanced_dead_ahead():
    """Facing the goal produces equal lobes and therefore no net turn."""
    left, right = steering_drive(0.0)
    assert math.isclose(left, right), (left, right)
    left45, right45 = steering_drive(0.0, 45.0)
    assert math.isclose(left45, math.cos(math.radians(45.0)), rel_tol=1e-9)
    assert math.isclose(right45, left45, rel_tol=1e-9)


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
        left, right = steering_drive(error, 45.0)
        assert left == 0.0 and right == 0.0, (error, left, right)


def test_steering_drive_default_width_still_steers_past_135():
    """The 70-degree default keeps driving where the 45 pair gives up.

    A fly pointing away from its route used to get no directional drive at
    all past +/-135 and fell back to a constant -- measured live, a median
    heading error of 122 degrees with the command positive on 95% of frames.
    """
    for error in (136.0, 150.0, 159.0):
        left, right = steering_drive(error)
        assert left > 0.0 and right == 0.0, (error, left, right)
        mirror_left, mirror_right = steering_drive(-error)
        assert mirror_right > 0.0 and mirror_left == 0.0, error
        assert steering_drive(error, 45.0) == (0.0, 0.0), error


def test_steering_drive_default_width_keeps_the_overlap():
    """Dead ahead must drive *both* lobes, or left turns stop existing.

    DNa02-left rests silent while DNa02-right runs at ~101 Hz, so a left
    turn is DNa02-right pushed below its balanced baseline. Take the overlap
    away -- a 90-degree half angle, where dead ahead is cos(90) = 0 on both
    sides -- and there is no baseline to push below: measured at 90, the
    command came out positive on 100% of frames, a fly that could only turn
    right.
    """
    left, right = steering_drive(0.0)
    assert math.isclose(left, right, rel_tol=1e-9)
    assert left > 0.3, left
    assert math.isclose(steering_drive(0.0, 90.0)[0], 0.0, abs_tol=1e-12)
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
    # Stated at the 45-degree lobe width, because the failure it documents
    # is specific to that geometry: it is the width whose dead-ahead lobes
    # (0.707/0.707) and hard-right lobes (1.0/0.0) both clip to the same
    # 0.4. The wider default never puts two different headings on the same
    # side of the ceiling, so it cannot show the bug at all.
    ahead = [v * SENSORY_CEILING for v in steering_drive(0.0, 45.0)]
    hard_right = [v * SENSORY_CEILING for v in steering_drive(45.0, 45.0)]
    assert ahead[0] != hard_right[0], (ahead, hard_right)
    assert math.isclose(ahead[0] - ahead[1], 0.0, abs_tol=1e-12)
    assert hard_right[0] - hard_right[1] > 0.3, hard_right

    clipped_ahead = min(SENSORY_CEILING, steering_drive(0.0, 45.0)[0])
    clipped_right = min(SENSORY_CEILING, steering_drive(45.0, 45.0)[0])
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


def _route_distance_fixture(route_len, edge_pos, crow_flies, last_route,
                            edge="-123624220#4"):
    """A World with only the fields `_route_distance_now` reads.

    Built with `__new__` rather than `World(...)`: the real constructor
    launches a SUMO subprocess and parses an 862 MB network, and none of
    that has any bearing on the arithmetic under test here.

    `route_len` is measured from the start of `edge`, as `findRoute`'s is.
    The fly stands on that same edge, which is the ordinary case.
    `test_route_distance_refuses_an_anchor_from_another_edge` is what moves
    the two apart.
    """
    from dipteran.world import VehicleState, World

    world = World.__new__(World)
    world.state = VehicleState(lat=12.92143, lon=77.63576)
    world.state.edge = edge
    world.state.edge_pos = edge_pos
    world.state.distance_to_goal_m = crow_flies
    world._route_length_m = route_len
    world._route_anchor_edge = edge
    world._last_route_m = last_route
    return world


def test_route_distance_holds_its_units_through_a_routing_failure():
    """A routing failure must not silently republish crow-flies metres.

    The regression, measured on the 2026-09-19 twisty run: that route is
    2,398 m along roads to cover 502 m direct, a 4.8x ratio because it is a
    one-way loop system. `findRoute` failed once mid-episode and the
    published route distance became the crow-flies 536 m, which everything
    downstream read as 1,862 m of progress in a single frame. The harness
    duly reported "closed 1862 m of route" for an episode that travelled
    790 m in total.

    Holding the last known route distance is the fix. Going stale is the
    correct failure mode; changing units is not.
    """
    # A live route, 2,398 m, fly at the start of its edge.
    world = _route_distance_fixture(2398.0, 0.0, 502.0, None)
    assert world.route_valid
    assert world._route_distance_now() == 2398.0

    # 100 m along the edge: route distance falls by exactly that.
    world.state.edge_pos = 100.0
    assert world._route_distance_now() == 2298.0

    # findRoute fails. Crow-flies has meanwhile closed to 536 m -- the exact
    # value that produced the 1,862 m artifact.
    world._route_length_m = None
    world.state.distance_to_goal_m = 536.0
    assert not world.route_valid
    assert world._route_distance_now() == 2298.0, (
        "a routing failure republished the crow-flies distance as a route "
        "distance -- this is the 1,862 m bug")

    # Recovery: a fresh, shorter route. `findRoute` measures it from the
    # start of the edge, and the fly is still 100 m along that edge.
    world._route_length_m = 2200.0
    assert world.route_valid
    assert world._route_distance_now() == 2100.0


def test_route_distance_seeds_from_crow_flies_before_any_route():
    """Before the first route resolves there is nothing to hold.

    Crow-flies is a legitimate *starting* value in that window -- what it
    must never be is a mid-episode substitute. `route_valid` is what tells
    the two apart from outside.
    """
    world = _route_distance_fixture(None, 0.0, 502.0, None)
    assert not world.route_valid
    assert world._route_distance_now() == 502.0


def test_route_distance_refuses_an_anchor_from_another_edge():
    """`edge_pos` is a position along *one* edge; the anchor must match it.

    Reachable on every junction crossing, not just on the one-way loop the
    2026-09-22 handoff proposed. `_pull_to_road` sets `state.edge` from raw
    `_edge_at`, which answers with an internal `:cluster_...` edge inside a
    junction -- the same behaviour `_road_edge_near` exists to work around --
    while `_update_route` early-returns on internal edges and so leaves the
    anchor pointing at the edge before the junction.

    Measured on the synthetic repro against the pre-fix code: route 2,000 m,
    anchor 0 on edge A, fly 400 m along A (1,600 m remaining), then one frame
    on a junction edge at `edge_pos` 2.0 published **1,998 m**. A 398 m
    backward step in one 0.1 s frame is 3,980 m/s of apparent wrong-way
    driving, against a detector whose threshold is 0.5 m/s and whose full
    scale is 4.0 m/s -- so it saturates, every junction.

    (The handoff's own proposed mechanism -- a lap revisiting an edge ID --
    does not reach this: `_update_route` reassigns `_route_edge` on every
    non-internal edge, so after a lap the guard compares against the
    *previous* edge and the recompute fires normally.)
    """
    world = _route_distance_fixture(2000.0, 400.0, 480.0, None,
                                    edge="-123624220#4")
    assert world.route_valid
    assert world._route_distance_now() == 1600.0

    # One frame inside the junction. Same route, but `edge_pos` is now
    # measured along a six-metre internal edge.
    world.state.edge = ":cluster_7_0"
    world.state.edge_pos = 2.0
    assert not world.route_valid, (
        "an anchor measured on another edge was reported as a live "
        "measurement -- this is the junction spike")
    assert world._route_distance_now() == 1600.0, (
        "the junction frame republished 1998 m: a 398 m backward jump that "
        "reads as 3980 m/s of wrong-way driving and fires PPL1")

    # Out the far side: a fresh route, measured from the new edge.
    world._route_length_m = 1500.0
    world._route_anchor_edge = "27785733#1"
    world.state.edge = "27785733#1"
    world.state.edge_pos = 0.0
    assert world.route_valid
    assert world._route_distance_now() == 1500.0


def test_reset_episode_seeds_the_ratchet_in_route_metres():
    """Episode 2 onward must start measuring in the same units as episode 1.

    `_spawn_ego` resolves the origin with `_road_edge_near`, which searches
    outward past junctions to a normal edge. `reset_episode` used raw
    `_edge_at`, which at the test routes' origins answers with an internal
    junction edge. `_update_route` early-returns on those, no route resolves,
    and `_route_distance_now` falls back to its crow-flies seed.

    Measured across every multi-episode run on record: episode 1 starts its
    route ratchet at 658.5 m (500 m route) / 2,397.7 m (twisty) -- route
    metres -- and **all 20 later episodes** start at exactly 426.0 / 502.2 m,
    the crow-flies distance. So `rewards()`'s `best_distance_m` was seeded
    4.8x too tight on the twisty route in every episode after the first --
    the failure mode c8fc19c describes and set out to fix, still live through
    the edge lookup that feeds it -- and `closed_route_m` for those episodes
    was a crow-flies start minus a route minimum.
    """
    from types import SimpleNamespace

    from dipteran.world import ROUTE_RETRY_FRAMES, VehicleState, World

    origin, goal = (12.92143, 77.63576), (12.92581, 77.63463)
    route_len = 2398.0

    def convert_road(lon, lat, isGeo=True, vClass="passenger"):
        # The origin itself sits in a junction; a few metres off it does not.
        if (lat, lon) == origin:
            return ":cluster_0_0", 1.5, 0
        return "310456161", 4.0, 0

    def find_route(frm, to):
        assert not frm.startswith(":"), "findRoute asked to route from a junction"
        return SimpleNamespace(edges=[frm, "27789866#2"], length=route_len)

    world = World.__new__(World)
    world.origin, world.goal = origin, goal
    world.state = VehicleState(lat=12.9250, lon=77.6340)   # wherever ep 1 ended
    world.traci = SimpleNamespace(
        simulation=SimpleNamespace(convertRoad=convert_road,
                                   convertGeo=lambda *a, **k: (0.0, 0.0),
                                   findRoute=find_route),
        vehicle=SimpleNamespace(moveToXY=lambda *a, **k: None,
                                setSpeed=lambda *a, **k: None))
    world.v = SimpleNamespace(steer_calibration_s=2.0)
    world.step_length = 0.1
    world._goal_edge = "27789866#2"
    world._route_retry = ROUTE_RETRY_FRAMES
    world._no_route_frames = 0

    world.reset_episode()

    crow = world.state.distance_to_goal_m
    assert 500.0 < crow < 505.0, crow        # the 502.2 m the runs recorded
    assert world.route_valid, (
        "no live route after reset -- the origin resolved to a junction edge")
    # `convert_road` puts the fly 4 m along the resolved edge, and the route
    # length is measured from that edge's start.
    left = route_len - 4.0
    assert world.state.route_distance_to_goal_m == left, (
        f"reset seeded {world.state.route_distance_to_goal_m:.1f} m; the "
        f"crow-flies distance is {crow:.1f} m and the route is {left} m")
    assert world.best_distance_m == left, (
        "the rewards() ratchet started in crow-flies metres -- 4.8x too "
        "tight on this route, so progress could not earn dopamine")


def test_route_distance_counts_from_the_fly_not_the_edge_start():
    """`findRoute`'s length runs from the *start* of the first edge.

    SUMO sums every edge's full length, the first one included, so a route
    asked for while the fly stands partway along its edge is longer than
    what is left by exactly `edge_pos`. That is every spawn and every
    `reset_episode`: the origin is wherever the click landed, not the start
    of an edge. `_route_point_ahead` already knew this -- it skips `edge_pos`
    into the first edge -- but the published distance did not.

    Measured on the 500 m route: the origin sits ~118 m into its start edge,
    so every episode started 118 m too far out and dropped by that much in
    one frame at the first edge change. A drop is a new best, so `rewards()`
    paid progress dopamine for ground never driven, and `closed_route_m` read
    118 m over the truth. Once 1b52eb7 seeded episodes 2+ in route metres,
    fast episodes closed more route than they drove and tripped run_route's
    assertion: "episode 1 closed 386.0 m of route on 268.4 m travelled".
    """
    from types import SimpleNamespace

    from dipteran.world import ROUTE_RETRY_FRAMES, VehicleState, World

    # Edge A is 200 m long; the route is 1,000 m from A's start, 800 m from
    # the start of B, the next edge along it.
    length_from = {"A": 1000.0, "B": 800.0}

    world = World.__new__(World)
    world.state = VehicleState(lat=12.93143, lon=77.62261)
    world.state.distance_to_goal_m = 426.0
    world.traci = SimpleNamespace(simulation=SimpleNamespace(
        findRoute=lambda frm, to: SimpleNamespace(
            edges=[frm, "goal"], length=length_from[frm])))
    world._goal_edge = "goal"
    world._route_edge = None
    world._route_edges = []
    world._route_length_m = None
    world._route_anchor_edge = None
    world._route_retry = ROUTE_RETRY_FRAMES
    world._no_route_frames = 0
    world._last_route_m = None

    # Spawned 118 m into A.
    world.state.edge, world.state.edge_pos = "A", 118.0
    world._update_route()
    assert world._route_distance_now() == 882.0, (
        f"spawned 118 m into the edge, published "
        f"{world._route_distance_now():.1f} m -- the full route from the "
        f"edge's start, not what is left of it")

    # To the far end of A, then onto B: a fresh route from B's start.
    world.state.edge_pos = 200.0
    world._update_route()
    before = world._route_distance_now()
    world.state.edge, world.state.edge_pos = "B", 0.0
    world._update_route()
    after = world._route_distance_now()
    assert before == after == 800.0, (
        f"crossing onto the next edge moved the route distance "
        f"{before:.1f} -> {after:.1f} m without driving a metre -- a new "
        f"best that pays progress dopamine for nothing")


def test_route_seed_settles_on_the_edge_the_fly_drives():
    """The spawn's edge is a lookup; the seed belongs to the edge driven.

    Both test routes' origins sit inside junctions, so `_road_edge_near`
    resolves them by searching outward -- clockwise from north, first normal
    edge wins -- and routes are one-directional. Traced on the 500 m route
    (2026-09-23): the lookup gives `38042287#0`, 658.4 m to the goal around a
    loop of four more edges. The fly, pointed at the goal, is on `1150874214`
    after 0.7 m of travel, 535.6 m out. That 122.8 m drop on 0.7 m driven was
    a new best -- progress dopamine at the start of every episode -- and
    run_route's closure ran ~120 m ahead of the distance driven until its
    assertion caught it: "episode 1 closed 383.3 m of route on 268.6 m
    travelled".
    """
    from types import SimpleNamespace

    from dipteran.world import ROUTE_RETRY_FRAMES, VehicleState, World

    origin, goal = (12.93143, 77.62261), (12.92760, 77.62251)
    # Lengths from each edge's start, as findRoute gives them.
    length_from = {"38042287#0": 658.5, "1150874214": 535.75}

    def convert_road(lon, lat, isGeo=True, vClass="passenger"):
        # The origin itself is in a junction; the search north of it finds
        # the lookup edge.
        if (lat, lon) == origin:
            return ":cluster_1620344798_304771894_0", 8.75, 0
        return "38042287#0", 0.25, 0

    world = World.__new__(World)
    world.origin, world.goal = origin, goal
    world.state = VehicleState(lat=12.9281, lon=77.6213)   # last episode's end
    world.traci = SimpleNamespace(
        simulation=SimpleNamespace(
            convertRoad=convert_road,
            convertGeo=lambda *a, **k: (0.0, 0.0),
            findRoute=lambda frm, to: SimpleNamespace(
                edges=[frm, "-27785750"], length=length_from[frm])),
        vehicle=SimpleNamespace(moveToXY=lambda *a, **k: None,
                                setSpeed=lambda *a, **k: None))
    world.v = SimpleNamespace(steer_calibration_s=2.0)
    world.step_length = 0.1
    world._goal_edge = "-27785750"
    world._route_retry = ROUTE_RETRY_FRAMES
    world._no_route_frames = 0

    def frame(edge, pos, travelled):
        s = world.state
        s.edge, s.edge_pos, s.travelled_m = edge, pos, travelled
        world._update_route()
        s.route_distance_to_goal_m = world._route_distance_now()
        world._settle_route_seed()
        return s.route_distance_to_goal_m

    world.reset_episode()
    assert world.start_route_m == 658.25 and not world.route_seeded, (
        "the spawn's route distance is a lookup and must stay provisional")

    # The calibration blink: parked on the lookup edge, nothing driven yet.
    frame("38042287#0", 0.25, 0.0)
    assert not world.route_seeded

    # 0.7 m later `_pull_to_road` has the fly on the edge it really drives.
    here = frame("1150874214", 0.25, 0.7)
    assert here == 535.5
    assert world.route_seeded
    assert (world.start_route_m == world.best_distance_m
            == world._prev_route_m == 535.5), (
        f"the 122.75 m between the lookup edge's route and the driven edge's "
        f"was left in the seed (start {world.start_route_m:.2f}, best "
        f"{world.best_distance_m:.2f}) -- progress dopamine and closure for "
        f"0.7 m of driving")

    # From here on a drop is progress again: the seed does not follow it.
    assert frame("1150874214", 20.25, 20.7) == 515.5
    assert world.start_route_m == world.best_distance_m == 535.5


def test_pursuit_target_stays_ahead_through_a_junction():
    """On a junction edge the fly has already left the route's first edge.

    `_route_point_ahead` skipped `edge_pos` into the first route edge only
    while the fly stood *on* it. Inside a junction `state.edge` is the
    internal `:cluster_...` edge, so the skip was zero and the lookahead was
    measured from the start of the edge the fly had just driven -- a target
    behind it. Measured on three live arrival recordings (a37cda7 and
    de4d8e2): 31-56% of junction frames had the target more than 120
    degrees off the heading, against 0-15% of frames on normal edges. Every
    junction crossing told the fly to turn round.
    """
    from dipteran.world import VehicleState, World

    world = World.__new__(World)
    world.state = VehicleState(lat=12.93, lon=77.62)
    # Route A then B along the x axis; A ends where the junction starts.
    world._route_edges = ["A", "B"]
    world._lane_shapes = {"A": [(0.0, 0.0), (100.0, 0.0)],
                          "B": [(106.0, 0.0), (300.0, 0.0)]}

    # Out of A, a few metres into the junction between A and B.
    world.state.edge, world.state.edge_pos = ":J_0", 3.0
    x, y = world._route_point_ahead(45.0)
    assert x > 100.0, (
        f"inside the junction the pursuit target is at x={x:.1f}, back on "
        f"the edge the fly just left (A ends at x=100) -- a turn-round cue")
    assert (x, y) == (151.0, 0.0), (x, y)   # 45 m into B

    # On A itself the target still counts from where the fly stands.
    world.state.edge, world.state.edge_pos = "A", 80.0
    assert world._route_point_ahead(45.0) == (131.0, 0.0)


def test_a_pinned_fly_senses_that_it_is_not_moving():
    """Optic flow and the stall timer read realised motion, not intent.

    Silk Board trace, 2026-09-23: a panic swerve left the fly on the
    reverse carriageway facing the roadside, and it sat at one spot for
    5,600+ frames -- `s.speed` ~4 m/s, while `_pull_to_road` cancelled
    nearly every step (0.4-0.5 m/s realised, 2 m off-road). Optic flow and
    the stall timer both read `s.speed`, what the fly *asked* for, so T5's
    "still air on a fly that is trying to move" never rose and the 45 s
    stall punishment never came. The full run spent 105 of its 120 minutes
    like that.
    """
    from types import SimpleNamespace

    from dipteran.world import VehicleState, World

    world = World.__new__(World)
    world.state = VehicleState(lat=12.917343, lon=77.628487)
    world.goal = (13.0039, 77.6653)
    world.v = SimpleNamespace(max_speed_ms=8.33)
    world.cfg = SimpleNamespace(snn=SimpleNamespace(steer_half_angle_deg=70.0))
    world.pothole_density = 0.0
    # No routing in this test: an unresolved goal edge makes `_update_route`
    # a no-op, and an empty route makes the target the goal itself.
    world._goal_edge = ""
    world._route_edge = None
    world._route_edges = []
    world._route_length_m = None
    world._route_anchor_edge = None
    world._route_retry = 0
    world._last_route_m = None
    world.route_seeded = True
    world._calibrating = 0
    world._perceive = lambda dt: ([], 0.0, 0.0)

    s = world.state
    s.edge = "38098070#1"
    s.speed, s.throttle_cmd = 4.0, 0.5      # asking for 4 m/s...
    s.realised_speed_ms = 0.45              # ...the road pull allows 0.45

    for _ in range(460):                    # 46 s at 0.1 s per frame
        sensed = world.sense(0.1)

    assert sensed["optic_flow_t4"] < 0.1, (
        f"optic flow {sensed['optic_flow_t4']:.2f} on a fly realising 0.45 "
        f"m/s -- it is reading the speed asked for, not achieved")
    assert sensed["optic_flow_t5"] > 0.4, (
        "no jam signal: throttle open, nothing moving, and T5 stayed quiet")
    assert s.stalled_s > 45.0, (
        f"stall timer {s.stalled_s:.1f} s after 46 s pinned -- the 45 s "
        f"stall punishment can never fire on a pinned fly")


def test_the_kerb_turns_a_blocked_fly_along_the_lane():
    """Silk Board pin (93358a2 trace): facing 108 degrees at the end of a
    lane running ~190, target to the right inside the decode dead zone, the
    road pull cancelling ~90% of each step. The heading never changed, so
    neither did anything else for 7,000 frames."""
    from types import SimpleNamespace

    from dipteran.world import VehicleState, World

    world = World.__new__(World)
    world.state = VehicleState(lat=12.9173, lon=77.6285)
    world.v = SimpleNamespace(kerb_turn_deg_s=60.0)
    # A lane running due south in SUMO x/y (y decreasing).
    world._lane_shapes = {"L": [(0.0, 100.0), (0.0, 0.0)]}
    s = world.state
    s.edge, s.edge_pos, s.heading = "L", 95.0, 108.0

    s.speed, s.realised_speed_ms = 4.0, 0.4          # pinned
    for _ in range(20):                               # 2 s
        world._kerb_steer(0.1)
    assert abs(s.heading - 180.0) < 1e-6, (
        f"heading {s.heading:.1f} after 2 s pressed against a lane running "
        f"180 -- the kerb did not turn it")

    # Driving freely, the kerb leaves the connectome's heading alone.
    s.heading, s.realised_speed_ms = 108.0, 3.9
    world._kerb_steer(0.1)
    assert s.heading == 108.0


def test_a_panic_burst_swerves_once_not_every_frame():
    """A four-frame DNp01 burst used to add 45 degrees per frame -- a
    180-degree flip onto the reverse carriageway at Silk Board."""
    from types import SimpleNamespace

    from dipteran.navigator import MotorOutput
    from dipteran.world import VehicleState, World

    world = World.__new__(World)
    world.state = VehicleState(lat=12.9173, lon=77.6285, heading=0.0)
    world.v = SimpleNamespace(
        steer_smoothing_s=0.0, steer_bias_tau_s=0.0, steer_scale=1.0,
        throttle_scale=1.0, max_turn_deg_s=0.0, max_speed_ms=8.0,
        max_accel_ms2=2.5, brake_ms2=8.0, kerb_turn_deg_s=0.0,
        steer_right_gain=1.0,
        stall_window_s=5.0)
    world._calibrating = 0
    world._steer_rest = 0.0
    world.traci = SimpleNamespace()               # every call inside raises
    world._pull_to_road = lambda lat, lon, dt: (lat, lon)

    for _ in range(4):
        world.apply(MotorOutput(panic=0.5, steer=0.1), 0.1)
    assert world.state.heading == 45.0, world.state.heading

    world.apply(MotorOutput(panic=0.0, steer=0.1), 0.1)
    world.apply(MotorOutput(panic=0.5, steer=0.1), 0.1)
    assert world.state.heading == 90.0, "a fresh burst should jink again"


def test_right_turns_get_the_measured_extra_gain():
    """Test F: the connectome swings DNa02 R-L +14.2 Hz for a right turn
    and -26.1 Hz for a left, so the same readout magnitude has to turn the
    fly harder to the right. Left turns are left untouched."""
    from types import SimpleNamespace

    from dipteran.navigator import MotorOutput
    from dipteran.world import VehicleState, World

    def turned(steer):
        world = World.__new__(World)
        world.state = VehicleState(lat=12.9173, lon=77.6285, heading=180.0)
        world.v = SimpleNamespace(
            steer_smoothing_s=0.0, steer_bias_tau_s=0.0, steer_scale=10.0,
            throttle_scale=1.0, max_turn_deg_s=45.0, max_speed_ms=8.0,
            max_accel_ms2=2.5, brake_ms2=8.0, kerb_turn_deg_s=0.0,
            stall_window_s=5.0, steer_right_gain=1.84)
        world._calibrating = 0
        world._steer_rest = 0.0
        world.traci = SimpleNamespace()
        world._pull_to_road = lambda lat, lon, dt: (lat, lon)
        world.apply(MotorOutput(steer=steer), 0.1)
        return world.state.heading - 180.0

    assert abs(turned(0.01) - 0.01 * 10.0 * 1.84 * 45.0 * 0.1) < 1e-9
    assert abs(turned(-0.01) + 0.01 * 10.0 * 45.0 * 0.1) < 1e-9


def test_the_live_steering_fit_finds_the_true_offset_from_a_crab():
    """Tin Factory trace, 2026-09-28: the parked offset was -0.013, the true
    zero-error readout +0.001, and the fly settled at a standing -35 deg
    error where the old +/-10 deg window never opened. The fit has to
    recover the true offset from one-sided data, then even out the sides
    once both are seen."""
    import math
    from types import SimpleNamespace

    from dipteran.world import World

    def readout(err):
        return 0.001 + 0.0003 * min(err, 0.0) + 0.0006 * max(err, 0.0)

    world = World.__new__(World)
    world.v = SimpleNamespace(steer_bias_tau_s=30.0, steer_fit_window_deg=60.0)
    fit = None
    for i in range(600):                     # 60 s crabbing at -20..-50
        err = -35.0 + 15.0 * math.sin(i / 20.0)
        fit = world._steer_fit(readout(err), err, 4.0, 0.1) or fit
    assert fit is not None and abs(fit[0] - 0.001) < 1e-4, fit
    assert fit[1:] == (1.0, 1.0), "one side only: no gain correction"

    for i in range(3000):                    # 300 s across both sides
        err = 40.0 * math.sin(i / 20.0)
        fit = world._steer_fit(readout(err), err, 4.0, 0.1) or fit
    rest, gain_left, gain_right = fit
    assert abs(rest - 0.001) < 1e-4, rest
    # Slopes 0.0003 left, 0.0006 right: mean 0.00045.
    assert abs(gain_left - 1.5) < 0.02 and abs(gain_right - 0.75) < 0.02, fit

    # Parked, or a target far behind: no update.
    assert world._steer_fit(0.5, 0.0, 0.0, 0.1) is None
    assert world._steer_fit(0.5, 120.0, 4.0, 0.1) is None


def test_the_fly_is_not_teleported_onto_an_unconnected_road():
    """Tin Factory trace, 2026-09-28: 1-2 m off its lane, the fly was
    snapped onto an overlapping road it has no connection to (a parallel
    road, and the reverse carriageway), adding ~400-1,200 m of route.
    Legal successors must still be accepted."""
    from types import SimpleNamespace

    from dipteran.world import World

    world = World.__new__(World)
    world.v = SimpleNamespace(lane_keep_m=8.0)
    # A runs east along y=0 into junction J, then B continues east; R is an
    # unconnected road 2 m north, running the other way.
    world._lane_shapes = {"A": [(0.0, 0.0), (100.0, 0.0)],
                          "B": [(105.0, 0.0), (200.0, 0.0)],
                          ":J_0": [(100.0, 0.0), (105.0, 0.0)],
                          "R": [(200.0, 2.0), (0.0, 2.0)]}
    world.traci = SimpleNamespace(
        edge=SimpleNamespace(getLaneNumber=lambda e: 1),
        lane=SimpleNamespace(getLinks=lambda lane: (
            [("B_0", True, True, False, ":J_0_0", "G", "s", 5.0)]
            if lane == "A_0" else [])),
        simulation=SimpleNamespace(
            convertGeo=lambda lon, lat, fromGeo=False: (lon, lat)))
    world._last_normal_edge = "A"

    # 1.5 m north of A at x=60: nearest is R, which A cannot reach.
    edge, pos, (x, y) = world._connected_snap(1.5, 60.0, "R")
    assert edge == "A" and abs(pos - 60.0) < 1e-9 and (x, y) == (60.0, 0.0)

    # Past A's end, before B starts, the legal road is the junction lane.
    # Without it the fly was pulled back to A's end every frame, pinned.
    edge, pos, (x, y) = world._connected_snap(1.5, 102.5, "R")
    assert edge == ":J_0" and abs(pos - 2.5) < 1e-9, (edge, pos)

    # Legal moves are left alone: A's successor, and junction J's lanes.
    assert world._connected_snap(0.5, 150.0, "B") is None
    assert world._connected_snap(0.5, 102.0, ":J_0") is None

    # Far from every legal lane, the fly really is elsewhere.
    assert world._connected_snap(50.0, 60.0, "R") is None

    # Stuck long enough, lane-keeping lets go rather than hold the fly.
    world._release_s = 3.0
    assert world._connected_snap(1.5, 60.0, "R") is None


def test_the_stall_timer_sees_a_jittering_pin():
    """Pinned at Silk Board, realised speed swung 0.1-1.9 m/s frame to frame
    with no net progress, so the instantaneous stall test kept resetting."""
    from types import SimpleNamespace

    from dipteran.world import VehicleState, World

    world = World.__new__(World)
    world.state = VehicleState(lat=12.917343, lon=77.628487)
    world.v = SimpleNamespace(stall_window_s=5.0)
    lat0 = world.state.lat
    for i in range(51):                     # 5 s oscillating +/- 1 m
        world._track_position(lat0 + (1 if i % 2 else -1) / 111_320.0,
                              77.628487, 0.1)
    net = world._net_speed_ms()
    assert net is not None and net < 0.5, net

    for i in range(51):                     # 5 s at 4 m/s due north
        world._track_position(lat0 + 0.4 * i / 111_320.0, 77.628487, 0.1)
    assert abs(world._net_speed_ms() - 4.0) < 0.05


def test_pothole_is_a_jolt_not_a_buzz():
    """A potholed edge held the 1,417-neuron bristle channel on for its
    whole length and flipped right turns left. One 0.5 s pulse per edge."""
    from dipteran.world import POTHOLE_JOLT_S, World

    world = World.__new__(World)
    frames = [world._pothole_jolt(True, "e1", 0.1) for _ in range(30)]
    assert sum(frames) == round(POTHOLE_JOLT_S / 0.1), frames
    assert frames[0] and not frames[-1]
    assert world._pothole_jolt(True, "e2", 0.1)        # next potholed edge
    assert not world._pothole_jolt(False, "e3", 0.1)
    assert world._pothole_jolt(True, "e1", 0.1)        # re-entry jolts again


def test_narrow_fork_follows_the_route():
    """Two legal branches a few metres apart: the route's one wins; a branch
    well clear of the route's is left to the fly."""
    from types import SimpleNamespace

    from dipteran.world import World

    world = World.__new__(World)
    world.v = SimpleNamespace(lane_keep_m=8.0)
    world._last_normal_edge = "a"
    world._route_edges = ["a", "route"]
    world._reachable = lambda e: ({"route", "loop"}, set(), set())
    world.traci = SimpleNamespace(simulation=SimpleNamespace(
        convertGeo=lambda lon, lat, fromGeo: (lon, lat)))
    shapes = {"route": [(3.0, 0.0), (3.0, 50.0)], "far": [(30.0, 0.0), (30.0, 50.0)]}
    world._lane_shape = lambda e: shapes.get(e)

    edge, pos, _xy = world._connected_snap(10.0, 0.0, "loop")   # lat=y, lon=x
    assert edge == "route" and abs(pos - 10.0) < 1e-6, (edge, pos)
    shapes["route"] = shapes["far"]
    assert world._connected_snap(10.0, 0.0, "loop") is None     # fly's call
    world._route_edges = ["a", "loop"]
    assert world._connected_snap(10.0, 0.0, "loop") is None     # on route


def test_edge_end_hands_over_across_a_gap():
    """Clamped at the end of a ramp whose successor starts 9 m on: the
    successor takes over instead of the ramp's end pinning the fly."""
    from types import SimpleNamespace

    from dipteran.world import World

    world = World.__new__(World)
    world.v = SimpleNamespace(lane_keep_m=8.0)
    world._last_normal_edge = "ramp"
    world._route_edges = []
    world._reachable = lambda e: ({"next"}, set(), set())
    world.traci = SimpleNamespace(simulation=SimpleNamespace(
        convertGeo=lambda lon, lat, fromGeo: (lon, lat)))
    shapes = {"ramp": [(0.0, 0.0), (0.0, 94.0)], "next": [(0.0, 103.0), (0.0, 110.0)]}
    world._lane_shape = lambda e: shapes.get(e)

    edge, pos, _xy = world._connected_snap(95.0, 0.0, "ramp")   # 1 m past end
    assert edge == "next" and pos == 0.0, (edge, pos)
    assert world._connected_snap(50.0, 0.0, "ramp") is None     # mid-ramp


def test_eases_off_for_a_sharp_turn():
    from types import SimpleNamespace

    from dipteran.navigator import MotorOutput
    from dipteran.world import TURN_SPEED_FLOOR, VehicleState, World

    def speed_after(error_deg):
        world = World.__new__(World)
        world.state = VehicleState(lat=12.9, lon=77.6, heading=0.0,
                                   heading_error_deg=error_deg)
        world.v = SimpleNamespace(
            steer_smoothing_s=0.0, steer_bias_tau_s=0.0, steer_scale=1.0,
            throttle_scale=1.0, max_turn_deg_s=0.0, max_speed_ms=10.0,
            max_accel_ms2=100.0, brake_ms2=100.0, kerb_turn_deg_s=0.0,
            steer_right_gain=1.0, stall_window_s=5.0)
        world._calibrating = 0
        world._steer_rest = 0.0
        world.traci = SimpleNamespace()
        world._pull_to_road = lambda lat, lon, dt: (lat, lon)
        world.apply(MotorOutput(throttle=1.0), 0.1)
        return world.state.speed

    assert abs(speed_after(0.0) - 10.0) < 1e-9
    assert abs(speed_after(60.0) - 5.0) < 1e-9
    assert abs(speed_after(170.0) - 10.0 * TURN_SPEED_FLOOR) < 1e-9


def test_repeat_fork_widens_the_window():
    """Route branch 12 m off: first pass is the fly's call, a repeat lap
    through the same edge gets the route's branch."""
    from types import SimpleNamespace

    from dipteran.world import World

    world = World.__new__(World)
    world.v = SimpleNamespace(lane_keep_m=8.0)
    world._last_normal_edge = "a"
    world._route_edges = ["a", "route"]
    world._reachable = lambda e: ({"route", "loop"}, set(), set())
    world.traci = SimpleNamespace(simulation=SimpleNamespace(
        convertGeo=lambda lon, lat, fromGeo: (lon, lat)))
    world._lane_shape = lambda e: {"route": [(12.0, 0.0), (12.0, 50.0)]}.get(e)

    assert world._connected_snap(10.0, 0.0, "loop") is None
    world._visits = {"a": 2}
    edge, _pos, _xy = world._connected_snap(10.0, 0.0, "loop")
    assert edge == "route", edge


def test_dead_end_turns_round():
    """Pinned at the end of an edge with no successors: heading flips and
    the lane becomes the reverse carriageway. Mid-edge, nothing happens."""
    from types import SimpleNamespace

    from dipteran.world import VehicleState, World

    world = World.__new__(World)
    world.state = VehicleState(lat=30.5, lon=0.0, heading=240.0, speed=3.5,
                               realised_speed_ms=0.0, edge="-800628086")
    world._reachable = lambda e: (set(), set(), set())
    world.traci = SimpleNamespace(simulation=SimpleNamespace(
        convertGeo=lambda lon, lat, fromGeo: (lon, lat)))
    shapes = {"-800628086": [(0.0, 0.0), (0.0, 31.0)],
              "800628086": [(0.0, 31.0), (0.0, 0.0)]}
    world._lane_shape = lambda e: shapes.get(e)

    world._turn_at_dead_end()
    assert world.state.heading == 60.0, world.state.heading
    assert world._last_normal_edge == "800628086"

    world.state.lat, world.state.heading = 10.0, 240.0
    world._turn_at_dead_end()
    assert world.state.heading == 240.0


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
