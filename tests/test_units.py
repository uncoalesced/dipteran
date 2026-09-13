"""Unit tests for the pure logic Phase 1 shipped without any.

GATES.md G1/G2 record this gap: the `.net.xml` parser, the point-to-polyline
geometry the whole OSM<->SUMO join rests on, the `osmid` normaliser, the
way-index strategy selection and the BFS all went to production untested.
Phase 2 stacks the navigator on top of them, so they get tests first.

Nothing here touches the 1 GB downloaded corpus -- the SUMO parser is
exercised against a synthetic `.net.xml` written to a temp dir, which is the
point: these have to run on a fresh clone with no data fetched.

    python tests/test_units.py
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

from dipteran.roads import (  # noqa: E402
    DEG_LAT_M,
    SumoNet,
    _osmids,
    _way_index,
    point_to_polyline_m,
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
# runner
# --------------------------------------------------------------------------

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
