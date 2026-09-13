"""Both road-network representations, from one OSM extract, plus the join.

Phase 1 tasks 6-8. Pipeline:

  1. download_osm_extract  -- one Overpass query -> cache/bangalore.osm.xml
  2. build_osmnx_graph     -- ox.graph_from_xml on that file  -> cache/*.graphml
  3. build_sumo_net        -- netconvert on the same file     -> cache/*.net.xml
  4. build_id_map          -- join OSMnx edges <-> SUMO edges -> cache/*.json

Both graphs come from the *same* local .osm file rather than two independent
downloads. That is what makes step 4 a join on shared OSM way IDs instead of a
geometry-matching guess.

The join is two-stage, and the second stage is not optional. Matching on OSM
way ID alone is correct but far too coarse: one long way (Outer Ring Road, say)
becomes hundreds of OSMnx edges *and* hundreds of SUMO edges, and a pure ID
join maps every one of those OSMnx edges to every one of those SUMO edges. So
the way ID narrows the candidate set -- cheaply and without false negatives --
and geometry then picks the segments that actually overlap.

Run `python -m dipteran.roads` to execute all four steps and verify the join.
"""

from __future__ import annotations

import datetime as dt
import gc
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from dipteran.config import Config, load_config

OVERPASS_ENDPOINTS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
)

# Metres per degree, used for a local equirectangular frame. Over a ~30 km
# extract the distortion is well under the tolerances below.
DEG_LAT_M = 110_574.0

# OSMnx's own `drive` way filter, inlined so the Overpass query, the OSMnx
# graph and the SUMO network all see an identical way set. Preferred source is
# osmnx's private helper; this is the fallback when that moves.
OSM_FILTERS = {
    "drive": (
        '["highway"]["area"!~"yes"]'
        '["highway"!~"abandoned|bridleway|bus_guideway|construction|corridor|'
        "cycleway|elevator|escalator|footway|ladder|no|path|pedestrian|planned|"
        'platform|proposed|raceway|service|service_road|steps|track"]'
        '["motor_vehicle"!~"no"]["motorcar"!~"no"]'
        '["service"!~"alley|driveway|emergency_access|parking|parking_aisle|private"]'
    ),
    "drive_service": (
        '["highway"]["area"!~"yes"]'
        '["highway"!~"abandoned|bridleway|construction|corridor|cycleway|elevator|'
        "escalator|footway|ladder|no|path|pedestrian|planned|platform|proposed|"
        'raceway|steps|track"]'
        '["motor_vehicle"!~"no"]["motorcar"!~"no"]'
        '["service"!~"emergency_access|parking|parking_aisle|private"]'
    ),
}

# netconvert derives edge IDs from OSM way IDs: an optional '-' for the
# reverse direction, the way ID, and an optional '#n' segment suffix when a
# way is split at a junction. ':' prefixes are internal junction lanes.
SUMO_EDGE_ID = re.compile(r"^(?P<rev>-?)(?P<way>\d+)(?:#(?P<seg>-?\d+))?$")


def _deg_lon_m(lat_deg: float) -> float:
    return 111_320.0 * math.cos(math.radians(lat_deg))


def osm_filter(network_type: str) -> str:
    """Overpass way filter for a network_type, preferring OSMnx's own string."""
    try:
        from osmnx._overpass import _get_osm_filter

        return _get_osm_filter(network_type)
    except Exception:
        if network_type not in OSM_FILTERS:
            raise ValueError(
                f"no inlined Overpass filter for network_type={network_type!r}; "
                f"have {sorted(OSM_FILTERS)}"
            ) from None
        return OSM_FILTERS[network_type]


# --------------------------------------------------------------------------
# 1. OSM extract
# --------------------------------------------------------------------------

def extract_bbox(cfg: Config) -> tuple[float, float, float, float]:
    """(south, west, north, east) for the extract.

    The `dist_m` square around the origin, clipped to the configured city
    envelope so the extract never covers ground the frontend will not let a
    viewer look at.
    """
    r = cfg.route
    dlat = math.degrees(r.dist_m / 6_371_008.8)
    dlon = dlat / max(math.cos(math.radians(r.origin.lat)), 1e-6)
    south = max(r.origin.lat - dlat, r.bbox.south)
    north = min(r.origin.lat + dlat, r.bbox.north)
    west = max(r.origin.lon - dlon, r.bbox.west)
    east = min(r.origin.lon + dlon, r.bbox.east)

    dest = r.destination
    if not (south <= dest.lat <= north and west <= dest.lon <= east):
        raise ValueError(
            f"extract bbox ({south:.4f},{west:.4f},{north:.4f},{east:.4f}) "
            f"excludes the destination ({dest.lat}, {dest.lon}); raise "
            "route.dist_m or widen route.bbox"
        )
    return south, west, north, east


def download_osm_extract(cfg: Config, force: bool = False) -> Path:
    """Fetch the drivable ways for the configured area as raw OSM XML."""
    out = cfg.paths.osm_extract
    if out.exists() and not force and out.stat().st_size > 0:
        print(f"[osm] reusing {out} ({out.stat().st_size / 1e6:.1f} MB)")
        return out

    south, west, north, east = extract_bbox(cfg)
    query = (
        "[out:xml][timeout:900];\n"
        f"(way{osm_filter(cfg.route.network_type)}({south},{west},{north},{east}););\n"
        "(._;>;);\n"
        "out body;\n"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")

    last_error: Exception | None = None
    for endpoint in OVERPASS_ENDPOINTS:
        print(f"[osm] querying {endpoint} for bbox "
              f"({south:.4f},{west:.4f},{north:.4f},{east:.4f})", flush=True)
        request = urllib.request.Request(
            endpoint,
            data=urllib.parse.urlencode({"data": query}).encode(),
            headers={"User-Agent": "dipteran/0.1 (github.com/uncoalesced/dipteran)"},
        )
        try:
            started = time.monotonic()
            with urllib.request.urlopen(request, timeout=960) as response, \
                    open(tmp, "wb") as fh:
                shutil.copyfileobj(response, fh, length=1 << 20)
            size = tmp.stat().st_size
            if size < 10_000:
                raise RuntimeError(f"Overpass returned only {size} bytes")
            tmp.replace(out)
            print(f"[osm] wrote {out} ({size / 1e6:.1f} MB) in "
                  f"{time.monotonic() - started:.0f}s", flush=True)
            return out
        except Exception as exc:
            last_error = exc
            tmp.unlink(missing_ok=True)
            print(f"[osm] {endpoint} failed: {exc}", flush=True)

    raise RuntimeError(f"all Overpass endpoints failed; last error: {last_error}")


# --------------------------------------------------------------------------
# 2. OSMnx graph
# --------------------------------------------------------------------------

def build_osmnx_graph(cfg: Config, force: bool = False):
    """The OSMnx/networkx graph the SNN reasons over (bearing, distance)."""
    import networkx as nx
    import osmnx as ox

    out = cfg.paths.graphml
    if out.exists() and not force:
        print(f"[osmnx] loading cached {out}", flush=True)
        graph = ox.load_graphml(out)
    else:
        source = download_osm_extract(cfg)
        print(f"[osmnx] building graph from {source}", flush=True)
        graph = ox.graph_from_xml(source, simplify=True, retain_all=False)
        out.parent.mkdir(parents=True, exist_ok=True)
        ox.save_graphml(graph, out)

    n_scc = nx.number_strongly_connected_components(graph)
    print(f"[osmnx] {graph.number_of_nodes()} nodes, "
          f"{graph.number_of_edges()} edges, {n_scc} strongly-connected "
          f"components", flush=True)
    return graph


def route_endpoints(cfg: Config, graph) -> tuple[int, int]:
    """Graph nodes nearest to the configured origin and destination."""
    import osmnx as ox

    r = cfg.route
    orig = ox.nearest_nodes(graph, r.origin.lon, r.origin.lat)
    dest = ox.nearest_nodes(graph, r.destination.lon, r.destination.lat)
    return int(orig), int(dest)


# --------------------------------------------------------------------------
# 3. SUMO network
# --------------------------------------------------------------------------

def _netconvert_binary(cfg: Config) -> str:
    if cfg.sumo.binary_dir:
        candidate = cfg.sumo.binary_dir / "netconvert"
        if candidate.exists():
            return str(candidate)
    found = shutil.which("netconvert")
    if found:
        return found
    sumo_home = os.environ.get("SUMO_HOME")
    if sumo_home:
        candidate = Path(sumo_home) / "bin" / "netconvert"
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(
        "netconvert not found -- install SUMO (apt-get install sumo sumo-tools) "
        "or set sumo.binary_dir in config.yaml"
    )


def _typemap() -> list[str]:
    """netconvert --type-files for OSM input, if SUMO's data dir is present."""
    for root in (os.environ.get("SUMO_HOME"), "/usr/share/sumo", "/usr/local/share/sumo"):
        if not root:
            continue
        typ = Path(root) / "data" / "typemap" / "osmNetconvert.typ.xml"
        if typ.exists():
            return ["--type-files", str(typ)]
    print("[sumo] osmNetconvert.typ.xml not found; using netconvert defaults")
    return []


def build_sumo_net(cfg: Config, force: bool = False) -> Path:
    """netconvert the same OSM extract into a routable SUMO network."""
    out = cfg.paths.sumo_net
    if out.exists() and not force:
        print(f"[sumo] reusing {out}", flush=True)
        return out

    source = download_osm_extract(cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        _netconvert_binary(cfg),
        "--osm-files", str(source),
        "--output-file", str(out),
        *_typemap(),
        # Keep the source OSM way ID on every lane. This is what makes stage
        # one of the join a lookup rather than a geometry guess, and it
        # survives --geometry.remove merging edges together.
        "--output.original-names", "true",
        # Drivable-only, routable topology.
        "--keep-edges.by-vclass", "passenger",
        "--remove-edges.isolated", "true",
        "--geometry.remove", "true",
        "--roundabouts.guess", "true",
        "--ramps.guess", "true",
        "--junctions.join", "true",
        "--tls.guess-signals", "true",
        "--tls.discard-simple", "true",
        "--tls.join", "true",
        "--no-turnarounds", "true",
    ]
    print(f"[sumo] {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0 or not out.exists():
        raise RuntimeError(
            f"netconvert failed (exit {result.returncode})\n"
            f"{result.stderr.strip()[-4000:]}"
        )
    warnings = [ln for ln in result.stderr.splitlines() if ln.startswith("Warning")]
    print(f"[sumo] wrote {out} ({out.stat().st_size / 1e6:.1f} MB), "
          f"{len(warnings)} netconvert warnings", flush=True)
    return out


# --------------------------------------------------------------------------
# 4. Geometry
# --------------------------------------------------------------------------

def point_to_polyline_m(point: np.ndarray, poly: np.ndarray) -> float:
    """Shortest distance from a point to a polyline, in a local metric frame.

    `point` is (2,), `poly` is (n, 2). Both already in metres.
    """
    if len(poly) == 1:
        return float(np.hypot(*(poly[0] - point)))
    a = poly[:-1]
    b = poly[1:]
    ab = b - a
    ap = point - a
    denom = np.einsum("ij,ij->i", ab, ab)
    t = np.where(denom > 0, np.einsum("ij,ij->i", ap, ab) / np.where(denom > 0, denom, 1), 0.0)
    t = np.clip(t, 0.0, 1.0)
    closest = a + t[:, None] * ab
    return float(np.min(np.hypot(*(closest - point).T)))


class SumoNet:
    """The bits of a SUMO .net.xml this project needs, without sumolib.

    sumolib would do this, but it ships with the SUMO tools rather than on the
    Python import path. All we need per edge is its origId params, its shape,
    its length, and the network's projection -- and we need the shapes as one
    vectorised array, because the join touches every edge.
    """

    def __init__(self, net_xml: Path, with_connections: bool = False):
        self.path = net_xml
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.proj_params: str | None = None

        self.edge_ids: list[str] = []
        self.origids: list[set[str]] = []
        lengths: list[float] = []
        shapes: list[list[tuple[float, float]]] = []
        # Buffering ~1.6M edge-id strings before `index` exists costs a few
        # hundred MB, and only the routability check needs them -- the ID-map
        # check already holds the OSMnx graph and has nothing to spare.
        conn_from: list[str] = []
        conn_to: list[str] = []
        self.with_connections = with_connections
        self.skipped_no_shape = 0
        self._parse(lengths, shapes, conn_from, conn_to)

        self.index = {edge_id: i for i, edge_id in enumerate(self.edge_ids)}
        if with_connections:
            self._build_connections(conn_from, conn_to)
        del conn_from, conn_to
        gc.collect()
        self.lengths = np.asarray(lengths, dtype=np.float64)
        counts = np.asarray([len(s) for s in shapes], dtype=np.int64)
        self._offsets = np.concatenate([[0], np.cumsum(counts)])
        flat = [pt for shape in shapes for pt in shape]
        self._xy = (np.asarray(flat, dtype=np.float64)
                    if flat else np.zeros((0, 2), dtype=np.float64))
        self._point_edge = np.repeat(np.arange(len(self.edge_ids)), counts)

        self.lonlat = self._project_all()
        self.centroids = self._per_edge_mean(self.lonlat)
        if len(self.lonlat):
            self.lon0 = float(self.lonlat[:, 0].mean())
            self.lat0 = float(self.lonlat[:, 1].mean())
        else:
            self.lon0 = self.lat0 = 0.0
        self._kx = _deg_lon_m(self.lat0)
        self.local = self.to_local(self.lonlat)
        self.local_centroids = self.to_local(self.centroids)
        self._max_seg_m = self._longest_segment()
        self._tree = self._build_tree()
        print(f"[sumo] parsed {len(self.edge_ids)} edges, "
              f"{len(self._xy)} shape points"
              + (f", {self.skipped_no_shape} without geometry"
                 if self.skipped_no_shape else ""), flush=True)

    # -- parsing ---------------------------------------------------------

    def _parse(self, lengths: list[float],
               shapes: list[list[tuple[float, float]]],
               conn_from: list[str], conn_to: list[str]) -> None:
        for _, elem in ET.iterparse(self.path, events=("end",)):
            if elem.tag == "connection":
                if self.with_connections:
                    source, target = elem.get("from", ""), elem.get("to", "")
                    if source and target and not source.startswith(":") \
                            and not target.startswith(":"):
                        conn_from.append(source)
                        conn_to.append(target)
                elem.clear()
            elif elem.tag == "location":
                off_x, off_y = elem.get("netOffset", "0,0").split(",")
                self.offset_x, self.offset_y = float(off_x), float(off_y)
                self.proj_params = elem.get("projParameter")
                elem.clear()
            elif elem.tag == "edge":
                edge_id = elem.get("id", "")
                # ':' prefixed edges are internal junction connectors.
                if edge_id.startswith(":") or elem.get("function") == "internal":
                    elem.clear()
                    continue
                origids: set[str] = set()
                for param in elem.iter("param"):
                    if param.get("key") == "origId":
                        origids.update(param.get("value", "").split())
                shape: list[tuple[float, float]] = []
                length = 0.0
                for lane in elem.iter("lane"):
                    lane_shape = lane.get("shape")
                    if not shape and lane_shape:
                        for pt in lane_shape.split():
                            x, y = pt.split(",")[:2]
                            shape.append((float(x), float(y)))
                    length = max(length, float(lane.get("length", 0.0)))
                elem.clear()
                if not shape:
                    self.skipped_no_shape += 1
                    continue
                self.edge_ids.append(edge_id)
                self.origids.append(origids)
                lengths.append(length)
                shapes.append(shape)

    # -- connectivity ----------------------------------------------------

    def _build_connections(self, conn_from: list[str],
                           conn_to: list[str]) -> None:
        """Edge-to-edge adjacency as a CSR pair of int32 arrays.

        A dict of sets over ~1.5M connections costs a few hundred MB; two
        int32 arrays cost ~12 MB, which matters because this network already
        pushes the memory budget.
        """
        index = self.index
        pairs = [(index[a], index[b]) for a, b in zip(conn_from, conn_to)
                 if a in index and b in index]
        n = len(self.edge_ids)
        if not pairs:
            self._conn_offsets = np.zeros(n + 1, dtype=np.int64)
            self._conn_targets = np.zeros(0, dtype=np.int32)
            return
        sources = np.fromiter((p[0] for p in pairs), dtype=np.int32,
                              count=len(pairs))
        targets = np.fromiter((p[1] for p in pairs), dtype=np.int32,
                              count=len(pairs))
        order = np.argsort(sources, kind="stable")
        self._conn_targets = targets[order]
        counts = np.bincount(sources, minlength=n)
        self._conn_offsets = np.concatenate([[0], np.cumsum(counts)])
        print(f"[sumo] {len(pairs)} edge-to-edge connections", flush=True)

    def successors(self, edge_index: int) -> np.ndarray:
        start = self._conn_offsets[edge_index]
        stop = self._conn_offsets[edge_index + 1]
        return self._conn_targets[start:stop]

    def reachable(self, from_edge: str, to_edge: str) -> tuple[bool, int]:
        """(is `to_edge` reachable from `from_edge`, edges explored).

        Breadth-first over SUMO's own `<connection>` elements, which encode
        the permitted turns. This is the in-process answer to "are origin and
        destination in the same routable component" -- duarouter gives the
        authoritative route, but needs 4.4 GB to do it.
        """
        if not self.with_connections:
            raise RuntimeError(
                "this SumoNet was built without connections; construct it "
                "with SumoNet(path, with_connections=True)"
            )
        start, goal = self.index[from_edge], self.index[to_edge]
        seen = np.zeros(len(self.edge_ids), dtype=bool)
        seen[start] = True
        frontier = [start]
        explored = 0
        while frontier:
            nxt: list[int] = []
            for edge_index in frontier:
                explored += 1
                if edge_index == goal:
                    return True, explored
                for target in self.successors(edge_index):
                    if not seen[target]:
                        seen[target] = True
                        nxt.append(int(target))
            frontier = nxt
        return False, explored

    # -- projection ------------------------------------------------------

    def _project_all(self) -> np.ndarray:
        """All shape points as (n, 2) lon/lat, in one vectorised call."""
        if not len(self._xy):
            return np.zeros((0, 2), dtype=np.float64)
        if not self.proj_params:
            raise RuntimeError(
                f"{self.path} has no projParameter; netconvert produced an "
                "unprojected network and geographic lookups are impossible"
            )
        from pyproj import Proj

        proj = Proj(self.proj_params)
        lon, lat = proj(self._xy[:, 0] - self.offset_x,
                        self._xy[:, 1] - self.offset_y, inverse=True)
        return np.column_stack([np.asarray(lon), np.asarray(lat)])

    def _per_edge_mean(self, values: np.ndarray) -> np.ndarray:
        if not len(values):
            return np.zeros((0, 2), dtype=np.float64)
        sums = np.add.reduceat(values, self._offsets[:-1], axis=0)
        counts = np.diff(self._offsets).astype(np.float64)
        return sums / counts[:, None]

    def to_local(self, lonlat: np.ndarray) -> np.ndarray:
        """lon/lat -> local metres, the frame all distances are measured in."""
        lonlat = np.atleast_2d(lonlat)
        return np.column_stack([
            (lonlat[:, 0] - self.lon0) * self._kx,
            (lonlat[:, 1] - self.lat0) * DEG_LAT_M,
        ])

    def _longest_segment(self) -> float:
        """Longest single shape segment, in metres.

        This is the exact extra search radius `nearest_edge` needs -- see the
        bound argued there. Consecutive points from *different* edges are not
        a segment, so those gaps are masked out.
        """
        if len(self.local) < 2:
            return 0.0
        steps = np.hypot(*(self.local[1:] - self.local[:-1]).T)
        same_edge = self._point_edge[1:] == self._point_edge[:-1]
        return float(steps[same_edge].max()) if same_edge.any() else 0.0

    def _build_tree(self):
        if not len(self.local):
            return None
        try:
            from scipy.spatial import cKDTree

            return cKDTree(self.local)
        except Exception as exc:
            print(f"[sumo] scipy unavailable ({exc}); nearest-edge lookups "
                  "will brute-force", flush=True)
            return None

    # -- lookups ---------------------------------------------------------

    def edge_lonlat(self, edge_id: str) -> np.ndarray:
        i = self.index[edge_id]
        return self.lonlat[self._offsets[i]:self._offsets[i + 1]]

    def edge_local(self, edge_id: str) -> np.ndarray:
        i = self.index[edge_id]
        return self.local[self._offsets[i]:self._offsets[i + 1]]

    def nearest_edge(self, lon: float, lat: float) -> tuple[str, float]:
        """(edge id, metres) of the SUMO edge nearest a coordinate.

        Pure geometry -- no OSM IDs involved. This is what makes it usable as
        an independent check on the ID-based join.

        The KD-tree indexes shape *points*, so its nearest neighbour is a
        vertex, not an edge. Taking the owner of that vertex is wrong whenever
        the query sits mid-segment: a point 2 m off the middle of a 100 m edge
        is 50 m from either of its endpoints, so the nearest *vertex* belongs
        to a neighbouring edge and the answer comes back 50 m out. Phase 2
        uses this to decide which edge the vehicle is on, so it scores every
        candidate edge with the real point-to-polyline distance and returns
        the true minimum.

        The radius is a bound, not a guess. The best achievable gap is at most
        `d0`, the distance to the nearest vertex (that vertex lies on its own
        edge). The winning edge's closest point sits on a segment no longer
        than `_max_seg_m`, so one of that segment's endpoints is within
        `sqrt(d0**2 + (_max_seg_m / 2)**2) <= d0 + _max_seg_m / 2`.
        """
        query = self.to_local(np.asarray([[lon, lat]]))[0]
        if self._tree is not None:
            d0, _ = self._tree.query(query)
            radius = float(d0) + self._max_seg_m / 2.0
            near = self._tree.query_ball_point(query, radius)
            edge_idx = np.unique(self._point_edge[near])
        else:
            deltas = self.local - query
            d2 = np.einsum("ij,ij->i", deltas, deltas)
            radius = math.sqrt(float(d2.min())) + self._max_seg_m / 2.0
            edge_idx = np.unique(self._point_edge[d2 <= radius * radius])

        best_id, best_gap = "", float("inf")
        for i in edge_idx:
            edge_id = self.edge_ids[int(i)]
            gap = point_to_polyline_m(query, self.edge_local(edge_id))
            if gap < best_gap:
                best_id, best_gap = edge_id, gap
        return best_id, best_gap


# --------------------------------------------------------------------------
# 5. The join
# --------------------------------------------------------------------------

def _osmids(data: dict) -> list[int]:
    """OSMnx stores `osmid` as an int, a list, or a stringified list."""
    raw = data.get("osmid")
    if raw is None:
        return []
    if isinstance(raw, int):
        return [raw]
    if isinstance(raw, str):
        return [int(tok) for tok in re.findall(r"\d+", raw)]
    out: list[int] = []
    for item in raw:
        out.extend(_osmids({"osmid": item}))
    return out


def _way_index(net: SumoNet) -> tuple[dict[str, list[str]], str]:
    """OSM way ID -> SUMO edge IDs, plus which strategy produced it.

    Preference order, per the Phase 1 brief: the explicit `origId` params that
    `--output.original-names` writes, then parsing the edge ID itself (SUMO
    derives it from the way ID), then nothing -- geometry alone.
    """
    from_origid: dict[str, list[str]] = {}
    from_edge_id: dict[str, list[str]] = {}

    for i, edge_id in enumerate(net.edge_ids):
        for origid in net.origids[i]:
            stem = origid.lstrip("-").split("#")[0]
            if stem.isdigit():
                from_origid.setdefault(stem, []).append(edge_id)
        match = SUMO_EDGE_ID.match(edge_id)
        if match:
            from_edge_id.setdefault(match.group("way"), []).append(edge_id)

    if from_origid:
        return from_origid, "origId"
    if from_edge_id:
        return from_edge_id, "edge_id_parse"
    return {}, "none"


def _osmnx_edge_lonlat(graph, u, v, key, data: dict | None) -> np.ndarray:
    """OSMnx edge shape as (n, 2) lon/lat, falling back to its endpoints."""
    if data:
        geom = data.get("geometry")
        if geom is not None and hasattr(geom, "coords"):
            return np.asarray(geom.coords, dtype=np.float64)[:, :2]
    try:
        nu, nv = graph.nodes[u], graph.nodes[v]
    except KeyError:
        return np.zeros((0, 2), dtype=np.float64)
    return np.asarray(
        [[float(nu["x"]), float(nu["y"])], [float(nv["x"]), float(nv["y"])]],
        dtype=np.float64,
    )


def _refine(poly_local: np.ndarray, candidates: list[str], net: SumoNet,
            tolerance_m: float) -> list[str]:
    """Keep only the candidate SUMO edges that actually overlap this edge.

    Stage one hands us every SUMO edge sharing an OSM way ID, which for a long
    way is the whole way. Distance from each candidate's centroid to this
    edge's polyline separates the overlapping segments from the rest.
    """
    if len(candidates) <= 1 or not len(poly_local):
        return candidates
    idx = np.fromiter((net.index[c] for c in candidates), dtype=np.int64,
                      count=len(candidates))
    centroids = net.local_centroids[idx]
    gaps = np.fromiter(
        (point_to_polyline_m(c, poly_local) for c in centroids),
        dtype=np.float64, count=len(candidates),
    )
    keep = gaps <= tolerance_m
    if not keep.any():
        # A SUMO edge much longer than this OSMnx edge can have a distant
        # centroid while still overlapping. Fall back to the other direction:
        # does the candidate's own polyline pass close to this edge?
        centre = poly_local.mean(axis=0)
        gaps = np.fromiter(
            (point_to_polyline_m(centre, net.edge_local(c)) for c in candidates),
            dtype=np.float64, count=len(candidates),
        )
        keep = gaps <= tolerance_m
    if not keep.any():
        # Never return nothing for an edge whose way ID did match: the single
        # closest candidate is the best available answer.
        keep = gaps == gaps.min()
    return [c for c, k in zip(candidates, keep) if k]


def build_id_map(cfg: Config, graph=None, net: SumoNet | None = None,
                 force: bool = False, tolerance_m: float = 30.0) -> dict:
    """Join OSMnx edges to SUMO edges: way ID first, then geometry."""
    out = cfg.paths.id_map
    if out.exists() and not force:
        print(f"[map] reusing {out}", flush=True)
        return json.loads(out.read_text(encoding="utf-8"))

    graph = graph if graph is not None else build_osmnx_graph(cfg)
    net = net if net is not None else SumoNet(build_sumo_net(cfg))

    way_to_sumo, strategy = _way_index(net)
    print(f"[map] stage 1 strategy={strategy}: {len(way_to_sumo)} OSM ways "
          f"indexed over {len(net.edge_ids)} SUMO edges", flush=True)

    entries: list[dict] = []
    unmapped = 0
    candidates_total = 0
    kept_total = 0
    started = time.monotonic()
    for u, v, key, data in graph.edges(keys=True, data=True):
        ways = _osmids(data)
        candidates: list[str] = []
        for way in ways:
            candidates.extend(way_to_sumo.get(str(way), ()))
        candidates = sorted(set(candidates))
        if candidates:
            poly = _osmnx_edge_lonlat(graph, u, v, key, data)
            poly_local = net.to_local(poly) if len(poly) else poly
            sumo_edges = _refine(poly_local, candidates, net, tolerance_m)
        else:
            sumo_edges = []
            unmapped += 1
        candidates_total += len(candidates)
        kept_total += len(sumo_edges)
        entries.append({
            "u": int(u), "v": int(v), "key": int(key),
            "osmid": ways, "sumo_edges": sumo_edges,
        })

    mapped = len(entries) - unmapped
    payload = {
        "strategy": strategy,
        "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "refinement": {
            "tolerance_m": tolerance_m,
            "candidates_from_way_id": candidates_total,
            "kept_after_geometry": kept_total,
        },
        "counts": {
            "osmnx_edges": len(entries),
            "sumo_edges": len(net.edge_ids),
            "osm_ways_indexed": len(way_to_sumo),
            "mapped": mapped,
            "unmapped": unmapped,
        },
        "osm_way_to_sumo": way_to_sumo,
        "osmnx_edge_to_sumo": entries,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload), encoding="utf-8")
    print(f"[map] stage 2 geometry: {candidates_total} candidates -> "
          f"{kept_total} kept ({kept_total / max(candidates_total, 1):.1%})",
          flush=True)
    print(f"[map] {mapped}/{len(entries)} OSMnx edges mapped "
          f"({100 * mapped / max(len(entries), 1):.1f}%) in "
          f"{time.monotonic() - started:.0f}s, wrote {out}", flush=True)
    return payload


def verify_id_map(cfg: Config, id_map: dict, graph, net: SumoNet,
                  sample: int = 400, tolerance_m: float = 30.0) -> dict:
    """Assert the join agrees with geometry, by metrics it was not built on.

    Two things get measured, in the opposite direction from the refinement
    (which compared SUMO centroids against OSMnx polylines):

    `midpoint_to_mapped` -- distance from each OSMnx edge's own midpoint to
    the nearest of its mapped SUMO shapes. If the join put an edge on the
    wrong road this is hundreds of metres; if it is right it is a lane offset.
    This is the hard gate.

    `shape_gap` -- Hausdorff-style worst-point distance from the OSMnx
    polyline to the union of its mapped SUMO shapes, i.e. does the mapping
    cover the whole edge and not just one end.

    Also reported, deliberately *not* as a hard gate: how often the SUMO edge
    that pure geometry calls nearest to the midpoint is inside the mapping.
    That number is depressed by real 2D ambiguity rather than by join errors
    -- SUMO offsets each lane's shape from the OSM centreline, and Bangalore
    stacks flyovers directly over surface roads, so the closest geometry to a
    point genuinely lying on road A is sometimes a lane of crossing road B.
    The floor below is loose enough to tolerate that and still fail a join
    that has come loose from geometry altogether.
    """
    entries = [e for e in id_map["osmnx_edge_to_sumo"] if e["sumo_edges"]]
    if not entries:
        raise AssertionError("no OSMnx edge mapped to any SUMO edge")

    stride = max(len(entries) // sample, 1)
    checked = entries[::stride][:sample]

    hits = 0
    tested = 0
    shape_gaps: list[float] = []
    midpoint_gaps: list[float] = []
    worst: list[dict] = []
    for entry in checked:
        data = graph.get_edge_data(entry["u"], entry["v"], entry["key"])
        poly = _osmnx_edge_lonlat(graph, entry["u"], entry["v"], entry["key"], data)
        if len(poly) < 1:
            continue
        poly_local = net.to_local(poly)
        mid = poly[len(poly) // 2]
        tested += 1

        mapped_pts = np.vstack([net.edge_local(e) for e in entry["sumo_edges"]])
        mid_local = net.to_local(np.asarray([mid]))[0]
        mid_gap = min(
            point_to_polyline_m(mid_local, net.edge_local(e))
            for e in entry["sumo_edges"]
        )
        midpoint_gaps.append(mid_gap)
        shape_gaps.append(max(
            point_to_polyline_m(p, mapped_pts) for p in poly_local
        ))

        nearest, _gap = net.nearest_edge(float(mid[0]), float(mid[1]))
        if nearest in entry["sumo_edges"]:
            hits += 1
        elif len(worst) < 5:
            worst.append({"entry": {k: entry[k] for k in ("u", "v", "key")},
                          "osmid": entry["osmid"],
                          "nearest_by_geometry": nearest,
                          "mapped": entry["sumo_edges"][:6],
                          "midpoint_gap_m": round(mid_gap, 2)})

    if not tested:
        raise AssertionError("no sampled OSMnx edge had usable geometry")

    def pct(values: list[float], q: float) -> float:
        return round(sorted(values)[int(q * (len(values) - 1))], 2)

    stats = {
        "sampled": tested,
        "midpoint_gap_median_m": round(statistics.median(midpoint_gaps), 2),
        "midpoint_gap_p90_m": pct(midpoint_gaps, 0.9),
        "midpoint_gap_max_m": round(max(midpoint_gaps), 2),
        "shape_gap_median_m": round(statistics.median(shape_gaps), 2),
        "shape_gap_p90_m": pct(shape_gaps, 0.9),
        "shape_gap_max_m": round(max(shape_gaps), 2),
        "geometry_only_agreement": round(hits / tested, 4),
        "tolerance_m": tolerance_m,
        "examples_where_geometry_disagrees": worst,
    }
    print(f"[map] verify on {tested} edges", flush=True)
    print(f"[map]   midpoint -> nearest mapped SUMO edge: median "
          f"{stats['midpoint_gap_median_m']} m, p90 "
          f"{stats['midpoint_gap_p90_m']} m, max "
          f"{stats['midpoint_gap_max_m']} m", flush=True)
    print(f"[map]   worst-point shape coverage: median "
          f"{stats['shape_gap_median_m']} m, p90 {stats['shape_gap_p90_m']} m, "
          f"max {stats['shape_gap_max_m']} m", flush=True)
    print(f"[map]   pure-geometry nearest edge is in the mapping for "
          f"{hits}/{tested} ({stats['geometry_only_agreement']:.1%})",
          flush=True)

    assert stats["midpoint_gap_median_m"] <= tolerance_m, (
        f"median distance from an OSMnx edge's midpoint to its own mapped "
        f"SUMO shapes is {stats['midpoint_gap_median_m']} m (> {tolerance_m} m)"
        " -- the join is putting edges on the wrong roads"
    )
    assert stats["midpoint_gap_p90_m"] <= 4 * tolerance_m, (
        f"p90 midpoint gap {stats['midpoint_gap_p90_m']} m exceeds "
        f"{4 * tolerance_m} m"
    )
    assert stats["shape_gap_p90_m"] <= 10 * tolerance_m, (
        f"p90 worst-point gap {stats['shape_gap_p90_m']} m between OSMnx edges "
        f"and their mapped SUMO shapes exceeds {10 * tolerance_m} m"
    )
    assert stats["geometry_only_agreement"] >= 0.6, (
        f"pure geometry agrees with the ID join on only "
        f"{stats['geometry_only_agreement']:.1%} of sampled edges, which is "
        f"below what lane offsets and flyovers alone explain. examples: {worst}"
    )
    return stats


def verify_determinism(cfg: Config, graph, net: SumoNet,
                       tolerance_m: float = 30.0) -> dict:
    """Rebuild the map and require byte-identical output.

    "Deterministic" is an explicit Phase 1 exit criterion, and the join walks
    Python sets and dicts along the way -- worth proving, not assuming.
    """
    first = build_id_map(cfg, graph=graph, net=net, force=True,
                         tolerance_m=tolerance_m)
    second = build_id_map(cfg, graph=graph, net=net, force=True,
                          tolerance_m=tolerance_m)
    a = json.dumps(first["osmnx_edge_to_sumo"], sort_keys=True)
    b = json.dumps(second["osmnx_edge_to_sumo"], sort_keys=True)
    assert a == b, "the OSMnx<->SUMO mapping is not reproducible across runs"
    print("[map] rebuild is byte-identical: mapping is deterministic", flush=True)
    return {"deterministic": True, "entries": len(first["osmnx_edge_to_sumo"])}


def route_endpoint_edges(cfg: Config, net: SumoNet) -> dict:
    """The SUMO edges nearest the configured origin and destination.

    Returns plain data, not the network, so the caller can drop its several
    hundred MB before `verify_routable` spawns duarouter -- which peaks at
    4.4 GB on this network and will not co-exist with it in a 7.7 GiB VM.
    """
    r = cfg.route
    from_edge, from_gap = net.nearest_edge(r.origin.lon, r.origin.lat)
    to_edge, to_gap = net.nearest_edge(r.destination.lon, r.destination.lat)
    print(f"[route] origin -> SUMO edge {from_edge} ({from_gap:.1f} m away)",
          flush=True)
    print(f"[route] destination -> SUMO edge {to_edge} ({to_gap:.1f} m away)",
          flush=True)
    return {
        "from_edge": from_edge,
        "to_edge": to_edge,
        "origin_gap_m": round(from_gap, 1),
        "destination_gap_m": round(to_gap, 1),
        "lengths": {e: float(net.lengths[i])
                    for i, e in enumerate(net.edge_ids)},
    }


def verify_routable(cfg: Config, endpoints: dict | None = None,
                    net: SumoNet | None = None) -> dict:
    """Ask SUMO itself whether A->B is drivable on the generated network.

    "No fatal topology errors" is not the same as routable: netconvert exits 0
    on networks whose components are disconnected. duarouter answering with an
    actual edge list between the origin and destination edges is the decisive
    check, and it is the same query Phase 2's traffic demand will make.

    Pass `endpoints` from `route_endpoint_edges` and drop the SumoNet first;
    passing `net` here is the convenient path, not the memory-safe one.
    """
    if endpoints is None:
        net = net or SumoNet(build_sumo_net(cfg))
        endpoints = route_endpoint_edges(cfg, net)
    from_edge = endpoints["from_edge"]
    to_edge = endpoints["to_edge"]
    lengths = endpoints["lengths"]

    trips = cfg.paths.cache_dir / "phase1_check.trips.xml"
    routes = cfg.paths.cache_dir / "phase1_check.rou.xml"
    trips.write_text(
        "<routes>\n"
        f'  <trip id="phase1" depart="0" from="{from_edge}" to="{to_edge}"/>\n'
        "</routes>\n",
        encoding="utf-8",
    )

    duarouter = shutil.which("duarouter")
    if not duarouter:
        raise FileNotFoundError("duarouter not found; install sumo-tools")
    result = subprocess.run(
        [duarouter, "--net-file", str(cfg.paths.sumo_net),
         "--route-files", str(trips), "--output-file", str(routes),
         "--no-warnings", "true"],
        capture_output=True, text=True, check=False,
    )

    # duarouter needs 4.4 GB to load this network, which in a 7.7 GiB WSL2 VM
    # is enough to destabilise the whole guest. A negative return code means
    # it was signalled, not that the network is unroutable -- that is an
    # environment limit, and the caller has the in-process BFS for the actual
    # connectivity claim. Reporting it as "not routable" would be a lie.
    if result.returncode is not None and result.returncode < 0:
        print(f"[route] duarouter was killed by signal "
              f"{-result.returncode} (it peaks at ~4.4 GB on this network); "
              f"falling back to the in-process connectivity check", flush=True)
        return {
            "from_edge": from_edge,
            "to_edge": to_edge,
            "origin_gap_m": endpoints["origin_gap_m"],
            "destination_gap_m": endpoints["destination_gap_m"],
            "duarouter": f"killed by signal {-result.returncode}",
            "straight_line_km": round(cfg.route.separation_m / 1000, 2),
        }

    if result.returncode != 0 or not routes.exists():
        raise AssertionError(
            f"duarouter could not route {from_edge} -> {to_edge} "
            f"(exit {result.returncode}); the network is not routable end to end\n"
            f"{result.stderr.strip()[-2000:]}"
        )

    route_elem = ET.parse(routes).getroot().find(".//route")
    if route_elem is None or not route_elem.get("edges"):
        raise AssertionError(
            f"duarouter produced no route between {from_edge} and {to_edge}; "
            "origin and destination are in disconnected components"
        )
    edges = route_elem.get("edges", "").split()
    length = sum(lengths.get(e, 0.0) for e in edges)
    print(f"[route] duarouter route: {len(edges)} edges, {length / 1000:.2f} km",
          flush=True)
    return {
        "from_edge": from_edge,
        "to_edge": to_edge,
        "origin_gap_m": endpoints["origin_gap_m"],
        "destination_gap_m": endpoints["destination_gap_m"],
        "duarouter": "ok",
        "route_edges": len(edges),
        "route_km": round(length / 1000, 2),
        "straight_line_km": round(cfg.route.separation_m / 1000, 2),
    }


def main() -> None:
    cfg = load_config()
    graph = build_osmnx_graph(cfg)
    net = SumoNet(build_sumo_net(cfg), with_connections=True)
    id_map = build_id_map(cfg, graph=graph, net=net, force=True)
    verify_id_map(cfg, id_map, graph, net)
    orig, dest = route_endpoints(cfg, graph)
    print(f"[route] OSMnx origin node {orig} -> destination node {dest}",
          flush=True)
    endpoints = route_endpoint_edges(cfg, net)
    connected, explored = net.reachable(endpoints["from_edge"],
                                        endpoints["to_edge"])
    print(f"[route] BFS over connections: destination reachable={connected} "
          f"after exploring {explored:,} edges", flush=True)
    assert connected, "origin and destination are in disconnected components"
    del graph, net
    gc.collect()
    # See check_phase1.check_sumo_net: duarouter wants ~4.4 GB on this
    # network and will take a small WSL2 guest down with it.
    if os.environ.get("DIPTERAN_DUAROUTER") == "1":
        verify_routable(cfg, endpoints=endpoints)
    print("\nroads self-check OK")


if __name__ == "__main__":
    main()
