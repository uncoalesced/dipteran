# Silk Board -> Tin Factory, 2026-10-03

**Result: ARRIVED, 19.9 m from Tin Factory** (`main.py` live run, code
`61a5e03`). First arrival on this route; best before was 324 m (09-28).

- Route: 16.0 km by road, 10.6 km crow-flies, 135 edges.
- Arrival: the server's `ARRIVAL_RADIUS_M` = 20 m.
- Recorded off `/ws` until the recorder's 2 h limit (t = 5,925 s, 3.0 km
  out and closing); the server ran on and logged `[sim] arrived, 19.9 m
  from goal`. Arrival time ~2 h sim (not captured to the second).
- To t = 5,925 s: 20.5 km driven, heading error median 14.7 deg, sign
  agreement 0.67, four dead ends turned round (`744783934`,
  `-800628086`, `1322857017` twice).

## Full runs on the way here

Each failure was traced to one junction or edge, fixed, given a unit test,
and the next run went further.

| code | outcome | cause | fix |
|---|---|---|---|
| `5162418` (10-02) | looped 5.7 km out | potholes flip right turns; narrow forks | `162fad9` |
| `162fad9` | 326 m, then pinned 871 m out | 55-deg right missed at speed; ramp ends 9 m short of successor | `a472f46` |
| `a472f46` | pinned 18 min in, 95 min | dead end `-800628086`, kerb steer holds heading | `02e69fb` |
| `02e69fb` | looped 3.9 km out | repeat fork past the 8 m window | `10cd9f5` |
| `10cd9f5` (5 km rung) | dead-end side streets | off-route dead-end branches legal | `f723e2e` |
| `f723e2e` | looped 8.3 km out, 34 laps | edge-end handover ignored the route | `61a5e03` |
| **`61a5e03`** | **ARRIVED 19.9 m** | | |

Short rungs on the Tin Factory end (see `ladder_tinfactory_2026-10-02.md`):
3.2 km arrived on `162fad9`, 5 km arrived on `f723e2e`.

## Fixes (all in `dipteran/world.py`)

1. **Pothole jolt** -- one 0.5 s `mechano_bristle` pulse per potholed edge.
   Held, it shifted steer -0.07, more than the connectome's whole
   right-turn authority.
2. **Narrow forks follow the route** within `lane_keep_m`; the window
   widens by `lane_keep_m` per repeat visit to the edge (max 4x).
3. **Ease off for turns** -- speed x max(0.4, cos heading error).
4. **Edge-end handover** -- a fly clamped at its edge's end moves onto a
   successor up to 2x `lane_keep_m` away, the route's first.
5. **Dead end** -- clamped at the end of an edge with no successors, turn
   round onto the reverse carriageway.
6. **Off-route dead-end branches** are not snap targets.

Connectome mapping was audited first (FlyWire v783) and is correct; see the
ladder doc.

## Known limits

- Right turns have about half the left-turn authority (probe: +0.028 vs
  -0.063 at 75 deg). Fixes 2-4 work around it in the vehicle layer; the
  network's own asymmetry is untouched.
- One arrival. Run-to-run variance is high (different traps each run);
  needs repeats before calling it reliable.
- The Silk Board start still detours ~20 min through the one-way system
  and a dead end before heading north.
