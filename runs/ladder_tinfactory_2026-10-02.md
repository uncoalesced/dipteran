# Tin Factory ladder + root-cause fixes, 2026-10-02

**Result: 3.2 km rung ARRIVED (19.8 m from goal, ~17 sim-min).** First arrival
on the Tin Factory approach, the stretch every Silk Board run since 09-27b
failed on.

Recorded live through the page (`main.py`, `DIPTERAN_LIVE=1`), one route per
server, frames captured off `/ws`. Arrival is the server's `ARRIVAL_RADIUS_M`
= 20 m, stricter than `run_route.py`'s 200 m.

## Why the fly kept looping

1. **Connectome mapping: correct.** Audited against FlyWire v783 (current):
   every cluster resolves to the right cells; DNa02 is the true L/R pair;
   PFL3 -> DNa02 is cleanly contralateral (258 / 238 synapses, 0
   ipsilateral -- the old "12 synapses" comments were the cell count);
   DNa02 inputs are symmetric (12,268 vs 13,159 synapses). The brain's
   heading-error sweep is monotonic both ways (-0.063 at 75 deg left,
   +0.028 at 75 deg right).
2. **Potholes flipped right turns left.** `mechano_bristle` drives 1,417
   neurons and was held on for a whole potholed edge. Probe: dead-ahead
   steer -0.016 -> -0.085; route 60 deg right +0.011 -> -0.014. Live
   (rung 1 at Silk Board) right-side errors of 45-75 deg averaged -0.02 to
   -0.03 steer. Fix: one 0.5 s jolt per potholed edge
   (`World._pothole_jolt`).
3. **Narrow forks were decided by lane distance.** South of Tin Factory the
   route leaves `1298941842` onto `1370245924`; the fly took
   `172938155#0` every lap (both branches start metres apart). Fix: where
   the route's branch is within `lane_keep_m` (8 m), the snap prefers it
   (`World._connected_snap`); wider forks are still the fly's call.

## Rungs

| rung | code | result | closest | driven | heading err med | sign agree |
|---|---|---|---|---|---|---|
| Silk Board -> 1 km (915 m crow, 2,963 m road) | `5162418` | cap 15 min | 510 m | 3,143 m | 17.2 | 0.47 |
| Silk Board -> 1 km | + pothole fix | stopped, congestion (528 m in 8 min) | -- | 528 m | 5.5 | 0.52 |
| 1 km -> Tin Factory (1,238 m road) | + pothole fix | cap 15 min, goal loop | 325 m | 3,125 m | 13.0 | 0.72 |
| 1 km -> Tin Factory | + fork fix | cap 15 min, on final edge | 114 m | 3,114 m | 15.0 | 0.65 |
| **3.2 km -> Tin Factory (3,378 m road)** | + fork fix | **ARRIVED 19.8 m** | -- | 3,733 m | 10.8 | 0.59 |

Start points were taken along OSMnx's shortest Silk Board -> Tin Factory
path: 1 km `12.99691,77.66233`, 3.2 km `12.98722,77.64852`. Silk Board
itself was dropped for the short rungs at Joel's call: the junction's
congestion ate the time budget.

## Also fixed

- `index.html`: a reload during a live run showed only the route picker;
  the page now attaches to the running simulation.
- `.claude/launch.json` (local): the live server for the Claude browser pane.

Full Silk Board -> Tin Factory on this code: see the 2026-10-03 entry.
