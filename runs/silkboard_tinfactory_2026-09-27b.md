# Silk Board -> Tin Factory, 2026-09-27 (run b)

**Result: FAILED, did not arrive.** It timed out at the 72,000-frame cap
(120 sim-min) **785 m** crow-flies from Tin Factory. Arrival means within
`ARRIVAL_M` = 200 m. This is still the best result on this route by far:
the two earlier runs never got closer than 10.19 km.

- Data: `silkboard_tinfactory_2026-09-27b.json`, sha256 `b0343bfc31d0e42dac7aff38a3dfb38475af71d8718b54fe8ccc8ae4225bf225`
  (the same on the WSL copy and the E: copy)
- Code: `d7705b0` on `dev` (kerb steering, one swerve per panic burst,
  windowed stall), on top of `c3bebdb`
- Unit tests: 87/87
- Command:

```
./.venv/bin/python scripts/run_route.py --origin 12.9177,77.6238 --dest 13.0039,77.6653 \
  --episodes 1 --episode-minutes 120 --max-minutes 130 --out runs/silkboard_tinfactory_2026-09-27b.json
```

## Headline numbers

| metric | 09-23 (`de4d8e2`) | 09-27 (`30fa530`) | **09-27b (`d7705b0`)** |
|---|---|---|---|
| arrived | no | no | **no** |
| closest approach (crow-flies) | 10,191.8 m | 10,194.5 m | **785.1 m** |
| closed crow-flies, of 10,587.5 m | 395.7 m | 393.0 m | **9,802.5 m (92.6%)** |
| closed route, of 16,002 m | 928.6 m | n/a | **14,195.6 m (88.7%)** |
| best route distance left | n/a | n/a | 1,806.1 m |
| travelled_m | 4,646 | 4,955 | 26,315 |
| edges_visited | 35 | 41 | **582** |
| pinned at Silk Board | from frame ~7,660 | from frame ~4,720 | **never** |
| off-road median | 2.02 m | 2.0 m | 0.63 m |
| heading error median / p90 | 7.6 / 20.5 deg | 8.0 / 21.6 deg | 18.2 / 55.8 deg |
| steer sign agreement | 0.761 | 0.764 | 0.576 |
| panic frames / stall peak | 4 / 0.0 s | 9 / 0.5 s | 9 / 0.2 s |
| realtime factor | 1.24 | 0.96 | 1.28 |

Invariant: closed route (14,195.6 m) ≤ travelled (26,314.7 m). There were
no `!!` lines and no routing failures. Of 8,476 no-route frames, 8,216
were junction transits (internal edges).

**Reading the heading figures.** Heading error and sign agreement look
worse than in the two earlier runs, but those runs are not a fair
baseline. A pinned fly sits pointing at its target, so it scores well on
heading. This run crossed 582 edges of real junctions. The fair comparison
is the 500 m route, where agreement was 0.489 to 0.664.

## Shape of the run

```
 frames  to goal  travelled  edges
      1  10.59 km   0.00 km      1
   5200   9.87      1.93        84   past the old pin at frame ~1,600
  11458  10.10      4.31       137   a detour loses ~0.2 km
  17743   9.16      6.65       231
  23311   8.02      8.61       320
  29780   5.81     11.10       328   fastest stretch, ~0.18 km per sim-min
  35254   4.42     13.11       397
  40911   3.74     15.19       469
  47366   3.08     17.64       522   circles near 3 km for ~25 sim-min
  53526   3.16     19.87       550
  59894   2.72     22.26       554
  66708   1.99     24.52       568
  71629   0.91     26.16       580
  72000   0.785    26.31       582   cap
```

## Why it failed

1. **Time.** At its final-9-minute rate (~0.14 km per sim-min) it needed
   about 4 more sim-minutes to get inside 200 m.
2. **Route efficiency.** It travelled 26.3 km to close 14.2 km of route,
   1.85 m driven per route metre. That came mostly from a ~25 sim-min
   circle near 3 km out (frames ~45,000–60,000), the same shape as the
   500 m route's "swarming" (the fly misses a turn and a one-way loop
   carries it round). The likely trigger is still the right-side decode
   dead zone at +30 to +60 deg. Kerb steering only acts on a fly the road
   is holding back, so it does not help a fly that is moving freely down
   the wrong branch.

## What freed the pin

At the end of `38098070#1` the fly faced 108 deg. The lane and its route
target lay ~46 deg to its right, the SNN steer read ±0.01 (the dead zone),
and the road pull cancelled ~90% of each step, so the heading never
changed. `World._kerb_steer` now turns a fly held back like that toward
the lane's tangent at 60 deg/s. A panic burst now swerves 45 deg once,
not 45 deg every frame; four frames of that was a 180 deg flip onto the
reverse carriageway.

## Next

- Run with `--episode-minutes 150` to see if it finishes.
- The right-side decode dead zone (+30 to +60 deg) now limits the whole
  route. It is behind both the ~3 km circle and item 4's goal swarming.
