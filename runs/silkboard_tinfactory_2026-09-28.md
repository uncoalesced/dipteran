# Silk Board -> Tin Factory, 2026-09-28

**Result: FAILED, did not arrive.** It timed out at the 90,000-frame cap
(150 sim-min). Its closest approach was **324 m** crow-flies from Tin
Factory, and it finished 355 m out. Arrival means within `ARRIVAL_M` =
200 m. This is the closest any run on this route has come; the previous
best was 785 m.

- Data: `silkboard_tinfactory_2026-09-28.json`, sha256 `ab693fcb86f533841994455441217ffe74e4bcc4297d06e917ef1d6ed833202f`
  (the same on the WSL copy and the E: copy)
- Code: `e360c89` (right-turn gain `vehicle.steer_right_gain` 1.84, on top
  of the `d7705b0` kerb steering)
- Unit tests: 88/88
- Command:

```
./.venv/bin/python scripts/run_route.py --origin 12.9177,77.6238 --dest 13.0039,77.6653 \
  --episodes 1 --episode-minutes 150 --max-minutes 165 --out runs/silkboard_tinfactory_2026-09-28.json
```

## Headline numbers

| metric | 09-27b (`d7705b0`, 120 min) | **09-28 (`e360c89`, 150 min)** |
|---|---|---|
| arrived | no | **no** |
| closest approach (crow-flies) | 785.1 m | **324.4 m** |
| closed crow-flies, of 10,587.5 m | 9,802.5 m | **10,263.1 m (96.9%)** |
| closed route, of 16,002 m | 14,195.6 m | **15,016.5 m (93.8%)** |
| best route distance left | 1,806.1 m | 985.2 m |
| travelled_m | 26,315 | 33,983 |
| edges_visited | 582 | 621 |
| to goal at frame ~47,000 | 3.08 km (circling) | 2.27 km |
| off-road median | 0.63 m | 0.46 m |
| heading error median / p90 | 18.2 / 55.8 deg | **13.4 / 48.0 deg** |
| steer sign agreement | 0.576 | **0.669** |
| panic frames / stall peak | 9 / 0.2 s | 5 / 0.3 s |

Invariant: closed route (15,016.5 m) ≤ travelled (33,983.1 m). There were
no `!!` lines and no routing failures. Of 11,116 no-route frames, 10,731
were junction transits.

## Shape of the run

```
 frames  to goal  travelled  edges
      1  10.59 km   0.00 km      1
   7789   9.85      2.98       115
  15770   8.97      6.08       171
  23661   8.00      9.09       282
  32085   5.19     12.55       321
  40130   3.40     15.57       413   no circle near 3 km this time
  48646   1.87     18.68       439
  56747   0.64     21.59       482   reaches the goal area at ~95 sim-min
  64796   0.55     24.55       553
  72812   0.60     27.50       595
  80834   0.38     30.54       608   closest stretch (324 m minimum)
  88857   0.61     33.55       621
  90000   0.355    33.98       621   cap
```

## What changed

The ~3 km circle from 09-27b is gone. The right-turn gain brought the fly
to the goal area at ~57,000 frames, where the 09-27b run was still 2.7 km
out. Heading error and sign agreement both improved, so the fly is
following its route target more closely.

## Why it failed

It reached 640 m at frame ~57,000, then spent the last **~55 sim-min**
(33,000 frames) circling 0.38–0.61 km from the goal. Edges visited nearly
stopped rising (608 -> 621 over the final 10,000 frames), so it was
driving the same loop over and over. The best *route* distance it reached
was 985 m, three times its best crow-flies distance (324 m). The goal
edge `362292202#1` sits on a one-way system near Tin Factory, and from
where the fly circles the legal path goes a long way round. It never took
that path.

This is the Silk Board version of item 4 ("goal swarming", 500 m route,
SESSION_2026-09-23 §8): the fly misses the approach, the one-way loop
carries it round, and it misses again.

## Next

- Trace the last 30 sim-min near Tin Factory with the scratch trace
  harness (`~/scratch/trace_route.py`), starting ~1 km out, to name the
  loop and the turn it keeps missing.
- Check how the goal resolves. `run_route.py` counts arrival at 200 m
  crow-flies, but the fly steers along the route to `362292202#1`. If that
  edge is only reachable the long way round, a goal edge picked closer to
  the fly's actual approach would count as arrival.
