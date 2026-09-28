# Runs

Each `*.json` file is `scripts/run_route.py --out` output, recorded as it
was written. The copy in WSL `~/dipteran/runs` is checked against this one
by sha256 before commit. Arrival means within `ARRIVAL_M` = 200 m
crow-flies of the goal. `scripts/regression_check.py` diffs a new run
against `scripts/baselines.json`.

Some runs have a write-up alongside them as a `.md` file with the same
name. The commit that added each run describes it.

## Routes

| name | origin -> dest | size |
|---|---|---|
| oneturn_500m | 12.93143,77.62261 -> 12.92760,77.62251 | 500 m, one turn |
| twisty_2400m | 12.92143,77.63576 -> 12.92581,77.63463 | 2.4 km of route, one-way loops, 502 m crow-flies |
| straight_silkboard | relaxed straight road from Silk Board | 566 m crow-flies |
| silkboard_tinfactory | 12.9177,77.6238 -> 13.0039,77.6653 | 16,002 m of route, 10,588 m crow-flies |

## Log

| run | result | episodes arrived | closest (m) | closed crow-flies (m) | travelled (m) |
|---|---|---|---|---|---|
| oneturn_500m_2026-09-19 | SUCCESS | 11/15 | 280.9 | 145.1 | 192.3 |
| oneturn_500m_2026-09-22pm | SUCCESS | 14/15 | 199.9 | 226.1 | 266.5 |
| oneturn_500m_2026-09-23 | SUCCESS | 13/15 | 271.1 | 154.9 | 361.2 |
| oneturn_500m_2026-09-24 | SUCCESS | 15/15 | 199.9 | 226.1 | 262.4 |
| oneturn_500m_2026-09-28 | SUCCESS | 15/15 | 200.0 | 226.0 | 268.5 |
| twisty_0209 | FAILED | 0/3 | 361.7 | 140.5 | 503.9 |
| twisty_2400m_1b52eb7 | FAILED | 0/3 | 497.7 | 4.5 | 1,000.4 |
| twisty_2400m_2026-09-22 | FAILED | 0/3 | 378.3 | 123.9 | 1,062.2 |
| twisty_2400m_2026-09-23 | FAILED | 0/3 | 494.6 | 7.6 | 839.3 |
| twisty_2400m_2026-09-24 | SUCCESS (partial) | 1/3 | 442.1 | 60.1 | 1,141.9 |
| twisty_2400m_2026-09-28 | FAILED (regressed from 1/3) | 0/3 | 442.9 | 59.3 | 1,166.5 |
| straight_silkboard_2026-09-23 | SUCCESS | 1/1 | 199.8 | 366.3 | 380.8 |
| silkboard_tinfactory_2026-09-23 | FAILED, pinned at Silk Board | 0/1 | 10,191.8 | 395.7 | 4,646.0 |
| silkboard_tinfactory_2026-09-27 | FAILED, pinned at Silk Board | 0/1 (wall cap) | 10,194.5 | 393.0 | 4,955.2 |
| [silkboard_tinfactory_2026-09-27b](silkboard_tinfactory_2026-09-27b.md) | FAILED, 785 m short | 0/1 | 785.1 | 9,802.5 | 26,314.7 |
| [silkboard_tinfactory_2026-09-28](silkboard_tinfactory_2026-09-28.md) | **FAILED, 324 m short at best, circling the goal** | 0/1 (150 min) | **324.4** (ended 355.4) | **10,263.1** | 33,983.1 |

"Closest" is `distance_min_m` and "closed" is `closed_m`, from the run JSON;
in multi-episode runs they come from the last episode.
