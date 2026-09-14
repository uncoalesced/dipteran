# Phase 2 report

What Phase 2 set out to do: turn the Phase 1 foundations into a running
system. A real `FlyBangaloreNavigator` over the full connectome, LIF
parameters calibrated rather than guessed, PAM/PPL1 reinforcement acting on
weights, SUMO background traffic, the hybrid motion model, a real telemetry
payload, and — pulled forward from Phase 3 by decision — the neural-activity
HUD that consumes it.

Every figure below is transcribed from a command's output or from an evidence
file under `cache/`. Where a number was wrong in an earlier draft of this
work, the correction is kept rather than smoothed over: the wrong numbers are
how several of the real defects were found.

---

## Environment as built

Unchanged from Phase 1 apart from two discoveries:

- `traci` and `sumolib` **ship with the apt SUMO** at `/usr/share/sumo/tools`.
  No pip install is needed; only `SUMO_HOME` has to be set. Phase 1 never
  imported them, so this was not known.
- The `.wslconfig` change from Phase 1 took effect. The guest reports
  **10,952 MB** total, against the 7.7 GiB that OOM-killed two Phase 1 runs.

SUMO 1.25.0, Python 3.12.14, `torch 2.11.0+cu128`, CUDA available, RTX 5050.

---

## The grilling round: does the locked sync actually hold?

This was named as the thing to settle *before* writing Phase 2 code — the
SUMO 10 Hz / SNN 50 Hz cadence was decided in planning and had never been run
against live `traci` with traffic demand. `scripts/bench_sync.py` runs it in
three stages, each in its own process, because Phase 1 learned that two heavy
stages in one process get OOM-killed.

**The framing in `PHASE1_REPORT.md` was backwards.** It asks whether SUMO can
fit "in the remaining ~10 ms" of a 20 ms tick. But SUMO steps once per *five*
SNN ticks, so the budget is a 100 ms frame, not a 20 ms one:

| Term | Measured |
|---|---|
| 5 SNN ticks (10 LIF sub-ticks each) | 56.74 ms |
| 1 SUMO step, 235 background vehicles | 3.50 ms |
| traci context read + control | 0.45 ms |
| **Whole frame** (budget 100 ms) | **61.03 ms median, 67.59 ms p95** |
| Real-time factor | **1.64× median, 1.48× p95** |

SUMO uses 3.5% of the frame; the SNN uses 57%. SUMO was never the half at
risk. Isolated, the SNN tick is 11.21 ms median / 14.11 ms p95 against its
20 ms budget, and a bare SUMO step is 3.07 ms median / 6.07 ms p95 / 15.34 ms
max with 235 vehicles.

### Memory: the constraint is ordering, not capacity

| | Measured |
|---|---|
| Host total (WSL2) | 10,952 MB |
| Resident `sumo` on the 862 MB network | **7,454 MB** |
| SNN side, steady state | 1,087 MB |
| Connectome build, transient peak | 4,344 MB |
| **Concurrent peak** | **8,506 MB** |
| Headroom | **2,446 MB** |
| VRAM peak | 174.5 MB of 7.93 GiB |

The build peak and the resident SUMO do not fit together — 4,344 + 7,454
exceeds the ceiling — but they do not have to. The build peak is transient and
precedes SUMO's launch. `main.py` and `scripts/run_route.py` both build W,
free the pandas intermediates, and only then start SUMO. Reversing that order
OOMs, and the ordering is documented in both.

**A reporting defect, corrected.** The bench's first version reported
`combined_peak_rss_mb: 11229.5` on a 10,952 MB host — an impossible number,
produced by summing two `VmHWM` high-water marks that never coexisted. It now
samples both processes' live RSS every frame and reports the true concurrent
maximum.

### What the bench under-measured, and the two wrong guesses on the way

The bench's `tick()` is a bare LIF loop: sparse matmul, threshold, reset. The
real navigator also maintains an eligibility trace and a plastic weight
overlay, and the bench models neither. The first real route run managed
**2.65 frames/s — 0.27× real time**, not 1.64×.

Two guesses at the cause, both wrong, both discarded on measurement:

1. *"It must be the per-vehicle `traci` calls."* `_perceive()` did call
   `traci.simulation.convertGeo()` once per nearby vehicle. Fixing that
   changed the frame rate not at all — because with only ~2 neighbours in the
   context radius it was two calls, not twenty.
2. *"Then it must be `convertRoad` on a 385k-edge network."* Also wrong.

Per-phase instrumentation settled it in one run, and the answer was not in
`traci` at all:

| phase | before | after |
|---|---|---|
| SNN, 5 ticks | **364.29 ms** | **66.70 ms** |
| `apply` (traci writes) | 10.03 ms | 9.94 ms |
| `sense` (traci reads) | 6.50 ms | 1.24 ms |
| `step_sumo` | 3.62 ms | 2.65 ms |
| **frame total** | ~384 ms | **80.53 ms** |

The cost was the plasticity machinery, and specifically its **dense** storage.
`trace` and `plastic` were `(n_motor × n)` tensors — 356 × 138,639 floats,
**198 MB each** — and the sub-tick loop touched both twice per sub-tick. That
is ~1.5 G element-operations per tick, against a sparse matmul over 15.1 M
non-zeros.

Two fixes:

- **Maintain the trace once per tick, not once per sub-tick.** Tick-level
  rates are the right resolution regardless: `learning_rate` is 1e-4 and the
  rule is designed to need ~1,000 ticks to move anything, so sub-tick
  precision is detail the update can never act on. 364 → 196 ms.
- **Store the overlay sparsely.** The dense form is ~99.9% structural zeros,
  because mean in-degree in this connectome is ~109. The motor rows hold
  **116,704** actual synapses against 49.4 M dense cells — a 423× reduction.
  196 → 66.7 ms.

Steady-state frame cost is now **80.53 ms against the 100 ms budget**, roughly
**1.24× real time** — below the bench's 1.64×, as it should be, since the
bench frame does less work.

**A second reporting defect, same family as the RSS one.** `frames_per_s`
initially divided frames by total wall time, which includes the connectome
build and SUMO's 75–130 s parse of the network. On a short run that is most of
the elapsed time, and it understated the steady-state rate by about 40%. The
runner now times the loop separately from startup and reports both.

---

## LIF calibration: measured, not chosen

`PHASE1_REPORT.md` warned that `PLAN.md` §5's `v_thresh = 1.0` "will not
behave" against aggregate synapse counts reaching 2,405 with in-degree running
to hundreds. It does not.

The fix is to normalise each row by its own total input weight, which turns
synaptic drive into "what fraction of my input is firing" and leaves exactly
one excitability knob, `snn.gain`. That knob was swept:

| gain | network Hz | motor Hz | |
|---|---|---|---|
| 1 | 0.54 | 0.00 | motors silent |
| 2 | 1.63 | 0.30 | |
| 4 | 3.43 | 0.30 | |
| 8 | 7.41 | 1.10 | |
| **16** | **33.54** | **17.66** | **selected** |
| 32 | 55.41 | 50.96 | |
| 64 | 76.24 | 82.16 | |
| 128 | 83.63 | 131.32 | |
| 256 | 81.43 | 134.43 | saturating |

Reproduce with `python -m dipteran.navigator`; the table is written to
`cache/lif_calibration.json`.

**The selection rule had to be corrected twice.** The first version measured
the whole-population mean while clamping sensory rows at `i_ext = 1.0` against
`v_thresh = 1.0` — so those neurons fired every sub-tick regardless of gain,
and the reported rate barely moved across a 64× sweep. It was measuring its
own input clamp. Fixed by dropping the sensory drive below threshold and
reporting the *network* rate with clamped rows excluded.

The second correction is the `motor Hz` column, and it is an engineering
constraint rather than a biological preference. A tick samples 10 sub-ticks at
2 ms, so a motor population firing at *r* Hz contributes *r*/50 spikes to a
tick's rate estimate. Below about 10 Hz that estimate is zero on almost every
tick, the steering difference is identically zero, and **the vehicle does not
turn no matter how well the rest of the network is tuned**. The rule now
requires 10 Hz at the motors and picks the *lowest* gain that clears it, since
gain scales the anatomy and the smallest working value distorts it least.
Left to the earlier rule, the sweep selected gain 4.0 — a network at a
textbook-plausible 3.43 Hz driving motors at 0.30 Hz, which is a parked car.

---

## Monoamines no longer act as fast excitation

`PLAN.md` §11 lists this as an open defect: `NT_SIGN` signs dopamine,
serotonin and octopamine `+1` as a placeholder, "wrong as biology and only
harmless while PAM/PPL1 are unwired".

**8,316 monoamine rows** (5,909 dopaminergic, 2,282 serotonergic, 216
octopaminergic) are now dropped from the fast current path entirely.
Neuromodulators do not drive their targets on a 20 ms timescale; they change
how much other synapses count for. Dopamine acts on weights instead, through
the three-factor rule.

---

## Reinforcement: the update was 500× the anatomy

The three-factor rule is what `PLAN.md` §4 asks for — an eligibility trace of
recent (motor, presynaptic) coincidence, gated by PAM minus PPL1, no
backpropagation. The first implementation was catastrophically mis-scaled, and
only a long run exposed it.

`plastic` holds row-normalised weights, so a typical entry is ~1e-4. The
eligibility trace saturates at `1/(1 − trace_decay)` = 5.0. An **absolute**
update of `learning_rate × dopamine × trace` at `learning_rate = 0.01` is
therefore 0.05 per tick — about **500× the synapse it was modifying**.

A route run caught it within 70 seconds:

```
[route]  53 frames  sim  5s  speed 3.32 m/s  drift 9.56e-03
[route] 388 frames  sim 39s  speed 0.00 m/s  drift 5.98e-01
[route] 722 frames  sim 72s  speed 0.00 m/s  drift 6.01e-01
```

Relative drift went 9.56e-03 → 6.01e-01 in 40 seconds, pinned against the
clamp, and the vehicle **flatlined at 0.00 m/s permanently from frame 388**.
A 12-second smoke test cannot show this; it had passed cleanly.

Three changes:

1. **Multiplicative update**, scaled by each synapse's own anatomical weight.
   This also gets the biology right — you cannot strengthen a synapse that
   does not exist.
2. **`learning_rate = 1e-4`**, capping drift at 0.05% of baseline per tick, so
   meaningful adaptation needs ~1,000 ticks (20 s) of sustained dopamine
   rather than ten.
3. **Homeostatic return** toward the anatomy every tick, dopamine or not.
   Without it a sustained punishment — a long stall, which is the normal case
   in Bangalore traffic — depresses the very motor pathway the fly needs to
   escape, and the stall becomes self-reinforcing and permanent.

Bounds are `plastic0 ± |plastic0| × plastic_span`. The sign-symmetric form
matters: **6,071,458 of the 15,091,983 pairs are inhibitory**, and a
`clamp(min=w×0.2, max=w×2.5)` form inverts for negative *w* — `min` ends up
above `max` and torch silently returns the max. `tests/test_units.py` pins
this.

---

## The firing-rate / command boundary

Two separate bugs, one root cause: a DNg02 **population firing rate** was
being handed to code written for a **command** in [0, 1]. The measured rate
band is 0.05–0.24, because descending neurons sit in balanced excitation and
inhibition and fire sparsely — which is what a real brain does.

- **`sense()` and `rewards()` tested `throttle_cmd > 0.3`.** A 0.24-maximum
  rate can never clear 0.3, so the condition was unreachable: gridlock
  detection, the stall punishment, and the `GRIDLOCK STAGNATION` state were
  all dead code.
- **The motion model computed `target_speed = throttle × max_speed`**, pinning
  the vehicle at ~5% throttle — an accurate brain wired to a mis-scaled servo.

Raising `snn.sensory_gain` does **not** fix the second one. Swept against the
real sensory magnitudes:

| sensory_gain | 1 | 2 | 4 | 8 | 12 | 16 | 24 | 32 |
|---|---|---|---|---|---|---|---|---|
| mean throttle | 0.043 | 0.041 | 0.044 | 0.014 | 0.030 | 0.050 | 0.050 | 0.040 |
| panic fraction | 0 | 0 | 0 | **0.70** | **0.45** | 0 | 0 | 0 |

Throttle is flat across a 32× range, and past 8× the giant fiber starts firing
on 45–70% of ticks. The ceiling is the recurrent excitation/inhibition
balance, not the input.

The fix is an explicit transfer, `vehicle.throttle_scale` and
`vehicle.steer_scale` in `config.yaml`, and the scaled command now lives on
`VehicleState` so it cannot be confused with the rate again — `sense()` no
longer accepts it as an argument at all.

The transfer was calibrated against the ranges the *live* loop produces
(throttle 0.05–0.24, |steer| 0.0–0.40), not a static stimulus. At
`throttle_scale = 12.0` the vehicle sat pinned at `max_speed_ms` on nearly
every frame — a saturated controller, not a fast one. `4.0` leaves headroom.

---

## Corrections to Phase 1 work

### `SumoNet.nearest_edge` was a nearest-*vertex* query

The KD-tree indexes shape **points**, and the function returned whichever edge
owned the nearest one. For a query sitting mid-segment that is the wrong edge,
and the error scales with segment length. Measured against the synthetic
fixture:

```
query 2.0 m from the midpoint of edge 300#0
  true polyline gap to 300#0:  2.000 m
  true polyline gap to 200#0: 49.981 m
  nearest_edge returned:      200#0 at 49.981 m
```

`PHASE1_REPORT.md` explicitly tells Phase 2 to use this call to decide which
SUMO edge the vehicle is on. It now gathers every candidate edge with a vertex
inside `d0 + max_segment/2` — an exact bound, since the winning edge's closest
point lies on a segment no longer than `max_segment`, putting one of its
endpoints within that radius — and scores them with real point-to-polyline
distance.

**The fix moves no Phase 1 number.** Re-running `check_phase1.py` gives 99.2%
coverage, 7.16 m median midpoint gap, 7.74 m p90 and 68.8% pure-geometry
agreement — identical to Phase 1. On a network with ~7 m shape resolution the
vertex answer and the polyline answer nearly always coincide; the fix removes
a hazard rather than changing a statistic.

### `PLAN.md` §3's pothole pathway had no resolved cluster

§3 routes potholes to "chordotonal organ & campaniform sensilla". Those
afferents are in the ventral nerve cord, and FAFB is a brain-only volume —
precisely the reason §4's `b1`/`b2` throttle had to become DNg02 in Phase 1.
The same mistake was sitting unnoticed in the sensory column.

Added `mechano_bristle`, resolving to the bristle mechanosensory afferents
v783 does curate (`cell_class = mechanosensory`, 2,668 neurons; the head and
eye bristle sub-classes are 1,417 of them). The substitution is recorded as a
declared fallback, so it can never masquerade as the primary mapping. Cluster
count goes 14 → 15: **9 sensory, 6 motor**.

### Vehicle position lookup, superseded by Phase 1's own memory finding

`PHASE1_REPORT.md` recommends `SumoNet.nearest_edge` for "which SUMO edge is
the vehicle on". That advice predates the memory numbers. A resident
`SumoNet` is a second full copy of the 862 MB network inside our process, and
with `sumo` itself at 7,454 MB of a 10,952 MB ceiling there is no room for it.

`dipteran/world.py` parses no network at all. Every geometric question goes to
the already-running SUMO process over traci, at zero extra resident cost.
`SumoNet` remains what Phase 1 used it for: offline construction and
verification of the ID map, one stage at a time.

---

## The HUD

Pulled forward from Phase 3 by decision, because the design produced a backend
payload shape and a frontend component together.

`scripts/extract_neuropils.py` turns the annotation table into the point
cloud. Two corrections to the sketch it was based on, both found by reading
the actual file:

- **There are no `x`/`y`/`z` columns.** v783 has `pos_x`/`pos_y`/`pos_z`
  (139,248 rows, complete) and `soma_x`/`soma_y`/`soma_z` (118,104 rows,
  84.8%). The sketched `usecols=[..., "x", "y", "z"]` raises `KeyError` on the
  first line that touches the file.
- **Those coordinates are FAFB voxel indices, and the voxels are not cubes** —
  4 nm in x and y, **40 nm in z**. Plotting them raw squashes the brain
  tenfold along one axis.

Region tagging composes the two schemes that exist rather than picking one:
the 15 resolved clusters win wherever they apply, and a crude
`super_class`/`cell_class` heuristic fills visual texture for the rest.

| Region | sampled | of total |
|---|---|---|
| Optic lobes | 726 | 97,180 |
| Looming | 314 | 314 |
| Antennal lobes | 369 | 2,288 |
| Central complex | 51 | 51 |
| Johnston's organ | 366 | 390 |
| Mechanosensory | 726 | 2,278 |
| Descending steer | 6 | 6 |
| Descending throttle | 388 | 1,295 |
| Giant fiber | 2 | 2 |
| Dopaminergic | 323 | 323 |
| Unassigned scaffold | 363 | 35,121 |
| **Total** | **3,634 points, 86 KiB** | |

**A sampling defect, corrected.** The first version kept every precise neuron
outright. That is 14,777 neurons — mostly the 12,246 T4/T5 optic-flow cells —
which overran the 4,000-point budget and left **zero** scaffold points: named
clusters floating in empty space instead of inside a brain. Sampling is now
per-region with a quota, floored so that no region is ever sampled below 128.
The floor is what keeps 2-neuron DNa02 and 2-neuron DNp01 on screen at all,
and both have legend entries.

---

## The full route run

35 minutes of wall clock, **25,727 frames, 2,572.7 s (43 min) of simulated
time**, run rather than deferred to Phase 3. `scripts/run_route.py`, output in
`cache/route_run.json`.

### Performance, on the real loop

| | Measured |
|---|---|
| Loop time (excluding 76.8 s startup) | 2,023.2 s |
| Simulated time | 2,572.7 s |
| **Real-time factor** | **1.27×** |
| Frames per second | 12.72 |
| SNN, 5 ticks | 67.82 ms |
| `apply` (traci writes) | 9.80 ms |
| `sense` (traci reads) | 0.50 ms |
| `step_sumo` | 0.49 ms |

78.6 ms of a 100 ms frame. Below the bench's 1.64×, as expected — the bench
frame carries no plasticity and no world I/O — and comfortably above real
time.

### Stability: every question the long run existed to answer

- **Plastic drift is bounded and flat.** Sampled every two minutes across the
  whole run it oscillates in **1.74e-04 – 2.23e-04** and ends at **2.01e-04**.
  It does not grow. Against a permitted `plastic_span` of 0.5 that is 0.04% of
  the band. The pre-fix version reached 6.01e-01 and pinned there within 40
  seconds. `plastic_finite` is true over 116,704 synapses.
- **The vehicle never stalled.** `stall_peak_s` 0.1, speed median 3.82 m/s,
  max 5.36 m/s, sustained for 25,727 consecutive frames.
- **Stranding did not happen.** `off_network_frames` is **0**. The soft
  edge-pull held the free body within a median of 1.15 m and a maximum of
  2.39 m of a road for the entire run. PLAN.md §11's worry about the extract's
  239 strongly-connected components is answered by construction: a body that
  is never route-constrained cannot be trapped in a dead component.
- Circuits fire in context: 15 giant-fiber escapes, 9 looming events, 1
  gridlock frame.

### Behaviour: the fly orbits, and that is the honest result

It travelled **6,776 m** and closed **103.6 m**. Distance to goal went
10,587.5 m → 10,488.3 m.

`edges_visited` reached 8 by frame ~3,400 and **stayed at 8 for the remaining
22,000 frames**. State occupancy was IDLE DRIFT 24,518 (95.3%), ODOR TRACKING
1,193 (4.6%), PANIC SWERVE 15, GRIDLOCK STAGNATION 1.

The standing decision is that arrival is not a pass/fail gate, and it is not
being treated as one here. But the agreed Phase 2 metrics were *net distance
closed* and *trajectory entropy*, and this run scores poorly on both. It is
not wandering toward Point B taking wrong turns; it is **circling a fixed
8-edge neighbourhood**, with exploration collapsing after the first six
minutes.

The mechanism is visible in the numbers rather than mysterious.
`steer_abs_median` is 0.075 with a persistent sign; at `steer_scale` 2.5 that
is a command of 0.19, and against `max_turn_deg_s` 45 over a 0.1 s frame it is
a steady ~0.84° per frame in one direction. A constant turn rate is a circle.
The EPG compass drive is not strong enough relative to that bias to pull
heading back toward the goal — `compass_epg` activation sat around 0.12–0.21
in sampled telemetry.

So the correct reading is that **the plumbing is right and the control law is
not**. Sensory encoding reaches the brain, the brain drives the motors, the
motors move a vehicle that stays on the road and never diverges — and the
resulting closed-loop policy is a slow orbit. Correcting it is control tuning
against a substrate that is now stable, which is what PLAN.md's Phase 3
already owns under "directionally-correct steering response to sensory
input". Recorded as open rather than presented as success.

---

## Phase 1 exit criteria, re-verified

No regression is acceptable, and none occurred. After the `nearest_edge` fix:

| Check | Result |
|---|---|
| config | PASS 0.0s |
| scaffold | PASS 1.0s |
| osmnx | PASS 20.8s |
| sumo_net | PASS 25.1s |
| id_map | PASS 286.7s |
| connectome | PASS 24.6s |
| cell_ids | PASS 2.8s |

7 PASS, 0 FAIL, exit 0.

`main.py` keeps mock mode as its default specifically to preserve this: the
scaffold criterion starts `main.py` under a 45-second timeout, and the live
path needs ~28 s to build W plus 75–130 s for SUMO to parse the network. Live
mode is opt-in behind `DIPTERAN_LIVE=1`.

`tests/test_units.py`: **56/56**, with no dependency on the downloaded corpus.
`ruff check --select F,E9`: clean.

---

## What Phase 2 did not settle

- **The closed-loop policy is an orbit.** 6,776 m travelled, 103.6 m closed,
  8 edges for 22,000 frames. A persistent steering bias of ~0.84°/frame
  outweighs the compass. This is the single most important thing Phase 2 did
  not settle, and it is control tuning rather than plumbing: every layer
  underneath it now measures clean.
- **Reinforcement is wired but its behavioural effect is unmeasured.** The
  rule runs, is bounded, and no longer diverges — drift held in
  1.74e-04–2.23e-04 across 25,727 frames. Nothing yet shows the fly *learns*:
  that a punished junction becomes less likely on a second approach. That
  needs a controlled repeat-exposure test, and the orbit above means the fly
  barely encounters enough distinct junctions to make one meaningful.
- **Plasticity sits at the wrong anatomical site.** It is applied to synapses
  onto the descending motor clusters; biologically the modulated site is
  Kenyon cell → MBON in the mushroom body. Marked as a simplification in the
  code with its upgrade path.
- **`compute.min_syn_count` is still 0 and still unsettled.** 80% of the
  15.1 M pairs sit below FlyWire's own ≥5-synapse reliability bar. Deferred to
  Phase 3 deliberately: dropping them re-normalises every row and invalidates
  the calibrated gain and the motor transfer at once, so it is a sensitivity
  analysis against a stable baseline, not a config flip.
- **`traci` rerouting is a latency hazard.** `changeTarget` costs 7.8–9.5 ms
  median but was measured spiking to **670–1,825 ms** — 7 to 18 consecutive
  blown frames. Nothing in the current loop calls it per-frame, but any future
  in-loop rerouting must run off the critical path.
