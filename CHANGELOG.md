# Changelog

## Phase 2 — Finish the Backend, plus the HUD (2026-09-14)

The real `FlyBangaloreNavigator` over the full 138,639-neuron connectome,
driving a free-body vehicle through live SUMO traffic, streaming a telemetry
payload to a neural-activity HUD. Scope covers PLAN.md §10's Phase 2 list plus
the Three.js HUD, pulled forward from Phase 3 by decision.

Every figure here is quoted from a command's output or from an evidence file
under `cache/`, not written from memory. Narrative in
[PHASE2_REPORT.md](PHASE2_REPORT.md).

### Added

- `dipteran/navigator.py` — the LIF loop over the real adjacency matrix, the
  three-factor PAM/PPL1 reinforcement rule, motor decoding, state
  classification and the telemetry payload builder.
- `dipteran/world.py` — SUMO lifecycle over traci, sensory encoding for all
  six PLAN.md §3 pathways, the hybrid motion model, and the reward/punishment
  triggers. Parses no network of its own; every geometric question goes to the
  running SUMO process.
- `scripts/bench_sync.py` — the grilling round the sync design had never had:
  SNN half, SUMO half and both-resident, each in its own process.
- `scripts/run_route.py` — a full Point A → Point B run reporting geography,
  stalls and plastic-weight drift, not just "it did not crash".
- `scripts/extract_neuropils.py` — one-time ETL turning the annotation table
  into the HUD's tagged point cloud.
- `tests/test_units.py` — 56 assertions over the pure logic, with no
  dependency on the 1 GB corpus. Closes `GATES.md` G1/G2.
- `config.yaml` gained `snn:` and `vehicle:` blocks; `dipteran/config.py`
  gained the matching `Snn` and `Vehicle` models.
- `main.py` — live mode behind `DIPTERAN_LIVE=1`, and `GET /neuropils.json`.
  Mock mode stays the default so `check_phase1.py`'s 45-second scaffold
  criterion still passes without a GPU, SUMO, or the FlyWire download.
- `index.html` — Three.js point cloud, state banner, per-region legend with
  meters, and a GeoJSON layer for background traffic.

### Measured results

- **The sync design holds, and the risk was on the other side.** 5 SNN ticks
  + 1 SUMO step + control = **61.03 ms median / 67.59 ms p95** against a
  100 ms frame: **1.64× real time median, 1.48× at p95**. SUMO is **3.5 ms**
  of that frame and traci control **0.45 ms**; the SNN is 56.74 ms.
- **Host RAM fits, with the build ordered correctly.** Concurrent resident
  **8,506 MB of 10,952 MB, 2,446 MB spare**. The two peaks — 4,343 MB
  building the connectome, 7,454 MB for resident `sumo` — do not coexist, and
  do not have to: the build peak is transient and precedes SUMO's launch.
- LIF gain swept, not guessed. Network / motor firing rates by gain: 1 →
  0.54 / 0.0 Hz (motors silent), 8 → 7.41 / 1.1, **16 → 33.54 / 17.66**,
  32 → 55.41 / 50.96, 256 → 81.43 / 134.43. **16.0** is the lowest gain whose
  descending motor population clears the 10 Hz floor a 50 Hz control rate
  needs.
- 8,316 monoamine rows dropped from the fast current path.
- SUMO parses the 862 MB network in **75–127 s**; per-step cost at 235
  background vehicles is 3.07 ms median, 6.07 ms p95, 15.34 ms max.
- `changeTarget` reroutes cost 7.8–9.5 ms median but spike to **670–1,825 ms**
  — 7 to 18 blown frames. Any in-loop rerouting has to be off the critical
  path.
- Neuropil ETL: 139,248 annotated neurons, all with usable coordinates;
  14,777 tagged from resolved clusters; 3,634 sampled points (1,816 precise +
  1,818 scaffold), 86 KiB. Every HUD region survives sampling, including
  6-neuron `steer` and 2-neuron `giant`.
- **Full Point A → Point B run: 25,727 frames, 2,572.7 s (43 min) of
  simulated time, at 1.27× real time** — 78.6 ms of a 100 ms frame. Every
  stability question the run existed to answer came back clean: plastic drift
  held in **1.74e-04 – 2.23e-04** and did not grow, `off_network_frames` was
  **0** with the vehicle a median 1.15 m from a road, and it never stalled
  (`stall_peak_s` 0.1, speed median 3.82 m/s).
- **The closed-loop behaviour is an orbit, and that is recorded as unfixed,
  not as success.** The fly travelled 6,776 m and closed only 103.6 m,
  visiting 8 edges by frame ~3,400 and no more across the remaining 22,000. A
  persistent ~0.84°/frame steering bias outweighs the EPG compass, and a
  constant turn rate is a circle. Arrival was never a pass/fail gate, but net
  distance closed and trajectory entropy were the agreed metrics and this run
  scores poorly on both. The substrate underneath measures clean; the control
  law does not. Phase 3 owns it.
- All seven Phase 1 exit criteria re-verified green after every change here.
- `tests/test_units.py`: 56/56. `ruff check --select F,E9`: clean.
- HUD verified rendering in a browser: 3,634-point cloud, 10-region legend
  with meters, state banner, vitals and event log, no console errors, and
  1,503 map features painted.

### Fixed

- **`SumoNet.nearest_edge` was a nearest-*vertex* query, not a nearest-edge
  one.** It took whichever edge owned the closest shape point, so a query
  mid-segment was attributed to a neighbour: measured on a synthetic net, a
  point **2.000 m** from edge `300#0` came back as `200#0` at **49.981 m**.
  PHASE1_REPORT.md tells Phase 2 to use exactly this call to decide which edge
  the vehicle is on. It now resolves candidates by real point-to-polyline
  distance inside a bound of `d0 + max_segment/2`, which is exact rather than
  a guess. Phase 1's ID-map statistics are unchanged by the fix — 99.2%
  coverage, 7.16 m median and 68.8% geometry agreement all identical.
- **The reinforcement update was ~500× the anatomy it modified.** `plastic`
  holds row-normalised weights (~1e-4) while the eligibility trace saturates
  at `1/(1 − trace_decay)` = 5, so an absolute `learning_rate` of 0.01 applied
  0.05 per tick. A route run caught it: relative drift went **9.56e-03 →
  6.01e-01 in 40 seconds**, pinned against the clamp, and the vehicle
  flatlined at **0.00 m/s** permanently from frame 388. The update is now
  scaled by each synapse's own weight, `learning_rate` is 1e-4, and a
  homeostatic term returns slowly toward the anatomy every tick so a punished
  episode is survivable.
- **Firing rates were leaking into thresholds written for commands.**
  `sense()` and `rewards()` tested `throttle_cmd > 0.3`, but received
  `MotorOutput.throttle` — a DNg02 population firing rate whose measured band
  is 0.05–0.24. The condition was unreachable, so gridlock detection, the
  stall punishment and the `GRIDLOCK STAGNATION` state were all dead code.
  The scaled command now lives on `VehicleState` and `sense()` no longer
  accepts it as an argument.
- **The motion model treated that same firing rate as a [0,1] throttle**,
  pinning the vehicle at 5% throttle — an accurate brain wired to a mis-scaled
  servo. Raising `snn.sensory_gain` does not fix it: throttle holds at
  0.04–0.05 from 1× to 32×, and past 8× the giant fiber fires on 45–70% of
  ticks. The ceiling is the recurrent excitation/inhibition balance, not the
  input. Fixed with explicit transfer gains in `config.yaml`.
- **The plasticity overlay was stored dense, and cost 4.5× the whole frame
  budget.** `trace` and `plastic` were `(356 × 138,639)` tensors — 198 MB each
  — touched twice per sub-tick, ~1.5 G element-ops per tick. The real loop ran
  at **2.65 frames/s (0.27× real time)** against the bench's predicted 1.64×.
  Two guesses at the cause (per-vehicle `convertGeo`, then `convertRoad`) were
  both wrong and were discarded on measurement; per-phase instrumentation
  showed 364 ms of a ~384 ms frame was the SNN, not `traci`. Fixed by
  maintaining the trace once per tick rather than per sub-tick (364 → 196 ms)
  and storing the overlay sparsely — the motor rows hold **116,704** real
  synapses against 49.4 M dense cells, a 423× reduction (196 → **66.7 ms**).
  Steady-state frame is now **80.53 ms of the 100 ms budget**.
- **`frames_per_s` counted startup as run time**, dividing frames by total
  wall including the connectome build and SUMO's 75–130 s network parse. Same
  family of defect as the RSS sum. The runner now times the loop separately.

### Decided

- **All HUD work lands in Phase 2**, rather than split across Phase 2/3 as
  PLAN.md §10 had it. The brainstorm that produced the HUD design produced a
  backend telemetry shape and a frontend component together, and splitting
  them would have meant specifying a schema with nothing to render it.
- **The two region-tagging schemes compose.** The 15 clusters resolved from
  real FlyWire root IDs drive the glow and the state logic; a crude
  `super_class`/`cell_class` heuristic fills in visual texture for the ~120k
  neurons in no named cluster. Precision where it drives behaviour, cheap
  scaffolding everywhere else.
- **Arrival is not a pass/fail gate.** The premise is purgatory, not
  autonomous navigation. Tuning the odour plume or the compass until the agent
  ignores local topology and beelines to Tin Factory would make it an
  artificial potential-field pathfinder wearing SNN clothing. Phase 2's metric
  is net distance closed and realistic circuit activation under disruption.
- **`compute.min_syn_count` stays at 0** for Phase 2. Raising it to FlyWire's
  ≥5-synapse reliability threshold drops ~80% of the 15.1 M pairs, which
  changes every row normalisation and invalidates the calibrated gain and the
  motor transfer together. Treated as a Phase 3 sensitivity analysis against a
  full-fidelity Phase 2 baseline.
- **G1/G2 tests were pulled into Phase 2**; the remaining `GATES.md` defects
  (stale README and DATA_LICENSE figures, committed references to gitignored
  paths) stay in Phase 4, where that file already puts them.
- Reference images shared during the HUD brainstorm are **layout reference
  only**, from an unrelated third-party visualiser. None of their framing or
  copy carries over. HUD copy stays emoji-free.

### Corrected in PLAN.md

- **§3's pothole pathway had no resolved cluster.** It routes potholes to
  "chordotonal organ & campaniform sensilla", but those afferents are ventral
  nerve cord neurons and FAFB is a brain-only volume — the same reason §4's
  b1/b2 throttle had to become DNg02. Added `mechano_bristle`, which resolves
  to the bristle mechanosensory afferents v783 does curate, with the
  substitution recorded as a fallback so it can never masquerade as the
  primary mapping. The cluster count goes from 14 to 15.
- **§5's reference LIF engine does not survive the real matrix.** Its
  `v_thresh = 1.0` is meaningless against aggregate synapse counts reaching
  2,405; its per-neuron Python loop over external currents would dominate the
  tick; and it treats dopamine as fast excitation, which §11 already flags as
  wrong biology.
- **PHASE1_REPORT.md's tick-budget framing inverted the risk.** It asks
  whether SUMO fits "in the remaining ~10 ms" of a 20 ms tick, but SUMO steps
  once per *five* SNN ticks — the real frame is 100 ms, and SUMO uses 3.5% of
  it while the SNN uses 57%.
- **PHASE1_REPORT.md's advice to use `SumoNet.nearest_edge` for vehicle
  position is superseded by its own memory finding.** A resident `SumoNet` is
  a second copy of the 862 MB network inside our process, and with `sumo`
  itself at 7,454 MB there is no room. `world.py` asks the running SUMO
  process instead, at zero extra resident cost.

### Environment findings

- `traci` and `sumolib` ship with the apt SUMO at `/usr/share/sumo/tools`;
  no pip install is needed, only `SUMO_HOME`.
- `.wslconfig` at `memory=11GB` took effect: the guest reports 10,952 MB.
- The FlyWire annotation table has **no `x`/`y`/`z` columns**. It has
  `pos_x`/`pos_y`/`pos_z` (139,248 rows, complete) and `soma_x`/`soma_y`/
  `soma_z` (118,104 rows, 84.8%). Those are FAFB voxel indices at 4 nm in x
  and y but **40 nm in z**, so plotting them raw squashes the brain tenfold
  along one axis.

## Phase 1 — Basic Build & Backend Foundations (2026-09-12)

Every exit criterion in PLAN.md Phase 1 met and asserted by
`python check_phase1.py`, which writes its raw findings to
`cache/phase1_evidence.json`. Full narrative in
[PHASE1_REPORT.md](PHASE1_REPORT.md).

Phase 2 deliberately not started: no real `FlyBangaloreNavigator` dynamics,
no LIF calibration, no PAM/PPL1 reinforcement wiring, no SUMO/SNN sync loop.
The mock `main.py` streamer remains the running baseline.

### Added

- `config.yaml` + `dipteran/config.py` — Pydantic-validated route, compute and
  path configuration. No lat/lon constants anywhere in code. Rejects a
  `dist_m` smaller than the origin→destination separation (otherwise the OSM
  extract silently excludes Point B) and requires both endpoints inside the
  configured city envelope.
- `dipteran/roads.py` — one Overpass query into one local `.osm.xml`, fed to
  **both** `ox.graph_from_xml` and `netconvert` so the two graphs are
  guaranteed to share OSM way IDs. Contains the OSMnx↔SUMO join, the
  `SumoNet` parser, a KD-tree point query (`nearest_edge`), BFS reachability
  over SUMO connections, and all the verification functions.
- `dipteran/connectome.py` — fetches the FlyWire v783 archive, collapses the
  edge list to one signed weight per connected neuron pair, builds a
  `torch.sparse_csr_tensor` on the configured device, and resolves the named
  sensory/motor clusters to real FlyWire root IDs.
- `main.py` + `index.html` — README's mock scaffold materialized as real
  files. Port overridable via `DIPTERAN_PORT` so tests can bind a free one.
- `check_phase1.py` — asserts all seven criteria, one subprocess per check,
  and records the evidence each one actually saw.
- `requirements.txt` — direct dependencies pinned to verified versions, with
  the cu128 index requirement called out.
- `LICENSE` (MIT, code) and `DATA_LICENSE.md` (FlyWire treated as
  CC BY-NC 4.0, with the required Dorkenwald/Schlegel/Zheng citations).
- `PHASE1_REPORT.md`, this changelog.

### Measured results

| Criterion | Result |
|---|---|
| `config.yaml` validates | PASS — cross-checked against `index.html`'s `BANGALORE_BOUNDS` |
| `python main.py` serves the mock stack | PASS — 3,963-byte page, 5 live WebSocket frames, agent moving inside the bbox |
| OSMnx graph loads cleanly | PASS — 162,861 nodes, 406,970 edges; `has_path` A→B, 13.29 km vs 10.59 km straight line |
| `netconvert` output valid and routable | PASS — 385,141 edges, 793,563 connections, UTM zone 43; destination reachable after exploring 178,131 edges |
| OSM↔SUMO lookup deterministic, matches geometry | PASS — `origId` strategy, 99.2% coverage, **7.16 m** median midpoint gap, byte-identical rebuild |
| Connectome GPU-resident sparse CSR, VRAM logged | PASS — 138,639², 15,091,983 non-zeros, **116.5 MiB** fp32 of 7.93 GiB |
| Every cluster resolves to a valid row index | PASS — 14/14 to real root IDs, all rows round-trip |

Environment: WSL2 Ubuntu 26.04, SUMO 1.25.0, Python 3.12.14,
torch 2.11.0+cu128 on an RTX 5050 (sm_120), osmnx 2.1.1.

### Decided

- **OSMnx↔SUMO join is two-stage.** `netconvert --output.original-names true`
  writes the source OSM way ID onto every lane as an `origId` param, which
  indexes candidates; geometry then keeps the segments that actually overlap.
  The ID stage alone is far too coarse — one OSM way becomes hundreds of
  OSMnx edges *and* hundreds of SUMO edges, so a pure ID join maps every one
  of the former to every one of the latter (4,078,524 candidate pairs, p90
  centroid gap 130 m). Refinement cuts that to 1,424,648 kept and a 7.16 m
  median gap.
- **Adjacency orientation**: `W[i, j]` is presynaptic *j* onto postsynaptic
  *i* — rows postsynaptic — matching the `W @ v` in PLAN.md §5.
- **Transmitter signs per presynaptic neuron**, from the annotation table's
  consensus `top_nt` (Dale's principle), not per synapse. 40.2% of pairs come
  out inhibitory. Glutamate is treated as inhibitory (GluCl-α).
- **fp32 is the default dtype**, not fp16. It costs 29 MiB and 5% more time,
  and removes a real hazard: weights are raw synapse counts up to 2405, so
  fp16 products saturate to `inf` within a couple of LIF sub-ticks.
- **`compute.min_syn_count`** knob added, default 0. The median v783 pair has
  only two synapses and the FlyWire papers treat ≥5 as reliable, so ~80% of
  the matrix is below their own bar — but PLAN.md's standing decision is
  maximum fidelity.
- **No FlyWire account is required.** Task 4 assumed one gated the data; it
  does not. Both the v783 connectome (Zenodo 10676866, 852 MB) and the
  annotation table are openly downloadable.
- **Connectome file**: `proofread_connections_783.feather`, already aggregated
  per (pair, neuropil), rather than the 9.5 GB per-synapse table.
- **One subprocess per check.** CPython does not return freed arenas to the
  OS, so holding multi-gigabyte artefacts across checks left each later check
  starting near the memory ceiling.

### Corrected in PLAN.md

Two clusters PLAN.md named do not exist in FAFB, and are fixed rather than
quietly substituted:

- **`B1`/`B2`** (throttle) are basalar wing motor neurons in the ventral nerve
  cord. FAFB/FlyWire is a brain-only volume and cannot contain them. Throttle
  now resolves to **DNg02** (`DNg02_a`…`DNg02_h`, 25 descending neurons), the
  brain-side population that sets wingbeat amplitude. This is the one cluster
  resolving via a fallback, and it is flagged as such in the output.
- **`aPN`** (acoustic) has no label in v783. The release curates
  `cell_sub_class = auditory` instead: 390 sensory neurons, all `JO-A*`/`JO-B*`
  types — the sound-sensitive Johnston's organ subgroups, as distinct from the
  wind/gravity and grooming subgroups of the same organ. Renamed
  `acoustic_jo` so nobody mistakes them for projection neurons.

Also corrected: the brief's sizing estimate of ~2–3M non-zeros. The real
figure is **15,091,983** (54,492,922 synapses over 138,639 neurons, 3.61 per
pair) — about 5× more. Its conclusion held anyway: tens of MB, not gigabytes.

### Environment findings

- **Host RAM, not VRAM, is the binding constraint.** The GPU never exceeded
  118 MiB of 7.93 GiB. WSL2 defaults to half of host RAM, and that was not
  enough: `duarouter` peaks at 4.42 GB and the OSM↔SUMO join needs the 3–4 GB
  OSMnx graph and the parsed network simultaneously. Exceeding the limit
  crashed the WSL VM rather than producing a clean OOM —
  `Wsl/Service/E_UNEXPECTED` plus `dxgk` GPU-passthrough ioctl failures, a
  silent exit 1 with no traceback, and a different apparent bug each run.
  Mitigated by setting `memory=11GB`/`swap=4GB` in `%USERPROFILE%\.wslconfig`.
  With that raised, all seven checks pass in one invocation (~7 minutes,
  `id_map` 321.5 s of it).
- **`DIPTERAN_DATA_ROOT`** added to `dipteran/config.py`: relative paths in
  `config.yaml` resolve against it instead of the repo, so the gigabyte
  artefacts can live on ext4 rather than a 9p mount —
  `DIPTERAN_DATA_ROOT=/opt/dipteran python check_phase1.py`. Measurably
  faster, and recommended on WSL, but *not* a correctness fix: it came out of
  a misdiagnosis. The 11 GB run's log appeared frozen for minutes at the line
  the crashes had occurred on, which was 9p write buffering delaying the log,
  not another crash.
- **The tick benchmark needs warming and repetition.** `check_phase1.py` no
  longer derives a real-time factor from its single 10-iteration loop: that
  loop reported 32 ms per matmul unwarmed, 8.7 ms warmed, and 13.4 ms after
  an "optimisation" — while `cache/bench.py` (warmed, 200 reps, bare
  `torch.sparse.mm`) gives 0.94/0.99 ms consistently. Reading a GPU scalar
  into Python inside the loop (`if scale > 0`) forces a host sync every
  sub-tick and is a real trap for Phase 2.
- **The SNN loop has ~2× real-time headroom**: 0.943 ms per sparse matmul
  (fp16) / 0.989 ms (fp32), so PLAN.md's 10 LIF sub-ticks cost 9.4 ms of the
  20 ms tick. An earlier version of the check reported 32 ms because it timed
  cuSPARSE's first calls — which would have wrongly condemned the design as
  16× too slow.
- **RTX 5050 is Blackwell, sm_120**, and needs cu128 or later wheels. Earlier
  builds import cleanly and then fail at the first kernel launch, so
  provisioning verifies an actual matmul rather than trusting
  `torch.cuda.is_available()`.
- `duarouter` is opt-in via `DIPTERAN_DUAROUTER=1`. It is authoritative and
  Phase 2 will want it, but the in-process BFS settles the routability
  criterion without 4.42 GB.

### Removed

- `handoff.md` — the Phase 1 brief. Its decisions are folded into PLAN.md §9
  as instructed, its outcomes into PHASE1_REPORT.md, and its summary here.
