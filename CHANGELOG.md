# Changelog

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
