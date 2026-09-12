# Phase 1 report

Executed 2026-09-12 against PLAN.md Phase 1. Every number below came out of
`python check_phase1.py`, which writes its raw findings to
`cache/phase1_evidence.json`. Nothing here is asserted from a file merely
existing.

Phase 2 was not started: no real `FlyBangaloreNavigator` dynamics, no LIF
calibration, no PAM/PPL1 wiring, no SUMO/SNN sync loop. The mock
`main.py`/`index.html` streamer is still the running baseline.

## Environment as built

| | |
|---|---|
| Platform | WSL2 **Ubuntu 26.04 LTS**, installed `wsl --install -d Ubuntu --no-launch` |
| GPU | RTX 5050 Laptop, 8151 MiB, driver 616.56, CUDA passthrough confirmed inside WSL |
| SUMO | 1.25.0 from apt (`sumo sumo-tools sumo-doc`), built with Proj and GDAL |
| Python | 3.12.14, standalone CPython managed by `uv` 0.12.13 |
| venv | `/opt/dipteran/.venv` |
| Key libs | osmnx 2.1.1, networkx 3.6.1, pydantic 2.13.5, pyproj 3.8.0, pandas 3.0.5, fastapi 0.141.1, scipy, scikit-learn |

Three environment decisions were forced and are worth knowing about:

**No UNIX account was created.** `wsl --install -d Ubuntu --no-launch` skips
the first-boot account prompt, and everything runs as root. Nothing in Phase 1
needs a non-root user.

**Python 3.12, not the system 3.14.** Ubuntu 26.04 ships Python 3.14, which
has no torch or geopandas wheels. Rather than build geospatial stacks from
source, `uv` provides a standalone 3.12.

**The venv is on ext4, not in the repo.** `/mnt/e` is a 9p mount; torch is
~3 GB across thousands of small files and installing it there crawls.
`.venv/` stays in `.gitignore` for anyone who prefers it in-tree.

## What got built

| File | What it does |
|---|---|
| `config.yaml` + `dipteran/config.py` | Pydantic-validated route/compute/paths config. No lat/lon anywhere in code. |
| `dipteran/roads.py` | Overpass extract → OSMnx graph + SUMO network + the ID-mapping layer, with its verification. |
| `dipteran/connectome.py` | Fetches the v783 archive, aggregates to one signed weight per connected pair, builds the GPU CSR matrix, resolves named clusters to real root IDs. |
| `main.py` + `index.html` | README's mock scaffold, materialized. Port is overridable via `DIPTERAN_PORT` so tests can bind a free one. |
| `check_phase1.py` | Asserts all seven criteria; writes `cache/phase1_evidence.json`. |
| `LICENSE`, `DATA_LICENSE.md` | MIT for code; FlyWire treated as CC BY-NC 4.0, with the required citations. |

Two validations beyond the letter of the brief, because the stated criteria
could each pass while being useless:

- `config.yaml` is cross-checked against `index.html`'s hardcoded
  `BANGALORE_BOUNDS`. If they drift, the map silently refuses to show the
  agent where the backend puts it.
- The config rejects a `dist_m` smaller than the origin→destination
  separation. Otherwise the OSM extract simply does not contain Point B, and
  every downstream step works perfectly on a route that cannot be completed.

## Exit criteria — what each check actually showed

### OSMnx graph loads cleanly for the configured route — PASS

162,861 nodes, 406,970 edges from a 50.1 MB Overpass extract.

"Loads cleanly" was read as *the route has to exist*, not just that the file
parses: origin snaps to node 2385133858, destination to node 3666262565, and
`nx.has_path` confirms a directed route between them, 13.29 km against a
10.59 km straight line (1.26× detour — plausible for Bangalore).

One caveat, carried into PLAN.md §11: the extract has **239 strongly-connected
components**. A and B are in the same one, but a vehicle that wanders into a
smaller component can be stranded. Phase 2's motion model has to cope.

### netconvert produces a valid, routable .net.xml — PASS

862.0 MB, 385,141 non-internal edges, 1,065,099 shape points, 793,563
edge-to-edge connections, projected `+proj=utm +zone=43 +ellps=WGS84`. 200
netconvert warnings, no fatal errors.

Exiting 0 is not the same as routable — netconvert is perfectly happy to emit
a network whose components are disconnected, so "no fatal topology errors"
proves nothing on its own. Two independent answers, and the interesting part
is why there are two:

**In-process BFS over SUMO's own `<connection>` elements** (which encode the
permitted turns): the origin coordinate snaps to edge `1299917517` 5.1 m
away, the destination to `-307845469#2` 1.8 m away, and the destination is
**reachable after exploring 178,131 edges**. This is the hard assertion, and
it runs in 38 s inside the check.

**duarouter**, run standalone, routes the same pair in **135 edges, 14.49 km**
against a 10.59 km straight line. It is the authoritative SUMO answer and the
same query Phase 2's demand generation will make — but it needs 4.42 GB of
RSS to load this network (measured, `/usr/bin/time -v`), and on a 7.7 GiB
WSL2 guest that reliably crashes the *guest*, not just the process. So it is
opt-in:

```bash
DIPTERAN_DUAROUTER=1 python check_phase1.py --only sumo_net
```

Letting an environment memory limit fail a criterion the BFS already settles
would have been the wrong call; so would reporting "not routable" when the
truth is "the router could not be loaded here". `verify_routable` now
distinguishes a negative return code (signalled) from a non-zero one
(genuinely unroutable).

### Deterministic OSM-edge↔SUMO-edge lookup matching on geometry — PASS

This was flagged as the hardest, least-proven piece, and it was. Its own
section below.

### Connectome as a GPU-resident sparse CSR, VRAM logged — PASS

`torch.sparse_csr_tensor` on `cuda:0`, **138,639 × 138,639** with
**15,091,983 non-zeros** (density 7.85 × 10⁻⁴), from torch 2.11.0+cu128 on
an RTX 5050 at compute capability **sm_120**.

| dtype | matrix | predicted | VRAM consumed | of total |
|---|---|---|---|---|
| float16 | 87.3 MiB | 86.9 MiB | 90.0 MiB | 7.93 GiB |
| float32 | 116.5 MiB | 115.7 MiB | 118.0 MiB | 7.93 GiB |

Predicted against measured agrees to within 0.5%, which is the real check
here — it confirms the matrix costs exactly `nnz × (value + int32 index) +
(n+1) × int32` and nothing is quietly materialising dense.

**The brief's sizing estimate was wrong; its conclusion was right.** It
predicted ~2–3M non-zeros. The actual figure is **15,091,983** — about 5×
more. 54,492,922 synapses (matching the documented 54.5M) collapse to 15.1M
connected pairs, 3.61 synapses per pair. The reason the estimate was low is
that the v783 edge list is mostly *weak* pairs: the median connected pair has
just **two** synapses, and the FlyWire papers conventionally treat ≥5 as the
threshold for a reliable connection. Thresholding there would land near the
2–3M figure. The conclusion — "tens of MB, nowhere near the 8 GB budget" —
survives unscathed at 5× the non-zeros, so nothing needed rethinking, but
`compute.min_syn_count` now exists as a knob (default 0, honouring PLAN.md's
maximum-fidelity decision).

**Signs come per presynaptic neuron, not per synapse.** The annotation
table's consensus `top_nt` (acetylcholine 86,193, glutamate 24,875, GABA
19,171, dopamine 5,909, serotonin 2,282, octopamine 216) is applied via
Dale's principle — one neuron, one fast transmitter. That is less noisy than
the per-synapse predictions and much cheaper: the edge list's six probability
columns over 16.8M rows are never read. **40.2%** of pairs come out
inhibitory (6,071,458); 394 pairs have a presynaptic neuron with no
annotation and are left excitatory.

`W[i, j]` is presynaptic *j* onto postsynaptic *i* — rows postsynaptic — to
match the `W @ v` in PLAN.md §5. Getting that backwards would silently run
the brain in reverse, so the orientation is now stated in the module
docstring and the decisions log.

**Speed, and a measurement mistake worth recording.** PLAN.md's tick is 10
LIF sub-ticks per 20 ms. The first version of this check reported **32 ms per
sparse matmul**, which would have made the design 16× too slow for real time.
That number was almost entirely cuSPARSE first-call handle setup and
algorithm selection, plus a host sync every iteration. Warmed up and timed
over 200 reps:

| dtype | per matmul | 10 sub-ticks | vs 20 ms tick | effective bandwidth |
|---|---|---|---|---|
| float16 | 0.943 ms | 9.4 ms | **2.12× real-time** | 97.1 GB/s |
| float32 | 0.989 ms | 9.9 ms | **2.02× real-time** | 123.5 GB/s |

So the loop fits, with roughly 2× headroom, before SUMO's share of the tick.
The check now warms up before timing.

**fp32 is the new default.** It costs 29 MiB and 5% more time — both trivial
against 8 GB — and removes a real hazard: with weights up to 2405 and
in-degree in the hundreds, fp16 products saturate to `inf` within a couple of
sub-ticks unless every step is rescaled. `config.yaml` documents the
trade-off; fp16 remains one line away.

### Every sensory/motor cluster resolves to a valid row index — PASS

All 14, against `Supplemental_file1_neuron_annotations.tsv` (139,248 annotated
neurons). Every row index is checked to round-trip: `neurons[row]` must equal
the root ID the cluster claims for it.

| Cluster | Pattern | Rows | Example root ID |
|---|---|---|---|
| `odor_dm1` | `^DM1[_ ]` | 2 | 720575940619071005 |
| `odor_dm4` | `^DM4[_ ]` | 4 | 720575940615366055 |
| `compass_epg` | `^EPGt?$` | 51 | 720575940602917088 |
| `optic_flow_t4` | `^T4[a-d]$` | 6,241 | 720575940599457990 |
| `optic_flow_t5` | `^T5[a-d]$` | 6,005 | 720575940596125868 |
| `looming_lc4` | `^LC4$` | 104 | 720575940603637438 |
| `looming_lplc2` | `^LPLC2$` | 210 | 720575940603989420 |
| `acoustic_jo` | `^auditory$` | 387 | 720575940600433181 |
| `steer_l_dna01` | `^DNa01$` | 4 | 720575940618167579 |
| `steer_r_dna02` | `^DNa02$` | 2 | 720575940604737708 |
| `panic_dnp01` | `^DNp01$` | 2 | 720575940622838154 |
| `throttle` | `^DNg02` (**fallback**) | 25 | 720575940610863310 |
| `reward_pam` | `^PAM` | 307 | 720575940604479590 |
| `punish_ppl1` | `^PPL1` | 16 | 720575940609180611 |

The counts are the check that these are the right cells, not just *some*
cells: EPG at 51 against a literature figure of ~46–50, DNp01 at exactly 2
for a bilateral Giant Fiber pair, PPL1 at 16, T4/T5 in the low thousands per
subtype-set. DM1 resolves to `DM1_lPN` and DM4 to `DM4_vPN` + `DM4_adPN`, all
`cell_class = ALPN`, `cell_sub_class = uniglomerular` — the uniglomerular
olfactory projection neurons for those two glomeruli, one per hemisphere.

Resolution is by regex against `cell_type`, `hemibrain_type`, `cell_class`,
`super_class` and `cell_sub_class`, with primary and fallback patterns tried
in order and **which one matched recorded in the output**, so a fallback can
never quietly pass itself off as the intended mapping. A cluster that matches
nothing raises rather than returning an empty list. Three of the 390 annotated
auditory neurons have no connections in the edge list and so have no matrix
row; that count is reported per cluster as `missing_from_matrix`.

#### The two clusters PLAN.md named that do not exist in FAFB

Both are corrected in PLAN.md (§3 and §4) rather than quietly substituted.

**`B1`/`B2` — throttle.** The b1 and b2 basalar wing motor neurons live in the
ventral nerve cord. FAFB/FlyWire is a brain-only volume, so they are not in
this dataset and no pattern will ever find them. Wiring the real wing motor
neurons would mean pulling in MANC, a separate connectome, which is out of
scope. The brain-side equivalent is **DNg02**, the descending population that
sets wingbeat amplitude — 25 neurons, `super_class = descending`, matched as a
prefix because v783 splits it into subtypes `DNg02_a` … `DNg02_h`. This is the
one cluster resolving via its fallback, and it is flagged as such in
`cache/cluster_ids_783.json`.

**`aPN` — acoustic.** No `aPN` label exists in v783 under any type column.
What the release does curate is `cell_sub_class = auditory`: 390 neurons,
every one a `JO-A*`/`JO-B*` cell type with `super_class = sensory`. Those are
the sound-sensitive Johnston's organ subgroups, as distinct from the
`JO-C`/`D`/`E` wind-and-gravity (484) and grooming (191) subgroups of the same
organ — so the curation is doing exactly the discrimination this project needs
for a 500–800 Hz horn. They are afferents rather than projection neurons,
which arguably makes them the better injection site: it is where sound
actually enters the brain. The cluster is named `acoustic_jo` rather than
`acoustic_apn` so nobody reading Phase 2 code mistakes them for PNs.

### Also checked

- **`python main.py` serves the mock stack** — PASS. The check starts the real
  process on a free port, fetches `/` (3,963 bytes, containing
  `maplibregl.Map`, `BANGALORE_BOUNDS` and the event terminal), opens
  `ws://.../ws`, reads 5 frames, and asserts every payload key is present,
  every streamed coordinate falls inside the configured bbox, heading stays in
  [0, 360), and the agent actually moved.
- **`config.yaml` validates** — PASS, including both extra guards above.

## Step 8: what the ID-mapping approach ended up being

Nobody had validated an approach for this, so here is what it turned into.

**First finding: both graphs must come from one extract.** Not two downloads
of the same area — the same local `.osm.xml` file, fed to both
`ox.graph_from_xml` and `netconvert`. The Overpass query uses OSMnx's own
`drive` way filter so the two see an identical way set. Everything downstream
rests on them sharing way IDs; two independent downloads do not guarantee it.

**Second finding: `origId` exists and works.** Running netconvert with
`--output.original-names true` writes the source OSM way ID onto every lane
as a `<param key="origId">`. It survives `--geometry.remove` merging edges.
That gave a way index covering **105,030 OSM ways over 385,141 SUMO edges**,
and **99.2%** of OSMnx edges (403,706 / 406,970) matched by ID alone. The
fallback paths in the code — parsing the way ID out of the SUMO edge ID, then
pure nearest-geometry — were never needed, and `check_phase1.py` now asserts
the strategy is `origId` so a silent regression to them fails.

**Third finding, and the important one: the ID join alone is nowhere near
enough.** A way ID is not a segment ID. Real example straight out of the
network:

```
edge '-100001035#0'  origId=['100001035']
edge '-100001035#1'  origId=['100001035']
...
edge '-100001035#5'  origId=['100001035']
```

One OSM way becomes six SUMO edges per direction, and OSMnx independently
splits the same way into its own edges at intersections. Joining on way ID
alone maps *every* OSMnx edge on that way to *every* SUMO edge of it. Across
the network that produced **4,078,524 candidate pairs** — an average of ten
SUMO edges per OSMnx edge, and for a long arterial like Outer Ring Road,
hundreds. Measured on the first attempt: the centroid of an OSMnx edge sat a
median 34 m and a p90 of **130 m** from the centroid of its mapped SUMO edges,
worst case 1.59 km. Useless for reading per-segment congestion, which is the
entire point of the layer.

**So the join is two-stage.** The way ID narrows candidates cheaply and with
no false negatives; geometry then keeps the SUMO segments whose centroid lies
within 30 m of the OSMnx edge's own polyline. 4,078,524 candidates →
**1,424,648 kept (34.9%)**, coverage unchanged at 99.2%.

After refinement:

| Metric | Median | p90 | Max |
|---|---|---|---|
| OSMnx edge midpoint → nearest of its mapped SUMO edges | **7.16 m** | 7.74 m | 25.48 m |
| Worst-point shape coverage (Hausdorff-style) | 7.37 m | 8.35 m | 55.99 m |

7 m is lane-offset scale — SUMO offsets each lane's shape from the OSM
centreline — so this is about as tight as the two representations can agree.

**On verifying it without circular reasoning.** Refining on centroid distance
and then asserting centroid distance proves nothing. The assertions measure
the opposite direction (OSMnx midpoint against mapped SUMO polylines, rather
than SUMO centroids against the OSMnx polyline), and determinism is proven by
rebuilding the whole map twice and requiring byte-identical output — it is,
across all 406,970 entries.

**A check that had to be relaxed, and why.** The first verification demanded
that the SUMO edge which *pure geometry* calls nearest to an OSMnx edge's
midpoint always be inside that edge's mapping. It came out at 68.8%, and
inspecting the disagreements showed the check was wrong, not the join:

```
OSMnx edge on way 32261256 → mapped '32261256#0', midpoint gap 7.72 m
                             pure geometry prefers '201196348#6'
```

The mapped edge is 7.72 m from the midpoint — correct. A *different* way's
lane geometry simply happens to be marginally closer, which is what you get
when lane shapes are offset by metres and Bangalore stacks flyovers directly
over surface roads: 2D nearest-geometry is genuinely ambiguous at that scale.
So that number is now reported as a statistic with a loose floor (≥60%)
rather than used as the gate; the hard gate is the midpoint-to-own-mapping
distance above, which is what Phase 2 actually depends on.

`SumoNet.nearest_edge` also exists as a KD-tree point query — given the
fly-vehicle's position, which SUMO edge is it on — and needs no ID map at all.
The ID map's job is the set-wise view: which SUMO edges make up this OSMnx
edge, for aggregating congestion across it.

## Things that did not go as planned

**No FlyWire account was needed.** Task 4 said to create a Codex/CAVE account.
It turned out not to gate anything: the v783 connectome is openly downloadable
from Zenodo record 10676866 and the annotation table from
`flyconnectome/flywire_annotations`. Both were fetched unattended. An account
is still worth having for interactive lookup in Codex, and the CC BY-NC terms
apply regardless — see `DATA_LICENSE.md`. (Creating accounts on someone's
behalf is also something I will not do, so this was a lucky escape rather than
a deliberate shortcut.)

**A cheaper connectome file than expected.** `proofread_connections_783.feather`
(852 MB) is already aggregated per (neuron pair, neuropil) — 16,847,997 rows
with `pre_pt_root_id`, `post_pt_root_id`, `syn_count` and six per-transmitter
probability columns. Summing over neuropils gives exactly the per-pair weight
wanted, so the 9.5 GB per-synapse table was never downloaded at all.

**torch took roughly 80 minutes.** Not a code problem — `pypi.nvidia.com` and
`download.pytorch.org` were slow, and the first attempt died on a timeout
fetching `nvidia-nvjitlink-cu12`. Retried with `UV_HTTP_TIMEOUT=300`. Worth
knowing that the RTX 5050 is Blackwell (**sm_120**) and requires **cu128 or
later**; earlier wheels import cleanly and then fail at the first kernel
launch, so `provision.sh` verifies an actual matmul rather than trusting
`torch.cuda.is_available()`.

**Two named clusters do not exist in FAFB under the names PLAN.md used.** Both
are now corrected in PLAN.md rather than papered over — see the cluster
section.

**RAM is tighter than VRAM, and this cost the most time of anything here.**
The whole exercise was expected to be GPU-memory-bound. It was not — the GPU
never went above 118 MiB of 7.93 GiB. Host RAM is the real ceiling. WSL2
defaults to half of physical memory, so 7.7 GiB of the 15.3 GiB host, and
`/usr/bin/time -v` puts **`duarouter` at 4.42 GB peak RSS** (39 s, exit 0)
just to load the 862 MB network and route one trip.

That does not co-exist with a parsed `SumoNet` in the same process, and the
failure mode is nasty: the OOM killer takes the Python parent, the whole WSL
session dies, and nothing is printed — a silent exit 1 with no traceback, at
the same line every time. It took three runs and a standalone
`/usr/bin/time -v` to pin down, having first been misread as two concurrent
jobs (which was *also* happening, and which is what the earlier kills were).

Three fixes, escalating, because the first two were not enough:

1. `route_endpoint_edges` returns plain data — the two edge IDs and an
   edge→length map — so the caller can drop the parsed network before
   spawning a subprocess. Not sufficient on its own.
2. The cross-check memoisation of the OSMnx graph and SUMO network was
   reverted. Holding a 3–4 GB graph across checks to save a 25 s re-read is a
   bad trade at this budget. Still not sufficient.
3. **A subprocess per check.** This is the one that works. `gc.collect()`
   frees the objects but CPython does not return the arenas to the OS, so RSS
   stays at 3–4 GB after the OSMnx check and the *next* check starts near the
   ceiling — the suite died parsing the `.net.xml`, having earlier died at
   duarouter, each time looking like a different bug. `check_phase1.py` now
   re-invokes itself once per check (`--worker`, writing its result to a temp
   JSON) and merges the results. `--in-process` keeps the old behaviour for
   anyone with the RAM.

The lesson worth carrying: in a memory-capped guest, a checker that holds
multi-gigabyte artefacts across steps is not slow, it is unreliable, and the
failures do not point at the cause. Peak RSS per step, for reference: 3–4 GB
OSMnx graph, ~2 GB parsing the `.net.xml`, 2.65 GiB connectome aggregation,
4.42 GB duarouter.

Phase 2 needs SUMO resident *alongside* the SNN, so raise the WSL allocation
in `.wslconfig` before starting.

**WSL itself needed restarting once.** After the OOM kills, the distro came
back but the WSL service started answering
`Wsl/Service/E_UNEXPECTED — Catastrophic failure` to every command, with
`dxgk: dxgkio_reserve_gpu_va: Ioctl failed: -75` in the kernel log — the GPU
virtual-address passthrough had come unstuck. `wsl --terminate Ubuntu` fixed
it; the full `wsl --shutdown` was avoided because Docker Desktop shares the
subsystem. Worth knowing that repeated OOM kills in a CUDA-using WSL distro
can leave the GPU passthrough in a state that only a distro restart clears,
and that a silent exit 1 with no traceback is what that looks like from the
outside.

**Two self-inflicted bugs worth recording.**

`ET.iterparse` with a blanket `elem.clear()` wipes child `<param>` attributes
before the parent `<edge>` end-event fires, silently destroying every
`origId`. `SumoNet._parse` clears only `edge` and `location` elements for
exactly this reason.

The first version of the GPU check timed sparse matmuls without warming up,
and so reported 32 ms each — which would have meant PLAN.md's 10-sub-tick
20 ms loop was 16× too slow for real time, a conclusion that would have
misdirected all of Phase 2. The true steady-state figure is 0.94 ms. Anything
measuring cuSPARSE has to discard the first few calls.

## Notes for Phase 2

- **Raise the WSL memory allocation** before running SUMO alongside the SNN.
  7.7 GiB is enough for one stage at a time and not much more; `duarouter`
  alone wants 3.85 GB.
- The SNN loop has ~2× real-time headroom (9.4 ms of a 20 ms tick). SUMO at
  `--step-length 0.1` has to fit in what is left, and that is still unproven
  on a 385k-edge network with traffic demand. Measure it early — the sync
  design depends on it, and the tick budget has no slack for a surprise.
- Edge weights are raw aggregate synapse counts up to 2405, and in-degree runs
  to hundreds. LIF calibration has to scale them: `v_thresh = 1.0` against a
  weight of 2,405 will not behave. This is also why the default dtype is now
  fp32 — see the connectome section.
- Monoamine (octopamine, serotonin, dopamine) edges are excitatory-signed in
  `W` as a placeholder. Harmless only while PAM/PPL1 are unwired; the
  reinforcement loop must make dopamine act on weights instead.
- Consider whether the weak tail belongs in the matrix at all. 80% of the
  15.1M pairs are below FlyWire's own ≥5-synapse reliability threshold. The
  knob is `compute.min_syn_count`; the decision is a fidelity-versus-noise
  call that Phase 2's dynamics should settle empirically.
- `cache/osm_sumo_edge_map.json` is 64 MB of JSON and takes ~80 s to rebuild.
  If Phase 2 reloads it often, convert it to `.npz`.
- For "which SUMO edge is the vehicle on", use `SumoNet.nearest_edge` (KD-tree
  point query, no ID map needed). Use the ID map for the set-wise direction:
  which SUMO edges to aggregate congestion over for a given OSMnx edge.
