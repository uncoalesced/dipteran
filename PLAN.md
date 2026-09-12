# PLAN: Drosophila Bangalore Purgatory

## 1. Concept

Take a real fly's brain — reconstructed as a spiking neural network (SNN) from an actual dissected connectome — and drop it, with zero routing instructions, into a virtual vehicle navigating simulated Bangalore traffic. Watch survival circuits evolved for predator evasion and odor-tracking collide with gridlock, auto-rickshaw cut-ins, potholes, and horns.

No 3D game engine. This is a headless, event-driven loop feeding live geospatial/traffic data into the SNN, whose motor output drives a vehicle along a real road network.

This is a research-grade simulation, not a zero-dependency toy: it needs a CUDA GPU, a SUMO install, and a FlyWire account. See [Known Risks / Blockers](#11-known-risks--blockers) before starting.

## 2. System Architecture

```
+-------------------------------------------------------+
|                  THE ENVIRONMENT                      |
|  - Road network: OSMnx/networkx graph (SNN reasoning) |
|    + SUMO network (netconvert), kept in sync          |
|  - SUMO microsimulation: background traffic agents     |
|    -> emergent congestion IS the traffic signal        |
|  - Synthetic Hazards: Potholes, Horns                  |
+---------------------------+---------------------------+
                            |
              Sensory Current Vector (I_ext)
                            v
+-------------------------------------------------------+
|                 THE CONNECTOME (SNN)                  |
|  - Full FlyWire/FAFB connectome (~139k neurons,        |
|    ~54.5M synapses), GPU sparse matrix                 |
|  - Central Complex: Ring attractor heading compass    |
|  - Antennal Lobes: DM1 sugar odor concentration        |
|  - Optic Lobes: T4/T5 optical flow & LC4 looming       |
|  - Johnston's Organ: Acoustic / vibration mechanosens  |
+---------------------------+---------------------------+
                            |
                Motor Spike Outflow (S_out)
                            v
+-------------------------------------------------------+
|               VEHICLE / AGENT CONTROLLER               |
|  - DNa01 / DNa02: Steering angle (Steer L/R)           |
|  - Wing-beat / Leg Motor: Throttle / Brake             |
|  - Giant Fiber (DNp01): Emergency Brake / Panic Swerve |
|  - Hybrid motion: continuous heading/position,          |
|    soft-pulled toward nearest road edge                 |
+-------------------------------------------------------+
```

Two road-network representations feed the loop: an OSMnx/networkx graph the SNN reasons over (bearing, distance, "which way is the goal"), and a SUMO network built from the same OSM extract via `netconvert` that actually runs the traffic microsimulation. A mapping layer keeps the fly-vehicle's position and nearby traffic state synced between the two (see [Roadmap Phase 1](#10-roadmap)) — this is a build foundation, before backend work starts.

There is no separate live-traffic API (Google/TomTom) and no external routing-engine congestion estimate (OSRM/GraphHopper) — both were considered during design and dropped once SUMO entered the picture (see [Decisions Log](#9-decisions-log)). SUMO's own simulated vehicle density on each edge is the only congestion signal.

## 3. Biological Sensory Mapping

How the fly navigates without turn-by-turn GPS, and how simulated Bangalore traffic gets translated into the sensory language of an insect brain:

| Real-world input | Biological pathway | Effect |
|---|---|---|
| Destination (Point B) proximity | Odor beacon → DM1/DM4 olfactory projection neurons | Point B emits an inverse-square odor gradient, `Concentration(x,y) = 1 / dist((x,y), Target)`, injected as continuous current. Concentration rises as the vehicle nears the goal. |
| Target bearing | Central Complex ring attractor (EPG neurons, Ellipsoid Body) | Current injected into the EPG ring encodes bearing of target relative to current heading — the fly's internal "sun compass." |
| Traffic gridlock (zero motion despite throttle) | Lobula Plate Tangential Cells (LPTCs), optic flow | Optic flow is derived from the fly-vehicle's actual displacement each tick inside SUMO's simulated traffic. Zero flow while attempting to move (stuck behind a SUMO-simulated jam) reads as being trapped in a spider web/sap — escalates into sustained panic convulsions. |
| Auto-rickshaw / vehicle cutting in | LC4 / LPLC2 looming detectors → Giant Fiber (DNp01) | Geometric near-collision detection between the fly-vehicle and nearby SUMO agents: rapid angular expansion of an agent in the forward field fires directly into DNp01, forcing instantaneous hard braking or an uncontrolled swerve. |
| Potholes / broken roads | Chordotonal organ & campaniform sensilla (mechanosensory) | Vertical acceleration spikes (synthetic hazard triggers on specific edges) trigger grooming/freezing circuits, temporarily disabling forward movement. |
| Horn blasts (500–800 Hz) | Johnston's Organ → the sound-sensitive JO-A/JO-B afferents | Synthetic horn events (proximity-triggered by nearby SUMO agents) fall inside the fly's natural courtship/predator-warning frequency band; destabilize the internal compass. **Correction from Phase 1**: this row originally said "acoustic projection neurons (aPNs)", but no aPN label exists in FlyWire v783. The release instead curates `cell_sub_class = auditory` — 390 sensory neurons, all `JO-A*`/`JO-B*` cell types, as distinct from the JO-C/D/E wind-and-gravity and grooming subgroups of the same organ. Current is injected there, which is where sound actually enters the brain. |

## 4. Motor Decoding

Three motor neuron clusters are read out of the simulation every tick:

- **Steering** — `DNa01` (left turn) and `DNa02` (right turn) descending neurons. Continuous steering torque:

  $$\tau_{\text{steer}} = \text{Spikes}(\text{DNa01}) - \text{Spikes}(\text{DNa02})$$

  Under the hybrid motion model, this torque continuously rotates the vehicle's free heading; a separate soft force pulls position back toward the nearest road edge, so the vehicle can wander (matching the "wrong turns down unpaved alleys" framing) without fully decoupling from the road network.

- **Acceleration / Brake** — throttle is read from **DNg02** (subtypes `DNg02_a`…`DNg02_h`, 25 neurons), the descending population that sets wingbeat amplitude. If the **Giant Fiber (DNp01)** fires, override throttle to 0 and apply full emergency braking.

  **Correction from Phase 1**: this originally read "forward wing-motor circuits (B1, B2)". The b1 and b2 basalar wing motor neurons are in the ventral nerve cord, and FAFB/FlyWire is a brain-only volume — they are not in this dataset and cannot be. DNg02 is the brain-side equivalent, and is what the code resolves. Wiring the actual wing motor neurons would mean bringing in the MANC ventral-nerve-cord connectome, which is a separate dataset and out of scope.

### Reinforcement loop (no backpropagation)

Real-time neuromodulatory updates run directly on mushroom body synapses:

- **PAM cluster (reward)** — every 100m closer to Point B, or sustained clean cruising above 30 km/h, pulses PAM dopaminergic neurons.
- **PPL1 cluster (punishment)** — stationary at an intersection >45s, hitting a pothole, or getting stuck in a dead end pulses PPL1, weakening synapses between the current visual context and the motor path that led there. This is the fly's only learning mechanism in this system — synaptic depression from simulated pain, not gradient descent.

## 5. Implementation Stack

**Step 1 — Build both road-network representations from the same OSM extract:**

```python
import osmnx as ox
import networkx as nx

point_a = (12.9177, 77.6238)  # Silk Board
point_b = (13.0039, 77.6653)  # Tin Factory

G = ox.graph_from_point(point_a, dist=15000, network_type='drive')
```

In parallel, run SUMO's `netconvert` against the same region to produce the SUMO network used for microsimulation. Build a mapping layer between OSMnx edge/node IDs and SUMO edge/lane IDs (both derive from the same underlying OSM way/node IDs, so this is a join, not a re-derivation) — this is the trickiest integration point in the stack and its own roadmap phase.

**Step 2 — Run SUMO as the traffic source**, with background vehicle agents populating the network. Read each edge's live simulated speed/density directly from SUMO's traffic state each tick — this is the congestion signal; there is no external traffic API call.

**Step 3 — Run the spiking connectome loop**, on GPU, every 20ms tick:
1. Calculate target heading and distance from the OSMnx graph.
2. Read current SUMO edge state (speed, density) and nearby SUMO agent positions.
3. Populate the sensory input array (`I_ext`): odor, compass bearing, optic flow, looming, mechanosensory, acoustic.
4. Run 10 sub-ticks of leaky integrate-and-fire (LIF) sparse matrix multiplication over the full ~139k-neuron adjacency matrix.
5. Decode `DNa01`, `DNa02`, `DNp01` spikes into steering torque / throttle / panic override; apply hybrid motion update.

Reference LIF engine (extend to load the real FlyWire adjacency matrix, not placeholder dimensions):

```python
import torch

class FlyBangaloreNavigator:
    def __init__(self, adj_matrix_csr, sensory_ids, motor_ids):
        self.W = adj_matrix_csr  # real FlyWire adjacency matrix, on GPU
        self.n_neurons = adj_matrix_csr.shape[0]  # ~139,255
        self.v = torch.zeros(self.n_neurons, device='cuda')
        self.v_rest = 0.0
        self.v_thresh = 1.0
        self.v_reset = -0.2
        self.decay = 0.9

        self.sensory_ids = sensory_ids  # {'odor': [...], 'looming': [...]}
        self.motor_ids = motor_ids      # {'steer_l': id, 'steer_r': id, 'panic': id}

    def step(self, external_currents):
        i_ext = torch.zeros(self.n_neurons, device='cuda')
        for nid, val in external_currents.items():
            i_ext[nid] = val

        synaptic_input = torch.sparse.mm(self.W, self.v.unsqueeze(1)).squeeze(1)
        self.v = (self.v * self.decay) + synaptic_input + i_ext

        spikes = (self.v >= self.v_thresh).float()
        self.v[spikes.bool()] = self.v_reset

        steer = spikes[self.motor_ids['steer_l']] - spikes[self.motor_ids['steer_r']]
        panic = spikes[self.motor_ids['panic']]

        return steer.item(), panic.item()
```

`sensory_ids` and `motor_ids` must be populated with real FlyWire cell IDs for each named cluster (DM1/DM4, EPG, T4/T5, LC4/LPLC2, Johnston's organ aPNs, DNa01/DNa02, DNp01, B1/B2, PAM, PPL1) — resolved via FlyWire/CAVE's annotation search, not invented. This ID-resolution work is its own roadmap phase.

**Step 4 — Visualize**: a dashboard showing the OSM route with the agent's GPS dot swerving/erratic, SUMO's background traffic, plus a live spike raster of the fly's brain regions.

## 6. Frontend / Deployment Architecture

```
[ Headless SNN Engine + SUMO microsimulation ]
        | (Inter-thread Queue)
        v
[ FastAPI WebSocket Server ]
        |  JSON Stream @ 20-30 Hz: { lat, lon, heading, brain_state, nearby_agents, event }
        v
[ Single-file index.html ]
   +-- MapLibre GL JS (hardware-accelerated 60 FPS icon interpolation)
   +-- Bounding Box Mask (locked strictly to Bangalore)
   +-- Bottom-Right HUD (neurological telemetry & live event terminal)
```

**Bangalore bounding box** — MapLibre `maxBounds` locks pan/zoom to city limits (SW `[77.40, 12.80]` to NE `[77.80, 13.15]`); the camera bounces back if dragged past it.

**Map layers**:
- Point A — origin pin (e.g. Silk Board junction).
- Point B — pulsating fermentation/sugar beacon (target).
- The Fly Agent — custom SVG marker, position/rotation updated per WebSocket message with CSS interpolation for smooth motion.
- Background SUMO traffic — nearby simulated vehicle agents, rendered from the `nearby_agents` payload field.
- Purgatory Trail — a GeoJSON LineString trailing the agent's erratic path, red-coded wherever it hit severe traffic or panic-swerved.

**Bottom-right HUD** (`backdrop-blur-md bg-black/80`):
- Dopamine level (PAM reward vs. PPL1 pain) as a split bar.
- Giant Fiber (DNp01) voltage — sparkline/gauge for panic proximity.
- Compass heading bump — radial ring showing active EPG cluster.
- Live event terminal, e.g.:
  ```
  [12.924, 77.631] LC4 SPIKE: Auto-rickshaw cut-in detected (0.8m margin)
  [12.926, 77.632] DNp01 FIRED: Panic swerve executed into bus lane
  [12.929, 77.635] MECHANOSENSORY SHOCK: Unmapped crater pothole hit
  [12.929, 77.635] PPL1 BURST: 450mV pain injection (Synaptic depression)
  [12.931, 77.639] LPTC ZERO-FLOW: Silk Board stagnation (Spiderweb dread)
  ```

The scaffold's mock random-walk streamer (`main.py` using `random.random()` for heading/pain/panic) is the placeholder this pipeline replaces — see [Roadmap Phase 2](#10-roadmap).

## 7. Scientific Provenance

What "taking the brain" actually meant — this is not a scan, it's a reconstruction from a physically destroyed specimen:

- **Source**: A single adult female *Drosophila melanogaster*, dissected at HHMI Janelia Research Campus (Davi Bock, Zhihao Zheng), published as the **FAFB dataset** (*Cell*, 2018).
- **Process**: Brain dissected → chemically fixed (glutaraldehyde/formaldehyde) → stained with osmium tetroxide/heavy metals → embedded in epoxy resin → sliced into 7,050 ribbons at 40nm thickness → imaged via serial-section electron microscopy at 4nm/pixel → 100+ terabytes of 2D cross-sections. The physical brain was destroyed; only the digital blueprint remains.
- **Reconstruction**: The **FlyWire Consortium**, co-led by Sebastian Seung and Mala Murthy (Princeton), with Gregory Jefferis and Philipp Schlegel (MRC LMB Cambridge). CNN/3D U-Net segmentation traced cell boundaries; 287 researchers across 76 labs crowdsourced proofreading via the FlyWire platform. Published in *Nature*, October 2024.
- **Deliverable**: 139,255 proofread neurons, ~54.5 million chemical synapses, ML-predicted neurotransmitter type per synapse (acetylcholine, GABA, glutamate, dopamine, serotonin, octopamine).
- **Access terms**: FlyWire/FAFB data is distributed under a data use agreement via the CAVE platform, not freely redistributable. This repo does not and cannot bundle the connectome — each user fetches it under their own FlyWire account (see [Roadmap Phase 1](#10-roadmap)).

**What the dataset includes vs. lacks:**

| Included | Missing |
|---|---|
| Physical wiring (which neuron touches which) | Ion channel dynamics (Hodgkin-Huxley conductances, resting voltages, activation curves) |
| Synaptic counts and spatial coordinates | Neuromodulatory volume transmission (extracellular dopamine/neuropeptide diffusion) |
| Predicted neurotransmitter types (excitatory/inhibitory) | Electrical synapses (gap junctions — invisible at 40nm EM resolution) |
| Static morphology (axon/dendrite geometry) | Intracellular state (ATP, second-messenger signaling, gene transcription) |
| Proofread cell-type categorization | Glia (non-neuronal support cells regulating neurotransmitter reuptake/ionic balance) |

## 8. Philosophical Framing

Does running this connectome replicate the fly's consciousness, or produce an illusion of it?

- Flies lack higher-order metacognition or linguistic self-awareness, but behavioral evidence shows primary subjective states: valence (hunger, fear, arousal) driven by dopamine/octopamine circuits, selective attention, and state-dependent behavioral persistence after aversive events.
- **Running the connectome is emulation, not reincarnation.** The circuit topology is biologically real — the Central Complex ring-attractor compass and Giant Fiber looming-escape loop work in simulation because the wiring pattern is authentic evolutionary structure. But the physics driving it (leaky integrate-and-fire equations, chosen resting potential/threshold/decay) are artificial approximations layered on top.
- **Searle's "wet rain" analogy**: a simulated hurricane computes Navier-Stokes with precision, but no molecule of water gets wet. Simulating the fly's membrane potential equations computes the math of neural transmission without reproducing the biological substrate that generated the original animal's living internal state.
- Net framing: this project takes the structural skeleton of an insect's evolved decision-tree, supplies the missing biophysical laws arbitrarily, and observes how that fixed geometry reacts to synthetic inputs — not resurrection, but structural emulation of an input-output transfer function.

## 9. Decisions Log

Resolved 2026-09-12 via a design-review interview. Recorded so later contributors don't re-litigate them without cause:

| Question | Decision | Why |
|---|---|---|
| Connectome scope | Full ~139k-neuron FlyWire connectome, not a curated subset or mock | Chose maximum fidelity over the smaller/faster paths |
| Compute | GPU torch sparse ops | Required to keep full-connectome ticks near real-time |
| "Zero-setup" README claim | Dropped | Incompatible with a GPU + large connectome-data requirement; honesty over a broken promise |
| Connectome data acquisition | User fetches via their own FlyWire/CAVE account | FlyWire data carries a use agreement; not freely redistributable in-repo |
| Traffic data source | SUMO microsimulation only (no OSRM/GraphHopper, no Google/TomTom) | Once real SUMO agents were chosen, their emergent congestion made an external traffic API redundant |
| Road network source | Both OSMnx/networkx (SNN reasoning) and SUMO netconvert network (microsim), kept in sync | SNN needs a lightweight graph to compute bearing/distance; SUMO needs its own network format to run agents — no single library does both well |
| Vehicle motion model | Hybrid: continuous heading/position, soft-pulled toward nearest road edge | Preserves the "wanders into unpaved alleys" narrative while keeping the vehicle roughly road-bound |
| Other traffic (auto-rickshaws etc.) | Real SUMO agents with geometric near-collision detection, not scripted random events | Real agent geometry gives genuine looming signals instead of a canned event list |
| PLAN.md role | Living roadmap, not a one-time explainer | Project is now committed to build-out, not just concept documentation |
| Roadmap granularity | Consolidated into 4 phases (basic build + backend foundations / finish backend / frontend + testing / ship) | User wanted a small, trackable phase count rather than 8 micro-phases |

### Resolved in later planning rounds (2026-09-12)

| Topic | Decision | Why |
|---|---|---|
| GPU/VRAM sizing | Non-issue. The adjacency matrix holds one entry per *connected neuron pair* (aggregate synapse weight), not one per raw synapse. Do not build a per-synapse matrix. | ~2–3M non-zeros, not 54.5M. At fp16 values with int32 indices that is tens of MB against an 8GB budget |
| FlyWire data access | Static v783 archive (Feather edge list) via Zenodo/Codex download — **not** the live CAVE query API for bulk connectivity | A `caveclient` account is still useful for interactive annotation lookup, but it does not gate the download |
| FlyWire data license | Treat as **CC BY-NC 4.0** (attribution + non-commercial) | Sources conflict — Zenodo metadata says CC-BY-4.0, flywire.ai/guidelines says CC BY-NC 4.0. NC is the binding conservative assumption. Never commit the raw connectome or any matrix compiled from it. See `DATA_LICENSE.md` |
| Code license | MIT for Dipteran's own code | Independent of the CC BY-NC 4.0 data license above |
| Route config | `config.yaml`, Pydantic-validated — origin/destination/bbox/network_type/device/data paths as fields | No hardcoded lat/lon constants in code |
| SUMO/SNN sync | SUMO at `--step-length 0.1` (10Hz), SNN at 50Hz (20ms), sample SUMO state and issue batched `traci` control every 5th SNN tick (100ms boundary); frontend interpolates for 60fps | Phase 2 implements this; Phase 1 only had to avoid foreclosing it |

### Resolved while building Phase 1 (2026-09-12)

| Topic | Decision | Why |
|---|---|---|
| Platform | WSL2 Ubuntu 26.04, installed `--no-launch` and driven as root | SUMO/OSMnx/GDAL are far cleaner than Windows-native, and CUDA passthrough works untouched. `--no-launch` skips first-boot UNIX account creation, which nothing here needs |
| Python version | 3.12, from a `uv`-managed standalone CPython | Ubuntu 26.04 ships Python 3.14, which has no torch or geopandas wheels; source builds are not worth it |
| venv location | `/opt/dipteran/.venv` on ext4, not inside the repo on `/mnt/e` | torch pulls ~3GB across thousands of small files and the 9p mount makes that crawl. `.venv/` stays in `.gitignore` for anyone who does put it in-repo |
| torch build | `--index-url https://download.pytorch.org/whl/cu128` | The RTX 5050 is Blackwell, compute capability sm_120. Pre-cu128 wheels import fine and then fail at the first kernel launch |
| OSM extract | One Overpass query into one local `.osm.xml`, feeding **both** `ox.graph_from_xml` and `netconvert` | Two independent downloads are not guaranteed to share way IDs, and the entire ID-mapping layer rests on them doing so. OSMnx's own `drive` filter is used in the Overpass query so both graphs see an identical way set |
| Connectome file | `proofread_connections_783.feather` (852MB, Zenodo record 10676866) — *not* `flywire_synapses_783.feather` (9.5GB) | The proofread file is already aggregated per (neuron pair, neuropil); summing over neuropils yields exactly the per-pair weight wanted, without downloading and grouping 130M individual synapses |
| FlyWire account | Not a prerequisite for running Dipteran | Both the connectome and the annotation table are openly downloadable. The use terms still apply — see `DATA_LICENSE.md` |
| Annotation source | `Supplemental_file1_neuron_annotations.tsv` from `flyconnectome/flywire_annotations` | Carries `root_id` against `cell_type`/`hemibrain_type`/`super_class`/`cell_class`, which is what cluster resolution needs |
| Adjacency orientation | `W[i, j]` is presynaptic *j* onto postsynaptic *i* — rows postsynaptic, columns presynaptic | Matches the `W @ v` in §5's LIF step. Getting this backwards silently runs the brain in reverse |
| Matrix dtype | **fp32** (`compute.dtype`), not fp16 | Measured: fp32 costs 116.5 MiB against fp16's 87.3 MiB and is 5% slower per sparse matmul (0.989 vs 0.943 ms). Both are trivial against 8GB. fp16 products saturate to `inf` within a couple of LIF sub-ticks unless every step is rescaled, because weights are raw aggregate synapse counts up to 2405 and in-degree runs to hundreds. 29 MiB is not worth that hazard |
| Synapse-count threshold | `compute.min_syn_count`, default **0** (keep all 15.1M pairs) | The median connected pair in v783 has only *two* synapses and the FlyWire papers treat ≥5 as a reliable connection, so ~80% of the matrix is below their own reliability bar. Default stays at 0 to honour the maximum-fidelity decision above, but the knob exists because Phase 2's dynamics may well be dominated by that weak tail |
| Transmitter signs | Per *presynaptic neuron*, from the annotation table's consensus `top_nt` | Dale's principle: one neuron releases one fast transmitter. The per-neuron consensus is far less noisy than per-synapse predictions, and it avoids reading or reducing six probability columns over 16.8M rows. 40.2% of pairs come out inhibitory |
| Glutamate sign | Inhibitory | The dominant postsynaptic glutamate receptor in the fly CNS is the glutamate-gated chloride channel GluCl-α. ACh excitatory, GABA inhibitory as usual; monoamines are left excitatory-signed as a placeholder, since Phase 2 has PAM/PPL1 dopamine acting on weights rather than as direct current |

## 10. Roadmap

Phases are dependency-ordered; each should be a mergeable milestone.

**Phase 1 — Basic Build & Backend Foundations** — complete. Measured results in [`PHASE1_REPORT.md`](PHASE1_REPORT.md); `python check_phase1.py` re-asserts every item below.

- [x] Get the repo skeleton running: the mock `main.py` + `index.html` scaffold, end to end, as the baseline before real work starts.
- [x] Document hardware requirement (CUDA GPU, VRAM sized for a ~139k-node sparse adjacency matrix run alongside SUMO).
- [x] FlyWire data-access instructions — `DATA_LICENSE.md`. An account turned out not to be required; the v783 archive and the annotation table are both openly downloadable, and `dipteran/connectome.py` fetches them.
- [x] Install SUMO, OSMnx, networkx, CUDA-enabled torch, FastAPI.
- [x] Extract the OSMnx/networkx graph for the configured A→B route.
- [x] Build the SUMO network via `netconvert` from the same OSM extract — literally the same local `.osm.xml`, not a second download.
- [x] Build the ID-mapping layer between OSMnx edges and SUMO edges: way-ID index from `origId`, then geometry refinement.
- [x] Fetch the connectivity graph and build the GPU-resident sparse adjacency matrix (`W`), one signed weight per connected neuron pair.
- [x] Resolve named clusters to real FlyWire root IDs from the v783 annotation table, producing the `sensory_ids`/`motor_ids` lookup tables.
- ~~**Risk**: the OSMnx↔SUMO ID mapping is an unproven integration point~~ — it was indeed the hard part and did need custom code; see §11.

**Phase 2 — Finish the Backend**
- Implement `FlyBangaloreNavigator` against the real adjacency matrix on GPU.
- Calibrate LIF parameters (`v_thresh`, `v_reset`, `decay`) against observed spike rates.
- Wire PAM/PPL1 reinforcement into synaptic weight updates.
- Run SUMO with background traffic agents on the synced network.
- Implement the hybrid motion model (continuous heading/position + soft edge-pull).
- Wire all sensory inputs each tick (odor, compass, optic flow, looming, mechanosensory, acoustic) from live OSMnx/SUMO state.
- Replace the mock random-walk WebSocket streamer in `main.py` with real `FlyBangaloreNavigator` + SUMO output.
- Finalize the JSON payload schema (extend the mock's fields with `nearby_agents` and any additional brain telemetry the HUD needs).
- **Risk**: SUMO's default step length (1s) is far coarser than the SNN's 20ms tick — needs `--step-length` tuned down, with a performance check that SUMO can actually keep up at that resolution.

**Phase 3 — Frontend & Testing**
- Wire `index.html`'s existing MapLibre/HUD scaffold to the real payload schema.
- Add the background SUMO traffic rendering layer.
- Sanity checks: plausible spike rates, directionally-correct steering response to sensory input, Giant Fiber firing on a staged looming test.
- End-to-end run: full pipeline (SUMO + SNN + FastAPI + frontend) driving the agent from Point A toward Point B without crashing the server.

**Phase 4 — Close Off & Ship on GitHub**
- Update README to reflect the final architecture and drop any remaining mock-scaffold framing.
- Add a license, `.gitignore` (exclude any fetched FlyWire data/credentials), and a short setup guide covering the Phase 1 prerequisites.
- Final pass confirming no FlyWire connectome data or credentials are committed to the repo.
- Tag a release and publish.

## 11. Known Risks / Blockers

Resolved during Phase 1 — kept here with their outcomes rather than deleted, since the reasoning is the useful part:

- ~~**FlyWire data license**~~ — **resolved**: treated as CC BY-NC 4.0, the stricter of two conflicting upstream statements. No connectome data and no artifact derived from it (adjacency matrix, neuron index, resolved cluster table) is committed or shared; every contributor fetches it themselves. Recorded in `DATA_LICENSE.md` with the required citations.
- ~~**OSMnx↔SUMO ID mapping**~~ — **resolved**, and it did need custom code. The join is two-stage: `netconvert --output.original-names true` writes the source OSM way ID onto every lane as an `origId` param, which indexes candidates; geometry then picks the segments that actually overlap. Way ID alone is far too coarse — one long way becomes hundreds of OSMnx edges *and* hundreds of SUMO edges, and a pure ID join maps every one of the former to every one of the latter. See `PHASE1_REPORT.md` for the measured numbers.
- ~~**GPU/VRAM sizing**~~ — **resolved**: a per-connected-pair matrix, not a per-synapse one. Measured footprint is in `PHASE1_REPORT.md`; it is nowhere near the 8GB budget.
- ~~**No zero-setup path anymore**~~ — **partly softened**: a CUDA GPU and a SUMO install are still required, but a FlyWire account turned out *not* to be. Both the v783 connectome and the annotation table are openly downloadable, so `python -m dipteran.connectome` fetches them unattended. README still needs updating to match (Phase 4).

Still open:

- **Host RAM, not VRAM, is the binding constraint.** This was expected to be a GPU-memory problem and is not: the adjacency matrix takes 116 MiB of 7.93 GiB. But WSL2 defaults to half of physical memory (7.7 GiB of 15.3 GiB here), and `duarouter` alone peaks at 3.85 GB loading the 862 MB network while the OSMnx graph costs 3–4 GB resident. Phase 1 runs fine one stage at a time; Phase 2 needs SUMO resident *alongside* the SNN, so raise the allocation in `.wslconfig` first. Two concurrent Phase 1 runs were OOM-killed discovering this.
- **SUMO tick resolution vs. the 20ms SNN loop**: SUMO's default step length is 1 second. The design decision is now made — SUMO at `--step-length 0.1`, SNN at 50Hz, batched `traci` control every 5th SNN tick — but it is unproven for performance on a network this size (385k edges) and has not been run with traffic demand yet. What *is* now measured is the SNN half: 10 sparse LIF sub-ticks cost 9.4 ms of the 20 ms tick, about 2× real-time headroom. SUMO has to fit in the remaining ~10 ms, and there is no slack for a surprise. Phase 2.
- **OSMnx graph connectivity**: the extract has 239 strongly-connected components. The configured origin and destination are in the same one and `duarouter` routes between them, but a vehicle that wanders into a smaller component can get stranded. Phase 2's hybrid motion model needs to handle that rather than assume it away.
- **Monoamine signs are placeholders**: octopamine, serotonin and dopamine edges are currently excitatory-signed in `W`. That is wrong as biology and only harmless while PAM/PPL1 are unwired; Phase 2's reinforcement loop has to make dopamine act on weights instead.
