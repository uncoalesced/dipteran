<p align="center">
  <img src="assets/banner/dipteran-banner-github.png" alt="Dipteran" width="760">
</p>

# Dipteran

A fruit fly's brain drives a car through Bangalore traffic.

The 138,639-neuron FlyWire reconstruction of an adult *Drosophila
melanogaster* runs as a spiking network on the GPU. Its sensory neurons get
fed a live simulation of Bangalore: an odour gradient pointing at wherever you
told it to go, a compass, optic flow, looming auto-rickshaws, horn blasts,
potholes. Its descending motor neurons are read back out as steering and
throttle. There's no route planner anywhere in the loop. The fly smells its
destination and tries to get there, the way a real one would smell fermenting
fruit.

It's bad at this, which is most of the point.

---

## Setup

You need a CUDA GPU, roughly 11 GB of free RAM, and about 30 GB of disk for
the OSM extract, the SUMO network and the connectome. Linux or WSL2 is the
tested path. Native macOS and Linux support, and support for non-NVIDIA GPUs,
are coming in a future release. Neither the connectome nor the road network
is in this repo, for licensing reasons covered in
[DATA_LICENSE.md](DATA_LICENSE.md); both get downloaded on first run.

**On Windows, everything below runs inside WSL2, not in a native PowerShell
or cmd prompt.** There's no native-Windows path yet (that's the "coming
soon" above). Open a WSL2 terminal first -- from PowerShell or cmd, run
`wsl` to drop into your Linux distro (install one with `wsl --install` if
you haven't), or open the distro directly from the Start menu. Everything
from here on is a bash command run inside that WSL2 shell.

**1. Python dependencies.** Run the setup script:

```bash
bash scripts/setup_wsl.sh
```

This creates `.venv`, installs `requirements.txt`, and installs torch from
the CUDA 12.8 index (required -- see the comment at the top of
`requirements.txt`). If the repo is sitting on a Windows drive
(`/mnt/c/...`, `/mnt/e/...`, etc.) it stops immediately and prints the two
commands that fix that, rather than failing partway through a 20-minute
install with a `Operation not permitted` error that gives no hint why: in
short, `uv`/pip reliably fail installing into a venv on a DrvFs-mounted
Windows drive, so the script insists the repo live on WSL2's own
filesystem (e.g. `~/dipteran`) first.

Prefer to do it by hand, or troubleshooting a failure? That's all the
script above does:

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
```

The torch line has to come from the CUDA 12.8 index and it has to be separate.
This was built on a Blackwell card (sm_120), and older wheels ship no kernels
for it. They import fine and then die at the first kernel launch, which is a
genuinely annoying way to find out.

**2. SUMO.** Not pip-installable. Get it from your package manager so
`netconvert` lands on PATH, or point `sumo.binary_dir` in `config.yaml` at
wherever you put it.

```bash
sudo apt-get install sumo sumo-tools sumo-doc
```

**3. Build the road network.** One Overpass query, then netconvert, then the
OSMnx-to-SUMO edge join. A few minutes, all of it into `cache/`.

```bash
python -m dipteran.roads
```

**4. Fetch the connectome.** Pulls the FlyWire v783 edge list from Zenodo and
the annotation table from GitHub, collapses 54.5 million synapses into one
signed weight per neuron pair, and resolves the 19 named sensory and motor
populations to real root IDs.

```bash
python -m dipteran.connectome
```

You don't need a FlyWire account for the bulk archive. Redistribution is a
different matter, which is why `data/` and `cache/` are gitignored in full.

**5. Optional: the brain visualiser.** Builds the per-neuropil point cloud the
HUD renders. Skip it and the HUD just tells you it's missing.

```bash
python scripts/extract_neuropils.py
```

If you're moving the multi-GB artifacts off a slow mount, set
`DIPTERAN_DATA_ROOT` and every relative path in `config.yaml` resolves against
that instead of the repo root.

---

## Launch

```bash
python main.py
```

Serves the map at http://127.0.0.1:8000 in mock mode: a random walk, no
connectome, starts instantly. Good for checking the frontend works and nothing
else.

For the real thing:

```bash
DIPTERAN_LIVE=1 python main.py
```

Mock is the default because live mode costs about 28 seconds to build the
connectome plus another 75 to 130 for SUMO to parse the 862 MB network, which
isn't what you want from `python main.py` with no arguments. Either way,
nothing boots until you've picked a route, so you won't pay that cost by
accident.

---

## How to use it

Open the page. It asks two questions and then gets out of the way.

1. **Click your destination.** This is the odour source. The fly gets a
   concentration gradient toward it and nothing else: no waypoints, no turn
   list. Anywhere inside the road network extract works.
2. **Click your starting point.** The fly spawns on the nearest road to your
   click, dropped into 300 background SUMO vehicles.
3. **Press "Release the fly."** In live mode that button is the boot trigger.
   The connectome build and the SUMO parse both happen behind it, so expect
   two or three minutes of status text before anything moves.

There's a "Use default" button if you just want the route everything was
measured on, Central Silk Board to Tin Factory, 10.6 km.

Once it's running you get three things. The map, with the fly rotated to its
actual heading and background traffic drawn as a GeoJSON layer. The point
cloud top-left, where each neuropil lights up with its own firing rate. And
the vitals panel bottom-right, collapsed by default, which opens into the PPL1
pain bar, the DNp01 giant-fiber bar, and a running log of what the brain just
did to itself.

One SUMO process and one GPU-resident connectome exist per server, so everyone
connected watches the same fly, and changing the route means a restart.

### Other things you can run

```bash
python tests/test_units.py                     # 74 unit tests, no FlyWire download needed
python scripts/run_route.py --max-minutes 60   # full route, with drift and geography reporting
python scripts/probe_diagnostics.py            # reproduces every steering figure below
python -m dipteran.navigator                   # the LIF gain calibration sweep
python -m dipteran.roads                       # rebuilds and re-verifies the road network
```

---

## How it works

Two road-network representations, built from one shared OSM extract and joined
by ID. OSMnx gives the graph the fly reasons over, bearing and distance to
goal. netconvert gives SUMO the network its traffic actually drives on. The
join matches OSM way IDs first to narrow the candidates cheaply, then uses
polyline distance to decide which of those candidates really overlap. One
long OSM way splits into many edges in both graphs, so a pure way-ID join is
far too coarse.

The live loop never parses `.net.xml` itself. It can't afford to: `sumo` alone
measures 7.4 GB resident against an 11 GB ceiling, so every geometric question
gets asked of the already-running SUMO process over traci rather than a second
resident copy of the network.

The connectome becomes a sparse CSR tensor `W` where `W[i,j]` is presynaptic
`j` onto postsynaptic `i`, so one LIF step is `W @ v`. Signs come from each
presynaptic neuron's consensus transmitter, following Dale's principle instead
of going per-synapse. Dopaminergic, serotonergic and octopaminergic rows are
dropped from that fast path entirely and act through a plastic overlay, which
is closer to what monoamines do anyway.

The vehicle is a free body: `moveToXY` with `keepRoute=2`, soft-pulled toward
the nearest road rather than constrained to a route. It can end up somewhere
stupid. That's allowed.

### Steering, and why the first version of it was wrong

The original readout was `rate(DNa01) - rate(DNa02)`. It looked reasonable and
it was a category error. DNa01 and DNa02 are functionally distinct circuits
with largely non-overlapping inputs, and both patterns pooled across the left
and right copies, so the difference was measuring baseline excitability
between two cell types rather than any left-right asymmetry. Probed at rest
with no directional input at all it returns +0.063, which is almost exactly
the 0.075 constant turn rate the vehicle had been orbiting at for 25,727
frames.

It's now `rate(DNa02_right) - rate(DNa02_left)`, side-split, matching the
relationship the 2024 Nature paper measured directly against rotational
velocity. A calibrated `snn.steer_bias` of 0.191 comes off the top, because
DNa02-left sits at 0.0 Hz while DNa02-right runs at about 95. That isn't a
reconstruction gap: in-degree is 742 against 751 and input weight 12,237
against 13,116, so both cells are reconstructed about equally well. The left
one is just quiet.

Directional drive goes into PFL3, not the odour PNs. Six projection neurons
can't move a cell with in-degree 742, and measuring it confirmed as much: they
shift DNa02 by 2.2 Hz in the wrong direction. PFL3 gives 40.3 Hz of monotonic
authority. The projection is contralateral, 12 synapses from PFL3-right onto
DNa02-left and zero ipsilateral, read straight out of `W`, so a goal on the
right drives the *left* lobe. There's a unit test asserting that sign, because
a plausible-looking backwards formula is exactly the bug that got caught twice
during design.

Measured end to end: -4.1°/s for a goal 90° left, +5.2°/s for 45° right,
+0.005°/s dead ahead.

### Reward

Dopamine scales with how close the goal is: full strength within 500 m,
falling to 2% at 15 km, mapped through `snn.pam_current_min/max` because PAM's
current-to-firing transfer is superlinear near zero. Inject 0.02 raw and you
get 0.5% of peak firing, not 2%.

The progress threshold that gates it dropped from 100 m to 12 m. At the
measured median speed the old one fired at most twice across a 25,727-frame
run, which meant plasticity ran homeostasis-only for essentially the entire
experiment. Punishment comes from PPL1 on stalls, potholes and
near-collisions.

---

## Where it actually is

The unit suite passes, ruff is clean, and each module self-checks on
`python -m dipteran.<module>`. The full loop runs at 1.27x real time (78.6 ms
of work in a 100 ms frame) for 25,727 consecutive frames with no stalls and
bounded plastic weight drift.

Phase 3 fixed the steering, and the steering now demonstrably points the right
way. Two things about it are still wrong. `steer` saturates on the left,
because DNa02-left is floored, so left turns come out roughly constant no
matter how large the error is. And the dead zone past 135° of error relies on
a small residual right-drift to sweep out of it, which works but isn't a
mechanism anyone designed.

**No learning has been demonstrated.** Episodic training is implemented and
the dopamine channel can now fire, which it previously couldn't. Nobody has
run the multi-episode experiment yet. Don't read the reinforcement section
above as a claim that anything has actually been learned.

---

## Where the fly's brain came from

Worth being precise about, because "we took a fly's brain" invites the wrong
picture.

The dataset came from an irreversible physical dissection of one adult female
*Drosophila melanogaster*, done at HHMI Janelia by Davi Bock and Zhihao Zheng
and published in Cell in 2018 as FAFB. The fly was killed and its 1 mm brain
micro-dissected out of the head capsule. Glutaraldehyde and formaldehyde
crosslinked the proteins; osmium tetroxide and heavy metal stains bound to the
lipid membranes to make them opaque to electrons. The tissue was dehydrated
and embedded in epoxy resin, at which point it was a piece of plastic-encased
carbon and metal. An ultramicrotome with a diamond blade then sliced that
block into 7,050 consecutive sections at 40 nanometres each, about a
two-thousandth the thickness of a sheet of paper, and every slice was
photographed on a custom serial-section transmission electron microscope at
4 nm per pixel. The output was over 100 terabytes of grey cross-sections.

Turning that into a searchable graph took most of a decade and landed as a
package of Nature papers in October 2024. The FlyWire consortium was co-led by
Sebastian Seung and Mala Murthy at Princeton, with Gregory Jefferis and
Philipp Schlegel at the MRC LMB in Cambridge. 3D U-Nets traced cell boundaries
across the slices. Automated segmentation made thousands of errors, fusing
adjacent neurons and severing thin axons, so 287 researchers across 76 labs
proofread the result by hand on an open web platform. What came out: 139,255
proofread neurons, roughly 54.5 million chemical synapses, and a
machine-learning transmitter call for nearly every one of them.

What the dataset doesn't contain is most of what makes a neuron work. There
are no ion channel dynamics in it, no conductances or resting voltages. It has
nothing on volume transmission, the extracellular washes of dopamine and
neuropeptides that diffuse nowhere near a classical synapse. Gap junctions are
generally invisible at 40 nm, so they aren't there either, and neither is
metabolic state, second messengers, or transcription. Glia are absent
entirely.

So this project supplies all of that itself, badly. The resting potential,
firing threshold and decay rate in `config.yaml` are numbers somebody picked.
The circuit topology is real and evolutionarily optimised; the physics laid
over it is invented.

Searle's version of the problem: a simulation of a hurricane can compute
Navier-Stokes to arbitrary precision and not one molecule of water gets wet.
Simulating a membrane potential computes the arithmetic of neural
transmission. It doesn't reproduce the substrate. Nothing here is a fly.
Nothing here is conscious. If it looks like it's trying to get somewhere,
that's the topology doing its job.

---

## Layout

| Path | What it is |
| --- | --- |
| `dipteran/roads.py` | OSM extract, both graph representations, the ID join, a from-scratch `.net.xml` parser |
| `dipteran/connectome.py` | FlyWire fetch, adjacency build, the 19 named cell populations |
| `dipteran/navigator.py` | The LIF step and the three-factor reinforcement rule |
| `dipteran/world.py` | SUMO over traci, sensory encoding, motor decoding, the vehicle's motion |
| `dipteran/config.py` | Pydantic schema for `config.yaml` |
| `main.py` | FastAPI, the route endpoints, the WebSocket stream |
| `index.html` | The whole frontend, one file, no build step |
| `scripts/probe_diagnostics.py` | Every steering and reward figure in this README |

Everything tunable lives in [config.yaml](config.yaml): coordinates, bounding
box, device and dtype, every LIF and reinforcement parameter, every motor
transfer scale, all file paths. No hardcoded coordinates or tuning constants
anywhere in the code. Run `python -m dipteran.config` and it'll tell you if the
file is wrong.

---

## Contributing

Bug reports, pull requests and platform-support help are welcome.
[CONTRIBUTING.md](CONTRIBUTING.md) has the setup, the checklist a PR needs to
pass, and the project's conventions.

---

## Licensing

The code is MIT. The data isn't, and the two are independent of each other.
FlyWire gets fetched at runtime under your own agreement and should be treated
as CC BY-NC 4.0. The map marker derives from
[TuragaLab/flybody](https://github.com/TuragaLab/flybody), which is Apache-2.0
and therefore the one asset actually committed here. OpenStreetMap is ODbL.
[DATA_LICENSE.md](DATA_LICENSE.md) has the detail and the citations you owe.

---

<p align="center"><sub>Engineered by uncoalesced.</sub></p>
