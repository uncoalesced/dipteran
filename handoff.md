# Phase 1 Handoff — Dipteran

You're picking this up cold. Read `README.md` and `PLAN.md` (both in this repo) first — architecture, biological mapping, roadmap, Decisions Log (§9), Known Risks (§11) all live there. This file only adds what those two don't have yet: decisions from two planning rounds run after PLAN.md was written, condensed to what Phase 1 needs. Fold the table below into PLAN.md §9 as you go so it stays the single source of truth.

**Your job**: execute PLAN.md's Phase 1 in full, end to end, autonomously. Do not start Phase 2 — no real `FlyBangaloreNavigator` dynamics, no PAM/PPL1 wiring, no SUMO/SNN sync loop. Keep the mock `main.py`/`index.html` (from README.md's embedded scaffold, once materialized) as the running baseline until Phase 2 replaces it.

## Environment

- Hardware: RTX 5050 Laptop GPU, 8GB VRAM. This is enormous headroom for this workload (see Decisions) — don't spend time on GPU memory optimization.
- Platform: **WSL2 Ubuntu**, not Windows-native. SUMO/OSMnx/GDAL are far cleaner there; CUDA passthrough works out of the box.
  - Check first: `wsl -l -v` (lists *all* installed distros — `wsl --status` only shows the default, which on this machine is currently `docker-desktop`, i.e. Docker's own WSL integration, not a general dev distro). If no Ubuntu distro exists: `wsl --install -d Ubuntu`, reboot, then run `nvidia-smi` inside it before anything else — confirm the RTX 5050 actually shows up under CUDA passthrough.
  - Repo is at `E:\Dipteran`, auto-mounted at `/mnt/e/Dipteran` inside WSL2. Work there directly.
- Repo state right now: git initialized, `README.md` committed and pushed to `origin/main` (`github.com/uncoalesced/dipteran`, private). **`PLAN.md` and `.gitignore` are not committed yet** — add both first (gitignore below).

## Decisions since PLAN.md was written (2026-09-12 planning rounds)

| Topic | Decision |
|---|---|
| GPU/VRAM sizing | Non-issue. A proper adjacency matrix has one entry per *connected neuron pair* (aggregate synapse weight), not one per raw synapse — expect ~2–3M nnz, not 54.5M. At fp16 values + int32 indices that's tens of MB, nowhere near the 8GB budget. Don't build a per-synapse matrix. |
| FlyWire data access | Static v783 archive via FlyWire Codex/Zenodo download (Feather/Parquet edge list) — **not** the live CAVE query API for the bulk connectome. A CAVE/`caveclient` account is still worth having for annotation lookups (step 9 below) but isn't a gate for downloading the connectome. |
| FlyWire data license | Treat as **CC BY-NC 4.0** (attribution + non-commercial) — sources conflict (Zenodo metadata says CC-BY-4.0, flywire.ai/guidelines says CC BY-NC 4.0), NC is the binding conservative assumption. Never commit the raw connectome or a compiled matrix to the repo (already in .gitignore below). Cite Dorkenwald et al. 2024 and Schlegel et al. 2024 in a `DATA_LICENSE.md`. |
| Code license | MIT for Dipteran's own code, independent of the CC BY-NC 4.0 data license above. |
| Route config | `config.yaml`, Pydantic-validated — origin/destination/bbox/network_type/device/data paths as fields. No hardcoded lat/lon constants in code. |
| SUMO/SNN sync (Phase 2, don't implement yet — just don't design Phase 1 in a way that forecloses it) | SUMO at `--step-length 0.1` (10Hz), SNN at 50Hz (20ms), sample SUMO state + issue batched `traci` control every 5th SNN tick (100ms boundary), frontend interpolates for 60fps. |

## `.gitignore` — add this verbatim, commit before anything else

```
# Python & Environment
__pycache__/
*.py[cod]
.venv/
env/
.env

# Large Biological Datasets & Compiled Models (never commit these)
data/
cache/
*.csv
*.parquet
*.feather
*.pt
*.pth
*.npy
*.npz

# SUMO & OSM outputs
*.net.xml
*.rou.xml
*.poly.xml
*.osm
*.sumocfg

# Credentials & Tokens
*token*
*credential*
*.secret
```

## Phase 1 task order

1. **First commit**: add `PLAN.md` and the `.gitignore` above, commit, push.
2. **Materialize the scaffold**: write README.md's embedded `main.py` + `index.html` out as real files (they don't exist on disk yet). Confirm `python main.py` actually serves the mock WebSocket + MapLibre frontend before touching anything real — this is PLAN.md's own first Phase 1 line item.
3. **Toolchain**: `sudo apt-get install sumo sumo-tools sumo-doc`; CUDA-matched `torch`; `osmnx`, `networkx`, `fastapi`, `uvicorn`, `pydantic`.
4. **FlyWire**: create a Codex/CAVE account, download the static v783 edge list.
5. **`config.yaml`**: origin/destination/bbox/network_type/device/data-path fields, Pydantic model to load it.
6. **OSMnx extraction**: pull the drivable graph for the configured route.
7. **SUMO network**: `netconvert` on the same OSM extract.
8. **OSMnx↔SUMO ID-mapping layer** — the hardest, least-proven piece; budget real time here. Both graphs derive from the same OSM extract, so check whether SUMO's `.net.xml` edges carry an `origId`/shape attribute tracing back to the source OSM way ID before falling back to nearest-geometry coordinate matching.
9. **Connectome → GPU**: load the v783 edge list, aggregate to one weighted entry per neuron pair, build a `torch.sparse_csr_tensor` on `cuda:0`. Log actual VRAM used (sanity check against the "tens of MB" expectation above).
10. **Cell-ID resolution**: resolve `sensory_ids` (DM1/DM4, EPG, T4/T5, LC4/LPLC2, Johnston's organ aPNs) and `motor_ids` (DNa01, DNa02, DNp01, B1/B2, PAM, PPL1) to real FlyWire root IDs via Codex/CAVE annotation search — not invented indices.

## Exit criteria — Phase 1 is done when every one of these is actually checked, not assumed

- [ ] OSMnx graph loads cleanly for the configured route
- [ ] `netconvert` produces a valid, routable `.net.xml` — no fatal topology errors
- [ ] A deterministic OSM-edge↔SUMO-edge/lane lookup matches on geometry for a sample set of edges (write a small script that asserts this, don't eyeball it)
- [ ] Connectome loaded as a GPU-resident `torch.sparse_csr_tensor`, VRAM usage logged
- [ ] Every named `sensory_ids`/`motor_ids` cluster resolves to a valid row index — no placeholders left in

## Out of scope — do not do this in Phase 1

Real SNN loop replacing the mock streamer, LIF parameter calibration, PAM/PPL1 reinforcement wiring, implementing the SUMO/SNN sync loop (decision is made above, building it is Phase 2).

## When done

Update PLAN.md's roadmap checkboxes and Decisions Log in-repo, commit, push. Leave a short `PHASE1_REPORT.md` at repo root: what got built, what each exit-criteria check actually showed, anything that didn't go as planned (especially step 8 — report whatever the ID-mapping approach ended up being, since nobody has validated one yet).
