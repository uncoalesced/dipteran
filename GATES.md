# GATES — Phase 1 hardening pass

Scope: Phase 1 shipped and all seven exit criteria pass. This pass closes the
gaps between "done" and "done well" found by auditing the shipped repo. Solo
mode, tree depth 3.

The skill's `scripts/gate-check.mjs` is not present in this install (only
`SKILL.md` ships), so CHECK lines are run as shell commands and the deciding
output is pasted into EVIDENCE.

Audited defects this pass must close:

1. Seven references in committed files point at `cache/*.py` and
   `cache/provision.sh`, which are gitignored — including a runtime `print()`
   in `check_phase1.py` telling users to consult a file a fresh clone does
   not have.
2. `README.md:78` states a FlyWire/CAVE account is required to fetch the
   connectome. Phase 1 established that is false.
3. `DATA_LICENSE.md:35` states the annotation table is ~25 MB; it is 31.7 MB.
4. No tests exist for any of the non-trivial pure logic — the `.net.xml`
   parser, the point-to-polyline geometry the whole OSM↔SUMO join rests on,
   the `osmid` normaliser, the way-index strategy selection, the BFS.
5. `compute.min_syn_count > 1` is a shipped code path that has never run.

---

## Status: partly closed during Phase 2 (2026-09-14)

This pass was originally stopped by the user with "You can stop, I'm happy
with how it is so far", leaving all twelve gates abandoned. Phase 2 reopened
four of them, because Phase 2 stacks a navigator directly on top of the
untested code this file flags — the `.net.xml` parser, the point-to-polyline
geometry and the BFS — and adding weight to untested foundations was the one
item worth paying for before building further.

That judgement was vindicated immediately: the first test written against
`nearest_edge` failed, and the root cause was a real defect in a function
`PHASE1_REPORT.md` explicitly tells Phase 2 to rely on. See G2.

**Closed in Phase 2: G1, G2, G10, G11.** Evidence recorded inline below.

**Still deferred: G3, G4, G5, G6, G7, G8, G9, G12.** These are documentation
staleness, provisioning hygiene, and a never-exercised config path. None
blocks Phase 2 or Phase 3, and PLAN.md's Phase 4 already owns the README and
setup-guide work most of them belong to.

ABANDON: G3 deferred to Phase 4; committed references to gitignored cache/ paths remain
ABANDON: G4 deferred to Phase 4; provisioning still only in gitignored cache/provision.sh
ABANDON: G5 deferred to Phase 4; bench_tick not moved into the repo. Phase 2 added
         scripts/bench_sync.py, which measures the SNN tick as one term of the
         full sync budget, but that is not the script this gate names
ABANDON: G6 deferred to Phase 3; min_syn_count > 1 still never exercised, now folded
         into the planned fidelity-versus-noise sensitivity analysis, because
         changing it re-normalises every row and invalidates the calibrated
         snn.gain and the vehicle transfer scales together
ABANDON: G7 deferred to Phase 4; README.md:78 still claims a FlyWire account is required
ABANDON: G8 deferred to Phase 4; DATA_LICENSE.md:35 still says ~25 MB, actual 31.7 MB
ABANDON: G9 deferred to Phase 4; Phase 1 report figures not audited as a set.
         PHASE2_REPORT.md's own figures were audited against command output
ABANDON: G12 deferred; Phase 2 work is committed locally, but the push to
         origin/main is the user's to make

---

- [x] **G1** Unit tests cover the non-trivial pure logic and pass, with no
      dependency on the 1 GB downloaded corpus.
      CHECK: `cd /mnt/e/Dipteran && python tests/test_units.py`
      EXPECT: exit 0, every test passes, count reported
      EVIDENCE: `57/57 passed`, exit 0. Covers point_to_polyline_m (6 cases
      including the t-clamp and a zero-length segment), _osmids (6), the
      SumoNet parser and projection (8), _way_index strategy selection (3),
      the reachability BFS (6), world geometry (12), state classification and
      telemetry (7), the firing-rate/command boundary (8), and that every
      resolved cluster has a HUD region (an unmapped one is dropped silently).
      None touches the downloaded corpus; the SUMO cases run against a
      synthetic .net.xml written to a temp dir.

- [x] **G2** The synthetic `.net.xml` fixture exercises the real `SumoNet`
      parser end to end: origId params, lane shapes, projection,
      `nearest_edge`, `reachable`.
      CHECK: `python tests/test_units.py` includes SumoNet cases
      EXPECT: parser/geometry/BFS assertions present and passing
      EVIDENCE: present and passing. The fixture covers internal-edge and
      shapeless-edge rejection, multi-value origId params on lanes, length as
      the max over lanes, UTM projection landing inside the configured bbox,
      the local frame preserving a 100 m edge, one-way BFS reachability, an
      unreachable island, and connections referencing unindexed internal
      edges.
      This gate paid for itself. `test_nearest_edge` failed on first run:
      `nearest_edge` was a nearest-*vertex* query, returning whichever edge
      owned the closest shape point. A query 2.000 m from edge `300#0` came
      back as `200#0` at 49.981 m. PHASE1_REPORT.md tells Phase 2 to use that
      exact call to decide which edge the vehicle is on. Fixed in
      dipteran/roads.py to resolve candidates by true point-to-polyline
      distance within an exact bound of `d0 + max_segment/2`; Phase 1's
      ID-map statistics are unchanged by the fix.

- [ ] **G3** No committed file references a gitignored path.
      CHECK: `git grep -n "cache/bench\.py\|cache/provision\.sh\|cache/check_patterns\|cache/check_pairs\|cache/probe\.py"`
      EXPECT: no matches
      EVIDENCE: pending

- [ ] **G4** Toolchain provisioning is reproducible from the repo, not from a
      gitignored scratch file.
      CHECK: `git ls-files scripts/ && bash -n scripts/setup_wsl.sh`
      EXPECT: file tracked, exit 0
      EVIDENCE: pending

- [ ] **G5** The tick benchmark that PHASE1_REPORT.md and `check_phase1.py`
      both cite is in the repo and runs.
      CHECK: `git ls-files scripts/bench_tick.py && python scripts/bench_tick.py`
      EXPECT: tracked, exit 0, prints ms per matmul for fp16 and fp32
      EVIDENCE: pending

- [ ] **G6** `compute.min_syn_count > 1` actually works: it drops weak pairs
      and every surviving pair meets the threshold.
      CHECK: run `aggregate_pairs` with `min_syn_count=5` and assert
      EXPECT: fewer pairs than unthresholded, `count.min() >= 5`
      EVIDENCE: pending

- [ ] **G7** `README.md` no longer claims a FlyWire account is required.
      CHECK: `git grep -n "your own FlyWire/CAVE account to fetch" README.md`
      EXPECT: no matches; replacement text states the data is openly
      downloadable and points at DATA_LICENSE.md
      EVIDENCE: pending

- [ ] **G8** `DATA_LICENSE.md` states the measured annotation file size.
      CHECK: `git grep -n "31.7 MB" DATA_LICENSE.md`
      EXPECT: one match, and no "~25 MB"
      EVIDENCE: pending

- [ ] **G9** Every number stated in PHASE1_REPORT.md is re-measured against
      `cache/phase1_evidence.json` or a command, not written from memory.
      This is the skill's report-audit rule; the shipped report was written
      across several sessions and its figures have not been checked as a set.
      CHECK: compare each figure to the evidence JSON
      EXPECT: every figure matches, or is corrected
      EVIDENCE: pending

- [x] **G10** `ruff check` clean over all tracked Python.
      CHECK: `python -m ruff check dipteran/ check_phase1.py main.py scripts/ tests/ --select F,E9`
      EXPECT: "All checks passed!"
      EVIDENCE: `All checks passed!` over dipteran/ (config, connectome,
      roads, navigator, world), scripts/ (bench_sync, extract_neuropils,
      run_route), tests/, check_phase1.py and main.py. One finding was raised
      and fixed on the way: an unused `import math` left in navigator.py after
      the gain-selection rule was rewritten.

- [x] **G11** All seven Phase 1 exit criteria still pass after every change in
      this pass. No regression is acceptable.
      CHECK: `DIPTERAN_DATA_ROOT=/opt/dipteran python check_phase1.py`
      EXPECT: 7 PASS, 0 FAIL, exit 0
      EVIDENCE: 7 PASS, 0 FAIL, exit 0 — config 0.0s, scaffold 1.0s, osmnx
      16.4s, sumo_net 22.0s, id_map 251.5s, connectome 19.2s, cell_ids 2.3s.
      Final run, after both the `nearest_edge` fix and the added
      `mechano_bristle` cluster; cell_ids reports "9 sensory and 6 motor
      clusters resolved", with mechano_bristle matching 1417 rows through its
      declared fallback. The ID-map figures are unchanged from Phase 1's:
      99.2% coverage, 7.16 m median midpoint gap, 7.74 m p90 and 68.8%
      pure-geometry agreement — so the fix removed a real hazard without
      moving any Phase 1 number.

- [ ] **G12** Work committed and pushed to origin/main; working tree clean.
      CHECK: `git status --short && git log --oneline -1 && git rev-parse HEAD origin/main`
      EXPECT: empty status, HEAD == origin/main
      EVIDENCE: pending
