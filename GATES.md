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

## Status: stopped by the user before any gate was worked

The user ended this pass with "You can stop, I'm happy with how it is so far."
No gate below was completed, and none should be read as met. Nothing was
half-built either: the pass had produced only this file and an audit, so the
repo is exactly as it was at commit `98172fd`, with Phase 1 complete and all
seven of its own exit criteria passing.

The five defects listed above are real, reproduced, and unfixed. Four of them
land naturally in PLAN.md's Phase 4 ("Update README to reflect the final
architecture", "a short setup guide covering the Phase 1 prerequisites"); the
missing tests do not, and are the item I would pick up first.

ABANDON: G1 user stopped the pass; no tests written
ABANDON: G2 user stopped the pass; no SumoNet fixture written
ABANDON: G3 user stopped the pass; 7 committed references to gitignored cache/ paths remain
ABANDON: G4 user stopped the pass; provisioning still only in gitignored cache/provision.sh
ABANDON: G5 user stopped the pass; bench_tick not moved into the repo
ABANDON: G6 user stopped the pass; min_syn_count > 1 still never exercised
ABANDON: G7 user stopped the pass; README.md:78 still claims a FlyWire account is required
ABANDON: G8 user stopped the pass; DATA_LICENSE.md:35 still says ~25 MB, actual 31.7 MB
ABANDON: G9 user stopped the pass; report figures not audited as a set
ABANDON: G10 not run
ABANDON: G11 not re-run; last full run was 7/7 PASS at 98172fd
ABANDON: G12 only this file is uncommitted

---

- [ ] **G1** Unit tests cover the non-trivial pure logic and pass, with no
      dependency on the 1 GB downloaded corpus.
      CHECK: `cd /mnt/e/Dipteran && python tests/test_units.py`
      EXPECT: exit 0, every test passes, count reported
      EVIDENCE: pending

- [ ] **G2** The synthetic `.net.xml` fixture exercises the real `SumoNet`
      parser end to end: origId params, lane shapes, projection,
      `nearest_edge`, `reachable`.
      CHECK: `python tests/test_units.py` includes SumoNet cases
      EXPECT: parser/geometry/BFS assertions present and passing
      EVIDENCE: pending

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

- [ ] **G10** `ruff check` clean over all tracked Python.
      CHECK: `python -m ruff check dipteran/ check_phase1.py main.py scripts/ tests/ --select F,E9`
      EXPECT: "All checks passed!"
      EVIDENCE: pending

- [ ] **G11** All seven Phase 1 exit criteria still pass after every change in
      this pass. No regression is acceptable.
      CHECK: `DIPTERAN_DATA_ROOT=/opt/dipteran python check_phase1.py`
      EXPECT: 7 PASS, 0 FAIL, exit 0
      EVIDENCE: pending

- [ ] **G12** Work committed and pushed to origin/main; working tree clean.
      CHECK: `git status --short && git log --oneline -1 && git rev-parse HEAD origin/main`
      EXPECT: empty status, HEAD == origin/main
      EVIDENCE: pending
