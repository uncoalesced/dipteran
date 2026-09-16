# Contributing to Dipteran

Thanks for taking a look at this. It's a one-person project so far, so
outside eyes on it are genuinely useful, not just a formality.

## Before you write code

For anything beyond a small fix, open an issue first and describe what
you want to change. This project has a specific, opinionated architecture
(described in the README's "How it works" section), and some things that
look like bugs are deliberate trade-offs measured against real numbers.
Saying what you're planning before you write it saves both of us a
rejected PR.

## Setup

Follow the README's Setup section, step 1 -- it has separate,
copy-paste-safe command blocks for Linux/WSL2 (bash) and native Windows
(PowerShell).

You don't need a CUDA GPU or the FlyWire connectome to work on most of the
codebase. `python main.py` (mock mode) runs the server, the route picker
and the HUD with a random-walk streamer, no GPU required. The unit suite
also runs without a GPU or a FlyWire download:

```bash
python tests/test_units.py
```

## Before you open a PR

Run the unit suite and the linter. Both have to be clean:

```bash
python tests/test_units.py
python -m ruff check dipteran/ main.py scripts/ tests/ --select F,E9
```

If your change touches a module with its own self-check
(`dipteran/config.py`, `dipteran/roads.py`, `dipteran/connectome.py`,
`dipteran/navigator.py`), run it:

```bash
python -m dipteran.<module>
```

If your change touches steering, reward, or anything else with a number
attached to it in the README, back it up the same way the existing code
does: measure it, don't eyeball it. `scripts/probe_diagnostics.py` is the
existing example of that pattern.

## Conventions

- No hardcoded coordinates or tuning constants in code. Everything tunable
  lives in `config.yaml`, validated by `dipteran/config.py`.
- Comments that cite `PLAN.md §N` are referring to the design doc, which
  isn't in the repo (see the README and `CLAUDE.md`). If you don't have
  it, ask in your issue or PR and I'll pull the relevant section into the
  conversation.
- Keep changes to `dipteran/roads.py`'s two-graph join, the connectome's
  `W` construction, and the steering readout especially well-commented --
  these are the parts that look simple and are not.

## What's useful right now

Check `TODO`-style notes in the repo and open issues for the current
short list. Platform support (native macOS/Linux, non-NVIDIA GPUs) and
performance work are both welcome but worth discussing in an issue first,
since they touch assumptions baked pretty deep into the pipeline.

## License

The code is MIT (see `LICENSE`); contributions are accepted under the same
license. The FlyWire connectome and OpenStreetMap data are not MIT and
have their own terms -- see `DATA_LICENSE.md` -- so don't commit any of
that data itself, only code that fetches or processes it.
