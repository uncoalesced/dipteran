## What this changes and why

<!-- The "why" matters more than the "what" here -- a lot of this codebase
     encodes measured trade-offs (see the README's "How it works" and
     "Steering" sections), so explain the reasoning, not just the diff. -->

## Related issue

<!-- Link it. If there isn't one, open one first for anything non-trivial;
     see CONTRIBUTING.md. -->

## Checklist

- [ ] `python tests/test_units.py` passes
- [ ] `python -m ruff check dipteran/ main.py scripts/ tests/ --select F,E9` is clean
- [ ] Any relevant module self-check (`python -m dipteran.<module>`) still passes
- [ ] No hardcoded coordinates or tuning constants -- new constants go in `config.yaml`
- [ ] If this changes steering, reward, or any other measured behavior, the
      numbers in the PR description are reproducible (script, command, or
      both)

## Screenshots / recordings

<!-- If this touches index.html or the HUD, a screenshot or short clip
     helps a lot -- delete this section if not applicable. -->
