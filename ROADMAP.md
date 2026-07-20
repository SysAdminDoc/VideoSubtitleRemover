# Video Subtitle Remover Pro -- Roadmap

Remaining-work backlog only: every item below is NOT yet implemented.
Completed items are deleted from this file; history lives in CHANGELOG.md and git.

## Research-Driven Additions (2026-07-18)

### P3 — Under Consideration

- [ ] P3 — Community translation enablement guide
  Why: The gettext i18n infrastructure is complete (POT template, pseudo-locale, check/update tooling, coverage reporting) but no community translations exist. A contributor guide would enable crowdsourced translations.
  Evidence: Only `locale/qps-Ploc/` exists. The tooling (`scripts/i18n_catalogs.py`) is ready.
  Touches: `CONTRIBUTING.md` or `docs/translating.md` (new), `README.md` (translation section)
  Acceptance: Guide explains: how to create a new locale, run the check/update workflow, test coverage, and submit a contribution. At least one community translation started.
  Complexity: S
