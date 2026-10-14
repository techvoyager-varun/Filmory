# Supplemental file review — 14 October 2026

Scope: research draft/audits, original and corrected notebooks, the separate
architecture explorer, dated review records, and previously uncommitted
training/evaluation metadata and validation rows. This pass did not start model
training, change serving weights, collect human feedback, or submit a manuscript
to a journal/plagiarism service.

## Checks

- Existing `docs/research/filmory_paper.tex`: built-in desktop compiler returned
  success. Verified metric tables are sourced from retained evidence; the draft
  remains provisional, with publication and Turnitin limitations documented.
- Corrected notebook regenerated with `backend/scripts/build_training_notebook.py`.
  Its six code cells parse as Python, have no saved outputs, and embedded training
  and architecture sources match the reviewed current files. No cells were run.
- Original notebook: 130 historical code cells; shell install/download and ZIP
  commands were inspected without execution. The multiline IPython ZIP cell is
  intentionally not plain Python. Original outputs are archival evidence only.
- Separate explorer: snapshot refreshed to **43 modules, 76 evidence-backed
  relationships, 12 routes, 37 API endpoints, 13 tables and 203 source/config
  files**. All **7 explorer tests passed**, including evidence anchors, source
  identity, endpoint/FK coverage, filtering, walkthrough bounds and SVG/Mermaid
  export escaping. Type checking and production build passed; the build emitted
  a nonfatal chunk-size warning and Node emitted its experimental type-stripping
  notice.
- Corrected stale explorer claims: local-only assistant behaviour is the default;
  explicit opt-in is required for cloud AI. Reversible exclusions, unique positive
  history and the owned optional study are now mapped. Six walkthroughs include
  the study, and the third migration is represented.
- Source hashes now normalize line endings/BOM for Windows/Linux consistency.
  Frozen experimental byte hashes and historical data are unchanged.
- Review of the 83 previously untracked nonignored files: **132,282,816 bytes**;
  16 JSON files, 20 JSONL files, 7 Python files and 2 notebooks passed applicable
  structural checks. No Google/GitHub key, private-key block, JWT or nonplaceholder
  PostgreSQL credential pattern was found. This targeted scan is not a security
  certification. New documentation records what is archival, incomplete or
  provisional rather than relabeling it as a successful result.

The 100 backend and 9 main-frontend tests, TypeScript/lint/build and 27-entry active
bundle integrity check passed in the preceding application commit. They were not
repeated here because application code and model checkpoints were unchanged.
Source and data are committed without dependencies, compiled output, local
databases, credentials or large training binaries. Those excluded files remain
local. The explorer server remains stopped after this source/build verification.
