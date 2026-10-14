# Filmory architecture explorer

A standalone React Flow viewer of the **actual repository source**, with an n8n-inspired dark canvas, coral controls, module cards, grouped layers and a source inspector. Filmory's frontend, backend and model artifacts are unchanged by this tool.

## Start locally (PowerShell)

Use Node.js **22.17 or newer** (Node 22 LTS recommended) and npm. Python **3.10+** is needed only to refresh/verify the source snapshot; the saved viewer runs without Python, PostgreSQL, API keys or Filmory's backend.

```powershell
cd E:\Filmory\tools\architecture-explorer
npm ci
npm run dev
```

Open **http://127.0.0.1:5191/**. Dependencies, including `@xyflow/react`, are already installed in this package in the current workspace. `npm ci` reproduces the lockfile on a fresh setup.

The server binds only to loopback and uses a strict port. If 5191 is occupied, use `npm run dev -- --port 5192` and open that port. **Ctrl+C** stops the viewer and frees its port. Filmory's usual ports are not needed.

```powershell
npm run verify
npm run build
npm run preview
```

`build` typechecks and produces the viewer in its own `dist/`. `preview` serves that build on port 5191; stop the dev server first. Do not open `dist/index.html` using `file://`; serve it with preview.

## Explore

- **Complete architecture** shows frontend, backend, database and external-service groups, plus offline training/artifacts/tooling.
- Select a **focused view** for sign-in, recommendations, Ask Filmory, activity updates or training/inference.
- **Click a card** or use the visible-module list to inspect responsibilities, technologies, actual repository paths, endpoints, routes, table columns, declared foreign keys and incoming/outgoing connections. File/evidence buttons copy an absolute path, with a line number for evidence.
- **Click an arrow** to inspect its precise meaning and source anchor. Solid arrows represent source relationships; dashed arrows indicate a conditional branch. The FK filter exposes real schema foreign keys separately from calls.
- **Search** matches modules, source paths, technologies, endpoints and table names. Typing a graph search returns to the complete map and retains the selected layer/status/relationship filters. In the source inventory it matches file paths, declared imports and Python symbols. Search narrows the visible nodes; only arrows with both endpoints visible remain.
- **Layer, module-status and relationship filters** narrow the graph. Reset map clears them. Unused modules are marked explicitly.
- **Zoom, pan, drag cards and use the minimap**. Cards stay inside their group. Changing a view/filter recalculates layout; refresh/reset discards temporary positions.
- **Start walkthrough** follows one of six explained flows, including the optional classmate review. Next/Previous highlight and focus each step; Exit stops guidance. These steps describe branches as well as calls, so an ordered explanation is not an assertion that every step always executes.
- **Source inventory** lists all scanned application, migration, script, test, style and CI source files. Select a file for imports, Python class/function symbols and its snapshot hash.
- **Evidence & scope** exposes verification limits, actual declared dependencies and the selected model configuration.

The diagram is intentionally read-only with respect to connections: viewers cannot invent an arrow or alter Filmory configuration. Card dragging changes only the display layout. Copy and downloads are the only persistent user actions.

## Exports

**Export diagram** downloads:

| Format | Content |
|---|---|
| SVG | Standalone vector drawing of visible groups, cards and directed arrows; edge titles retain relationship labels. No external fonts/images are needed. |
| PNG | Raster of the same drawing, up to 4096 pixels on its longest side. |
| Mermaid | Editable text for the visible nodes and labeled conditional/solid connections. |
| JSON | Complete source snapshot plus current filter settings and React Flow layout/viewport. |

SVG/PNG exports use source data and current card positions, not a screenshot of UI chrome. Search/filter first if you want a focused diagram. JSON contains source paths and architecture metadata, so review it before sharing publicly. It does not include `.env` values, model bytes, database rows or raw user datasets. Importing saved layouts is not implemented.

## Refresh after repository changes

```powershell
npm run scan
npm run verify
```

`scripts/scan.py` finds the repository relative to itself and **reads without importing application code**. It uses Python AST for FastAPI decorators/router prefixes, ORM tables/foreign keys, Python symbols and imports; bounded regexes extract TypeScript imports and `createFileRoute` declarations. It scans owned source in `frontend`, `backend` and `.github`, prunes dependency/cache/build folders and reads selected dependency/config metadata. Raw catalog JSON, tensors/checkpoints, `.env`, third-party source and generated builds are excluded. Training source snapshots are inventoried but do not create runtime relationships.

`src/architecture.json` is the checked-in snapshot. Module descriptions and arrows are curated in `scan.py`; **every node/arrow has a path, line, source anchor and excerpt**. Missing anchors fail scanning. `verify` also detects stale snapshots and validates evidence, hashes, graph integrity, endpoint/route coverage, FK targets, filters, guided steps, group bounds and export escaping. Source hashes normalize line endings to LF and remove a UTF-8 BOM, so Windows and Linux checkouts agree. This is separate from the raw-byte hashes in the frozen training/evaluation manifests. A matching anchor can survive a semantic change: review descriptions and edges after changing behavior, rather than treating text matching as a proof of unchanged execution.

The snapshot records its capture root for absolute-path copying. On a different checkout, run `npm run scan` once to update that root. Verification compares source identity independently of the capture location.

## Findings and limits

The current map aggregates **43 modules**, **76 relationships**, **12 frontend routes**, **37 declared API endpoints** (including root and health), and **13 ORM tables** across **203 scanned source/config files**. The default call view hides the schema FK arrows.

- Actual frontend requests use the typed `api/client.ts` seam, JSON fetch and localStorage bearer tokens; this is not a cookie-only auth architecture.
- `variant=auto` is resolved from the active configuration. The selected mapped-user bundle uses Transformer retrieval and `normalized_fixed`; new accounts can use checkpoint-bound history-bin sequence/neighbor policies. DAMR remains a separate supported/discovery branch. The repository's older overview text does not describe every current branch precisely.
- `semantic_search.py` currently executes SQL metadata/title/keyword filtering. Embedding settings and the metadata table alone do **not** demonstrate active vector search or pgvector integration.
- Local-only intent parsing and catalogue-evidence explanations are the default. Gemini calls require explicitly disabling `ASSISTANT_LOCAL_ONLY` and have heuristic/template fallbacks. TMDB enrichment requires a key and missing artwork. Keys, provider availability, quota, remote model identifiers and responses were not checked.
- Reversible not-interested/show-again feedback and unique current positive history feed recommendation exclusions and genre signals. The optional owned recommendation study freezes a slate and saves judgments separately from plays/likes; its existence is not evidence of human participation.
- The Poster component renders any catalog image URL. TMDB produces image URLs but is not a browser host restriction. Static curated Wikimedia URLs in the unused local catalog helper do not establish an active integration.
- Browser preview error hooks are optional. Their implementation/destination is outside the repository and remains unverified. SSR uses TanStack Start's server entry; framework internals are not inspected as repository source.
- SQLAlchemy defaults to PostgreSQL, while tests can use SQLite. Live database availability, migration state and actual stored rows were not queried.
- Model pointer/config and artifact readers are verified; scanner does not deserialize checkpoints or rerun training/evaluation. `/api/metrics` reads offline metrics, not live accuracy.
- Play is a simulated activity event. No video streaming service is evidenced.
- CI workflows/tests and research documents are present; this diagram does not certify their current run status, paper claims, recommendation quality, data licensing or plagiarism scores.

Implementation follows the official [React Flow node/subflow API](https://reactflow.dev/api-reference/types/node) and [React Flow instance API](https://reactflow.dev/api-reference/types/react-flow-instance). This is an architecture viewer inspired by n8n's visual style, not an n8n workflow executor.
