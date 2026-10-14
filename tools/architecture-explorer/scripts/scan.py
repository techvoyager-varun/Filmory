"""Read-only source inventory. No application imports, .env reads, or model loads."""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import os
import re
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
OUT = HERE / 'src' / 'architecture.json'
SKIP = {'node_modules', '.venv', 'venv', '__pycache__', '.git', '.output', '.nitro',
        '.pytest_cache', '.test-tmp', 'dist', '.wrangler'}
SOURCE_SUFFIXES = {'.py', '.ts', '.tsx', '.js', '.mjs', '.css', '.html', '.yml', '.yaml'}

def read(path):
    return (ROOT / path).read_text(encoding='utf-8-sig')

def evidence(path, needle):
    lines = read(path).splitlines()
    matches = [i for i, line in enumerate(lines, 1) if needle in line]
    if not matches:
        raise ValueError(f'Missing evidence: {path}: {needle}')
    line = matches[0]
    return {'path': path, 'line': line, 'anchor': needle, 'excerpt': lines[line - 1].strip()}

def inventory():
    paths = []
    for folder in ['frontend', 'backend', '.github']:
        for directory, children, names in os.walk(ROOT / folder):
            children[:] = [name for name in children if name not in SKIP]
            for name in names:
                p = Path(directory) / name
                if p.suffix in SOURCE_SUFFIXES: paths.append(p)
    for p in [ROOT / 'frontend/package.json', ROOT / 'backend/requirements.txt',
              ROOT / 'backend/ml/active_bundle.json', ROOT / 'backend/ml/serving_policy.json']:
        paths.append(p)
    result, routes, endpoints, tables = [], [], [], []
    for p in sorted(set(paths)):
        text = p.read_text(encoding='utf-8-sig')
        path = p.relative_to(ROOT).as_posix()
        item = {'path': path, 'lines': len(text.splitlines()), 'sha256': hashlib.sha256(text.encode('utf-8')).hexdigest(),
                'imports': [], 'symbols': []}
        if p.suffix == '.py':
            tree = ast.parse(text)
            prefix = ''
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    item['imports'].append({'module': node.module or '' if isinstance(node, ast.ImportFrom) else ', '.join(a.name for a in node.names), 'line': node.lineno})
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    item['symbols'].append({'name': node.name, 'line': node.lineno})
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'APIRouter':
                    for kw in node.keywords:
                        if kw.arg == 'prefix' and isinstance(kw.value, ast.Constant): prefix = kw.value.value
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    for dec in node.decorator_list:
                        if (isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute)
                            and isinstance(dec.func.value, ast.Name) and dec.func.value.id in ('router', 'app')
                            and dec.func.attr in ('get', 'post', 'put', 'patch', 'delete') and dec.args
                            and isinstance(dec.args[0], ast.Constant)):
                            signature = ast.get_source_segment(text, node).split('):', 1)[0]
                            auth = 'required' if 'Depends(get_current_user)' in signature else 'optional' if 'Depends(get_optional_current_user)' in signature else 'public'
                            endpoints.append({'method': dec.func.attr.upper(), 'route': prefix + dec.args[0].value,
                                              'file': path, 'line': dec.lineno, 'handler': node.name, 'auth': auth})
                if isinstance(node, ast.ClassDef):
                    table, columns, fks = None, [], []
                    for field in node.body:
                        if isinstance(field, ast.Assign) and isinstance(field.targets[0], ast.Name):
                            name = field.targets[0].id
                            if name == '__tablename__' and isinstance(field.value, ast.Constant): table = field.value.value
                            if isinstance(field.value, ast.Call) and isinstance(field.value.func, ast.Name) and field.value.func.id == 'Column':
                                columns.append({'name': name, 'line': field.lineno})
                                for call in ast.walk(field.value):
                                    if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == 'ForeignKey':
                                        fks.append({'column': name, 'target': ast.literal_eval(call.args[0]), 'line': call.lineno})
                    if table: tables.append({'name': table, 'className': node.name, 'file': path, 'line': node.lineno, 'columns': columns, 'foreignKeys': fks})
        elif p.suffix in {'.ts', '.tsx', '.js', '.mjs'}:
            for match in re.finditer(r'(?:from\s+|import\s*)([\'\"])([^\'\"]+)\1', text):
                item['imports'].append({'module': match.group(2), 'line': text.count('\n', 0, match.start()) + 1})
            for match in re.finditer(r'createFileRoute\([\'\"]([^\'\"]+)[\'\"]\)', text):
                routes.append({'route': match.group(1), 'file': path, 'line': text.count('\n', 0, match.start()) + 1})
        result.append(item)
    return result, routes, endpoints, tables

def build():
    files, routes, endpoints, tables = inventory()
    nodes, edges = [], []
    def node(id, title, layer, description, tech, paths, anchor, note='', status='source'):
        nodes.append({'id': id, 'title': title, 'layer': layer, 'description': description,
                      'technologies': tech.split('|'), 'files': paths, 'evidence': [evidence(paths[0], anchor)],
                      'note': note, 'status': status})
    def edge(source, target, label, path, anchor, kind='call', conditional=False, note=''):
        edges.append({'id': f'e{len(edges)+1}', 'source': source, 'target': target, 'label': label,
                      'kind': kind, 'conditional': conditional, 'note': note, 'evidence': [evidence(path, anchor)]})
    F = 'frontend/src/'
    B = 'backend/app/'
    node('shell', 'App shell & routing', 'frontend', 'TanStack file routes render through Outlet inside QueryClientProvider, AuthProvider and UserDataProvider. The router supplies a request-scoped query cache.', 'React 19|TanStack Router / Start|TanStack Query|TypeScript', [F+'routes/__root.tsx', F+'router.tsx', F+'routeTree.gen.ts', F+'components/Navbar.tsx', F+'components/Footer.tsx'], '            <Outlet />')
    node('discovery', 'Discovery & catalog screens', 'frontend', 'Home, catalog, search, movie details, profile and model transparency screens query typed API clients and render movie rails, grids and taste state.', 'React|TanStack Query', [F+'routes/'+x for x in ['index.tsx', 'movies.index.tsx', 'movies.$movieId.tsx', 'search.tsx', 'profile.tsx', 'model.tsx']], 'getRecommendations')
    node('auth-ui', 'Session & onboarding', 'frontend', 'Login/register/demo handlers save tokens; AuthProvider restores /me, handles expiry and cross-tab changes. RequireAuth redirects unauthenticated visitors. Onboarding saves genre and movie preferences.', 'React context|localStorage|TanStack Router', [F+'context/AuthContext.tsx', F+'components/RequireAuth.tsx', F+'components/onboarding/OnboardingWizard.tsx', F+'routes/login.tsx', F+'routes/register.tsx', F+'routes/onboarding.tsx', F+'api/auth.ts'], 'restoreSession')
    node('activity-ui', 'Activity & query refresh', 'frontend', 'UserDataProvider loads history, likes and watchlist, serializes duplicate actions, and invalidates recommendation/taste queries after supported mutations.', 'React context|TanStack Query|Sonner', [F+'context/UserDataContext.tsx', F+'components/MovieActions.tsx', F+'routes/history.tsx', F+'routes/my-list.tsx', F+'api/interactions.ts'], 'recordInteraction', 'Play records a simulated viewing event; no streaming/video delivery service is present.')
    node('chat-ui', 'Ask Filmory workspace', 'frontend', 'The chat route manages sessions and message requests. ChatPanel displays responses, active filters, recommended cards and grounded evidence; feedback and filter removal use the assistant API.', 'React|TypeScript', [F+'routes/ask-filmory.tsx'] + [F+'components/assistant/'+x for x in ['ChatPanel.tsx','ChatInput.tsx','ActiveFilters.tsx','AssistantMovieCards.tsx','EvidencePanel.tsx']] + [F+'api/assistant.ts'], 'sendMessage')
    node('http', 'Typed HTTP client', 'frontend', 'request() sends JSON fetches to VITE_API_BASE_URL (localhost:8000 default), attaches a localStorage Bearer token and reports HTTP errors. Six domain modules define the endpoint calls.', 'Fetch API|TypeScript|JWT bearer', [F+'api/'+x for x in ['client.ts','auth.ts','movies.ts','recommendations.ts','interactions.ts','assistant.ts','recommendationStudy.ts']], 'await fetch(')
    node('components', 'Movie presentation & UI', 'frontend', 'MovieCard/Row/Grid, HeroBanner, Poster, loading/error/empty states and Radix wrappers compose the screens. Poster uses catalog image URLs with generated artwork on failure.', 'React|Radix UI|Tailwind CSS 4|Lucide', [F+'components/'+x for x in ['MovieCard.tsx','MovieRow.tsx','MovieGrid.tsx','HeroBanner.tsx','Poster.tsx','LoadingSkeleton.tsx','ErrorState.tsx','EmptyState.tsx']] + [x['path'] for x in files if '/components/ui/' in x['path']] + [F+'styles.css',F+'types/movie.ts',F+'types/assistant.ts'], 'MovieActions')
    node('ssr', 'SSR & error handling', 'frontend', 'Vite uses the Lovable TanStack config. Start installs CSRF protection for server functions; the custom server entry catches catastrophic SSR errors and renders an error page.', 'Vite|TanStack Start|Nitro|CSRF middleware', [F+'start.ts',F+'server.ts',F+'lib/error-capture.ts',F+'lib/error-page.ts',F+'lib/lovable-error-reporting.ts','frontend/vite.config.ts'], 'createCsrfMiddleware', 'Allowed preview hosts and optional browser reporting hooks are configuration, not proof of a deployed integration.')
    node('legacy-data', 'Local catalog helper (unused)', 'frontend', 'data/movies.ts contains curated movie extras and an exported image-proxy URL helper. No current route/component import of this module was found; API-backed routes are the active path.', 'TypeScript|Curated metadata', [F+'data/movies.ts'], 'const CURATED', 'Wikimedia and images.weserv.nl URL declarations here do not prove a current browser integration. No active edge is drawn.', 'inactive')
    node('app', 'FastAPI entry & lifespan', 'backend', 'Registers six routers, configures CORS and exposes /health. Lifespan attempts schema creation and model loading; health checks SQL connectivity and model readiness.', 'FastAPI|Uvicorn|Pydantic Settings|CORS', [B+'main.py',B+'config.py','backend/requirements.txt'], 'app.include_router')
    node('auth-api', 'Auth & preferences API', 'backend', 'Registers users with bcrypt hashes, issues JWTs for login/demo, restores authenticated profiles and validates onboarding preferences stored in SQLAlchemy.', 'FastAPI|Pydantic|SQLAlchemy', [B+'routers/auth.py',B+'schemas/schemas.py'], 'create_access_token')
    node('security', 'JWT & identity dependencies', 'backend', 'HS256 token creation/verification with expiration. HTTPBearer resolves the sub claim to a database User; required and optional dependencies are separate.', 'python-jose|Passlib / bcrypt|HTTPBearer', [B+'core/security.py',B+'core/deps.py',B+'core/time.py'], 'jwt.encode')
    node('catalog-api', 'Catalog API', 'backend', 'Lists genres, paginates and filters movies, performs title/genre search, and resolves movie details. Missing artwork may trigger TMDB enrichment.', 'FastAPI|SQLAlchemy', [B+'routers/movies.py'], 'ensure_movie_posters')
    node('rec-api', 'Recommendation & metrics API', 'backend', 'Serves personalized, cold-start, similar, most-rated, popular and genre lists plus taste-state and offline metric reports. Numeric user targets resolve only to the requester.', 'FastAPI|Pydantic|SQLAlchemy', [B+'routers/recommendations.py'], 'get_personalized_recommendations(', 'The metrics endpoint reads an offline artifact; it does not measure live accuracy.')
    node('activity-api', 'Activity API', 'backend', 'Records play/like/list events, maintains watch history, current likes/watchlist and refreshes genre signals. User identity comes from the bearer dependency.', 'FastAPI|SQLAlchemy', [B+'routers/interactions.py'], 'refresh_genre_signals')
    node('signals', 'Live preference signals', 'backend', 'Rebuilds genre weights from current onboarding genres, plays and current likes, replacing UserGenrePreference rows. This is request-time profile updating, not model training.', 'SQLAlchemy|Python Counter', [B+'services/preference_signals.py'], 'db.query(UserGenrePreference)')
    node('assistant-api', 'Assistant API', 'backend', 'Authenticated chat, session, filter and feedback endpoints enforce ownership. Chat requests pass per-user and per-IP in-memory rate limits before orchestration.', 'FastAPI|Pydantic|deque rate limiter', [B+'routers/assistant.py',B+'schemas/assistant.py'], '_enforce_chat_rate_limit')
    node('orchestrator', 'Conversation orchestration', 'backend', 'Persists messages and intent deltas, excludes seen/watched titles and enforces hard constraints. Catalog lookup returns title matches; discovery branches rank candidates and produce evidence-backed explanations.', 'SQLAlchemy|Pydantic intent schemas|Python', [B+'services/assistant_orchestrator.py'], 'def handle_message(')
    node('search', 'Metadata / keyword retrieval', 'backend', 'Searches SQL movie titles/descriptions and structured genre, year, runtime and rating filters. Reference titles can seed item similarity. No vector retrieval call exists in this service.', 'SQLAlchemy|Regex|SQL ILIKE', [B+'services/semantic_search.py'], 'def search_movies_by_intent(', 'Despite the filename semantic_search.py, this implementation does not perform embedding similarity.')
    node('llm', 'Intent & explanation adapter', 'backend', 'Local-only mode defaults to deterministic intent parsing and catalogue-evidence explanations. Explicitly disabling local mode permits google-genai calls; missing keys, exceptions or unusable responses fall back to heuristics/templates.', 'Local parsing|google-genai (optional)|Pydantic|ContextVar', [B+'services/llm_client.py',B+'config.py'], 'if settings.ASSISTANT_LOCAL_ONLY:', 'Cloud calls are disabled by default. Provider credentials, configured model availability, quota and live responses were not checked.', 'conditional')
    node('study-ui', 'Optional recommendation review', 'frontend', 'The signed-in homepage offers an opt-in review of a frozen recommendation slate. Interesting/not-for-me/unsure answers are saved separately from plays and likes.', 'React|TypeScript|Fetch API', [F+'components/RecommendationReview.tsx',F+'api/recommendationStudy.ts'], 'startRecommendationStudy', 'UI verification is not human participation or evidence of satisfaction.')
    node('study-api', 'Owned recommendation study API', 'backend', 'Requires consent and bearer identity to create a frozen slate and policy snapshot. Owned read/update endpoints save per-movie answers without changing recommendation history.', 'FastAPI|Pydantic|SQLAlchemy', [B+'routers/recommendation_study.py'], 'def start_review(')
    node('feedback', 'Reversible exclusions & positive history', 'backend', 'Last not-interested/show-again decision controls exclusions. Positive history deduplicates movies, excludes hidden titles and ignores removed likes. A play is activity rather than proof of enjoyment.', 'SQLAlchemy|Interaction log', [B+'services/recommendation_feedback.py'], 'def positive_history(')
    node('recommender', 'Recommendation engine', 'backend', 'Combines live SQL profiles/history with the loaded model catalog. Mapped users use the active serving variant; unmapped accounts use a selected sequence/neighbor policy or cold-start fallback. Discovery may use DAMR.', 'PyTorch|NumPy|SQLAlchemy', [B+'ml/recommender.py'], 'def get_personalized_recommendations(')
    node('ranking', 'Frozen ranking & DAMR', 'backend', 'ranking_policy selects history bins, candidate positions and normalized fusion. DAMR separately computes drift/focus/maturity/freshness, adaptive expert weights, momentum, quality and diversity.', 'PyTorch|NumPy|Training-free ranking', [B+'ml/ranking_policy.py',B+'ml/damr.py'], 'def choose_history_policy(', 'These are alternate/conditional ranking branches, not one compulsory serial pipeline.')
    node('loader', 'Model loader & inference', 'backend', 'Singleton ModelService resolves active_bundle.json, loads maps/matrices and three checkpoint architectures, switches to eval mode and validates serving-policy checkpoint hashes.', 'PyTorch|JSON|Pickle maps|SHA-256', [B+'ml/model_service.py',B+'ml/architectures.py'], 'def load_all(')
    node('tmdb-adapter', 'Artwork enrichment adapter', 'backend', 'Uses MovieLens/TMDB ID links or title/year searches to fetch metadata with urllib, stores missing artwork/synopsis/runtime in movies and returns CDN URLs.', 'urllib|ThreadPoolExecutor|SQLAlchemy', [B+'ml/tmdb.py'], 'def ensure_movie_posters(', 'Enabled only with a TMDB key and missing poster data; provider calls were not checked.', 'conditional')
    node('db', 'SQLAlchemy session & schema', 'database', 'Configured PostgreSQL engine with pool health checks. get_db opens/closes request sessions; declarative ORM maps 13 tables. Tests may use SQLite independently.', 'PostgreSQL (configured)|SQLAlchemy 2', [B+'database.py',B+'models/db_models.py'], 'SessionLocal =')
    node('study-db', 'Recommendation study records', 'database', 'recommendation_studies stores an owned frozen movie slate, serving policy snapshot and replaceable per-movie judgments. No participant results are invented.', 'SQLAlchemy|JSON snapshots|User foreign key', [B+'models/db_models.py'], 'class RecommendationStudy(Base):')
    node('identity-db', 'Users & preference tables', 'database', 'users, user_preferences and user_genre_preferences store account identities, optional MovieLens identity mapping, onboarding selections and live genre weights.', 'SQLAlchemy|JSON columns|Foreign keys', [B+'models/db_models.py'], 'class User(Base):')
    node('catalog-db', 'Movie catalog table', 'database', 'movies stores canonical MovieLens movie_id, title, genres, rating/count, runtime, description and poster/backdrop URLs.', 'SQLAlchemy|JSON genres', [B+'models/db_models.py'], 'class Movie(Base):')
    node('activity-db', 'Activity tables', 'database', 'interactions, watch_history, my_list and likes relate users to movies. Current likes/watchlist uniqueness and interaction history are distinct state.', 'SQLAlchemy|Foreign keys|Unique constraints', [B+'models/db_models.py'], 'class Interaction(Base):')
    node('assistant-db', 'Assistant persistence', 'database', 'assistant_sessions stores user-owned accumulated intent; assistant_messages stores transcript and returned IDs; assistant_feedback records feedback. Message/movie feedback IDs are not declared foreign keys.', 'SQLAlchemy|JSON snapshots|UUID sessions', [B+'models/db_models.py'], 'class AssistantSession(Base):')
    node('embedding-db', 'Embedding metadata (unused)', 'database', 'movie_text_embeddings declares model/dimension/hash metadata and a movie foreign key. No embedding vector column or active reader/writer is found in app services.', 'SQLAlchemy|Schema only', [B+'models/db_models.py',B+'config.py'], 'class MovieTextEmbedding(Base):', 'SEMANTIC_SEARCH_ENABLED and GEMINI_EMBEDDING_MODEL settings do not establish a vector-search integration.', 'inactive')
    node('gemini', 'Google Gemini', 'external', 'Optional remote model provider used by the intent/explanation adapter through google-genai.', 'Gemini API|google-genai', [B+'services/llm_client.py'], 'genai.Client(api_key=', 'Configuration and call sites verified; external execution is unverified.', 'conditional')
    node('tmdb', 'TMDB API & images', 'external', 'Metadata/search provider and image.tmdb.org CDN URL source. Browsers render returned artwork URLs directly; API keys stay in backend configuration.', 'TMDB REST|Image CDN', [B+'ml/tmdb.py',F+'components/Poster.tsx'], 'https://api.themoviedb.org', 'Actual stored image URL hosts and current API availability were not queried.', 'conditional')
    node('fonts', 'Google Fonts', 'external', 'The root document preconnects and loads Sora/Manrope font CSS from Google Fonts.', 'Hosted font CSS', [F+'routes/__root.tsx'], 'https://fonts.googleapis.com/css2', 'Browser reachability was not checked.', 'conditional')
    node('preview-hooks', 'Preview reporting hooks', 'external', 'The root error boundary calls optional window.__lovableEvents / __lovableReportRuntimeError functions if supplied by the host.', 'Browser window hooks', [F+'lib/lovable-error-reporting.ts'], 'window.__lovableEvents?', 'Hook implementation, destination and availability are outside this repository; no remote telemetry destination is asserted.', 'conditional')
    node('data', 'MovieLens / catalog files', 'offline', 'Local catalog/link JSON is consumed by seed/enrichment scripts. Training data preparation can read MovieLens ratings/movies CSV or stored histories and produces mappings/sequences/matrices.', 'MovieLens format|JSON / CSV|Pandas|Pickle', [B+'scripts/seed_movies.py','backend/scripts/prepare_training_data.py','backend/scripts/train_models.py'], 'movie_catalog', 'Data contents and licensing provenance were not independently audited; only readers and local file locations are mapped.')
    node('train', 'Offline training & evaluation', 'offline', 'Fresh training splits sequences into training/validation/test, writes checkpoints/config/logs and sampled/full-catalog metrics. Evaluation scripts select policies and produce research evidence.', 'PyTorch|NumPy|Pandas|Python CLI', ['backend/scripts/train_models.py','backend/scripts/prepare_training_data.py'] + [x['path'] for x in files if x['path'].startswith('backend/scripts/') and x['path'] not in ['backend/scripts/train_models.py','backend/scripts/prepare_training_data.py']], 'torch.save(model.state_dict()', 'Only source/metadata inspection was done here; training/evaluation was not rerun. This explorer does not modify weights.')
    node('bundle', 'Active artifacts & policy', 'offline', 'active_bundle.json points to the clean_extended run. Its config selects Transformer retrieval and normalized_fixed serving. serving_policy.json holds checkpoint-bound history-bin policies for new accounts.', 'PyTorch checkpoints|ID maps|Metrics JSON', ['backend/ml/active_bundle.json','backend/ml/serving_policy.json','backend/training/runs/20261001_clean_extended/model_config.json','backend/training/runs/20261001_clean_extended/metrics.json'], '20261001_clean_extended', 'Pointer/config files exist. Checkpoint bytes were not loaded or revalidated by the explorer scanner.', 'artifact')
    node('schema-tools', 'Migrations & catalog seeding', 'offline', 'Alembic imports Base.metadata for schema migrations, including assistant and recommendation study tables. Seed scripts populate catalog and demo records; a separate script enriches database posters.', 'Alembic|SQLAlchemy|Python CLI', ['backend/migrations/env.py','backend/migrations/versions/0001_initial_tables.py','backend/migrations/versions/0002_assistant_tables.py','backend/migrations/versions/0003_recommendation_study.py',B+'scripts/seed_movies.py',B+'scripts/enrich_database_posters.py',B+'scripts/tmdb_helper.py'], 'target_metadata = Base.metadata', 'Migration state of a live database was not checked.')
    node('checks', 'Tests, CI & research docs', 'offline', 'Backend pytest and frontend node tests cover behavior; GitHub CodeQL/Codacy workflows are declared. Research reports/manuscript describe evaluations, but are not executable call relationships.', 'pytest|Node test runner|CodeQL|Codacy|LaTeX', ['backend/tests/conftest.py'] + [x['path'] for x in files if ('/tests/' in x['path'] or x['path'].startswith('.github/')) and x['path'] != 'backend/tests/conftest.py'] + ['docs/research/filmory_paper.tex','docs/research/PAPER_AUDIT.md','docs/research/FULL_PROJECT_AUDIT.md'], 'import', 'Tests/CI status and manuscript claims were not re-certified in this architecture-only task.')

    # Each arrow represents the stated call/import/FK/config use, not inferred runtime success.
    edge('shell','auth-ui','provides session context',F+'routes/__root.tsx','<AuthProvider>', 'composition')
    edge('shell','activity-ui','provides user data context',F+'routes/__root.tsx','<UserDataProvider>', 'composition')
    edge('shell','discovery','renders route outlet',F+'routes/__root.tsx','            <Outlet />','composition')
    edge('discovery','http','queries catalog / recommendations',F+'routes/index.tsx','queryFn: () => getRecommendations','call')
    edge('auth-ui','http','login / restore / preferences',F+'api/auth.ts','request<AuthResponse>("/api/auth/login"','call')
    edge('activity-ui','http','activity requests',F+'context/UserDataContext.tsx','interactionsApi.recordInteraction','call')
    edge('chat-ui','http','chat / session requests',F+'api/assistant.ts','request<ChatResponse>','call')
    edge('discovery','components','renders movie rails',F+'routes/index.tsx','<MovieRow','composition')
    edge('components','activity-ui','reads action context',F+'components/MovieActions.tsx','useUserData','call')
    edge('http','app','JSON fetch + Bearer headers',F+'api/client.ts','await fetch(','http')
    for target, module in [('auth-api','auth'),('catalog-api','movies'),('rec-api','recommendations'),('activity-api','interactions'),('assistant-api','assistant'),('study-api','study')]:
        edge('app',target,'mounts router',B+'main.py',f'app.include_router({module}_router)','mount')
    edge('app','db','create tables / health check',B+'main.py','Base.metadata.create_all','call')
    edge('app','loader','loads models in lifespan',B+'main.py','model_service.load_all()','call')
    edge('auth-api','security','hashes / signs JWT',B+'routers/auth.py','token = create_access_token(','call')
    for source, module in [('auth-api','auth'),('activity-api','interactions'),('assistant-api','assistant')]:
        edge(source,'security','requires bearer identity',B+f'routers/{module}.py','Depends(get_current_user)','dependency')
    edge('rec-api','security','optional bearer identity',B+'routers/recommendations.py','Depends(get_optional_current_user)','dependency')
    edge('security','identity-db','resolves JWT subject',B+'core/deps.py','db.query(User).filter','sql')
    edge('auth-api','identity-db','reads / writes account preferences',B+'routers/auth.py','db.query(UserPreference)','sql')
    edge('catalog-api','catalog-db','queries catalog',B+'routers/movies.py','db.query(Movie)','sql')
    edge('catalog-api','tmdb-adapter','enriches missing posters',B+'routers/movies.py','ensure_movie_posters(','call',True)
    edge('rec-api','recommender','dispatches recommendation request',B+'routers/recommendations.py','rec_type, recommendations = get_personalized_recommendations','call')
    edge('rec-api','bundle','reads offline metrics via active loader',B+'routers/recommendations.py','model_service.ml_dir + "/metrics.json"','artifact')
    edge('activity-api','activity-db','writes / reads activity state',B+'routers/interactions.py','db.query(WatchHistory)','sql')
    edge('activity-api','signals','rebuilds genre signals',B+'routers/interactions.py','refresh_genre_signals(current_user.id, db)','call')
    edge('auth-api','signals','refreshes onboarding genre signals',B+'routers/auth.py','refresh_genre_signals(user.id, db)','call')
    edge('signals','identity-db','replaces genre preference rows',B+'services/preference_signals.py','db.add_all([UserGenrePreference','sql')
    edge('signals','activity-db','reads distinct plays and current likes',B+'services/preference_signals.py','played_ids = db.query(Interaction.movie_id)','sql')
    edge('signals','feedback','excludes hidden titles from genre signals',B+'services/preference_signals.py','excluded = hidden_movie_ids(user_id, db)','call')
    edge('discovery','study-ui','renders optional study review',F+'routes/index.tsx','<RecommendationReview','composition')
    edge('study-ui','http','sends consent and study answers',F+'api/recommendationStudy.ts','return request("/api/recommendation-study"','call')
    edge('study-api','security','requires bearer identity',B+'routers/recommendation_study.py','Depends(get_current_user)','dependency')
    edge('study-api','recommender','freezes ten recommendations',B+'routers/recommendation_study.py','get_personalized_recommendations(user,db,top_k=10)','call')
    edge('study-api','study-db','persists frozen slate and answers',B+'routers/recommendation_study.py','study = RecommendationStudy(','sql')
    edge('assistant-api','orchestrator','handles chat message',B+'routers/assistant.py','handle_message(','call')
    edge('orchestrator','assistant-db','persists conversation',B+'services/assistant_orchestrator.py','db.add(assistant_msg)','sql')
    edge('orchestrator','activity-db','excludes watched titles',B+'services/assistant_orchestrator.py','db.query(WatchHistory.movie_id)','sql')
    edge('orchestrator','llm','extracts intent / explains evidence',B+'services/assistant_orchestrator.py','delta = extract_intent(','call',True)
    edge('orchestrator','search','retrieves constrained candidates',B+'services/assistant_orchestrator.py','search_movies, reference_ids = search_movies_by_intent(','call')
    edge('search','catalog-db','SQL metadata / title search',B+'services/semantic_search.py','query = db.query(Movie)','sql')
    edge('orchestrator','recommender','discovery reranking (not title lookup)',B+'services/assistant_orchestrator.py','scored_movies, damr_audit = rerank_candidate_pool_with_damr(','call',True)
    edge('llm','gemini','generate_content only when local-only mode is disabled',B+'services/llm_client.py','response = client.models.generate_content(','external',True,'ASSISTANT_LOCAL_ONLY defaults to true; these external calls are optional.')
    edge('recommender','loader','uses loaded model service',B+'ml/recommender.py','model_service.load_all()','call')
    edge('recommender','ranking','selects history policy / DAMR branch',B+'ml/recommender.py','policy = choose_history_policy(','call',True)
    edge('recommender','catalog-db','hydrates scored movie IDs',B+'ml/recommender.py','db_movies = db.query(Movie)','sql')
    edge('recommender','feedback','reads unique current positive history',B+'ml/recommender.py','recent_interactions = reversed(positive_history(user.id, db))','call')
    edge('feedback','activity-db','reads plays and current likes',B+'services/recommendation_feedback.py','db.query(Interaction.movie_id, Interaction.timestamp)','sql')
    edge('orchestrator','feedback','excludes rejected movies',B+'services/assistant_orchestrator.py','watched_ids.update(hidden_movie_ids(user.id, db))','call')
    edge('recommender','identity-db','reads live genre preferences',B+'ml/recommender.py','db.query(UserGenrePreference)','sql')
    edge('loader','bundle','resolves active bundle / policy',B+'ml/model_service.py','"active_bundle.json"','artifact')
    edge('tmdb-adapter','tmdb','fetches metadata / builds image URLs',B+'ml/tmdb.py','urllib.request.urlopen(','external',True)
    edge('tmdb-adapter','catalog-db','persists artwork / metadata',B+'ml/tmdb.py','m.poster_url = meta["poster_url"]','sql',True)
    edge('components','tmdb','renders catalog artwork URL',F+'components/Poster.tsx','src={movie.posterUrl}','external',True,'Poster accepts any catalog URL; TMDB is a URL producer, not a hard-coded browser host restriction.')
    edge('shell','fonts','loads hosted font CSS',F+'routes/__root.tsx','https://fonts.googleapis.com/css2','external',True)
    edge('shell','preview-hooks','optional error reporting',F+'routes/__root.tsx','reportLovableError(error','external',True)
    edge('train','data','reads maps / sequences or CSV', 'backend/scripts/train_models.py','sequences = pickle.load(f)','artifact')
    edge('train','bundle','writes training bundles (activation separate)', 'backend/scripts/train_models.py','torch.save(model.state_dict()','artifact',True,'Training writes args.output; the active pointer chooses a bundle separately. No automatic deployment edge is implied.')
    edge('schema-tools','db','uses Base.metadata for migrations','backend/migrations/env.py','target_metadata = Base.metadata','schema')
    edge('schema-tools','data','reads movie catalog seed input',B+'scripts/seed_movies.py','movie_catalog','artifact')
    # Actual FK declarations, including the inactive metadata table.
    table_node = {'users':'identity-db','user_preferences':'identity-db','user_genre_preferences':'identity-db',
                  'movies':'catalog-db','interactions':'activity-db','watch_history':'activity-db','my_list':'activity-db',
                  'likes':'activity-db','assistant_sessions':'assistant-db','assistant_messages':'assistant-db',
                  'assistant_feedback':'assistant-db','movie_text_embeddings':'embedding-db','recommendation_studies':'study-db'}
    for table in tables:
        for fk in table['foreignKeys']:
            source, target = table_node[table['name']], table_node[fk['target'].split('.')[0]]
            if source != target:
                edge(source,target, f"FK {table['name']}.{fk['column']} → {fk['target']}",table['file'],f'ForeignKey("{fk["target"]}"','foreign-key')
    for item in nodes:
        item['routes'] = [r for r in routes if r['file'] in item['files']]
        item['endpoints'] = [e for e in endpoints if e['file'] in item['files']]
        item['tables'] = [t for t in tables if table_node.get(t['name']) == item['id']]
    walks = [
        {'id':'auth','title':'Sign in & restore','summary':'Credentials → bcrypt → signed token → bearer identity → restored session.', 'steps':[
            ['auth-ui','Enter credentials or restore a saved session. The frontend guard is only a UI guard.'],
            ['http','Auth API wrappers POST JSON; later requests attach the saved Bearer token.'],
            ['auth-api','Login reads the account and verifies its password before returning accessToken and user.'],
            ['security','Passlib checks bcrypt. python-jose signs/verifies exp and sub using backend configuration.'],
            ['identity-db','Protected dependencies resolve sub to a current users row; /me restores the profile.']]},
        {'id':'recommendations','title':'Movie recommendations','summary':'Home feed → user-aware policy → frozen model inference → hydrated movie results.', 'steps':[
            ['discovery','Home queries recommendations keyed by the current user; guests receive a fallback feed.'],
            ['http','The wrapper sends GET /api/recommendations/{target}?top_k=12&variant=auto.'],
            ['rec-api','Only the requester can resolve a numeric target. The handler dispatches to the engine.'],
            ['recommender','Mapped identities use model scores. New accounts branch to sequence/neighbor or cold-start logic.'],
            ['loader','Inference uses loaded PyTorch models and genre matrices from the selected bundle, without retraining.'],
            ['ranking','Auto uses normalized_fixed for the mapped bundle; new-user history bins choose a frozen policy. DAMR is a separate supported branch.'],
            ['catalog-db','The engine hydrates scored IDs into movie schemas for the response and UI rails.']]},
        {'id':'assistant','title':'Ask Filmory','summary':'Chat → intent delta → strict catalog filters → lookup or ranking → evidence/explanation.', 'steps':[
            ['chat-ui','Send a message with the selected session ID; the route guards stale responses.'],
            ['assistant-api','Bearer identity, ownership checks and in-memory rate limits precede the chat handler.'],
            ['orchestrator','Create/load a session, persist the user message and merge explicit intent changes.'],
            ['llm','Parse locally by default. Cloud intent/explanation calls require disabling local-only mode; external success is unverified.'],
            ['search','Apply title/keyword and metadata constraints plus watched/session exclusions in SQL.'],
            ['recommender','Title lookup bypasses reranking. Discovery may score candidates with the configured DAMR branch.'],
            ['assistant-db','Store transcript and returned IDs, and return grounded evidence plus an explanation/template.']]},
        {'id':'activity','title':'Feedback & taste updates','summary':'Like/play/save → authenticated persistence → live genre signals → query invalidation.', 'steps':[
            ['components','MovieActions delegates play, like and list actions to UserDataContext.'],
            ['activity-ui','Action requests are guarded per user/action. Play is a simulated interaction, not a video stream.'],
            ['activity-api','The authenticated user controls whose history/likes/list rows are changed.'],
            ['activity-db','Persist event history and current activity state with user/movie foreign keys.'],
            ['signals','Refresh genre weights from onboarding genres, plays and current likes. No checkpoint training occurs.'],
            ['activity-ui','Relevant query keys are invalidated; subsequent recommendation requests see the new profile.']]},
        {'id':'study','title':'Optional classmate review','summary':'Explicit consent → frozen slate → owned judgments, separate from viewing history.', 'steps':[
            ['study-ui','Opt in from Home and review a frozen movie list; participation is optional.'],
            ['http','Typed requests send consent when creating a review and movie/choice when answering.'],
            ['study-api','Bearer ownership gates every operation. A movie must belong to the frozen slate.'],
            ['recommender','Create the slate from the current policy before collecting any answers.'],
            ['study-db','Store movie/policy snapshots and replaceable judgments without changing plays or likes.']]},
        {'id':'training','title':'Offline artifacts → inference','summary':'Local histories/CSV → split & train → bundle files → explicit active pointer → loader.', 'steps':[
            ['data','Data preparation reads local MovieLens-format inputs or stored ordinal histories.'],
            ['train','Fresh training holds out validation/test targets and writes checkpoints and genuine metric logs into args.output.'],
            ['bundle','The checked-in pointer selects a specific bundle; serving-policy hashes bind overlays to checkpoint bytes. Activation is separate.'],
            ['loader','ModelService reads the pointer and config, loads matrices/maps/models and verifies policy hashes.'],
            ['rec-api','/api/metrics reads selected offline metrics; these are not live user satisfaction or overall accuracy.']]}
    ]
    return {'schemaVersion':1, 'repository':'Filmory', 'root':str(ROOT), 'scope':'Static source snapshot; no live service verification',
            'nodes':nodes, 'edges':edges, 'walkthroughs':walks, 'files':files, 'routes':routes, 'endpoints':endpoints, 'tables':tables,
            'dependencies': {'frontend':json.loads(read('frontend/package.json')), 'backend':read('backend/requirements.txt').splitlines()},
            'activeModelConfig':json.loads(read('backend/training/runs/20261001_clean_extended/model_config.json')),
            'limitations':[
                'Arrows are direct source calls, composition, router registration, artifact access or declared foreign keys. They do not certify that the live branch executed.',
                'No application startup, database query, provider request, training, migration or checkpoint deserialization was performed by the scanner.',
                'Python is parsed with AST. TypeScript routes/imports use bounded regex extraction; dynamic imports, build-time plugin internals and runtime-generated paths may be incomplete.',
                'The graph aggregates modules and tables for readability; the inventory lists all scanned application, script, migration, test, style and CI source files. Third-party internals, binaries, raw data, secrets and generated builds are excluded.',
                'Embedding metadata and settings exist, but active vector search and pgvector use are not demonstrated by this application source.',
                'Gemini/TMDB credentials, provider model availability, live database schema and optional preview hook destinations remain unverified.',
                'Research claims, model quality, source data provenance, streaming support and deployed service availability cannot be certified by an architecture diagram.'
            ]}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true', help='Fail when the saved snapshot is stale; never write it.')
    args = parser.parse_args()
    data = build()
    serialized = json.dumps(data, indent=2, ensure_ascii=False) + '\n'
    if args.check:
        saved = json.loads(OUT.read_text(encoding='utf-8')) if OUT.exists() else None
        if saved:
            # Capture location affects copied absolute paths, not source identity.
            saved['root'] = data['root']
        if saved != data:
            raise SystemExit('Architecture snapshot is stale. Run npm run scan, review changed evidence, then npm run verify.')
        print(f'Fresh snapshot: {len(data["files"])} files, {len(data["nodes"])} nodes, {len(data["edges"])} evidence-backed edges.')
    else:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(serialized, encoding='utf-8')
        print(f'Wrote {OUT}: {len(data["routes"])} frontend routes, {len(data["endpoints"])} endpoints, {len(data["tables"])} tables.')
