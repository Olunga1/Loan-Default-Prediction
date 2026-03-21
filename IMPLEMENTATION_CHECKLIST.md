# Fix-First Checklist (Execution Order)

## P0: Start/Build Blockers
- [x] Add missing backend modules imported by API (`src/api/middleware/*`, `src/utils/config.py`).
- [x] Add missing preprocessing module used by training (`src/data/preprocessing.py`).
- [x] Fix API startup order and config initialization in `src/api/main.py`.
- [x] Align `/predict/batch` request contract with frontend (`{ applications: [...] }`).
- [x] Remove frontend unresolved imports by replacing page with self-contained baseline UI.
- [x] Add minimal Next.js project config files (`frontend/tsconfig.json`, `frontend/next.config.js`, `frontend/next-env.d.ts`).

## P1: CI and Compose Path Integrity
- [x] Add files referenced by docker-compose mounts (`scripts/init_db.sql`, Prometheus/Grafana placeholders).
- [x] Add missing CI references (`src/models/evaluate.py`, `tests/integration/test_api_endpoints.py`).
- [x] Update frontend test command to pass when no tests exist.

## P2: Runtime Correctness
- [x] Fix metrics label usage for cache-hit path.
- [x] Make health endpoint non-failing for no-model local baseline (`status: degraded`, HTTP 200).
- [x] Update unit tests to match current schema and endpoint behavior.

## P3: Documentation Integrity
- [x] Create docs files linked from `README_PRODUCTION.md`.
- [ ] Reconcile/merge `README.md` and `README_PRODUCTION.md` into single source of truth.
- [ ] Replace placeholder benchmark/accuracy claims with measured values from this repo.

## P4: Hardening Before Public Portfolio
- [ ] Add real auth flow (JWT issuance + verification) and disable anonymous mode in production.
- [ ] Add strict request/response logging policy with PII redaction.
- [ ] Add model artifact build step and pinned inference contract tests.
- [ ] Add load test targets and latency SLO checks in CI.
