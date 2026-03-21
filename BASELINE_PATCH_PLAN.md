# Minimal Working Baseline Patch Plan

## Goal
Get to a coherent baseline where:
1. FastAPI app starts and serves required endpoints.
2. Frontend compiles and can call backend endpoints.
3. CI does not fail due to missing files/paths.

## Scope Applied
- Backend:
  - Replaced brittle API wiring with runnable baseline in `src/api/main.py`.
  - Added middleware modules and configuration settings module.
  - Standardized request/response schemas in `src/api/schemas.py`.
- ML pipeline:
  - Added `src/data/preprocessing.py` and simplified `src/models/train.py`.
  - Added `src/models/evaluate.py` used by CI.
  - Simplified `scripts/optimize_model.py` to produce actionable report output.
- Frontend:
  - Replaced unresolved component imports with a self-contained form page.
  - Added Next.js config/typescript bootstrap files.
- Infra/ops:
  - Added `scripts/init_db.sql`, Prometheus and Grafana placeholders, and baseline Kubernetes manifest.
  - Added docs files referenced by production README.
  - Added integration test placeholder.

## Remaining Work to Declare "Production-Ready"
- Replace temporary auth bypass with enforced JWT + RBAC.
- Add actual frontend component system (form validation, charts, design system).
- Add real model registry flow and reproducible artifact packaging.
- Add E2E tests and load testing.
- Add deployment manifests beyond baseline single deployment.

## Verification Commands
```bash
python -m compileall src
python src/models/train.py
python src/models/evaluate.py --model-path models/production/model.pkl
uvicorn src.api.main:app --reload
```

## Risk Notes
- Current auth middleware allows anonymous requests for local baseline; unsafe for production.
- Current docs include placeholders; use measured metrics before publishing.
- CI still depends on external secrets/services for full train/deploy jobs.
