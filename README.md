# JamJamBeat

JamJamBeat is a hand-gesture music interaction project with:

- `frontend/`: Vite web app
- `test_backend/`: lightweight inference server (`/infer`, `/health`)

This README documents the deployment baseline that was missing in earlier checks.

## Prerequisites

- Docker 24+
- Docker Compose v2

## Quick Deploy (WSL)

From project root:

```bash
docker compose up -d --build
```

Services:

- Frontend: `http://localhost:3003`
- Backend health: `http://localhost:8008/health`

The frontend proxies `/infer` and `/health` to the backend through Nginx.

## Environment Variables

Backend container supports:

- `JAMJAM_HOST` (default: `0.0.0.0`)
- `JAMJAM_PORT` (default: `8008`)
- `JAMJAM_MODEL_PATH` (default: `/app/checkpoints/best_model.pth`)

Frontend build arg:

- `VITE_MODEL_ENDPOINT` (default: `/infer`)

## Useful Commands

```bash
# status
docker compose ps

# logs
docker compose logs -f backend
docker compose logs -f frontend

# stop
docker compose down
```

## Notes

- Existing local development flow (`npm run dev`, local Python execution) remains unchanged.
- `Makefile` still contains legacy absolute training paths; deployment runtime does not rely on them.
