# Test Backend (Temporary)

This folder is an isolated inference backend for local testing only.

## Run

From project root:

```bash
.venv/bin/python test_backend/infer_server.py
```

## Frontend hook

Use one of these:

- `?inferEndpoint=http://127.0.0.1:8008/infer`
- `window.__JAMJAM_MODEL_ENDPOINT = "http://127.0.0.1:8008/infer"`
