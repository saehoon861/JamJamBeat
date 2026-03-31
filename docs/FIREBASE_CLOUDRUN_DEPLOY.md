# Firebase Hosting + Cloud Run Deployment

This guide deploys JamJamBeat with:

- Frontend: Firebase Hosting (`frontend/dist`)
- Backend: Cloud Run (`jamjam-backend`)

## 1) One-time setup

```bash
# Login
gcloud auth login
firebase login

# Set project
gcloud config set project <YOUR_PROJECT_ID>

# Enable required APIs
gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com
```

Create `frontend/.firebaserc` from `frontend/.firebaserc.example` and set your project id.

## 2) Deploy backend to Cloud Run

Run from repo root:

```bash
export PROJECT_ID="<YOUR_PROJECT_ID>"
export IMAGE="gcr.io/${PROJECT_ID}/jamjam-backend:$(date +%Y%m%d%H%M%S)"

gcloud auth configure-docker gcr.io
docker build -f Dockerfile.backend -t "$IMAGE" .
docker push "$IMAGE"

gcloud run deploy jamjam-backend \
  --image "$IMAGE" \
  --region asia-northeast3 \
  --allow-unauthenticated \
  --memory 1Gi \
  --timeout 60 \
  --set-env-vars JAMJAM_HOST=0.0.0.0,JAMJAM_MODEL_PATH=/app/checkpoints/best_model.pth
```

Notes:

- Backend listens on Cloud Run `PORT` automatically.
- Increase memory/timeout if inference gets slow or fails.

## 3) Build and deploy frontend to Firebase Hosting

Run from `frontend`:

```bash
npm ci
npm run build:production
firebase deploy --only hosting
```

`frontend/firebase.json` already rewrites these paths to Cloud Run service `jamjam-backend` in `asia-northeast3`:

- `/infer`
- `/health`

## 4) Verification checklist

```bash
# Cloud Run direct health
curl -sS "https://<CLOUD_RUN_URL>/health"

# Hosting health through rewrite
curl -sS "https://<YOUR_FIREBASE_DOMAIN>/health"
```

In browser devtools, confirm frontend requests are sent to relative path `/infer` (same origin).

## 5) Rollback

- Backend issue: redeploy prior backend revision in Cloud Run.
- Frontend issue: rollback Hosting to previous release in Firebase console.
