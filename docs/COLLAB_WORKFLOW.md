# Collaboration Workflow (Design/Effects vs Inference)

Lightweight collaboration rules for this project.

## 1) Ownership Boundaries

### Design/Effects owner
- `frontend/src/js/main.js`
- `frontend/src/js/theme.js`
- `frontend/src/styles/*`
- `frontend/public/assets/*`
- Gesture reaction mapping and UI/FX behavior

### Inference owner
- `src/training/*`
- `src/models/*`
- `src/dataset/*`
- `data/*` tooling and labeling pipeline
- `test_backend/*`
- `frontend/src/js/model_inference.js` endpoint integration semantics

### Shared boundary files (both reviewers required)
- `docs/GESTURE_CONTRACT.md`
- `frontend/src/js/gestures.js` (canonical label normalization)

## 2) PR and Review Rules

### PR types
- UI PR: visual/effect/layout only
- Inference PR: model/pipeline/server endpoint behavior only
- Contract PR: labels/schema/confidence semantics changes

### Required reviewers
- UI PR -> Design/Effects owner
- Inference PR -> Inference owner
- Contract PR -> Both owners

## 3) Handoff Checklist

Before merge, check all that apply:

- [ ] Contract changed? (labels/schema/confidence semantics)
- [ ] `docs/GESTURE_CONTRACT.md` updated if changed
- [ ] Frontend fallback behavior verified for unknown label (`None`/no-op)
- [ ] Endpoint failure behavior verified (`model_inference.js` fail-open behavior)
- [ ] Mapping impact checked in `main.js` (sound/effect/object trigger)
- [ ] Quick smoke test run in browser (`index.html`, `theme.html`)

## 4) Daily Communication Protocol (5-10 min)

Use this exact async template:

```text
Yesterday:
Today:
Contract risk: none | label change | confidence semantics | schema change
Blocked:
```

Keep discussion focused on:
1. Label set changes
2. Confidence meaning/range changes
3. Integration breakages between inference and UI

## 5) Change Management Rules

### Label changes
- Prefer add + alias over rename/remove.
- If rename is unavoidable, keep old alias temporarily and announce deprecation window.

### Confidence semantics changes
- Treat as contract change.
- Include before/after example in PR description.

### Schema changes
- Document request/response field changes in `GESTURE_CONTRACT.md`.
- Frontend owner validates graceful handling before merge.

## 6) Acceptance Criteria Examples

### Design/Effects done criteria
- Gesture events trigger intended object/effect/sound mapping.
- Unknown label does not crash UI.
- Visual feedback remains responsive under active inference traffic.

### Inference done criteria
- Endpoint returns valid schema fields (`label`, `confidence`).
- Confidence values are stable and bounded [0, 1].
- Repeated request failures degrade gracefully (no frontend lockup).
