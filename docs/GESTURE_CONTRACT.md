# Gesture Contract (Frontend <-> Inference)

This document is the single boundary contract between:
- Design/Effects owner (frontend visuals, interactions)
- Inference owner (gesture prediction pipeline)

If labels, confidence semantics, or payload shape change, update this file in the same PR.

## 1) Scope

- Inference owner controls how predictions are generated.
- Frontend owner controls how predictions are mapped to effects.
- Integration boundary is `frontend/src/js/model_inference.js` -> `frontend/src/js/gestures.js`.

## 2) Runtime Data Flow

1. `model_inference.js` sends landmarks payload to inference endpoint.
2. Endpoint returns prediction (`label`, `confidence`, optional `probs`, `model_version`).
3. `gestures.js` normalizes label IDs and applies stabilization rules.
4. `main.js` consumes resolved gesture label and triggers visual/audio effects.

## 3) Inference Request Schema

Current request body from `frontend/src/js/model_inference.js`:

```json
{
  "version": 1,
  "ts_ms": 123456789,
  "landmarks": [
    { "x": 0.12, "y": 0.34, "z": -0.02 }
  ]
}
```

Rules:
- `landmarks` contains 21 points.
- `x`, `y`, `z` must be finite numbers.
- `ts_ms` is frame timestamp in milliseconds.

## 4) Inference Response Schema

Current accepted response shape:

```json
{
  "label": "Fist",
  "confidence": 0.91,
  "probs": [0.01, 0.91, 0.03, 0.02, 0.01, 0.01, 0.01],
  "model_version": "v1"
}
```

Rules:
- `label`: string (required)
- `confidence`: number in [0, 1] (required)
- `probs`: array<number> (optional)
- `model_version`: string (optional)

## 5) Canonical Gesture Labels

Canonical labels used by frontend:
- `None`
- `Fist`
- `OpenPalm`
- `V`
- `Pinky`
- `Animal`
- `KHeart`

Current alias normalization in `gestures.js`:
- `class1`, `isfist` -> `Fist`
- `class2`, `paper`, `open_palm` -> `OpenPalm`
- `isv`, `class3` -> `V`
- `class4`, `ispinky` -> `Pinky`
- `class5`, `isanimal` -> `Animal`
- `class6`, `is_k_heart`, `k-heart` -> `KHeart`

## 6) Confidence and Stabilization Semantics

Current frontend semantics (`gestures.js`):
- Enter threshold: `CONFIDENCE_ENTER = 0.72`
- Stable activation: `STABLE_FRAMES = 3`
- Clear condition: `CLEAR_FRAMES = 2`

If inference owner changes meaning/range of `confidence`, this is a contract change.

## 7) Frontend Gesture-to-Effect Mapping (Current)

Current mapping in `main.js`:
- `Fist` -> `drum`
- `OpenPalm` -> `xylophone`
- `V` -> `tambourine`
- `Pinky` -> `owl`
- `Animal` -> `owl`
- `KHeart` -> `fern`

This mapping is UI policy (design/effects domain), not inference domain.

## 8) Versioning Policy

- Minor change: add optional field, add alias.
- Breaking change: rename/remove label, change required field, change confidence meaning.

When breaking change occurs:
1. Update this file.
2. Add migration note in PR description.
3. Keep compatibility alias for at least one sprint when possible.
