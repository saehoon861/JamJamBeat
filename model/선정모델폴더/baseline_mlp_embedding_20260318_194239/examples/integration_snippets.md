# integration_snippets - 메인스트림 연결용 참고 코드

## AI가 먼저 읽으면 좋은 파일
1. `MANIFEST.json`
2. `runtime/input_spec.json`
3. `runtime/inference.py`
4. `examples/sample_input.json`
5. `examples/expected_output.json`

## 순수 Python 서비스에 붙이는 최소 예시

```python
from pathlib import Path
from runtime import load_bundle

bundle_dir = Path("baseline_mlp_embedding_20260318_194239")
bundle = load_bundle(bundle_dir, device="cpu")

raw_joint63 = [0.0] * 63
result = bundle.predict(raw_joint63, tau=0.90)
print(result["pred_label"], result["confidence"])
```

## FastAPI에 붙이는 예시

```python
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

from runtime import load_bundle


class PredictRequest(BaseModel):
    raw_joint63: list[float] = Field(min_length=63, max_length=63)
    tau: float | None = None


bundle = load_bundle(Path("baseline_mlp_embedding_20260318_194239"), device="cpu")
app = FastAPI()


@app.post("/predict")
def predict(req: PredictRequest):
    return bundle.predict(req.raw_joint63, tau=req.tau)
```

## Flask에 붙이는 예시

```python
from pathlib import Path

from flask import Flask, jsonify, request

from runtime import load_bundle


bundle = load_bundle(Path("baseline_mlp_embedding_20260318_194239"), device="cpu")
app = Flask(__name__)


@app.post("/predict")
def predict():
    payload = request.get_json(force=True)
    raw_joint63 = payload["raw_joint63"]
    tau = payload.get("tau")
    return jsonify(bundle.predict(raw_joint63, tau=tau))
```

## 통합 시 주의사항
- 입력은 반드시 `x0,y0,z0,...,x20,y20,z20` 순서의 63개 float 이어야 합니다.
- `tau`를 주지 않으면 pure top-1 argmax 입니다.
- `tau`를 주면 `confidence < tau` 일 때 `neutral` 로 강제합니다.
- 이 번들은 `baseline/raw landmark` 전용이라 sequence, bone, angle feature와 바로 호환되지 않습니다.
