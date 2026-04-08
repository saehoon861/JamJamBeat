import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import torch

try:
    from test_backend.mlp_classifier_copy import GestureMLP
except ModuleNotFoundError:
    from mlp_classifier_copy import GestureMLP


DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[1] / "checkpoints" / "best_model.pth"
)
HOST = os.getenv("JAMJAM_HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", os.getenv("JAMJAM_PORT", "8008")))
MODEL_PATH = Path(os.getenv("JAMJAM_MODEL_PATH", str(DEFAULT_MODEL_PATH)))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 학습 시 사용한 클래스 인덱스를 사람이 읽는 라벨로 변환한다.
LABEL_MAP = {
    0: "None",
    1: "Class1",
    2: "Class2",
    3: "Class3",
    4: "Class4",
    5: "Class5",
    6: "Class6",
}


def landmarks_to_features(landmarks):
    # 프론트에서 전달한 21개 랜드마크를 모델 입력용 63차원 벡터로 펼친다.
    if not isinstance(landmarks, list) or len(landmarks) < 21:
        raise ValueError("landmarks must be a list with at least 21 points")

    features = []
    for point in landmarks[:21]:
        if not isinstance(point, dict):
            raise ValueError("each landmark must be an object")
        features.extend(
            [
                float(point.get("x", 0.0)),
                float(point.get("y", 0.0)),
                float(point.get("z", 0.0)),
            ]
        )
    return features


def load_model():
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model = GestureMLP(
        input_dim=63,
        num_classes=7,
        hidden_dims=[128, 64],
        dropout=0.003,
        use_batchnorm=True,
    )
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


MODEL = load_model()


class InferenceHandler(BaseHTTPRequestHandler):
    def _send_json(self, status_code, payload):
        # 브라우저와 바로 연결할 수 있도록 모든 응답을 JSON+CORS 형태로 통일한다.
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        # 로컬 프론트엔드의 preflight 요청을 허용한다.
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"status": "ok"})
            return
        if self.path == "/infer":
            self._send_json(405, {"error": "use POST /infer"})
            return
        if self.path == "/favicon.ico":
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/infer":
            self._send_json(404, {"error": "not found"})
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(content_length)
            payload = json.loads(raw.decode("utf-8"))

            # 이미 feature 벡터가 있으면 그대로 쓰고, 없으면 landmarks를 펼쳐서 사용한다.
            if "features" in payload:
                features = [float(v) for v in payload["features"]]
            else:
                features = landmarks_to_features(payload.get("landmarks", []))

            if len(features) != 63:
                raise ValueError("features length must be 63")

            x = torch.tensor(features, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                logits = MODEL(x)
                probs = torch.softmax(logits, dim=1).squeeze(0)
                # 가장 높은 확률의 클래스를 최종 예측으로 반환한다.
                pred_idx = int(torch.argmax(probs).item())
                confidence = float(probs[pred_idx].item())

            response = {
                "label": LABEL_MAP.get(pred_idx, str(pred_idx)),
                "class_id": pred_idx,
                "confidence": confidence,
                "probs": [float(v) for v in probs.tolist()],
                "model_version": "best_model.pth-test",
            }
            self._send_json(200, response)
        except Exception as exc:
            self._send_json(400, {"error": str(exc)})


if __name__ == "__main__":
    print(f"[test-backend] loading model: {MODEL_PATH}")
    print(
        f"[test-backend] torch: {torch.__version__}, torch.cuda: {torch.version.cuda}"
    )
    if DEVICE.type == "cuda":
        print(f"[test-backend] cuda device: {torch.cuda.get_device_name(0)}")
    else:
        print("[test-backend] device: cpu")
    print(f"[test-backend] listening on http://{HOST}:{PORT}")
    server = HTTPServer((HOST, PORT), InferenceHandler)
    server.serve_forever()
