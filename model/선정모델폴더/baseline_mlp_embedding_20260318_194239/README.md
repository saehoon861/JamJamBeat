# baseline_mlp_embedding_20260318_194239 - 메인스트림 전달용 self-contained 모델 번들

이 폴더는 JamJamBeat에서 선정한 `baseline / mlp_embedding / 20260318_194239` 체크포인트를 다른 Python 서비스에 바로 붙이기 좋게 정리한 전달용 번들입니다.

핵심 특징:
- 원본 학습 파이프라인 없이 `runtime/` 만으로 checkpoint 로드 가능
- 입력 스펙, 클래스 순서, 선택 근거, 평가 요약을 파일별로 분리
- AI가 읽어서 integration code를 생성하기 쉬운 구조

추천 읽기 순서:
1. `MANIFEST.json`
2. `MODEL_CARD.md`
3. `runtime/input_spec.json`
4. `runtime/inference.py`
5. `examples/example_usage.py`

폴더 구성:
- `runtime/`: 실제 서비스 연결용 최소 추론 코드와 checkpoint
- `artifacts/`: 원본 run 메타와 선택 근거
- `examples/`: 샘플 입력/출력과 통합 예제

빠른 시작:

```bash
cd model/선정모델폴더/baseline_mlp_embedding_20260318_194239
python -m venv .venv
. .venv/bin/activate
python -m pip install -r runtime/requirements.txt
python examples/example_usage.py
```

핵심 선택 근거:
- `accuracy`: `0.7747`
- `macro_f1`: `0.7948`
- `class0_fpr`: `0.0884`
- `class0_fnr`: `0.5929`

입력 요약:
- 모델 타입: `mlp_embedding`
- 모드: `frame`
- 입력: raw joint 63차원 한 프레임
- 입력 순서: `x0,y0,z0,...,x20,y20,z20`

중요 주의사항:
- 이 모델은 `baseline` 계열 raw landmark 입력 전용입니다.
- bone/angle/normalized image 입력에는 바로 사용할 수 없습니다.
- `tau`는 학습 원본 run에 기록된 source-of-truth 값이 없어서, 번들 기본값은 `null` 입니다.
- 필요 시 호출부에서 `tau=0.90` 같은 값을 넘겨 neutral fallback을 적용하면 됩니다.
