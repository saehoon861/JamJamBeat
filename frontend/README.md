# frontend/

## 1. mediapipe_infer.js
**역할:**  
브라우저에서 **MediaPipe Hands**를 실행하여 매 프레임마다 손 랜드마크(21개)를 추출합니다.

---

## 2. feature_vectorizer.js
**역할:**  
MediaPipe에서 추출한 랜드마크 좌표 데이터를 **모델 입력(feature vector)** 형태로 변환합니다.

예시  
- 21 landmarks × (x, y, z)  
- → **63차원 feature vector**

---

## 3. model_inference.js
**역할:**  
feature sequence를 입력으로 받아 **제스처 분류 결과를 예측**합니다.

모델 추론 과정에서  
- feature sequence 입력
- gesture classification 수행
- 예측된 제스처(label) 반환

---

## 4. sound_engine.js
**역할:**  
모델이 예측한 **제스처(label)**에 따라 해당하는 **사운드를 재생**합니다.

예시  
- `tap` → drum sound  
- `pinch` → snare sound