# preprocessing/

데이터 전처리와 관련된 코드가 있는 폴더입니다.

영상 데이터를 모델 학습에 사용할 수 있도록  
랜드마크 추출 및 feature 변환을 수행합니다.

---

## 1. mediapipe_infer.py
**역할:** 영상 또는 이미지에서 MediaPipe를 이용해 손 랜드마크를 추출합니다.

출력 데이터는 다음과 같습니다.

21개의 랜드마크  
(x, y, z 좌표)

---

## 2. landmark_vectorize.py
**역할:** 랜드마크 좌표를 모델 입력 형태로 변환합니다.

예시

21 landmarks × 3 좌표 = 63차원 feature vector

이 벡터는 이후 모델 학습에 사용됩니다.

---

## 3. labeling.py
**역할:** gesture 데이터에 라벨을 붙이는 작업을 수행합니다.

예시

clap  
tap  
pinch  
swipe

이러한 라벨을 데이터에 연결하여  
gesture classification 학습에 사용합니다.