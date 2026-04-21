# models/

제스처 인식 모델이 정의되어 있는 폴더입니다.

MLP baseline 모델과  
Transformer 기반 모델이 포함되어 있습니다.

---

## 1. embedding/landmark_embedding.py
**역할:** 랜드마크 feature를 embedding vector로 변환합니다.

예시

63 → 128 dimension

이 과정은 모델이 더 의미 있는 feature를 학습하도록 돕습니다.

---

## 2. transformer/temporal_transformer.py
**역할:** 시간 순서(sequence)를 학습하는 Transformer Encoder 모델입니다.

여러 프레임의 랜드마크 데이터를 입력으로 받아  
gesture 패턴을 학습합니다.

---

## 3. baseline/mlp_classifier.py
**역할:** 간단한 MLP 모델을 이용한 gesture classification baseline입니다.

Transformer 모델과 성능 비교를 위해 사용됩니다.

---

## 4. gesture_model.py
**역할:** 전체 gesture recognition 모델을 구성하는 코드입니다.

구조

embedding  
↓  
transformer encoder  
↓  
classification layer