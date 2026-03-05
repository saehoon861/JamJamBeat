# training/

모델 학습 및 평가 관련 코드가 있는 폴더입니다.

---

## 1. train.py
**역할:** gesture recognition 모델을 학습합니다.

dataset을 불러와 모델을 학습하고  
checkpoint를 저장합니다.

---

## 2. evaluation.py
**역할:** 학습된 모델의 성능을 평가합니다.

예시

accuracy  
confusion matrix

---

## 3. predict.py
**역할:** 학습된 모델을 이용해 gesture를 예측합니다.

테스트 데이터 또는 새로운 입력 데이터를 사용하여  
모델의 추론 결과를 확인합니다.