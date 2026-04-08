# pipelines/

모델 학습 및 추론 전체 과정을 관리하는 코드입니다.

---

## 1. train_pipeline.py
**역할:** 모델 학습 전체 과정을 실행합니다.

과정

dataset 생성  
↓  
model training  
↓  
evaluation  
↓  
checkpoint 저장

---

## 2. inference_pipeline.py
**역할:** 학습된 모델을 이용해 제스처를 추론하는 전체 과정을 실행합니다.