# dataset/

모델 학습에 사용할 데이터셋을 구성하는 코드입니다.

랜드마크 데이터를 불러오고  
모델 입력 형태로 변환합니다.

---

## 1. gesture_dataset.py
**역할:** PyTorch Dataset 클래스를 정의합니다.

dataset을 통해 모델 학습 시 다음 데이터를 제공합니다.

입력 데이터  
gesture sequence

라벨  
gesture class

---

## 2. sliding_window.py
**역할:** 랜드마크 시퀀스를 일정한 길이의 프레임으로 분할합니다.

예시

16 frames × 63 feature

이 방식은 제스처의 시간 흐름을 학습하기 위해 사용됩니다.