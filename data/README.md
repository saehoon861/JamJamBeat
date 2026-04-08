# 학습용 데이터 준비

학습용 데이터 준비를 위한 코드 파일들입니다.

- 영상의 랜드마크를 추출할 수 있습니다.
- 영상과 추출된 랜드마크를 보면서 제스처에 대한 수동 라벨링을 할 수 있습니다.
  - 추출된 랜드마크가 없어도 영상을 보면서 수동 라벨링을 진행할 수 있으나, 추천하지 않습니다.

---

## 랜드마크 추출

→ [`data/landmark_extractor/README.md`](./landmark_extractor/README.md)

## 영상과 랜드마크를 보며 수동 라벨링

→ [`data/totalcheck_tool/README.md`](./totalcheck_tool/README.md)

## 영상별 CSV 통합 (랜드마크 + 라벨링)

`data/integration.py` 5번째 줄에서 출력 파일명을 본인 것으로 반드시 변경해주세요.

```python
OUTPUT_FILE = Path(__file__).parent / "total_data" / "sample.csv"  # sample.csv 를 변경
```

이름을 바꾸지 않으면 실행이 차단됩니다.

변경 후 아래 명령어를 실행하면 `total_data/` 안에 병합된 CSV가 생성됩니다.

```bash
cd data
uv run python integration.py
```