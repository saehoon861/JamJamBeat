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

## 라벨링 클래스 분포 현황 확인

total_data 폴더 안에 있는 CSV 파일들의 라벨링 클래스에 대한 전체 분포 현황과 파일별 분포 현황을 확인할 수 있습니다.

> 병합본과 원본 CSV 파일의 구분을 잘 하고 작업해주세요.

> 병합본과 원본 csv 파일 두 종류가 전부 total_data 폴더 안에 있으면 전체 분포의 내용이 중복되어 출력됩니다.

```bash
cd data
uv run python check_labels.py
```