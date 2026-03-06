"""core/annotation_state.py - [Model] 프레임 라벨 상태 추적 모듈

labeling_tool과의 차이점:
- 초기화 시 기존 labeled_data CSV의 gesture 컬럼 값을 주입받아 labels 배열을 구성.
  (0으로 가득 찬 초기 배열 대신, 기존 라벨링 결과를 복원)
- 나머지 toggle / set_range / is_range_active 로직은 동일.
"""


class AnnotationState:
    """영상 한 건에 종속된 프레임별 라벨 상태를 관리하는 Model 클래스.

    - 기존 labeled_data CSV의 gesture 값으로 초기화 (검수 목적)
    - Space bar 토글, S/F 구간 덮어쓰기 지원
    - S 마커 활성 여부로 Enter 저장 잠금 판단 제공
    """

    def __init__(self, existing_labels: list[int], target_class: int):
        """
        Args:
            existing_labels: labeled_data CSV의 gesture 컬럼을 리스트로 변환한 값.
                             'n' 또는 'Esc' 키 입력 시 이 값은 파일에 저장되지 않아
                             원본 CSV가 안전하게 보존됩니다.
            target_class: 파일명에서 파싱한 목표 제스처 클래스 ID.
        """
        self.total_frames = len(existing_labels)
        self.target_class = target_class
        # 기존 labeled_data의 gesture 값으로 State 초기화 (복사본으로 보관)
        self.labels: list[int] = list(existing_labels)
        self.start_marker: int | None = None

    def toggle_frame(self, index: int) -> None:
        """해당 프레임의 라벨을 0 ↔ target_class로 토글합니다."""
        if 0 <= index < self.total_frames:
            if self.labels[index] == 0:
                self.labels[index] = self.target_class
            else:
                self.labels[index] = 0

    def set_label(self, index: int, class_id: int) -> None:
        """해당 프레임의 라벨을 직접 지정된 class_id로 설정합니다.
        (숫자키 0~6 직접 라벨링용)
        """
        if 0 <= index < self.total_frames:
            self.labels[index] = class_id

    def set_range_start(self, index: int) -> None:
        """S 키: 구간 라벨링의 시작 프레임을 마킹합니다."""
        self.start_marker = index

    def set_range_end(self, index: int, target_class: int) -> None:
        """F 키: 구간 라벨링의 종료 프레임을 마킹하고 일괄 덮어쓰기합니다.

        - start_marker가 None이면 조용히 무시(Pass)
        - index+1까지 슬라이싱하여 끝 프레임 자체도 포함
        - 덮어쓰기 후 start_marker를 None으로 리셋
        """
        if self.start_marker is None:
            return

        start = min(self.start_marker, index)
        end = max(self.start_marker, index)

        for i in range(start, end + 1):
            if 0 <= i < self.total_frames:
                self.labels[i] = target_class

        self.start_marker = None

    def set_range_end_with_class(self, index: int, class_id: int) -> None:
        """F 키: 특정 클래스 ID로 구간을 마킹합니다. (set_label과 조합용)"""
        self.set_range_end(index, class_id)

    def is_range_active(self) -> bool:
        """S 마커가 활성 상태인지 반환합니다.
        True이면 Enter 저장이 잠깁니다."""
        return self.start_marker is not None

    def get_label(self, index: int) -> int:
        """특정 프레임의 현재 라벨 값을 반환합니다."""
        if 0 <= index < self.total_frames:
            return self.labels[index]
        return 0
