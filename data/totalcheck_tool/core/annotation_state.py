"""core/annotation_state.py - [Model] 프레임 라벨 상태 추적 모듈 (Phase 2)

Phase 2 변경사항:
- 초기화 시 existing_labels가 None이면 total_frames만큼 0으로 채움 (백지 라벨링).
- 3단계 구간 라벨링 State Machine:
  1단계: S → start_marker 설정
  2단계: S 재입력 → cancel_range() (전면 취소)
  3단계: F → end_marker 설정 (대기 상태)
  확정: 0~6 입력 → fill_range_with_class(id) (일괄 덮어쓰기)
"""


class AnnotationState:
    """영상 한 건에 종속된 프레임별 라벨 상태를 관리하는 Model 클래스.

    - 기존 labeled_data CSV의 gesture 값으로 초기화 (검수 목적)
    - None이면 전부 0으로 초기화 (백지 라벨링)
    - Space bar 토글, S/F 구간 덮어쓰기 지원
    - S 마커 활성 여부로 Enter 저장 잠금 판단 제공
    """

    def __init__(self, existing_labels: list[int] | None, target_class: int, total_frames: int = 0):
        """
        Args:
            existing_labels: labeled_data CSV의 gesture 컬럼을 리스트로 변환한 값.
                             None이면 백지 라벨링 (전부 0으로 초기화).
            target_class: 파일명에서 파싱한 목표 제스처 클래스 ID.
            total_frames: 백지 라벨링 시 프레임 수 (existing_labels가 None일 때 사용).
        """
        if existing_labels is not None and len(existing_labels) > 0:
            self.total_frames = len(existing_labels)
            self.labels: list[int] = list(existing_labels)
        else:
            self.total_frames = total_frames
            self.labels = [0] * total_frames

        self.target_class = target_class
        self.start_marker: int | None = None
        self.end_marker: int | None = None

    def toggle_frame(self, index: int) -> None:
        """해당 프레임의 라벨을 0 ↔ target_class로 토글합니다."""
        if 0 <= index < self.total_frames:
            if self.labels[index] == 0:
                self.labels[index] = self.target_class
            else:
                self.labels[index] = 0

    def set_label(self, index: int, class_id: int) -> None:
        """해당 프레임의 라벨을 직접 지정된 class_id로 설정합니다.
        (숫자키 0~6 직접 라벨링용, 기존 함수 그대로 유지)
        """
        if 0 <= index < self.total_frames:
            self.labels[index] = class_id

    def set_range_start(self, index: int) -> None:
        """S 키: 구간 라벨링의 시작 프레임을 마킹합니다. (1단계)"""
        self.start_marker = index
        self.end_marker = None  # 새로 시작하면 end는 초기화

    def set_range_end(self, index: int) -> None:
        """F 키: 구간 라벨링의 종료 프레임을 마킹합니다. (3단계 대기 모드 진입)

        Phase 2 변경: 기존처럼 즉시 덮어쓰지 않고 end_marker만 지정합니다.
        이후 0~6 숫자키를 눌러 fill_range_with_class()를 호출해야 확정됩니다.
        """
        if self.start_marker is None:
            return
        self.end_marker = index

    def fill_range_with_class(self, class_id: int) -> None:
        """0~6 확정키: start_marker ~ end_marker 범위를 class_id로 일괄 덮어쓰기합니다.

        - 두 마커가 모두 None이 아닐 때만 동작
        - 덮어쓰기 후 두 마커를 모두 None으로 리셋
        """
        if self.start_marker is None or self.end_marker is None:
            return

        start = min(self.start_marker, self.end_marker)
        end = max(self.start_marker, self.end_marker)

        for i in range(start, end + 1):
            if 0 <= i < self.total_frames:
                self.labels[i] = class_id

        self.start_marker = None
        self.end_marker = None

    def cancel_range(self) -> None:
        """S 재입력 또는 기타 취소: 두 마커를 모두 강제 초기화합니다."""
        self.start_marker = None
        self.end_marker = None

    def is_range_active(self) -> bool:
        """S 마커가 활성 상태인지 반환합니다. (1단계 or 3단계)
        True이면 Enter 저장이 잠깁니다."""
        return self.start_marker is not None

    def is_waiting_for_class(self) -> bool:
        """3단계 대기 상태인지 반환합니다.
        start_marker와 end_marker가 모두 지정되었으나 클래스 미확정 상태.

        정확한 조건: (self.start_marker is not None) and (self.end_marker is not None)
        """
        return (self.start_marker is not None) and (self.end_marker is not None)

    def get_label(self, index: int) -> int:
        """특정 프레임의 현재 라벨 값을 반환합니다."""
        if 0 <= index < self.total_frames:
            return self.labels[index]
        return 0
