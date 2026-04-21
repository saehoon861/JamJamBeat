"""core/annotation_state.py - [Model] 프레임 라벨 상태 추적 모듈"""


class AnnotationState:
    """영상 한 건에 종속된 프레임별 라벨 상태를 관리하는 Model 클래스.
    
    - 모든 프레임은 초기 라벨 0 (Neutral)으로 초기화
    - Space bar 토글, S/F 구간 덮어쓰기 등을 지원
    - S 마커 활성 여부로 Enter 저장 잠금 판단 제공
    """

    def __init__(self, total_frames: int, target_class: int):
        self.total_frames = total_frames
        self.target_class = target_class
        self.labels: list[int] = [0] * total_frames
        self.start_marker: int | None = None

    def toggle_frame(self, index: int) -> None:
        """해당 프레임의 라벨을 0 ↔ target_class로 토글합니다."""
        if 0 <= index < self.total_frames:
            if self.labels[index] == 0:
                self.labels[index] = self.target_class
            else:
                self.labels[index] = 0

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

        # end+1: 파이썬 슬라이싱에서 끝 인덱스를 포함하기 위함
        for i in range(start, end + 1):
            if 0 <= i < self.total_frames:
                self.labels[i] = target_class

        self.start_marker = None

    def is_range_active(self) -> bool:
        """S 마커가 활성 상태인지 반환합니다.
        True이면 Enter 저장이 잠깁니다."""
        return self.start_marker is not None

    def get_label(self, index: int) -> int:
        """특정 프레임의 현재 라벨 값을 반환합니다."""
        if 0 <= index < self.total_frames:
            return self.labels[index]
        return 0
