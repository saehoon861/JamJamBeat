"""core/class_pipeline.py - Singleton 패턴 기반 라벨 설정 관리 모듈

labeling_tool의 동일 파일을 재사용합니다.
config_path를 외부에서 주입받아 totalcheck_tool/config/ 대신
labeling_tool/config/label_config.yaml을 읽습니다.
"""

from pathlib import Path

import yaml


class LabelConfigManager:
    """YAML 설정 파일에서 클래스 ID↔Name, Color 매핑을 로드하여
    전역적으로 한 번만 읽어 여러 모듈에서 재사용하는 싱글톤 클래스.

    totalcheck_tool에서는 labeling_tool의 label_config.yaml을 그대로 재사용합니다.
    """

    _instance = None
    _config = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load_config(self, config_path: str | None = None) -> None:
        """label_config.yaml 파일을 로드합니다.

        config_path 미지정 시, 자동으로 labeling_tool/config/label_config.yaml을 참조합니다.
        (totalcheck_tool은 별도 config를 두지 않고 labeling_tool의 것을 재사용)
        """
        if config_path is None:
            # totalcheck_tool → data → labeling_tool/config
            tool_dir = Path(__file__).resolve().parent.parent   # totalcheck_tool/
            data_dir = tool_dir.parent                          # data/
            config_path = data_dir / "labeling_tool" / "config" / "label_config.yaml"
        else:
            config_path = Path(config_path)

        with open(config_path, "r", encoding="utf-8") as f:
            self._config = yaml.safe_load(f)

    @property
    def config(self) -> dict:
        if self._config is None:
            self.load_config()
        return self._config

    @property
    def fps(self) -> int:
        return self.config.get("fps", 30)

    @property
    def timeline_half_range(self) -> int:
        return self.config.get("timeline", {}).get("half_range", 60)

    @property
    def timeline_bar_height(self) -> int:
        return self.config.get("timeline", {}).get("bar_height", 40)

    def get_class_name(self, class_id: int) -> str:
        """클래스 ID 기반 매칭된 이름 텍스트 반환."""
        classes = self.config.get("classes", {})
        entry = classes.get(class_id, {})
        return entry.get("name", "Unknown")

    def get_class_color(self, class_id: int) -> tuple:
        """타임라인 표시용 BGR 색상 튜플 반환."""
        classes = self.config.get("classes", {})
        entry = classes.get(class_id, {})
        color = entry.get("color", [128, 128, 128])
        return tuple(color)

    def get_display_label(self, class_id: int) -> str:
        """화면 표시용 '{ID}_{Name}' 형태의 라벨 문자열 반환."""
        name = self.get_class_name(class_id)
        return f"{class_id}_{name}"

    @staticmethod
    def parse_target_from_filename(filename: str) -> int:
        """파일명에서 Target 클래스 ID를 추출합니다.

        예: '1_slow_Left_woman1.mp4' -> 1
            '0_neutral_right_woman1.mp4' -> 0
        """
        stem = Path(filename).stem
        while stem.endswith(".mp4"):
            stem = stem[:-4]

        first_part = stem.split("_")[0]
        try:
            return int(first_part)
        except ValueError:
            raise ValueError(
                f"파일명 '{filename}'에서 클래스 ID를 추출할 수 없습니다. "
                f"'{first_part}'은(는) 정수로 변환할 수 없습니다."
            )
