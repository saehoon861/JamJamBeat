"""dataset_variant_runtime.py - Infer run dataset variants and recreate runtime landmark transforms."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DATASET_VARIANTS = ("baseline", "pos_only", "scale_only", "pos_scale")
BASELINE_VARIANT = "baseline"
_EPS = 1e-8

_POS_SCALE_RE = re.compile(r"(?:^|[_-])pos[_-]?scale(?:$|[_\.-])")
_SCALE_RE = re.compile(r"(?:^|[_-])scale(?:$|[_\.-])")
_POS_RE = re.compile(r"(?:^|[_-])pos(?:$|[_\.-])")


@dataclass(slots=True)
class VariantResolution:
    """Resolved dataset variant plus lightweight provenance for logging/debugging."""

    variant: str
    source: str
    warning: str | None = None


def _variant_from_name(name: str) -> str | None:
    lower = name.strip().lower()
    if not lower:
        return None
    if "pos_scale" in lower or _POS_SCALE_RE.search(lower):
        return "pos_scale"
    if "scale_only" in lower or _SCALE_RE.search(lower):
        return "scale_only"
    if "pos_only" in lower or _POS_RE.search(lower):
        return "pos_only"
    if (
        "baseline" in lower
        or "none" in lower
        or "poc_base" in lower
        or "_for_poc" in lower
        or "_notnull" in lower
    ):
        return BASELINE_VARIANT
    return None


def _single_variant(candidates: list[str]) -> str | None:
    cleaned = [candidate for candidate in candidates if candidate in DATASET_VARIANTS]
    if not cleaned:
        return None
    unique = set(cleaned)
    if len(unique) == 1:
        return cleaned[0]
    return None


def infer_dataset_variant(run_dir: Path, summary_path: Path | None) -> VariantResolution:
    """Infer which landmark-coordinate variant a run was trained on."""
    summary_candidates: list[str] = []

    if summary_path is not None and summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            for raw_path in summary.get("input_csv_paths", []):
                csv_name = Path(str(raw_path)).name
                variant = _variant_from_name(csv_name)
                if variant is not None:
                    summary_candidates.append(variant)
        except Exception as exc:
            return VariantResolution(
                variant=BASELINE_VARIANT,
                source="summary-error",
                warning=f"failed to parse run_summary.json: {exc}",
            )

    variant = _single_variant(summary_candidates)
    if variant is not None:
        return VariantResolution(variant=variant, source="input_csv_paths")

    name_candidates: list[str] = []
    for candidate in (
        run_dir.name,
        run_dir.parent.name,
        run_dir.parent.parent.name if run_dir.parent.parent != run_dir.parent else "",
    ):
        variant = _variant_from_name(candidate)
        if variant is not None:
            name_candidates.append(variant)

    variant = _single_variant(name_candidates)
    if variant is not None:
        warning = None
        if summary_candidates:
            warning = (
                "mixed or ambiguous input_csv_paths variants detected; "
                f"falling back to run/suite name -> {variant}"
            )
        return VariantResolution(variant=variant, source="run-name", warning=warning)

    warning = None
    if summary_candidates:
        warning = (
            "could not resolve a single dataset variant from input_csv_paths or run name; "
            "falling back to baseline"
        )
    return VariantResolution(variant=BASELINE_VARIANT, source="fallback", warning=warning)


def apply_dataset_variant(raw_landmarks: np.ndarray, dataset_variant: str) -> np.ndarray:
    """Recreate the landmark-coordinate variant used by training CSVs."""
    pts = raw_landmarks.astype(np.float32).copy()

    if dataset_variant == BASELINE_VARIANT:
        return pts

    origin = pts[0].copy()
    middle_knuckle = pts[9]
    denom = float(np.linalg.norm(middle_knuckle - origin))
    scale = 1.0 if denom <= _EPS else 1.0 / denom

    if dataset_variant == "pos_only":
        return (pts - origin).astype(np.float32)
    if dataset_variant == "scale_only":
        return (pts * scale).astype(np.float32)
    if dataset_variant == "pos_scale":
        return ((pts - origin) * scale).astype(np.float32)

    raise ValueError(f"Unsupported dataset_variant: {dataset_variant}")
