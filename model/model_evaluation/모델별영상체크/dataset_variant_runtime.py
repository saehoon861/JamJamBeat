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
    dataset_key: str | None = None
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


def _summary_dict(summary_path: Path) -> dict | None:
    loaded = json.loads(summary_path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else None


def _summary_value(summary: dict, *keys: str) -> object | None:
    current: object = summary
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _path_variants(raw_values: object) -> list[str]:
    if not isinstance(raw_values, list):
        return []
    candidates: list[str] = []
    for raw_path in raw_values:
        csv_name = Path(str(raw_path)).name
        variant = _variant_from_name(csv_name)
        if variant is not None:
            candidates.append(variant)
    return candidates


def infer_dataset_variant(run_dir: Path, summary_path: Path | None) -> VariantResolution:
    """Infer which landmark-coordinate variant a run was trained on."""
    if summary_path is not None and summary_path.exists():
        try:
            summary = _summary_dict(summary_path)
        except Exception as exc:
            return VariantResolution(
                variant=BASELINE_VARIANT,
                source="summary-error",
                dataset_key=None,
                warning=f"failed to parse run_summary.json: {exc}",
            )
        if summary is not None:
            dataset_info_key = _summary_value(summary, "dataset_info", "dataset_key")
            top_level_key = _summary_value(summary, "dataset_key")
            dataset_key_candidates = [
                candidate
                for candidate in (
                    _variant_from_name(str(dataset_info_key or "")),
                    _variant_from_name(str(top_level_key or "")),
                )
                if candidate is not None
            ]
            dataset_key = str(dataset_info_key or top_level_key or "").strip() or None

            variant = _single_variant(dataset_key_candidates)
            if variant is not None:
                source = "dataset_info.dataset_key" if dataset_info_key else "dataset_key"
                return VariantResolution(variant=variant, source=source, dataset_key=dataset_key)

            path_sources = [
                ("dataset_info.input_csv_paths", _summary_value(summary, "dataset_info", "input_csv_paths")),
                ("inputs", _summary_value(summary, "inputs")),
                ("input_csv_paths", _summary_value(summary, "input_csv_paths")),
            ]
            for source_name, raw_values in path_sources:
                path_candidates = _path_variants(raw_values)
                variant = _single_variant(path_candidates)
                if variant is not None:
                    return VariantResolution(
                        variant=variant,
                        source=source_name,
                        dataset_key=dataset_key,
                    )

            summary_warning = None
            if dataset_key_candidates or any(_path_variants(values) for _, values in path_sources):
                summary_warning = (
                    "could not resolve a single dataset variant from run_summary metadata; "
                    "falling back to run/suite name"
                )
        else:
            dataset_key = None
            summary_warning = "run_summary.json did not contain a JSON object; falling back to run/suite name"
    else:
        dataset_key = None
        summary_warning = None

    name_candidates: list[str] = []
    for candidate in (
        run_dir.name,
        run_dir.parent.parent.name if run_dir.parent.parent != run_dir.parent else "",
    ):
        variant = _variant_from_name(candidate)
        if variant is not None:
            name_candidates.append(variant)

    variant = _single_variant(name_candidates)
    if variant is not None:
        return VariantResolution(
            variant=variant,
            source="run-name",
            dataset_key=dataset_key,
            warning=summary_warning,
        )

    return VariantResolution(
        variant=BASELINE_VARIANT,
        source="fallback",
        dataset_key=dataset_key,
        warning=summary_warning,
    )


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
