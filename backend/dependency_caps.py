"""Runtime checks for dependency major-version ceilings."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.metadata as metadata
import re
import sys
from typing import Iterable, List, Tuple


@dataclass(frozen=True)
class DependencyCap:
    package: str
    minimum: str
    maximum: str


OCR_DEPENDENCY_CAPS: Tuple[DependencyCap, ...] = (
    DependencyCap("rapidocr", "2.0.0", "4.0.0"),
    DependencyCap("rapidocr-onnxruntime", "1.4.0", "2.0.0"),
    DependencyCap("paddleocr", "3.0.0", "4.0.0"),
)


def _version_key(value: str) -> Tuple[int, int, int]:
    numbers = [int(part) for part in re.findall(r"\d+", value)[:3]]
    while len(numbers) < 3:
        numbers.append(0)
    return tuple(numbers)  # type: ignore[return-value]


def _within_cap(version: str, cap: DependencyCap) -> bool:
    parsed = _version_key(version)
    return _version_key(cap.minimum) <= parsed < _version_key(cap.maximum)


def check_ocr_dependency_caps(
    caps: Iterable[DependencyCap] = OCR_DEPENDENCY_CAPS,
) -> List[str]:
    """Return installed OCR engine packages that exceed supported ranges."""
    problems: List[str] = []
    for cap in caps:
        try:
            installed = metadata.version(cap.package)
        except metadata.PackageNotFoundError:
            continue
        if not _within_cap(installed, cap):
            problems.append(
                f"{cap.package}=={installed} outside supported range "
                f">={cap.minimum},<{cap.maximum}"
            )
    return problems


def enforce_ocr_dependency_caps() -> None:
    problems = check_ocr_dependency_caps()
    if problems:
        joined = "\n- ".join(problems)
        raise RuntimeError(f"OCR dependency version cap check failed:\n- {joined}")


def main() -> int:
    try:
        enforce_ocr_dependency_caps()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print("OCR dependency version caps OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
