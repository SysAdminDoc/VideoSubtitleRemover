"""RM-39: INT8 dynamic quantization for the RapidOCR detection ONNX.

RapidOCR ships FP32 weights. Dynamic INT8 quantization via
onnxruntime cuts detection cost roughly in half on CPU with <1% F1
loss in practice. Output checkpoint is drop-in -- set RAPIDOCR_DET_PATH
(rapidocr 1.x) or pass the quantized path in the rapidocr 2.x API
when constructing the detector.

Usage:
    python scripts/quantize_ocr.py path/to/det.onnx path/to/det_int8.onnx

The script is intentionally standalone (no `backend.processor` import)
so it can run inside a clean venv that only has `onnxruntime`
installed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def quantize_path(input_onnx: str, output_onnx: str) -> int:
    """Quantize one ONNX file. Returns 0 on success, non-zero on
    failure."""
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType  # type: ignore
    except ImportError:
        print(
            "onnxruntime.quantization is not available. Install via "
            "`pip install onnxruntime>=1.16`.",
            file=sys.stderr,
        )
        return 2
    src = Path(input_onnx)
    if not src.is_file():
        print(f"Input ONNX not found: {input_onnx}", file=sys.stderr)
        return 3
    dst = Path(output_onnx)
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        quantize_dynamic(
            model_input=str(src),
            model_output=str(dst),
            weight_type=QuantType.QUInt8,
            optimize_model=False,
        )
    except Exception as exc:
        print(f"Quantization failed: {exc}", file=sys.stderr)
        return 4
    src_size = src.stat().st_size
    dst_size = dst.stat().st_size
    print(
        f"Quantized {src.name} -> {dst.name}: "
        f"{src_size / 1e6:.1f} MB -> {dst_size / 1e6:.1f} MB "
        f"({(1 - dst_size / src_size) * 100:.0f}% smaller)"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Quantize a RapidOCR detection ONNX to INT8 for ~2x CPU speedup."
    )
    parser.add_argument("input_onnx", help="Source FP32 detection ONNX")
    parser.add_argument("output_onnx", help="Destination INT8 ONNX")
    args = parser.parse_args()
    return quantize_path(args.input_onnx, args.output_onnx)


if __name__ == "__main__":
    sys.exit(main())
