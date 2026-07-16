"""Tests for HDR codec/bit-depth preservation decisions."""

from __future__ import annotations

from backend.hdr import (
    ColorMetadata,
    hdr_pixel_format_args,
    hdr_safe_codec,
)


def _hdr_meta() -> ColorMetadata:
    return ColorMetadata(
        color_primaries="bt2020",
        color_transfer="smpte2084",
        color_space="bt2020nc",
        color_range="tv",
    )


def _sdr_meta() -> ColorMetadata:
    return ColorMetadata(
        color_primaries="bt709",
        color_transfer="bt709",
        color_space="bt709",
    )


def test_is_hdr_detection():
    assert _hdr_meta().is_hdr is True
    assert ColorMetadata(color_transfer="arib-std-b67").is_hdr is True
    assert _sdr_meta().is_hdr is False


def test_label_composition():
    assert _hdr_meta().label == "bt2020 / smpte2084 / bt2020nc"
    assert ColorMetadata().label == "unknown"


def test_hdr_source_promotes_incompatible_codec_to_hevc():
    assert hdr_safe_codec("h264", _hdr_meta()) == "h265"


def test_hdr_source_keeps_compatible_codec():
    for codec in ("h265", "av1", "vvc"):
        assert hdr_safe_codec(codec, _hdr_meta()) == codec


def test_sdr_source_leaves_codec_untouched():
    assert hdr_safe_codec("h264", _sdr_meta()) == "h264"
    assert hdr_safe_codec("h264", None) == "h264"


def test_hdr_forces_10bit_pixel_format():
    assert hdr_pixel_format_args(_hdr_meta(), "h265") == [
        "-pix_fmt", "yuv420p10le"]
    assert hdr_pixel_format_args(_hdr_meta(), "h265", hardware=True) == [
        "-pix_fmt", "p010le"]


def test_no_pixel_format_override_for_sdr_or_incompatible_codec():
    assert hdr_pixel_format_args(_sdr_meta(), "h265") == []
    # HDR meta but a codec that cannot carry HDR -> no 10-bit surface.
    assert hdr_pixel_format_args(_hdr_meta(), "h264") == []
