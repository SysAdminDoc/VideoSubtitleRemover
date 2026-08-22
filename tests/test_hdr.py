"""Tests for HDR codec/bit-depth preservation decisions."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from backend.hdr import (
    ColorMetadata,
    hdr_proxy_from_high_bit,
    hdr_proxy_to_linear,
    hdr_encode_args,
    hdr_pixel_format_args,
    hdr_repair_block_reason,
    hdr_repair_ready,
    hdr_safe_codec,
    hdr_signal_to_linear,
    linear_to_hdr_signal,
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


def test_pq_and_hlg_transfer_round_trip_keeps_high_bit_precision():
    values = np.linspace(0, 65535, 4096, dtype=np.uint16).reshape(1, -1, 1)
    for transfer in ("smpte2084", "arib-std-b67"):
        source = np.repeat(values, 3, axis=2)
        linear = hdr_signal_to_linear(source, transfer)
        encoded = linear_to_hdr_signal(linear, transfer)
        assert len(np.unique(encoded)) > 256
        assert int(np.max(np.abs(encoded.astype(np.int32)
                               - source.astype(np.int32)))) <= 2
        proxy = hdr_proxy_from_high_bit(source, transfer)
        lifted = hdr_proxy_to_linear(proxy, transfer)
        assert lifted.dtype == np.float32
        assert np.isfinite(lifted).all()


def test_hdr_repair_requires_consistent_color_tags():
    assert hdr_repair_ready(None) is False
    assert "metadata is unavailable" in hdr_repair_block_reason(None)
    assert hdr_repair_ready(ColorMetadata()) is False
    assert "incomplete or invalid" in hdr_repair_block_reason(ColorMetadata())
    assert hdr_repair_ready(_hdr_meta()) is True
    assert hdr_repair_ready(ColorMetadata(
        color_primaries="bt709",
        color_transfer="not-a-transfer",
        color_space="bt709",
    )) is False
    assert hdr_repair_ready(ColorMetadata(
        pixel_format="yuv420p10le",
    )) is False
    assert hdr_repair_ready(ColorMetadata(
        color_primaries="bt709",
        color_transfer="smpte2084",
        color_space="bt2020nc",
        color_range="tv",
    )) is False
    assert hdr_repair_ready(ColorMetadata(
        color_primaries="bt2020",
        color_transfer="smpte2084",
        color_space="",
        color_range="tv",
    )) is False
    conflicted = ColorMetadata(
        color_primaries="bt2020",
        color_transfer="smpte2084",
        color_space="bt2020nc",
        color_range="tv",
        tag_conflicts=("color_transfer",),
    )
    assert hdr_repair_ready(conflicted) is False
    assert "color_transfer" in hdr_repair_block_reason(conflicted)


def test_linear_light_merge_preserves_outside_pixels_and_sub_8bit_detail():
    width = 768
    source = np.zeros((4, width, 3), dtype=np.uint16)
    ramp = np.linspace(0, 65535, width, dtype=np.uint16)
    source[:] = ramp[None, :, None]
    mask = np.zeros((4, width), dtype=np.uint8)
    mask[:, 128:640] = 255
    remover = __import__("backend.processor", fromlist=["SubtitleRemover"])
    instance = remover.SubtitleRemover.__new__(remover.SubtitleRemover)
    instance.config = SimpleNamespace(mask_feather_px=0)
    instance._color_metadata = _hdr_meta()
    instance._hdr_repair_ready = True
    for transfer in ("smpte2084", "arib-std-b67"):
        instance._color_metadata = ColorMetadata(
            color_primaries="bt2020",
            color_transfer=transfer,
            color_space="bt2020nc",
            color_range="tv",
        )
        proxy = hdr_proxy_from_high_bit(source, transfer)
        repaired_proxy = proxy.copy()
        repaired_proxy[mask > 0] = np.clip(
            repaired_proxy[mask > 0].astype(np.int16) + 4,
            0,
            255,
        ).astype(np.uint8)
        output = instance._merge_high_bit_output(source, repaired_proxy, mask)
        assert output.dtype == np.uint16
        assert np.array_equal(output[mask == 0], source[mask == 0])
        assert np.any(output[mask > 0] != source[mask > 0])
        assert len(np.unique(output[mask > 0, 0])) > 256
        assert hdr_encode_args(instance._color_metadata) == [
            "-color_primaries", "bt2020",
            "-color_trc", transfer,
            "-colorspace", "bt2020nc",
            "-color_range", "tv",
        ]
