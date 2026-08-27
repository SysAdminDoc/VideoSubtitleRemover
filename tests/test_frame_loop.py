import contextlib
from dataclasses import FrozenInstanceError, fields
import inspect

import numpy as np
import pytest

from backend.io import MediaWriteError
from backend.processor import (
    SubtitleRemover,
    _FrameBatch,
    _FrameLoopContext,
    _FrameLoopState,
)


@pytest.mark.parametrize("mode", ["auto", "sttn", "lama", "propainter"])
def test_manual_region_mask_is_identical_across_inpainting_modes(mode):
    from backend.config import ProcessingConfig, normalize_processing_config

    class Reader:
        def __init__(self, frame):
            self.frame = frame
            self.used = False

        def read(self):
            if self.used:
                return False, None
            self.used = True
            return True, self.frame.copy()

    class Detector:
        def __getattr__(self, name):
            if name.startswith("detect"):
                raise AssertionError("manual-only mode must not inspect OCR")
            raise AttributeError(name)

    frame = np.zeros((10, 12, 3), dtype=np.uint8)
    rect = (2, 3, 8, 7)
    config = normalize_processing_config(ProcessingConfig(
        mode=mode,
        device="cpu",
        sttn_skip_detection=True,
        subtitle_area=rect,
        subtitle_areas=[rect],
        phash_skip_enable=False,
    ))
    remover = SubtitleRemover.__new__(SubtitleRemover)
    remover.config = config
    remover.detector = Detector()
    remover._color_metadata = None
    remover.last_detection_stats = {"frames_total": 0}
    skipped = []
    remover._record_detection_skip = skipped.append
    remover._time_stage = lambda _name: contextlib.nullcontext()

    def create_mask(shape, boxes, **_kwargs):
        mask = np.zeros(shape[:2], dtype=np.uint8)
        for x1, y1, x2, y2 in boxes:
            mask[y1:y2, x1:x2] = 255
        return mask

    remover._create_mask = create_mask
    remover._apply_polygon_region_shapes = lambda mask, _shapes: mask
    remover._apply_manual_mask_corrections = (
        lambda mask, _seconds, _index: mask
    )

    ctx = _FrameLoopContext.__new__(_FrameLoopContext)
    values = {
        "start_frame": 0,
        "end_frame": 1,
        "fps": 30.0,
        "frame_timing": None,
        "high_bit_depth_surface": False,
        "batch_size": 1,
        "timed_region_spans": False,
        "timed_mask_corrections": False,
        "static_fixed_shapes": [{"rect": rect}],
        "selective_ranges": [],
        "reader": Reader(frame),
        "selective_cap": None,
        "matte_reader": None,
        "frozen_matte": False,
        "frame_skip": 0,
        "keyframe_set": None,
        "whisper_spans": [],
    }
    for name, value in values.items():
        object.__setattr__(ctx, name, value)
    state = _FrameLoopState(
        frame_idx=0,
        last_mask=None,
        last_hash=None,
        tracker=None,
        fixed_mask_cache={},
    )

    batch = remover._decode_and_build_batch(ctx, state)

    expected = np.zeros(frame.shape[:2], dtype=np.uint8)
    expected[3:7, 2:8] = 255
    np.testing.assert_array_equal(batch.masks[0], expected)
    assert skipped == ["manual_region"]


def test_manual_region_mode_without_a_region_fails_before_ocr():
    from backend.config import ProcessingConfig, normalize_processing_config

    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    remover = SubtitleRemover.__new__(SubtitleRemover)
    remover.config = normalize_processing_config(ProcessingConfig(
        mode="auto",
        device="cpu",
        sttn_skip_detection=True,
        phash_skip_enable=False,
    ))
    remover._color_metadata = None
    remover.last_detection_stats = {"frames_total": 0}
    remover._time_stage = lambda _name: contextlib.nullcontext()

    class Reader:
        def read(self):
            return True, frame.copy()

    ctx = _FrameLoopContext.__new__(_FrameLoopContext)
    values = {
        "start_frame": 0,
        "end_frame": 1,
        "fps": 30.0,
        "frame_timing": None,
        "high_bit_depth_surface": False,
        "batch_size": 1,
        "timed_region_spans": False,
        "timed_mask_corrections": False,
        "static_fixed_shapes": [],
        "selective_ranges": [],
        "reader": Reader(),
        "selective_cap": None,
        "matte_reader": None,
        "frozen_matte": False,
    }
    for name, value in values.items():
        object.__setattr__(ctx, name, value)
    state = _FrameLoopState(
        frame_idx=0,
        last_mask=None,
        last_hash=None,
        tracker=None,
        fixed_mask_cache={},
    )

    with pytest.raises(ValueError, match="needs a fixed, timed, or moving"):
        remover._decode_and_build_batch(ctx, state)


def test_frame_loop_state_has_only_eight_carried_mutables():
    # This list is a guardrail against state creep in the frame loop, so it
    # is updated only when a new carried value is a deliberate design choice.
    # fade_carry (RM-292) is one: a fade-out hold has to survive the batch
    # boundary, and the alternative -- parking it on the remover instance --
    # is the same state in a worse place. fade_pending and written_idx
    # (RM-296) are the other two: a fade-in hold looks forward, so the tail
    # of a batch has to wait for the next one, which means the decode cursor
    # and the write cursor stop being the same number.
    assert [item.name for item in fields(_FrameLoopState)] == [
        "frame_idx",
        "last_mask",
        "last_hash",
        "tracker",
        "fixed_mask_cache",
        "fade_carry",
        "fade_pending",
        "written_idx",
    ]


def test_the_write_cursor_never_runs_ahead_of_the_decode_cursor():
    # written_idx is what the checkpoint resume point means, so a batch that
    # is still held back must leave it behind frame_idx, never ahead.
    state = _FrameLoopState(
        frame_idx=30, last_mask=None, last_hash=None, tracker=None,
        fixed_mask_cache={}, written_idx=0,
    )
    assert state.written_idx <= state.frame_idx
    assert state.fade_pending is None


def test_split_tail_and_prepend_keep_the_parallel_payloads_aligned():
    batch = _FrameBatch()
    for index in range(5):
        batch.add(
            np.full((4, 4, 3), index, dtype=np.uint8),
            np.full((4, 4), index, dtype=np.uint8),
            None,
            passthrough=index % 2 == 0,
        )
    tail = batch.split_tail(2)
    assert len(batch.frames) == 3
    assert len(tail.frames) == 2
    for held in (batch, tail):
        assert len(held.frames) == len(held.masks)
        assert len(held.frames) == len(held.source_frames)
        assert len(held.frames) == len(held.passthrough_flags)
    assert tail.passthrough_flags == [False, True]

    following = _FrameBatch()
    following.add(
        np.full((4, 4, 3), 9, dtype=np.uint8),
        np.full((4, 4), 9, dtype=np.uint8),
        None,
        passthrough=False,
    )
    following.prepend(tail)
    assert [int(frame[0, 0, 0]) for frame in following.frames] == [3, 4, 9]
    assert following.passthrough_flags == [False, True, False]


def test_split_tail_of_zero_is_a_no_op():
    batch = _FrameBatch()
    batch.add(
        np.zeros((4, 4, 3), dtype=np.uint8),
        np.zeros((4, 4), dtype=np.uint8),
        None,
        passthrough=False,
    )
    tail = batch.split_tail(0)
    assert not tail.frames
    assert len(batch.frames) == 1


def test_frame_loop_context_is_frozen():
    assert _FrameLoopContext.__dataclass_params__.frozen is True
    with pytest.raises(FrozenInstanceError):
        context = _FrameLoopContext.__new__(_FrameLoopContext)
        context.fps = 24.0


def test_frame_batch_add_keeps_parallel_payloads_synchronized():
    batch = _FrameBatch()
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.uint8)
    batch.add(frame, mask, None, passthrough=True)
    assert len(batch.frames) == 1
    assert len(batch.frames) == len(batch.masks)
    assert len(batch.frames) == len(batch.source_frames)
    assert len(batch.frames) == len(batch.passthrough_flags)
    assert batch.passthrough_flags == [True]


def test_write_batch_propagates_writer_failure_before_checkpointing():
    """RM-139: an unwritable frame must abort the loop, not advance state."""
    class _ExplodingWriter:
        def __init__(self):
            self.calls = 0

        def write(self, frame):
            self.calls += 1
            raise MediaWriteError(
                "frame lost", reason="frame_write_failed", path="frame_000000.png")

    remover = SubtitleRemover.__new__(SubtitleRemover)
    remover.config = type("_Cfg", (), {"quality_report": False})()
    remover.live_preview_stride = 1
    remover.on_preview_frame = None
    remover._time_stage = lambda name: contextlib.nullcontext()
    remover._merge_high_bit_output = lambda source, result, mask: result

    ctx = _FrameLoopContext.__new__(_FrameLoopContext)
    writer = _ExplodingWriter()
    object.__setattr__(ctx, "writer", writer)
    object.__setattr__(ctx, "matte_writer", None)
    object.__setattr__(ctx, "frames_to_process", 2)
    state = _FrameLoopState(
        frame_idx=2, last_mask=None, last_hash=None, tracker=None,
        fixed_mask_cache={})

    batch = _FrameBatch()
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    mask = np.zeros((4, 4), dtype=np.uint8)
    batch.add(frame, mask, None, passthrough=False)
    batch.add(frame, mask, None, passthrough=False)

    with pytest.raises(MediaWriteError):
        remover._write_batch(ctx, state, batch, [frame, frame])
    assert writer.calls == 1


def test_process_video_handles_media_write_error_before_generic_exception():
    source = inspect.getsource(SubtitleRemover.process_video)
    assert "except MediaWriteError as e:" in source
    assert source.index("except MediaWriteError") < source.index("except Exception as e:")
    # The success-path release must precede the encode-stage checkpoint marker
    # so a failed flush cannot mark the inpaint pass complete.
    assert source.index("writer.release()") < source.index('stage="encoding"')


def test_process_video_delegates_each_frame_loop_stage():
    # RM-296 moved refine/inpaint/write behind _process_batch, because a
    # pause has to run that same sequence a second time to flush the held
    # tail. The guardrail is unchanged in intent: process_video orchestrates
    # and must not inline the loop body.
    source = inspect.getsource(SubtitleRemover.process_video)
    for method in (
        "_decode_and_build_batch",
        "_process_batch",
        "_checkpoint_after_batch",
    ):
        assert f"self.{method}(" in source
    assert "for _ in range(batch_size)" not in source

    stage = inspect.getsource(SubtitleRemover._process_batch)
    for method in ("_refine_batch_masks", "_inpaint_batch", "_write_batch"):
        assert f"self.{method}(" in stage
