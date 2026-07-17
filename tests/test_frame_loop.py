from dataclasses import FrozenInstanceError, fields
import inspect

import numpy as np
import pytest

from backend.processor import (
    SubtitleRemover,
    _FrameBatch,
    _FrameLoopContext,
    _FrameLoopState,
)


def test_frame_loop_state_has_only_five_carried_mutables():
    assert [item.name for item in fields(_FrameLoopState)] == [
        "frame_idx",
        "last_mask",
        "last_hash",
        "tracker",
        "fixed_mask_cache",
    ]


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


def test_process_video_delegates_each_frame_loop_stage():
    source = inspect.getsource(SubtitleRemover.process_video)
    for method in (
        "_decode_and_build_batch",
        "_refine_batch_masks",
        "_inpaint_batch",
        "_write_batch",
        "_checkpoint_after_batch",
    ):
        assert f"self.{method}(" in source
    assert "for _ in range(batch_size)" not in source
