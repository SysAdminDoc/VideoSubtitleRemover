from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np
import pytest
from xml.etree import ElementTree

from backend import io
from backend import resume_checkpoint
from backend.frozen_matte import FrozenMatteError, freeze_matte, validate_frozen_matte
from backend.matte_interchange import MaskInterchangeReader, MaskInterchangeWriter
from backend.nle_sidecar import write_edl, write_fcpxml


def _probe_result(payload: str):
    return SimpleNamespace(returncode=0, stdout=payload, stderr="")


def test_probe_keeps_1001_timeline_as_integer_ticks():
    stream = json.dumps({
        "streams": [{
            "avg_frame_rate": "60000/1001",
            "r_frame_rate": "60000/1001",
            "time_base": "1/60000",
            "start_time": "0.5",
            "duration": "0.0834166667",
        }],
    })
    frames = (
        "best_effort_timestamp=30000|duration=1001|"
        "best_effort_timestamp_time=0.5|duration_time=0.0166833333333333|"
        "pkt_duration=77|pkt_duration_time=0.0012833333333333\n"
        "best_effort_timestamp=31001|duration=1001|"
        "best_effort_timestamp_time=0.5166833333333333|"
        "duration_time=0.0166833333333333|pkt_duration=77|"
        "pkt_duration_time=0.0012833333333333\n"
        "best_effort_timestamp=33003|duration=1001|"
        "best_effort_timestamp_time=0.55005|"
        "duration_time=0.0166833333333333|pkt_duration=77|"
        "pkt_duration_time=0.0012833333333333\n"
    )
    runner = mock.Mock(
        side_effect=[_probe_result(stream), _probe_result(frames)]
    )
    with mock.patch.object(io.shutil, "which", return_value="ffprobe"), \
            mock.patch.object(io, "run_process", runner):
        timing = io._probe_video_frame_timing("clip.mkv", timeout=30.0)

    assert timing is not None
    assert timing.timestamp_ticks == [0, 1001, 3003]
    assert timing.duration_ticks == [1001, 2002, 1001]
    assert (timing.time_base_num, timing.time_base_den) == (1, 60000)
    assert timing.source_start_ticks == 30000
    assert timing.stream_start_ticks == 30000
    assert timing.total_duration_ticks == 4004
    assert abs(timing.duration - (4004 / 60000)) < 1e-15
    frame_command = runner.call_args_list[1].args[0]
    entries = frame_command[frame_command.index("-show_entries") + 1]
    assert "duration" in entries
    assert "duration_time" in entries
    assert "pkt_duration" in entries
    assert "pkt_duration_time" in entries


def test_probe_prefers_ffmpeg9_duration_fields_over_packet_fallbacks():
    stream = json.dumps({
        "streams": [{
            "avg_frame_rate": "25/1",
            "r_frame_rate": "25/1",
            "time_base": "1/1000",
            "start_time": "0",
            "duration": "0.205",
        }],
    })
    frames = (
        "best_effort_timestamp=0|duration=40|"
        "best_effort_timestamp_time=0|duration_time=0.040|"
        "pkt_duration=900|pkt_duration_time=0.900\n"
        "best_effort_timestamp=40|duration=40|"
        "best_effort_timestamp_time=0.040|duration_time=0.040|"
        "pkt_duration=900|pkt_duration_time=0.900\n"
        "best_effort_timestamp=80|duration=N/A|"
        "best_effort_timestamp_time=0.080|duration_time=0.125|"
        "pkt_duration=40|pkt_duration_time=0.040\n"
    )
    with mock.patch.object(io.shutil, "which", return_value="ffprobe"), \
            mock.patch.object(
                io,
                "run_process",
                side_effect=[_probe_result(stream), _probe_result(frames)],
            ):
        timing = io._probe_video_frame_timing("ffmpeg9.mkv", timeout=30.0)

    assert timing is not None
    assert timing.timestamp_ticks == [0, 40, 80]
    assert timing.duration_ticks == [40, 40, 125]
    assert timing.total_duration_ticks == 205


def test_probe_keeps_authoritative_long_final_frame_duration():
    stream = json.dumps({
        "streams": [{
            "avg_frame_rate": "25/1",
            "r_frame_rate": "25/1",
            "time_base": "1/1000",
            "start_time": "0",
            "duration": "1.080",
        }],
    })
    frames = (
        "best_effort_timestamp=0|duration=40|"
        "best_effort_timestamp_time=0|duration_time=0.040\n"
        "best_effort_timestamp=40|duration=40|"
        "best_effort_timestamp_time=0.040|duration_time=0.040\n"
        "best_effort_timestamp=80|duration=1000|"
        "best_effort_timestamp_time=0.080|duration_time=1.000\n"
    )
    with mock.patch.object(io.shutil, "which", return_value="ffprobe"), \
            mock.patch.object(
                io,
                "run_process",
                side_effect=[_probe_result(stream), _probe_result(frames)],
            ):
        timing = io._probe_video_frame_timing("long-tail.mkv", timeout=30.0)

    assert timing is not None
    assert timing.duration_ticks == [40, 40, 1000]
    assert timing.total_duration_ticks == 1080
    assert not any(
        item["kind"] in {"missing_duration", "repaired_duration"}
        for item in timing.anomalies
    )


def test_probe_logs_and_repairs_missing_repeated_and_non_monotonic_pts(caplog):
    stream = json.dumps({
        "streams": [{
            "avg_frame_rate": "25/1",
            "r_frame_rate": "25/1",
            "time_base": "1/1000",
            "start_time": "0",
            "duration": "0.2",
        }],
    })
    frames = (
        "N/A,40,N/A,0.04\n"
        "1000,40,1.0,0.04\n"
        "1000,40,1.0,0.04\n"
        "900,40,0.9,0.04\n"
    )
    with mock.patch.object(io.shutil, "which", return_value="ffprobe"), \
            mock.patch.object(
                io,
                "run_process",
                side_effect=[_probe_result(stream), _probe_result(frames)],
            ), caplog.at_level("WARNING", logger="backend.io"):
        timing = io._probe_video_frame_timing("damaged.mkv", timeout=30.0)

    assert timing is not None
    kinds = {entry["kind"] for entry in timing.anomalies}
    assert {
        "missing_timestamp",
        "repaired_timestamp",
        "repeated_pts",
        "non_monotonic_pts",
    } <= kinds
    messages = "\n".join(record.message for record in caplog.records)
    assert "missing_timestamp" in messages
    assert "repeated_pts" in messages
    assert "non_monotonic_pts" in messages
    assert "repaired_timestamp" in messages
    assert timing.timestamp_ticks == [0, 1000, 1040, 1080]


def test_frame_range_does_not_round_exact_subframe_boundaries():
    timing = io.VideoFrameTiming(
        timestamp_ticks=[0, 1, 2, 3],
        duration_ticks=[1, 1, 1, 1],
        time_base_num=1,
        time_base_den=60000,
        average_fps=60000.0,
    )
    assert timing.frame_range(
        Fraction(1, 60000), Fraction(3, 60000), 4
    ) == (1, 3)


def test_sparse_timing_rows_keep_timestamp_time_in_the_right_column():
    stream = json.dumps({
        "streams": [{
            "avg_frame_rate": "25/1",
            "r_frame_rate": "25/1",
            "time_base": "1/1000",
            "start_time": "0",
            "duration": "1.08",
        }],
    })
    frames = (
        "0,0.000000\n"
        "40,0.040000\n"
        "80,0.080000\n"
        "1040,1.040000\n"
    )
    with mock.patch.object(io.shutil, "which", return_value="ffprobe"), \
            mock.patch.object(
                io,
                "run_process",
                side_effect=[_probe_result(stream), _probe_result(frames)],
            ):
        timing = io._probe_video_frame_timing("sparse.mkv", timeout=30.0)

    assert timing is not None
    assert timing.timestamp_ticks == [0, 40, 80, 1040]
    assert timing.duration_ticks == [40, 40, 960, 40]


def test_exact_ticks_survive_pause_checkpoint_round_trip(tmp_path: Path):
    source = tmp_path / "source.mkv"
    source.write_bytes(b"source")
    frame_dir = tmp_path / "job.frames"
    frame_dir.mkdir()
    cv2.imwrite(str(frame_dir / "frame_000000.png"), np.zeros((4, 4, 3), np.uint8))
    timing = io.VideoFrameTiming(
        timestamp_ticks=[0, 1001, 2002],
        duration_ticks=[1001, 1001, 1001],
        time_base_num=1,
        time_base_den=60000,
        average_fps=60000 / 1001,
        is_vfr=False,
    )
    metadata = timing.checkpoint_metadata(0, 3, 1)
    payload = resume_checkpoint.write_pause_checkpoint(
        tmp_path,
        "job",
        input_path=str(source),
        output_path=str(tmp_path / "out.mkv"),
        config_hash="abc",
        frame_dir=frame_dir,
        next_frame=1,
        total_frames=3,
        width=4,
        height=4,
        fps=timing.average_fps,
        status="paused",
        timing=metadata,
    )
    assert payload["timing"]["time_base_den"] == 60000
    state = resume_checkpoint.load_pause_checkpoint(
        tmp_path,
        "job",
        input_path=str(source),
        output_path=str(tmp_path / "out.mkv"),
        config_hash="abc",
        total_frames=3,
        width=4,
        height=4,
        fps=timing.average_fps,
        timing=timing.checkpoint_metadata(0, 3, 0),
    )
    assert state.next_frame == 1
    assert state.warning == ""
    changed_timing = timing.checkpoint_metadata(0, 3, 0)
    changed_timing["stream_start_ticks"] = 1
    changed = resume_checkpoint.load_pause_checkpoint(
        tmp_path,
        "job",
        input_path=str(source),
        output_path=str(tmp_path / "out.mkv"),
        config_hash="abc",
        total_frames=3,
        width=4,
        height=4,
        fps=timing.average_fps,
        timing=changed_timing,
    )
    assert "exact source timing" in changed.warning


def test_exact_ticks_are_written_and_validated_in_matte_manifest(tmp_path: Path):
    output = tmp_path / "cleaned.mp4"
    ticks = [0, 1001]
    durations = [1001, 1001]
    writer = MaskInterchangeWriter(
        output,
        "png",
        width=4,
        height=4,
        fps=60000 / 1001,
        start_frame=0,
        end_frame=2,
        timestamps=[0.0, 1001 / 60000],
        durations=[1001 / 60000, 1001 / 60000],
        is_vfr=False,
        source_time_base=1 / 60000,
        timestamp_ticks=ticks,
        duration_ticks=durations,
        source_time_base_num=1,
        source_time_base_den=60000,
        source_start_ticks=30000,
        stream_start_ticks=30000,
    )
    writer.write(np.zeros((4, 4), np.uint8))
    writer.write(np.full((4, 4), 255, np.uint8))
    evidence = writer.finalize()
    manifest = json.loads(Path(evidence["manifest"]).read_text(encoding="utf-8"))
    assert manifest["timestamp_ticks"] == ticks
    assert manifest["duration_ticks"] == durations
    reader = MaskInterchangeReader(
        evidence["manifest"],
        width=4,
        height=4,
        start_frame=0,
        end_frame=2,
        timestamps=[0.0, 1001 / 60000],
        durations=[1001 / 60000, 1001 / 60000],
        is_vfr=False,
        source_time_base=1 / 60000,
        timestamp_ticks=ticks,
        duration_ticks=durations,
        source_time_base_num=1,
        source_time_base_den=60000,
        source_start_ticks=30000,
        stream_start_ticks=30000,
        mode="replace",
    )
    reader.close()
    with pytest.raises(ValueError, match="stream_start_ticks"):
        MaskInterchangeReader(
            evidence["manifest"],
            width=4,
            height=4,
            start_frame=0,
            end_frame=2,
            timestamps=[0.0, 1001 / 60000],
            durations=[1001 / 60000, 1001 / 60000],
            is_vfr=False,
            source_time_base=1 / 60000,
            timestamp_ticks=ticks,
            duration_ticks=durations,
            source_time_base_num=1,
            source_time_base_den=60000,
            source_start_ticks=30000,
            stream_start_ticks=30001,
            mode="replace",
        )
    source = tmp_path / "source.mkv"
    source.write_bytes(b"source")
    frozen = freeze_matte(evidence["manifest"], source)
    validated = validate_frozen_matte(
        frozen,
        source_path=source,
        width=4,
        height=4,
        start_frame=0,
        end_frame=2,
        timestamps=[0.0, 1001 / 60000],
        durations=[1001 / 60000, 1001 / 60000],
        is_vfr=False,
        source_time_base=1 / 60000,
        timestamp_ticks=ticks,
        duration_ticks=durations,
        source_time_base_num=1,
        source_time_base_den=60000,
        source_start_ticks=30000,
        stream_start_ticks=30000,
    )
    assert validated["stream_start_ticks"] == 30000
    with pytest.raises(FrozenMatteError, match="edit-list timing"):
        validate_frozen_matte(
            frozen,
            source_path=source,
            width=4,
            height=4,
            start_frame=0,
            end_frame=2,
            timestamps=[0.0, 1001 / 60000],
            durations=[1001 / 60000, 1001 / 60000],
            is_vfr=False,
            source_time_base=1 / 60000,
            timestamp_ticks=ticks,
            duration_ticks=durations,
            source_time_base_num=1,
            source_time_base_den=60000,
            source_start_ticks=30000,
            stream_start_ticks=30001,
        )


def test_fcpxml_uses_exact_rational_boundary(tmp_path: Path):
    path = tmp_path / "exact.fcpxml"
    write_fcpxml(
        str(path),
        "source.mkv",
        "cleaned.mkv",
        fps=60000 / 1001,
        start_s=0.0,
        end_s=0.0,
        start_ticks=30000,
        end_ticks=34004,
        time_base_num=1,
        time_base_den=60000,
    )
    text = path.read_text(encoding="utf-8")
    assert 'offset="1/2s"' in text
    assert 'duration="1001/15000s"' in text


def test_multi_segment_nle_does_not_apply_overall_exact_range_to_segment_one(
    tmp_path: Path,
):
    xml_path = tmp_path / "multi.fcpxml"
    write_fcpxml(
        str(xml_path),
        "source.mkv",
        "cleaned.mkv",
        fps=30.0,
        start_s=10.0,
        end_s=21.0,
        segments=[(10.0, 11.0), (20.0, 21.0)],
        start_ticks=600000,
        end_ticks=1260000,
        time_base_num=1,
        time_base_den=60000,
    )
    root = ElementTree.parse(xml_path).getroot()
    clips = root.findall(".//asset-clip")
    assert [clip.attrib["duration"] for clip in clips] == ["1/1s", "1/1s"]

    edl_path = tmp_path / "multi.edl"
    write_edl(
        str(edl_path),
        "source.mkv",
        "cleaned.mkv",
        fps=30.0,
        start_s=10.0,
        end_s=21.0,
        segments=[(10.0, 11.0), (20.0, 21.0)],
        start_ticks=600000,
        end_ticks=1260000,
        time_base_num=1,
        time_base_den=60000,
    )
    events = [
        line for line in edl_path.read_text(encoding="utf-8").splitlines()
        if line.startswith("001") or line.startswith("002")
    ]
    assert "00:00:10:00 00:00:11:00" in events[0]
