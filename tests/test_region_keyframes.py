import json
import unittest

import numpy as np

from backend.config import ProcessingConfig, normalize_processing_config
from backend.config_schema import (
    CONFIG_SCHEMA_VERSION,
    GUI_SETTINGS_FORMAT,
    backend_config_cli_args,
    migrate_gui_settings,
    parse_cli_assignments,
    serialize_backend_config,
)
from backend.processor import SubtitleRemover
from backend.region_keyframes import (
    normalize_region_keyframe_tracks,
    region_shapes_at,
)
from gui.config import ProcessingConfig as GuiProcessingConfig


RECT_TRACK = {
    "keyframes": [
        {"time": 2.0, "rect": [20, 30, 60, 50]},
        {"time": 0.0, "rect": [0, 10, 40, 30]},
    ],
}

POLYGON_TRACK = {
    "keyframes": [
        {"time": 1.0, "polygon": [2, 2, 10, 2, 6, 10]},
        {"time": 3.0, "polygon": [4, 4, 12, 4, 8, 12]},
    ],
}


class RegionKeyframeTests(unittest.TestCase):
    def test_rect_track_sorts_and_interpolates_only_inside_span(self):
        tracks = normalize_region_keyframe_tracks([RECT_TRACK])
        self.assertIsNotNone(tracks)
        self.assertEqual(tracks[0]["start"], 0.0)
        self.assertEqual(tracks[0]["end"], 2.0)
        self.assertEqual(region_shapes_at(tracks, -1.0), [{"rect": [0, 10, 40, 30]}])
        self.assertEqual(region_shapes_at(tracks, 1.0), [{"rect": [10, 20, 50, 40]}])
        self.assertEqual(region_shapes_at(tracks, 2.01), [])

    def test_polygon_track_interpolates_matching_vertices(self):
        tracks = normalize_region_keyframe_tracks([POLYGON_TRACK])
        self.assertEqual(
            region_shapes_at(tracks, 2.0),
            [{"polygon": [3, 3, 11, 3, 7, 11]}],
        )

    def test_polygon_track_rejects_topology_changes(self):
        malformed = {
            "keyframes": [
                {"time": 0, "polygon": [0, 0, 10, 0, 5, 5]},
                {"time": 1, "polygon": [0, 0, 10, 0, 10, 10, 0, 10]},
            ]
        }
        self.assertIsNone(normalize_region_keyframe_tracks([malformed]))

    def test_backend_fills_interpolated_polygon_not_its_bounds(self):
        remover = SubtitleRemover.__new__(SubtitleRemover)
        remover.config = normalize_processing_config(ProcessingConfig(
            subtitle_region_keyframes=[POLYGON_TRACK],
        ))
        shapes = remover._fixed_region_shapes(2.0)
        mask = np.zeros((20, 20), dtype=np.uint8)
        result = remover._apply_polygon_region_shapes(mask, shapes)
        self.assertEqual(int(result[7, 7]), 255)
        self.assertEqual(int(result[10, 3]), 0)

    def test_gui_and_backend_round_trip_tracks(self):
        gui_config = GuiProcessingConfig.from_dict({
            "subtitle_region_keyframes": [RECT_TRACK],
            "vsr_settings_format": 4,
        })
        restored = GuiProcessingConfig.from_dict(gui_config.to_dict())
        self.assertEqual(
            restored.subtitle_region_keyframes,
            gui_config.subtitle_region_keyframes,
        )
        backend = normalize_processing_config(ProcessingConfig(
            subtitle_region_keyframes=restored.subtitle_region_keyframes,
        ))
        self.assertEqual(
            serialize_backend_config(backend)["subtitle_region_keyframes"],
            restored.to_dict()["subtitle_region_keyframes"],
        )

    def test_cli_override_round_trip_tracks(self):
        config = normalize_processing_config(ProcessingConfig(
            subtitle_region_keyframes=[RECT_TRACK],
        ))
        args = backend_config_cli_args(config)
        assignments = [args[index + 1] for index, value in enumerate(args[:-1])
                       if value == "--set"]
        payload = parse_cli_assignments(assignments)
        self.assertEqual(
            payload["subtitle_region_keyframes"],
            serialize_backend_config(config)["subtitle_region_keyframes"],
        )
        self.assertIn(str(CONFIG_SCHEMA_VERSION), args)

    def test_format_four_migration_keeps_timed_regions(self):
        legacy = {
            "vsr_settings_format": 4,
            "subtitle_region_spans": [
                {"rect": [1, 2, 11, 12], "start": 3.0, "end": 4.0},
            ],
        }
        migrated = migrate_gui_settings(json.loads(json.dumps(legacy)))
        self.assertEqual(migrated["vsr_settings_format"], GUI_SETTINGS_FORMAT)
        self.assertEqual(
            migrated["subtitle_region_spans"], legacy["subtitle_region_spans"])


if __name__ == "__main__":
    unittest.main()
