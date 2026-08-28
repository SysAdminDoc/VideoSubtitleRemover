"""RM-341: a way back from a bad configuration or a skipped walkthrough.

There was no reset across 140 settings, no way to see the welcome flow
again once it was dismissed, and no undo for the queue operations that
discard work. A beginner who dismissed the walkthrough and then broke their
settings had nothing to go back to.
"""

from __future__ import annotations

import ast
import unittest
from dataclasses import fields
from pathlib import Path

from gui.config import (
    RESET_PRESERVED_FIELDS,
    ProcessingConfig,
    ProcessingStatus,
    QueueItem,
    processing_defaults,
    reset_summary,
)

ROOT = Path(__file__).resolve().parent.parent


class ResetToDefaultsTests(unittest.TestCase):
    def test_reset_equals_the_dataclass_defaults_for_every_declared_field(self):
        """The acceptance, stated literally: every field, not a chosen few."""
        fresh = ProcessingConfig()
        dirtied = ProcessingConfig()
        for field in fields(dirtied):
            value = getattr(dirtied, field.name)
            if isinstance(value, bool):
                setattr(dirtied, field.name, not value)
            elif isinstance(value, int):
                setattr(dirtied, field.name, value + 7)
            elif isinstance(value, float):
                setattr(dirtied, field.name, min(0.99, value + 0.11))
            elif isinstance(value, str) and value:
                setattr(dirtied, field.name, value + "-changed")

        reset = processing_defaults(dirtied)
        mismatches = []
        for field in fields(fresh):
            if field.name in RESET_PRESERVED_FIELDS:
                continue
            if getattr(reset, field.name) != getattr(fresh, field.name):
                mismatches.append(field.name)
        self.assertEqual(mismatches, [], mismatches)

    def test_the_preserved_fields_survive_a_reset(self):
        config = ProcessingConfig()
        config.high_contrast = True
        config.text_scale_percent = 150
        config.onboarding_seen = True
        config.subtitle_area = (4, 5, 60, 70)
        config.mask_dilate_px = 19

        reset = processing_defaults(config)
        self.assertTrue(reset.high_contrast)
        self.assertEqual(reset.text_scale_percent, 150)
        self.assertTrue(reset.onboarding_seen)
        self.assertEqual(tuple(reset.subtitle_area), (4, 5, 60, 70))
        # And the thing the user actually wanted reset is reset.
        self.assertEqual(reset.mask_dilate_px, ProcessingConfig().mask_dilate_px)

    def test_the_summary_names_what_will_change_and_nothing_else(self):
        config = ProcessingConfig()
        config.mask_dilate_px = 19
        config.high_contrast = True

        changed = reset_summary(config)
        self.assertIn("mask_dilate_px", changed)
        self.assertNotIn("high_contrast", changed)
        for name in RESET_PRESERVED_FIELDS:
            self.assertNotIn(name, changed)

    def test_a_config_already_at_defaults_reports_nothing_to_change(self):
        self.assertEqual(reset_summary(ProcessingConfig()), [])

    def test_the_control_asks_before_acting(self):
        source = (ROOT / "gui" / "settings_controller.py").read_text(
            encoding="utf-8")
        tree = ast.parse(source)
        target = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "_reset_processing_settings"
        )
        body = ast.dump(target)
        self.assertIn("show_confirm", body)
        # It names what it will change before acting.
        self.assertIn("reset_summary", body)


class WalkthroughReplayTests(unittest.TestCase):
    def test_the_welcome_flow_can_be_replayed(self):
        from gui.onboarding import OnboardingMixin

        self.assertTrue(hasattr(OnboardingMixin, "replay_onboarding"))

    def test_the_replay_clears_the_in_session_guard(self):
        """`_onboarding_scheduled` would otherwise block a second showing."""
        from gui.onboarding import OnboardingMixin

        class _Host(OnboardingMixin):
            def __init__(self):
                self._onboarding_scheduled = True
                self.shown = 0

            def _show_onboarding(self):
                self.shown += 1

        host = _Host()
        host.replay_onboarding()
        self.assertEqual(host.shown, 1)
        self.assertFalse(host._onboarding_scheduled)

    def test_help_offers_the_control(self):
        source = (ROOT / "gui" / "support_controller.py").read_text(
            encoding="utf-8")
        self.assertIn("replay_onboarding", source)
        self.assertIn("Show walkthrough", source)


class _QueueHost:
    """The queue slice of the app, without tkinter."""

    def __init__(self, items):
        import threading

        self.queue = list(items)
        self.queue_lock = threading.RLock()
        self.is_processing = False
        self._selected_queue_item_id = None
        self.statuses = []
        self.queue_undo_btn = None

    # Methods the undo touches, stubbed to observable no-ops.
    def _has_active_processing_thread(self):
        return False

    def _update_queue_display(self):
        pass

    def _update_status(self, text, tone="info", **_kwargs):
        self.statuses.append((str(text), tone))


def _host(count=3):
    from gui.app import VideoSubtitleRemoverApp

    items = [
        QueueItem(id=f"item-{index}", file_path=f"{index}.mp4",
                  output_path=f"{index}_clean.mp4",
                  config=ProcessingConfig())
        for index in range(count)
    ]
    host = _QueueHost(items)
    for name in ("_record_queue_undo", "_refresh_queue_undo_control",
                 "_undo_queue_removal", "_try_dequeue_queue_item",
                 "_remove_from_queue", "_clear_completed_queue_items"):
        setattr(type(host), name, getattr(VideoSubtitleRemoverApp, name))
    return host


class QueueUndoTests(unittest.TestCase):
    def test_a_removed_item_comes_back_in_its_original_position(self):
        host = _host(3)
        item = host.queue[1]
        removed, result, position = host._try_dequeue_queue_item(item.id)
        self.assertEqual(result, "removed")
        self.assertEqual(position, 1)
        host._record_queue_undo([(removed, position)], "removal")
        self.assertEqual([i.id for i in host.queue], ["item-0", "item-2"])

        host._undo_queue_removal()
        self.assertEqual(
            [i.id for i in host.queue], ["item-0", "item-1", "item-2"])

    def test_undo_is_a_single_step(self):
        host = _host(3)
        for index in (0, 1):
            item = host.queue[0]
            removed, _result, position = host._try_dequeue_queue_item(item.id)
            host._record_queue_undo([(removed, position)], "removal")
        host._undo_queue_removal()
        # The most recent removal comes back; the earlier one does not.
        self.assertEqual(len(host.queue), 2)
        host._undo_queue_removal()
        self.assertEqual(len(host.queue), 2)
        self.assertTrue(
            any("nothing to undo" in text.lower() for text, _tone
                in host.statuses))

    def test_clearing_completed_items_is_undoable_as_one_step(self):
        host = _host(4)
        for index in (1, 3):
            host.queue[index].status = ProcessingStatus.COMPLETE
        host._clear_completed_queue_items()
        self.assertEqual([i.id for i in host.queue], ["item-0", "item-2"])

        host._undo_queue_removal()
        self.assertEqual(
            [i.id for i in host.queue],
            ["item-0", "item-1", "item-2", "item-3"],
        )

    def test_undo_is_refused_while_a_batch_is_running(self):
        host = _host(2)
        removed, _result, position = host._try_dequeue_queue_item("item-0")
        host._record_queue_undo([(removed, position)], "removal")
        host.is_processing = True
        host._undo_queue_removal()
        self.assertEqual(len(host.queue), 1)
        self.assertTrue(
            any("finish" in text.lower() for text, _tone in host.statuses))

    def test_undoing_an_item_that_is_already_back_does_not_duplicate_it(self):
        host = _host(2)
        removed, _result, position = host._try_dequeue_queue_item("item-0")
        host._record_queue_undo([(removed, position)], "removal")
        with host.queue_lock:
            host.queue.insert(0, removed)
        host._undo_queue_removal()
        self.assertEqual([i.id for i in host.queue], ["item-0", "item-1"])

    def test_a_position_beyond_the_current_queue_is_clamped(self):
        host = _host(3)
        removed, _result, _position = host._try_dequeue_queue_item("item-2")
        host._record_queue_undo([(removed, 99)], "removal")
        host._undo_queue_removal()
        self.assertIn("item-2", [i.id for i in host.queue])


if __name__ == "__main__":
    unittest.main()
