"""RM-152: logical-to-physical layout mirroring for RTL locales."""

from pathlib import Path
import sys
import tkinter as tk
import unittest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from gui import direction
from gui.config import ProcessingConfig, QueueItem
from gui.theme import Theme
from gui.widgets import (
    ModernButton,
    ModernProgressBar,
    ModernSlider,
    QueueItemWidget,
    SegmentedPicker,
)


class PureMirrorTests(unittest.TestCase):
    """The mapping functions, with no Tk involved."""

    def setUp(self):
        Theme.RTL_LAYOUT = True

    def tearDown(self):
        Theme.RTL_LAYOUT = False

    def test_sides_swap(self):
        self.assertEqual(direction.mirror_side("left"), "right")
        self.assertEqual(direction.mirror_side("right"), "left")

    def test_direction_neutral_sides_are_untouched(self):
        for value in ("top", "bottom"):
            self.assertEqual(direction.mirror_side(value), value)

    def test_anchors_swap_west_and_east_but_keep_north_and_south(self):
        self.assertEqual(direction.mirror_anchor("w"), "e")
        self.assertEqual(direction.mirror_anchor("nw"), "ne")
        self.assertEqual(direction.mirror_anchor("se"), "sw")
        for value in ("n", "s", "center"):
            self.assertEqual(direction.mirror_anchor(value), value)

    def test_sticky_masks_mirror_componentwise(self):
        self.assertEqual(direction.mirror_sticky("nw"), "ne")
        self.assertEqual(direction.mirror_sticky("nsew"), "nswe")
        # Already symmetric: mirroring must not lose either edge.
        self.assertEqual(set(direction.mirror_sticky("ew")), {"e", "w"})
        self.assertEqual(direction.mirror_sticky("ns"), "ns")

    def test_justification_swaps_but_center_holds(self):
        self.assertEqual(direction.mirror_justify("left"), "right")
        self.assertEqual(direction.mirror_justify("center"), "center")

    def test_spaced_arrows_flip(self):
        self.assertEqual(
            direction.mirror_text("Filename (A -> Z)"), "Filename (A <- Z)")
        self.assertEqual(
            direction.mirror_text("Filename (Z <- A)"), "Filename (Z -> A)")

    def test_symmetric_and_embedded_arrows_are_left_alone(self):
        # `<->` means "both ways" and has no direction to mirror.
        self.assertEqual(
            direction.mirror_text("source <-> cleaned"), "source <-> cleaned")
        # A bare `->` inside a path or annotation is not an affordance.
        self.assertEqual(
            direction.mirror_text("def f()->str"), "def f()->str")

    def test_everything_is_inert_while_ltr(self):
        Theme.RTL_LAYOUT = False
        self.assertEqual(direction.mirror_side("left"), "left")
        self.assertEqual(direction.mirror_anchor("nw"), "nw")
        self.assertEqual(direction.mirror_sticky("nw"), "nw")
        self.assertEqual(direction.mirror_justify("left"), "left")
        self.assertEqual(direction.mirror_text("A -> Z"), "A -> Z")

    def test_non_string_values_pass_through(self):
        for value in (None, 3, ("left",)):
            self.assertIs(direction.mirror_side(value), value)
            self.assertIs(direction.mirror_text(value), value)

    def test_mirroring_a_mapping_never_mutates_the_caller_dict(self):
        original = {"side": "left", "padx": 4}
        mirrored = direction.mirror_options(original)
        self.assertEqual(original, {"side": "left", "padx": 4})
        self.assertEqual(mirrored["side"], "right")
        self.assertEqual(mirrored["padx"], 4)

    def test_a_mapping_with_nothing_to_mirror_is_returned_as_is(self):
        original = {"padx": 4, "text": "plain"}
        self.assertIs(direction.mirror_options(original), original)

    def test_suspending_the_mirror_restores_the_previous_state(self):
        with direction.no_mirror():
            self.assertEqual(direction.mirror_options({"side": "left"}),
                             {"side": "left"})
            with direction.no_mirror():
                pass
            self.assertEqual(direction.mirror_options({"side": "left"}),
                             {"side": "left"})
        self.assertEqual(
            direction.mirror_options({"side": "left"})["side"], "right")


class LiveTkMirrorTests(unittest.TestCase):
    """The interception point, against a real Tk interpreter."""

    @classmethod
    def setUpClass(cls):
        # One root per class, torn down like the other GUI test files. A
        # stale pointer left by an earlier file's destroyed root is the
        # documented cause of intermittent "Tcl wasn't installed
        # properly" errors, so clear it before asking for a new one.
        try:
            tk._default_root = None
        except Exception:
            pass
        try:
            cls.root = tk.Tk()
        except Exception as exc:  # pragma: no cover - headless CI
            raise unittest.SkipTest(f"Tk display unavailable: {exc}")
        cls.root.geometry("1000x800")
        cls.root.withdraw()

    @classmethod
    def tearDownClass(cls):
        cls.root.destroy()
        try:
            tk._default_root = None
        except Exception:
            pass

    def setUp(self):
        # This module patches a *global* -- tkinter.Misc._options. Register
        # the teardown before patching so a test that raises mid-way can
        # never leak the patch into the rest of the suite.
        self.addCleanup(lambda: setattr(Theme, "RTL_LAYOUT", False))
        self.addCleanup(direction.uninstall_direction_mirror)
        Theme.RTL_LAYOUT = True
        direction.install_direction_mirror()
        self.holder = tk.Frame(self.root)
        self.holder.pack(fill="both", expand=True)
        self.addCleanup(self.holder.destroy)

    def test_pack_side_and_anchor_mirror(self):
        widget = tk.Label(self.holder, text="x")
        widget.pack(side="left", anchor="nw")
        info = widget.pack_info()
        self.assertEqual(str(info["side"]), "right")
        self.assertEqual(str(info["anchor"]), "ne")

    def test_widget_construction_options_mirror(self):
        widget = tk.Label(self.holder, text="x", anchor="w", justify="left")
        self.assertEqual(str(widget.cget("anchor")), "e")
        self.assertEqual(str(widget.cget("justify")), "right")

    def test_configure_after_construction_mirrors_too(self):
        widget = tk.Label(self.holder, text="x")
        widget.configure(anchor="w")
        self.assertEqual(str(widget.cget("anchor")), "e")

    def test_grid_sticky_mirrors(self):
        widget = tk.Label(self.holder, text="x")
        widget.grid(row=0, column=0, sticky="nw")
        self.assertEqual(str(widget.grid_info()["sticky"]), "ne")

    def test_menu_captions_mirror_their_arrows(self):
        menu = tk.Menu(self.root, tearoff=0)
        menu.add_command(label="Filename (A -> Z)")
        self.assertEqual(
            str(menu.entrycget(0, "label")), "Filename (A <- Z)")
        menu.destroy()

    def test_canvas_items_keep_their_coordinate_coupled_anchor(self):
        # A Canvas item's anchor is paired with an x the caller already
        # mirrored by hand; flipping it here would tear the two apart.
        canvas = tk.Canvas(self.holder, width=40, height=20)
        item = canvas.create_text(5, 5, text="A -> Z", anchor="w")
        self.assertEqual(str(canvas.itemcget(item, "anchor")), "w")
        self.assertEqual(str(canvas.itemcget(item, "text")), "A -> Z")
        canvas.itemconfigure(item, anchor="nw")
        self.assertEqual(str(canvas.itemcget(item, "anchor")), "nw")
        canvas.destroy()

    def test_an_ltr_session_is_untouched_even_while_installed(self):
        Theme.RTL_LAYOUT = False
        widget = tk.Label(self.holder, text="A -> Z", anchor="w")
        widget.pack(side="left")
        self.assertEqual(str(widget.cget("anchor")), "w")
        self.assertEqual(str(widget.cget("text")), "A -> Z")
        self.assertEqual(str(widget.pack_info()["side"]), "left")

    def test_install_and_uninstall_are_idempotent_and_reversible(self):
        # setUp already installed it.
        self.assertFalse(direction.install_direction_mirror())
        self.assertTrue(direction.direction_mirror_installed())
        self.assertTrue(direction.uninstall_direction_mirror())
        self.assertFalse(direction.direction_mirror_installed())
        self.assertFalse(direction.uninstall_direction_mirror())

        widget = tk.Label(self.holder, text="x", anchor="w")
        self.assertEqual(str(widget.cget("anchor")), "w")
        direction.install_direction_mirror()

    def test_uninstall_restores_the_exact_original_callables(self):
        direction.uninstall_direction_mirror()
        self.assertIs(tk.Canvas.itemconfig, tk.Canvas.itemconfigure)
        baseline = tk.Misc._options
        direction.install_direction_mirror()
        self.assertIsNot(tk.Misc._options, baseline)
        direction.uninstall_direction_mirror()
        self.assertIs(tk.Misc._options, baseline)
        direction.install_direction_mirror()

    def test_canvas_button_places_icon_after_text_in_rtl(self):
        button = ModernButton(self.holder, text="Open", icon=">", width=120)
        texts = [
            str(button.itemcget(item, "text"))
            for item in button.find_all()
            if button.type(item) == "text"
        ]
        self.assertEqual(texts, ["Open >"])

    def test_slider_value_geometry_and_fill_mirror_in_rtl(self):
        slider = ModernSlider(
            self.holder, from_=0, to=100, value=25, width=120)
        slider.pack(fill="x")
        self.root.update_idletasks()

        self.assertGreater(slider._value_to_x(0), slider._value_to_x(100))
        self.assertAlmostEqual(
            slider._x_to_value(slider._value_to_x(25)), 25, places=0)
        fill_items = [
            item for item in slider.canvas.find_all()
            if slider.canvas.type(item) == "rectangle"
            and str(slider.canvas.itemcget(item, "fill")) == Theme.BLUE_PRIMARY
        ]
        self.assertEqual(len(fill_items), 1)
        coords = slider.canvas.coords(fill_items[0])
        self.assertGreater(min(coords[::2]), slider.THUMB_R)
        self.assertEqual(max(coords[::2]), slider._width - slider.THUMB_R)

    def test_progress_fill_grows_from_the_right_in_rtl(self):
        progress = ModernProgressBar(self.holder, width=120, height=6)
        progress.pack(fill="x")
        progress.set_progress(0.25, animate=False)
        fill_items = [
            item for item in progress.find_all()
            if progress.type(item) == "polygon"
            and str(progress.itemcget(item, "fill")) == progress.fill_color
        ]
        self.assertEqual(len(fill_items), 1)
        coords = progress.coords(fill_items[0])
        self.assertGreater(min(coords[::2]), 0)
        self.assertEqual(max(coords[::2]), progress.bar_width)

    def test_segment_order_mirrors_for_packed_and_grid_layouts(self):
        packed = SegmentedPicker(
            self.holder, [("a", "A"), ("b", "B"), ("c", "C")])
        packed.pack(fill="x")
        grid = SegmentedPicker(
            self.holder,
            [("a", "A"), ("b", "B"), ("c", "C"), ("d", "D")],
            columns=2,
        )
        grid.pack(fill="x")
        self.root.update_idletasks()

        self.assertEqual(
            [str(packed._segments[value].pack_info()["side"])
             for value in ("a", "b", "c")],
            ["right", "right", "right"],
        )
        self.assertEqual(
            int(grid._segments["a"].grid_info()["column"]), 1)
        self.assertEqual(
            int(grid._segments["b"].grid_info()["column"]), 0)
        self.assertEqual(
            int(grid._segments["c"].grid_info()["column"]), 1)
        self.assertEqual(
            int(grid._segments["d"].grid_info()["column"]), 0)

    def test_selected_queue_accent_stripe_moves_to_the_reading_end(self):
        item = QueueItem(
            "id", "input.mp4", "output.mp4", ProcessingConfig())
        queue_item = QueueItemWidget(
            self.holder, item, on_remove=lambda _item_id: None)
        queue_item.pack(fill="x")
        queue_item.set_selected(True)
        self.root.update_idletasks()

        self.assertEqual(
            str(queue_item.accent_stripe.pack_info()["side"]), "right")
        self.assertEqual(
            str(queue_item.accent_stripe.cget("bg")), Theme.BLUE_PRIMARY)


if __name__ == "__main__":
    unittest.main()
