"""UI builder methods extracted from app.py."""

from __future__ import annotations

import logging
from pathlib import Path

try:
    import tkinter as tk
    from tkinter import ttk
except ImportError:
    pass

from gui.theme import (
    Theme, f, text_scale_percent,
)
from gui.config import (
    APP_VERSION, InpaintMode, list_presets,
)
from gui.utils import (
    _build_language_list,
)
from gui.widgets import (
    Tooltip, ModernButton, ModernProgressBar, ModernToggle,
    ModernSlider, SegmentedPicker, DragDropFrame,
)
from backend.a11y import set_accessible_metadata
from backend.i18n import available_catalogs, tr

logger = logging.getLogger(__name__)


class LayoutBuildMixin:
    """UI builder methods extracted from app.py."""

    def _build_ui(self):
        """Build a preview-first workbench with a dedicated inspector and queue."""
        main_container = tk.Frame(self.root, bg=Theme.BG_DARK)
        main_container.pack(fill="both", expand=True)

        # Header
        self._build_header(main_container)
        self._build_command_strip(main_container)

        # Reserve persistent bottom surfaces before the expanding workbench so
        # they cannot be clipped at the minimum 980x720 viewport.
        self._build_footer(main_container)
        self._build_log_panel(main_container)

        # Keep the operational queue and primary action visible while the
        # preview/inspector workbench scrolls independently above it.
        queue_row = tk.Frame(main_container, bg=Theme.BG_DARK)
        queue_row.pack(side="bottom", fill="x")
        self._queue_row = queue_row
        self._build_queue_section(queue_row)

        # A single scroll surface keeps the three-part workbench usable at the
        # 980x720 minimum without compromising the desktop hierarchy.
        content_shell = tk.Frame(main_container, bg=Theme.BG_DARK)
        content_shell.pack(fill="both", expand=True)
        self._content_canvas = tk.Canvas(
            content_shell, bg=Theme.BG_DARK, highlightthickness=0)
        content_scroll = ttk.Scrollbar(
            content_shell, orient="vertical",
            command=self._content_canvas.yview,
            style="Dark.Vertical.TScrollbar")
        self._content_canvas.configure(yscrollcommand=content_scroll.set)
        content_scroll.pack(side="right", fill="y")
        self._content_canvas.pack(side="left", fill="both", expand=True)

        content = tk.Frame(self._content_canvas, bg=Theme.BG_DARK)
        self._content_window = self._content_canvas.create_window(
            (0, 0), window=content, anchor="nw")
        content.bind("<Configure>", self._on_content_configure)
        self._content_canvas.bind("<Configure>", self._on_content_canvas_configure)
        self._content_canvas.bind("<MouseWheel>", self._on_content_mousewheel)
        content.bind("<MouseWheel>", self._on_content_mousewheel)
        content.columnconfigure(
            0, weight=17, minsize=500, uniform="workbench")
        content.columnconfigure(
            1, weight=8, minsize=360, uniform="workbench")
        content.columnconfigure(2, weight=0, minsize=0)
        content.rowconfigure(0, weight=1)
        self._content = content

        # Keep the legacy workflow state widgets alive for status/a11y updates,
        # but do not spend viewport space on an onboarding rail.
        workflow_col = tk.Frame(content, bg=Theme.BG_DARK)
        self._workflow_col = workflow_col
        self._build_workflow_rail(workflow_col)

        # Preview is the primary work surface.
        preview_col = tk.Frame(content, bg=Theme.BG_DARK)
        preview_col.grid(row=0, column=0, sticky="nsew",
                         padx=(0, 0))
        self._preview_col = preview_col
        self._build_preview_section(preview_col)

        # Focused inspector: cleanup profile, region, output, then details.
        settings_col = tk.Frame(content, bg=Theme.BG_DARK)
        settings_col.grid(row=0, column=1, sticky="nsew")
        self._settings_col = settings_col
        self._build_settings_section(settings_col)

    def _build_command_strip(self, parent):
        """Build the high-priority command row from the v3 visual target."""
        strip = self._create_surface(parent)
        strip.pack(fill="x", pady=(1, 0))
        self._command_strip = strip

        inner = tk.Frame(strip, bg=Theme.BG_SECONDARY)
        inner.pack(fill="x", padx=Theme.S_LG, pady=Theme.S_SM)
        self._command_inner = inner

        import_block = tk.Frame(inner, bg=Theme.BG_SECONDARY)
        self.drop_area = DragDropFrame(
            import_block, self._on_files_dropped, height=38, compact=True)
        self.drop_area.pack(fill="x")
        self._import_section = import_block

        mode_block = tk.Frame(inner, bg=Theme.BG_SECONDARY)
        tk.Label(
            mode_block, text=tr("Cleanup profile"), font=f(Theme.F_BODY_SM),
            bg=Theme.BG_SECONDARY, fg=Theme.TEXT_SECONDARY,
        ).pack(anchor="w", pady=(0, Theme.S_XS))
        self._command_mode_combo = ttk.Combobox(
            mode_block, textvariable=self.mode_var,
            values=[mode.value for mode in InpaintMode], state="readonly",
            style="Dark.TCombobox", font=f(Theme.F_BODY_SM), width=18,
        )
        self._command_mode_combo.pack(fill="x")
        self._command_mode_combo.bind(
            "<<ComboboxSelected>>", lambda _event: self._on_mode_picker_changed(
                self.mode_var.get()))

        region_block = tk.Frame(inner, bg=Theme.BG_SECONDARY)
        tk.Label(
            region_block, text=tr("Subtitle region"), font=f(Theme.F_BODY_SM),
            bg=Theme.BG_SECONDARY, fg=Theme.TEXT_SECONDARY,
        ).pack(anchor="w", pady=(0, Theme.S_XS))
        self._command_region_var = tk.StringVar()
        self._command_region_combo = ttk.Combobox(
            region_block, textvariable=self._command_region_var,
            values=(tr("Automatic"), tr("Manual region"), tr("Set region...")),
            state="readonly", style="Dark.TCombobox",
            font=f(Theme.F_BODY_SM), width=18,
        )
        self._command_region_combo.pack(fill="x")
        self._command_region_combo.bind(
            "<<ComboboxSelected>>", self._on_command_region_changed)
        self.skip_detection_var.trace_add("write", self._sync_command_region)
        self._sync_command_region()

        output_block = tk.Frame(inner, bg=Theme.BG_SECONDARY)
        tk.Label(
            output_block, text=tr("Output"), font=f(Theme.F_BODY_SM),
            bg=Theme.BG_SECONDARY, fg=Theme.TEXT_SECONDARY,
        ).pack(anchor="w", pady=(0, Theme.S_XS))
        self._command_output_btn = ModernButton(
            output_block, text=tr("Same as source"), width=176,
            command=self._choose_output_dir, style="secondary", size="md",
        )
        self._command_output_btn.pack(fill="x")

        start_block = tk.Frame(inner, bg=Theme.BG_SECONDARY)
        self.command_start_btn = ModernButton(
            start_block, text=tr("Start cleanup"), width=176,
            command=self._start_processing, style="primary", size="lg", icon=">",
        )
        self.command_start_btn.pack(fill="x", pady=(Theme.S_LG, 0))

        self._command_blocks = (
            import_block, mode_block, region_block, output_block, start_block,
        )
        self._layout_command_strip(compact=False)
        self._divider(strip)

    def _build_header(self, parent):
        """Compact command bar with product identity and live readiness signals."""
        header = self._create_surface(parent, bg=Theme.BG_DARK)
        header.pack(fill="x")

        inner = tk.Frame(header, bg=Theme.BG_DARK)
        inner.pack(fill="x", padx=Theme.S_LG, pady=(Theme.S_SM, Theme.S_SM))

        header_top = tk.Frame(inner, bg=Theme.BG_DARK)
        header_top.pack(fill="x")

        left = tk.Frame(header_top, bg=Theme.BG_DARK)
        left.pack(side="left", fill="y")
        self._header_left = left

        if getattr(self, "_header_icon_photo", None) is not None:
            self._header_icon_label = tk.Label(
                left,
                image=self._header_icon_photo,
                bg=Theme.BG_DARK,
            )
            self._header_icon_label.pack(side="left", padx=(0, Theme.S_SM))

        self._header_title_label = tk.Label(
            left,
            text=tr("Video Subtitle Remover Pro"),
            font=f(Theme.F_DISPLAY, "bold"),
            bg=Theme.BG_DARK,
            fg=Theme.TEXT_PRIMARY,
        )
        self._header_title_label.pack(side="left", anchor="w")
        Tooltip(self._header_title_label, f"Video Subtitle Remover v{APP_VERSION}")
        self._header_version_label = tk.Label(
            left,
            text=f"v{APP_VERSION}",
            font=f(Theme.F_META, "bold"),
            bg=Theme.BG_DARK,
            fg=Theme.TEXT_SECONDARY,
            padx=8,
            pady=3,
        )
        self._header_intro_label = tk.Label(
            left,
            text=tr("Private, local cleanup"),
            font=f(Theme.F_META, "bold"),
            bg=Theme.BG_DARK,
            fg=Theme.GREEN_PRIMARY,
        )

        right = tk.Frame(header_top, bg=Theme.BG_DARK)
        right.pack(side="right", anchor="n")
        self._header_right = right

        settings_btn = ModernButton(
            right, text=tr("Settings"), width=92,
            command=self._focus_settings_panel, style="toolbar",
            size="sm",
        )
        settings_btn.pack(side="left")
        self._header_settings_btn = settings_btn

        help_btn = ModernButton(right, text=tr("Help"), width=80,
                                command=self._show_about, style="toolbar",
                                size="sm")
        help_btn.pack(side="left", padx=(Theme.S_SM, 0))
        self._header_help_btn = help_btn

        chips = tk.Frame(header_top, bg=Theme.BG_DARK)
        chips.pack(side="right", padx=(Theme.S_XL, Theme.S_LG))
        self._header_chips = chips
        self._render_header_chips()
        self._divider(header)

    def _build_workflow_rail(self, parent):
        """Build the three-step workflow rail from the redesign reference."""
        rail = self._create_surface(parent)
        rail.pack(fill="both", expand=True)
        self._workflow_rail = rail

        tk.Label(
            rail, text=tr("Workflow").upper(), font=f(Theme.F_EYEBROW, "bold"),
            bg=Theme.BG_SECONDARY, fg=Theme.TEXT_MUTED,
        ).pack(anchor="w", padx=Theme.S_LG, pady=(Theme.S_LG, Theme.S_MD))

        steps = tk.Frame(rail, bg=Theme.BG_SECONDARY)
        steps.pack(fill="x", padx=Theme.S_MD)
        self._workflow_steps_row = steps
        self._workflow_step_blocks = []
        self._workflow_connectors = []
        step_data = (
            (tr("Import"), tr("Add video or image files")),
            (tr("Configure"), tr("Choose cleanup and region")),
            (tr("Process"), tr("Run and monitor the batch")),
        )
        for idx, (step_label, description) in enumerate(step_data, start=1):
            block = tk.Frame(steps, bg=Theme.BG_SECONDARY)
            block.pack(fill="x")
            badge_lbl = tk.Label(
                block, text=str(idx), width=3, height=1,
                font=f(Theme.F_BODY_SM, "bold"),
                bg=Theme.BG_TERTIARY, fg=Theme.TEXT_MUTED,
                highlightthickness=1, highlightbackground=Theme.BORDER,
            )
            badge_lbl.pack(side="left", anchor="n", pady=2)
            copy = tk.Frame(block, bg=Theme.BG_SECONDARY)
            copy.pack(side="left", fill="x", expand=True,
                      padx=(Theme.S_SM, 0))
            text_lbl = tk.Label(
                copy, text=step_label, font=f(Theme.F_BODY_SM, "bold"),
                bg=Theme.BG_SECONDARY, fg=Theme.TEXT_SECONDARY,
            )
            text_lbl.pack(anchor="w")
            description_lbl = tk.Label(
                copy, text=description, font=f(Theme.F_META),
                bg=Theme.BG_SECONDARY, fg=Theme.TEXT_MUTED,
                wraplength=112, justify="left",
            )
            description_lbl.pack(anchor="w", pady=(2, 0))
            self._workflow_step_blocks.append(block)
            self._workflow_pills.append({
                "frame": block, "badge": badge_lbl, "text": text_lbl,
                "description": description_lbl,
            })
            if idx < len(step_data):
                connector = tk.Frame(
                    steps, bg=Theme.BORDER_STRONG, width=2, height=24)
                connector.pack(anchor="w", padx=14, pady=3)
                connector.pack_propagate(False)
                self._workflow_connectors.append(connector)

        self._header_guidance_panel = self._create_card(rail)
        self._header_guidance_panel.pack(
            side="bottom", fill="x", padx=Theme.S_MD,
            pady=(Theme.S_XL, Theme.S_MD))

        guidance_copy = tk.Frame(self._header_guidance_panel, bg=Theme.BG_CARD)
        guidance_copy.pack(fill="x", padx=Theme.S_MD, pady=Theme.S_MD)
        self._header_guidance_copy = guidance_copy

        self.header_guidance_title = tk.Label(
            guidance_copy,
            text=tr("Build your batch"),
            font=f(Theme.F_TITLE, "bold"),
            bg=Theme.BG_CARD,
            fg=Theme.TEXT_PRIMARY,
        )
        self.header_guidance_title.pack(anchor="w")
        self.header_guidance_body = tk.Label(
            guidance_copy,
            text=tr("Import files or choose a folder to start."),
            font=f(Theme.F_BODY_SM),
            wraplength=124,
            justify="left",
            bg=Theme.BG_CARD,
            fg=Theme.TEXT_MUTED,
        )
        self.header_guidance_body.pack(anchor="w", fill="x", pady=(4, 0))

    def _build_input_section(self, parent):
        """Compact import surface above the primary preview."""
        section = self._create_surface(parent)
        section.pack(fill="x")
        self._import_section = section

        self.drop_area = DragDropFrame(
            section, self._on_files_dropped, height=52, compact=True)
        self.drop_area.pack(fill="x", padx=Theme.S_MD, pady=Theme.S_SM)

    def _build_output_card(self, parent):
        """Build the output destination group inside the inspector."""
        out_surface = self._create_card(parent)
        out_surface.pack(fill="x")
        self._card_header(out_surface, "Output", "Output")

        out_row = tk.Frame(out_surface, bg=Theme.BG_CARD)
        out_row.pack(fill="x", padx=Theme.S_MD, pady=(0, Theme.S_MD))

        label_col = tk.Frame(out_row, bg=Theme.BG_CARD)
        label_col.pack(fill="x")

        self.output_dir_label = tk.Label(label_col, text="", font=f(Theme.F_BODY, "bold"),
                                         bg=Theme.BG_CARD, fg=Theme.TEXT_PRIMARY, anchor="w")
        self.output_dir_label.pack(anchor="w")

        self.output_dir_meta = tk.Label(label_col, text="", font=f(Theme.F_META),
                                        bg=Theme.BG_CARD, fg=Theme.TEXT_MUTED, anchor="w")
        self.output_dir_meta.pack(anchor="w", pady=(2, 0))

        actions = tk.Frame(out_row, bg=Theme.BG_CARD)
        actions.pack(fill="x", pady=(Theme.S_SM, 0))

        choose_btn = ModernButton(actions, text=tr("Choose folder"), width=120,
                                  command=self._choose_output_dir, style="accent",
                                  size="sm")
        choose_btn.pack(side="left")

        reset_btn = ModernButton(actions, text=tr("Reset"), width=76,
                                 command=self._reset_output_dir, style="ghost",
                                 size="sm")
        reset_btn.pack(side="left", padx=(Theme.S_SM, 0))

        self._update_output_label()
        return out_surface

    def _build_inspector_primary_action(self, parent):
        """Keep a compatibility action instance; the command bar owns layout."""
        self._inspector_primary_frame = tk.Frame(
            parent, bg=Theme.BG_SECONDARY)
        self.inspector_start_btn = ModernButton(
            self._inspector_primary_frame,
            text=tr("Start cleanup"),
            width=320,
            height=44,
            command=self._start_processing,
            style="primary",
            size="lg",
            icon=">",
        )
        self._refresh_action_states()
        self._layout_queue_actions(compact=False, dense=False)

    def _build_profile_settings_group(self, settings):
        # ---- Profile card -----------------------------------------------
        profile_panel = self._create_card(settings)
        profile_panel.pack(fill="x")

        self._card_header(profile_panel, "Cleanup profile", "Cleanup profile")

        self.inspector_mode_value = tk.Label(
            profile_panel, textvariable=self.mode_var,
            font=f(Theme.F_BODY, "bold"), bg=Theme.BG_CARD,
            fg=Theme.TEXT_SECONDARY,
        )
        self.inspector_mode_value.pack(
            anchor="w", padx=Theme.S_MD, pady=(0, Theme.S_MD))

        profile_details = tk.Frame(profile_panel, bg=Theme.BG_CARD)
        self._inspector_profile_details = profile_details

        # Preset picker -- one-click recipe application. Built-ins + user-saved.
        preset_row = tk.Frame(profile_details, bg=Theme.BG_CARD)
        preset_row.pack(fill="x", padx=Theme.S_MD, pady=(Theme.S_XS, Theme.S_SM))

        tk.Label(preset_row, text=tr("Preset"), font=f(Theme.F_BODY_SM),
                 bg=Theme.BG_CARD, fg=Theme.TEXT_SECONDARY).pack(anchor="w")

        self.preset_var = tk.StringVar(value="(custom)")
        preset_names = ["(custom)"] + [n for n, _ in list_presets()]
        self.preset_combo = ttk.Combobox(
            preset_row, textvariable=self.preset_var, values=preset_names,
            state="readonly", style="Dark.TCombobox", width=20,
            font=f(Theme.F_BODY_SM),
        )
        self.preset_combo.pack(fill="x", pady=(Theme.S_XS, Theme.S_SM))
        self.preset_combo.bind("<<ComboboxSelected>>", self._on_preset_applied)

        preset_actions = tk.Frame(preset_row, bg=Theme.BG_CARD)
        preset_actions.pack(fill="x")

        save_preset_btn = ModernButton(
            preset_actions, text=tr("Save as..."), command=self._save_preset_dialog,
            size="sm", style="ghost",
        )
        save_preset_btn.pack(side="left")

        export_preset_btn = ModernButton(
            preset_actions, text=tr("Export"), command=self._export_preset_dialog,
            size="sm", style="ghost",
        )
        export_preset_btn.pack(side="left", padx=(Theme.S_XS, 0))
        Tooltip(export_preset_btn, tr("Write the current preset to a shareable JSON file."))

        import_preset_btn = ModernButton(
            preset_actions, text=tr("Import"), command=self._import_preset_dialog,
            size="sm", style="ghost",
        )
        import_preset_btn.pack(side="left", padx=(Theme.S_XS, 0))
        Tooltip(import_preset_btn, tr("Load a preset JSON file into the user library."))

        # Algorithm -- segmented picker replaces the Combobox for speed + clarity
        self._legacy_mode_label = tk.Label(
            profile_panel, text=tr("Algorithm"), font=f(Theme.F_BODY_SM),
            bg=Theme.BG_CARD, fg=Theme.TEXT_SECONDARY,
        )

        self.mode_picker = SegmentedPicker(
            profile_panel,
            options=[(m.value, m.value) for m in InpaintMode],
            value=self.mode_var.get(),
            command=self._on_mode_picker_changed,
            bg=Theme.BG_CARD,
            group_label=tr("Cleanup algorithm"),
            columns=2,
        )

        self.algo_desc = tk.Label(profile_panel, text=self._get_algo_description(),
                                  font=f(Theme.F_BODY_SM), bg=Theme.BG_CARD,
                                  fg=Theme.TEXT_SECONDARY, justify="left", anchor="w",
                                  wraplength=320)
        Tooltip(self.mode_picker, self._get_algo_description())

        row2 = tk.Frame(profile_details, bg=Theme.BG_CARD)
        row2.pack(fill="x", padx=Theme.S_MD, pady=(0, Theme.S_SM))

        tk.Label(row2, text=tr("Compute device"), font=f(Theme.F_BODY_SM),
                 bg=Theme.BG_CARD, fg=Theme.TEXT_SECONDARY).pack(anchor="w")

        self.gpu_combo = ttk.Combobox(row2, textvariable=self.gpu_var, width=20,
                                      values=[tr("Detecting hardware...")],
                                      style="Dark.TCombobox", state="disabled",
                                      font=f(Theme.F_BODY_SM))
        self.gpu_combo.pack(fill="x", pady=(Theme.S_XS, 0))
        self.gpu_combo.bind("<<ComboboxSelected>>", self._on_gpu_changed)
        self._refresh_gpu_selector()

        lang_row = tk.Frame(profile_details, bg=Theme.BG_CARD)
        lang_row.pack(fill="x", padx=Theme.S_MD, pady=(0, Theme.S_MD))

        tk.Label(lang_row, text=tr("Subtitle language"), font=f(Theme.F_BODY_SM),
                 bg=Theme.BG_CARD, fg=Theme.TEXT_SECONDARY).pack(anchor="w")

        # F-5: language list = curated friendly names plus compatible
        # OCR engine codes so users can pick e.g. Thai or Polish without
        # modifying code. Backend status reports engine capacity separately.
        self._lang_display = _build_language_list()
        self._lang_labels = [f"{name} ({code})" for code, name in self._lang_display]
        self._lang_by_label = {label: code for label, (code, _) in
                               zip(self._lang_labels, self._lang_display)}
        self._lang_display_var = tk.StringVar()
        self._set_lang_display(self.lang_var.get())

        language_picker = tk.Frame(lang_row, bg=Theme.BG_CARD)
        language_picker.pack(fill="x", pady=(Theme.S_XS, 0))
        self.lang_combo = ttk.Combobox(language_picker, textvariable=self._lang_display_var,
                                       width=20, values=self._lang_labels,
                                       style="Dark.TCombobox",
                                       state="readonly", font=f(Theme.F_BODY_SM))
        self.lang_combo.pack(side="left", fill="x", expand=True)
        self.lang_combo.bind("<<ComboboxSelected>>", self._on_lang_changed)
        self._lang_detect_btn = ModernButton(
            language_picker, text=tr("Detect"), width=68,
            command=self._probe_language_from_preview,
            style="ghost", size="sm")
        self._lang_detect_btn.pack(side="right", padx=(Theme.S_SM, 0))
        Tooltip(self._lang_detect_btn,
                tr("Auto-detect the subtitle language from a sample frame."))

        return profile_panel, profile_details

    def _build_workflow_settings_group(self, settings):
        # ---- Workflow card ----------------------------------------------
        workflow_panel = self._create_card(settings)
        workflow_panel.pack(fill="x")

        self._card_header(workflow_panel, "Subtitle region", "Subtitle region")

        workflow_details = tk.Frame(workflow_panel, bg=Theme.BG_CARD)
        self._inspector_workflow_details = workflow_details
        checks_frame = tk.Frame(workflow_details, bg=Theme.BG_CARD)
        checks_frame.pack(fill="x", padx=Theme.S_MD, pady=(0, Theme.S_MD))

        self.skip_check = ModernToggle(
            checks_frame,
            text=tr("Use a fixed subtitle region"),
            variable=self.skip_detection_var,
            wraplength=300,
        )
        self.skip_check.pack(anchor="w")
        Tooltip(self.skip_check, tr("Skip repeated detection when you have already set a precise subtitle region."))

        self.lama_check = ModernToggle(
            checks_frame,
            text=tr("Fast LaMa cleanup"),
            variable=self.lama_fast_var,
            wraplength=300,
        )
        self.lama_check.pack(anchor="w", pady=(Theme.S_SM, 0))
        Tooltip(self.lama_check, tr("LaMa fast mode is useful for quick passes and lower-resolution drafts."))

        self.preserve_audio_check = ModernToggle(
            checks_frame,
            text=tr("Preserve source audio"),
            variable=self.preserve_audio_var,
            wraplength=300,
        )
        self.preserve_audio_check.pack(anchor="w", pady=(Theme.S_SM, 0))
        self.ffmpeg_warning_label = tk.Label(
            checks_frame,
            text=tr("Checking FFmpeg availability for audio preservation..."),
            font=f(Theme.F_META),
            bg=Theme.BG_CARD,
            fg=Theme.INFO,
            wraplength=320,
            justify="left",
        )
        self._refresh_ffmpeg_warning()

        # The region reads as a control group, not a card nested in a card.
        region_surface = tk.Frame(workflow_panel, bg=Theme.BG_CARD,
                                  highlightthickness=0)
        self._inspector_region_surface = region_surface
        region_surface.pack(fill="x", padx=Theme.S_MD, pady=(0, Theme.S_MD))

        region_text = tk.Frame(region_surface, bg=Theme.BG_CARD)
        region_text.pack(side="left", fill="x", expand=True,
                         pady=Theme.S_XS)

        self.region_label = tk.Label(region_text, text="", font=f(Theme.F_BODY, "bold"),
                                     bg=Theme.BG_CARD, fg=Theme.TEXT_PRIMARY,
                                     anchor="w")
        self.region_label.pack(anchor="w", pady=(2, 0))

        self.region_meta = tk.Label(region_text, text="", font=f(Theme.F_META),
                                    bg=Theme.BG_CARD, fg=Theme.TEXT_MUTED,
                                    anchor="w")
        self.region_meta.pack(anchor="w", pady=(2, 0))

        region_actions = tk.Frame(region_surface, bg=Theme.BG_CARD)
        region_actions.pack(side="right", pady=Theme.S_XS)

        self.region_btn = ModernButton(region_actions, text=tr("Set region"), width=88,
                                       command=self._open_region_selector_modal, style="accent",
                                       size="sm")
        self.region_btn.pack(side="left")

        self.region_reset_btn = ModernButton(region_actions, text=tr("Reset"), width=64,
                                             command=self._reset_region, style="ghost",
                                             size="sm")
        self.region_reset_btn.pack(side="left", padx=(Theme.S_SM, 0))
        return workflow_panel, workflow_details, region_surface

    def _build_sttn_settings_group(self):
        # STTN Motion card
        sttn_frame = self._create_card(self.adv_panel)
        sttn_frame.pack(fill="x", pady=(Theme.S_MD, Theme.S_SM))
        self._card_header(sttn_frame, "STTN motion", "Motion smoothing")

        self._create_slider(sttn_frame, "Neighbor stride", 5, 30,
                            self.config.sttn_neighbor_stride, "sttn_neighbor_stride",
                            hint="How far apart nearby frames are sampled for motion "
                                 "context. Lower is more thorough but slower.")
        self._create_slider(sttn_frame, "Reference length", 5, 30,
                            self.config.sttn_reference_length, "sttn_reference_length",
                            hint="How many reference frames are kept in view. More "
                                 "references steady long, slow-moving shots.")
        self._create_slider(sttn_frame, "Max load frames", 10, 100,
                            self.config.sttn_max_load_num, "sttn_max_load_num",
                            hint="Most frames held in memory per pass. Lower this if "
                                 "you run out of GPU or system memory.")
        tk.Frame(sttn_frame, bg=Theme.BG_CARD, height=Theme.S_SM).pack(fill="x")

    def _build_detection_settings_group(self):
        # Detection Precision card
        det_frame = self._create_card(self.adv_panel)
        det_frame.pack(fill="x", pady=(0, Theme.S_SM))
        self._card_header(det_frame, "Detection", "Precision tuning")

        self._create_slider(det_frame, "Sensitivity", 10, 90,
                            int(self.config.detection_threshold * 100),
                            "_detection_threshold_pct",
                            hint="Higher catches more text (lower confidence floor). Lower is stricter.")
        engine_row = tk.Frame(det_frame, bg=Theme.BG_CARD)
        engine_row.pack(fill="x", padx=Theme.S_LG, pady=(Theme.S_SM, 0))
        tk.Label(
            engine_row,
            text=tr("OCR engine"),
            font=f(Theme.F_BODY_SM),
            bg=Theme.BG_CARD,
            fg=Theme.TEXT_SECONDARY,
        ).pack(side="left")
        self.ocr_engine_combo = ttk.Combobox(
            engine_row,
            textvariable=self.ocr_engine_var,
            values=tuple(self._ocr_engine_by_label),
            width=24,
            state="readonly",
            style="Dark.TCombobox",
            font=f(Theme.F_BODY_SM),
        )
        self.ocr_engine_combo.pack(side="right")
        self.ocr_engine_combo.bind(
            "<<ComboboxSelected>>", self._on_ocr_engine_changed)
        Tooltip(
            self.ocr_engine_combo,
            tr("Automatic uses the best installed detector. Select an engine "
               "to compare results or reproduce a run."),
        )
        self.language_filter_var = tk.BooleanVar(
            value=self.config.language_mask_filter)
        self.language_filter_toggle = ModernToggle(
            det_frame,
            text=tr("Only remove the selected language"),
            variable=self.language_filter_var,
        )
        self.language_filter_toggle.pack(
            anchor="w", padx=Theme.S_LG, pady=(Theme.S_SM, 0))
        Tooltip(
            self.language_filter_toggle,
            tr("Keep OCR boxes whose recognized script does not match the "
               "subtitle language. Latin-script languages share one family."),
        )
        self._create_slider(det_frame, "Frame skip", 0, 10,
                            self.config.detection_frame_skip, "detection_frame_skip",
                            hint="Reuse the last mask for N frames. At 5, scheduled OCR drops by up to about 83% on stable footage.")
        self._create_slider(det_frame, "Mask dilate", 0, 20,
                            self.config.mask_dilate_px, "mask_dilate_px",
                            hint="Expand detected regions for cleaner fill edges.")
        self.conf_dilate_var = tk.BooleanVar(
            value=self.config.confidence_weighted_dilation)
        conf_dilate_toggle = ModernToggle(
            det_frame,
            text=tr("Adaptive edge padding"),
            variable=self.conf_dilate_var,
        )
        conf_dilate_toggle.pack(anchor="w", padx=Theme.S_LG, pady=(Theme.S_SM, 0))
        Tooltip(conf_dilate_toggle,
                tr("Add more padding around uncertain text detections while "
                   "keeping confident detections tight. Helps catch faint edges."))
        self._create_slider(det_frame, "Mask feather", 0, 15,
                            self.config.mask_feather_px, "mask_feather_px",
                            hint="Soft-blend the removal edge for seamless boundaries.")
        self._create_slider(det_frame, "Colour match ring", 0, 8,
                            self.config.edge_ring_px, "edge_ring_px",
                            hint="Post-inpaint edge-ring colour correction to kill faint seams.")

        self.auto_band_var = tk.BooleanVar(value=self.config.auto_band)
        auto_band_toggle = ModernToggle(
            det_frame,
            text=tr("Auto-detect subtitle band on load"),
            variable=self.auto_band_var,
        )
        auto_band_toggle.pack(anchor="w", padx=Theme.S_LG, pady=(Theme.S_SM, 0))
        Tooltip(auto_band_toggle, tr("Scan the first 30 frames and pin the dominant subtitle band before processing."))

        self.flow_warp_var = tk.BooleanVar(value=self.config.tbe_flow_warp)
        flow_toggle = ModernToggle(
            det_frame,
            text=tr("Motion-aligned background recovery"),
            variable=self.flow_warp_var,
        )
        flow_toggle.pack(anchor="w", padx=Theme.S_LG, pady=(Theme.S_SM, 0))
        Tooltip(flow_toggle, tr("Align nearby frames before rebuilding hidden "
                                "background pixels. Slower, but cleaner on pans and zooms."))

        self.scene_split_var = tk.BooleanVar(value=self.config.tbe_scene_cut_split)
        scene_toggle = ModernToggle(
            det_frame,
            text=tr("Keep scene changes separate"),
            variable=self.scene_split_var,
        )
        scene_toggle.pack(anchor="w", padx=Theme.S_LG, pady=(Theme.S_SM, 0))
        Tooltip(scene_toggle, tr("Keep background recovery from mixing frames "
                                 "across scene changes. Recommended for edited video."))

        self.kalman_var = tk.BooleanVar(value=self.config.kalman_tracking)
        kalman_toggle = ModernToggle(
            det_frame,
            text=tr("Smooth detection between frames"),
            variable=self.kalman_var,
        )
        kalman_toggle.pack(anchor="w", padx=Theme.S_LG, pady=(Theme.S_SM, 0))
        Tooltip(kalman_toggle, tr("Smooths per-frame OCR jitter and fills single-frame misses. Recommended."))

        self.phash_var = tk.BooleanVar(value=self.config.phash_skip_enable)
        phash_toggle = ModernToggle(
            det_frame,
            text=tr("Reuse masks on unchanged frames"),
            variable=self.phash_var,
        )
        phash_toggle.pack(anchor="w", padx=Theme.S_LG, pady=(Theme.S_SM, 0))
        Tooltip(phash_toggle, tr("Skip OCR on frames nearly identical to the last detected one. Static shots can avoid most OCR calls."))
        self._create_slider(
            det_frame, "pHash distance", 0, 16,
            self.config.phash_skip_distance, "phash_skip_distance",
            hint="Higher values reuse masks across more near-identical frames; 4 is conservative.",
        )

        self.colour_tune_var = tk.BooleanVar(value=self.config.colour_tune_enable)
        colour_toggle = ModernToggle(
            det_frame,
            text=tr("Colour-tuned mask expansion"),
            variable=self.colour_tune_var,
        )
        colour_toggle.pack(anchor="w", padx=Theme.S_LG, pady=(Theme.S_SM, 0))
        Tooltip(colour_toggle, tr("Grow the mask to cover serifs / drop shadows that match the subtitle colour. Catches decorative lettering."))

        # RM-24: vertical-text toggle for Japanese tategaki / classical CN.
        self.vertical_text_var = tk.BooleanVar(value=getattr(self.config, "detection_vertical", False))
        vertical_toggle = ModernToggle(
            det_frame,
            text=tr("Vertical text mode (Japanese tategaki / classical Chinese)"),
            variable=self.vertical_text_var,
        )
        vertical_toggle.pack(anchor="w", padx=Theme.S_LG, pady=(Theme.S_SM, Theme.S_MD))
        Tooltip(vertical_toggle, tr("Improve detection for top-to-bottom Japanese "
                                    "or Chinese text. Saved masks stay aligned to the source."))

        tk.Frame(det_frame, bg=Theme.BG_CARD, height=Theme.S_SM).pack(fill="x")

    def _build_output_settings_group(self):
        # Output Quality card
        quality_frame = self._create_card(self.adv_panel)
        quality_frame.pack(fill="x", pady=(0, Theme.S_SM))
        self._card_header(quality_frame, "Output", "Encoding quality")

        self._create_slider(quality_frame, "CRF target", 15, 35,
                            self.config.output_quality, "output_quality",
                            hint="Lower = higher quality. 23 is a balanced default.")

        self.hw_encode_var = tk.BooleanVar(value=self.config.use_hw_encode)
        self.hw_encode_check = ModernToggle(
            quality_frame,
            text=tr("Hardware encoding (NVENC / QSV / AMF) with software fallback"),
            variable=self.hw_encode_var,
        )
        self.hw_encode_check.pack(anchor="w", padx=Theme.S_LG, pady=(Theme.S_SM, 0))
        Tooltip(self.hw_encode_check, tr("Use the GPU's media encoder when "
                                         "available. The app retries with CPU encoding if it fails."))

        self.d3d12_accel_var = tk.BooleanVar(
            value=getattr(self.config, "d3d12_accel", False))
        d3d12_toggle = ModernToggle(
            quality_frame,
            text=tr("Try FFmpeg D3D12 acceleration (experimental)"),
            variable=self.d3d12_accel_var,
        )
        d3d12_toggle.pack(anchor="w", padx=Theme.S_LG, pady=(Theme.S_SM, 0))
        Tooltip(d3d12_toggle, tr(
            "Windows only. Requires FFmpeg 8.1 or newer and uses D3D12 only "
            "after a byte-valid runtime test; NVENC, QSV, AMF, or CPU encoding "
            "remains the automatic fallback."))

        # F-8: output codec selector lives next to the HW-encode toggle.
        codec_row = tk.Frame(quality_frame, bg=Theme.BG_CARD)
        codec_row.pack(fill="x", padx=Theme.S_LG, pady=(Theme.S_SM, 0))
        tk.Label(codec_row, text=tr("Output codec"), font=f(Theme.F_BODY_SM),
                 bg=Theme.BG_CARD, fg=Theme.TEXT_SECONDARY).pack(side="left")
        self.output_codec_var = tk.StringVar(value=getattr(self.config, "output_codec", "h264"))
        codec_combo = ttk.Combobox(
            codec_row, textvariable=self.output_codec_var, width=10,
            values=["h264", "h265", "av1", "vvc"],
            state="readonly", style="Dark.TCombobox", font=f(Theme.F_BODY_SM),
        )
        codec_combo.pack(side="right")
        Tooltip(codec_combo,
                tr("H.264 plays almost everywhere. H.265 and AV1 make smaller "
                   "files but take longer; VVC needs a compatible FFmpeg build."))

        self.adaptive_batch_var = tk.BooleanVar(value=self.config.adaptive_batch)
        adaptive_toggle = ModernToggle(
            quality_frame,
            text=tr("Fit frame batches to GPU memory"),
            variable=self.adaptive_batch_var,
        )
        adaptive_toggle.pack(anchor="w", padx=Theme.S_LG, pady=(Theme.S_SM, 0))
        Tooltip(adaptive_toggle, tr("Adjust how many frames are processed together "
                                    "to fit available GPU memory. Recommended for 4K video."))

        self.temporal_mask_union_var = tk.BooleanVar(
            value=self.config.temporal_mask_union)
        temporal_mask_toggle = ModernToggle(
            quality_frame,
            text=tr("Carry masks across missed detections"),
            variable=self.temporal_mask_union_var,
        )
        temporal_mask_toggle.pack(anchor="w", padx=Theme.S_LG, pady=(Theme.S_SM, 0))
        Tooltip(temporal_mask_toggle, tr(
            "Automatic detection only. Carry recent text masks forward for a "
            "few frames so a single missed detection does not leave a flash."))
        self._create_slider(
            quality_frame, "Mask carry window", 1, 15,
            self.config.temporal_mask_window, "temporal_mask_window",
            hint="Number of recent masks available to repair a missed detection.",
        )

        self.export_srt_var = tk.BooleanVar(value=self.config.export_srt)
        srt_toggle = ModernToggle(
            quality_frame,
            text=tr("Export detected text as .srt sidecar"),
            variable=self.export_srt_var,
        )
        srt_toggle.pack(anchor="w", padx=Theme.S_LG, pady=(Theme.S_SM, 0))
        Tooltip(srt_toggle, tr("Writes an .srt file next to the output using OCR text and timings."))

        self.export_mask_var = tk.BooleanVar(value=self.config.export_mask_video)
        mask_toggle = ModernToggle(
            quality_frame,
            text=tr("Export lossless mask / alpha matte"),
            variable=self.export_mask_var,
        )
        mask_toggle.pack(anchor="w", padx=Theme.S_LG, pady=(Theme.S_SM, 0))
        Tooltip(mask_toggle, tr(
            "Writes an exact grayscale matte and versioned timestamp manifest "
            "alongside the output."))

        matte_format_row = tk.Frame(quality_frame, bg=Theme.BG_CARD)
        matte_format_row.pack(fill="x", padx=Theme.S_LG, pady=(Theme.S_XS, 0))
        tk.Label(
            matte_format_row, text=tr("Matte format"), font=f(Theme.F_META),
            bg=Theme.BG_CARD, fg=Theme.TEXT_MUTED,
        ).pack(side="left")
        self.mask_export_format_var = tk.StringVar(
            value=self.config.mask_export_format)
        matte_format = ttk.Combobox(
            matte_format_row, textvariable=self.mask_export_format_var,
            values=("ffv1", "png"), state="readonly", width=9,
            style="Dark.TCombobox", font=f(Theme.F_BODY_SM),
        )
        matte_format.pack(side="right")
        set_accessible_metadata(
            matte_format, role="combo box", label=tr("Lossless matte format"),
            description=tr("Choose FFV1 video or an ordered PNG sequence."))

        matte_import_row = tk.Frame(quality_frame, bg=Theme.BG_CARD)
        matte_import_row.pack(fill="x", padx=Theme.S_LG, pady=(Theme.S_SM, 0))
        self.mask_import_label_var = tk.StringVar(value=(
            Path(self.config.mask_import_path).name
            if self.config.mask_import_path else tr("No imported matte")
        ))
        tk.Label(
            matte_import_row, textvariable=self.mask_import_label_var,
            font=f(Theme.F_META), bg=Theme.BG_CARD, fg=Theme.TEXT_MUTED,
            anchor="w",
        ).pack(side="left", fill="x", expand=True)
        ModernButton(
            matte_import_row, text=tr("Import matte"), width=112,
            command=self._choose_mask_import_manifest,
            style="ghost", size="sm",
        ).pack(side="right")

        matte_mode_row = tk.Frame(quality_frame, bg=Theme.BG_CARD)
        matte_mode_row.pack(fill="x", padx=Theme.S_LG, pady=(Theme.S_XS, 0))
        ModernButton(
            matte_mode_row, text=tr("Clear import"), width=100,
            command=self._clear_mask_import_manifest,
            style="ghost", size="sm",
        ).pack(side="left")
        self.mask_import_mode_var = tk.StringVar(
            value=self.config.mask_import_mode)
        matte_mode = ttk.Combobox(
            matte_mode_row, textvariable=self.mask_import_mode_var,
            values=("replace", "add", "subtract"), state="readonly",
            width=11, style="Dark.TCombobox", font=f(Theme.F_BODY_SM),
        )
        matte_mode.pack(side="right")
        set_accessible_metadata(
            matte_mode, role="combo box", label=tr("Imported matte mode"),
            description=tr(
                "Replace, add to, or subtract from the native composed mask."))

        self.deinterlace_var = tk.BooleanVar(value=self.config.deinterlace_auto)
        deinterlace_toggle = ModernToggle(
            quality_frame,
            text=tr("Clean up interlaced video automatically"),
            variable=self.deinterlace_var,
        )
        deinterlace_toggle.pack(anchor="w", padx=Theme.S_LG, pady=(Theme.S_SM, 0))
        Tooltip(deinterlace_toggle, tr("Detect and remove comb-like lines from "
                                       "older interlaced video before cleanup."))

        self.keyframe_var = tk.BooleanVar(value=self.config.keyframe_detection)
        keyframe_toggle = ModernToggle(
            quality_frame,
            text=tr("Faster keyframe-only detection"),
            variable=self.keyframe_var,
        )
        keyframe_toggle.pack(anchor="w", padx=Theme.S_LG, pady=(Theme.S_SM, 0))
        Tooltip(keyframe_toggle, tr("Check text on major scene frames and reuse "
                                    "masks between them. Faster on long videos, less precise on motion."))

        self.quality_report_var = tk.BooleanVar(value=self.config.quality_report)
        quality_toggle = ModernToggle(
            quality_frame,
            text=tr("Check output quality after processing"),
            variable=self.quality_report_var,
        )
        quality_toggle.pack(anchor="w", padx=Theme.S_LG, pady=(Theme.S_SM, Theme.S_MD))
        Tooltip(quality_toggle, tr("Sample the finished video for visual damage, "
                                   "remaining text, and flicker, then flag results that need review."))

        translation_frame = self._create_card(self.adv_panel)
        translation_frame.pack(fill="x", pady=(0, Theme.S_SM))
        self._card_header(
            translation_frame, "Localization", "Erase, translate, and re-embed")

        self.translation_enabled_var = tk.BooleanVar(
            value=self.config.translation_enabled)
        translation_toggle = ModernToggle(
            translation_frame,
            text=tr("Re-embed translated subtitles after cleanup"),
            variable=self.translation_enabled_var,
        )
        translation_toggle.pack(
            anchor="w", padx=Theme.S_LG, pady=(Theme.S_XS, 0))
        Tooltip(translation_toggle, tr(
            "Use a provided translated SRT, or send OCR/Whisper cues to a "
            "local translation command. VSR itself does not contact a "
            "translation service; the command you choose controls text handling."))

        self.translation_srt_var = tk.StringVar(
            value=self.config.translation_srt)
        self.translation_srt_label_var = tk.StringVar(value=(
            Path(self.config.translation_srt).name
            if self.config.translation_srt else tr("No translated SRT selected")
        ))
        translated_row = tk.Frame(translation_frame, bg=Theme.BG_CARD)
        translated_row.pack(
            fill="x", padx=Theme.S_LG, pady=(Theme.S_SM, 0))
        tk.Label(
            translated_row,
            textvariable=self.translation_srt_label_var,
            font=f(Theme.F_META),
            bg=Theme.BG_CARD,
            fg=Theme.TEXT_MUTED,
            anchor="w",
        ).pack(side="left", fill="x", expand=True)
        ModernButton(
            translated_row,
            text=tr("Choose translated SRT"),
            width=148,
            command=self._choose_translated_srt,
            style="ghost",
            size="sm",
        ).pack(side="right")
        ModernButton(
            translated_row,
            text=tr("Clear"),
            width=68,
            command=self._clear_translated_srt,
            style="ghost",
            size="sm",
        ).pack(side="right", padx=(0, Theme.S_XS))

        language_row = tk.Frame(translation_frame, bg=Theme.BG_CARD)
        language_row.pack(
            fill="x", padx=Theme.S_LG, pady=(Theme.S_SM, 0))
        tk.Label(
            language_row, text=tr("Language tags"),
            font=f(Theme.F_BODY_SM), bg=Theme.BG_CARD,
            fg=Theme.TEXT_SECONDARY,
        ).pack(side="left")
        language_controls = tk.Frame(language_row, bg=Theme.BG_CARD)
        language_controls.pack(side="right")
        self.translation_source_lang_var = tk.StringVar(
            value=self.config.translation_source_lang or "auto")
        tk.Label(
            language_controls, text=tr("Source"),
            font=f(Theme.F_META), bg=Theme.BG_CARD,
            fg=Theme.TEXT_MUTED,
        ).pack(side="left", padx=(0, Theme.S_XS))
        source_lang_entry = tk.Entry(
            language_controls, width=8, bg=Theme.BG_TERTIARY,
            fg=Theme.TEXT_PRIMARY, font=f(Theme.F_BODY_SM),
            insertbackground=Theme.TEXT_PRIMARY, highlightthickness=1,
            highlightbackground=Theme.BORDER,
            highlightcolor=Theme.BORDER_FOCUS, relief="flat", bd=6,
            textvariable=self.translation_source_lang_var,
        )
        source_lang_entry.pack(side="left")
        self.translation_target_lang_var = tk.StringVar(
            value=self.config.translation_target_lang)
        tk.Label(
            language_controls, text=tr("Target"),
            font=f(Theme.F_META), bg=Theme.BG_CARD,
            fg=Theme.TEXT_MUTED,
        ).pack(side="left", padx=(Theme.S_SM, Theme.S_XS))
        target_lang_entry = tk.Entry(
            language_controls, width=8, bg=Theme.BG_TERTIARY,
            fg=Theme.TEXT_PRIMARY, font=f(Theme.F_BODY_SM),
            insertbackground=Theme.TEXT_PRIMARY, highlightthickness=1,
            highlightbackground=Theme.BORDER,
            highlightcolor=Theme.BORDER_FOCUS, relief="flat", bd=6,
            textvariable=self.translation_target_lang_var,
        )
        target_lang_entry.pack(side="left")
        set_accessible_metadata(
            source_lang_entry, role="text box", label=tr("Source language tag"))
        set_accessible_metadata(
            target_lang_entry, role="text box", label=tr("Target language tag"))

        self.translation_provider_var = tk.StringVar(
            value=self.config.translation_provider or "command")
        provider_row = tk.Frame(translation_frame, bg=Theme.BG_CARD)
        provider_row.pack(
            fill="x", padx=Theme.S_LG, pady=(Theme.S_SM, 0))
        tk.Label(
            provider_row, text=tr("Local provider"),
            font=f(Theme.F_BODY_SM), bg=Theme.BG_CARD,
            fg=Theme.TEXT_SECONDARY,
        ).pack(side="left")
        provider_entry = tk.Entry(
            provider_row, width=16, bg=Theme.BG_TERTIARY,
            fg=Theme.TEXT_PRIMARY, font=f(Theme.F_BODY_SM),
            insertbackground=Theme.TEXT_PRIMARY, highlightthickness=1,
            highlightbackground=Theme.BORDER,
            highlightcolor=Theme.BORDER_FOCUS, relief="flat", bd=6,
            textvariable=self.translation_provider_var,
        )
        provider_entry.pack(side="right")

        self.translation_command_var = tk.StringVar(
            value=self.config.translation_command)
        command_row = tk.Frame(translation_frame, bg=Theme.BG_CARD)
        command_row.pack(
            fill="x", padx=Theme.S_LG, pady=(Theme.S_SM, 0))
        tk.Label(
            command_row, text=tr("Translator command"),
            font=f(Theme.F_BODY_SM), bg=Theme.BG_CARD,
            fg=Theme.TEXT_SECONDARY,
        ).pack(side="left")
        command_entry = tk.Entry(
            command_row, bg=Theme.BG_TERTIARY, fg=Theme.TEXT_PRIMARY,
            font=f(Theme.F_BODY_SM), insertbackground=Theme.TEXT_PRIMARY,
            highlightthickness=1, highlightbackground=Theme.BORDER,
            highlightcolor=Theme.BORDER_FOCUS, relief="flat", bd=6,
            textvariable=self.translation_command_var,
        )
        command_entry.pack(
            side="left", fill="x", expand=True, padx=(Theme.S_SM, Theme.S_SM))
        ModernButton(
            command_row, text=tr("Browse"), width=78,
            command=self._choose_translation_command,
            style="ghost", size="sm",
        ).pack(side="right")

        self.translation_style_var = tk.StringVar(
            value=self.config.translation_style)
        style_row = tk.Frame(translation_frame, bg=Theme.BG_CARD)
        style_row.pack(
            fill="x", padx=Theme.S_LG, pady=(Theme.S_SM, Theme.S_MD))
        tk.Label(
            style_row, text=tr("ASS style override"),
            font=f(Theme.F_BODY_SM), bg=Theme.BG_CARD,
            fg=Theme.TEXT_SECONDARY,
        ).pack(side="left")
        style_entry = tk.Entry(
            style_row, bg=Theme.BG_TERTIARY, fg=Theme.TEXT_PRIMARY,
            font=f(Theme.F_BODY_SM), insertbackground=Theme.TEXT_PRIMARY,
            highlightthickness=1, highlightbackground=Theme.BORDER,
            highlightcolor=Theme.BORDER_FOCUS, relief="flat", bd=6,
            textvariable=self.translation_style_var,
        )
        style_entry.pack(
            side="right", fill="x", expand=True, padx=(Theme.S_SM, 0))
        Tooltip(style_entry, tr(
            "Optional FFmpeg force_style text, for example FontSize=24."))

        return quality_frame

    def _build_range_settings_group(self):
        # Video Range card
        time_frame = self._create_card(self.adv_panel)
        time_frame.pack(fill="x")
        self._card_header(time_frame, "Video range", "Trim (videos only)")

        time_inner = tk.Frame(time_frame, bg=Theme.BG_CARD)
        time_inner.pack(fill="x", padx=Theme.S_LG, pady=(0, Theme.S_MD))

        tk.Label(time_inner, text=tr("Start (s)"), font=f(Theme.F_BODY_SM),
                 bg=Theme.BG_CARD, fg=Theme.TEXT_SECONDARY).pack(side="left")
        self.time_start_entry = tk.Entry(
            time_inner, width=7, bg=Theme.BG_TERTIARY,
            fg=Theme.TEXT_PRIMARY, font=f(Theme.F_BODY_SM),
            insertbackground=Theme.TEXT_PRIMARY,
            highlightthickness=1,
            highlightbackground=Theme.BORDER,
            highlightcolor=Theme.BORDER_FOCUS,
            relief="flat", bd=6)
        self.time_start_entry.insert(0, str(self.config.time_start or 0))
        self.time_start_entry.pack(side="left", padx=(Theme.S_SM, Theme.S_MD))

        tk.Label(time_inner, text=tr("End (s)"), font=f(Theme.F_BODY_SM),
                 bg=Theme.BG_CARD, fg=Theme.TEXT_SECONDARY).pack(side="left")
        self.time_end_entry = tk.Entry(
            time_inner, width=7, bg=Theme.BG_TERTIARY,
            fg=Theme.TEXT_PRIMARY, font=f(Theme.F_BODY_SM),
            insertbackground=Theme.TEXT_PRIMARY,
            highlightthickness=1,
            highlightbackground=Theme.BORDER,
            highlightcolor=Theme.BORDER_FOCUS,
            relief="flat", bd=6)
        self.time_end_entry.insert(0, str(self.config.time_end or 0))
        self.time_end_entry.pack(side="left", padx=(Theme.S_SM, 0))

        tk.Label(time_inner, text=tr("0 uses the full clip"), font=f(Theme.F_META),
                 bg=Theme.BG_CARD, fg=Theme.TEXT_MUTED).pack(side="left", padx=(Theme.S_MD, 0))

    def _build_performance_settings_groups(self):
        # ---- v3.13 GUI-exposed knobs ------------------------------------
        # Editorial: chyron vs subtitle filter + karaoke grouping
        editorial_frame = self._create_card(self.adv_panel)
        editorial_frame.pack(fill="x", pady=(Theme.S_MD, Theme.S_SM))
        self._card_header(editorial_frame, "Editorial", "Filter what gets removed")

        self.remove_subs_var = tk.BooleanVar(value=self.config.remove_subtitles)
        remove_subs_toggle = ModernToggle(
            editorial_frame,
            text=tr("Remove dialogue subtitles (short-lived OCR tracks)"),
            variable=self.remove_subs_var,
        )
        remove_subs_toggle.pack(anchor="w", padx=Theme.S_LG, pady=(Theme.S_XS, 0))
        Tooltip(remove_subs_toggle, tr("Remove short-lived dialogue text. Persistent "
                                       "logos and lower-thirds are controlled separately below."))

        self.remove_chyrons_var = tk.BooleanVar(value=self.config.remove_chyrons)
        remove_chyrons_toggle = ModernToggle(
            editorial_frame,
            text=tr("Remove persistent text (logos, tickers, lower-thirds)"),
            variable=self.remove_chyrons_var,
        )
        remove_chyrons_toggle.pack(anchor="w", padx=Theme.S_LG, pady=(Theme.S_SM, 0))
        Tooltip(remove_chyrons_toggle, tr("Remove text that stays in one place for "
                                          "about three seconds or longer, such as lower-thirds."))

        self.karaoke_grouping_var = tk.BooleanVar(value=self.config.karaoke_grouping)
        karaoke_toggle = ModernToggle(
            editorial_frame,
            text=tr("Karaoke grouping: fuse per-syllable boxes on the same line"),
            variable=self.karaoke_grouping_var,
        )
        karaoke_toggle.pack(anchor="w", padx=Theme.S_LG, pady=(Theme.S_SM, Theme.S_MD))
        Tooltip(karaoke_toggle, tr("Stops karaoke captions leaking original text through the gaps between syllables."))

        # Audio card: loudnorm target + multi-track passthrough
        audio_frame = self._create_card(self.adv_panel)
        audio_frame.pack(fill="x", pady=(0, Theme.S_SM))
        self._card_header(audio_frame, "Audio", "Loudness + tracks")

        self.multi_audio_var = tk.BooleanVar(value=self.config.multi_audio_passthrough)
        multi_audio_toggle = ModernToggle(
            audio_frame,
            text=tr("Pass through every audio stream (Bluray/DVD multi-track)"),
            variable=self.multi_audio_var,
        )
        multi_audio_toggle.pack(anchor="w", padx=Theme.S_LG, pady=(Theme.S_XS, 0))
        Tooltip(multi_audio_toggle, tr("Mux every audio stream from the source. Off keeps only the first track."))

        loudnorm_row = tk.Frame(audio_frame, bg=Theme.BG_CARD)
        loudnorm_row.pack(fill="x", padx=Theme.S_LG, pady=(Theme.S_SM, Theme.S_MD))
        tk.Label(loudnorm_row, text=tr("EBU R128 loudness target"), font=f(Theme.F_BODY_SM),
                 bg=Theme.BG_CARD, fg=Theme.TEXT_SECONDARY).pack(side="left")
        self.loudnorm_var = tk.StringVar(value=str(self.config.loudnorm_target or 0.0))
        loudnorm_entry = tk.Entry(
            loudnorm_row, width=7, bg=Theme.BG_TERTIARY,
            fg=Theme.TEXT_PRIMARY, font=f(Theme.F_BODY_SM),
            insertbackground=Theme.TEXT_PRIMARY,
            highlightthickness=1,
            highlightbackground=Theme.BORDER,
            highlightcolor=Theme.BORDER_FOCUS,
            relief="flat", bd=6, textvariable=self.loudnorm_var)
        loudnorm_entry.pack(side="left", padx=(Theme.S_SM, Theme.S_MD))
        tk.Label(loudnorm_row,
                 text=tr("LUFS. 0 = off. YouTube -14, Apple -16, broadcast -23."),
                 font=f(Theme.F_META),
                 bg=Theme.BG_CARD, fg=Theme.TEXT_MUTED).pack(side="left")

        # Performance card: decode HW accel + prefetch
        perf_frame = self._create_card(self.adv_panel)
        perf_frame.pack(fill="x", pady=(0, Theme.S_SM))
        self._card_header(perf_frame, "Performance", "Decode pipeline")

        accel_row = tk.Frame(perf_frame, bg=Theme.BG_CARD)
        accel_row.pack(fill="x", padx=Theme.S_LG, pady=(Theme.S_XS, 0))
        tk.Label(accel_row, text=tr("Hardware-decode hint"), font=f(Theme.F_BODY_SM),
                 bg=Theme.BG_CARD, fg=Theme.TEXT_SECONDARY).pack(side="left")
        self.decode_accel_var = tk.StringVar(value=self.config.decode_hw_accel or "off")
        accel_combo = ttk.Combobox(
            accel_row, textvariable=self.decode_accel_var, width=10,
            values=["off", "auto", "any", "d3d11", "vaapi", "mfx", "pynv", "nvdec"],
            state="readonly", style="Dark.TCombobox", font=f(Theme.F_BODY_SM),
        )
        accel_combo.pack(side="right")
        Tooltip(accel_combo, tr("Try hardware video decoding for faster input. "
                                "The app falls back to software decoding if needed."))

        rife_row = tk.Frame(perf_frame, bg=Theme.BG_CARD)
        rife_row.pack(fill="x", padx=Theme.S_LG, pady=(Theme.S_SM, 0))
        tk.Label(rife_row, text=tr("Frame interpolation stride"), font=f(Theme.F_BODY_SM),
                 bg=Theme.BG_CARD, fg=Theme.TEXT_SECONDARY).pack(side="left")
        self.rife_stride_var = tk.StringVar(
            value=str(getattr(self.config, "rife_fast_stride", 0) or 0)
        )
        rife_entry = tk.Entry(
            rife_row, width=5, bg=Theme.BG_TERTIARY,
            fg=Theme.TEXT_PRIMARY, font=f(Theme.F_BODY_SM),
            insertbackground=Theme.TEXT_PRIMARY,
            highlightthickness=1,
            highlightbackground=Theme.BORDER,
            highlightcolor=Theme.BORDER_FOCUS,
            relief="flat", bd=6, textvariable=self.rife_stride_var)
        rife_entry.pack(side="right")
        Tooltip(rife_entry, tr("0 cleans every frame. Values above 1 clean fewer "
                               "frames and recreate the gaps when Practical-RIFE is installed. "
                               "Stride 3 can reduce inpaint calls by up to about 67%."))

        self.prefetch_var = tk.BooleanVar(value=self.config.prefetch_decode)
        prefetch_toggle = ModernToggle(
            perf_frame,
            text=tr("Read frames ahead in the background"),
            variable=self.prefetch_var,
        )
        prefetch_toggle.pack(anchor="w", padx=Theme.S_LG, pady=(Theme.S_SM, Theme.S_MD))
        Tooltip(prefetch_toggle, tr("Read upcoming frames in the background so "
                                    "cleanup spends less time waiting on the source file."))

    def _build_accessibility_storage_settings(
        self, quality_frame,
    ):
        # Quality sheet toggle (lives under Output but kept separate so we
        # don't disturb the existing Output card layout)
        self.quality_sheet_var = tk.BooleanVar(value=self.config.quality_report_sheet)
        quality_sheet_toggle = ModernToggle(
            quality_frame,
            text=tr("Quality report sheet (side-by-side PNG comparison)"),
            variable=self.quality_sheet_var,
        )
        quality_sheet_toggle.pack(anchor="w", padx=Theme.S_LG, pady=(0, Theme.S_SM))
        Tooltip(quality_sheet_toggle, tr("Save a side-by-side source/output image "
                                         "for sampled frames. Also enables the numeric quality report."))

        self.json_log_var = tk.BooleanVar(value=getattr(self.config, "json_log_enabled", False))
        json_log_toggle = ModernToggle(
            quality_frame,
            text=tr("Structured JSON log"),
            variable=self.json_log_var,
        )
        json_log_toggle.pack(anchor="w", padx=Theme.S_LG, pady=(0, Theme.S_SM))
        Tooltip(json_log_toggle, tr("Write a structured JSON-lines log alongside the text log. Useful for long batch runs and scripted post-processing."))

        text_scale_row = tk.Frame(quality_frame, bg=Theme.BG_CARD)
        text_scale_row.pack(
            fill="x", padx=Theme.S_LG, pady=(0, Theme.S_SM))
        tk.Label(
            text_scale_row,
            text=tr("Interface text size (restart required)"),
            font=f(Theme.F_BODY_SM),
            bg=Theme.BG_CARD,
            fg=Theme.TEXT_SECONDARY,
        ).pack(side="left")
        self.text_scale_var = tk.StringVar(
            value=f"{text_scale_percent()}%")
        text_scale_combo = ttk.Combobox(
            text_scale_row,
            textvariable=self.text_scale_var,
            values=("100%", "125%", "150%", "175%", "200%"),
            width=7,
            state="readonly",
            style="Dark.TCombobox",
            font=f(Theme.F_BODY_SM),
        )
        text_scale_combo.pack(side="right")
        Tooltip(
            text_scale_combo,
            tr("Scales interface text and dependent controls on the next launch."),
        )

        locale_row = tk.Frame(quality_frame, bg=Theme.BG_CARD)
        locale_row.pack(fill="x", padx=Theme.S_LG, pady=(0, Theme.S_SM))
        tk.Label(
            locale_row,
            text=tr("Interface language (restart required)"),
            font=f(Theme.F_BODY_SM),
            bg=Theme.BG_CARD,
            fg=Theme.TEXT_SECONDARY,
        ).pack(side="left")
        self._locale_display_to_tag = {
            tr("System"): "system",
            tr("English"): "en",
        }
        for locale_tag in available_catalogs():
            self._locale_display_to_tag.setdefault(locale_tag, locale_tag)
        saved_locale = getattr(self.config, "ui_locale", "system")
        locale_display = next(
            (
                display
                for display, locale_tag in self._locale_display_to_tag.items()
                if locale_tag.lower() == str(saved_locale).lower()
            ),
            str(saved_locale),
        )
        if locale_display not in self._locale_display_to_tag:
            self._locale_display_to_tag[locale_display] = str(saved_locale)
        self.locale_var = tk.StringVar(value=locale_display)
        self.locale_combo = ttk.Combobox(
            locale_row,
            textvariable=self.locale_var,
            values=tuple(self._locale_display_to_tag),
            width=14,
            state="readonly",
            style="Dark.TCombobox",
            font=f(Theme.F_BODY_SM),
        )
        self.locale_combo.pack(side="right")
        Tooltip(
            self.locale_combo,
            tr("Uses the system language or a compiled catalog on the next launch."),
        )

        # RM-96: high-contrast theme toggle. Takes effect on next launch
        # because re-skinning every live widget mid-session would force
        # a tree-wide redraw the design tokens were not built for.
        self.high_contrast_var = tk.BooleanVar(value=getattr(self.config, "high_contrast", False))
        hc_toggle = ModernToggle(
            quality_frame,
            text=tr("High-contrast theme (restart required)"),
            variable=self.high_contrast_var,
        )
        hc_toggle.pack(anchor="w", padx=Theme.S_LG, pady=(0, Theme.S_MD))
        Tooltip(hc_toggle, tr("Alternative palette tuned for low-vision users. Persists across sessions."))

        self.update_check_var = tk.BooleanVar(value=getattr(self.config, "update_check", False))
        uc_toggle = ModernToggle(
            quality_frame,
            text=tr("Check for updates on startup"),
            variable=self.update_check_var,
        )
        uc_toggle.pack(anchor="w", padx=Theme.S_LG, pady=(0, Theme.S_MD))
        Tooltip(uc_toggle, tr("When enabled, checks GitHub Releases for a newer version on launch. Off by default; no telemetry."))

        storage_frame = self._create_card(self.adv_panel)
        storage_frame.pack(fill="x", pady=(Theme.S_MD, Theme.S_SM))
        self._card_header(
            storage_frame,
            "Storage",
            "Temporary, mask, checkpoint, and resume files",
        )
        storage_row = tk.Frame(storage_frame, bg=Theme.BG_CARD)
        storage_row.pack(fill="x", padx=Theme.S_LG, pady=(0, Theme.S_MD))
        self.work_dir_var = tk.StringVar(
            value=getattr(self.config, "work_directory", ""))
        self.work_dir_entry = tk.Entry(
            storage_row,
            textvariable=self.work_dir_var,
            bg=Theme.BG_TERTIARY,
            fg=Theme.TEXT_PRIMARY,
            insertbackground=Theme.TEXT_PRIMARY,
            font=f(Theme.F_BODY_SM),
            relief="flat",
            bd=6,
            highlightthickness=1,
            highlightbackground=Theme.BORDER,
            highlightcolor=Theme.BORDER_FOCUS,
        )
        self.work_dir_entry.pack(side="left", fill="x", expand=True)
        Tooltip(
            self.work_dir_entry,
            tr("Leave empty for the system temporary directory. The selected "
               "folder is write-tested before a batch starts."),
        )
        work_browse_btn = ModernButton(
            storage_row,
            text=tr("Choose folder"),
            width=118,
            command=self._choose_work_directory,
            style="accent",
            size="sm",
        )
        work_browse_btn.pack(side="left", padx=(Theme.S_SM, 0))
        work_reset_btn = ModernButton(
            storage_row,
            text=tr("Use system"),
            width=96,
            command=self._reset_work_directory,
            style="ghost",
            size="sm",
        )
        work_reset_btn.pack(side="left", padx=(Theme.S_SM, 0))
        self._work_directory_buttons = (work_browse_btn, work_reset_btn)

    def _build_inspector_summary(self, settings):
        """Build the first-viewport inspector as separator-led disclosure rows."""
        header = tk.Frame(settings, bg=Theme.BG_SECONDARY)
        header.pack(fill="x", pady=(Theme.S_XS, Theme.S_SM))
        tk.Label(
            header,
            text=tr("Cleanup settings"),
            font=f(Theme.F_HEADING, "bold"),
            bg=Theme.BG_SECONDARY,
            fg=Theme.TEXT_PRIMARY,
        ).pack(anchor="w")
        self._inspector_profile_summary_var = tk.StringVar()
        tk.Label(
            header,
            textvariable=self._inspector_profile_summary_var,
            font=f(Theme.F_BODY_SM),
            bg=Theme.BG_SECONDARY,
            fg=Theme.TEXT_MUTED,
        ).pack(anchor="w", pady=(Theme.S_XS, 0))

        self._inspector_encoding_var = tk.StringVar(
            value=str(getattr(self.config, "output_codec", "h264")).upper())
        rows = (
            ("detection", tr("Detection")),
            ("inpainting", tr("Inpainting")),
            ("encoding", tr("Encoding")),
            ("advanced", tr("Advanced")),
        )
        self._inspector_summary_rows = []
        self._inspector_summary_chevrons = {}
        for section_key, title in rows:
            self._divider(settings)
            row = tk.Frame(settings, bg=Theme.BG_SECONDARY)
            row.pack(fill="x")
            button = tk.Button(
                row,
                text=title,
                command=lambda name=section_key: self._open_inspector_details(name),
                font=f(Theme.F_BODY),
                bg=Theme.BG_SECONDARY,
                fg=Theme.TEXT_SECONDARY,
                activebackground=Theme.BG_CARD_HOVER,
                activeforeground=Theme.TEXT_PRIMARY,
                anchor="w",
                relief="flat",
                bd=0,
                highlightthickness=0,
                takefocus=1,
                cursor="hand2",
                padx=0,
                pady=Theme.S_LG,
            )
            button.pack(side="left", fill="x", expand=True)
            chevron = tk.Label(
                row,
                text="v",
                font=f(Theme.F_BODY_SM, "bold"),
                bg=Theme.BG_SECONDARY,
                fg=Theme.TEXT_MUTED,
            )
            chevron.pack(side="right", padx=(Theme.S_SM, 0))
            row.bind(
                "<Button-1>",
                lambda _event, name=section_key: self._open_inspector_details(name),
            )
            chevron.bind(
                "<Button-1>",
                lambda _event, name=section_key: self._open_inspector_details(name),
            )
            self._inspector_summary_rows.append(button)
            self._inspector_summary_chevrons[section_key] = chevron
            if section_key == "advanced":
                self._inspector_advanced_button = button
                self._inspector_advanced_chevron = chevron
        self._refresh_inspector_summary()
        self.mode_var.trace_add("write", self._refresh_inspector_summary)

    def _build_inspector_detail_surfaces(self, settings):
        """Construct the hidden detailed controls behind the flat summary."""
        profile_panel, profile_details = (
            self._build_profile_settings_group(settings)
        )
        workflow_panel, workflow_details, region_surface = (
            self._build_workflow_settings_group(settings)
        )
        output_panel = self._build_output_card(settings)
        self._build_inspector_primary_action(settings)

        profile_panel.pack_forget()
        workflow_panel.pack_forget()
        output_panel.pack_forget()
        self._inspector_profile_panel = profile_panel
        self._inspector_workflow_panel = workflow_panel
        self._inspector_output_panel = output_panel
        self._inspector_detail_panels = (
            (profile_panel, {"fill": "x"}),
            (profile_details, {
                "fill": "x", "padx": Theme.S_MD,
                "pady": (Theme.S_SM, Theme.S_MD),
            }),
            (workflow_panel, {"fill": "x"}),
            (workflow_details, {
                "fill": "x", "before": region_surface,
            }),
            (output_panel, {"fill": "x"}),
        )

    def _build_settings_section(self, parent):
        """Flat inspector grouped by separators and progressive disclosure."""
        section = self._create_surface(parent)
        section.pack(fill="both", expand=True)

        tk.Frame(
            section, bg=Theme.BORDER_SUBTLE, width=1,
        ).pack(side="left", fill="y")
        settings = tk.Frame(section, bg=Theme.BG_SECONDARY)
        settings.pack(
            side="left", fill="both", expand=True,
            padx=Theme.S_LG, pady=(Theme.S_SM, Theme.S_MD),
        )

        self._build_inspector_summary(settings)
        self._build_inspector_detail_surfaces(settings)

        # ---- Advanced toggle --------------------------------------------
        adv_frame = tk.Frame(settings, bg=Theme.BG_SECONDARY)
        self._advanced_compat_frame = adv_frame

        self.adv_visible = False
        self.adv_toggle = ModernButton(adv_frame, text=tr("Advanced settings"), width=188,
                                       command=self._toggle_advanced,
                                       style="ghost", size="sm", icon="+")
        self._sync_inspector_disclosure_state()

        self.adv_panel = tk.Frame(settings, bg=Theme.BG_SECONDARY)

        self._build_sttn_settings_group()
        self._build_detection_settings_group()
        quality_frame = self._build_output_settings_group()
        self._build_range_settings_group()
        self._build_performance_settings_groups()
        self._build_accessibility_storage_settings(quality_frame)

        self.output_codec_var.trace_add("write", self._sync_inspector_encoding)

        self._update_region_label_display()
        self._update_mode_options()

    def _build_preview_section(self, parent):
        """Build the central 16:9 preview and its contextual tools."""
        section = self._create_surface(parent)
        section.pack(fill="both", expand=True)
        self._preview_frame = section

        preview_header = tk.Frame(section, bg=Theme.BG_SECONDARY)
        preview_header.pack(fill="x", padx=Theme.S_MD,
                            pady=(Theme.S_MD, Theme.S_SM))

        preview_text = tk.Frame(preview_header, bg=Theme.BG_SECONDARY)
        preview_text.pack(side="left", fill="x", expand=True)
        self._preview_heading_label = tk.Label(
            preview_text, text=tr("Preview"),
            font=f(Theme.F_HEADING, "bold"),
            bg=Theme.BG_SECONDARY, fg=Theme.TEXT_PRIMARY,
        )
        self._preview_heading_label.pack(anchor="w")
        self.preview_title_label = tk.Label(
            preview_text, text=tr("Preview"),
            font=f(Theme.F_META), bg=Theme.BG_SECONDARY,
            fg=Theme.TEXT_MUTED,
        )
        self.preview_meta_label = tk.Label(
            preview_text,
            text=tr("Select a queue item to inspect its subtitle region."),
            font=f(Theme.F_META), wraplength=520, justify="left",
            bg=Theme.BG_SECONDARY, fg=Theme.TEXT_MUTED,
        )
        # Kept for live status and accessibility; the visual header stays terse.

        self.preview_status_chip = tk.Label(
            preview_header, text=tr("Waiting"),
            font=f(Theme.F_META, "bold"),
            bg=Theme.BG_SECONDARY, fg=Theme.TEXT_MUTED,
        )
        self.preview_ab_btn = ModernButton(
            preview_header, text=tr("Before / after"), width=118,
            command=self._open_ab_scrubber, style="ghost", size="sm",
        )
        Tooltip(
            self.preview_ab_btn,
            tr("Wipe between the source frame and cleaned output."),
        )

        self._preview_tools_btn = ModernButton(
            preview_header,
            text=tr("Preview tools"),
            width=112,
            command=self._open_preview_tools_menu,
            style="ghost",
            size="sm",
        )

        media_surface = tk.Frame(
            section, bg=Theme.BG_CARD, highlightthickness=1,
            highlightbackground=Theme.BORDER_SUBTLE,
        )
        media_surface.pack(
            fill="both", expand=True, padx=Theme.S_MD,
            pady=(0, Theme.S_MD))
        self._preview_media_surface = media_surface

        self._preview_label = tk.Label(
            media_surface, bg=Theme.BG_CARD, text="",
            font=f(Theme.F_META), fg=Theme.TEXT_MUTED,
            compound="bottom", justify="center", cursor="hand2", takefocus=1,
        )
        self._preview_label.pack(fill="both", expand=True, padx=Theme.S_SM,
                                 pady=Theme.S_SM)
        self._preview_photo = None
        self._preview_label.bind("<Double-Button-1>", self._open_preview_zoom)
        self._preview_label.bind(
            "<ButtonPress-1>", self._on_preview_region_press, add="+")
        self._preview_label.bind(
            "<B1-Motion>", self._on_preview_region_drag, add="+")
        self._preview_label.bind(
            "<ButtonRelease-1>", self._on_preview_region_release, add="+")
        self._preview_label.bind(
            "<Button-3>", self._open_preview_tools_menu, add="+")
        self._preview_label.bind(
            "<Shift-F10>", self._open_preview_tools_menu, add="+")
        self._preview_label.bind(
            "<Menu>", self._open_preview_tools_menu, add="+")
        Tooltip(
            self._preview_label,
            tr("Double-click for full size, draw a region, or right-click for preview tools."),
        )

        self.preview_action_hint = tk.Label(
            section,
            text=tr("Add media, then select a queue item to enable preview tools."),
            font=f(Theme.F_META), bg=Theme.BG_SECONDARY,
            fg=Theme.TEXT_MUTED, wraplength=520, justify="left", anchor="w",
        )
        # State is already expressed in the preview title, status, and disabled
        # controls. Keep this label for accessibility/status updates, not layout.

        preview_actions = tk.Frame(section, bg=Theme.BG_SECONDARY)
        self._preview_primary_actions = preview_actions

        self.preview_region_btn = ModernButton(
            preview_actions, text=tr("Set region"), width=104,
            command=self._open_region_selector, style="accent", size="sm",
        )
        self.preview_region_btn.pack(side="left")
        Tooltip(self.preview_region_btn,
                tr("Draw the subtitle region directly on the preview frame."))
        self.preview_mask_btn = ModernButton(
            preview_actions, text=tr("Review mask"), width=108,
            command=self._open_selected_mask_preview,
            style="ghost", size="sm",
        )
        self.preview_mask_btn.pack(side="left", padx=(Theme.S_SM, 0))
        Tooltip(self.preview_mask_btn,
                tr("Detect subtitles and inspect the first-frame mask."))
        self.preview_inpaint_btn = ModernButton(
            preview_actions, text=tr("Test cleanup"), width=112,
            command=self._open_selected_inpaint_preview,
            style="ghost", size="sm",
        )
        self.preview_inpaint_btn.pack(side="left", padx=(Theme.S_SM, 0))
        Tooltip(self.preview_inpaint_btn,
                tr("Run detection and cleanup on one sample frame."))

        self.preview_zoom_btn = ModernButton(
            preview_actions, text=tr("Full size"), width=86,
            command=self._open_preview_zoom, style="ghost", size="sm",
        )
        self.preview_zoom_btn.pack(side="left", padx=(Theme.S_SM, 0))
        self.preview_correction_btn = ModernButton(
            preview_actions, text=tr("Correct mask"), width=112,
            command=self._open_selected_mask_correction,
            style="ghost", size="sm",
        )
        self.preview_correction_btn.pack(side="left", padx=(Theme.S_SM, 0))

        Tooltip(self.preview_zoom_btn, tr("Open the source frame at full size."))
        Tooltip(
            self.preview_correction_btn,
            tr("Paint frame-local mask corrections for a selective rerun."),
        )

        self.preview_mask_tuning = tk.Frame(section, bg=Theme.BG_SECONDARY)
        tuning_label = tk.Frame(
            self.preview_mask_tuning, bg=Theme.BG_SECONDARY)
        tuning_label.pack(side="left")
        tk.Label(
            tuning_label,
            text=tr("Mask dilation"),
            font=f(Theme.F_BODY_SM, "bold"),
            bg=Theme.BG_SECONDARY,
            fg=Theme.TEXT_PRIMARY,
        ).pack(anchor="w")
        tk.Label(
            tuning_label,
            text=tr("Adjust the cached detection without running OCR again."),
            font=f(Theme.F_META),
            bg=Theme.BG_SECONDARY,
            fg=Theme.TEXT_MUTED,
        )
        self.preview_mask_dilate_slider = ModernSlider(
            self.preview_mask_tuning,
            from_=0,
            to=20,
            value=self.config.mask_dilate_px,
            command=self._on_preview_mask_dilate_changed,
            bg=Theme.BG_SECONDARY,
            width=190,
            accessible_label=tr("Preview mask dilation"),
        )
        self.preview_mask_dilate_slider.pack(
            side="left", fill="x", expand=True, padx=(Theme.S_LG, Theme.S_SM))
        self.preview_mask_dilate_value_var = tk.StringVar(
            value=f"{self.config.mask_dilate_px} px")
        tk.Label(
            self.preview_mask_tuning,
            textvariable=self.preview_mask_dilate_value_var,
            font=f(Theme.F_META, "bold"),
            bg=Theme.BG_SECONDARY,
            fg=Theme.TEXT_PRIMARY,
        ).pack(side="right")
        self._set_preview_placeholder(
            "Preview",
            "Select a queue item to inspect its subtitle region before cleanup.",
        )

    def _build_queue_section(self, parent):
        """Build the full-width processing queue and batch controls."""
        section = self._create_surface(parent)
        section.pack(fill="both", expand=True)
        self._divider(section)

        header = tk.Frame(section, bg=Theme.BG_SECONDARY)
        header.pack(fill="x", padx=Theme.S_MD, pady=(Theme.S_MD, Theme.S_XS))

        btn_frame = tk.Frame(header, bg=Theme.BG_SECONDARY)
        btn_frame.pack(side="right")
        self._queue_action_frame = btn_frame

        heading = tk.Frame(header, bg=Theme.BG_SECONDARY)
        heading.pack(side="left", fill="x", expand=True)

        tk.Label(heading, text=tr("Processing queue"),
                 font=f(Theme.F_HEADING, "bold"),
                 bg=Theme.BG_SECONDARY, fg=Theme.TEXT_PRIMARY).pack(anchor="w")
        self._queue_subtitle_label = tk.Label(
            heading,
            text=tr("Review files, progress, and output status in one place."),
            font=f(Theme.F_BODY_SM), bg=Theme.BG_SECONDARY,
            fg=Theme.TEXT_MUTED, wraplength=360, justify="left",
        )

        # One inline summary replaces a cluster of decorative status pills.
        count_cluster = tk.Frame(header, bg=Theme.BG_SECONDARY)
        count_cluster.pack(side="right", anchor="n", padx=(0, Theme.S_MD))
        self._queue_count_cluster = count_cluster

        def _mk_stat_pill(fg=Theme.TEXT_SECONDARY, bg=Theme.BG_SECONDARY):
            pill = tk.Frame(count_cluster, bg=Theme.BG_SECONDARY)
            lbl = tk.Label(pill, text="", font=f(Theme.F_META),
                           bg=Theme.BG_SECONDARY, fg=fg)
            lbl.pack()
            return pill, lbl

        self.queue_total_pill, self.queue_count = _mk_stat_pill(
            fg=Theme.TEXT_SECONDARY)
        self.queue_done_pill, self.queue_done_lbl = _mk_stat_pill(
            fg=Theme.SUCCESS)
        self.queue_err_pill, self.queue_err_lbl = _mk_stat_pill(
            fg=Theme.WARNING)

        self.queue_total_pill.pack(side="left")
        # done/err pills get shown conditionally in _update_queue_display
        self.queue_count.config(text=tr("{count} items").format(count=0))

        # Sort button -- hidden until queue has >= 3 items
        self._sort_btn = ModernButton(
            count_cluster, text=tr("Sort"), width=72,
            command=self._open_sort_menu, style="ghost", size="sm")
        # packed conditionally in _update_queue_display

        # Batch progress -- labels row above the bar
        batch_frame = tk.Frame(section, bg=Theme.BG_SECONDARY)
        batch_frame.pack(fill="x", padx=Theme.S_MD, pady=(Theme.S_SM, 0))
        self._queue_batch_frame = batch_frame

        meta_row = tk.Frame(batch_frame, bg=Theme.BG_SECONDARY)
        meta_row.pack(fill="x")

        self.batch_label = tk.Label(meta_row, text=tr("Ready"),
                                    font=f(Theme.F_META, "bold"),
                                    bg=Theme.BG_SECONDARY, fg=Theme.TEXT_MUTED)
        self.batch_label.pack(side="left")

        self.batch_percent_label = tk.Label(meta_row, text="",
                                            font=f(Theme.F_META, "bold"),
                                            bg=Theme.BG_SECONDARY, fg=Theme.TEXT_SECONDARY)
        self.batch_percent_label.pack(side="right")

        batch_bar_frame = tk.Frame(section, bg=Theme.BG_SECONDARY)
        batch_bar_frame.pack(fill="x", padx=Theme.S_MD, pady=(4, Theme.S_SM))
        self._queue_batch_bar_frame = batch_bar_frame

        self.batch_progress = ModernProgressBar(batch_bar_frame, width=300, height=5,
                                                 fill=Theme.BLUE_PRIMARY)
        self.batch_progress.pack(fill="x")

        # Queue filter input -- appears when there are >5 items
        self._queue_filter_var = tk.StringVar()
        self._queue_filter_frame = tk.Frame(
            section, bg=Theme.BG_TERTIARY,
            highlightthickness=1, highlightbackground=Theme.BORDER)
        # Packed/unpacked dynamically in _update_queue_display
        filter_inner = tk.Frame(self._queue_filter_frame, bg=Theme.BG_TERTIARY)
        filter_inner.pack(fill="x", padx=Theme.S_SM, pady=2)

        tk.Label(filter_inner, text=tr("Filter"), font=f(Theme.F_META, "bold"),
                 bg=Theme.BG_TERTIARY, fg=Theme.TEXT_MUTED).pack(
                     side="left", padx=(Theme.S_SM, Theme.S_SM))
        self._queue_filter_entry = tk.Entry(
            filter_inner, textvariable=self._queue_filter_var,
            bg=Theme.BG_TERTIARY, fg=Theme.TEXT_PRIMARY,
            insertbackground=Theme.TEXT_PRIMARY,
            font=f(Theme.F_BODY_SM), relief="flat", bd=6,
            highlightthickness=0)
        self._queue_filter_entry.pack(side="left", fill="x", expand=True)
        self._queue_filter_entry.bind(
            "<FocusIn>",
            lambda e: self._queue_filter_frame.config(highlightbackground=Theme.BORDER_FOCUS),
        )
        self._queue_filter_entry.bind(
            "<FocusOut>",
            lambda e: self._queue_filter_frame.config(highlightbackground=Theme.BORDER),
        )
        self._queue_filter_clear = ModernButton(
            filter_inner, text=tr("Clear"), width=68,
            command=lambda: self._queue_filter_var.set(""),
            style="ghost", size="sm")
        self._queue_filter_clear.pack(side="right", padx=(Theme.S_SM, 0))
        self._queue_filter_var.trace_add(
            "write", lambda *_: self._apply_queue_filter())

        self._queue_container = tk.Frame(section, bg=Theme.BG_SECONDARY)
        self._queue_container.pack(fill="both", expand=True,
                                   padx=Theme.S_MD, pady=(0, Theme.S_SM))
        queue_container = self._queue_container

        table_header = tk.Frame(
            section, bg=Theme.BG_TERTIARY,
            highlightthickness=0,
        )
        table_header.pack(
            fill="x", padx=Theme.S_MD, pady=(0, 1),
            before=self._queue_container,
        )
        table_header.columnconfigure(0, weight=5, uniform="queue_columns")
        table_header.columnconfigure(1, weight=3, uniform="queue_columns")
        table_header.columnconfigure(2, weight=2, uniform="queue_columns")
        for column, label in enumerate(("File name", "Details", "Status")):
            tk.Label(
                table_header, text=tr(label), font=f(Theme.F_META),
                bg=Theme.BG_TERTIARY, fg=Theme.TEXT_MUTED,
                anchor="w",
            ).grid(
                row=0, column=column, sticky="ew",
                padx=Theme.S_MD, pady=Theme.S_SM,
            )
        self._queue_table_header = table_header

        self.queue_canvas = tk.Canvas(queue_container, bg=Theme.BG_SECONDARY,
                                     highlightthickness=0, height=88)
        scrollbar = ttk.Scrollbar(queue_container, orient="vertical",
                                 command=self.queue_canvas.yview,
                                 style="Dark.Vertical.TScrollbar")

        self.queue_frame = tk.Frame(self.queue_canvas, bg=Theme.BG_SECONDARY)

        self.queue_canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.queue_canvas.pack(side="left", fill="both", expand=True)

        self.queue_window = self.queue_canvas.create_window((0, 0), window=self.queue_frame,
                                                            anchor="nw")

        self.queue_frame.bind("<Configure>", self._on_queue_configure)
        self.queue_canvas.bind("<Configure>", self._on_canvas_configure)

        # Mousewheel scrolling
        self.queue_canvas.bind("<Enter>", self._bind_mousewheel)
        self.queue_canvas.bind("<Leave>", self._unbind_mousewheel)

        self._build_queue_empty_state()

        # One quiet command cluster shares the queue header.
        self.queue_add_btn = ModernButton(
            btn_frame,
            text=tr("Add media"),
            width=104,
            command=self.drop_area._open_file_dialog,
            style="secondary",
            size="sm",
        )
        self.queue_add_btn.pack(side="right")

        self.queue_remove_btn = ModernButton(
            btn_frame,
            text=tr("Remove"),
            width=92,
            command=self._remove_selected_queue_item,
            style="toolbar",
            size="sm",
        )
        self.queue_clear_completed_btn = ModernButton(
            btn_frame,
            text=tr("Clear completed"),
            width=132,
            command=self._clear_completed_queue_items,
            style="toolbar",
            size="sm",
        )
        self.queue_move_up_btn = ModernButton(
            btn_frame,
            text="^",
            width=40,
            command=lambda: self._move_selected_queue_item(-1),
            style="toolbar",
            size="sm",
        )
        self.queue_move_down_btn = ModernButton(
            btn_frame,
            text="v",
            width=40,
            command=lambda: self._move_selected_queue_item(1),
            style="toolbar",
            size="sm",
        )
        self._queue_action_separators = tuple(
            tk.Frame(btn_frame, bg=Theme.BORDER_SUBTLE, width=1)
            for _index in range(3)
        )
        Tooltip(self.queue_remove_btn, tr("Remove the selected item from the queue."))
        Tooltip(
            self.queue_clear_completed_btn,
            tr("Remove completed items without deleting their output files."),
        )
        Tooltip(self.queue_move_up_btn, tr("Move the selected item up."))
        Tooltip(self.queue_move_down_btn, tr("Move the selected item down."))

        self.start_btn = ModernButton(btn_frame, text=tr("Start batch"), width=180,
                                     height=44,
                                     command=self._start_processing,
                                     style="primary", size="lg", icon=">")
        self.start_btn.pack(side="right")

        self.open_output_btn = ModernButton(btn_frame, text=tr("Open output"), width=132,
                                            command=self._open_output_folder,
                                            style="ghost", size="sm", icon="^")
        self.open_output_btn.pack(side="right", padx=(0, Theme.S_SM))

        self.retry_btn = ModernButton(btn_frame, text=tr("Retry failed"), width=124,
                                      command=self._retry_failed,
                                      style="ghost", size="sm")
        self.retry_btn.pack(side="left")

        self.repeat_btn = ModernButton(btn_frame, text=tr("Repeat last"), width=120,
                                      command=self._repeat_last_job,
                                      style="ghost", size="sm")
        self.repeat_btn.pack(side="left", padx=(Theme.S_SM, 0))

        self.clear_btn = ModernButton(btn_frame, text=tr("Clear queue"), width=120,
                                     command=self._clear_queue,
                                     style="ghost", size="sm")
        self.clear_btn.pack(side="left", padx=(Theme.S_SM, 0))

        self._queue_more_btn = ModernButton(
            btn_frame, text=tr("Queue actions"), width=124,
            command=self._open_queue_actions_menu,
            style="ghost", size="sm", icon="...",
        )

        self._refresh_action_states()
        self._layout_queue_actions(
            compact=self._layout_mode == "stacked",
            dense=self._text_scale_percent >= 150,
        )

    def _build_queue_empty_state(self):
        """Render an empty queue as one quiet table row."""
        self.empty_container = tk.Frame(self.queue_frame, bg=Theme.BG_SECONDARY)
        self.empty_container.pack(fill="x")
        tk.Label(
            self.empty_container, text=tr("No media queued"),
            font=f(Theme.F_BODY_SM), bg=Theme.BG_SECONDARY,
            fg=Theme.TEXT_MUTED,
        ).pack(anchor="w", padx=Theme.S_MD, pady=Theme.S_MD)

    def _build_footer(self, parent):
        """Footer status bar with a colored dot + message and a right-side hint."""
        footer = tk.Frame(parent, bg=Theme.BG_DARK)
        footer.pack(side="bottom", fill="x")
        self._footer = footer
        self._divider(footer)

        left = tk.Frame(footer, bg=Theme.BG_DARK)
        left.pack(side="left", padx=Theme.S_LG, pady=Theme.S_XS)
        self._footer_left = left

        # Keep the status canvas for color/state updates and accessibility,
        # but the footer itself follows the reference's single quiet label.
        self.status_dot = tk.Canvas(left, width=10, height=10, bg=Theme.BG_DARK,
                                    highlightthickness=0)
        self._status_dot_item = self.status_dot.create_oval(
            1, 1, 9, 9, fill=Theme.TEXT_SECONDARY, outline="")

        self.status_label = tk.Label(left, text=tr("Ready"),
                                     font=f(Theme.F_BODY_SM),
                                     bg=Theme.BG_DARK, fg=Theme.TEXT_SECONDARY, anchor="w")
        self.status_label.pack(side="left")
        try:
            from backend.a11y import set_accessible_metadata
            set_accessible_metadata(
                self.status_label,
                role="status",
                label=tr("Application status"),
                state="neutral",
                value=tr("Ready"),
            )
        except Exception:
            pass

        self.status_hint = tk.Label(
            footer,
            text=tr("Add files, review a sample frame, then start."),
            font=f(Theme.F_META),
            bg=Theme.BG_DARK,
            fg=Theme.TEXT_MUTED,
        )
        self._footer_activity_btn = ModernButton(
            footer, text=tr("Activity"), width=92,
            command=self._toggle_log_panel,
            style="ghost", size="sm",
        )
