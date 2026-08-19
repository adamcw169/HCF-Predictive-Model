"""Tab 1 - Extract steady-state anchors from a draw run and calibrate on them.

Three steps, in the order the work has to happen: a block cannot be matched to
an analytic estimate before it exists, and nothing can be calibrated before
both sides are present.

    1. Load & extract   - the run in, steady-state blocks out
    2. Match analytic   - the estimator's prediction for each block
    3. Fit & inspect    - the correction between the two

They are separate pages rather than one long scroll. Everything in v1.1 was
visible at once, which meant the four numbers an operator acts on - block
counts, the shape chosen per channel, the fit scatter - arrived buried in the
evidence behind them. The evidence has not been removed; each page shows its
conclusion first and keeps the detail one click away in a collapsed section.

A step's page stays populated after it has run, so moving back to check what
the calibration is standing on never costs a re-run.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QTabWidget,
    QTableView,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

import analytic_export
import analytic_source as analytic
import calibration as calib
import ingest
import paths
import provenance
import schema
import steady_state as ss
from ui_common import (
    COLOR_ACCENT,
    COLOR_ACCENT_SOFT,
    COLOR_ERROR,
    COLOR_LINE,
    COLOR_MUTED,
    COLOR_OK,
    COLOR_SECONDARY,
    Banner,
    CollapsibleSection,
    DataFrameModel,
    PlotPanel,
    badge,
    fit_table,
    number_input,
    subheading,
)


# How long the sensitivity slider must sit still before the extraction re-runs.
# Long enough that a drag across the range does not queue a dozen extractions,
# short enough that letting go feels immediate.
SENSITIVITY_DEBOUNCE_MS = 250

# Sentinel stored on the order combo for the Auto entry. Not an order, so it
# cannot be mistaken for one by anything that reads the box.
AUTO_ORDER = "auto"


def _order_code(orders: dict) -> str:
    """Compact shape signature for an axis tick, e.g. "L/Q"."""
    letters = {1: "L", 2: "Q", 3: "C"}
    return "/".join(letters.get(order, str(order)) for order in orders.values())


class ExtractCalibrateTab(QWidget):
    """Load a run, extract anchors, match analytic estimates, fit and save."""

    calibration_saved = Signal(object)

    def __init__(self, preform_id: str, app_version: str = "", parent=None):
        super().__init__(parent)
        self._preform_id = preform_id
        self._app_version = app_version
        # The active geometry's columns. Everything per-channel in this tab -
        # the section list, the fit-against dropdown, the suggested defaults -
        # is built from this rather than from a module-level channel list, so a
        # preform with six setpoints gets six sections without new UI code.
        self._schema = schema.schema_for_preform(preform_id)
        # Which channels this geometry watches, with what thresholds, and which
        # must be jointly steady per mapping. Falls back to the shipped
        # defaults for an id with no profile, which is what any pre-v1.9 caller
        # already got.
        self._extraction_profile = ss.EXTRACTION_PROFILES.get(
            preform_id,
            ss.ExtractionProfile(
                monitored_channels=ss.MONITORED_CHANNELS,
                channel_thresholds=dict(ss.DEFAULT_CHANNEL_THRESHOLDS),
                stability_groups=dict(ss.DEFAULT_STABILITY_GROUPS),
            ),
        )

        self._raw: pd.DataFrame | None = None
        self._raw_path: Path | None = None
        # Every experimental file currently loaded, in selection order. The
        # first is the time reference the others are offset against.
        self._sources: list[ingest.SourceFile] = []
        self._merge_report: ingest.MergeReport | None = None
        self._offset_boxes: dict[int, QDoubleSpinBox] = {}
        self._per_channel: ss.PerChannelExtraction | None = None
        self._analytic_source: analytic.StaticDatasetAnalyticSource | None = None
        self._analytic_path: Path | None = None
        self._estimates: dict[str, pd.DataFrame] = {}
        self._anchor_tables: dict[str, pd.DataFrame] = {}
        self._drawdown_mode = calib.DRAWDOWN_KINEMATIC
        self._calibration: calib.CalibrationSet | None = None
        # {channel: {term key: order combo}}, populated by the order panel.
        self._order_boxes: dict[str, dict[str, QComboBox]] = {}
        self._order_scan: dict[str, calib.OrderScan] = {}
        self._form_boxes: dict[str, QComboBox] = {}
        self._form_reasons: dict[str, QLabel] = {}
        self._correction_plots: dict[str, PlotPanel] = {}
        # The layout holding each channel's order combos, so the row can be
        # rebuilt when the chosen variable changes how many there should be.
        self._order_rows: dict[str, QHBoxLayout] = {}
        # {channel: {block_id}} excluded from that channel's anchors only.
        self._excluded_blocks: dict[str, set[int]] = {}
        # Set when a dev-training recommendation has been adopted, so the fit
        # report can say the selection came from held-out evidence.
        self._adopted_from_training: dict[str, str] = {}
        # {channel: Adoption} - the dev-training adoption still standing for that
        # channel. Dropped as soon as the operator moves one of that channel's
        # dropdowns away from what was adopted, so the provenance line cannot go
        # on crediting a held-out search for a shape someone has since changed.
        self._adoptions: dict[str, provenance.Adoption] = {}
        # Debounces the sensitivity slider so a drag does not re-extract on
        # every pixel.
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.timeout.connect(self._run_sensitivity_preview)
        # Guards the live-refit signal while the order boxes are being set
        # programmatically, so applying a suggestion refits once, not per box.
        self._suspend_order_signals = False

        self._build_ui()
        self.lbl_sensitivity.setText(f"{self._sensitivity:.1f}x")
        self.lbl_sensitivity_effect.setText(
            "Load a run to see what this sensitivity keeps."
        )
        self._sync_enabled()

    # ------------------------------------------------------------------ ui

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        self.steps = QTabWidget()
        self.steps.setDocumentMode(True)
        self.steps.addTab(self._build_step_extract(), "1 · Load && extract")
        self.steps.addTab(self._build_step_match(), "2 · Match analytic")
        self.steps.addTab(self._build_step_fit(), "3 · Fit && inspect")
        outer.addWidget(self.steps)

    @staticmethod
    def _page(*widgets: QWidget) -> QScrollArea:
        """One scrollable step page."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)
        for widget in widgets:
            layout.addWidget(widget)
        layout.addStretch(1)
        scroll.setWidget(body)
        return scroll

    def _build_step_extract(self) -> QWidget:
        return self._page(
            subheading(
                "A draw run sampled at 1 Hz is a handful of settled operating "
                "points separated by ramps, not thousands of experiments. This "
                "step finds those settled stretches - once per correction pair, "
                "since each one cares about a different set of channels."
            ),
            self._build_load_group(),
            self._build_settings_group(),
            self._build_diagnostic_group(),
            self._build_blocks_group(),
        )

    def _build_step_match(self) -> QWidget:
        return self._page(
            subheading(
                "Every block needs the fast estimator's prediction for it, so "
                "the calibration has two sides to compare. Blocks differ per "
                "correction pair, so each one is matched over its own windows."
            ),
            self._build_analytic_group(),
        )

    def _build_step_fit(self) -> QWidget:
        return self._page(
            subheading(
                "The correction between the estimator and the tower: a few "
                "parameters per channel, weighted by how well each block was "
                "measured."
            ),
            self._build_calibration_group(),
        )

    # -- 1 -------------------------------------------------------------

    def _build_load_group(self) -> QWidget:
        group = QGroupBox("Load raw data")
        layout = QVBoxLayout(group)

        layout.addWidget(
            subheading(
                "Raw experimental timeseries, one row per sample, with a "
                f"'{schema.TIME_COLUMN}' column and whatever QC flags the "
                "logger wrote. Select several files at once if acquisition "
                "split the channels across them - they are binned to 1 s and "
                "merged on time, and the schema is checked against the union "
                "of their columns rather than against each file separately."
            )
        )
        row = QHBoxLayout()
        self.btn_load_raw = QPushButton("Browse for experimental CSV(s)...")
        self.btn_load_raw.setProperty("accent", "true")
        self.btn_load_raw.clicked.connect(self._on_load_raw)
        row.addWidget(self.btn_load_raw)
        self.lbl_raw_path = QLabel("No file loaded.")
        self.lbl_raw_path.setProperty("role", "mono")
        self.lbl_raw_path.setWordWrap(True)
        row.addWidget(self.lbl_raw_path, 1)
        layout.addLayout(row)

        # One row per selected file. The first is the time reference and has no
        # offset control; every other file gets one, because a fixed clock error
        # between two instruments is a real and common fault and the operator is
        # the only one who can know it happened. The app does not estimate it -
        # cross-correlating instruments that measure different quantities would
        # be guessing, and a wrong guess misaligns every block silently.
        self.box_sources = QWidget()
        self._sources_layout = QVBoxLayout(self.box_sources)
        self._sources_layout.setContentsMargins(0, 0, 0, 0)
        self._sources_layout.setSpacing(4)
        self.box_sources.setVisible(False)
        layout.addWidget(self.box_sources)

        self.banner_raw = Banner()
        layout.addWidget(self.banner_raw)

        self.section_merge = CollapsibleSection(
            "Merge detail",
            summary="which file gave which column, and how full each second is",
            expanded=False,
        )
        self.section_merge.add_widget(
            subheading(
                "Per column: how many 1-second bins were an average of several "
                "raw samples, how many rested on a single sample, and how many "
                "had none at all. A column that is mostly empty seconds is "
                "thin evidence however good the instrument was."
            )
        )
        self.model_merge_sources = DataFrameModel(float_format="{:.4g}")
        self.table_merge_sources = QTableView()
        self.table_merge_sources.setModel(self.model_merge_sources)
        fit_table(self.table_merge_sources, stretch_last=True)
        self.table_merge_sources.setMinimumHeight(90)
        self.section_merge.add_widget(self.table_merge_sources)

        self.model_merge_coverage = DataFrameModel(float_format="{:.1f}")
        self.table_merge_coverage = QTableView()
        self.table_merge_coverage.setModel(self.model_merge_coverage)
        fit_table(self.table_merge_coverage, stretch_last=True)
        self.table_merge_coverage.setMinimumHeight(180)
        self.section_merge.add_widget(self.table_merge_coverage)
        self.section_merge.setVisible(False)
        layout.addWidget(self.section_merge)
        return group

    def _rebuild_source_rows(self) -> None:
        """One row per loaded file, with an offset control on all but the first."""
        while self._sources_layout.count():
            item = self._sources_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._offset_boxes = {}

        if len(self._sources) < 2:
            self.box_sources.setVisible(False)
            return

        header = QLabel(
            "Clock offsets. The first file is the time reference; shift another "
            "file if you know its instrument's clock is out. Applied before "
            "binning, so a corrected file lands in the right seconds."
        )
        header.setProperty("role", "subheading")
        header.setWordWrap(True)
        self._sources_layout.addWidget(header)

        for index, source in enumerate(self._sources):
            row = QHBoxLayout()
            name = QLabel(source.name)
            name.setProperty("role", "mono")
            row.addWidget(name, 1)
            if index == 0:
                reference = QLabel("time reference")
                reference.setProperty("role", "subheading")
                row.addWidget(reference)
            else:
                spin = number_input(-3600.0, 3600.0, 1, 0.5, "s", source.offset_s)
                spin.setToolTip(
                    "Seconds added to this file's timestamps before binning. "
                    "Positive moves it later."
                )
                spin.editingFinished.connect(self._on_offsets_changed)
                self._offset_boxes[index] = spin
                row.addWidget(spin)
            holder = QWidget()
            holder.setLayout(row)
            self._sources_layout.addWidget(holder)

        self.box_sources.setVisible(True)

    def _on_offsets_changed(self) -> None:
        """Re-merge with the offsets as they now stand."""
        if not self._sources:
            return
        updated = []
        for index, source in enumerate(self._sources):
            spin = self._offset_boxes.get(index)
            offset = float(spin.value()) if spin is not None else 0.0
            updated.append(ingest.SourceFile(source.path, offset))
        if [s.offset_s for s in updated] == [s.offset_s for s in self._sources]:
            return
        self._sources = updated
        self._load_sources(rebuild_rows=False)

    # -- 2 -------------------------------------------------------------

    def _build_settings_group(self) -> QWidget:
        group = QGroupBox("Extract steady-state blocks")
        layout = QVBoxLayout(group)
        layout.addWidget(
            subheading(
                "A sample counts as steady when the channels relevant to a "
                "given correction hold still together and no QC flag is set. "
                "The defaults are the values already validated against a real "
                "draw run, so the usual path is to press Extract and read the "
                "result - the settings below only need opening if the block "
                "counts come out wrong."
            )
        )

        # The one control that matters. A single sensitivity, scaled against
        # each channel's own measured drift, instead of six absolute thresholds
        # in six different units.
        slider_row = QHBoxLayout()
        slider_row.addWidget(QLabel("Extraction sensitivity"))
        self.slider_sensitivity = QSlider(Qt.Horizontal)
        # Integer slider over tenths, since QSlider is integer-only.
        self.slider_sensitivity.setRange(5, 100)
        self.slider_sensitivity.setValue(int(round(ss.DEFAULT_SENSITIVITY * 10)))
        self.slider_sensitivity.setTickInterval(5)
        self.slider_sensitivity.setTickPosition(QSlider.TicksBelow)
        self.slider_sensitivity.setMinimumWidth(280)
        self.slider_sensitivity.setToolTip(
            "How many multiples of a channel's own normal drift still count as "
            "settled. Higher admits longer, less settled stretches; lower keeps "
            "only the flattest. The per-channel scaling is derived from the "
            "loaded run - see the settings below."
        )
        self.slider_sensitivity.valueChanged.connect(self._on_sensitivity_changed)
        slider_row.addWidget(self.slider_sensitivity, 1)
        self.lbl_sensitivity = QLabel("")
        self.lbl_sensitivity.setProperty("role", "mono")
        self.lbl_sensitivity.setMinimumWidth(64)
        slider_row.addWidget(self.lbl_sensitivity)
        layout.addLayout(slider_row)

        self.lbl_sensitivity_effect = QLabel("")
        self.lbl_sensitivity_effect.setProperty("role", "subheading")
        self.lbl_sensitivity_effect.setWordWrap(True)
        layout.addWidget(self.lbl_sensitivity_effect)

        # The run button comes before the settings, not after them: pressing it
        # with the defaults is the normal path, and burying it under six
        # threshold spin boxes implies they need attention first.
        row = QHBoxLayout()
        self.btn_extract = QPushButton("Extract steady-state blocks")
        self.btn_extract.setProperty("accent", "true")
        self.btn_extract.clicked.connect(self._on_extract)
        row.addWidget(self.btn_extract)
        self.btn_reset_settings = QPushButton("Reset to proven defaults")
        self.btn_reset_settings.clicked.connect(self._reset_settings)
        row.addWidget(self.btn_reset_settings)
        row.addStretch(1)
        layout.addLayout(row)

        self.section_settings = CollapsibleSection(
            "Extraction settings", expanded=False
        )
        layout.addWidget(self.section_settings)

        grid = QGridLayout()
        grid.setHorizontalSpacing(24)

        timing = QFormLayout()
        self.spin_window = number_input(3, 3600, 0, 1, "s", 61)
        self.spin_window.setToolTip(
            "Length of the centred rolling window used to measure steadiness. "
            "Longer windows demand a longer settled stretch and reject short "
            "plateaus."
        )
        self.spin_gap = number_input(0, 600, 0, 1, "s", 5)
        self.spin_gap.setToolTip(
            "Runs of unsteady samples no longer than this, sitting between two "
            "steady stretches, are bridged rather than allowed to split a block."
        )
        self.spin_min_block = number_input(1, 3600, 0, 5, "s", 60)
        self.spin_min_block.setToolTip(
            "Blocks shorter than this after edge trimming are discarded: their "
            "median is too poorly determined to be worth an anchor."
        )
        self.spin_edge_trim = number_input(0, 600, 0, 1, "s", 5)
        self.spin_edge_trim.setToolTip(
            "Trimmed from each end of a block. The rolling window straddles the "
            "boundary there, so the edges are the least trustworthy samples."
        )
        self.spin_temp_tol = number_input(0, 500, 1, 5, "degC", 20)
        self.spin_temp_tol.setToolTip(
            "Blocks this far from the run's modal furnace temperature are "
            "flagged as a different process point rather than a repeat."
        )
        timing_heading = QLabel("Window and block rules")
        timing_heading.setProperty("role", "subheading")
        timing.addRow(timing_heading)
        # The lag B: secondary to the slider, adjustable but not the primary
        # interaction.
        self.spin_lag = number_input(1, 600, 0, 5, "s", ss.DEFAULT_LAG_WINDOW_S)
        self.spin_lag.setToolTip(
            "The lag B in |A(t) - A(t-B)| / max(|A(t)|, floor). Long enough that "
            "a slow ramp shows as movement, short enough not to smear the "
            "boundary between two adjacent operating points."
        )
        self.spin_lag.valueChanged.connect(self._on_sensitivity_changed)
        timing.addRow("Percent-change lag (B)", self.spin_lag)
        timing.addRow("Rolling window (absolute criterion only)", self.spin_window)
        timing.addRow("Bridge gaps up to", self.spin_gap)
        timing.addRow("Minimum block duration", self.spin_min_block)
        timing.addRow("Edge trim per side", self.spin_edge_trim)
        timing.addRow("Non-nominal temperature flag", self.spin_temp_tol)
        for spin in (
            self.spin_window,
            self.spin_gap,
            self.spin_min_block,
            self.spin_edge_trim,
        ):
            spin.valueChanged.connect(self._refresh_settings_summary)
        timing_box = QWidget()
        timing_box.setLayout(timing)
        grid.addWidget(timing_box, 0, 0)

        thresholds = QFormLayout()
        thresholds_heading = QLabel("Channels watched")
        thresholds_heading.setProperty("role", "subheading")
        thresholds.addRow(thresholds_heading)
        self._threshold_rows: dict[str, tuple[QCheckBox, object]] = {}
        # From the active geometry's extraction profile, not a module-level
        # list: a nested preform watches three capillary bores the non-nested
        # one does not have.
        for channel, default in self._extraction_profile.channel_thresholds.items():
            check = QCheckBox(channel)
            check.setChecked(True)
            spin = number_input(0.0, 1000.0, 4, 0.01, schema.unit_of(channel), default)
            spin.setToolTip(
                f"Absolute rolling-SD threshold for {channel}. Used only by the "
                "v1.0-v1.3 absolute criterion, which the app keeps for "
                "comparison; the percent criterion in use derives its own "
                "per-channel scale from the data."
            )
            check.toggled.connect(spin.setEnabled)
            check.toggled.connect(self._refresh_settings_summary)
            spin.valueChanged.connect(self._refresh_settings_summary)
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.addWidget(spin)
            thresholds.addRow(check, row)
            self._threshold_rows[channel] = (check, spin)
        thresholds_box = QWidget()
        thresholds_box.setLayout(thresholds)
        grid.addWidget(thresholds_box, 0, 1)
        grid.setColumnStretch(1, 1)
        self.section_settings.add_layout(grid)
        self.section_settings.add_widget(
            subheading(
                "Each correction pair judges steadiness on its own subset of "
                "these channels - unchecking one removes it from every criteria "
                "set that uses it."
            )
        )
        self.section_settings.add_widget(
            subheading(
                "What the sensitivity slider means per channel. The reference "
                "drift is measured from the loaded run, so one slider covers "
                "channels whose natural drift differs by three orders of "
                "magnitude without anyone entering a number per channel."
            )
        )
        self.model_references = DataFrameModel(float_format="{:.4g}")
        self.table_references = QTableView()
        self.table_references.setModel(self.model_references)
        fit_table(self.table_references, stretch_last=True)
        self.table_references.setMinimumHeight(150)
        self.section_settings.add_widget(self.table_references)
        self._refresh_settings_summary()

        self.banner_extract = Banner()
        layout.addWidget(self.banner_extract)
        return group

    def _refresh_settings_summary(self) -> None:
        """One line describing the settings, so closing the panel loses nothing."""
        active = [
            channel
            for channel, (check, _spin) in self._threshold_rows.items()
            if check.isChecked()
        ]
        timing_changed = (
            int(self.spin_lag.value()) != int(ss.DEFAULT_LAG_WINDOW_S)
            or int(self.spin_gap.value()) != 5
            or int(self.spin_min_block.value()) != 60
            or int(self.spin_edge_trim.value()) != 5
        )
        summary = (
            f"sensitivity {self._sensitivity:.1f}x, "
            f"{int(self.spin_lag.value())} s lag, "
            f"{int(self.spin_min_block.value())} s minimum block, "
            f"{len(active)} of {len(self._threshold_rows)} channel(s) watched"
        )
        if timing_changed or len(active) != len(self._threshold_rows):
            summary += " - modified from the defaults"
        else:
            summary += " - at the shipped defaults"
        self.section_settings.set_summary(summary)

    # -- 3 -------------------------------------------------------------

    def _build_diagnostic_group(self) -> QWidget:
        group = QGroupBox("Extraction diagnostic")
        layout = QVBoxLayout(group)
        layout.addWidget(
            subheading(
                "Shaded spans are the blocks kept for the selected mapping. "
                "Everything unshaded was rejected - look at what was thrown "
                "away as well as what was not. Each mapping has its own "
                "criteria set, so the shading changes with the selector."
            )
        )

        row = QHBoxLayout()
        row.addWidget(QLabel("Show blocks for"))
        self.combo_channel = QComboBox()
        self.combo_channel.setMinimumWidth(260)
        self.combo_channel.currentIndexChanged.connect(self._on_channel_changed)
        row.addWidget(self.combo_channel)
        self.lbl_criteria = QLabel("")
        self.lbl_criteria.setProperty("role", "mono")
        self.lbl_criteria.setWordWrap(True)
        row.addWidget(self.lbl_criteria, 1)
        layout.addLayout(row)

        self.lbl_group_rationale = subheading("")
        layout.addWidget(self.lbl_group_rationale)

        self.plot_diagnostic = PlotPanel(width=9.0, height=7.0)
        self.plot_diagnostic.setMinimumHeight(460)
        self.plot_diagnostic.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.plot_diagnostic.show_placeholder(
            "Load a raw timeseries and extract blocks to see the diagnostic."
        )
        layout.addWidget(self.plot_diagnostic)
        return group

    # -- 4 -------------------------------------------------------------

    def _build_blocks_group(self) -> QWidget:
        group = QGroupBox("Blocks found per correction pair")
        layout = QVBoxLayout(group)
        layout.addWidget(
            subheading(
                "How many anchors each mapping got, and how much settled time "
                "they cover, against the stricter all-channel rule."
            )
        )
        self.model_comparison = DataFrameModel(float_format="{:.4g}")
        self.table_comparison = QTableView()
        self.table_comparison.setModel(self.model_comparison)
        fit_table(self.table_comparison, stretch_last=True)
        self.table_comparison.setMinimumHeight(140)
        layout.addWidget(self.table_comparison)

        self.lbl_limiting = QLabel("")
        self.lbl_limiting.setProperty("role", "subheading")
        self.lbl_limiting.setWordWrap(True)
        layout.addWidget(self.lbl_limiting)

        detail = CollapsibleSection(
            "Block detail",
            summary="per-block medians and errors, and every window the "
            "extraction accepted or rejected",
            expanded=False,
        )
        self.section_block_detail = detail

        detail.add_widget(
            subheading(
                "Median, standard error and sample count per channel, for the "
                "mapping selected above. The standard error is what weights "
                "each block in the fit, so a long stable block counts for more "
                "than a short noisy one."
            )
        )
        self.model_blocks = DataFrameModel(float_format="{:.5g}")
        self.table_blocks = QTableView()
        self.table_blocks.setModel(self.model_blocks)
        fit_table(self.table_blocks)
        self.table_blocks.setMinimumHeight(180)
        detail.add_widget(self.table_blocks)

        detail.add_widget(
            subheading(
                "Every stretch of the run for the selected mapping, accepted "
                "and rejected alike, with the channel that was the binding "
                "constraint. This is where to look when a window that appears "
                "flat did not become a block."
            )
        )
        self.model_windows = DataFrameModel(float_format="{:.4g}")
        self.table_windows = QTableView()
        self.table_windows.setModel(self.model_windows)
        fit_table(self.table_windows, stretch_last=True)
        self.table_windows.setMinimumHeight(180)
        detail.add_widget(self.table_windows)
        layout.addWidget(detail)

        # Excluding a block is a judgement about one anchor - a block the
        # operator can see is contaminated - so it lives next to the block table
        # rather than in a settings panel. It is per channel by construction:
        # the block ids being excluded belong to this mapping's own extraction.
        exclude_row = QHBoxLayout()
        self.btn_exclude_block = QPushButton("Exclude selected block")
        self.btn_exclude_block.setToolTip(
            "Removes the selected block from this mapping's anchor set only. "
            "Other mappings keep their own blocks, and the calibration refits "
            "immediately."
        )
        self.btn_exclude_block.clicked.connect(self._on_exclude_block)
        exclude_row.addWidget(self.btn_exclude_block)
        self.btn_restore_blocks = QPushButton("Restore all")
        self.btn_restore_blocks.setToolTip(
            "Puts every excluded block for this mapping back. Exclusions are "
            "never destructive - the extraction is not re-run."
        )
        self.btn_restore_blocks.clicked.connect(self._on_restore_blocks)
        exclude_row.addWidget(self.btn_restore_blocks)
        self.lbl_excluded = QLabel("")
        self.lbl_excluded.setProperty("role", "subheading")
        self.lbl_excluded.setWordWrap(True)
        exclude_row.addWidget(self.lbl_excluded, 1)
        detail.add_layout(exclude_row)

        # The estimator hand-off. Deliberately here, after exclusion: the file
        # should describe the blocks the calibration will actually rest on, so
        # excluding a block and then exporting must not still send it.
        export_row = QHBoxLayout()
        self.btn_export_analytic = QPushButton("Export analytic input spreadsheet...")
        self.btn_export_analytic.setToolTip(
            "Writes one row per surviving block, using that block's median "
            "values, in the column format the supervisor's estimator reads. "
            "Run the estimator on it yourself, then load its output back with "
            "'Load analytic estimates...' - nothing here connects to it."
        )
        self.btn_export_analytic.clicked.connect(self._on_export_analytic)
        export_row.addWidget(self.btn_export_analytic)
        self.lbl_export_note = QLabel("")
        self.lbl_export_note.setProperty("role", "subheading")
        self.lbl_export_note.setWordWrap(True)
        export_row.addWidget(self.lbl_export_note, 1)
        detail.add_layout(export_row)

        row = QHBoxLayout()
        self.btn_export_blocks = QPushButton("Export blocks to CSV...")
        self.btn_export_blocks.clicked.connect(self._on_export_blocks)
        row.addWidget(self.btn_export_blocks)
        row.addStretch(1)
        layout.addLayout(row)
        return group

    # -- 5 -------------------------------------------------------------

    def _build_analytic_group(self) -> QWidget:
        group = QGroupBox("Analytic estimate per block")
        layout = QVBoxLayout(group)
        layout.addWidget(
            subheading(
                "A file the fast estimator has already been run over: either a "
                "full dataset carrying analytic_* columns, or just those "
                "columns plus a timestamp. Each block takes the median of the "
                "analytic rows whose timestamp falls inside its window."
            )
        )

        row = QHBoxLayout()
        self.btn_load_analytic = QPushButton("Browse for analytic estimates CSV...")
        self.btn_load_analytic.clicked.connect(self._on_load_analytic)
        row.addWidget(self.btn_load_analytic)
        self.lbl_analytic_path = QLabel("No file loaded.")
        self.lbl_analytic_path.setProperty("role", "mono")
        self.lbl_analytic_path.setWordWrap(True)
        row.addWidget(self.lbl_analytic_path, 1)
        layout.addLayout(row)

        form = QHBoxLayout()
        self.spin_tolerance = number_input(0, 3600, 0, 5, "s", 0)
        self.spin_tolerance.setToolTip(
            "Widens each block's window at both ends before matching. Leave at "
            "zero unless the analytic file is stamped more coarsely than the "
            "raw data and blocks are coming up empty."
        )
        form.addWidget(QLabel("Match tolerance"))
        form.addWidget(self.spin_tolerance)
        self.spin_min_rows = number_input(1, 1000, 0, 1, "rows", analytic.DEFAULT_MIN_ROWS)
        self.spin_min_rows.setToolTip(
            "A block matching fewer analytic rows than this is reported as "
            "thin rather than presented as solid."
        )
        form.addWidget(QLabel("Flag below"))
        form.addWidget(self.spin_min_rows)
        self.btn_match = QPushButton("Match analytic estimates to blocks")
        self.btn_match.setProperty("accent", "true")
        self.btn_match.clicked.connect(self._on_match_analytic)
        form.addWidget(self.btn_match)
        form.addStretch(1)
        layout.addLayout(form)

        self.banner_analytic = Banner()
        layout.addWidget(self.banner_analytic)

        self.model_estimates = DataFrameModel(float_format="{:.5g}")
        self.table_estimates = QTableView()
        self.table_estimates.setModel(self.model_estimates)
        fit_table(self.table_estimates, stretch_last=True)
        self.table_estimates.setMinimumHeight(180)
        layout.addWidget(self.table_estimates)
        return group

    # -- 6 -------------------------------------------------------------

    def _build_calibration_group(self) -> QWidget:
        group = QGroupBox("Fit calibration")
        layout = QVBoxLayout(group)

        row = QHBoxLayout()
        self.btn_fit = QPushButton("Fit calibration")
        self.btn_fit.setProperty("accent", "true")
        self.btn_fit.clicked.connect(self._on_fit)
        row.addWidget(self.btn_fit)
        self.btn_save = QPushButton("Save and use this calibration")
        self.btn_save.clicked.connect(self._on_save)
        row.addWidget(self.btn_save)
        self.lbl_anchor_badge = badge("No calibration yet")
        row.addWidget(self.lbl_anchor_badge)
        row.addStretch(1)
        layout.addLayout(row)

        self.banner_fit = Banner()
        layout.addWidget(self.banner_fit)
        self.banner_orders = Banner()
        layout.addWidget(self.banner_orders)

        layout.addWidget(self._build_channel_summaries())
        layout.addWidget(self._build_fit_plot_panel())
        layout.addWidget(self._build_fit_options_section())
        layout.addWidget(self._build_full_report_section())
        return group

    def _build_fit_plot_panel(self) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(
            subheading(
                "The calibration itself: the correction applied to each anchor, "
                "against what that anchor measured."
            )
        )
        self.plot_fit = PlotPanel(width=9.0, height=6.0)
        self.plot_fit.setMinimumHeight(400)
        self.plot_fit.show_placeholder(
            "Fit a calibration to see predicted against actual, with each "
            "anchor's own measurement error."
        )
        layout.addWidget(self.plot_fit)
        return box

    def _build_fit_options_section(self) -> QWidget:
        """Terms, intervals and weighting - rarely touched, so folded away."""
        section = CollapsibleSection(
            "Fit options",
            summary="correction terms, confidence level, standard-error floor",
            expanded=False,
        )
        self.section_fit_options = section

        grid = QGridLayout()
        grid.setHorizontalSpacing(24)

        terms = QVBoxLayout()
        terms.addWidget(QLabel("Correction terms"))
        self._term_checks: dict[str, QCheckBox] = {}
        for term in calib.FEATURE_TERMS:
            check = QCheckBox(term.label)
            check.setToolTip(term.description)
            check.setChecked(term.always_on or term.default_on)
            if term.always_on:
                check.setEnabled(False)
            terms.addWidget(check)
            self._term_checks[term.key] = check
        terms.addStretch(1)
        terms_box = QWidget()
        terms_box.setLayout(terms)
        grid.addWidget(terms_box, 0, 0)

        options = QFormLayout()
        self.spin_preform_od = number_input(0.0, 1000.0, 3, 1.0, "mm", 0.0)
        self.spin_preform_od.setSpecialValueText("not known - use draw speed")
        self.spin_preform_od.setToolTip(
            "Optional. With a preform outer diameter the draw-down ratio is "
            "computed geometrically, from preform OD over fiber OD, which is a "
            "property of the geometry alone. Left at zero, the ratio is "
            "computed kinematically from the analytic draw speed and the feed "
            "speed instead. Whichever is used is recorded in the calibration "
            "and reused at prediction time."
        )
        options.addRow("Preform outer diameter", self.spin_preform_od)

        self.spin_confidence = number_input(50.0, 99.9, 1, 1.0, "%", 95.0)
        self.spin_confidence.setToolTip(
            "Confidence level for the parameter intervals. Intervals use a t "
            "distribution on n - p degrees of freedom, which at this sample "
            "size is noticeably wider than a normal one."
        )
        options.addRow("Confidence level", self.spin_confidence)

        self.spin_se_floor = number_input(0.0, 50.0, 2, 0.5, "%", 2.0)
        self.spin_se_floor.setToolTip(
            "Floor on each anchor's standard error, as a percentage of that "
            "channel's spread across the anchor set. A channel that never moved "
            "during a block reports a standard error of exactly zero; without a "
            "floor that block would take infinite weight and the rest would be "
            "ignored."
        )
        options.addRow("Standard-error floor", self.spin_se_floor)
        options_box = QWidget()
        options_box.setLayout(options)
        grid.addWidget(options_box, 0, 1)
        grid.setColumnStretch(1, 1)
        section.add_layout(grid)
        section.add_widget(
            subheading(
                "Every term costs a degree of freedom out of a handful, so read "
                "the intervals before adding one."
            )
        )
        return section

    def _build_full_report_section(self) -> QWidget:
        """Coefficients, intervals and cautions in full."""
        section = CollapsibleSection(
            "Full fit report",
            summary="every coefficient, its interval, and the cautions raised",
            expanded=False,
        )
        self.section_full_report = section

        self.report_fit = QTextBrowser()
        self.report_fit.setMinimumHeight(240)
        self.report_fit.setOpenExternalLinks(False)
        section.add_widget(self.report_fit)

        self.model_parameters = DataFrameModel(float_format="{:.5g}")
        self.table_parameters = QTableView()
        self.table_parameters.setModel(self.model_parameters)
        fit_table(self.table_parameters, stretch_last=True)
        self.table_parameters.setMinimumHeight(200)
        section.add_widget(self.table_parameters)
        return section

    def _build_channel_summaries(self) -> QWidget:
        """One row per channel: what it rests on and what shape it chose.

        These three facts - anchor count, chosen shape, and the resulting fit -
        are what an operator acts on. Everything that justifies them sits in
        that channel's own Advanced section, shut until asked for.
        """
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # Permanent, not a tooltip and not dismissible. Auto and dev-training
        # had grown into two mechanisms with no stated relationship: the tab
        # explained what Auto does, the Dev menu explained what the split does,
        # and nothing said when to reach for the second instead of the first.
        # The distinction is not a preference - it is which data the choice was
        # checked against - so it belongs on screen permanently rather than
        # behind a hover.
        #
        # This replaces the old "each channel picks its own order..." caption
        # rather than joining it: that sentence said what the collapsed sections
        # below already show, and one standing line is the budget.
        self.lbl_selection_scope = QLabel(
            "Auto and manual selection here use only this run's anchors. "
            "Cross-validation checks a shape against other blocks of the same "
            "run, which cannot tell you whether the search found the process or "
            "found this run. For a configuration validated against data held "
            "back from the search, use Dev > Train Auto selection, then adopt "
            "the result - it will be credited on the channel's Source line."
        )
        self.lbl_selection_scope.setProperty("role", "explainer")
        self.lbl_selection_scope.setWordWrap(True)
        layout.addWidget(self.lbl_selection_scope)

        self._order_boxes = {}
        self._channel_sections: dict[str, CollapsibleSection] = {}
        self._order_suggestion_labels: dict[str, QLabel] = {}
        self._order_tables: dict[str, DataFrameModel] = {}
        self._order_plots: dict[str, PlotPanel] = {}
        self._training_checks: dict[str, QCheckBox] = {}

        self._equation_labels: dict[str, QLabel] = {}
        self._equation_notes: dict[str, QLabel] = {}
        self._provenance_labels: dict[str, QLabel] = {}
        self._variable_boxes: dict[str, QComboBox] = {}
        self._variable_captions: dict[str, QLabel] = {}

        for channel in self._schema.setpoint_names:
            # Two lines per channel outside every collapsible, and only two: the
            # equation, and who chose it. v1.3 set the bar at "the equation is
            # readable without opening anything"; v1.7 adds the one line that
            # says where it came from, and puts everything that merely justifies
            # it back behind the disclosure where the rest of the evidence lives.
            equation = QLabel("-")
            equation.setProperty("role", "equation")
            equation.setWordWrap(True)
            equation.setTextInteractionFlags(Qt.TextSelectableByMouse)
            self._equation_labels[channel] = equation
            layout.addWidget(equation)

            source = QLabel("")
            source.setProperty("role", "provenance")
            source.setWordWrap(True)
            source.setTextInteractionFlags(Qt.TextSelectableByMouse)
            self._provenance_labels[channel] = source
            layout.addWidget(source)

            section = CollapsibleSection(
                channel, summary="not fitted yet", expanded=False
            )
            section.toggle.toggled.connect(
                lambda checked, name=channel: self._on_channel_section_toggled(
                    name, checked
                )
            )
            self._channel_sections[channel] = section

            # What the equation means and what was left out of it. Visible by
            # default until v1.7, where it was three sentences per channel - the
            # additive/ratio reading, the terms whose interval spans zero, and
            # the comparison against the two-feature fit - times four channels,
            # above the fold, saying things that overlap the provenance line. It
            # is justification, so it sits with the rest of the justification.
            note = QLabel("")
            note.setProperty("role", "subheading")
            note.setWordWrap(True)
            self._equation_notes[channel] = note
            section.add_widget(note)

            variable_row = QHBoxLayout()
            variable_row.addWidget(QLabel("Fit against:"))
            combo = QComboBox()
            combo.setMinimumWidth(240)
            for key in calib.single_variable_keys(self._schema):
                combo.addItem(calib.variable_label(key), key)
            combo.addItem(
                calib.variable_label(calib.ENGINEERED_PAIR_KEY),
                calib.ENGINEERED_PAIR_KEY,
            )
            default_index = combo.findData(
                calib.default_variable(channel, self._schema)
            )
            combo.setCurrentIndex(max(default_index, 0))
            combo.currentIndexChanged.connect(self._on_variable_changed)
            self._variable_boxes[channel] = combo
            variable_row.addWidget(combo)

            caption = QLabel(
                "suggested: "
                f"{calib.variable_label(calib.default_variable(channel, self._schema))} "
                f"- {calib.suggested_variable_why(channel, self._schema)}. "
                "A hypothesis, not a restriction: every variable is offered."
            )
            caption.setProperty("role", "subheading")
            caption.setWordWrap(True)
            self._variable_captions[channel] = caption
            variable_row.addWidget(caption, 1)
            variable_holder = QWidget()
            variable_holder.setLayout(variable_row)
            section.add_widget(variable_holder)

            section.add_widget(self._build_form_row(channel))

            # The plot the variable and the shape are chosen from: what the
            # analytic estimate got wrong, against the one quantity being
            # blamed for it, with the fitted curve through it.
            scatter = PlotPanel(width=9.0, height=3.0)
            scatter.setMinimumHeight(260)
            scatter.show_placeholder(
                "Fit a calibration to see the correction against this variable."
            )
            self._correction_plots[channel] = scatter
            section.add_widget(scatter)

            # Rebuilt whenever the variable changes: a single-variable fit has
            # one order to choose and the engineered pair has two, so a fixed
            # row of boxes for terms that are no longer being fitted would sit
            # there looking live while doing nothing.
            holder = QWidget()
            holder_layout = QHBoxLayout(holder)
            holder_layout.setContentsMargins(0, 0, 0, 0)
            self._order_rows[channel] = holder_layout
            self._order_boxes[channel] = {}
            section.add_widget(holder)
            self._rebuild_order_boxes(channel)

            suggestion = QLabel("-")
            suggestion.setProperty("role", "subheading")
            suggestion.setWordWrap(True)
            self._order_suggestion_labels[channel] = suggestion
            section.add_widget(suggestion)

            plot = PlotPanel(width=9.0, height=2.8)
            plot.setMinimumHeight(240)
            plot.show_placeholder("Fit a calibration to compare shapes.")
            self._order_plots[channel] = plot
            section.add_widget(plot)

            caption_row = QHBoxLayout()
            caption_row.addWidget(
                subheading(
                    "Training error always improves with more parameters and is "
                    "not shown; only cross-validated error is evidence."
                ),
                1,
            )
            check = QCheckBox("show training error")
            check.setChecked(False)
            check.setToolTip(
                "Overlays the training-error curve for contrast. Off by default "
                "because it is a curve nobody should be selecting from - it "
                "falls with every added parameter whether or not the parameter "
                "means anything."
            )
            check.toggled.connect(
                lambda _checked, name=channel: self._draw_order_comparison(name)
            )
            self._training_checks[channel] = check
            caption_row.addWidget(check)
            caption_holder = QWidget()
            caption_holder.setLayout(caption_row)
            section.add_widget(caption_holder)

            model = DataFrameModel(float_format="{:.4g}")
            table = QTableView()
            table.setModel(model)
            fit_table(table, stretch_last=True)
            table.setMinimumHeight(180)
            self._order_tables[channel] = model
            section.add_widget(table)

            layout.addWidget(section)

        row = QHBoxLayout()
        self.btn_apply_suggested = QPushButton("Use the suggested orders")
        self.btn_apply_suggested.clicked.connect(self._apply_suggested_orders)
        row.addWidget(self.btn_apply_suggested)
        self.btn_all_linear = QPushButton("Set every channel to linear")
        self.btn_all_linear.clicked.connect(self._apply_linear_orders)
        row.addWidget(self.btn_all_linear)
        self.btn_all_auto = QPushButton("Set every channel to Auto")
        self.btn_all_auto.setToolTip(
            "Auto picks the simplest shape whose cross-validated error is "
            f"within {calib.AUTO_ORDER_TOLERANCE:.0%} of the best - not simply "
            "the lowest error - and stays linear below "
            f"{calib.MIN_ANCHORS_FOR_QUADRATIC} anchors regardless."
        )
        self.btn_all_auto.clicked.connect(self._apply_auto_orders)
        row.addWidget(self.btn_all_auto)
        row.addStretch(1)
        layout.addLayout(row)
        return box

    def _rebuild_order_boxes(self, channel: str) -> None:
        """One order combo per term the channel is currently fitted against."""
        layout = self._order_rows.get(channel)
        if layout is None:
            return
        previous = {
            key: box.currentData() for key, box in self._order_boxes[channel].items()
        }
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        layout.addWidget(QLabel("Function shape:"))
        boxes: dict[str, QComboBox] = {}
        variable = self._variable_boxes[channel].currentData()
        for key in calib.terms_for_variable(variable):
            if key == calib.TERM_CONST.key:
                continue
            label = QLabel(calib.variable_label(key))
            label.setProperty("role", "subheading")
            layout.addWidget(label)
            combo = QComboBox()
            combo.addItem("Auto", AUTO_ORDER)
            for order in range(1, calib.MAX_ORDER + 1):
                combo.addItem(calib.order_label(order), order)
            wanted = previous.get(key, AUTO_ORDER)
            combo.setCurrentIndex(max(combo.findData(wanted), 0))
            term = calib.TERM_BY_KEY.get(key)
            if term is not None:
                combo.setToolTip(term.description)
            combo.currentIndexChanged.connect(self._on_order_changed)
            layout.addWidget(combo)
            boxes[key] = combo
        layout.addStretch(1)
        self._order_boxes[channel] = boxes

    def _build_form_row(self, channel: str) -> QWidget:
        """Additive or ratio, with the ratio visibly refused where it must be.

        Greyed out rather than absent: a missing control looks like an
        oversight, and the operator is owed the reason the option is not there.
        """
        holder = QWidget()
        row = QHBoxLayout(holder)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(QLabel("Fit quantity:"))

        combo = QComboBox()
        combo.addItem("correction (actual - analytic)", calib.FORM_ADDITIVE)
        combo.addItem(f"ratio ({calib.RATIO_DIRECTION})", calib.FORM_RATIO)
        offered = calib.ratio_is_offered(channel)
        if not offered:
            item = combo.model().item(1)
            item.setEnabled(False)
            item.setToolTip(calib.ratio_unavailable_reason(channel))
        combo.setCurrentIndex(
            combo.findData(calib.DEFAULT_FORMS.get(channel, calib.FORM_ADDITIVE))
        )
        combo.currentIndexChanged.connect(self._on_variable_changed)
        self._form_boxes[channel] = combo
        row.addWidget(combo)

        reason = QLabel(
            "" if offered else f"ratio {calib.ratio_unavailable_reason(channel)}"
        )
        reason.setProperty("role", "subheading")
        reason.setWordWrap(True)
        row.addWidget(reason, 1)
        self._form_reasons[channel] = reason
        return holder

    def _on_variable_changed(self) -> None:
        """A new variable or form means a new fit, and a new order scan.

        Rescanning rather than only refitting: the candidate orders are scored
        against whatever variable is in play, so a scan from the previous
        variable would be recommending a shape for a function that is no longer
        being fitted.
        """
        if self._suspend_order_signals:
            return
        # The order controls belong to the terms, so they are rebuilt before the
        # refit reads them.
        self._suspend_order_signals = True
        try:
            for channel in self._variable_boxes:
                self._rebuild_order_boxes(channel)
        finally:
            self._suspend_order_signals = False
        if not self._anchor_tables:
            return
        self._refit(rescan_orders=True)

    def _selected_variables(self) -> dict[str, str]:
        return {
            channel: box.currentData()
            for channel, box in self._variable_boxes.items()
        }

    def _selected_forms(self) -> dict[str, str]:
        return {
            channel: box.currentData() for channel, box in self._form_boxes.items()
        }

    def _on_channel_section_toggled(self, channel: str, expanded: bool) -> None:
        """Draw a channel's order plot the first time it is opened.

        Four hidden matplotlib figures redrawn on every refit is wasted work
        for panels nobody has looked at, so the drawing waits for the click.
        """
        if expanded and channel in self._order_scan:
            self._draw_order_comparison(channel)
            self._draw_correction_scatter(channel)

    # -------------------------------------------------------------- state

    @property
    def _selected_channel(self) -> str | None:
        data = self.combo_channel.currentData()
        return str(data) if data else None

    @property
    def _selected_result(self) -> ss.ExtractionResult | None:
        if self._per_channel is None:
            return None
        channel = self._selected_channel
        if channel is None:
            return None
        return self._per_channel.results.get(channel)

    def _sync_enabled(self) -> None:
        has_raw = self._raw is not None
        has_blocks = self._per_channel is not None and any(
            result.n_blocks > 0 for result in self._per_channel.results.values()
        )
        has_analytic = self._analytic_source is not None
        has_anchors = bool(self._anchor_tables)
        self.btn_extract.setEnabled(has_raw)
        self.btn_export_blocks.setEnabled(has_blocks)
        self.btn_match.setEnabled(has_blocks and has_analytic)
        self.btn_fit.setEnabled(has_anchors)
        self.btn_save.setEnabled(self._calibration is not None)

    def _reset_settings(self) -> None:
        self.slider_sensitivity.setValue(int(round(ss.DEFAULT_SENSITIVITY * 10)))
        self.spin_lag.setValue(ss.DEFAULT_LAG_WINDOW_S)
        self.spin_window.setValue(61)
        self.spin_gap.setValue(5)
        self.spin_min_block.setValue(60)
        self.spin_edge_trim.setValue(5)
        self.spin_temp_tol.setValue(20)
        for channel, default in self._extraction_profile.channel_thresholds.items():
            check, spin = self._threshold_rows[channel]
            available = self._raw is None or channel in self._raw.columns
            check.setChecked(available)
            spin.setValue(default)
        self._refresh_settings_summary()
        self.banner_extract.show_message(
            "Extraction settings reset to the values validated against a real "
            "draw run.",
            "info",
        )

    @property
    def _sensitivity(self) -> float:
        return self.slider_sensitivity.value() / 10.0

    def _settings(self) -> ss.ExtractionSettings:
        # The checkboxes still choose *which* channels are watched; what they no
        # longer carry is a per-channel number, which the slider replaced.
        enabled = tuple(
            channel
            for channel, (check, _spin) in self._threshold_rows.items()
            if check.isChecked()
        )
        thresholds = {
            channel: float(spin.value())
            for channel, (check, spin) in self._threshold_rows.items()
            if check.isChecked()
        }
        return ss.ExtractionSettings(
            criterion=ss.CRITERION_PERCENT,
            sensitivity=self._sensitivity,
            lag_window_s=float(self.spin_lag.value()),
            monitored_channels=enabled,
            window_s=float(self.spin_window.value()),
            max_gap_bridge_s=float(self.spin_gap.value()),
            min_block_duration_s=float(self.spin_min_block.value()),
            edge_trim_s=float(self.spin_edge_trim.value()),
            channel_thresholds=thresholds,
            nominal_temp_tolerance_C=float(self.spin_temp_tol.value()),
            # Which channels have to be jointly steady for each mapping is a
            # geometry fact, not an operator setting - so it comes from the
            # preform's profile while everything above comes from the controls.
            stability_groups=dict(self._extraction_profile.stability_groups),
        )

    def _on_sensitivity_changed(self) -> None:
        """Show the new value at once; re-extract once the slider settles.

        Re-extracting on every pixel of a drag would run four per-mapping
        extractions per step over six thousand samples. The label moves with the
        handle so the control feels live, and the work waits for the operator to
        stop moving it - which is the difference between responsive and a
        separate Apply button.
        """
        self.lbl_sensitivity.setText(f"{self._sensitivity:.1f}x")
        self._refresh_settings_summary()
        if self._raw is None:
            self.lbl_sensitivity_effect.setText(
                "Load a run to see what this sensitivity keeps."
            )
            return
        self.lbl_sensitivity_effect.setText("re-extracting...")
        self._preview_timer.start(SENSITIVITY_DEBOUNCE_MS)

    def _run_sensitivity_preview(self) -> None:
        if self._raw is None:
            return
        self._on_extract()

    def _invalidate_downstream(self, from_blocks: bool = True) -> None:
        """Drop anything that a re-run upstream has made stale.

        A calibration that is still on screen after the blocks beneath it have
        changed is worse than none: it looks current.
        """
        if from_blocks:
            self._estimates = {}
            self.model_estimates.set_dataframe(pd.DataFrame())
            self.banner_analytic.clear_message()
        self._anchor_tables = {}
        self._calibration = None
        self._order_scan = {}
        self.model_parameters.set_dataframe(pd.DataFrame())
        self.report_fit.clear()
        self.banner_fit.clear_message()
        self.banner_orders.clear_message()
        self.lbl_anchor_badge.setText("No calibration yet")
        self.plot_fit.show_placeholder(
            "Fit a calibration to see predicted against actual, with each "
            "anchor's own measurement error."
        )
        for channel, section in self._channel_sections.items():
            section.set_summary("not fitted yet")
            self._order_tables[channel].set_dataframe(pd.DataFrame())
            self._order_suggestion_labels[channel].setText("-")
            self._equation_labels[channel].setText("-")
            self._equation_notes[channel].setText("")
            self._order_plots[channel].show_placeholder(
                "Fit a calibration to compare candidate function shapes."
            )
            self._correction_plots[channel].show_placeholder(
                "Fit a calibration to see the correction against this variable."
            )

    # --------------------------------------------------------------- load

    def _on_load_raw(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Open experimental CSV(s)",
            "",
            "CSV files (*.csv);;All files (*)",
        )
        if not paths:
            return
        self._sources = [ingest.SourceFile(str(path)) for path in paths]
        self._load_sources(rebuild_rows=True)

    def _load_sources(self, rebuild_rows: bool) -> None:
        """Merge whatever files are currently selected and adopt the result."""
        try:
            QGuiApplication.setOverrideCursor(Qt.WaitCursor)
            frame, report = ingest.load_sources(self._sources, self._schema)
        except (ingest.IngestError, ss.ExtractionError, ValueError, OSError) as exc:
            self.banner_raw.show_message(str(exc), "error")
            return
        finally:
            QGuiApplication.restoreOverrideCursor()

        notes = list(report.notes)
        self._merge_report = report
        if rebuild_rows:
            self._rebuild_source_rows()

        # The merge tables are only meaningful once there is a merge.
        self.section_merge.setVisible(report.resampled)
        if report.resampled:
            self.model_merge_sources.set_dataframe(report.source_table())
            self.model_merge_coverage.set_dataframe(report.coverage_table())

        path = self._sources[0].path
        self._raw = frame
        self._raw_path = Path(path)
        self._per_channel = None
        self.model_blocks.set_dataframe(pd.DataFrame())
        self.model_comparison.set_dataframe(pd.DataFrame())
        self.model_windows.set_dataframe(pd.DataFrame())
        self.plot_diagnostic.show_placeholder(
            "Press 'Extract steady-state blocks' to run the extraction."
        )
        self._invalidate_downstream()

        if len(self._sources) == 1:
            self.lbl_raw_path.setText(str(path))
        else:
            self.lbl_raw_path.setText(
                f"{len(self._sources)} files: "
                + ", ".join(source.name for source in self._sources)
            )
        span = frame[schema.TIME_COLUMN]
        duration_min = (span.iloc[-1] - span.iloc[0]).total_seconds() / 60.0
        present = [c for c in ss.DEFAULT_CHANNEL_THRESHOLDS if c in frame.columns]
        absent = [c for c in ss.DEFAULT_CHANNEL_THRESHOLDS if c not in frame.columns]
        for channel, (check, spin) in self._threshold_rows.items():
            available = channel in frame.columns
            check.setEnabled(available)
            check.setChecked(available)
            spin.setEnabled(available)
            check.setText(channel if available else f"{channel} (not in file)")
        self._refresh_settings_summary()

        unit = "1 s bins" if report.resampled else "samples"
        message = (
            f"Loaded {len(frame):,} {unit} spanning {duration_min:.1f} minutes "
            f"({span.iloc[0]:%Y-%m-%d %H:%M:%S} to {span.iloc[-1]:%H:%M:%S}). "
            f"{len(present)} of {len(ss.DEFAULT_CHANNEL_THRESHOLDS)} monitored "
            "channels present."
        )
        if report.resampled:
            message = report.describe() + " " + message
        # A column no file carried is a real problem; a column that is merely in
        # a different file than expected is not, and the two must not read the
        # same. Both lists below are computed against the *merged* frame, which
        # is what makes them the genuinely-absent sets.
        #
        # Reported separately because they are different in kind: a missing
        # steadiness criterion relaxes the extraction, while a missing anchor
        # column - cap_OD_um, say, which is not a criterion at all - means every
        # anchor will be short of something the calibration needs.
        missing_data = [
            c
            for c in report.missing_columns(self._schema)
            if c not in absent
        ]
        if missing_data:
            message += (
                " Absent from every supplied file, and needed for anchors: "
                + ", ".join(missing_data)
                + "."
            )
        if absent and report.resampled:
            message += (
                " Absent from every supplied file (not merely from the first), "
                "so not used as a steadiness criterion: "
                + ", ".join(absent)
                + "."
            )
        elif absent:
            message += (
                " Not present, so not used as a steadiness criterion: "
                + ", ".join(absent)
                + "."
            )
        if notes:
            message += " " + " ".join(notes)

        # A column two files both claim, with different values in it, is a data
        # problem worth stopping over rather than a preference to resolve
        # quietly. The merge keeps the first file's copy and says so here.
        kind = "warn" if (absent or missing_data) else "ok"
        if report.conflicts:
            kind = "warn"
            message += " CONFLICT: " + " ".join(report.conflicts)
        self.banner_raw.show_message(message, kind)
        self._sync_enabled()

    # ------------------------------------------------------------ extract

    def _on_extract(self) -> None:
        if self._raw is None:
            return
        try:
            QGuiApplication.setOverrideCursor(Qt.WaitCursor)
            per_channel = ss.extract_per_channel(self._raw, self._settings())
        except ss.ExtractionError as exc:
            self.banner_extract.show_message(str(exc), "error")
            return
        finally:
            QGuiApplication.restoreOverrideCursor()

        self._per_channel = per_channel
        # Block ids are assigned per extraction, so an exclusion carried over
        # from a previous sensitivity would silently remove a different window
        # than the operator chose.
        self._excluded_blocks = {}
        self._invalidate_downstream()

        previous = self._selected_channel
        self.combo_channel.blockSignals(True)
        self.combo_channel.clear()
        for channel, result in per_channel.results.items():
            self.combo_channel.addItem(
                f"{channel}  -  {result.n_blocks} block(s)", channel
            )
        index = self.combo_channel.findData(previous)
        self.combo_channel.setCurrentIndex(max(index, 0))
        self.combo_channel.blockSignals(False)

        self.model_comparison.set_dataframe(per_channel.comparison_table())
        self.model_references.set_dataframe(
            ss.reference_table(self._raw, self._settings())
        )
        self._refresh_channel_views()

        counts = per_channel.comparison_table()
        gained = int((counts["change"] > 0).sum())
        baseline = per_channel.global_result.n_blocks
        blocks_by_channel = ", ".join(
            f"{channel} {result.n_blocks}"
            for channel, result in per_channel.results.items()
        )
        self.lbl_sensitivity_effect.setText(
            f"At {self._sensitivity:.1f}x: {blocks_by_channel}. "
            f"{sum(r.total_steady_s for r in per_channel.results.values()):,.0f} s "
            "of settled time across all mappings."
        )
        message = (
            f"Per-mapping extraction from {len(per_channel.global_result.frame):,} "
            f"samples. Blocks per mapping: "
            + blocks_by_channel
            + f" (all-channel rule gives {baseline} for every channel)."
        )
        thin = [
            channel
            for channel, result in per_channel.results.items()
            if result.n_blocks < 4
        ]
        kind = "ok" if gained else "info"
        if per_channel.notes:
            message += " " + " ".join(per_channel.notes)
        if thin:
            kind = "warn"
            message += (
                " Very few anchors for: "
                + ", ".join(thin)
                + " - a three-parameter correction needs at least four blocks "
                "to have any residual degrees of freedom."
            )
        self.banner_extract.show_message(message, kind)
        self._sync_enabled()

    # ------------------------------------------------------ block exclusion

    def _kept_blocks(self, channel: str) -> pd.DataFrame:
        """One mapping's blocks with its excluded ones removed.

        Exclusion is a view over the extraction, never an edit to it: the
        extraction result keeps every block it found, so restoring costs
        nothing and an accidental exclusion is not a re-run.
        """
        result = self._per_channel.results.get(channel) if self._per_channel else None
        if result is None or result.blocks.empty:
            return pd.DataFrame()
        excluded = self._excluded_blocks.get(channel, set())
        if not excluded:
            return result.blocks
        keep = ~result.blocks["block_id"].isin(excluded)
        return result.blocks[keep]

    def _blocks_for_fitting(self) -> dict[str, pd.DataFrame]:
        if self._per_channel is None:
            return {}
        out: dict[str, pd.DataFrame] = {}
        for channel in self._per_channel.results:
            kept = self._kept_blocks(channel)
            if not kept.empty:
                out[channel] = kept
        return out

    def _on_export_analytic(self) -> None:
        """Write the estimator's input spreadsheet for the surviving blocks.

        Uses the currently-selected mapping's kept blocks: each mapping has its
        own block list since v1.1, so "the surviving blocks" is only a
        well-defined set once one of them is chosen.
        """
        channel = self._selected_channel
        if channel is None or self._per_channel is None:
            self.lbl_export_note.setText(
                "Extract steady-state blocks first, then choose a mapping."
            )
            return
        kept = self._kept_blocks(channel)
        if kept.empty:
            self.lbl_export_note.setText(
                f"{channel} has no surviving blocks to export - every one has "
                "been excluded."
            )
            return

        suggested = f"analytic_input_{self._preform_id}_{channel}.csv"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export analytic input spreadsheet", suggested, "CSV files (*.csv)"
        )
        if not path:
            return
        try:
            result = analytic_export.export_blocks(
                kept, path, preform_schema=self._schema
            )
        except (ValueError, OSError) as exc:
            self.lbl_export_note.setText(str(exc))
            return

        message = (
            f"Wrote {result.n_rows} block row(s) to {result.path.name} "
            f"({len(result.columns)} columns). Run the estimator on it, then "
            "load its output with 'Load analytic estimates...'."
        )
        if result.warnings:
            message += "  " + "  ".join(result.warnings)
        self.lbl_export_note.setText(message)

    def _on_exclude_block(self) -> None:
        channel = self._selected_channel
        result = self._selected_result
        if channel is None or result is None or result.blocks.empty:
            return
        rows = {
            index.row() for index in self.table_blocks.selectionModel().selectedRows()
        }
        if not rows:
            self.lbl_excluded.setText("Select a row in the block table first.")
            return
        table = self.model_blocks.dataframe()
        ids = {
            int(table.iloc[row]["block_id"])
            for row in rows
            if 0 <= row < len(table)
        }
        if not ids:
            return
        self._excluded_blocks.setdefault(channel, set()).update(ids)
        self._after_exclusion_change(channel)

    def _on_restore_blocks(self) -> None:
        channel = self._selected_channel
        if channel is None or not self._excluded_blocks.get(channel):
            return
        self._excluded_blocks[channel] = set()
        self._after_exclusion_change(channel)

    def _after_exclusion_change(self, channel: str) -> None:
        """Redraw the table, then refit if there is a calibration to refit.

        Same live-update behaviour as changing a variable or an order: an
        exclusion that did not visibly move the fit would be indistinguishable
        from one that had not registered.
        """
        self._refresh_block_table()
        if self._estimates and self._analytic_source is not None:
            # The analytic estimates were matched per block, so dropping a block
            # means dropping its estimate row too - `_refit` rebuilds the anchor
            # tables from whatever blocks survive.
            self._estimates = {
                name: self._analytic_source.estimates_for_blocks(frame)
                for name, frame in self._blocks_for_fitting().items()
            }
            self._refresh_estimates_table()
            if self._calibration is not None:
                self._refit(rescan_orders=True)
        self._sync_enabled()

    def _refresh_block_table(self) -> None:
        """Block table for the selected mapping, excluded rows marked as such."""
        result = self._selected_result
        channel = self._selected_channel
        if result is None or channel is None:
            return
        table = ss.block_table(result.blocks)
        if table.empty:
            self.model_blocks.set_dataframe(table)
            self.lbl_excluded.setText("")
            return
        excluded = self._excluded_blocks.get(channel, set())
        marked = table.copy()
        # An excluded block stays in the table, flagged - moving it somewhere
        # else or deleting the row would lose the operator's place and make the
        # action feel irreversible.
        marked.insert(
            1,
            "excluded",
            ["yes" if int(b) in excluded else "" for b in marked["block_id"]],
        )
        self.model_blocks.set_dataframe(marked)
        if excluded:
            self.lbl_excluded.setText(
                f"{len(excluded)} block(s) excluded from {channel}: "
                + ", ".join(str(b) for b in sorted(excluded))
                + f". {len(table) - len(excluded)} anchor(s) remain."
            )
        else:
            self.lbl_excluded.setText("")

    def _on_channel_changed(self) -> None:
        self._refresh_channel_views()

    def _refresh_channel_views(self) -> None:
        """Point the plot, block table and window table at the chosen mapping."""
        result = self._selected_result
        if result is None:
            return
        channel = self._selected_channel
        group = self._per_channel.groups.get(channel) if self._per_channel else None

        self.lbl_criteria.setText(f"criteria: {result.criteria_label}")
        self.lbl_group_rationale.setText(group.rationale if group else "")
        self._refresh_block_table()
        self.model_windows.set_dataframe(result.windows)

        limits = result.limiting_factor_counts()
        if limits.empty:
            self.lbl_limiting.setText("")
        else:
            top = limits.iloc[0]
            self.lbl_limiting.setText(
                f"Binding constraint for {channel}: {top['limiting channel']} "
                f"({top['share of unsteady']:.0%} of unsteady samples). "
                + (
                    "That is this mapping's own output channel, so it cannot be "
                    "relaxed - it is the quantity being measured."
                    if str(top["limiting channel"]) == channel
                    else "Loosening its threshold is what would add blocks here."
                )
            )
        self._draw_diagnostic(result)

    def _draw_diagnostic(self, result: ss.ExtractionResult) -> None:
        figure = self.plot_diagnostic.figure
        figure.clear()
        # Plot the criteria channels for this mapping, plus the mapping's own
        # setpoint even when it is not itself a criterion (Pocap is derived, so
        # it never appears in a criteria set but is the thing being corrected).
        channels = [c for c in result.channels_used if c in result.frame.columns]
        target = result.target_channel
        if target and target in result.frame.columns and target not in channels:
            channels = [*channels, target]
        if not channels:
            self.plot_diagnostic.show_placeholder("No channel to plot.")
            return

        frame = result.frame
        minutes = (
            frame[schema.TIME_COLUMN] - frame[schema.TIME_COLUMN].iloc[0]
        ).dt.total_seconds().to_numpy() / 60.0

        axes = figure.subplots(len(channels), 1, sharex=True, squeeze=False)[:, 0]
        steady = frame["steady_block_id"].to_numpy() > 0
        hoverable = []
        for ax, channel in zip(axes, channels):
            values = pd.to_numeric(frame[channel], errors="coerce").to_numpy(float)
            (line,) = ax.plot(
                minutes, values, color=COLOR_LINE, linewidth=0.7, zorder=2
            )
            line.set_label(channel)
            hoverable.append(line)
            for start, end in result.spans:
                ax.axvspan(
                    minutes[start],
                    minutes[end],
                    color=COLOR_ACCENT_SOFT,
                    alpha=0.55,
                    linewidth=0,
                    zorder=1,
                )
            kept = np.where(steady, values, np.nan)
            ax.plot(minutes, kept, color=COLOR_ACCENT, linewidth=1.8, zorder=3)
            # Channel name inside the axes, unit on the axis. A two-line ylabel
            # collides with its neighbour once there are six stacked panels.
            ax.set_ylabel(schema.unit_of(channel) or "", fontsize=7)
            ax.text(
                0.004,
                0.94,
                channel,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                color=COLOR_MUTED,
            )
            self.plot_diagnostic.style_axes(ax)

        for start, end in result.spans:
            block_id = int(frame["steady_block_id"].iloc[start])
            axes[0].annotate(
                str(block_id),
                xy=((minutes[start] + minutes[end]) / 2.0, 1.02),
                xycoords=("data", "axes fraction"),
                ha="center",
                va="bottom",
                fontsize=8,
                color=COLOR_ACCENT,
                fontweight="bold",
            )

        axes[-1].set_xlabel("time from start of run (min)")
        title = (
            f"{result.target_channel or 'all channels'}: {result.n_blocks} "
            "steady-state block(s) kept, shaded; everything else rejected"
        )
        axes[0].set_title(title, pad=18)
        figure.tight_layout()

        # Hovering a trace answers the question the window table answers in
        # bulk - "why isn't this bit a block?" - at the exact sample under the
        # pointer, which is how an operator actually asks it.
        def describe(selection):
            index = int(round(float(selection.target[0]) * 60.0 / result.sample_period_s))
            index = max(0, min(index, len(result.frame) - 1))
            channel = selection.artist.get_label()
            value = pd.to_numeric(result.frame[channel], errors="coerce").iloc[index]
            unit = schema.unit_of(channel)
            return (
                f"{channel} = {value:.6g} {unit}".strip()
                + "\n"
                + result.explain_at(index)
            )

        self.plot_diagnostic.set_hover(hoverable, describe)
        self.plot_diagnostic.draw_idle()

    def _on_export_blocks(self) -> None:
        if self._per_channel is None:
            return
        default = str(paths.data_dir() / "steady_state_blocks.csv")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save block summary", default, "CSV files (*.csv)"
        )
        if not path:
            return
        # One file with a `channel` column rather than four files: the block
        # boundaries genuinely differ per mapping, and a single table makes that
        # visible instead of hiding it in filenames.
        frames = []
        for channel, result in self._per_channel.results.items():
            if result.blocks.empty:
                continue
            frame = result.blocks.copy()
            frame.insert(0, "channel", channel)
            frame.insert(1, "criteria", result.criteria_label)
            frames.append(frame)
        if not frames:
            return
        try:
            pd.concat(frames, ignore_index=True).to_csv(path, index=False)
        except OSError as exc:
            QMessageBox.warning(self, "Could not save", str(exc))
            return
        self.banner_extract.show_message(
            f"Block summaries for {len(frames)} mapping(s) written to {path}.", "ok"
        )

    # ----------------------------------------------------------- analytic

    def _on_load_analytic(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open analytic estimates CSV",
            "",
            "CSV files (*.csv);;All files (*)",
        )
        if not path:
            return
        try:
            QGuiApplication.setOverrideCursor(Qt.WaitCursor)
            frame, notes = analytic.load_analytic_dataset(path)
            source = analytic.StaticDatasetAnalyticSource(
                frame,
                tolerance_s=float(self.spin_tolerance.value()),
                min_rows=int(self.spin_min_rows.value()),
                label=Path(path).name,
            )
        except (ValueError, OSError) as exc:
            self.banner_analytic.show_message(str(exc), "error")
            return
        finally:
            QGuiApplication.restoreOverrideCursor()

        self._analytic_source = source
        self._analytic_path = Path(path)
        self._estimates = {}
        self.model_estimates.set_dataframe(pd.DataFrame())
        self._invalidate_downstream(from_blocks=False)
        self.lbl_analytic_path.setText(str(path))

        first, last = source.time_span
        resolution = source.timestamp_resolution_s
        message = (
            f"Loaded {source.n_rows:,} analytic rows spanning "
            f"{first:%Y-%m-%d %H:%M:%S} to {last:%H:%M:%S}."
        )
        kind = "ok"
        if notes:
            message += " " + " ".join(notes)
        if resolution >= 30.0:
            kind = "warn"
            message += (
                f" Timestamps step in {resolution:g} s, which is coarser than "
                "1 Hz data. Short blocks may match few rows or none; if that "
                f"happens, raise the match tolerance to about {resolution:g} s."
            )
        self.banner_analytic.show_message(message, kind)
        self._sync_enabled()

    def _on_match_analytic(self) -> None:
        if self._per_channel is None or self._analytic_source is None:
            return
        self._analytic_source.tolerance_s = float(self.spin_tolerance.value())
        self._analytic_source.min_rows = int(self.spin_min_rows.value())

        # Each mapping has its own block boundaries, so each needs its own
        # match: the same analytic file, but different windows to median over.
        estimates: dict[str, pd.DataFrame] = {}
        blocks = self._blocks_for_fitting()
        for channel, frame in blocks.items():
            estimates[channel] = self._analytic_source.estimates_for_blocks(frame)
        self._estimates = estimates
        self._invalidate_downstream(from_blocks=False)
        if not estimates:
            self.banner_analytic.show_message(
                "No mapping produced any block to match against.", "error"
            )
            self._sync_enabled()
            return

        self._refresh_estimates_table()

        preform_od = float(self.spin_preform_od.value())
        try:
            tables, mode = calib.build_anchor_tables(
                blocks,
                estimates,
                preform_schema=self._schema,
                preform_OD_mm=preform_od if preform_od > 0 else None,
            )
        except calib.CalibrationError as exc:
            self.banner_analytic.show_message(str(exc), "error")
            self._sync_enabled()
            return
        self._anchor_tables = tables
        self._drawdown_mode = mode

        stacked = pd.concat(estimates.values(), ignore_index=True)
        empty = int((stacked["status"] == analytic.STATUS_EMPTY).sum())
        sparse = int((stacked["status"] == analytic.STATUS_SPARSE).sum())
        kind = "error" if empty == len(stacked) else ("warn" if empty or sparse else "ok")
        per_channel_counts = ", ".join(
            f"{channel} {len(frame)}" for channel, frame in estimates.items()
        )
        detail = (
            "Anchors per mapping: "
            + per_channel_counts
            + ". "
            + analytic.coverage_summary(stacked)
            + " Draw-down ratio computed "
            + (
                "geometrically from the preform outer diameter."
                if mode == calib.DRAWDOWN_GEOMETRIC
                else "kinematically from the analytic draw speed and the feed speed."
            )
        )
        self.banner_analytic.show_message(detail, kind)
        self._sync_enabled()

    def _refresh_estimates_table(self) -> None:
        if not self._estimates:
            self.model_estimates.set_dataframe(pd.DataFrame())
            return
        columns = ["block_id", "n_rows", "status", *schema.ANALYTIC_NAMES, "note"]
        frames = []
        for channel, frame in self._estimates.items():
            subset = frame[[c for c in columns if c in frame.columns]].copy()
            subset.insert(0, "channel", channel)
            frames.append(subset)
        self.model_estimates.set_dataframe(pd.concat(frames, ignore_index=True))

    # ---------------------------------------------------------------- fit

    def _selected_terms(self) -> tuple[str, ...]:
        return tuple(
            key for key, check in self._term_checks.items() if check.isChecked()
        )

    def _selected_orders(self) -> dict[str, dict[str, int]]:
        """Per-channel, per-term order, with Auto resolved to what it picked.

        Auto is a way of choosing an order, not a different kind of fit - by the
        time this returns, every value is a plain integer and nothing
        downstream needs to know which of them a human typed.
        """
        out: dict[str, dict[str, int]] = {}
        for channel, boxes in self._order_boxes.items():
            scan = self._order_scan.get(channel)
            auto = scan.auto_orders if scan is not None else {}
            resolved: dict[str, int] = {}
            for key, box in boxes.items():
                value = box.currentData()
                # `None` means the combo has no current item - a `findData` miss
                # setting index -1. Since v1.6 that is reachable by asking for an
                # order the dropdown no longer offers, and it must not become a
                # `TypeError` two calls further down inside the fit. Falls back
                # to whatever Auto resolves to, which is the same thing the box
                # would have shown had it never been touched.
                if value is None or value == AUTO_ORDER:
                    resolved[key] = int(auto.get(key, 1))
                else:
                    resolved[key] = int(value)
            out[channel] = resolved
        return out

    def _channel_uses_auto(self, channel: str) -> bool:
        return any(
            box.currentData() == AUTO_ORDER
            for box in self._order_boxes.get(channel, {}).values()
        )

    # ------------------------------------------------------- order shapes

    def _on_order_changed(self) -> None:
        """Refit live when the operator overrides a suggested order.

        Refitting rather than only redrawing is the point: the override has to
        show its consequences - wider intervals, a different residual pattern -
        at the moment it is made, not after a separate button press.
        """
        if self._suspend_order_signals or not self._anchor_tables:
            return
        self._refit(rescan_orders=False)

    def _apply_suggested_orders(self) -> None:
        self._set_orders(
            {channel: scan.suggested for channel, scan in self._order_scan.items()}
        )

    def _apply_auto_orders(self) -> None:
        self._set_orders(
            {
                channel: {key: AUTO_ORDER for key in boxes}
                for channel, boxes in self._order_boxes.items()
            }
        )

    def _apply_linear_orders(self) -> None:
        self._set_orders(
            {channel: scan.linear for channel, scan in self._order_scan.items()}
        )

    def _set_order_boxes_silently(self, orders) -> None:
        """Set every order box without triggering a refit per box."""
        self._suspend_order_signals = True
        try:
            for channel, per_term in orders.items():
                for key, box in self._order_boxes.get(channel, {}).items():
                    wanted = per_term.get(key, 1)
                    if wanted != AUTO_ORDER:
                        wanted = int(wanted)
                    index = box.findData(wanted)
                    if index >= 0:
                        box.setCurrentIndex(index)
        finally:
            self._suspend_order_signals = False

    def _set_orders(self, orders: dict[str, dict[str, int]]) -> None:
        if not orders:
            return
        self._set_order_boxes_silently(orders)
        self._on_order_changed()

    def _refresh_order_panel(self) -> None:
        """Show what cross-validation found, and warn where n is too small."""
        guarded: list[str] = []
        for channel, scan in self._order_scan.items():
            suggestion = self._order_suggestion_labels[channel]
            lines = [
                f"Cross-validation prefers: {calib.orders_label(scan.loo_best)}"
                + (
                    " - not applied, see the caution above."
                    if scan.guardrail_applied and scan.loo_best != scan.suggested
                    else ""
                )
            ]
            if self._channel_uses_auto(channel):
                lines.append(scan.auto_explanation)
            suggestion.setText("  ".join(lines))
            suggestion.setToolTip(scan.guardrail_note or "")
            self._order_tables[channel].set_dataframe(scan.table())
            if self._channel_sections[channel].is_expanded:
                self._draw_order_comparison(channel)
                self._draw_correction_scatter(channel)
            if scan.guardrail_applied:
                guarded.append(channel)

        self._refresh_channel_summaries()

        # A quadratic actually in use below the threshold is the thing worth
        # saying loudest, because it is the one selection nothing in this tab
        # has tested. It outranks the general small-n notice.
        cautions = self._quadratic_cautions()
        if cautions:
            self.banner_orders.show_message(" ".join(cautions), "warn")
        elif guarded:
            self.banner_orders.show_message(
                f"{len(guarded)} channel(s) have fewer than "
                f"{calib.MIN_ANCHORS_FOR_QUADRATIC} anchors "
                f"({', '.join(f'{c}: n={self._order_scan[c].n_anchor}' for c in guarded)}). "
                "Auto stays linear for them regardless of what cross-validation "
                "prefers: below that many points the linear-vs-quadratic "
                "ranking is unstable, and a single block can flip it. Quadratic "
                "is not blocked - the suggestion above can be applied "
                "deliberately - but confirm it under Dev > Train Auto selection, "
                "which scores a shape against blocks held back from the search.",
                "warn",
            )
        else:
            # Nothing to report. Since v1.7 each channel's provenance line
            # already says why its shape was chosen, so a standing banner
            # announcing that nothing went wrong is a fifth line of text
            # competing with the four that carry information.
            self.banner_orders.clear_message()

    def _quadratic_cautions(self) -> list[str]:
        """One caution per channel fitted quadratic below the anchor threshold.

        Read off the orders actually in force - `_selected_orders` resolves Auto
        to the integer it picked - so a manual override is caught, which is the
        only way to reach a quadratic below the threshold in the first place.
        """
        selected = self._selected_orders()
        out: list[str] = []
        for channel, scan in self._order_scan.items():
            caution = calib.quadratic_caution(
                channel, selected.get(channel, {}), scan.n_anchor
            )
            if caution:
                out.append(caution)
        return out

    def _reconcile_adoptions(self) -> None:
        """Drop any adoption the operator has since overridden.

        Compared against the dropdowns rather than tracked through per-channel
        signals: every path that can change a selection already funnels through
        a refit, so one comparison there catches all of them, including the ones
        a future control would otherwise have to remember to notify.
        """
        variables = self._selected_variables()
        forms = self._selected_forms()
        orders = self._selected_orders()
        for channel, adoption in list(self._adoptions.items()):
            if not adoption.matches(
                variables.get(channel), forms.get(channel), orders.get(channel)
            ):
                del self._adoptions[channel]
                self._adopted_from_training.pop(channel, None)

    def _refresh_provenance(self) -> None:
        """The one always-visible line per channel saying what chose this."""
        variables = self._selected_variables()
        forms = self._selected_forms()
        orders = self._selected_orders()
        for channel, label in self._provenance_labels.items():
            fit = (
                self._calibration.channels.get(channel)
                if self._calibration is not None
                else None
            )
            if fit is None:
                label.setText("")
                continue
            entry = provenance.for_channel(
                channel=channel,
                variable=variables.get(channel, fit.variable),
                orders=fit.orders,
                form=forms.get(channel, fit.form),
                n_anchor=fit.n_anchor,
                uses_auto=self._channel_uses_auto(channel),
                scan=self._order_scan.get(channel),
                adoption=self._adoptions.get(channel),
            )
            text = entry.line()
            # The equation drops terms whose interval spans zero, so if any were
            # dropped the reader has to know the line is a summary rather than
            # the whole fit. A count here; the names are in the section below.
            dropped = sum(
                1
                for estimate in fit.estimates
                if estimate.key != calib.TERM_CONST.key and estimate.spans_zero
            )
            if dropped:
                text += (
                    f" {dropped} term(s) with an interval spanning zero are "
                    "omitted from the equation above; open the section for which."
                )
            label.setText(text)

    def _refresh_channel_summaries(self) -> None:
        """What is behind the fit, for the collapsed section header.

        Since v1.7 this is fit *quality* only. The anchor count, the variable
        and the shape moved to the provenance line directly above, and repeating
        them here put the same three facts on screen twice per channel, eight
        times down the page - which was a good part of why the page had stopped
        being readable. The header now carries what the provenance line does
        not: how well it fits, how much room it had, and whether it complained.
        """
        for channel, section in self._channel_sections.items():
            fit = (
                self._calibration.channels.get(channel)
                if self._calibration is not None
                else None
            )
            if fit is None:
                section.set_summary("not fitted yet")
                continue
            summary = (
                f"RMS {fit.residual_rms:.4g} {fit.unit}  ·  {fit.dof} residual dof"
            )
            if fit.warnings:
                summary += f"  ·  {len(fit.warnings)} caution(s)"
            section.set_summary(summary)

    def _refresh_equations(self, result: calib.CalibrationSet) -> None:
        """The fitted function, in words, for every channel.

        Deliberately outside every collapsible section: this is the line the
        change exists to put on screen, and it is worth nothing if it needs a
        click to reach.
        """
        for channel, label in self._equation_labels.items():
            fit = result.channels.get(channel)
            note = self._equation_notes[channel]
            if fit is None:
                label.setText("-")
                note.setText("")
                continue
            text = calib.equation_text(fit)
            label.setText(text.equation)
            parts = [text.form_note]
            if text.excluded_note:
                parts.append(text.excluded_note)
            comparison = calib.variable_comparison_note(fit)
            if comparison:
                parts.append(comparison)
            note.setText("  ".join(parts))

        # With anchors in hand the refusal can name what is actually true of
        # them - Pocap's analytic really does go negative here, core_dP's does
        # not - instead of the generic argument used before any data loaded.
        for channel, label in self._form_reasons.items():
            if calib.ratio_is_offered(channel):
                continue
            label.setText(
                "ratio "
                + calib.ratio_unavailable_reason(
                    channel, self._anchor_tables.get(channel)
                )
            )

    def _draw_correction_scatter(self, channel: str) -> None:
        """What the analytic estimate got wrong, against the chosen variable.

        The fitted curve is drawn over the anchors so the shape can be judged by
        eye rather than only by a cross-validation number - which is the whole
        point of reducing the fit to one variable.
        """
        panel = self._correction_plots.get(channel)
        fit = (
            self._calibration.channels.get(channel)
            if self._calibration is not None
            else None
        )
        if panel is None or fit is None:
            return
        if fit.variable == calib.ENGINEERED_PAIR_KEY:
            panel.show_placeholder(
                "The two-feature fit has no single axis to plot against. Choose "
                "one variable above to see the correction against it."
            )
            return

        table = self._anchor_tables.get(channel)
        if table is None or table.empty:
            panel.show_placeholder("No anchors to plot.")
            return
        mask, _ = calib.usable_mask(table, channel, fit.terms, fit.form)
        used = table[mask]
        x = pd.to_numeric(used[fit.variable], errors="coerce").to_numpy(float)
        if fit.is_ratio:
            y = fit.actual / fit.analytic
            y_label = f"{calib.RATIO_DIRECTION} (-)"
        else:
            y = fit.actual - fit.analytic
            y_label = f"correction needed ({fit.unit})"

        figure = panel.figure
        figure.clear()
        ax = figure.add_subplot(111)
        points = ax.scatter(
            x, y, s=42, color=COLOR_ACCENT, zorder=3, edgecolors="none"
        )
        for xi, yi, block in zip(x, y, fit.block_ids):
            if np.isfinite(xi) and np.isfinite(yi):
                ax.annotate(
                    str(int(block)),
                    (xi, yi),
                    textcoords="offset points",
                    xytext=(5, 4),
                    fontsize=7,
                    color=COLOR_MUTED,
                )

        finite = x[np.isfinite(x)]
        if finite.size >= 2:
            grid = np.linspace(finite.min(), finite.max(), 200)
            constant, by_variable = calib.uncentred_polynomial(fit)
            curve = np.full_like(grid, constant)
            for key, powers in by_variable.items():
                if key != fit.variable:
                    continue
                for power, coefficient in powers.items():
                    curve = curve + coefficient * grid**power
            ax.plot(
                grid,
                curve,
                color=COLOR_ERROR,
                linewidth=1.4,
                zorder=2,
                label=f"fitted, {calib.order_label(fit.orders.get(fit.variable, 1))}",
            )
            ax.legend(fontsize=7, frameon=False, labelcolor=COLOR_MUTED)

        ax.set_xlabel(calib.variable_label(fit.variable))
        ax.set_ylabel(y_label)
        ax.set_title(
            f"{channel}: correction against {calib.variable_label(fit.variable)}",
            fontsize=9,
        )
        panel.style_axes(ax, grid="both")
        figure.tight_layout()

        def describe(selection):
            index = max(0, min(int(selection.index), len(x) - 1))
            return (
                f"block {fit.block_ids[index]}\n"
                f"{calib.variable_label(fit.variable)} = {x[index]:.6g}\n"
                f"{y_label} = {y[index]:.6g}\n"
                f"measured {fit.actual[index]:.6g} {fit.unit}, "
                f"analytic {fit.analytic[index]:.6g} {fit.unit}"
            )

        panel.set_hover([points], describe)
        panel.draw_idle()

    def _draw_order_comparison(self, channel: str) -> None:
        scan = self._order_scan.get(channel)
        panel = self._order_plots.get(channel)
        if scan is None or panel is None:
            return
        figure = panel.figure
        figure.clear()
        ax = figure.add_subplot(111)

        usable = [c for c in scan.candidates if c.feasible and np.isfinite(c.loo_rmse)]
        if not usable:
            panel.show_placeholder(
                f"No candidate shape is usable for {scan.channel} at "
                f"{scan.n_anchor} anchor(s)."
            )
            return
        usable.sort(key=lambda c: c.n_parameters)
        x = np.arange(len(usable), dtype=float)
        loo = np.array([c.loo_rmse for c in usable])

        cv_line = ax.plot(
            x, loo, "o-", color=COLOR_ACCENT, linewidth=1.6, markersize=6,
            label="cross-validated error",
        )[0]
        hoverable = [cv_line]

        # Off by default. The training curve makes the overfitting vivid, but it
        # is a curve nobody should be choosing from, and leaving it on screen
        # invites exactly that. The computation stays either way - only the
        # drawing is behind the toggle.
        show_training = self._training_checks[channel].isChecked()
        train_line = None
        if show_training:
            training = np.array([c.training_rmse for c in usable])
            train_line = ax.plot(
                x, training, "s--", color=COLOR_SECONDARY, linewidth=1.2,
                markersize=4, label="training error (never selects)",
            )[0]
            hoverable.append(train_line)

        chosen = scan.candidate_for(self._current_orders_for(scan.channel))
        if chosen is not None and chosen in usable:
            index = usable.index(chosen)
            ax.axvline(index, color=COLOR_OK, linewidth=1.2, linestyle=":")
            # Axes fraction for y, so the marker cannot land outside the view
            # once the log scale has picked its own limits.
            ax.annotate(
                "in use",
                xy=(index, 0.97),
                xycoords=("data", "axes fraction"),
                ha="center",
                va="top",
                fontsize=7,
                color=COLOR_OK,
                fontweight="bold",
            )

        ax.set_xticks(x)
        # Parameter counts repeat across candidates, so label by shape instead:
        # "L/Q" is wall-ratio linear, draw-down quadratic. Unambiguous at a
        # glance, with the full name in the hover.
        ax.set_xticklabels([_order_code(c.orders) for c in usable], fontsize=7)
        ax.set_xlabel(
            "candidate shape (L linear, Q quadratic; "
            + "/".join(
                calib.TERM_BY_KEY[k].label for k in usable[0].orders
            )
            + ")"
        )
        ax.set_ylabel(f"cross-validated error ({scan.unit})")
        ax.set_yscale("log")
        ax.set_title(
            f"{scan.channel}: leave-one-block-out error per candidate shape",
            fontsize=9,
        )
        if show_training:
            ax.legend(fontsize=7, frameon=False, labelcolor=COLOR_MUTED)
        panel.style_axes(ax)
        figure.tight_layout()

        def describe(selection):
            index = int(round(float(selection.target[0])))
            index = max(0, min(index, len(usable) - 1))
            candidate = usable[index]
            which = "LOO-CV" if selection.artist is cv_line else "training"
            value = candidate.loo_rmse if which == "LOO-CV" else candidate.training_rmse
            return (
                f"{candidate.label}\n"
                f"{candidate.n_parameters} parameter(s)\n"
                f"{which} error {value:.5g} {scan.unit}\n"
                f"AIC {candidate.aic:.4g} | BIC {candidate.bic:.4g}"
            )

        panel.set_hover(hoverable, describe)
        panel.draw_idle()

    def _current_orders_for(self, channel: str) -> dict[str, int]:
        """The concrete orders this channel is currently fitted with.

        Delegates rather than reading the dropdowns again. `_selected_orders`
        is the single place that turns an Auto selection into a real order -
        via `OrderScan.auto_orders`, which also applies the small-anchor
        guardrail - and reading `currentData()` directly here meant this path
        saw the raw `"auto"` sentinel and tried to `int()` it. Two readers of
        the same widget with different ideas of what its value means is the
        bug, so there is now only one.
        """
        return self._selected_orders().get(channel, {})

    # ---------------------------------------------------------------- fit

    def _on_fit(self) -> None:
        if not self._anchor_tables or self._per_channel is None:
            return
        self._refit(rescan_orders=True)

    def _refit(self, rescan_orders: bool) -> None:
        if not self._anchor_tables or self._per_channel is None:
            return
        preform_od = float(self.spin_preform_od.value())
        # The draw-down feature has to be rebuilt if the preform OD changed
        # after the analytic match ran, or the fit would use a stale column.
        blocks = self._blocks_for_fitting()
        try:
            tables, mode = calib.build_anchor_tables(
                blocks,
                self._estimates,
                preform_OD_mm=preform_od if preform_od > 0 else None,
                preform_schema=self._schema,
            )
        except calib.CalibrationError as exc:
            self.banner_fit.show_message(str(exc), "error")
            return
        self._anchor_tables, self._drawdown_mode = tables, mode

        if rescan_orders:
            # Scanning is 8 refits per candidate shape per channel - fast here,
            # but it only has to happen when the anchors change, not on every
            # manual override.
            try:
                QGuiApplication.setOverrideCursor(Qt.WaitCursor)
                selection = calib.select_orders(
                    tables,
                    terms=self._selected_terms(),
                    alpha=1.0 - float(self.spin_confidence.value()) / 100.0,
                    se_floor_fraction=float(self.spin_se_floor.value()) / 100.0,
                    variables=self._selected_variables(),
                    forms=self._selected_forms(),
                    preform_schema=self._schema,
                )
            finally:
                QGuiApplication.restoreOverrideCursor()
            self._order_scan = selection.scans
            # Back to Auto rather than to the integer Auto happens to have
            # chosen this time: Auto is a standing instruction, and freezing it
            # into a number would silently stop it tracking the anchors as the
            # sensitivity or the excluded blocks change.
            self._set_order_boxes_silently(
                {
                    name: {key: AUTO_ORDER for key in boxes}
                    for name, boxes in self._order_boxes.items()
                }
            )

        try:
            QGuiApplication.setOverrideCursor(Qt.WaitCursor)
            result = calib.fit_calibration(
                tables,
                preform_id=self._preform_id,
                drawdown_mode=mode,
                preform_OD_mm=preform_od if preform_od > 0 else None,
                terms=self._selected_terms(),
                alpha=1.0 - float(self.spin_confidence.value()) / 100.0,
                se_floor_fraction=float(self.spin_se_floor.value()) / 100.0,
                preform_schema=self._schema,
                source_files={
                    "raw_timeseries": str(self._raw_path or ""),
                    "analytic_estimates": str(self._analytic_path or ""),
                },
                notes=tuple(self._per_channel.notes),
                app_version=self._app_version,
                orders=self._selected_orders(),
                variables=self._selected_variables(),
                forms=self._selected_forms(),
            )
        except calib.CalibrationError as exc:
            self.banner_fit.show_message(str(exc), "error")
            self._calibration = None
            self._sync_enabled()
            return
        finally:
            QGuiApplication.restoreOverrideCursor()

        self._calibration = result
        # Before anything reads the provenance: an adoption that no longer
        # matches the dropdowns has been overridden and must stop being cited.
        self._reconcile_adoptions()
        self.model_parameters.set_dataframe(calib.parameter_table(result))
        self.report_fit.setHtml(self._fit_report_html(result))
        self._refresh_order_panel()
        self._refresh_equations(result)
        self._refresh_provenance()
        self._draw_fit(result)
        thin = min(result.anchor_counts.values(), default=0) < 6
        self.lbl_anchor_badge.setText(result.anchor_label)
        self.lbl_anchor_badge.setProperty("role", "badge-thin" if thin else "badge")
        self.lbl_anchor_badge.style().unpolish(self.lbl_anchor_badge)
        self.lbl_anchor_badge.style().polish(self.lbl_anchor_badge)

        warnings = sum(len(c.warnings) for c in result.channels.values())
        # The per-channel anchor counts used to be spelled out here as well as
        # on the badge and, since v1.7, on every provenance line. Three copies
        # of the same four numbers, one of them the least specific. This keeps
        # the part only the banner says: it is fitted, it is not saved.
        message = (
            f"Fitted {len(result.channels)} channel(s). Not saved yet - read "
            "each channel's Source line above, then the report below."
        )
        if warnings:
            message += f" {warnings} caution(s) raised."
        self.banner_fit.show_message(message, "warn" if warnings else "ok")
        self._sync_enabled()

    def _fit_report_html(self, result: calib.CalibrationSet) -> str:
        css = (
            "<style>"
            "body { color: #1F2933; }"
            "h4 { margin: 10px 0 3px 0; color: #1F2933; }"
            "p { margin: 3px 0 8px 0; }"
            "p.note { color: #6B7684; margin: 2px 0 8px 0; }"
            "p.equation { font-family: Consolas, monospace; font-weight: bold;"
            " color: #1F3E75; background: #EDF2FB; padding: 8px 10px;"
            " border-left: 3px solid #2D6CDF; margin: 6px 0; }"
            "code { font-family: Consolas, monospace; color: #1F4FA8; }"
            "span.warn { color: #B7791F; }"
            "span.ok { color: #2F855A; }"
            "</style>"
        )
        parts = [css]
        parts.append(
            f"<p><b>{result.anchor_label}</b> &nbsp; {len(result.terms)} term(s) "
            f"per channel &nbsp;|&nbsp; "
            f"{'geometric' if result.drawdown_mode == calib.DRAWDOWN_GEOMETRIC else 'kinematic'}"
            " draw-down ratio &nbsp;|&nbsp; "
            f"{1 - result.alpha:.0%} intervals on a t distribution.</p>"
        )
        if result.n_anchor < 6:
            parts.append(
                "<p class='note'><span class='warn'>At this many anchors, every "
                "interval below is wide and every coefficient is provisional. "
                "Treat the calibration as a correction of known sign and "
                "roughly known size, not as a precise number.</span></p>"
            )

        for channel, fit in result.channels.items():
            parts.append(
                f"<h4>{channel} <span class='note'>({fit.unit}) - "
                f"{result.anchor_label_for(channel)}</span></h4>"
            )
            equation = calib.equation_text(fit)
            parts.append(f"<p class='equation'>{equation.equation}</p>")
            parts.append(
                f"<p class='note'>Fitted quantity: <b>{fit.form_label}</b>. "
                f"Fitted against: <b>{fit.variable_label}</b>, "
                f"{calib.order_label(max(fit.orders.values(), default=1))}. "
                f"{equation.form_note}</p>"
            )
            if equation.excluded_note:
                parts.append(f"<p class='note'>{equation.excluded_note}</p>")
            comparison = calib.variable_comparison_note(fit)
            if comparison:
                parts.append(f"<p class='note'>{comparison}</p>")
            parts.append(
                f"<p>{fit.n_anchor} anchor(s), {len(fit.estimates)} parameter(s), "
                f"{fit.dof} residual degree(s) of freedom. RMS residual "
                f"<code>{fit.residual_rms:.4g}</code> {fit.unit} (unweighted); "
                "leave-one-block-out CV error "
                f"<code>{fit.loo_rmse:.4g}</code> {fit.unit} (inverse-variance "
                "weighted, so the two are not directly comparable). The raw "
                "analytic estimate was off by RMS "
                f"<code>{np.sqrt(np.mean((fit.actual - fit.analytic) ** 2)):.4g}</code> "
                f"{fit.unit} before correction.</p>"
            )
            for estimate in fit.estimates:
                verdict = (
                    "<span class='warn'>interval spans zero</span>"
                    if estimate.spans_zero
                    else "<span class='ok'>distinguishable from zero</span>"
                )
                parts.append(
                    f"<p class='note'><code>{estimate.label}</code> = "
                    f"{estimate.value:.5g} &plusmn; {estimate.se:.3g}, "
                    f"[{estimate.ci_lo:.5g}, {estimate.ci_hi:.5g}] - {verdict}</p>"
                )
            for warning in fit.warnings:
                parts.append(f"<p class='note'><span class='warn'>{warning}</span></p>")
            for note in fit.excluded_notes:
                parts.append(f"<p class='note'>{note}</p>")

        if result.notes:
            parts.append("<h4>Notes carried from extraction</h4>")
            for note in result.notes:
                parts.append(f"<p class='note'>{note}</p>")
        return "".join(parts)

    def _draw_fit(self, result: calib.CalibrationSet) -> None:
        figure = self.plot_fit.figure
        figure.clear()
        channels = list(result.channels)
        rows = int(np.ceil(len(channels) / 2))
        axes = figure.subplots(rows, 2, squeeze=False).ravel()
        hover_points: list = []
        hover_meta: dict = {}

        for ax, channel in zip(axes, channels):
            fit = result.channels[channel]
            container = ax.errorbar(
                fit.actual,
                fit.fitted,
                xerr=fit.actual_se,
                fmt="o",
                markersize=5,
                color=COLOR_ACCENT,
                ecolor=COLOR_MUTED,
                elinewidth=1,
                capsize=3,
                zorder=3,
            )
            marker = container.lines[0]
            hover_points.append(marker)
            hover_meta[marker] = (channel, fit)
            # Scaled to the calibrated comparison alone. The uncorrected
            # analytic estimate can sit hundreds of units away - the furnace
            # channel is off by about 216 degC - and putting it on the same axes
            # flattens every calibrated point into a single dot, hiding exactly
            # what this plot exists to show. Its size is stated in the corner
            # instead.
            combined = np.concatenate([fit.actual, fit.fitted])
            combined = combined[np.isfinite(combined)]
            lo, hi = float(combined.min()), float(combined.max())
            pad = 0.10 * (hi - lo) if hi > lo else max(abs(hi), 1.0) * 0.1
            ax.plot(
                [lo - pad, hi + pad],
                [lo - pad, hi + pad],
                color=COLOR_ERROR,
                linewidth=0.9,
                linestyle="--",
                zorder=1,
            )
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_ylim(lo - pad, hi + pad)
            for x, y, block_id in zip(fit.actual, fit.fitted, fit.block_ids):
                ax.annotate(
                    str(block_id),
                    (x, y),
                    textcoords="offset points",
                    xytext=(5, 4),
                    fontsize=7,
                    color=COLOR_MUTED,
                )
            raw_rms = float(np.sqrt(np.mean((fit.actual - fit.analytic) ** 2)))
            ax.text(
                0.03,
                0.95,
                f"RMS error {fit.residual_rms:.3g} {fit.unit}\n"
                f"before correction {raw_rms:.3g}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=7,
                color=COLOR_MUTED,
            )
            ax.set_title(
                f"{channel} ({fit.unit}) - n={fit.n_anchor}, {fit.order_summary}",
                fontsize=8,
            )
            ax.set_xlabel("measured")
            ax.set_ylabel("calibrated prediction")
            self.plot_fit.style_axes(ax, grid="both")

        for ax in axes[len(channels) :]:
            ax.axis("off")
        figure.suptitle(
            "THE FITTED SCATTER: calibrated prediction against measured, with "
            "each anchor's own measurement error. Dashed line is 1:1; numbers "
            "are block ids.",
            fontsize=9,
            color=COLOR_MUTED,
        )
        figure.tight_layout()

        def describe(selection):
            channel, fit = hover_meta[selection.artist]
            index = int(selection.index)
            index = max(0, min(index, len(fit.actual) - 1))
            return (
                f"block {fit.block_ids[index]} ({channel})\n"
                f"measured {fit.actual[index]:.6g} +/- {fit.actual_se[index]:.3g} "
                f"{fit.unit}\n"
                f"calibrated {fit.fitted[index]:.6g} {fit.unit}\n"
                f"analytic {fit.analytic[index]:.6g} {fit.unit}\n"
                f"residual {fit.actual[index] - fit.fitted[index]:+.4g} {fit.unit}"
            )

        self.plot_fit.set_hover(hover_points, describe)
        self.plot_fit.draw_idle()

    # --------------------------------------------------------------- save

    def _on_save(self) -> None:
        if self._calibration is None:
            return
        path = paths.calibration_path(self._preform_id)
        try:
            calib.save_calibration(
                self._calibration, path, paths.anchor_blocks_path(self._preform_id)
            )
        except OSError as exc:
            QMessageBox.warning(self, "Could not save the calibration", str(exc))
            return
        self.banner_fit.show_message(
            f"Saved to {path}. The Predict tab is now using it.", "ok"
        )
        self.calibration_saved.emit(self._calibration)

    # --------------------------------------------------- dev-mode adoption

    def current_settings(self) -> ss.ExtractionSettings:
        """The extraction settings as configured, for dev-mode training."""
        return self._settings()

    def source_paths(self) -> tuple[str, str]:
        return (str(self._raw_path or ""), str(self._analytic_path or ""))

    def preform_schema(self) -> schema.PreformSchema:
        """The geometry this tab is working in, for the dev dialog."""
        return self._schema

    def source_files(self) -> list[ingest.SourceFile]:
        """Every experimental file loaded, with its offset, for dev training.

        The dev dialog re-runs extraction from the files rather than reusing
        this tab's frame. Handing it only the first path would have it train on
        a block list built from part of the data - silently, since a missing
        column merely relaxes a steadiness criterion rather than raising.
        """
        return list(self._sources)

    def adopt_auto_configuration(
        self,
        variables: dict[str, str],
        orders: dict[str, dict[str, int]],
        forms: dict[str, str],
        scope: str = provenance.SCOPE_SEARCH_SET,
        block_counts: dict[str, dict[str, int]] | None = None,
    ) -> None:
        """Apply a dev-training recommendation as each channel's selection.

        Explicit, never automatic: this runs only when someone presses one of
        the Adopt actions after looking at both the searched and the held-out
        error. The orders are set as concrete values rather than left on Auto,
        because they were chosen on a *subset* of the anchors - leaving Auto on
        would silently re-decide them against the full set and quietly discard
        the thing the held-out split was run to establish.

        `scope` records which of the two actions was pressed, and is reported
        rather than acted on. Both carry the same configuration - the selection
        is identical, only the coefficients the operator inspected differ - and
        this tab refits either way against its own anchor set, which is the one
        the calibration it saves has to describe. Saying which fit was reviewed
        keeps that honest instead of implying the tab inherited coefficients it
        did not.
        """
        if not variables:
            return
        self._suspend_order_signals = True
        try:
            for channel, variable in variables.items():
                box = self._variable_boxes.get(channel)
                if box is not None:
                    index = box.findData(variable)
                    if index >= 0:
                        box.setCurrentIndex(index)
                form_box = self._form_boxes.get(channel)
                form = forms.get(channel)
                if form_box is not None and form is not None:
                    index = form_box.findData(form)
                    if index >= 0:
                        form_box.setCurrentIndex(index)
            for channel in variables:
                self._rebuild_order_boxes(channel)
            self._set_order_boxes_silently(orders)
        finally:
            self._suspend_order_signals = False

        self._adopted_from_training = dict(variables)
        # Recorded per channel so the provenance line can name the date, the
        # block count behind the fit that was reviewed, and which of the two
        # Adopt actions produced it - and so `_reconcile_adoptions` can tell
        # later whether this is still what the dropdowns say.
        when = dt.datetime.now()
        counts = block_counts or {}
        self._adoptions = {
            channel: provenance.Adoption(
                channel=channel,
                when=when,
                scope=scope,
                variable=variable,
                form=forms.get(channel, calib.FORM_ADDITIVE),
                orders={k: int(v) for k, v in (orders.get(channel) or {}).items()},
                n_reviewed=int(counts.get(channel, {}).get("reviewed", 0)),
                n_search=int(counts.get(channel, {}).get("search", 0)),
                n_held_out=int(counts.get(channel, {}).get("held_out", 0)),
            )
            for channel, variable in variables.items()
        }
        if self._anchor_tables:
            self._refit(rescan_orders=True)
            # `_refit` puts every box back on Auto by design; the adopted orders
            # have to be reasserted afterwards, since they are a decision made
            # on held-out evidence rather than a standing instruction.
            self._suspend_order_signals = True
            try:
                self._set_order_boxes_silently(orders)
            finally:
                self._suspend_order_signals = False
            self._refit(rescan_orders=False)
        self.banner_fit.show_message(
            f"Adopted a dev-training configuration ({scope} reviewed): "
            + ", ".join(
                f"{channel} vs {calib.variable_label(variable)}"
                for channel, variable in variables.items()
            )
            + ". The selection came from the split-validated search; this tab "
            "has refitted it against its own full anchor set. Change any "
            "dropdown to override.",
            "info",
        )

    # ------------------------------------------------------------ adopted

    def adopt_existing(self, result: calib.CalibrationSet) -> None:
        """Show a calibration loaded from disk at startup.

        The anchor table travels with the calibration, so the report and the
        parameter table can be rebuilt without the original CSVs being present.
        The extraction steps above stay empty: those files are not reloaded, and
        pretending otherwise would misrepresent what is on screen.
        """
        self._calibration = result
        self._anchor_tables = dict(result.anchor_tables)
        self._drawdown_mode = result.drawdown_mode
        self.model_parameters.set_dataframe(calib.parameter_table(result))
        self.report_fit.setHtml(self._fit_report_html(result))
        self._draw_fit(result)
        self._refresh_channel_summaries()
        # The saved calibration carries its chosen shapes, so the boxes must
        # show them rather than sitting on their linear defaults and implying
        # the loaded fit is something it is not.
        self._set_order_boxes_silently(
            {name: fit.orders for name, fit in result.channels.items()}
        )
        self._suspend_order_signals = True
        try:
            for name, fit in result.channels.items():
                box = self._variable_boxes.get(name)
                if box is not None:
                    index = box.findData(fit.variable)
                    if index >= 0:
                        box.setCurrentIndex(index)
                form_box = self._form_boxes.get(name)
                if form_box is not None:
                    index = form_box.findData(fit.form)
                    if index >= 0:
                        form_box.setCurrentIndex(index)
        finally:
            self._suspend_order_signals = False
        self._refresh_equations(result)
        self.lbl_anchor_badge.setText(result.anchor_label)
        for key, check in self._term_checks.items():
            check.setChecked(key in result.terms)
        self.spin_confidence.setValue(round((1.0 - result.alpha) * 100.0, 1))
        self.spin_se_floor.setValue(result.se_floor_fraction * 100.0)
        if result.preform_OD_mm:
            self.spin_preform_od.setValue(float(result.preform_OD_mm))
        self.banner_fit.show_message(
            "Loaded the saved calibration for this preform: "
            + result.describe()
            + ". Load a run above to refit it against new or additional anchors.",
            "info",
        )
        self._sync_enabled()
