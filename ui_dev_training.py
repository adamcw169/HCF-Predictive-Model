"""The dev-mode training dialog: run the Auto search on a split, review, adopt.

Deliberately not part of Tab 1. It is a batch operation that takes seconds,
consumes a whole dataset, and produces a recommendation about how the app
should be configured - none of which belongs in the flow an operator walks
through to calibrate a draw. It lives behind a "Dev" menu, it never runs on its
own, and the configuration it produces is applied only when someone presses
Adopt after looking at both numbers.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QTableView,
    QVBoxLayout,
    QWidget,
)

import analytic_source as analytic
import calibration as calib
import dev_training as dev
import ingest
import provenance
import schema
import steady_state as ss
from ui_common import (
    Banner,
    CollapsibleSection,
    DataFrameModel,
    fit_table,
    number_input,
    subheading,
)


class DevTrainingDialog(QDialog):
    """Search on part of the anchors, score on the rest, adopt if it holds up."""

    # variables, orders, forms, which fit the operator reviewed before adopting
    # (SCOPE_SEARCH_SET or SCOPE_FULL_DATA), and the per-channel block counts
    # behind it. The scope changes nothing about the configuration - it is
    # identical either way - so it travels as a separate argument rather than
    # being inferred downstream from data that cannot distinguish the two, and
    # the counts travel with it so the tab's provenance line can say "n=" of
    # something real rather than of whatever it happens to hold.
    configuration_adopted = Signal(dict, dict, dict, str, dict)

    # Spelled once, in `provenance`, so the dialog and the tab cannot drift
    # apart on a string they pass between them.
    SCOPE_SEARCH_SET = provenance.SCOPE_SEARCH_SET
    SCOPE_FULL_DATA = provenance.SCOPE_FULL_DATA

    def __init__(
        self,
        settings: ss.ExtractionSettings,
        raw_path: str = "",
        analytic_path: str = "",
        parent=None,
        sources: list[ingest.SourceFile] | None = None,
        preform_schema: "schema.PreformSchema | None" = None,
    ):
        super().__init__(parent)
        self._settings = settings
        # Since v1.8 the calibration tab may have been fed several experimental
        # files. This dialog re-runs extraction itself, so it needs all of them:
        # training on the first file alone would silently search a block list
        # built from part of the data, because a missing column relaxes a
        # steadiness criterion rather than raising.
        self._raw_sources: list[ingest.SourceFile] = list(sources or [])
        # The geometry the calibration tab is working in, so re-extraction here
        # reconstructs the same absolute pressures from the same chain.
        self._preform_schema = preform_schema or schema.TUBULAR_SCHEMA
        self._report: dev.TrainingReport | None = None
        # The full-dataset anchor tables the last run was built from - every
        # block, both sides of the cut. Kept so "refit on full data" refits on
        # exactly what the search was withheld from, rather than re-extracting
        # and risking a different block list than the one that was scored.
        self._tables: dict[str, pd.DataFrame] = {}
        self._refit: dev.FullDataRefit | None = None

        self.setWindowTitle("Dev: train Auto selection")
        self.resize(1080, 800)
        self._build_ui()
        self.edit_raw.setText(raw_path)
        self.edit_analytic.setText(analytic_path)
        self._sync_enabled()

    # ------------------------------------------------------------------ ui

    def _build_ui(self) -> None:
        """Scrollable body, fixed action bar.

        The dialog used to lay its contents straight onto the QDialog with no
        scroll area at all. That was survivable while everything was shut and
        unusable the moment it was not: opening the candidate detail or the
        full-data refit added several hundred pixels, the dialog does not grow
        past the screen, and the Adopt buttons - the entire point of the dialog -
        went off the bottom edge with no way to bring them back.

        So the body scrolls, and the actions do not. Keeping the buttons out of
        the scrolled region is the part that matters: it means no amount of
        expanded detail can put the primary action out of reach, rather than
        merely making it reachable after enough scrolling.
        """
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        layout.addWidget(
            subheading(
                "Cross-validation inside the anchor set proves a candidate "
                "generalises to the other blocks of the same run. It cannot say "
                "whether searching two or three dozen candidates found something "
                "about the process or about this run's particular geometry "
                "sweep. This withholds part of the data from the search and "
                "scores the winner against it once."
            )
        )

        layout.addWidget(self._build_input_group())
        layout.addWidget(self._build_split_group())

        row = QHBoxLayout()
        self.btn_run = QPushButton("Run training")
        self.btn_run.setProperty("accent", "true")
        self.btn_run.clicked.connect(self._on_run)
        row.addWidget(self.btn_run)
        row.addStretch(1)
        layout.addLayout(row)

        self.banner = Banner()
        layout.addWidget(self.banner)

        layout.addWidget(self._build_results_group(), 1)
        layout.addStretch(1)
        self.scroll.setWidget(body)
        outer.addWidget(self.scroll, 1)

        # Everything below this line stays outside the scroll area.
        actions = QWidget()
        actions.setProperty("role", "actionbar")
        row = QHBoxLayout(actions)
        row.setContentsMargins(14, 10, 14, 12)
        self.btn_adopt = QPushButton("Adopt (search-set fit)")
        self.btn_adopt.setToolTip(
            "Carries the selected variable, shape and form per channel back to "
            "the calibration tab. The coefficients under review stay the ones "
            "fitted on the search set alone - the held-out blocks took no part "
            "in them, and stay that way. Use this when the held-out portion has "
            "to remain held out for some other reason.\n\n"
            "Nothing is applied until this is pressed, and a manual override "
            "still wins."
        )
        self.btn_adopt.clicked.connect(self._on_adopt)
        row.addWidget(self.btn_adopt)

        self.btn_adopt_full = QPushButton("Adopt and refit on full data")
        self.btn_adopt_full.setProperty("accent", "true")
        self.btn_adopt_full.setToolTip(
            "Same configuration - the (variable, shape, form) the split-validated "
            "search chose, unchanged. The search is NOT re-run. Only the "
            "coefficients are refitted, using every block: the search set, the "
            "held-out set, and anything the split did not use.\n\n"
            "The refit reports its own RMS and parameter intervals below before "
            "anything is applied. More data is not assumed to be a better fit - "
            "a term can lose its significance once the last blocks are in, and "
            "that is worth seeing."
        )
        self.btn_adopt_full.clicked.connect(self._on_adopt_full)
        row.addWidget(self.btn_adopt_full)

        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.reject)
        row.addWidget(self.btn_close)
        row.addStretch(1)
        outer.addWidget(actions)

    def _build_input_group(self) -> QWidget:
        group = QGroupBox("Dataset")
        layout = QVBoxLayout(group)
        layout.addWidget(
            subheading(
                "The full dataset, not a pre-truncated one - the split has to be "
                "made here, or the held-out portion is whatever a previous step "
                "happened to leave behind. Extraction uses the settings "
                "currently configured on the calibration tab."
            )
        )
        for label, attribute, handler in (
            ("Raw timeseries", "edit_raw", self._browse_raw),
            ("Analytic estimates", "edit_analytic", self._browse_analytic),
        ):
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            edit = QLineEdit()
            edit.setPlaceholderText("no file selected")
            edit.textChanged.connect(self._sync_enabled)
            setattr(self, attribute, edit)
            row.addWidget(edit, 1)
            button = QPushButton("Browse...")
            button.clicked.connect(handler)
            row.addWidget(button)
            layout.addLayout(row)
        return group

    def _build_split_group(self) -> QWidget:
        group = QGroupBox("Split")
        layout = QVBoxLayout(group)

        row = QHBoxLayout()
        row.addWidget(QLabel("Search fraction"))
        self.spin_fraction = number_input(
            10.0, 95.0, 0, 5.0, "%", dev.DEFAULT_SEARCH_FRACTION * 100.0
        )
        self.spin_fraction.setToolTip(
            "The earliest blocks go to the search, the rest are held out. "
            "Chronological rather than shuffled: a draw run walks through its "
            "operating points in order, so a random split would put a block's "
            "near-twin on the other side and report an interpolation error."
        )
        row.addWidget(self.spin_fraction)
        row.addStretch(1)
        layout.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Always hold out block ids"))
        self.edit_forced = QLineEdit()
        self.edit_forced.setPlaceholderText(
            "e.g. 1, 4, 7 - applies to every channel; blank for a blind cut"
        )
        self.edit_forced.setToolTip(
            "For blocks already treated as external validation. These are "
            "removed from the search whatever the chronological cut would have "
            "done with them."
        )
        row.addWidget(self.edit_forced, 1)
        layout.addLayout(row)
        return group

    def _build_results_group(self) -> QWidget:
        group = QGroupBox("Result")
        layout = QVBoxLayout(group)
        layout.addWidget(
            subheading(
                "Two different numbers, never to be read as one. The LOO-CV "
                "column is how the winner did on the blocks it was chosen on; "
                "the held-out column is how it did on blocks that took no part "
                "in choosing it. A held-out error much worse than the "
                "cross-validated one means the search found this run, not the "
                "process."
            )
        )
        self.model_results = DataFrameModel(float_format="{:.4g}")
        self.table_results = QTableView()
        self.table_results.setModel(self.model_results)
        fit_table(self.table_results, stretch_last=True)
        self.table_results.setMinimumHeight(180)
        layout.addWidget(self.table_results)

        # One line per channel, always visible, saying what the search chose and
        # whether it is in use anywhere yet. The table above has the numbers; a
        # table is not a sentence, and "what did this actually decide?" should
        # not need reading across eleven columns to answer.
        self.lbl_provenance = QLabel("")
        self.lbl_provenance.setProperty("role", "subheading")
        self.lbl_provenance.setWordWrap(True)
        self.lbl_provenance.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.lbl_provenance)

        self.section_detail = CollapsibleSection(
            "Candidate detail per channel",
            summary="every candidate the search ranked, and why the winner won",
            expanded=False,
        )
        self.model_detail = DataFrameModel(float_format="{:.4g}")
        self.table_detail = QTableView()
        self.table_detail.setModel(self.model_detail)
        fit_table(self.table_detail, stretch_last=True)
        self.table_detail.setMinimumHeight(240)
        self.section_detail.add_widget(self.table_detail)
        layout.addWidget(self.section_detail)

        self.section_refit = CollapsibleSection(
            "Full-data refit",
            summary="the same configuration, coefficients refitted on every block",
            expanded=False,
        )
        self.section_refit.add_widget(
            subheading(
                "Appears once 'Adopt and refit on full data' is pressed. The "
                "search is not re-run - only the coefficients move, and they "
                "move onto every block rather than the search set alone. Read "
                "the RMS columns against each other and check whether any term "
                "stopped being distinguishable from zero: more data is not the "
                "same thing as a better fit."
            )
        )
        self.model_refit = DataFrameModel(float_format="{:.4g}")
        self.table_refit = QTableView()
        self.table_refit.setModel(self.model_refit)
        fit_table(self.table_refit, stretch_last=True)
        self.table_refit.setMinimumHeight(140)
        self.section_refit.add_widget(self.table_refit)

        self.model_refit_params = DataFrameModel(float_format="{:.4g}")
        self.table_refit_params = QTableView()
        self.table_refit_params.setModel(self.model_refit_params)
        fit_table(self.table_refit_params, stretch_last=True)
        self.table_refit_params.setMinimumHeight(160)
        self.section_refit.add_widget(self.table_refit_params)
        layout.addWidget(self.section_refit)

        self.lbl_notes = QLabel("")
        self.lbl_notes.setProperty("role", "subheading")
        self.lbl_notes.setWordWrap(True)
        layout.addWidget(self.lbl_notes)
        return group

    # --------------------------------------------------------------- state

    def _sync_enabled(self) -> None:
        ready = bool(self.edit_raw.text().strip() and self.edit_analytic.text().strip())
        self.btn_run.setEnabled(ready)
        trained = self._report is not None and bool(self._report.results)
        self.btn_adopt.setEnabled(trained)
        # Both adopt actions need a completed search that reported a held-out
        # score. The full-data refit reuses that search's winner verbatim, so
        # without one there is nothing to refit.
        self.btn_adopt_full.setEnabled(trained)

    def _browse_raw(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open raw timeseries CSV", "", "CSV files (*.csv);;All files (*)"
        )
        if path:
            self.edit_raw.setText(path)

    def _browse_analytic(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open analytic estimates CSV", "", "CSV files (*.csv);;All files (*)"
        )
        if path:
            self.edit_analytic.setText(path)

    def _forced_ids(self) -> set[int]:
        text = self.edit_forced.text().replace(",", " ").split()
        ids: set[int] = set()
        for token in text:
            try:
                ids.add(int(token))
            except ValueError:
                continue
        return ids

    # ----------------------------------------------------------------- run

    def _on_run(self) -> None:
        raw_path = self.edit_raw.text().strip()
        analytic_path = self.edit_analytic.text().strip()
        try:
            QGuiApplication.setOverrideCursor(Qt.WaitCursor)
            report = self._run_training(raw_path, analytic_path)
        except (ss.ExtractionError, calib.CalibrationError, ValueError, OSError) as exc:
            self.banner.show_message(str(exc), "error")
            self._report = None
            self._sync_enabled()
            return
        finally:
            QGuiApplication.restoreOverrideCursor()

        self._report = report
        # A new search invalidates any refit shown against the previous one.
        self._refit = None
        self.model_refit.set_dataframe(pd.DataFrame())
        self.model_refit_params.set_dataframe(pd.DataFrame())
        self.model_results.set_dataframe(report.table())
        self._refresh_detail()
        # A fresh search supersedes any earlier adoption, so the lines go back
        # to saying nothing is in use.
        self._refresh_provenance()

        if not report.results:
            self.banner.show_message(
                "No channel produced a usable split. " + " ".join(report.notes),
                "warn",
            )
        else:
            degraded = [
                channel
                for channel, result in report.results.items()
                if result.held_out_n >= result.THIN_HELD_OUT
                and result.degradation > 2.0
            ]
            message = (
                f"Trained {len(report.results)} channel(s). Read both columns "
                "before adopting."
            )
            if degraded:
                message += (
                    " Held-out error is more than double the cross-validated "
                    "error for: "
                    + ", ".join(degraded)
                    + " - the search fitted this run rather than the process on "
                    "those channels."
                )
            self.banner.show_message(message, "warn" if degraded else "ok")
        self.lbl_notes.setText(" ".join(report.notes))
        self._sync_enabled()

    def _run_training(self, raw_path: str, analytic_path: str) -> dev.TrainingReport:
        """The same extraction and matching Tab 1 runs, then the split."""
        # Prefer the exact set of files the calibration tab merged, so the block
        # list trained on is the block list that tab is showing. Falls back to
        # the path in the box when this dialog was opened without them, or when
        # the operator has since pointed it somewhere else.
        sources = self._raw_sources
        if not sources or (
            raw_path and raw_path != str(sources[0].path)
        ):
            sources = [ingest.SourceFile(raw_path)]
        # The dev dialog re-extracts, so it needs the same pressure chain the
        # calibration tab used - otherwise a nested run would arrive with only
        # its outer layer reconstructed.
        frame, _report = ingest.load_sources(sources, self._preform_schema)
        per_channel = ss.extract_per_channel(frame, self._settings)

        analytic_frame, _ = analytic.load_analytic_dataset(analytic_path)
        source = analytic.StaticDatasetAnalyticSource(analytic_frame)
        blocks = {
            channel: result.blocks
            for channel, result in per_channel.results.items()
            if not result.blocks.empty
        }
        if not blocks:
            raise calib.CalibrationError(
                "The extraction produced no blocks with these settings."
            )
        estimates = {
            channel: source.estimates_for_blocks(frame_)
            for channel, frame_ in blocks.items()
        }
        tables, _mode = calib.build_anchor_tables(blocks, estimates)
        # Extraction is complete at this point and the block list is fixed. The
        # split below only decides which side of a boundary each block falls,
        # so the search fraction cannot change how many there are.
        self._tables = tables

        forced = self._forced_ids()
        spec = dev.SplitSpec(
            search_fraction=float(self.spin_fraction.value()) / 100.0,
            always_held_out={channel: set(forced) for channel in tables} if forced else {},
        )
        return dev.train_auto_selection(tables, spec)

    def _refresh_detail(self) -> None:
        if self._report is None or not self._report.results:
            self.model_detail.set_dataframe(pd.DataFrame())
            return
        frames = []
        for channel, result in self._report.results.items():
            if result.search is None:
                continue
            table = result.search.table(limit=8)
            table.insert(0, "channel", channel)
            frames.append(table)
        if frames:
            self.model_detail.set_dataframe(pd.concat(frames, ignore_index=True))

    def _block_counts(self, scope: str) -> dict[str, dict[str, int]]:
        """Per-channel block counts for the fit being adopted.

        `reviewed` is the count behind the coefficients the operator actually
        looked at, which differs by action: the search set for a search-set fit,
        every block for a full-data refit. It is the number the tab's provenance
        line quotes as "n=", so quoting the wrong one would put a specific,
        wrong figure on screen - worse than none.
        """
        if self._report is None:
            return {}
        counts: dict[str, dict[str, int]] = {}
        for channel, result in self._report.results.items():
            n_search = len(result.split.search)
            n_full = result.split.n_blocks
            counts[channel] = {
                "reviewed": n_full if scope == self.SCOPE_FULL_DATA else n_search,
                "search": n_search,
                "held_out": result.held_out_n,
            }
        return counts

    def _refresh_provenance(self, adopted_scope: str = "") -> None:
        """One line per channel: what the search chose, and on what.

        The dialog's counterpart to the tab's provenance line. Worded as
        "search selected" rather than "Source:" because nothing here is in use
        anywhere until an Adopt button is pressed, and borrowing the tab's
        phrasing would imply it already was.
        """
        if self._report is None or not self._report.results:
            self.lbl_provenance.setText("")
            return
        lines = [
            provenance.search_line(
                channel=channel,
                variable=result.variable,
                orders=result.orders,
                form=result.form,
                n_search=len(result.split.search),
                n_held_out=result.held_out_n,
                verdict=result.verdict,
                adopted_scope=adopted_scope,
            )
            for channel, result in self._report.results.items()
        ]
        self.lbl_provenance.setText("\n".join(lines))

    def _on_adopt(self) -> None:
        """v1.5 behaviour, unchanged: carry the search-set selection across.

        The coefficients reviewed here are the ones fitted on the search set
        alone. Nothing is refitted on the held-out blocks, which is the point -
        there are reasons to keep that portion untouched that have nothing to do
        with this dialog.
        """
        if self._report is None or not self._report.results:
            return
        self.configuration_adopted.emit(
            self._report.adopted_variables(),
            self._report.adopted_orders(),
            self._report.adopted_forms(),
            self.SCOPE_SEARCH_SET,
            self._block_counts(self.SCOPE_SEARCH_SET),
        )
        self._refresh_provenance(self.SCOPE_SEARCH_SET)
        self.banner.show_message(
            "Adopted the search-set selection. Each channel's variable, shape "
            "and form are now set on the calibration tab. The coefficients "
            "reviewed here came from the search set alone - the held-out blocks "
            "took no part in them. A manual override still overrides them.",
            "ok",
        )

    def _on_adopt_full(self) -> None:
        """Refit the chosen configuration on every block, then adopt it.

        The search is not re-run. The winning (variable, shape, form) is taken
        exactly as the split-validated search left it, and only the coefficients
        move - onto the full block list, held-out set included. The refit's own
        quality is shown before the adoption message, because a coefficient
        refitted on more data is worth inspecting rather than assuming better.
        """
        if self._report is None or not self._report.results or not self._tables:
            return
        try:
            QGuiApplication.setOverrideCursor(Qt.WaitCursor)
            refit = dev.refit_on_full_data(self._report, self._tables)
        except (calib.CalibrationError, ValueError) as exc:
            self.banner.show_message(str(exc), "error")
            return
        finally:
            QGuiApplication.restoreOverrideCursor()

        self._refit = refit
        self.model_refit.set_dataframe(refit.table())
        self.model_refit_params.set_dataframe(refit.parameter_table())
        self.section_refit.set_expanded(True)

        if not refit.results:
            self.banner.show_message(
                "Nothing could be refitted on the full block list. "
                + " ".join(refit.notes),
                "warn",
            )
            self.lbl_notes.setText(" ".join(self._report.notes + refit.notes))
            return

        self.configuration_adopted.emit(
            self._report.adopted_variables(),
            self._report.adopted_orders(),
            self._report.adopted_forms(),
            self.SCOPE_FULL_DATA,
            self._block_counts(self.SCOPE_FULL_DATA),
        )
        self._refresh_provenance(self.SCOPE_FULL_DATA)
        added = sum(r.n_added for r in refit.results.values())
        message = (
            f"Refitted {len(refit.results)} channel(s) on every block "
            f"({added} block(s) beyond the search sets) and adopted the result. "
            "The selection is unchanged - only the coefficients moved. Check "
            "the full-data refit panel below before relying on it."
        )
        if refit.notes:
            message += " " + " ".join(refit.notes)
        self.banner.show_message(message, "warn" if refit.notes else "ok")
        self.lbl_notes.setText(" ".join(self._report.notes + refit.notes))
