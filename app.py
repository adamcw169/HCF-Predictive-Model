"""HCF Anchor Predictor - entry point.

A native Windows desktop app that turns a raw draw run into a handful of
steady-state anchor points, fits a small physically-motivated correction
between the fast estimator's analytic prediction and what the tower actually
did, and applies that correction to a new target geometry.

Deliberately the small-data counterpart to the HCF Draw Predictor: two to four
parameters per channel fitted by weighted least squares, not a high-capacity
learner. At eight anchors across six input dimensions, a flexible model can
interpolate its own points and nothing else; a handful of dimensionless
parameters can be identified, checked against their confidence intervals, and
stands a chance of transferring to a new preform geometry.

Fully offline: no network access anywhere.

Run from source:   python app.py
Build the exe:     pyinstaller app.spec
"""

from __future__ import annotations

import datetime as dt
import multiprocessing
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtGui import QAction, QFont, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QMessageBox,
    QStatusBar,
    QTabWidget,
)

import calibration as calib
import paths
import preform
from ui_extract_tab import ExtractCalibrateTab
from ui_predict_tab import PredictTab

APP_TITLE = "HCF Anchor Predictor"
APP_VERSION = "1.10"


def resource_path(relative: str) -> Path:
    """Locate a bundled resource both from source and inside a PyInstaller exe."""
    base = getattr(sys, "_MEIPASS", None)
    if base:
        return Path(base) / relative
    return Path(__file__).resolve().parent / relative


class CalibrationLoader(QObject):
    """Loads a saved calibration off the UI thread so startup stays responsive."""

    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, preform_id: str):
        super().__init__()
        self._preform_id = preform_id

    def run(self) -> None:
        try:
            result = calib.load_calibration(paths.calibration_path(self._preform_id))
            self.finished.emit(result)
        except Exception as exc:  # noqa: BLE001 - shown in the UI
            self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self._preform = preform.get_preform(preform.DEFAULT_PREFORM_ID)
        self._loader_thread: QThread | None = None
        self._loader: CalibrationLoader | None = None

        self.setWindowTitle(f"{APP_TITLE} {APP_VERSION}")
        self.resize(1240, 940)
        self.setMinimumSize(980, 680)

        icon_path = resource_path("app.ico")
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

        self.tabs = QTabWidget()
        self.extract_tab = ExtractCalibrateTab(self._preform.id, APP_VERSION)
        self.predict_tab = PredictTab(APP_VERSION)
        self.tabs.addTab(self.extract_tab, "Extract && calibrate")
        self.tabs.addTab(self.predict_tab, "Predict")
        self.setCentralWidget(self.tabs)

        self.extract_tab.calibration_saved.connect(self._on_calibration_saved)

        self._build_menu()
        self._build_status_bar()
        self._start_load()

    def _build_menu(self) -> None:
        """A Dev menu, kept away from the operator flow.

        Training is a batch operation over a whole dataset that produces a
        recommendation about how the app should be configured. Putting it on
        Tab 1 would invite running it casually, which is the one way to use it
        that defeats the point.
        """
        menu = self.menuBar().addMenu("&Dev")
        action = QAction("Train Auto selection...", self)
        action.setStatusTip(
            "Search variable and shape on part of the anchors, score the "
            "winner on blocks held back from the search."
        )
        action.triggered.connect(self._open_dev_training)
        menu.addAction(action)

    def _open_dev_training(self) -> None:
        from ui_dev_training import DevTrainingDialog

        raw_path, analytic_path = self.extract_tab.source_paths()
        dialog = DevTrainingDialog(
            self.extract_tab.current_settings(),
            raw_path,
            analytic_path,
            self,
            # Since v1.8 the tab may have merged several experimental files;
            # the dialog re-extracts and needs all of them.
            sources=self.extract_tab.source_files(),
            preform_schema=self.extract_tab.preform_schema(),
        )
        dialog.configuration_adopted.connect(
            self.extract_tab.adopt_auto_configuration
        )
        dialog.exec()

    def _build_status_bar(self) -> None:
        bar = QStatusBar()
        self.setStatusBar(bar)

        self.status_label = QLabel("Checking for a saved calibration...")
        bar.addWidget(self.status_label, 1)

        storage = QLabel(paths.storage_note())
        storage.setProperty("role", "mono")
        storage.setToolTip(
            "Calibrations and the prediction log are stored here. The app never "
            "accesses the network."
        )
        bar.addPermanentWidget(storage)

    # ------------------------------------------------------------- startup

    def _start_load(self) -> None:
        self._loader_thread = QThread(self)
        self._loader = CalibrationLoader(self._preform.id)
        self._loader.moveToThread(self._loader_thread)
        self._loader_thread.started.connect(self._loader.run)
        self._loader.finished.connect(self._on_loaded)
        self._loader.failed.connect(self._on_load_failed)
        self._loader_thread.start()

    def _teardown_loader(self) -> None:
        if self._loader_thread is not None:
            self._loader_thread.quit()
            self._loader_thread.wait()
            self._loader_thread.deleteLater()
            self._loader_thread = None
        self._loader = None

    def _on_loaded(self, result: calib.CalibrationSet | None) -> None:
        self._teardown_loader()
        self.predict_tab.set_calibration(result)
        if result is None:
            # Nothing calibrated yet: land on the extraction tab, and leave
            # Predict reachable but visibly unable to run.
            self.tabs.setCurrentIndex(0)
            self.status_label.setText(
                "No calibration yet - load a draw run and fit one on the "
                "'Extract & calibrate' tab."
            )
            return
        self.extract_tab.adopt_existing(result)
        self.status_label.setText(result.describe())
        self.tabs.setCurrentIndex(1)

    def _on_load_failed(self, message: str) -> None:
        self._teardown_loader()
        self.predict_tab.set_calibration(None)
        self.tabs.setCurrentIndex(0)
        self.status_label.setText("The saved calibration could not be loaded.")
        QMessageBox.warning(self, "Could not load saved calibration", message)

    def _on_calibration_saved(self, result: calib.CalibrationSet) -> None:
        self.predict_tab.set_calibration(result)
        self.status_label.setText(result.describe())
        self.tabs.setCurrentIndex(1)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        self._teardown_loader()
        super().closeEvent(event)


def _install_exception_hook(window_title: str) -> None:
    """Show unexpected errors in a dialog.

    The exe is built with --windowed, so there is no console for a traceback to
    land in. Without this an unexpected error would close the app silently.
    """

    def hook(exc_type, exc_value, exc_tb):
        text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        sys.__stderr__ and sys.__stderr__.write(text)
        try:
            QMessageBox.critical(
                None,
                f"{window_title} - unexpected error",
                f"{exc_value}\n\nDetails:\n{text[-1800:]}",
            )
        except Exception:  # noqa: BLE001 - never fail inside the handler
            pass

    sys.excepthook = hook


def _selftest(raw_csv: str, analytic_csv: str, out_path: str | None) -> int:
    """Run the whole pipeline headlessly and write a report. Returns an exit code.

    Exists because the shipped artefact is a `--windowed` single-file exe: it
    has no console, so a missing hidden import or a bundling mistake shows up
    as a window that never appears and nothing else. This drives raw CSV ->
    blocks -> analytic match -> calibration -> prediction inside the bundle
    itself and writes down what happened, which is the only way to check the
    exe rather than the source tree it was built from.

    Usage:  HCFAnchorPredictor.exe --selftest RAW.csv ANALYTIC.csv [--out FILE]
    """
    import analytic_source as analytic
    import ingest
    import schema
    import steady_state as ss

    lines: list[str] = [f"{APP_TITLE} {APP_VERSION} self-test"]
    code = 0
    try:
        frame, notes = ss.load_raw_timeseries(raw_csv)
        lines.append(f"raw: {len(frame)} samples from {raw_csv}")
        lines += [f"  note: {n}" for n in notes]

        per_channel = ss.extract_per_channel(frame)
        lines.append(f"extract: {per_channel.describe()}")
        lines.append(
            f"criterion: percent variation, sensitivity "
            f"{ss.DEFAULT_SENSITIVITY:g}x each channel's own reference drift, "
            f"lag {ss.DEFAULT_LAG_WINDOW_S:g} s"
        )
        for row in ss.compare_criteria(frame).to_dict("records"):
            lines.append(
                f"  {row['channel']}: {row['blocks (percent)']} block(s) vs "
                f"{row['blocks (absolute SD)']} under the v1.0-v1.3 absolute "
                f"criterion ({row['change']:+d}), "
                f"{row['settled s (percent)']:.0f} s settled vs "
                f"{row['settled s (absolute SD)']:.0f} s"
            )
        comparison = per_channel.comparison_table()
        for row in comparison.to_dict("records"):
            lines.append(
                f"  {row['channel']}: {row['blocks (per-mapping)']} block(s) "
                f"vs {row['blocks (all-channel)']} all-channel "
                f"({row['change']:+d}), {row['steady seconds (per-mapping)']:.0f} s "
                f"settled vs {row['steady seconds (all-channel)']:.0f} s; "
                f"limited by {row['limiting channel']}"
            )
        if all(result.n_blocks == 0 for result in per_channel.results.values()):
            raise RuntimeError("no steady-state block was extracted")

        # v1.8.2: extracting a time slice of a run must reproduce what the
        # full run finds when filtered to that same window - loading less
        # data must not silently retune the extraction. Exercised here, not
        # only in the test suite, because the bug this guards against was a
        # run-derived statistic (a segmentation tolerance built from
        # `np.ptp` of whatever frame happened to be loaded) that looked fine
        # on the reference file and only broke once a shorter file was
        # loaded - exactly the situation a --selftest run cannot see unless
        # it manufactures one.
        slice_start = frame[schema.TIME_COLUMN].iloc[len(frame) // 2]
        slice_frame = frame[frame[schema.TIME_COLUMN] >= slice_start].reset_index(
            drop=True
        )
        if len(slice_frame) >= 200:
            slice_per_channel = ss.extract_per_channel(slice_frame)
            for channel, full_result in per_channel.results.items():
                slice_result = slice_per_channel.results.get(channel)
                if slice_result is None:
                    continue
                in_window = full_result.blocks[
                    full_result.blocks["start_time"] >= slice_start
                ]
                if len(in_window) == 0:
                    continue
                if slice_result.n_blocks == 0:
                    raise RuntimeError(
                        f"{channel}: the full run finds {len(in_window)} "
                        "block(s) in the second half of this file, but "
                        "extracting that half directly finds none - "
                        "extraction is not invariant to which slice of a "
                        "run is loaded"
                    )
            lines.append(
                "slice invariance: extracting the second half of this file "
                "directly found real blocks for every channel the full run "
                "found blocks for in that window"
            )

        analytic_frame, analytic_notes = analytic.load_analytic_dataset(analytic_csv)
        source = analytic.StaticDatasetAnalyticSource(analytic_frame)
        blocks = {
            channel: result.blocks
            for channel, result in per_channel.results.items()
            if not result.blocks.empty
        }
        estimates = {
            channel: source.estimates_for_blocks(frame_)
            for channel, frame_ in blocks.items()
        }
        for channel, frame_ in estimates.items():
            lines.append(f"analytic {channel}: {analytic.coverage_summary(frame_)}")
        lines += [f"  note: {n}" for n in analytic_notes]

        tables, mode = calib.build_anchor_tables(blocks, estimates)
        variables = {
            channel: calib.default_variable(channel) for channel in tables
        }
        selection = calib.select_orders(
            tables, app_version=APP_VERSION, variables=variables
        )
        fitted = calib.fit_calibration(
            tables,
            preform_id=preform.DEFAULT_PREFORM_ID,
            drawdown_mode=mode,
            app_version=APP_VERSION,
            orders=selection.chosen_orders(),
            variables=variables,
        )
        lines.append(f"calibration: {fitted.describe()}")
        for channel, fit in fitted.channels.items():
            lines.append(f"  {fit.describe()}")
            lines.append(f"    {calib.equation_text(fit).equation}")
            lines.append(
                f"    form: {fit.form_label} | orders: {fit.order_summary} | "
                f"LOO-CV RMSE {fit.loo_rmse:.4g} {fit.unit}"
            )
            comparison = calib.variable_comparison_note(fit)
            if comparison:
                lines.append(f"    {comparison}")
            scan = selection.scans.get(channel)
            if scan is not None:
                lines.append(f"    {scan.auto_explanation}")
                if scan.guardrail_note:
                    lines.append(f"    guardrail: {scan.guardrail_note}")
                # Auto must never be argmin, and must never exceed linear below
                # the guardrail. Asserted in the shipped artefact, not only in
                # the test suite.
                if scan.guardrail_applied and set(scan.auto_orders.values()) != {1}:
                    raise RuntimeError(
                        f"{channel}: Auto exceeded linear below the anchor "
                        "guardrail"
                    )
        # The full Auto search and the held-out split, exercised inside the
        # shipped artefact rather than only in the test suite - both are new in
        # v1.5 and both have guarantees that are invisible when they break.
        import dev_training as dev

        searches = calib.search_auto_all(tables)
        lines.append("full Auto search (variable and shape together):")
        for channel, search in searches.items():
            lines.append(
                f"  {channel}: {len(search.candidates)} candidate(s), "
                f"{len(search.eligible)} eligible -> "
                f"{search.chosen.full_label if search.chosen else 'none'}"
            )
            # A ratio candidate must never be generated for a gauge pressure
            # channel, anywhere in the search.
            if not calib.ratio_is_offered(channel):
                offending = [
                    c for c in search.candidates if c.form == calib.FORM_RATIO
                ]
                if offending:
                    raise RuntimeError(
                        f"{channel} produced {len(offending)} ratio candidate(s) "
                        "in the full search"
                    )
            if search.guardrail_applied and search.chosen is not None:
                if not search.chosen.is_linear:
                    raise RuntimeError(
                        f"{channel}: the search exceeded linear below the anchor "
                        "guardrail"
                    )

        # The dev dialog is imported only when its menu item is clicked, so
        # nothing else in the app would notice it failing to bundle. Imported
        # here - not constructed, which would need a running QApplication - so
        # the shipped exe proves the module loads.
        import ui_dev_training  # noqa: F401

        import provenance

        report = dev.train_auto_selection(tables)
        lines.append("dev training (search on a split, scored on held-out blocks):")
        for channel, result in report.results.items():
            search_ids = set(result.split.search_ids)
            held_ids = set(result.split.held_out_ids)
            if not search_ids.isdisjoint(held_ids):
                raise RuntimeError(f"{channel}: the split overlapped")
            if not set(result.fit.block_ids) <= search_ids:
                raise RuntimeError(
                    f"{channel}: the winning fit saw blocks outside the search set"
                )
            # The v1.6 ordering contract: the cut falls between blocks, so the
            # two sides account for every extracted block exactly once.
            if not result.split.is_partition:
                raise RuntimeError(
                    f"{channel}: the split did not partition the block list"
                )
            lines.append(
                f"  {channel}: {calib.variable_label(result.variable)} "
                f"({calib.orders_label(result.orders)}), "
                f"LOO-CV {result.loo_rmse:.4g} on {len(result.split.search)} "
                f"search block(s) vs held-out {result.held_out_rmse:.4g} on "
                f"{result.held_out_n} never-seen block(s) - {result.verdict}"
            )
        for note in report.notes:
            lines.append(f"  note: {note}")

        # The v1.6 second action: the same winner, coefficients refitted on
        # every block. Exercised here because it is the path that could silently
        # re-run the search and destroy the held-out guarantee above.
        refit = dev.refit_on_full_data(report, tables)
        lines.append("full-data refit (same selection, coefficients on 100%):")
        for channel, entry in refit.results.items():
            trained = report.results[channel]
            if (
                entry.variable != trained.variable
                or entry.orders != trained.orders
                or entry.form != trained.form
            ):
                raise RuntimeError(
                    f"{channel}: the full-data refit changed the selection"
                )
            if entry.n_full < entry.n_search:
                raise RuntimeError(
                    f"{channel}: the full-data refit used fewer blocks than the "
                    "search set"
                )
            lines.append(
                f"  {channel}: {entry.n_search} -> {entry.n_full} block(s) "
                f"(+{entry.n_added}), RMS "
                f"{entry.search_fit.residual_rms:.4g} -> "
                f"{entry.full_fit.residual_rms:.4g} {entry.unit}, "
                f"{entry.full_fit.dof} dof"
                + (
                    f", newly not significant: {', '.join(entry.newly_insignificant)}"
                    if entry.newly_insignificant
                    else ""
                )
            )
        for note in refit.notes:
            lines.append(f"  note: {note}")

        # Extraction must not depend on where the search/held-out cut is set.
        # Checked in the shipped artefact across the whole usable range, because
        # the symptom it guards against - a block count that falls as the search
        # fraction rises - looks exactly like a worse extraction and is not one.
        baseline = dict(report.block_counts)
        trained_at: dict[float, int] = {}
        for percent in (50, 60, 70, 75, 80, 85, 90):
            probe = dev.train_auto_selection(
                tables, dev.SplitSpec(search_fraction=percent / 100.0)
            )
            if probe.block_counts != baseline:
                raise RuntimeError(
                    f"extraction changed with the split fraction: {percent}% gave "
                    f"{probe.block_counts}, 80% gave {baseline}"
                )
            for channel, result in probe.results.items():
                if not result.split.is_partition:
                    raise RuntimeError(
                        f"{channel}: the split at {percent}% was not a partition"
                    )
            trained_at[percent / 100.0] = len(probe.results)
        # And a larger search fraction must never train fewer channels.
        counts = [trained_at[f] for f in sorted(trained_at)]
        if any(b < a for a, b in zip(counts, counts[1:])):
            raise RuntimeError(
                f"raising the search fraction reduced the channels trained: {trained_at}"
            )
        lines.append(
            "split/extract ordering: block counts "
            + ", ".join(f"{c}={n}" for c, n in baseline.items())
            + f" (total {report.total_blocks}) identical at every fraction "
              "50-90%; channels trained "
            + ", ".join(f"{f:.0%}->{n}" for f, n in sorted(trained_at.items()))
        )

        # The refusal is part of what the shipped artefact has to get right, so
        # it is asserted here rather than only in the test suite.
        for channel in ("core_dP_kPa", "Pocap_kPa"):
            if calib.ratio_is_offered(channel):
                raise RuntimeError(f"{channel} must not offer the ratio form")
        lines.append(
            "form policy: ratio refused for core_dP_kPa and Pocap_kPa, offered "
            "for furnace_temp_C and draw_speed_m_min"
        )

        # Cubic is removed, not hidden. Checked in the shipped exe: no label for
        # it, no candidate carrying it, and an explicit request refused rather
        # than quietly downgraded to quadratic.
        if calib.MAX_ORDER != 2 or "cubic" in calib.ORDER_LABELS:
            raise RuntimeError(
                f"cubic is still reachable: MAX_ORDER={calib.MAX_ORDER}, "
                f"labels={calib.ORDER_LABELS}"
            )
        for source, group in (
            ("select_orders", [s.candidates for s in selection.scans.values()]),
            ("search_auto", [s.candidates for s in searches.values()]),
            (
                "dev search",
                [r.search.candidates for r in report.results.values() if r.search],
            ),
        ):
            for candidates in group:
                deep = [c for c in candidates if any(o > 2 for o in c.orders.values())]
                if deep:
                    raise RuntimeError(
                        f"{source} produced {len(deep)} candidate(s) above quadratic"
                    )
        probe_channel = next(iter(tables))
        probe_terms = calib.terms_for_variable(calib.default_variable(probe_channel))
        try:
            calib.fit_channel(
                tables[probe_channel],
                probe_channel,
                terms=probe_terms,
                orders={k: 3 for k in probe_terms if k != calib.TERM_CONST.key},
            )
        except calib.CalibrationError:
            pass
        else:
            raise RuntimeError("an explicit cubic request was accepted")
        lines.append(
            f"shape policy: up to {calib.order_label(calib.MAX_ORDER)} "
            f"({calib.MAX_ORDER} powers); cubic removed everywhere and refused "
            "when named. Quadratic competes at "
            f"{calib.MIN_ANCHORS_FOR_QUADRATIC}+ anchors, cautioned below it "
            "rather than blocked."
        )
        for channel, scan in selection.scans.items():
            caution = calib.quadratic_caution(
                channel, scan.auto_orders, scan.n_anchor
            )
            if caution:
                lines.append(f"  caution: {caution}")

        # The v1.7 provenance line, rendered inside the shipped artefact. It is
        # the sentence an operator reads to decide whether to trust a number, so
        # a version of it that says "Source: Auto" with no reason attached - or
        # that credits a held-out search for coefficients fitted elsewhere - is
        # a defect the exe should catch rather than the reader.
        lines.append("provenance (as shown beside each equation):")
        for channel, fit in fitted.channels.items():
            entry = provenance.for_channel(
                channel=channel,
                variable=fit.variable,
                orders=fit.orders,
                form=fit.form,
                n_anchor=fit.n_anchor,
                uses_auto=True,
                scan=selection.scans.get(channel),
            )
            line = entry.line()
            if not line.startswith(f"Source: {provenance.SOURCE_AUTO}"):
                raise RuntimeError(f"{channel}: provenance did not report Auto")
            if "was chosen because" not in line:
                raise RuntimeError(
                    f"{channel}: an Auto provenance line gave no specific reason"
                )
            lines.append(f"  {line}")

        # An adopted channel must name the adoption *and* admit that the
        # coefficients on screen were refitted here. Checked on a synthetic
        # adoption so the assertion holds without a dev-training run.
        sample = next(iter(fitted.channels))
        sample_fit = fitted.channels[sample]
        adopted = provenance.for_channel(
            channel=sample,
            variable=sample_fit.variable,
            orders=sample_fit.orders,
            form=sample_fit.form,
            n_anchor=sample_fit.n_anchor,
            uses_auto=True,
            adoption=provenance.Adoption(
                channel=sample,
                when=dt.datetime.now(),
                scope=provenance.SCOPE_FULL_DATA,
                variable=sample_fit.variable,
                form=sample_fit.form,
                orders=dict(sample_fit.orders),
                n_reviewed=sample_fit.n_anchor,
            ),
        ).line()
        if provenance.SOURCE_DEV not in adopted:
            raise RuntimeError("an adopted channel was not credited as dev-trained")
        if "not the fit the held-out score was computed for" not in adopted:
            raise RuntimeError(
                "an adopted channel's provenance overclaimed: it did not say the "
                "coefficients were refitted on this tab's anchors"
            )
        lines.append(f"  (adopted example) {adopted}")

        path = paths.calibration_path(fitted.preform_id)
        calib.save_calibration(
            fitted, path, paths.anchor_blocks_path(fitted.preform_id)
        )
        reloaded = calib.load_calibration(path)
        if reloaded is None:
            raise RuntimeError("the calibration did not reload from disk")
        lines.append(f"persistence: saved and reloaded {path}")

        # Replay one real anchor as a target. Pocap's blocks are the most
        # tightly constrained set, so its last anchor is a geometry every
        # channel's calibration has seen.
        reference_channel = "Pocap_kPa" if "Pocap_kPa" in tables else next(iter(tables))
        row = tables[reference_channel].iloc[-1]
        last_block = blocks[reference_channel].iloc[-1]
        target = {
            "fibre_OD_um": float(last_block["fibre_OD_um_median"]),
            "fibre_ID_um": float(last_block["fibre_ID_um_median"]),
            "cap_OD_um": float(last_block["cap_OD_um_median"]),
            "cap_ID_um": float(last_block["cap_ID_um_median"]),
            "tension_g": float(last_block["tension_g_median"]),
            "feed_speed_mm_min": float(last_block["feed_speed_mm_min_median"]),
        }
        for name in schema.ANALYTIC_NAMES:
            target[name] = float(row[name])
        predictions = reloaded.predict(target)
        lines.append(
            f"prediction (last {reference_channel} block replayed as a target):"
        )
        for channel, prediction in predictions.items():
            lines.append(
                f"  {channel} = {prediction.value:.6g} {prediction.unit} "
                f"[{prediction.ci_lo:.6g}, {prediction.ci_hi:.6g}] "
                f"({reloaded.anchor_label_for(channel)})"
            )
        # --- v1.8: multi-file ingest and the derived delta_P ---------------
        #
        # Exercised inside the bundle because both are plumbing: a merge that
        # drops a column or launders a QC flag produces a slightly different
        # block list rather than an error, and nothing downstream would notice.
        single_frame, single_report = ingest.load_sources([raw_csv])
        if single_report.resampled:
            raise RuntimeError("a lone file must not be resampled")
        if not single_frame.equals(frame):
            raise RuntimeError(
                "the single-file ingest path diverged from load_raw_timeseries"
            )

        # Split the run across two files on deliberately different clocks, then
        # merge it back and check the extraction is unmoved.
        import tempfile

        original = pd.read_csv(raw_csv)
        cap_columns = [c for c in original.columns if c.startswith("cap_")]
        if cap_columns:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_dir = Path(tmp)
                a_path = tmp_dir / "a.csv"
                b_path = tmp_dir / "b.csv"
                original.drop(columns=cap_columns).to_csv(a_path, index=False)
                cap = original[[schema.TIME_COLUMN] + cap_columns].copy()
                stamps = pd.to_datetime(cap[schema.TIME_COLUMN])
                doubled = cap.loc[cap.index.repeat(2)].copy()
                shifts = pd.to_timedelta(
                    ([0.0, 0.5] * len(cap))[: len(doubled)], unit="s"
                )
                doubled[schema.TIME_COLUMN] = (
                    stamps.loc[stamps.index.repeat(2)].to_numpy() + shifts.to_numpy()
                )
                doubled.to_csv(b_path, index=False)

                merged, merge_report = ingest.load_sources(
                    [str(a_path), str(b_path)]
                )
                if merge_report.missing_columns():
                    raise RuntimeError(
                        "the merge lost column(s): "
                        + ", ".join(merge_report.missing_columns())
                    )
                for column in ingest.FLAGS_ANY | ingest.FLAGS_ALL:
                    if column in merged.columns and merged[column].dtype != bool:
                        raise RuntimeError(
                            f"{column} came out of the merge as "
                            f"{merged[column].dtype}, not bool - a QC flag that "
                            "is not a boolean is silently ignored downstream"
                        )
                merged_per_channel = ss.extract_per_channel(merged)
                before = {
                    c: r.n_blocks for c, r in per_channel.results.items()
                }
                after = {
                    c: r.n_blocks for c, r in merged_per_channel.results.items()
                }
                if before != after:
                    raise RuntimeError(
                        f"merging moved the block counts: {before} -> {after}"
                    )
                lines.append(
                    "multi-file ingest: "
                    + merge_report.describe()
                    + f" Block counts unchanged {before}."
                )
                lines.append(
                    "  coverage: "
                    + ", ".join(
                        f"{c}={merge_report.coverage[c].frac_averaged:.0%} averaged"
                        for c in sorted(cap_columns)
                        if c in merge_report.coverage
                    )
                )

        delta = schema.delta_p(
            predictions["core_dP_kPa"].value, predictions["Pocap_kPa"].value
        )
        expected = (
            predictions["Pocap_kPa"].value - predictions["core_dP_kPa"].value
        )
        if not np.isclose(delta, expected, rtol=0, atol=0):
            raise RuntimeError("delta_P is not the plain difference it claims to be")
        lines.append(
            f"  {schema.DELTA_P_NAME} = {delta:.6g} {schema.DELTA_P_UNIT} "
            f"(Pocap_kPa {predictions['Pocap_kPa'].value:.6g} - core_dP_kPa "
            f"{predictions['core_dP_kPa'].value:.6g}; derived, not fitted)"
        )

        # --- v1.9: the registry is generic, and the nested preform is real ---
        #
        # Checked inside the bundle because the failure this guards is silent:
        # a stage that still reads a module-level channel list instead of the
        # active preform's does not raise, it just quietly produces four
        # channels for a six-channel geometry.
        import preform as preform_registry

        nested = preform_registry.get_preform("dnanf")
        if not nested.is_implemented:
            raise RuntimeError("the nested preform should be implemented in v1.9")
        if nested.schema is None or len(nested.schema.setpoint_names) != 6:
            raise RuntimeError("the nested preform did not carry a six-setpoint schema")

        # The existing geometry must be untouched by its presence.
        existing = preform_registry.get_preform(preform.DEFAULT_PREFORM_ID)
        if existing.schema is not schema.NONNESTED_SCHEMA:
            raise RuntimeError("the non-nested preform's schema was replaced")
        if schema.NONNESTED_SCHEMA.setpoint_names != schema.SETPOINT_NAMES:
            raise RuntimeError("the non-nested setpoint list moved")
        if schema.NONNESTED_SCHEMA.wall_ratio_names != ("cap_wall_ratio",):
            raise RuntimeError("the non-nested wall ratio was renamed")
        if calib.single_variable_keys() != calib.SINGLE_VARIABLE_KEYS:
            raise RuntimeError("the non-nested fit-variable list widened")
        for channel in schema.SETPOINT_NAMES:
            if calib.default_variable(channel) != calib.SUGGESTED_VARIABLE[channel]:
                raise RuntimeError(f"{channel}: suggested variable moved")
        # A stored calibration carries only a preform id; the schema behind it
        # has to resolve from that alone, or old files stop predicting.
        if schema.schema_for_preform(fitted.preform_id) is not schema.NONNESTED_SCHEMA:
            raise RuntimeError(
                "a stored non-nested calibration did not resolve to its own schema"
            )

        lines.append(
            "preform registry: "
            + ", ".join(
                f"{p.id} ({len(p.target_names)} setpoints"
                + (", implemented)" if p.is_implemented else ", not implemented)")
                for p in preform_registry.REGISTRY
            )
        )
        nested_deltas = nested.schema.compute_deltas(
            {
                "core_dP_kPa": 5.0,
                "Pocap_kPa": 16.0,
                "Pmcap_kPa": 20.0,
                "Picap_kPa": 12.0,
            }
        )
        # The third step is negative on purpose here: the chain must report it
        # rather than clamp it.
        if nested_deltas["deltaPicap"] >= 0:
            raise RuntimeError("a negative pressure step was not reported as negative")
        lines.append(
            "nested pressure chain: "
            + ", ".join(f"{k}={v:+.4g}" for k, v in nested_deltas.items())
            + " (derived, not fitted; negatives reported, not clamped)"
        )
        lines.append(
            "nested wall ratios: "
            + ", ".join(nested.schema.wall_ratio_names)
            + "; suggested per pressure channel: "
            + ", ".join(
                f"{c}->{calib.default_variable(c, nested.schema)}"
                for c in ("Pocap_kPa", schema.MCAP_COLUMN, schema.ICAP_COLUMN)
            )
        )
        lines.append(
            "nested anchors: none yet - the geometry is implemented, not "
            "calibrated. Load a nested draw run to train one."
        )

        # --- tubular / NANF / DNANF ------------------------------------
        #
        # The chain is an ingestion concern and the deltas are a display one;
        # neither changes what is fitted. Checked in the bundle because a
        # broken chain does not raise - it produces an absolute pressure built
        # on the wrong base, which then fits perfectly well against nothing.
        import analytic_export as _ax

        ids = [p.id for p in preform_registry.REGISTRY]
        if ids != ["tubular", "nanf", "dnanf"]:
            raise RuntimeError(f"unexpected preform registry: {ids}")
        for legacy, current in (
            ("hc_10cap_nonnested", "tubular"),
            ("hc_nested_3layer", "dnanf"),
        ):
            if preform_registry.get_preform(legacy).id != current:
                raise RuntimeError(f"{legacy} no longer resolves to {current}")
            if schema.schema_for_preform(legacy) is not schema.schema_for_preform(
                current
            ):
                raise RuntimeError(f"{legacy} resolved to a different schema")
        lines.append(
            "preform registry: "
            + ", ".join(
                f"{p.id} ({len(p.target_names)} setpoints, "
                f"{len(p.schema.pressure_chain)} chain link(s))"
                for p in preform_registry.REGISTRY
            )
        )

        # Ingestion chain: raw sequential differentials -> absolutes, then the
        # display chain subtracts them back. The two must agree, or the tower
        # is being told to set a differential the model never saw.
        probe = pd.DataFrame(
            {"core_dP_kPa": [5.0], "outer_dP_kPa": [9.4], "inner_dP_kPa": [4.6],
             "mid_dP_kPa": [3.1]}
        )
        for entry in preform_registry.REGISTRY:
            derived, chain_notes = entry.schema.derive_pressures(probe)
            deltas = entry.schema.compute_deltas(derived.iloc[0].to_dict())
            for link in entry.schema.pressure_chain:
                if link.absolute not in derived.columns:
                    raise RuntimeError(
                        f"{entry.id}: {link.absolute} was not derived"
                    )
            # Each display delta must equal the raw differential it came from.
            for link, (name, value) in zip(entry.schema.pressure_chain, deltas.items()):
                expected = float(probe[link.raw_delta].iloc[0])
                if not np.isclose(value, expected, rtol=0, atol=1e-9):
                    raise RuntimeError(
                        f"{entry.id}: {name} = {value:g} does not round-trip to "
                        f"{link.raw_delta} = {expected:g}"
                    )
            lines.append(
                f"  {entry.id} chain: "
                + " | ".join(link.describe() for link in entry.schema.pressure_chain)
                + "  ->  deltas "
                + ", ".join(f"{k}={v:+g}" for k, v in deltas.items())
            )

        # NANF genuinely skips the middle rather than blanking it.
        nanf_schema = preform_registry.get_preform("nanf").schema
        if schema.MCAP_COLUMN in nanf_schema.setpoint_names:
            raise RuntimeError("NANF should have no middle capillary pressure")
        if any("middle" in name for name in nanf_schema.feature_names):
            raise RuntimeError("NANF should have no middle geometry")

        # The confirmed export format, asserted against the literal header.
        confirmed = (
            "sample,time_utc,feed_speed_mm_min,draw_speed_m_min,tension_g,"
            "furnace_temp_C,fibre_OD_um,fibre_ID_um,cap_OD_um,cap_ID_um,"
            "core_dP_kPa,outer_dP_kPa,atm_P_kPa"
        )
        tubular_columns = _ax.export_columns_for(schema.TUBULAR_SCHEMA)
        if ",".join(tubular_columns) != confirmed:
            raise RuntimeError(
                "the tubular analytic export no longer matches the confirmed "
                "13-column format"
            )
        for entry in preform_registry.REGISTRY:
            columns = _ax.export_columns_for(entry.schema)
            if columns[:13] != tubular_columns:
                raise RuntimeError(f"{entry.id}: export lost the confirmed prefix")
        nanf_columns = _ax.export_columns_for(nanf_schema)
        if any("middle" in c or c == schema.MID_DP_COLUMN for c in nanf_columns):
            raise RuntimeError("the NANF export carries middle-layer columns")
        lines.append(
            "analytic export: tubular = the confirmed 13 columns; "
            f"nanf = {len(nanf_columns)}, dnanf = "
            f"{len(_ax.export_columns_for(schema.DNANF_SCHEMA))} "
            "(multi-layer column names NOT yet confirmed against the "
            "estimator - see analytic_export.NANF_DNANF_FORMAT_IS_UNCONFIRMED)"
        )

        lines.append("RESULT: PASS")
    except Exception as exc:  # noqa: BLE001 - the point is to report it
        lines.append("".join(traceback.format_exception(exc)))
        lines.append("RESULT: FAIL")
        code = 1

    report = "\n".join(lines) + "\n"
    destination = Path(out_path) if out_path else paths.data_dir() / "selftest_report.txt"
    try:
        destination.write_text(report, encoding="utf-8")
    except OSError:
        pass
    if sys.stdout is not None:
        try:
            sys.stdout.write(report)
        except Exception:  # noqa: BLE001 - no console in a windowed build
            pass
    return code


def main() -> int:
    # Required before anything can spawn a worker process in a PyInstaller
    # onefile build. Nothing here parallelises today, but joblib is imported for
    # persistence and reaching a process-based backend without this guard makes
    # the exe re-launch itself, opening windows without end.
    multiprocessing.freeze_support()

    argv = sys.argv[1:]
    if argv and argv[0] == "--selftest":
        rest = argv[1:]
        out_path = None
        if "--out" in rest:
            index = rest.index("--out")
            out_path = rest[index + 1] if index + 1 < len(rest) else None
            rest = rest[:index] + rest[index + 2 :]
        if len(rest) != 2:
            return _selftest("", "", out_path)
        return _selftest(rest[0], rest[1], out_path)

    QApplication.setAttribute(Qt.AA_DontUseNativeMenuBar, False)
    app = QApplication(sys.argv)
    app.setApplicationName(APP_TITLE)
    app.setStyle("Fusion")
    app.setFont(QFont("Segoe UI", 10))

    icon_path = resource_path("app.ico")
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    qss_path = resource_path("style.qss")
    if qss_path.exists():
        app.setStyleSheet(qss_path.read_text(encoding="utf-8"))

    _install_exception_hook(APP_TITLE)

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
