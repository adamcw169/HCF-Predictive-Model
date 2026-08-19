"""Tab 2 - Predict calibrated setpoints for a target geometry.

The analytic estimate is typed in, not computed. That is deliberate and it is
not a placeholder: the fast estimator is not callable from this app, and the
alternative - inferring or defaulting those four numbers - would produce a
prediction that looks identical to a real one while resting on nothing. The
operator runs the estimator themselves and enters what it said; the calibration
corrects that. `LiveEstimatorAnalyticSource` is the seam for changing this
later, and nothing on this tab touches it.
"""

from __future__ import annotations

import csv
import datetime as dt

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QDoubleValidator
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

import calibration as calib
import paths
import preform
import schema
from ui_common import Banner, badge, number_input, subheading

# Target geometry and process inputs, with a sensible starting value and the
# step size an operator would actually nudge them by.
INPUT_FIELDS: tuple[tuple[str, float, float, int, float, float], ...] = (
    # (column, lo, hi, decimals, step, default)
    ("fibre_OD_um", 20.0, 2000.0, 2, 1.0, 183.0),
    ("fibre_ID_um", 1.0, 1900.0, 2, 1.0, 48.0),
    ("cap_OD_um", 0.1, 1000.0, 3, 0.1, 8.6),
    ("cap_ID_um", 0.05, 1000.0, 3, 0.1, 7.7),
    ("tension_g", 0.0, 2000.0, 1, 5.0, 391.0),
    ("feed_speed_mm_min", 0.0, 200.0, 3, 0.1, 5.0),
)

# Step size and starting value per *kind* of measurement. The bounds come from
# the column spec itself, so a new geometry needs no new entry here unless it
# introduces a genuinely new kind of quantity - a capillary diameter behaves
# like a capillary diameter whichever layer it is on.
_FIELD_STYLE: dict[str, tuple[int, float, float]] = {
    # column-name suffix: (decimals, step, default)
    "fibre_OD_um": (2, 1.0, 183.0),
    "fibre_ID_um": (2, 1.0, 48.0),
    "tension_g": (1, 5.0, 391.0),
    "feed_speed_mm_min": (3, 0.1, 5.0),
}
_CAP_OD_STYLE = (3, 0.1, 8.6)
_CAP_ID_STYLE = (3, 0.1, 7.7)


def input_fields_for(
    preform_schema: schema.PreformSchema,
) -> tuple[tuple[str, float, float, int, float, float], ...]:
    """The Predict tab's geometry inputs for one geometry.

    Built from the preform's own feature columns rather than a module constant,
    which is what lets a nested preform show ten inputs where the non-nested
    one shows six without a second hardcoded table. For the non-nested schema
    this reproduces `INPUT_FIELDS` exactly, which the tests assert - the
    existing tab must not shift a bound, a step or a default.
    """
    out = []
    for spec in preform_schema.features:
        if spec.name in _FIELD_STYLE:
            decimals, step, default = _FIELD_STYLE[spec.name]
        elif spec.name.startswith("cap_OD"):
            decimals, step, default = _CAP_OD_STYLE
        elif spec.name.startswith("cap_ID"):
            decimals, step, default = _CAP_ID_STYLE
        else:  # pragma: no cover - a genuinely new kind of column
            decimals, step, default = (3, 0.1, float(spec.lo))
        out.append((spec.name, spec.lo, spec.hi, decimals, step, default))
    return tuple(out)


def _header_of(path) -> list[str]:
    """First row of an existing log, or an empty list if it cannot be read."""
    try:
        with path.open("r", newline="", encoding="utf-8") as handle:
            return next(csv.reader(handle), [])
    except (OSError, UnicodeDecodeError):
        return []


class PredictTab(QWidget):
    """Target in, calibrated setpoints out - with the anchor count attached."""

    def __init__(self, app_version: str = "", parent=None):
        super().__init__(parent)
        self._app_version = app_version
        self._calibration: calib.CalibrationSet | None = None
        self._last_prediction: dict | None = None
        # The selected geometry's columns. Every form and the results grid are
        # built from this, so switching preform rebuilds them rather than
        # needing a second set of hardcoded widgets.
        self._schema = schema.schema_for_preform(preform.DEFAULT_PREFORM_ID)

        self._build_ui()
        self._sync_enabled()

    # ------------------------------------------------------------------ ui

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        self.banner_state = Banner()
        layout.addWidget(self.banner_state)
        layout.addWidget(self._build_preform_group())
        layout.addWidget(self._build_inputs_group())
        layout.addWidget(self._build_analytic_group())

        row = QHBoxLayout()
        self.btn_predict = QPushButton("Run prediction")
        self.btn_predict.setProperty("accent", "true")
        self.btn_predict.clicked.connect(self._on_predict)
        row.addWidget(self.btn_predict)
        self.lbl_gate = QLabel("")
        self.lbl_gate.setProperty("role", "subheading")
        self.lbl_gate.setWordWrap(True)
        row.addWidget(self.lbl_gate, 1)
        layout.addLayout(row)

        layout.addWidget(self._build_results_group())
        layout.addStretch(1)

        scroll.setWidget(body)
        outer.addWidget(scroll)

    def _build_preform_group(self) -> QWidget:
        group = QGroupBox("Preform")
        layout = QVBoxLayout(group)
        self.combo_preform = QComboBox()
        for entry in preform.REGISTRY:
            self.combo_preform.addItem(entry.label, entry.id)
            index = self.combo_preform.count() - 1
            if not entry.is_implemented:
                # Visible so the roadmap is legible, disabled so it cannot be
                # chosen and silently produce nothing.
                model_item = self.combo_preform.model().item(index)
                model_item.setEnabled(False)
                model_item.setToolTip(entry.unavailable_reason)
        layout.addWidget(self.combo_preform)
        self.lbl_preform_note = QLabel("")
        self.lbl_preform_note.setProperty("role", "subheading")
        self.lbl_preform_note.setWordWrap(True)
        layout.addWidget(self.lbl_preform_note)
        self.combo_preform.currentIndexChanged.connect(self._on_preform_changed)
        return group

    @staticmethod
    def _clear_form(form: QFormLayout) -> None:
        """Empty a form layout, deleting the widgets it owned."""
        while form.rowCount():
            form.removeRow(0)

    def _build_inputs_group(self) -> QWidget:
        group = QGroupBox("Target geometry and process inputs")
        layout = QVBoxLayout(group)
        self._inputs_form = QFormLayout()
        self._inputs: dict[str, object] = {}
        self._rebuild_inputs()
        layout.addLayout(self._inputs_form)

        self.row_preform_od = QWidget()
        od_layout = QHBoxLayout(self.row_preform_od)
        od_layout.setContentsMargins(0, 0, 0, 0)
        od_layout.addWidget(QLabel("preform_OD_mm"))
        self.spin_preform_od = number_input(0.0, 1000.0, 3, 1.0, "mm", 0.0)
        self.spin_preform_od.setToolTip(
            "Required only because this calibration was fitted with the "
            "geometric draw-down ratio."
        )
        od_layout.addWidget(self.spin_preform_od)
        od_layout.addStretch(1)
        self.row_preform_od.setVisible(False)
        layout.addWidget(self.row_preform_od)
        return group

    def _rebuild_inputs(self) -> None:
        """Geometry inputs for the selected preform.

        Rebuilt rather than fixed: the nested geometry has ten of these where
        the non-nested one has six, and the registry's whole premise is that a
        new preform needs no new UI code.
        """
        self._clear_form(self._inputs_form)
        self._inputs = {}
        specs = self._schema.spec_by_name
        for name, lo, hi, decimals, step, default in input_fields_for(self._schema):
            spec = specs[name]
            box = number_input(lo, hi, decimals, step, spec.unit, default)
            box.setToolTip(spec.description)
            box.valueChanged.connect(self._on_inputs_changed)
            self._inputs_form.addRow(f"{name}", box)
            self._inputs[name] = box

    def _rebuild_analytic(self) -> None:
        """One manual-entry field per analytic estimate this geometry needs.

        Manual entry is unchanged from v1.0: the supervisor's estimator output
        is typed in, not computed here. The nested geometry simply needs two
        more of them. Whether that estimator actually produces the middle and
        inner values is not yet known - if it does not, these stay as the entry
        path for whatever the operator does have, which is the same thing the
        existing four fields are.
        """
        self._clear_form(self._analytic_form)
        self._analytic_inputs = {}
        for spec in self._schema.analytics:
            field = QLineEdit()
            field.setPlaceholderText(f"required - {spec.unit}")
            # A validator rather than a spin box, so "not entered yet" is a real
            # state. A spin box showing 0.000 looks answered when it is not.
            validator = QDoubleValidator()
            validator.setNotation(QDoubleValidator.StandardNotation)
            field.setValidator(validator)
            field.setProperty("required", "pending")
            field.textChanged.connect(self._on_analytic_changed)
            self._analytic_form.addRow(spec.name, field)
            self._analytic_inputs[spec.name] = field

    def _build_analytic_group(self) -> QWidget:
        group = QGroupBox("Analytic estimate (required)")
        layout = QVBoxLayout(group)
        layout.addWidget(
            subheading(
                "Run the fast estimator for this target and enter its output. "
                "The calibration is a correction applied on top of these "
                "numbers - it does not replace them, and there is no default: "
                "a guessed analytic estimate would produce a prediction "
                "indistinguishable from a real one."
            )
        )
        self._analytic_form = QFormLayout()
        self._analytic_inputs: dict[str, QLineEdit] = {}
        self._rebuild_analytic()
        layout.addLayout(self._analytic_form)
        return group

    def _rebuild_results(self) -> None:
        """The differential pressures first, then the absolutes behind them.

        Differentials lead because they are what a fabricator sets at the
        tower: the number that matters operationally is the step across each
        capillary wall, not the absolute pressure either side of it. They were
        added as a derived footnote when the chain was introduced; this release
        promotes them to the primary output.

        The absolutes stay, below, and are not demoted in substance - they are
        what the model actually fits, they carry the confidence and prediction
        intervals, and a differential is only as trustworthy as the two
        predictions it is a difference of. Removing them would hide the fitted
        quantity behind an arithmetic convenience.

        Each geometry's chain decides the rows: one step for tubular, two for
        NANF (whose inner wall is crossed from the outer capillary, there being
        no middle layer), three for DNANF.
        """
        grid = self._results_grid
        while grid.count():
            item = grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._result_widgets = {}
        self._delta_widgets = {}

        def add_row(row: int, label: str, unit_text: str) -> tuple[QLabel, QLabel]:
            name = QLabel(label)
            name.setProperty("role", "subheading")
            grid.addWidget(name, row * 2, 0)

            value = QLabel("-")
            value.setProperty("role", "metric")
            value.setTextInteractionFlags(Qt.TextSelectableByMouse)
            grid.addWidget(value, row * 2, 1)

            unit = QLabel(unit_text)
            unit.setProperty("role", "metric-unit")
            grid.addWidget(unit, row * 2, 2)

            detail = QLabel("")
            detail.setProperty("role", "subheading")
            detail.setWordWrap(True)
            detail.setTextInteractionFlags(Qt.TextSelectableByMouse)
            grid.addWidget(detail, row * 2 + 1, 0, 1, 4)
            return value, detail

        def add_heading(row: int, text: str, hint: str) -> None:
            heading = QLabel(text)
            heading.setProperty("role", "heading")
            grid.addWidget(heading, row * 2, 0, 1, 2)
            note = QLabel(hint)
            note.setProperty("role", "subheading")
            note.setWordWrap(True)
            grid.addWidget(note, row * 2 + 1, 0, 1, 4)

        row = 0
        add_heading(
            row,
            "Differential pressures",
            "What to set at the tower: the step across each capillary wall. "
            "Derived by subtraction from the fitted absolutes below - nothing "
            "is fitted for these directly, so no interval is quoted on them.",
        )
        row += 1
        for delta in self._schema.deltas:
            value, detail = add_row(row, delta.name, delta.unit)
            self._delta_widgets[delta.name] = (value, detail)
            row += 1

        add_heading(
            row,
            "Absolute setpoints (reference)",
            "What the calibration actually fits, with its intervals. A "
            "differential above is only as well known as the two predictions "
            "it is the difference of.",
        )
        row += 1
        for spec in self._schema.setpoints:
            self._result_widgets[spec.name] = add_row(row, spec.name, spec.unit)
            row += 1

        grid.setColumnStretch(3, 1)

    def _build_results_group(self) -> QWidget:
        group = QGroupBox("Predicted setpoints")
        layout = QVBoxLayout(group)

        header = QHBoxLayout()
        self.lbl_anchor_badge = badge("No calibration loaded")
        header.addWidget(self.lbl_anchor_badge)
        self.lbl_calibration_note = QLabel("")
        self.lbl_calibration_note.setProperty("role", "subheading")
        self.lbl_calibration_note.setWordWrap(True)
        header.addWidget(self.lbl_calibration_note, 1)
        layout.addLayout(header)

        self._results_grid = QGridLayout()
        self._results_grid.setVerticalSpacing(10)
        self._result_widgets: dict[str, tuple[QLabel, QLabel]] = {}
        self._delta_widgets: dict[str, tuple[QLabel, QLabel]] = {}
        self._rebuild_results()
        layout.addLayout(self._results_grid)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("color: #DFE3E8;")
        layout.addWidget(line)

        self.banner_result = Banner()
        layout.addWidget(self.banner_result)

        row = QHBoxLayout()
        self.lbl_log = QLabel("")
        self.lbl_log.setProperty("role", "mono")
        self.lbl_log.setWordWrap(True)
        row.addWidget(self.lbl_log, 1)
        layout.addLayout(row)
        return group

    # -------------------------------------------------------------- state

    def set_calibration(self, result: calib.CalibrationSet | None) -> None:
        self._calibration = result
        if result is None:
            self.banner_state.show_message(
                "No calibration has been saved for this preform yet. Extract "
                "anchor blocks from a draw run on the 'Extract & calibrate' "
                "tab and save a calibration first.",
                "warn",
            )
            self.lbl_anchor_badge.setText("No calibration loaded")
            self.lbl_calibration_note.setText("")
            self.row_preform_od.setVisible(False)
        else:
            counts = result.anchor_counts
            thin = min(counts.values(), default=0) < 6
            self.banner_state.show_message(
                result.describe()
                + ". Anchors per channel: "
                + ", ".join(f"{name} {n}" for name, n in counts.items())
                + (
                    ". Each channel has its own steady-state blocks, so they do "
                    "not rest on the same evidence."
                    if len(set(counts.values())) > 1
                    else "."
                )
                + (
                    " Fewer than six anchors on at least one channel: its "
                    "interval below is wide, and its correction should be read "
                    "as approximate."
                    if thin
                    else ""
                ),
                "warn" if thin else "ok",
            )
            self.lbl_anchor_badge.setText(result.anchor_label)
            self.lbl_anchor_badge.setProperty(
                "role", "badge-thin" if thin else "badge"
            )
            self.lbl_anchor_badge.style().unpolish(self.lbl_anchor_badge)
            self.lbl_anchor_badge.style().polish(self.lbl_anchor_badge)
            shapes = "; ".join(
                f"{name} ({fit.order_summary})"
                for name, fit in result.channels.items()
            )
            self.lbl_calibration_note.setText(
                "Terms: "
                + ", ".join(calib.TERM_BY_KEY[t].label for t in result.terms)
                + f". Shapes per channel: {shapes}."
            )
            geometric = result.drawdown_mode == calib.DRAWDOWN_GEOMETRIC
            self.row_preform_od.setVisible(geometric)
            if geometric and result.preform_OD_mm:
                self.spin_preform_od.setValue(float(result.preform_OD_mm))
        self._clear_results()
        self._sync_enabled()

    def _on_preform_changed(self) -> None:
        # `currentData()` is None when the combo has no current item - a
        # `findData` miss setting index -1, which is reachable by asking for a
        # preform id this build does not have. Ignore rather than raising out
        # of a signal handler, where the traceback would be swallowed and the
        # tab left half-rebuilt.
        current = self.combo_preform.currentData()
        if current is None:
            return
        entry = preform.get_preform(current)
        self.lbl_preform_note.setText(
            "" if entry.is_implemented else entry.unavailable_reason
        )
        # Rebuild every form from the newly selected geometry. This is the
        # registry's original promise finally made true: before v1.9 the
        # selector only toggled an availability note, and the inputs, analytic
        # fields and results grid stayed on the non-nested columns whatever was
        # picked.
        new_schema = entry.schema or schema.schema_for_preform(entry.id)
        if new_schema is not self._schema:
            self._schema = new_schema
            self._rebuild_inputs()
            self._rebuild_analytic()
            self._rebuild_results()
            # A calibration fitted for a different geometry cannot describe
            # this one, so it is dropped rather than left on screen looking
            # applicable.
            if (
                self._calibration is not None
                and self._calibration.preform_id != entry.id
            ):
                self._calibration = None
                self.lbl_anchor_badge.setText("No calibration loaded")
                self.lbl_calibration_note.setText("")
                self.banner_state.show_message(
                    f"No calibration has been saved for {entry.label} yet. "
                    "Extract anchor blocks from a draw run on the "
                    "'Extract & calibrate' tab and save a calibration first.",
                    "warn",
                )
            self._clear_results()
        self._sync_enabled()

    def _on_inputs_changed(self, *_args) -> None:
        self._sync_enabled()

    def _on_analytic_changed(self, *_args) -> None:
        for field in self._analytic_inputs.values():
            filled = self._parse(field) is not None
            field.setProperty("required", "ok" if filled else "pending")
            field.style().unpolish(field)
            field.style().polish(field)
        self._sync_enabled()

    @staticmethod
    def _parse(field: QLineEdit) -> float | None:
        text = field.text().strip()
        if not text:
            return None
        try:
            value = float(text)
        except ValueError:
            return None
        return value if np.isfinite(value) else None

    def _missing_analytic(self) -> list[str]:
        return [
            name
            for name, field in self._analytic_inputs.items()
            if self._parse(field) is None
        ]

    def _sync_enabled(self) -> None:
        entry = preform.get_preform(self.combo_preform.currentData())
        missing = self._missing_analytic()
        reasons: list[str] = []
        if not entry.is_implemented:
            reasons.append("this preform is not implemented")
        if self._calibration is None:
            reasons.append("no calibration is loaded")
        if missing:
            reasons.append(
                f"{len(missing)} analytic value(s) still to enter: "
                + ", ".join(missing)
            )
        self.btn_predict.setEnabled(not reasons)
        self.lbl_gate.setText(
            "" if not reasons else "Waiting on: " + "; ".join(reasons) + "."
        )

    def _clear_results(self) -> None:
        for value, detail in self._result_widgets.values():
            value.setText("-")
            detail.setText("")
        for value_label, detail_label in self._delta_widgets.values():
            value_label.setText("-")
            detail_label.setText("")
        self.banner_result.clear_message()
        self.lbl_log.setText("")
        self._last_prediction = None

    # ---------------------------------------------------------- predicting

    def _collect_inputs(self) -> dict:
        values = {name: float(box.value()) for name, box in self._inputs.items()}
        for name, field in self._analytic_inputs.items():
            values[name] = self._parse(field)
        preform_od = float(self.spin_preform_od.value())
        if preform_od > 0:
            values["preform_OD_mm"] = preform_od
        return values

    def _on_predict(self) -> None:
        if self._calibration is None:
            return
        inputs = self._collect_inputs()
        try:
            predictions = self._calibration.predict(inputs)
        except calib.CalibrationError as exc:
            self.banner_result.show_message(str(exc), "error")
            return

        warnings: list[str] = []
        for channel, prediction in predictions.items():
            value, detail = self._result_widgets[channel]
            # Plain significant figures, no thousands separators: a furnace
            # setpoint of "1,997" reads as two numbers at a glance.
            value.setText(f"{prediction.value:.6g}")
            # A ratio channel says what it actually did - multiplied - rather
            # than only reporting the difference that multiplication produced.
            applied = (
                f"analytic {prediction.analytic:.6g} x "
                f"{prediction.factor:.5g}"
                if prediction.form == calib.FORM_RATIO
                else (
                    f"analytic {prediction.analytic:.6g} "
                    f"{'+' if prediction.correction >= 0 else '-'} "
                    f"{abs(prediction.correction):.5g} correction"
                )
            )
            detail.setText(
                f"{applied}  ·  "
                f"{prediction.confidence:.0%} CI on the correction "
                f"[{prediction.ci_lo:.6g}, {prediction.ci_hi:.6g}]  ·  "
                f"interval for a new draw "
                f"[{prediction.pi_lo:.6g}, {prediction.pi_hi:.6g}]  ·  "
                f"{self._calibration.anchor_label_for(channel)}"
            )
            spec = schema.SPEC_BY_NAME[channel]
            if not spec.in_range(np.array([prediction.value])).item():
                warnings.append(
                    f"{channel} = {prediction.value:.6g} {spec.unit} is outside "
                    f"the physical sanity range {spec.lo:g}-{spec.hi:g}."
                )

        # Derived from the pressures just displayed, whatever produced them.
        # No interval is quoted on any of them: the operands are correlated
        # predictions sharing an anchor set, and combining their intervals as
        # though independent would report a confidence this app has not earned.
        # The values are exact arithmetic on the numbers above; the uncertainty
        # behind each is its own channel's, shown on that channel's own line.
        #
        # Not clamped and not hidden when negative. A step that comes out the
        # wrong way round is either a real physical surprise or a bad estimate,
        # and both are things the operator needs to see - the same reason this
        # app surfaces a negative analytic pressure rather than suppressing it.
        predicted_values = {c: p.value for c, p in predictions.items()}
        deltas: dict[str, float] = {}
        for spec in self._schema.deltas:
            widgets = self._delta_widgets.get(spec.name)
            if widgets is None:
                continue
            value_label, detail_label = widgets
            if spec.inner not in predictions or spec.outer not in predictions:
                deltas[spec.name] = float("nan")
                value_label.setText("-")
                detail_label.setText(
                    f"Needs both {spec.inner} and {spec.outer} to be calibrated."
                )
                continue
            value = spec.compute(predicted_values)
            deltas[spec.name] = value
            value_label.setText(f"{value:.6g}")
            detail_label.setText(
                f"{spec.inner} {predicted_values[spec.inner]:.6g} - "
                f"{spec.outer} {predicted_values[spec.outer]:.6g}"
                "  ·  derived, not fitted: "
                f"{spec.description[0].lower()}{spec.description[1:].rstrip('.')}, "
                "computed from the two predictions above. See each channel's "
                "own interval for how well it is known."
            )
        # The single-delta geometry keeps the attribute the log and the tests
        # already know by name.
        # (the single-delta geometry keeps `delta_P` reachable by name above)

        features = self._calibration.features_for(inputs)
        # Per-channel anchor sets cover different ranges, so extrapolation is a
        # per-channel question now: a target can sit inside one channel's blocks
        # and outside another's.
        extrapolation: list[str] = []
        for channel in predictions:
            for message in self._extrapolation_warnings(
                features, self._calibration.anchors_for(channel)
            ):
                text = f"{channel}: {message}"
                if text not in extrapolation:
                    extrapolation.append(text)

        message = (
            f"Predicted from {self._calibration.anchor_label.lower()}. "
            "Each number is the analytic estimate you entered plus a fitted "
            "correction; the intervals are the calibration's, not the "
            "estimator's."
        )
        kind = "ok"
        if warnings or extrapolation:
            kind = "warn"
            message += " " + " ".join(warnings + extrapolation)
        self.banner_result.show_message(message, kind)

        record = self._build_log_record(inputs, features, predictions, deltas)
        self._last_prediction = record
        self._log_prediction(record)

    def _extrapolation_warnings(self, features: dict, anchors) -> list[str]:
        """Say so when a target sits outside the range the anchors covered.

        With eight anchors this matters more than usual: there is no density of
        points to fall back on, so a target just outside the anchor range is
        genuinely an extrapolation of a straight line rather than a small step.
        """
        out: list[str] = []
        for key, label in (
            (calib.TERM_WALL.key, "capillary wall ratio"),
            ("drawdown_ratio", "draw-down ratio"),
        ):
            if key not in anchors.columns:
                continue
            value = features.get(key)
            column = anchors[key].to_numpy(dtype=float)
            column = column[np.isfinite(column)]
            if value is None or not np.isfinite(value) or column.size == 0:
                continue
            lo, hi = float(column.min()), float(column.max())
            if value < lo or value > hi:
                out.append(
                    f"The target's {label} ({value:.5g}) sits outside the "
                    f"anchor range {lo:.5g} to {hi:.5g}, so the correction is "
                    "extrapolated."
                )
        return out

    def _build_log_record(
        self,
        inputs: dict,
        features: dict,
        predictions: dict,
        deltas: dict[str, float] | None = None,
    ) -> dict:
        deltas = deltas or {}
        calibration = self._calibration
        record: dict = {
            "logged_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "app_version": self._app_version,
            "preform_id": calibration.preform_id,
            "calibration_created_utc": calibration.created_utc,
            "n_anchor": calibration.n_anchor,
            "n_anchor_per_channel": "|".join(
                f"{name}:{count}" for name, count in calibration.anchor_counts.items()
            ),
            "orders_per_channel": "|".join(
                f"{name}:{fit.order_summary}"
                for name, fit in calibration.channels.items()
            ),
            "terms": "|".join(calibration.terms),
            "drawdown_mode": calibration.drawdown_mode,
            "confidence": 1.0 - calibration.alpha,
        }
        for name, _lo, _hi, _d, _s, _default in input_fields_for(self._schema):
            record[name] = inputs[name]
        # Always present, blank when unused, so the log has one stable header
        # rather than a shape that depends on which calibration was loaded.
        record["preform_OD_mm"] = inputs.get("preform_OD_mm", "")
        # Every wall ratio this geometry defines - one for the non-nested
        # preform, still under the `cap_wall_ratio` header the existing log
        # uses, three for the nested one.
        for ratio_name in self._schema.wall_ratio_names:
            record[ratio_name] = features.get(ratio_name)
        record["drawdown_ratio"] = features.get("drawdown_ratio")
        for channel, prediction in predictions.items():
            record[schema.analytic_name(channel)] = prediction.analytic
            record[f"predicted_{channel}"] = prediction.value
            record[f"predicted_{channel}_ci_lo"] = prediction.ci_lo
            record[f"predicted_{channel}_ci_hi"] = prediction.ci_hi
            record[f"predicted_{channel}_pi_lo"] = prediction.pi_lo
            record[f"predicted_{channel}_pi_hi"] = prediction.pi_hi
        # Logged as well as shown, so a row of the log reconstructs the screen.
        # `_log_prediction` rotates the file when the header changes, so extra
        # columns cannot misalign an existing log. The non-nested geometry
        # writes exactly one, still headed `predicted_delta_P`.
        for spec in self._schema.deltas:
            record[f"predicted_{spec.name}"] = deltas.get(spec.name, float("nan"))
        return record

    def _log_prediction(self, record: dict) -> None:
        """Append the prediction to a local CSV.

        Every future anchor comes from a real draw, and a real draw starts with
        a prediction. Logging what was asked for and what was answered is what
        makes it possible to come back later and add the result as a ninth
        anchor.
        """
        path = paths.prediction_log_path()
        fields = list(record)
        try:
            exists = path.exists() and path.stat().st_size > 0
            if exists and _header_of(path) != fields:
                # An older log with a different shape would silently misalign
                # every appended row. Move it aside rather than corrupt it.
                stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
                path.replace(path.with_name(f"{path.stem}_pre_{stamp}.csv"))
                exists = False
            with path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                if not exists:
                    writer.writeheader()
                writer.writerow(record)
        except OSError as exc:
            QMessageBox.warning(
                self,
                "Prediction not logged",
                f"The prediction is shown above but could not be written to "
                f"{path}:\n\n{exc}",
            )
            return
        self.lbl_log.setText(f"Logged to {path}")
