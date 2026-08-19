"""Column definitions for hollow-core fiber draw data.

Deliberately file-compatible with the HCF Draw Predictor's schema: the same
column names, units and meanings, so a CSV written for one app is readable by
the other. The code is a separate implementation - nothing is imported across
the two projects - but the names below must not drift from that app's
``schema.py`` or the file compatibility is lost.

This app consumes two kinds of file:

* a **raw 1 Hz timeseries** (``hcf_timeseries_1s.csv`` shape) - one row per
  sample of a live draw, with QC flag columns. Steady-state blocks are
  extracted from this.
* an **analytic estimate file** (``model_ready_data_ready2.csv`` shape) - rows
  carrying the fast-estimator's ``analytic_*`` columns and a timestamp, matched
  to blocks by time window.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ColumnSpec:
    """One column in the data schema.

    ``lo``/``hi`` are generous physical sanity bounds. They exist to catch unit
    mix-ups (mm vs um, bar vs kPa) and typos, not to enforce a process window.
    """

    name: str
    unit: str
    description: str
    lo: float
    hi: float
    # Sensor noise around a true zero can read slightly negative. For those
    # channels a small negative excursion is normal rather than suspicious.
    neg_tolerance: float = 0.0

    def in_range(self, values: pd.Series) -> pd.Series:
        """Boolean mask of values that fall inside the sanity bounds."""
        return (values >= self.lo - self.neg_tolerance) & (values <= self.hi)


# Geometry and process inputs the operator specifies for a target draw. These
# are also measured per steady-state block, which is what makes a block usable
# as a calibration anchor.
REQUIRED_FEATURE_COLUMNS: tuple[ColumnSpec, ...] = (
    ColumnSpec("fibre_OD_um", "um", "Target fiber outer diameter", 20.0, 2000.0),
    ColumnSpec("fibre_ID_um", "um", "Target fiber inner (core) diameter", 1.0, 1900.0),
    ColumnSpec("cap_OD_um", "um", "Target capillary outer diameter", 0.1, 1000.0),
    ColumnSpec("cap_ID_um", "um", "Target capillary inner diameter", 0.05, 1000.0),
    ColumnSpec("tension_g", "g", "Target draw tension", 0.0, 2000.0),
    ColumnSpec("feed_speed_mm_min", "mm/min", "Preform feed speed", 0.0, 200.0),
)

# Machine setpoints. These are what the calibration corrects and what the
# Predict tab reports.
SETPOINT_COLUMNS: tuple[ColumnSpec, ...] = (
    ColumnSpec(
        "furnace_temp_C", "degC", "Furnace temperature setpoint", 1500.0, 2200.0
    ),
    ColumnSpec("draw_speed_m_min", "m/min", "Draw (capstan) speed", 0.0, 500.0),
    ColumnSpec(
        "core_dP_kPa", "kPa (gauge)", "Core pressure", 0.0, 500.0, neg_tolerance=1.0
    ),
    ColumnSpec(
        "Pocap_kPa",
        "kPa (gauge)",
        "Capillary pressure (absolute gauge, not the raw differential)",
        0.0,
        500.0,
        neg_tolerance=1.0,
    ),
)

ANALYTIC_PREFIX = "analytic_"

# Analytic (fast physics estimator) columns mirror the setpoints one-for-one.
# The analytic estimate is unconstrained by the physical sanity bounds of the
# real setpoint it estimates - the fast estimator legitimately returns, for
# example, a negative capillary pressure - so the bounds are widened here
# rather than copied. A wild analytic value is information about the estimator,
# not a reason to reject the operator's file.
ANALYTIC_COLUMNS: tuple[ColumnSpec, ...] = tuple(
    ColumnSpec(
        name=f"{ANALYTIC_PREFIX}{spec.name}",
        unit=spec.unit,
        description=f"Analytic (fast physics model) estimate of {spec.name}",
        lo=-1.0e4,
        hi=1.0e4,
    )
    for spec in SETPOINT_COLUMNS
)

# Raw differential that Pocap_kPa is derived from when it is not supplied.
CORE_DP_COLUMN = "core_dP_kPa"
OUTER_DP_COLUMN = "outer_dP_kPa"
POCAP_COLUMN = "Pocap_kPa"

# Ambient pressure, carried through block statistics for reference. Not a
# model input.
ATM_P_COLUMN = "atm_P_kPa"

TIME_COLUMN = "time_utc"

SETPOINT_NAMES: tuple[str, ...] = tuple(c.name for c in SETPOINT_COLUMNS)
ANALYTIC_NAMES: tuple[str, ...] = tuple(c.name for c in ANALYTIC_COLUMNS)
REQUIRED_FEATURE_NAMES: tuple[str, ...] = tuple(
    c.name for c in REQUIRED_FEATURE_COLUMNS
)

SPEC_BY_NAME: dict[str, ColumnSpec] = {
    spec.name: spec
    for spec in (*REQUIRED_FEATURE_COLUMNS, *SETPOINT_COLUMNS, *ANALYTIC_COLUMNS)
}

# Channels summarised per steady-state block. Order is the display order in the
# block table. Everything the calibration needs is in here.
BLOCK_VALUE_COLUMNS: tuple[str, ...] = (
    "feed_speed_mm_min",
    "draw_speed_m_min",
    "tension_g",
    "furnace_temp_C",
    "fibre_OD_um",
    "fibre_ID_um",
    "cap_OD_um",
    "cap_ID_um",
    CORE_DP_COLUMN,
    OUTER_DP_COLUMN,
    POCAP_COLUMN,
    ATM_P_COLUMN,
)


# Raw block medians carried onto every anchor row. The calibration is fitted on
# dimensionless engineered features, but the anchor table is also the saved
# record of what a calibration rests on - written beside it as plain CSV - and
# that record is far more useful with the unprocessed geometry in it than with
# only the ratios derived from it.
ANCHOR_RAW_COLUMNS: tuple[str, ...] = (
    "fibre_OD_um",
    "fibre_ID_um",
    "cap_OD_um",
    "cap_ID_um",
    "tension_g",
    "feed_speed_mm_min",
    "furnace_temp_C",
    "draw_speed_m_min",
    CORE_DP_COLUMN,
    OUTER_DP_COLUMN,
)


def analytic_name(setpoint: str) -> str:
    """Analytic column name corresponding to a setpoint column."""
    return f"{ANALYTIC_PREFIX}{setpoint}"


def unit_of(column: str) -> str:
    spec = SPEC_BY_NAME.get(column)
    return spec.unit if spec is not None else ""


# --------------------------------------------------------------------------
# Derived quantities
# --------------------------------------------------------------------------


def add_pocap(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Ensure ``Pocap_kPa`` exists, deriving it from the differential if needed.

    Returns the frame and a note describing what was done, so the UI can say so
    rather than quietly inventing a column.
    """
    out = df.copy()
    if POCAP_COLUMN in out.columns and out[POCAP_COLUMN].notna().any():
        return out, f"{POCAP_COLUMN} read directly from the file."
    if CORE_DP_COLUMN in out.columns and OUTER_DP_COLUMN in out.columns:
        out[POCAP_COLUMN] = pd.to_numeric(
            out[CORE_DP_COLUMN], errors="coerce"
        ) + pd.to_numeric(out[OUTER_DP_COLUMN], errors="coerce")
        return out, (
            f"{POCAP_COLUMN} derived as {CORE_DP_COLUMN} + {OUTER_DP_COLUMN} "
            "(the file carried only the differential)."
        )
    return out, (
        f"{POCAP_COLUMN} is absent and cannot be derived: the file has neither "
        f"{POCAP_COLUMN} nor both of {CORE_DP_COLUMN} and {OUTER_DP_COLUMN}."
    )


# The pressure difference across the capillary wall: capillary minus core.
#
# Reported alongside the predicted setpoints because it is the quantity that
# physically inflates or collapses a capillary, and an operator reading two
# pressures off a screen should not have to subtract them in their head. It is
# *derived*, never fitted: nothing predicts delta_P, and adding it changed
# nothing about what the calibration does. Both operands are gauge pressures
# against the same reference, so the difference is well defined even though
# neither absolute value is.
DELTA_P_NAME = "delta_P"
DELTA_P_UNIT = "kPa"


def delta_p(core_dP_kPa: float, pocap_kPa: float) -> float:
    """``Pocap_kPa - core_dP_kPa``. Arithmetic, not a model."""
    core = float(core_dP_kPa)
    cap = float(pocap_kPa)
    if not np.isfinite(core) or not np.isfinite(cap):
        return float("nan")
    return cap - core


def cap_wall_ratio(cap_od_um: float, cap_id_um: float) -> float:
    """Capillary wall thickness as a fraction of its outer diameter.

    Dimensionless, and the single geometric ratio that most strongly governs
    how far the analytic capillary pressure sits from reality.
    """
    cap_od = float(cap_od_um)
    if not np.isfinite(cap_od) or cap_od == 0.0:
        return float("nan")
    return (cap_od - float(cap_id_um)) / cap_od


def kinematic_drawdown_ratio(
    draw_speed_m_min: float, feed_speed_mm_min: float
) -> float:
    """Velocity draw-down ratio, dimensionless.

    ``draw_speed`` is m/min and ``feed_speed`` is mm/min, so the 1000 converts
    the numerator. By mass conservation this equals the cross-sectional area
    reduction, which is the quantity the physics actually cares about.

    The draw speed used here must be the *analytic* one, never the measured
    one: at prediction time the measured draw speed does not exist yet, it is
    one of the things being predicted. Using the analytic value keeps the same
    feature computable during calibration and during prediction.
    """
    feed = float(feed_speed_mm_min)
    if not np.isfinite(feed) or feed == 0.0:
        return float("nan")
    return float(draw_speed_m_min) * 1000.0 / feed


def geometric_drawdown_ratio(preform_OD_mm: float, fibre_OD_um: float) -> float:
    """Diameter draw-down ratio from preform OD to fiber OD, dimensionless.

    Preferred over the kinematic ratio when the preform outer diameter is
    known, because it is a property of the geometry alone and does not inherit
    any error in the analytic draw speed. Returns NaN when the preform OD has
    not been supplied.
    """
    od_mm = float(preform_OD_mm)
    fibre_od = float(fibre_OD_um)
    if not np.isfinite(od_mm) or od_mm <= 0.0:
        return float("nan")
    if not np.isfinite(fibre_od) or fibre_od <= 0.0:
        return float("nan")
    return od_mm * 1000.0 / fibre_od


# --------------------------------------------------------------------------
# File checks
# --------------------------------------------------------------------------


def missing_columns(df: pd.DataFrame, required: tuple[str, ...]) -> list[str]:
    return [name for name in required if name not in df.columns]


def out_of_range_counts(df: pd.DataFrame) -> dict[str, int]:
    """Count values outside the sanity bounds, per known column present."""
    counts: dict[str, int] = {}
    for name, spec in SPEC_BY_NAME.items():
        if name not in df.columns:
            continue
        values = pd.to_numeric(df[name], errors="coerce")
        bad = int((~spec.in_range(values) & values.notna()).sum())
        if bad:
            counts[name] = bad
    return counts


# ==========================================================================
# Per-preform schemas (v1.9)
# ==========================================================================
#
# Everything above this line is the non-nested preform's schema, and it is
# deliberately untouched: it is what every stored calibration was fitted
# against, and the module-level names are what the whole codebase imports.
# The nested preform does not modify any of it - it defines its own columns
# below and both are bundled into `PreformSchema` objects, so downstream code
# can ask "what channels does *this* preform have" instead of reading a global.
#
# Why a bundle rather than more module constants
# ----------------------------------------------
# Before v1.9 the four setpoint channels were a global fact. Adding a second
# geometry makes them a property of *which preform is loaded*, and the
# distinction has to live somewhere a caller can pass around. A bundle keeps
# the related pieces - features, setpoints, analytics, block columns, the
# geometric ratios and the pressure chain - together and consistent, so a
# caller cannot pick up one preform's setpoints alongside another's ratios.
#
# Every consumer takes this as an optional argument defaulting to
# `NONNESTED_SCHEMA`, which is why the existing preform's numbers cannot move:
# an unchanged call site sees exactly the constants it always did.


# --- nested geometry: three concentric capillary layers -------------------
#
# Bounds mirror the single-capillary specs above: the layers are the same kind
# of measurement on the same instrument, so they get the same sanity range. The
# generous range exists to catch unit mix-ups (mm vs um), not to encode a
# process window, and that reasoning does not change per layer.

NESTED_FEATURE_COLUMNS: tuple[ColumnSpec, ...] = (
    ColumnSpec("fibre_OD_um", "um", "Target fiber outer diameter", 20.0, 2000.0),
    ColumnSpec("fibre_ID_um", "um", "Target fiber inner (core) diameter", 1.0, 1900.0),
    ColumnSpec(
        "cap_OD_outer_um", "um", "Outer-layer capillary outer diameter", 0.1, 1000.0
    ),
    ColumnSpec(
        "cap_ID_outer_um", "um", "Outer-layer capillary inner diameter", 0.05, 1000.0
    ),
    ColumnSpec(
        "cap_OD_middle_um", "um", "Middle-layer capillary outer diameter", 0.1, 1000.0
    ),
    ColumnSpec(
        "cap_ID_middle_um", "um", "Middle-layer capillary inner diameter", 0.05, 1000.0
    ),
    ColumnSpec(
        "cap_OD_inner_um", "um", "Inner-layer capillary outer diameter", 0.1, 1000.0
    ),
    ColumnSpec(
        "cap_ID_inner_um", "um", "Inner-layer capillary inner diameter", 0.05, 1000.0
    ),
    ColumnSpec("tension_g", "g", "Target draw tension", 0.0, 2000.0),
    ColumnSpec("feed_speed_mm_min", "mm/min", "Preform feed speed", 0.0, 200.0),
)

# The three capillary pressures. Each is a gauge pressure with the same
# arbitrary zero as the existing pair, so each carries the same negative
# tolerance and the same refusal of the ratio correction form.
MCAP_COLUMN = "Pmcap_kPa"
ICAP_COLUMN = "Picap_kPa"

NESTED_SETPOINT_COLUMNS: tuple[ColumnSpec, ...] = (
    ColumnSpec(
        "furnace_temp_C", "degC", "Furnace temperature setpoint", 1500.0, 2200.0
    ),
    ColumnSpec("draw_speed_m_min", "m/min", "Draw (capstan) speed", 0.0, 500.0),
    ColumnSpec(
        CORE_DP_COLUMN, "kPa (gauge)", "Core pressure", 0.0, 500.0, neg_tolerance=1.0
    ),
    ColumnSpec(
        POCAP_COLUMN,
        "kPa (gauge)",
        "Outer capillary pressure",
        0.0,
        500.0,
        neg_tolerance=1.0,
    ),
    ColumnSpec(
        MCAP_COLUMN,
        "kPa (gauge)",
        "Middle capillary pressure",
        0.0,
        500.0,
        neg_tolerance=1.0,
    ),
    ColumnSpec(
        ICAP_COLUMN,
        "kPa (gauge)",
        "Inner capillary pressure",
        0.0,
        500.0,
        neg_tolerance=1.0,
    ),
)

NESTED_ANALYTIC_COLUMNS: tuple[ColumnSpec, ...] = tuple(
    ColumnSpec(
        name=f"{ANALYTIC_PREFIX}{spec.name}",
        unit=spec.unit,
        description=f"Analytic (fast physics model) estimate of {spec.name}",
        lo=-1.0e4,
        hi=1.0e4,
    )
    for spec in NESTED_SETPOINT_COLUMNS
)


# --- geometric ratios, per capillary layer --------------------------------
#
# `cap_wall_ratio` above is this same arithmetic on the single-capillary
# preform's one layer. It stays exactly where it is and keeps its name: it is
# the feature every existing calibration was fitted against, and renaming it
# would invalidate stored coefficients for no benefit.


@dataclass(frozen=True)
class WallRatioSpec:
    """One dimensionless wall-thickness ratio, and the columns it comes from."""

    name: str
    od_column: str
    id_column: str
    label: str

    def compute(self, od_um: float, id_um: float) -> float:
        return cap_wall_ratio(od_um, id_um)


NONNESTED_WALL_RATIOS: tuple[WallRatioSpec, ...] = (
    WallRatioSpec("cap_wall_ratio", "cap_OD_um", "cap_ID_um", "Capillary wall ratio"),
)

NESTED_WALL_RATIOS: tuple[WallRatioSpec, ...] = (
    WallRatioSpec(
        "outer_wall_ratio",
        "cap_OD_outer_um",
        "cap_ID_outer_um",
        "Outer capillary wall ratio",
    ),
    WallRatioSpec(
        "middle_wall_ratio",
        "cap_OD_middle_um",
        "cap_ID_middle_um",
        "Middle capillary wall ratio",
    ),
    WallRatioSpec(
        "inner_wall_ratio",
        "cap_OD_inner_um",
        "cap_ID_inner_um",
        "Inner capillary wall ratio",
    ),
)


# --- the pressure chain ---------------------------------------------------
#
# Each delta is the step across one capillary wall: the pressure inside that
# layer minus the pressure of whatever is immediately outside it. On the
# non-nested preform there is one such step (`Pocap - core_dP`), which is the
# existing `delta_P` under a name that says which wall it crosses. On the
# nested preform there are three, chained outward-in.
#
# Derived, never fitted - the same status `delta_P` has had since v1.8. And
# deliberately not clamped: a negative step is either a real physical surprise
# or a bad estimate, and both are things the operator needs to see. The app
# already surfaces negative analytic pressures rather than hiding them, and
# this follows that.
#
# No interval is quoted on any of them, for the reason `delta_P` quotes none:
# the operands are correlated predictions sharing an anchor set, and combining
# their intervals as though independent would report a confidence not earned.


@dataclass(frozen=True)
class DeltaSpec:
    """One pressure step across a capillary wall: `inner - outer`."""

    name: str
    inner: str  # the pressure inside this wall
    outer: str  # the pressure immediately outside it
    description: str

    @property
    def unit(self) -> str:
        return DELTA_P_UNIT

    def compute(self, values: "Mapping[str, float]") -> float:
        """`inner - outer`, NaN if either operand is missing or not finite."""
        return delta_p(values.get(self.outer, float("nan")),
                       values.get(self.inner, float("nan")))


# One step. Named `delta_P` because that is what it has been called since
# v1.8 and what the prediction log column is; the nested names below are the
# generalised form of the same idea.
NONNESTED_DELTAS: tuple[DeltaSpec, ...] = (
    DeltaSpec(
        DELTA_P_NAME,
        inner=POCAP_COLUMN,
        outer=CORE_DP_COLUMN,
        description="Across the capillary wall: outer capillary minus core.",
    ),
)

NESTED_DELTAS: tuple[DeltaSpec, ...] = (
    DeltaSpec(
        "deltaPocap",
        inner=POCAP_COLUMN,
        outer=CORE_DP_COLUMN,
        description="Across the outer capillary wall: outer capillary minus core.",
    ),
    DeltaSpec(
        "deltaPmcap",
        inner=MCAP_COLUMN,
        outer=POCAP_COLUMN,
        description="Across the middle wall: middle capillary minus outer capillary.",
    ),
    DeltaSpec(
        "deltaPicap",
        inner=ICAP_COLUMN,
        outer=MCAP_COLUMN,
        description="Across the inner wall: inner capillary minus middle capillary.",
    ),
)


# ==========================================================================
# The ingestion pressure chain
# ==========================================================================
#
# Per the supervisor: `core_dP_kPa` is the only channel a draw tower logs as a
# true absolute pressure. Every capillary layer's raw channel is a *sequential
# differential* against the layer immediately outside it. So the file gives:
#
#     core_dP_kPa    absolute
#     outer_dP_kPa   Pocap - Pcore
#     mid_dP_kPa     Pmcap - Pocap      (DNANF only)
#     inner_dP_kPa   Picap - (the layer outside it)
#
# and the absolute pressures the model is fitted on have to be reconstructed by
# walking outward-in, each link adding its raw differential to the absolute
# already established for the layer before it.
#
# This is exactly what `add_pocap` has always done for the single-capillary
# case (`Pocap = core_dP_kPa + outer_dP_kPa`); it is generalised here to a
# chain of arbitrary length, chosen by the active preform. NANF's chain skips
# the middle link entirely: its `inner_dP_kPa` is measured against the *outer*
# layer, because there is no middle layer for it to be measured against.
#
# What is *fitted* does not change. Absolute pressures remain the native
# target for every layer, as they have been for nine releases - the branching,
# the order-selection tolerances and the held-out framework were all validated
# against them, and switching the fitted target would invalidate that
# validation to gain arithmetic that a prediction can do afterwards anyway.
# This chain is an *ingestion* concern: it turns what the tower logged into
# what the model has always been fitted on.

MID_DP_COLUMN = "mid_dP_kPa"
INNER_DP_COLUMN = "inner_dP_kPa"


@dataclass(frozen=True)
class PressureLink:
    """One step of the ingestion chain: ``absolute = base + raw_delta``.

    `base` is the absolute pressure of the layer immediately outside this one -
    `core_dP_kPa` for the first link, and the previously-derived absolute for
    every link after it, which is what makes the chain sequential rather than a
    set of independent sums.
    """

    absolute: str
    base: str
    raw_delta: str
    layer: str

    def describe(self) -> str:
        return f"{self.absolute} = {self.base} + {self.raw_delta}"


# Tubular: one capillary layer, so one link - and it is precisely the sum
# `add_pocap` has always computed.
TUBULAR_PRESSURE_CHAIN: tuple[PressureLink, ...] = (
    PressureLink(POCAP_COLUMN, CORE_DP_COLUMN, OUTER_DP_COLUMN, "outer"),
)

# NANF: outer and inner, no middle. The inner layer's raw differential is
# against the outer layer, because that is the layer physically outside it.
NANF_PRESSURE_CHAIN: tuple[PressureLink, ...] = (
    PressureLink(POCAP_COLUMN, CORE_DP_COLUMN, OUTER_DP_COLUMN, "outer"),
    PressureLink(ICAP_COLUMN, POCAP_COLUMN, INNER_DP_COLUMN, "inner"),
)

# DNANF: the full three-layer chain, each link against the one before it.
DNANF_PRESSURE_CHAIN: tuple[PressureLink, ...] = (
    PressureLink(POCAP_COLUMN, CORE_DP_COLUMN, OUTER_DP_COLUMN, "outer"),
    PressureLink(MCAP_COLUMN, POCAP_COLUMN, MID_DP_COLUMN, "middle"),
    PressureLink(ICAP_COLUMN, MCAP_COLUMN, INNER_DP_COLUMN, "inner"),
)


def derive_absolute_pressures(
    df: pd.DataFrame, chain: tuple[PressureLink, ...]
) -> tuple[pd.DataFrame, list[str]]:
    """Reconstruct each layer's absolute pressure by walking the chain.

    Returns the frame and one note per link, so the UI can say what was derived
    rather than quietly inventing a column - the same contract `add_pocap` has.

    A link whose absolute column is already present and populated is left
    alone: a file that carries real absolute pressures is more authoritative
    than anything reconstructed from differentials, and silently overwriting it
    would discard the better measurement.

    Order matters and is not incidental. Each link's base may be the absolute
    the *previous* link just derived, so the chain is walked in sequence and a
    broken link stops the ones downstream of it - reporting that, rather than
    computing a later layer against a base that was never established.
    """
    out = df.copy()
    notes: list[str] = []
    for link in chain:
        if link.absolute in out.columns and out[link.absolute].notna().any():
            notes.append(f"{link.absolute} read directly from the file.")
            continue
        if link.base not in out.columns or link.raw_delta not in out.columns:
            missing = [
                name
                for name in (link.base, link.raw_delta)
                if name not in out.columns
            ]
            notes.append(
                f"{link.absolute} is absent and cannot be derived: "
                f"{', '.join(missing)} not in the file. Any layer inside this "
                "one cannot be derived either."
            )
            continue
        out[link.absolute] = pd.to_numeric(
            out[link.base], errors="coerce"
        ) + pd.to_numeric(out[link.raw_delta], errors="coerce")
        notes.append(
            f"{link.absolute} derived as {link.base} + {link.raw_delta} "
            "(the file carried the sequential differential)."
        )
    return out, notes


# --- the bundle -----------------------------------------------------------


@dataclass(frozen=True)
class PreformSchema:
    """Every column fact about one preform geometry, in one object.

    Downstream code takes one of these instead of reading module globals, so
    "which channels exist" becomes a property of the loaded preform rather
    than of the process. `NONNESTED_SCHEMA` is built from the untouched
    constants above, so passing it is indistinguishable from the pre-v1.9
    behaviour - which is what lets the existing preform's stored calibrations
    keep reproducing exactly.
    """

    features: tuple[ColumnSpec, ...]
    setpoints: tuple[ColumnSpec, ...]
    analytics: tuple[ColumnSpec, ...]
    wall_ratios: tuple[WallRatioSpec, ...]
    deltas: tuple[DeltaSpec, ...]
    # Channels summarised per steady-state block, in display order.
    block_value_columns: tuple[str, ...]
    # Raw block medians carried onto every anchor row.
    anchor_raw_columns: tuple[str, ...]
    # How each layer's absolute pressure is reconstructed from the sequential
    # differentials the tower actually logs. Ingestion only - what is *fitted*
    # is the absolute pressures this produces, unchanged.
    pressure_chain: tuple[PressureLink, ...] = ()

    def derive_pressures(self, df: "pd.DataFrame") -> tuple["pd.DataFrame", list[str]]:
        """Walk this geometry's chain over a raw frame."""
        return derive_absolute_pressures(df, self.pressure_chain)

    @property
    def setpoint_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.setpoints)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.features)

    @property
    def analytic_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.analytics)

    @property
    def wall_ratio_names(self) -> tuple[str, ...]:
        return tuple(ratio.name for ratio in self.wall_ratios)

    @property
    def spec_by_name(self) -> dict[str, ColumnSpec]:
        return {
            spec.name: spec
            for spec in (*self.features, *self.setpoints, *self.analytics)
        }

    def wall_ratio(self, name: str) -> WallRatioSpec | None:
        for ratio in self.wall_ratios:
            if ratio.name == name:
                return ratio
        return None

    def compute_wall_ratios(self, inputs: Mapping[str, float]) -> dict[str, float]:
        """Every wall ratio this geometry defines, from raw or median columns.

        Accepts both the bare column name and the `_median` suffix the block
        summaries use, so the same function serves prediction inputs and
        anchor rows without either caller reshaping its data first.
        """
        out: dict[str, float] = {}
        for ratio in self.wall_ratios:
            od = inputs.get(ratio.od_column, inputs.get(f"{ratio.od_column}_median", np.nan))
            id_ = inputs.get(ratio.id_column, inputs.get(f"{ratio.id_column}_median", np.nan))
            out[ratio.name] = cap_wall_ratio(od, id_)
        return out

    def compute_deltas(self, values: Mapping[str, float]) -> dict[str, float]:
        """Every pressure step this geometry defines. Not clamped, not hidden."""
        return {spec.name: spec.compute(values) for spec in self.deltas}


NONNESTED_SCHEMA = PreformSchema(
    features=REQUIRED_FEATURE_COLUMNS,
    setpoints=SETPOINT_COLUMNS,
    analytics=ANALYTIC_COLUMNS,
    wall_ratios=NONNESTED_WALL_RATIOS,
    deltas=NONNESTED_DELTAS,
    block_value_columns=BLOCK_VALUE_COLUMNS,
    anchor_raw_columns=ANCHOR_RAW_COLUMNS,
    pressure_chain=TUBULAR_PRESSURE_CHAIN,
)

# The nested geometry carries three capillary layers through every stage: the
# block summary needs all six diameters, and the anchor record keeps them for
# the same reason the non-nested one keeps its two - the saved CSV is the
# evidence a calibration rests on, and it is far more useful with the raw
# geometry in it.
NESTED_BLOCK_VALUE_COLUMNS: tuple[str, ...] = (
    "feed_speed_mm_min",
    "draw_speed_m_min",
    "tension_g",
    "furnace_temp_C",
    "fibre_OD_um",
    "fibre_ID_um",
    "cap_OD_outer_um",
    "cap_ID_outer_um",
    "cap_OD_middle_um",
    "cap_ID_middle_um",
    "cap_OD_inner_um",
    "cap_ID_inner_um",
    CORE_DP_COLUMN,
    OUTER_DP_COLUMN,
    POCAP_COLUMN,
    MCAP_COLUMN,
    ICAP_COLUMN,
    ATM_P_COLUMN,
)

NESTED_ANCHOR_RAW_COLUMNS: tuple[str, ...] = (
    "fibre_OD_um",
    "fibre_ID_um",
    "cap_OD_outer_um",
    "cap_ID_outer_um",
    "cap_OD_middle_um",
    "cap_ID_middle_um",
    "cap_OD_inner_um",
    "cap_ID_inner_um",
    "tension_g",
    "feed_speed_mm_min",
    "furnace_temp_C",
    "draw_speed_m_min",
    CORE_DP_COLUMN,
    OUTER_DP_COLUMN,
)

NESTED_SCHEMA = PreformSchema(
    features=NESTED_FEATURE_COLUMNS,
    setpoints=NESTED_SETPOINT_COLUMNS,
    analytics=NESTED_ANALYTIC_COLUMNS,
    wall_ratios=NESTED_WALL_RATIOS,
    deltas=NESTED_DELTAS,
    block_value_columns=NESTED_BLOCK_VALUE_COLUMNS,
    anchor_raw_columns=NESTED_ANCHOR_RAW_COLUMNS,
    pressure_chain=DNANF_PRESSURE_CHAIN,
)




# Every block-value column any registered geometry defines, non-nested order
# first. Used where a frame is *filtered* against this list rather than driven
# by it - `if column in frame.columns` - so widening it cannot change what a
# non-nested frame produces: the nested columns simply are not there. That
# filtering is what makes one union list safe where a per-preform list would
# otherwise have to be threaded through every summary function.
ALL_BLOCK_VALUE_COLUMNS: tuple[str, ...] = tuple(
    dict.fromkeys((*BLOCK_VALUE_COLUMNS, *NESTED_BLOCK_VALUE_COLUMNS))
)


# ==========================================================================
# NANF: outer and inner capillary layers, no middle
# ==========================================================================
#
# Structurally between tubular and DNANF, and not merely DNANF with a layer
# blanked out: NANF has no middle geometry at all, so its inner layer's raw
# differential is measured against the *outer* layer and its pressure chain has
# two links rather than three. Giving it its own schema rather than reusing
# DNANF's with empty columns is what keeps "this preform has no middle layer"
# a fact the app can act on instead of a column full of NaN that every stage
# has to remember to skip.

NANF_FEATURE_COLUMNS: tuple[ColumnSpec, ...] = (
    ColumnSpec("fibre_OD_um", "um", "Target fiber outer diameter", 20.0, 2000.0),
    ColumnSpec("fibre_ID_um", "um", "Target fiber inner (core) diameter", 1.0, 1900.0),
    ColumnSpec(
        "cap_OD_outer_um", "um", "Outer-layer capillary outer diameter", 0.1, 1000.0
    ),
    ColumnSpec(
        "cap_ID_outer_um", "um", "Outer-layer capillary inner diameter", 0.05, 1000.0
    ),
    ColumnSpec(
        "cap_OD_inner_um", "um", "Inner-layer capillary outer diameter", 0.1, 1000.0
    ),
    ColumnSpec(
        "cap_ID_inner_um", "um", "Inner-layer capillary inner diameter", 0.05, 1000.0
    ),
    ColumnSpec("tension_g", "g", "Target draw tension", 0.0, 2000.0),
    ColumnSpec("feed_speed_mm_min", "mm/min", "Preform feed speed", 0.0, 200.0),
)

# Five setpoints: no middle capillary pressure. Absolute, and fitted directly -
# unchanged from how every other preform's pressures are handled.
NANF_SETPOINT_COLUMNS: tuple[ColumnSpec, ...] = (
    ColumnSpec(
        "furnace_temp_C", "degC", "Furnace temperature setpoint", 1500.0, 2200.0
    ),
    ColumnSpec("draw_speed_m_min", "m/min", "Draw (capstan) speed", 0.0, 500.0),
    ColumnSpec(
        CORE_DP_COLUMN, "kPa (gauge)", "Core pressure", 0.0, 500.0, neg_tolerance=1.0
    ),
    ColumnSpec(
        POCAP_COLUMN,
        "kPa (gauge)",
        "Outer capillary pressure",
        0.0,
        500.0,
        neg_tolerance=1.0,
    ),
    ColumnSpec(
        ICAP_COLUMN,
        "kPa (gauge)",
        "Inner capillary pressure",
        0.0,
        500.0,
        neg_tolerance=1.0,
    ),
)

NANF_ANALYTIC_COLUMNS: tuple[ColumnSpec, ...] = tuple(
    ColumnSpec(
        name=f"{ANALYTIC_PREFIX}{spec.name}",
        unit=spec.unit,
        description=f"Analytic (fast physics model) estimate of {spec.name}",
        lo=-1.0e4,
        hi=1.0e4,
    )
    for spec in NANF_SETPOINT_COLUMNS
)

NANF_WALL_RATIOS: tuple[WallRatioSpec, ...] = (
    WallRatioSpec(
        "outer_wall_ratio",
        "cap_OD_outer_um",
        "cap_ID_outer_um",
        "Outer capillary wall ratio",
    ),
    WallRatioSpec(
        "inner_wall_ratio",
        "cap_OD_inner_um",
        "cap_ID_inner_um",
        "Inner capillary wall ratio",
    ),
)

# Two steps, not three: the inner wall is crossed from the outer capillary
# directly, because there is no middle layer between them.
NANF_DELTAS: tuple[DeltaSpec, ...] = (
    DeltaSpec(
        "deltaPocap",
        inner=POCAP_COLUMN,
        outer=CORE_DP_COLUMN,
        description="Across the outer capillary wall: outer capillary minus core.",
    ),
    DeltaSpec(
        "deltaPicap",
        inner=ICAP_COLUMN,
        outer=POCAP_COLUMN,
        description=(
            "Across the inner wall: inner capillary minus outer capillary - "
            "this geometry has no middle layer between them."
        ),
    ),
)

NANF_BLOCK_VALUE_COLUMNS: tuple[str, ...] = (
    "feed_speed_mm_min",
    "draw_speed_m_min",
    "tension_g",
    "furnace_temp_C",
    "fibre_OD_um",
    "fibre_ID_um",
    "cap_OD_outer_um",
    "cap_ID_outer_um",
    "cap_OD_inner_um",
    "cap_ID_inner_um",
    CORE_DP_COLUMN,
    OUTER_DP_COLUMN,
    INNER_DP_COLUMN,
    POCAP_COLUMN,
    ICAP_COLUMN,
    ATM_P_COLUMN,
)

NANF_ANCHOR_RAW_COLUMNS: tuple[str, ...] = (
    "fibre_OD_um",
    "fibre_ID_um",
    "cap_OD_outer_um",
    "cap_ID_outer_um",
    "cap_OD_inner_um",
    "cap_ID_inner_um",
    "tension_g",
    "feed_speed_mm_min",
    "furnace_temp_C",
    "draw_speed_m_min",
    CORE_DP_COLUMN,
    OUTER_DP_COLUMN,
    INNER_DP_COLUMN,
)


NANF_SCHEMA = PreformSchema(
    features=NANF_FEATURE_COLUMNS,
    setpoints=NANF_SETPOINT_COLUMNS,
    analytics=NANF_ANALYTIC_COLUMNS,
    wall_ratios=NANF_WALL_RATIOS,
    deltas=NANF_DELTAS,
    block_value_columns=NANF_BLOCK_VALUE_COLUMNS,
    anchor_raw_columns=NANF_ANCHOR_RAW_COLUMNS,
    pressure_chain=NANF_PRESSURE_CHAIN,
)


# --- the three named geometries ------------------------------------------
#
# Tubular and DNANF are the two schemas above under the names the fabricators
# actually use. They are aliases, not copies: `TUBULAR_SCHEMA is
# NONNESTED_SCHEMA` is true, so every identity check elsewhere in the codebase
# (and every stored calibration resolving through it) keeps working, and there
# is no second object that could drift out of step with the first.
TUBULAR_SCHEMA = NONNESTED_SCHEMA
DNANF_SCHEMA = NESTED_SCHEMA

ALL_SCHEMAS: tuple[PreformSchema, ...] = (
    TUBULAR_SCHEMA,
    NANF_SCHEMA,
    DNANF_SCHEMA,
)


# Resolving a geometry from a stored calibration.
#
# Deliberately keyed on `preform_id`, which every `CalibrationSet` has carried
# since v1.0, rather than persisting the schema itself. That means a
# calibration saved before v1.9 resolves to `NONNESTED_SCHEMA` and reconstructs
# its features exactly as it always did - no format bump, no migration, and no
# stored file that has to be re-fitted to stay loadable.
SCHEMA_BY_PREFORM_ID: dict[str, PreformSchema] = {
    # Current ids.
    "tubular": TUBULAR_SCHEMA,
    "nanf": NANF_SCHEMA,
    "dnanf": DNANF_SCHEMA,
    # The ids these geometries were registered under before this release. A
    # stored calibration carries only its preform id, so these entries are what
    # keep a file saved under the old name resolving to the schema it was
    # actually fitted against.
    "hc_10cap_nonnested": TUBULAR_SCHEMA,
    "hc_nested_3layer": DNANF_SCHEMA,
}


def schema_for_preform(preform_id: str | None) -> PreformSchema:
    """The column bundle for a preform id, defaulting to the non-nested one.

    An unknown or missing id falls back rather than raising: a calibration is
    still loadable and still describes *something*, and the non-nested schema
    is what any file predating the registry was written against.
    """
    return SCHEMA_BY_PREFORM_ID.get(preform_id or "", NONNESTED_SCHEMA)
