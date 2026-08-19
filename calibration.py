"""Anchor calibration: a small, weighted correction from analytic to actual.

The model
---------
For each setpoint channel *c*, the analytic estimate is treated as most of the
answer and only the residual is learned::

    actual_c  =  analytic_c  +  theta0
                             +  theta1 * (cap_wall_ratio      - centre)
                             +  theta2 * (log drawdown ratio  - centre)
                            [+  theta3 * (analytic_c          - centre)]

Two to four parameters per channel, every feature dimensionless, and the
physics model carrying the bulk of the prediction. That is the whole model.

Why so small
------------
A real draw run yields something like eight independent steady-state blocks,
not thousands of rows. Across six input dimensions, eight points cannot
identify a flexible learner: a forest or a Gaussian process would interpolate
those eight and tell you nothing trustworthy anywhere else. Two to four
parameters expressed in dimensionless terms can be identified from eight
points, can be checked one by one against their confidence intervals, and can
plausibly transfer to a new preform geometry - which memorised structure
cannot.

Weighting
---------
Blocks are not equally good. A 1054-second block pins its median down far
better than a 112-second one, and the extraction already reports the standard
error behind every median. Fitting by inverse-variance weighted least squares
uses that: each anchor's weight is 1 / (se_actual^2 + se_analytic^2 + floor^2).

Both standard errors enter because the residual being fitted is a difference of
two measured medians, so both sides contribute uncertainty to it. The floor is
there because a channel that never moved during a block reports a standard
error of exactly zero, which is true of those samples and false of the
instrument; without a floor, that one block would receive infinite weight and
the other seven would be ignored.

Uncertainty
-----------
Parameter covariance is the usual `scale * (X' W X)^-1`, with `scale` the
weighted residual variance, and intervals come from a t distribution on
`n - p` degrees of freedom - not a normal one, because at n around 8 the
difference is large and in the direction of honesty.

This is ordinary weighted least squares. It is implemented here on numpy and
scipy rather than pulled from statsmodels: at two to four parameters the
normal equations are three lines, keeping them local means the covariance and
the interval convention are visible in the same file as the model, and it
leaves one less large dependency to bundle into the exe. `tests/` cross-checks
the numbers against statsmodels' `WLS` where that package is available.
"""

from __future__ import annotations

import datetime as dt
import itertools
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Mapping

import joblib
import numpy as np
import pandas as pd
from scipy import stats

import schema

# Bumped whenever the persisted shape changes, so a stale file is refused with
# a clear message instead of half-loading.
#
# 2: per-channel anchor tables (each setpoint channel now has its own
#    steady-state blocks) and per-term polynomial orders.
# 3: per-channel correction form (additive or ratio) and per-channel fit
#    variables. A prediction cannot be reconstructed without both, so a
#    format-2 file is refused rather than assumed additive.
CALIBRATION_FORMAT = 3

DEFAULT_ALPHA = 0.05  # 95% intervals

# The floor applied to every anchor's combined standard error, as a fraction of
# that channel's spread across the anchor set. 2% of the block-to-block spread
# is well below any real measurement uncertainty, so it never dominates a
# genuine standard error; it only stops a reported zero from taking over.
DEFAULT_SE_FLOOR_FRACTION = 0.02

DRAWDOWN_KINEMATIC = "kinematic"
DRAWDOWN_GEOMETRIC = "geometric"

# Below this many anchors, Auto stays linear whatever the cross-validation
# prefers, and a manual quadratic is cautioned rather than blocked.
#
# Set deliberately in v1.6, and deliberately *not* the 15 that preceded it. That
# 15 was sized for a candidate list that included cubic: a cubic single-variable
# fit costs 4 parameters and a cubic two-feature fit costs 7, so the threshold
# had to be high enough that the largest thing the search could reach still had
# room to be checked. Cubic is gone (see `MAX_ORDER`), so the constant is no
# longer guarding the same object and reusing its value would be a coincidence
# rather than a reason.
#
# 9 is chosen from residual degrees of freedom at the two shapes a quadratic
# search can actually produce:
#
#   single variable, quadratic   3 parameters   9 - 3 = 6 residual dof
#   engineered pair, both quad   5 parameters   9 - 5 = 4 residual dof
#
# Both are positive and non-trivial, and 4 is about where the t interval on a
# coefficient stops being so wide that "significant" and "not significant" are
# the same statement. Below 9 the narrower shape still has dof to spare, but the
# *ranking* between linear and quadratic is what goes unstable first - which is
# what this threshold governs, not whether a fit is arithmetically possible.
# `scan_orders` enforces that separately via `n_anchor >= n_parameters + 2`.
#
# A judgement call, not a law. The real check on a quadratic is the held-out
# score from `dev_training`, which is why crossing this line is a caution and
# not a refusal.
MIN_ANCHORS_FOR_QUADRATIC = 9

# How much worse than the best cross-validated error a candidate may be and
# still be preferred for being simpler.
#
# Auto is deliberately *not* argmin(CV error). At eight anchors the difference
# between a linear and a quadratic LOO-CV score is routinely a few percent and
# routinely turns on which single block was held out - picking the winner of
# that comparison spends a degree of freedom on noise. Within this band the
# candidates are treated as indistinguishable and the simplest one wins.
#
# Named and surfaced in the UI rather than buried: it is the whole difference
# between "Auto" and "lowest number", and anyone reading the selection is owed
# the rule that produced it.
AUTO_ORDER_TOLERANCE = 0.10

# A feature whose values barely move across the anchors carries no information
# about them, and a fit against it is rank-deficient: the slope and the offset
# become interchangeable, `lstsq` returns some enormous coefficient that cancels
# against the intercept, and the in-sample residual goes to zero. That fit then
# scores *better* on leave-one-out than every honest candidate, because the
# degeneracy is present in every fold.
#
# Found by the held-out split: `fibre_OD_um` is exactly constant across the
# first seven blocks of the reference run, and a search that ranked candidates
# on cross-validation alone picked it for `furnace_temp_C` with a coefficient of
# -1.9e15 and a held-out error fourteen orders of magnitude worse than its
# cross-validated one.
#
# So a candidate whose feature does not vary is refused entry to the search,
# with the reason recorded, rather than being allowed to win it. A manual
# override can still fit one - it comes with a loud condition-number warning -
# because refusing to compute what an operator explicitly asked for is a
# different thing from declining to choose it for them.
MIN_FEATURE_SPREAD_FRACTION = 1.0e-9

# Backstop for the near-degenerate case the spread test does not catch: a design
# this badly conditioned has no separately identifiable coefficients whatever
# the individual spreads look like. Far above the value that merely earns a
# collinearity warning (30), so this excludes only the hopeless.
MAX_SEARCH_CONDITION_NUMBER = 1.0e8


def feature_is_degenerate(values: np.ndarray) -> bool:
    """Whether a feature is too nearly constant to fit a coefficient against."""
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return True
    span = float(np.ptp(finite))
    if span <= 0.0:
        return True
    scale = max(abs(float(np.mean(finite))), float(np.max(np.abs(finite))), 1.0e-30)
    return span / scale < MIN_FEATURE_SPREAD_FRACTION


# --------------------------------------------------------------------------
# What is fitted: additive correction or ratio
# --------------------------------------------------------------------------
#
# The quantity being fitted is chosen per channel, and it is a correctness
# question rather than a presentation one.
#
# A ratio says "the analytic estimate is out by a factor". That is only
# meaningful when both sides are strictly positive and share a sign: no
# multiplier maps a negative analytic estimate onto a positive measurement, and
# near zero the ratio explodes. It suits `furnace_temp_C` (absolute
# temperature, ~1781 analytic against ~1997 measured) and `draw_speed_m_min`
# (a speed), both of which stay well away from zero.
#
# It does not suit the two pressure channels, and they are refused it outright:
#
#   * `Pocap_kPa`'s analytic estimate is observed going to about -17 kPa on
#     real anchors while the measurement stays positive. There is no factor
#     that maps one onto the other.
#   * `core_dP_kPa` happens to stay positive on the anchors seen so far, but
#     both are **gauge** pressures - the zero is a reference choice, not a
#     physical origin, and `schema` already allows them to read slightly
#     negative around true zero. A ratio of two gauge pressures changes value
#     if the reference changes, so it is not a physically meaningful multiplier
#     even when it is arithmetically computable.
#
# The refusal is therefore a declared property of the channel, not a check that
# happens to fail on today's data - a future anchor set with all-positive
# pressures must not quietly re-enable it.

FORM_ADDITIVE = "additive"
FORM_RATIO = "ratio"

# One direction, fixed. `actual / analytic` rather than its inverse because it
# reads as "multiply the estimate by this to get reality", which is both the
# way an operator uses it and a direct multiplication at prediction time rather
# than a division that would blow up as the fitted ratio approached zero.
RATIO_DIRECTION = "actual / analytic"
RATIO_PREDICTION_TEXT = "analytic x factor = actual"

# Channels whose quantity is strictly positive with a physical zero, so a
# multiplicative correction means something.
RATIO_CAPABLE_CHANNELS: frozenset[str] = frozenset(
    {"furnace_temp_C", "draw_speed_m_min"}
)

# Covers every channel any registered geometry defines. The nested capillary
# pressures land on FORM_ADDITIVE, which is not a special case: they are gauge
# pressures with the same arbitrary zero as the existing pair, so the same
# argument that refuses `Pocap_kPa` a ratio refuses them one. Listing them
# explicitly rather than relying on the `.get(..., FORM_ADDITIVE)` fallback
# keeps the dict a complete statement of policy rather than a partial one that
# happens to default correctly.
DEFAULT_FORMS: dict[str, str] = {
    channel: (FORM_RATIO if channel in RATIO_CAPABLE_CHANNELS else FORM_ADDITIVE)
    for channel in dict.fromkeys(
        (*schema.SETPOINT_NAMES, *schema.NESTED_SCHEMA.setpoint_names)
    )
}

FORM_LABELS: dict[str, str] = {
    FORM_ADDITIVE: "additive (actual - analytic)",
    FORM_RATIO: f"ratio ({RATIO_DIRECTION})",
}


def ratio_is_offered(channel: str) -> bool:
    """Whether the ratio form may be chosen for this channel at all."""
    return channel in RATIO_CAPABLE_CHANNELS


def allowed_forms(channel: str) -> tuple[str, ...]:
    """Every correction form this channel may legitimately use.

    The single source of truth for what a channel is allowed to be fitted as.
    The manual dropdown greys out what is not in here, and the Auto search
    enumerates exactly this - so a ratio candidate cannot appear for a gauge
    pressure channel by way of a second check that drifted out of step with the
    first.
    """
    if ratio_is_offered(channel):
        return (FORM_ADDITIVE, FORM_RATIO)
    return (FORM_ADDITIVE,)


def ratio_unavailable_reason(
    channel: str, anchors: pd.DataFrame | None = None
) -> str:
    """One line saying why the ratio form is not on offer for this channel.

    Reports what is actually true of the anchors in hand where that is the
    stronger statement, and falls back to the gauge-zero argument otherwise -
    rather than asserting a sign change that a given anchor set may not show.
    """
    if ratio_is_offered(channel):
        return ""
    crosses = False
    if anchors is not None and not anchors.empty:
        analytic_column = schema.analytic_name(channel)
        if analytic_column in anchors.columns:
            values = pd.to_numeric(
                anchors[analytic_column], errors="coerce"
            ).to_numpy(float)
            values = values[np.isfinite(values)]
            crosses = bool(values.size and values.min() <= 0.0)
    if crosses:
        return (
            "undefined: the analytic estimate goes negative in this data while "
            "the measurement stays positive"
        )
    return (
        "not offered: gauge pressure has an arbitrary zero, so a multiplier "
        "between analytic and actual is not a physical quantity"
    )


def resolve_form(channel: str, form: str | None) -> str:
    """The form to fit with, refusing a ratio the channel may not have."""
    if form is None:
        return DEFAULT_FORMS.get(channel, FORM_ADDITIVE)
    if form not in (FORM_ADDITIVE, FORM_RATIO):
        raise CalibrationError(
            f"Unknown correction form {form!r}. Expected "
            f"{FORM_ADDITIVE!r} or {FORM_RATIO!r}."
        )
    if form == FORM_RATIO and not ratio_is_offered(channel):
        raise CalibrationError(
            f"The ratio form is not available for {channel}: "
            + ratio_unavailable_reason(channel)
            + "."
        )
    return form


# --------------------------------------------------------------------------
# Feature terms
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureTerm:
    key: str
    label: str
    description: str
    always_on: bool = False
    default_on: bool = True


TERM_CONST = FeatureTerm(
    key="const",
    label="Offset",
    description=(
        "The mean correction between the analytic estimate and reality, in the "
        "channel's own units. Always fitted: without it, the model asserts the "
        "physics model is unbiased, which is the thing being tested."
    ),
    always_on=True,
)

TERM_WALL = FeatureTerm(
    key="cap_wall_ratio",
    label="Capillary wall ratio",
    description=(
        "(cap_OD - cap_ID) / cap_OD. Dimensionless. How the correction changes "
        "with capillary wall thickness relative to its diameter - the geometric "
        "ratio the analytic capillary pressure is most sensitive to."
    ),
)

TERM_DRAWDOWN = FeatureTerm(
    key="log_drawdown_ratio",
    label="log(draw-down ratio)",
    description=(
        "Natural log of the draw-down ratio. Dimensionless. Logged because "
        "draw-down spans orders of magnitude and the physics is multiplicative "
        "in it; a proportional change in draw-down should mean a fixed change "
        "in the correction."
    ),
)

TERM_ANALYTIC_LEVEL = FeatureTerm(
    key="analytic_level",
    label="Analytic level (gain)",
    description=(
        "The channel's own analytic estimate. Lets the correction scale with "
        "the size of the estimate rather than being a fixed offset - a gain "
        "error rather than a bias. Off by default: it costs a parameter, and "
        "at eight anchors that is a real cost."
    ),
    default_on=False,
)

FEATURE_TERMS: tuple[FeatureTerm, ...] = (
    TERM_CONST,
    TERM_WALL,
    TERM_DRAWDOWN,
    TERM_ANALYTIC_LEVEL,
)

# The raw measured inputs, offered as fit variables in their own right. They are
# already columns on every anchor table, and the Predict tab already collects
# all six from the operator, so a correction fitted against one of them is
# computable at prediction time with no new input.
#
# `default_on=False` throughout: adding these must not change what
# `DEFAULT_TERMS` means, so the two-engineered-feature fit stays exactly what it
# was in v1.0-v1.2.
RAW_INPUT_TERMS: tuple[FeatureTerm, ...] = tuple(
    FeatureTerm(
        key=spec.name,
        label=spec.name,
        description=f"{spec.description} ({spec.unit}). Measured per block.",
        default_on=False,
    )
    for spec in schema.REQUIRED_FEATURE_COLUMNS
)

# The nested geometry's own raw inputs and its three per-layer wall ratios
# (v1.9). Registered here so `TERM_BY_KEY` can resolve their labels and
# `fit_channel` accepts them - the registry is a lookup, not a menu. What is
# actually *offered* for a given fit is `single_variable_keys()` below, which
# is per-preform, so registering these cannot widen the non-nested search.
NESTED_RAW_INPUT_TERMS: tuple[FeatureTerm, ...] = tuple(
    FeatureTerm(
        key=spec.name,
        label=spec.name,
        description=f"{spec.description} ({spec.unit}). Measured per block.",
        default_on=False,
    )
    for spec in schema.NESTED_FEATURE_COLUMNS
    # The shared ones are already registered above; re-registering would make
    # `TERM_BY_KEY` depend on dict ordering for which description wins.
    if spec.name not in {t.key for t in RAW_INPUT_TERMS}
)

# `cap_wall_ratio` generalised: the same (OD - ID) / OD on each nested layer.
# `TERM_WALL` above is this quantity for the single-capillary geometry and
# keeps its name and key - it is what every stored calibration was fitted
# against, and renaming it would invalidate those coefficients for nothing.
WALL_RATIO_TERMS: tuple[FeatureTerm, ...] = tuple(
    FeatureTerm(
        key=ratio.name,
        label=ratio.label,
        description=(
            f"({ratio.od_column} - {ratio.id_column}) / {ratio.od_column}. "
            "Dimensionless. How the correction changes with that layer's wall "
            "thickness relative to its diameter - the geometric ratio a "
            "capillary pressure is most directly coupled to."
        ),
        default_on=False,
    )
    for ratio in schema.NESTED_WALL_RATIOS
)

ALL_TERMS: tuple[FeatureTerm, ...] = (
    FEATURE_TERMS + RAW_INPUT_TERMS + NESTED_RAW_INPUT_TERMS + WALL_RATIO_TERMS
)

TERM_BY_KEY = {term.key: term for term in ALL_TERMS}

DEFAULT_TERMS: tuple[str, ...] = tuple(
    term.key for term in FEATURE_TERMS if term.always_on or term.default_on
)

# What a single-variable fit may be fitted against: the six raw inputs, plus the
# two engineered features so the v1.0-v1.2 result stays directly comparable
# rather than being replaced by something that cannot be lined up against it.
SINGLE_VARIABLE_KEYS: tuple[str, ...] = tuple(
    [spec.name for spec in schema.REQUIRED_FEATURE_COLUMNS]
    + [TERM_WALL.key, TERM_DRAWDOWN.key]
)


def single_variable_keys(
    preform_schema: schema.PreformSchema | None = None,
) -> tuple[str, ...]:
    """What this geometry's fit-against dropdown and Auto search may offer.

    Per-preform since v1.9. Defaulting to the non-nested schema returns exactly
    `SINGLE_VARIABLE_KEYS`, which is what keeps the existing preform's Auto
    search enumerating the same candidate list it always has - a wider list
    would change which candidate wins by changing what it was compared against.
    """
    preform_schema = preform_schema or schema.NONNESTED_SCHEMA
    if preform_schema is schema.NONNESTED_SCHEMA:
        return SINGLE_VARIABLE_KEYS
    return tuple(
        [spec.name for spec in preform_schema.features]
        + list(preform_schema.wall_ratio_names)
        + [TERM_DRAWDOWN.key]
    )

# The two-feature engineered fit, kept selectable as a named option rather than
# deleted - it is the shape every earlier version validated.
ENGINEERED_PAIR_KEY = "__engineered_pair__"

# Where each channel's dropdown opens. Not a UI nicety: each is a hypothesis
# about which single quantity the correction is most coupled to, and the caption
# beside the dropdown says so. A default is not a restriction - every channel
# still offers the full list.
SUGGESTED_VARIABLE: dict[str, str] = {
    # Core pressure is the mechanism that sets the fiber ID.
    "core_dP_kPa": "fibre_ID_um",
    # Capillary pressure sets the capillary bore directly.
    "Pocap_kPa": "cap_ID_um",
    # Temperature governs viscosity, which is what draw tension measures.
    "furnace_temp_C": "tension_g",
    # v1.2's two-feature fit found nothing else mattered here, so this keeps the
    # one existing result rather than guessing a new default with no evidence.
    "draw_speed_m_min": TERM_WALL.key,
}

SUGGESTED_VARIABLE_WHY: dict[str, str] = {
    "core_dP_kPa": "core pressure is the mechanism that sets fiber ID",
    "Pocap_kPa": "capillary pressure sets the capillary ID and OD directly",
    "furnace_temp_C": (
        "temperature governs viscosity, which is the physical driver of draw "
        "tension"
    ),
    "draw_speed_m_min": (
        "the existing two-feature fit found no other predictor mattered here"
    ),
}

# Offered alongside the suggestion as the obvious second thing to try.
ALTERNATE_VARIABLE: dict[str, str] = {
    "core_dP_kPa": "fibre_OD_um",
    "Pocap_kPa": "cap_OD_um",
}

# --- nested geometry defaults (v1.9) --------------------------------------
#
# Same reasoning as the table above, one layer at a time: each capillary
# pressure opens on the wall ratio of the layer it actually inflates. That is
# the nested generalisation of "Pocap opens on cap_ID_um" - a hypothesis about
# the quantity the correction is most coupled to, not a restriction, and the
# operator still sees every variable in the dropdown.
#
# On the nested geometry `Pocap_kPa` moves from the single `cap_ID_um` to
# `outer_wall_ratio`, because on a three-layer preform the outer capillary
# pressure governs its own layer's wall rather than a bore the geometry no
# longer has a single one of.
NESTED_SUGGESTED_VARIABLE: dict[str, str] = {
    "furnace_temp_C": "tension_g",
    "draw_speed_m_min": "outer_wall_ratio",
    "core_dP_kPa": "fibre_ID_um",
    "Pocap_kPa": "outer_wall_ratio",
    schema.MCAP_COLUMN: "middle_wall_ratio",
    schema.ICAP_COLUMN: "inner_wall_ratio",
}

NESTED_SUGGESTED_VARIABLE_WHY: dict[str, str] = {
    "furnace_temp_C": (
        "temperature governs viscosity, which is the physical driver of draw "
        "tension"
    ),
    "draw_speed_m_min": (
        "no other predictor has been shown to matter here; the outer layer's "
        "wall ratio is the closest analogue of the non-nested default"
    ),
    "core_dP_kPa": "core pressure is the mechanism that sets fiber ID",
    "Pocap_kPa": (
        "the outer capillary pressure inflates the outer layer, so its wall "
        "ratio is the geometry it is most directly coupled to"
    ),
    schema.MCAP_COLUMN: (
        "the middle capillary pressure inflates the middle layer, so its own "
        "wall ratio is the coupled geometry - not another layer's"
    ),
    schema.ICAP_COLUMN: (
        "the inner capillary pressure inflates the inner layer, so its own "
        "wall ratio is the coupled geometry - not another layer's"
    ),
}

NESTED_ALTERNATE_VARIABLE: dict[str, str] = {
    "core_dP_kPa": "fibre_OD_um",
    "Pocap_kPa": "cap_ID_outer_um",
    schema.MCAP_COLUMN: "cap_ID_middle_um",
    schema.ICAP_COLUMN: "cap_ID_inner_um",
}


def suggested_variable(
    channel: str, preform_schema: schema.PreformSchema | None = None
) -> str:
    """Where this channel's fit-against dropdown opens, for this geometry."""
    preform_schema = preform_schema or schema.NONNESTED_SCHEMA
    table = (
        SUGGESTED_VARIABLE
        if preform_schema is schema.NONNESTED_SCHEMA
        else NESTED_SUGGESTED_VARIABLE
    )
    return table.get(channel, TERM_WALL.key)


def suggested_variable_why(
    channel: str, preform_schema: schema.PreformSchema | None = None
) -> str:
    preform_schema = preform_schema or schema.NONNESTED_SCHEMA
    table = (
        SUGGESTED_VARIABLE_WHY
        if preform_schema is schema.NONNESTED_SCHEMA
        else NESTED_SUGGESTED_VARIABLE_WHY
    )
    return table.get(channel, "")


def variable_label(key: str) -> str:
    if key == ENGINEERED_PAIR_KEY:
        return f"{TERM_WALL.label} + {TERM_DRAWDOWN.label} (the v1.2 fit)"
    term = TERM_BY_KEY.get(key)
    return term.label if term else key


def terms_for_variable(key: str) -> tuple[str, ...]:
    """The `terms` tuple that fits against one chosen variable."""
    if key == ENGINEERED_PAIR_KEY:
        return DEFAULT_TERMS
    if key not in TERM_BY_KEY:
        raise CalibrationError(f"Unknown fit variable {key!r}.")
    return (TERM_CONST.key, key)


def variable_of(terms: tuple[str, ...]) -> str:
    """Inverse of `terms_for_variable`, for reporting what a fit used."""
    active = [key for key in terms if key != TERM_CONST.key]
    if len(active) == 1:
        return active[0]
    return ENGINEERED_PAIR_KEY


def default_variable(
    channel: str, preform_schema: schema.PreformSchema | None = None
) -> str:
    return suggested_variable(channel, preform_schema)


# Every setpoint channel any registered geometry defines, in a stable order:
# the non-nested four first, then whatever a later geometry added. The order
# matters because it is the order channels are fitted and displayed in, and
# putting the original four first is what makes filtering this list to a
# non-nested anchor set return exactly `SETPOINT_NAMES` - which is why the
# existing preform's results cannot be reordered by this change.
ALL_KNOWN_SETPOINTS: tuple[str, ...] = tuple(
    dict.fromkeys((*schema.SETPOINT_NAMES, *schema.NESTED_SCHEMA.setpoint_names))
)


def ordered_channels(
    tables: Mapping[str, object],
    preform_schema: schema.PreformSchema | None = None,
) -> tuple[str, ...]:
    """Which channels to process, in display order.

    Replaces iterating a module-level channel list. Given an explicit schema it
    is that geometry's channels; otherwise it is whichever known channels the
    supplied tables actually carry, which is how a caller that never learned
    about preforms still does the right thing for either geometry.
    """
    if preform_schema is not None:
        return tuple(c for c in preform_schema.setpoint_names if c in tables)
    return tuple(c for c in ALL_KNOWN_SETPOINTS if c in tables)


class CalibrationError(ValueError):
    """Raised when a calibration cannot be fitted or loaded."""


# --------------------------------------------------------------------------
# Anchor assembly
# --------------------------------------------------------------------------


def build_anchor_table(
    blocks: pd.DataFrame,
    estimates: pd.DataFrame,
    preform_OD_mm: float | None = None,
    preform_schema: schema.PreformSchema | None = None,
) -> tuple[pd.DataFrame, str]:
    """Join blocks to their analytic estimates and compute the fit features.

    Returns the anchor table and the draw-down mode that was used, which the
    calibration records so prediction computes the same feature the same way.

    The draw-down ratio uses the *analytic* draw speed, never the measured one.
    The measured draw speed is one of the four things being predicted, so it
    does not exist at prediction time; building the feature from the analytic
    value keeps it computable on both sides. If a preform outer diameter is
    supplied, the geometric ratio (preform OD over fiber OD) is used instead -
    it is a property of the geometry alone and inherits no error from the
    analytic draw speed.
    """
    if blocks.empty:
        raise CalibrationError("No steady-state blocks to build anchors from.")
    if estimates.empty:
        raise CalibrationError("No analytic estimates to pair with the blocks.")

    merged = blocks.merge(estimates, on="block_id", how="left", suffixes=("", "_est"))
    preform_schema = preform_schema or schema.NONNESTED_SCHEMA

    use_geometric = (
        preform_OD_mm is not None
        and np.isfinite(float(preform_OD_mm))
        and float(preform_OD_mm) > 0
    )
    mode = DRAWDOWN_GEOMETRIC if use_geometric else DRAWDOWN_KINEMATIC

    rows: list[dict] = []
    for record in merged.to_dict("records"):
        row: dict = {
            "block_id": record["block_id"],
            "start_time": record.get("start_time"),
            "end_time": record.get("end_time"),
            "duration_s": record.get("duration_s"),
            "n_analytic_rows": record.get("n_rows", np.nan),
            "analytic_status": record.get("status", "missing"),
        }
        for channel in preform_schema.setpoint_names:
            row[f"actual_{channel}"] = record.get(f"{channel}_median", np.nan)
            row[f"actual_{channel}_se"] = record.get(f"{channel}_se", np.nan)
            analytic = schema.analytic_name(channel)
            row[analytic] = record.get(analytic, np.nan)
            row[f"{analytic}_se"] = record.get(f"{analytic}_se", np.nan)

        # Raw block medians travel with the anchor so the diagnostic grid can
        # plot the setpoint against the geometry itself, not only against the
        # engineered features derived from it.
        for column in preform_schema.anchor_raw_columns:
            row[column] = record.get(f"{column}_median", np.nan)
            row[f"{column}_se"] = record.get(f"{column}_se", np.nan)

        # Every wall ratio this geometry defines - one for the single-capillary
        # preform (still keyed `cap_wall_ratio`), three for the nested one.
        row.update(preform_schema.compute_wall_ratios(record))
        if use_geometric:
            ratio = schema.geometric_drawdown_ratio(
                float(preform_OD_mm), record.get("fibre_OD_um_median", np.nan)
            )
        else:
            ratio = schema.kinematic_drawdown_ratio(
                record.get(schema.analytic_name("draw_speed_m_min"), np.nan),
                record.get("feed_speed_mm_min_median", np.nan),
            )
        row["drawdown_ratio"] = ratio
        row["log_drawdown_ratio"] = (
            float(np.log(ratio)) if np.isfinite(ratio) and ratio > 0 else np.nan
        )
        rows.append(row)

    return pd.DataFrame(rows), mode


def build_anchor_tables(
    blocks_by_channel: Mapping[str, pd.DataFrame],
    estimates_by_channel: Mapping[str, pd.DataFrame],
    preform_OD_mm: float | None = None,
    preform_schema: schema.PreformSchema | None = None,
) -> tuple[dict[str, pd.DataFrame], str]:
    """One anchor table per setpoint channel.

    Since v1.1 each channel has its own steady-state blocks, so it has its own
    anchors - different rows, different count, different block ids. Nothing
    downstream may assume the channels share a table, and since v1.9 nothing
    may assume how many of them there are either.
    """
    tables: dict[str, pd.DataFrame] = {}
    mode = DRAWDOWN_KINEMATIC
    errors: list[str] = []
    for channel, blocks in blocks_by_channel.items():
        estimates = estimates_by_channel.get(channel)
        if estimates is None:
            errors.append(f"{channel}: no analytic estimates were matched.")
            continue
        try:
            table, mode = build_anchor_table(
                blocks, estimates, preform_OD_mm, preform_schema
            )
        except CalibrationError as exc:
            errors.append(f"{channel}: {exc}")
            continue
        table.insert(1, "channel", channel)
        tables[channel] = table
    if not tables:
        raise CalibrationError(
            "No channel produced a usable anchor table.\n" + "\n".join(errors)
        )
    return tables, mode


def combined_anchor_table(tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """All per-channel anchors stacked, for range checks and for the saved CSV."""
    frames = [table for table in tables.values() if not table.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def usable_mask(
    anchors: pd.DataFrame,
    channel: str,
    terms,
    form: str = FORM_ADDITIVE,
) -> tuple[np.ndarray, list[str]]:
    """Which anchors can be fitted for one channel, and why the rest cannot."""
    analytic = schema.analytic_name(channel)
    needed = [f"actual_{channel}", analytic]
    # Every term except the offset and the analytic level is a column that has
    # to be present and finite. Listed by iteration rather than by name so a
    # raw input works exactly like an engineered feature.
    for key in terms:
        if key in (TERM_CONST.key, TERM_ANALYTIC_LEVEL.key):
            continue
        needed.append(key)

    ok = np.ones(len(anchors), dtype=bool)
    reasons: list[str] = []
    for column in needed:
        if column not in anchors.columns:
            reasons.append(f"every anchor dropped: {column} is not in the table.")
            return np.zeros(len(anchors), dtype=bool), reasons
        finite = np.isfinite(pd.to_numeric(anchors[column], errors="coerce").to_numpy(float))
        dropped = int((ok & ~finite).sum())
        if dropped:
            reasons.append(f"{dropped} anchor(s) dropped: {column} is missing.")
        ok &= finite

    status = anchors.get("analytic_status")
    if status is not None:
        matched = status.to_numpy() != "empty"
        dropped = int((ok & ~matched).sum())
        if dropped:
            reasons.append(
                f"{dropped} anchor(s) dropped: no analytic row matched the block."
            )
        ok &= matched

    if form == FORM_RATIO:
        # A ratio needs both sides on the same side of zero. An anchor that
        # breaks that cannot contribute a factor, so it is dropped and said so
        # rather than producing a negative or exploding multiplier.
        analytic_values = pd.to_numeric(
            anchors[analytic], errors="coerce"
        ).to_numpy(float)
        actual_values = pd.to_numeric(
            anchors[f"actual_{channel}"], errors="coerce"
        ).to_numpy(float)
        with np.errstate(invalid="ignore"):
            same_sign = (
                np.isfinite(analytic_values)
                & np.isfinite(actual_values)
                & (analytic_values != 0.0)
                & (np.sign(analytic_values) == np.sign(actual_values))
            )
        dropped = int((ok & ~same_sign).sum())
        if dropped:
            reasons.append(
                f"{dropped} anchor(s) dropped: the ratio form needs the analytic "
                "estimate and the measurement on the same side of zero."
            )
        ok &= same_sign
    return ok, reasons


# --------------------------------------------------------------------------
# One channel's fit
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TermEstimate:
    key: str
    label: str
    value: float
    se: float
    ci_lo: float
    ci_hi: float
    # Feature centring and scaling used internally; kept for reporting the
    # range each coefficient was actually identified over.
    center: float
    spread: float
    # Which power of the (centred) feature this coefficient multiplies. 1 for a
    # linear term, 0 for the offset.
    power: int = 1

    @property
    def spans_zero(self) -> bool:
        return self.ci_lo <= 0.0 <= self.ci_hi


# Order names, used in the UI and in reports. Index is the polynomial order.
#
# Cubic was removed in v1.6 - from the manual dropdown, from Auto, and from the
# dev-training search alike. It never had a physical justification here: the
# correction being fitted is a small residual between an analytic estimate and a
# measurement, and nothing in the draw physics suggests an inflection in it. It
# also cannot be afforded. A cubic single-variable fit is 4 parameters and a
# cubic two-feature fit is 7, against the eight-to-sixteen anchors a real run
# yields, so the candidate existed mainly to be excluded by the feasibility test
# further down - and, when it was feasible, to win on training error while
# describing nothing.
#
# Removed rather than hidden: `MAX_ORDER` is the single bound, an order above it
# raises instead of being silently clamped, and no code path can name a third
# power.
ORDER_LABELS: tuple[str, ...] = ("constant", "linear", "quadratic")
MAX_ORDER = 2


def order_label(order: int) -> str:
    return ORDER_LABELS[order] if 0 <= order < len(ORDER_LABELS) else f"order {order}"


def quadratic_caution(
    channel: str,
    orders: Mapping[str, int],
    n_anchor: int,
    min_anchors: int = MIN_ANCHORS_FOR_QUADRATIC,
) -> str:
    """A caution when a quadratic is in use below the anchor threshold.

    A caution, never a refusal. The old behaviour was a hard wall: below the
    threshold a higher order was simply unavailable, on the reasoning that the
    cross-validated ranking that would justify it is unstable at small n. That
    reasoning is sound and the wall was still the wrong instrument, because the
    thing it was standing in for now exists - `dev_training`'s held-out split
    scores a shape against blocks that took no part in choosing it, which is a
    direct answer to "is this quadratic real?" rather than a proxy for it.

    So: manual and Auto, which have no held-out score behind them, say what the
    anchor count does and does not support and point at the check that would
    settle it. Empty string when there is nothing to caution about.
    """
    if n_anchor >= min_anchors:
        return ""
    quadratic = sorted(
        variable_label(key) for key, order in orders.items() if int(order) >= 2
    )
    if not quadratic:
        return ""
    return (
        f"{channel}: {', '.join(quadratic)} is fitted quadratic on {n_anchor} "
        f"anchor(s), below the {min_anchors} this app asks for before it will "
        "choose a quadratic on its own. It is not blocked - the fit is "
        "computable and may well be right - but nothing here has tested it: at "
        "this many anchors the cross-validated ranking of linear against "
        "quadratic can turn on a single block. Confirm it under Dev > Train "
        "Auto selection, which scores the shape against blocks held back from "
        "the search, before relying on it."
    )


def _term_label(key: str, power: int) -> str:
    base = TERM_BY_KEY[key].label if key in TERM_BY_KEY else key
    if power <= 1:
        return base
    return f"{base}^{power}"


def resolve_order(key: str, order: int) -> int:
    """One term's polynomial order, refusing anything above `MAX_ORDER`.

    Raises rather than clamping. Clamping is how a removed option comes back as
    a silent downgrade: asking for cubic and being handed quadratic without
    being told leaves a caller believing it got the shape it named. Since v1.6
    there is no cubic to ask for, so asking is a mistake in the request and is
    reported as one.
    """
    order = int(order)
    if order > MAX_ORDER:
        raise CalibrationError(
            f"{order_label(order) if order < len(ORDER_LABELS) else f'order {order}'} "
            f"is not an available shape for '{key}'. This app fits up to "
            f"{order_label(MAX_ORDER)} ({MAX_ORDER} powers); cubic was removed "
            "in v1.6 as unjustified at these anchor counts."
        )
    return max(1, order)


def expand_terms(
    terms: tuple[str, ...], orders: Mapping[str, int] | None
) -> tuple[tuple[str, int], ...]:
    """Turn {term: order} into the ordered list of (term, power) design columns.

    The offset is always a single column. Every other term contributes one
    column per power up to its order, on the *centred* feature - so a quadratic
    wall-ratio term is `(w - centre)` and `(w - centre)^2`, which keeps the
    linear coefficient meaning what it meant before the quadratic was added.
    """
    orders = orders or {}
    columns: list[tuple[str, int]] = []
    for key in terms:
        if key == TERM_CONST.key:
            columns.append((key, 0))
            continue
        order = resolve_order(key, orders.get(key, 1))
        for power in range(1, order + 1):
            columns.append((key, power))
    return tuple(columns)


@dataclass(frozen=True)
class ChannelCalibration:
    """The fitted correction for one setpoint channel."""

    channel: str
    unit: str
    terms: tuple[str, ...]
    theta: np.ndarray
    cov: np.ndarray
    centers: dict[str, float]
    n_anchor: int
    dof: int
    alpha: float
    scale: float
    residual_rms: float
    weighted_residual_rms: float
    condition_number: float
    se_floor: float
    estimates: tuple[TermEstimate, ...]
    block_ids: tuple[int, ...]
    actual: np.ndarray
    actual_se: np.ndarray
    analytic: np.ndarray
    fitted: np.ndarray
    weights: np.ndarray
    warnings: tuple[str, ...] = ()
    excluded_notes: tuple[str, ...] = ()
    # {term key: polynomial order}. Absent keys are linear.
    orders: dict[str, int] = field(default_factory=dict)
    # The (term, power) pair behind each design column, in column order.
    columns: tuple[tuple[str, int], ...] = ()
    # Leave-one-block-out CV error for the order combination actually used,
    # in the channel's own units. NaN when it was not computed.
    loo_rmse: float = float("nan")
    # Which quantity was fitted: FORM_ADDITIVE or FORM_RATIO.
    form: str = FORM_ADDITIVE
    # LOO-CV of the two-engineered-feature fit under the same form, so a
    # single-variable fit can be reported as better or worse than the shape
    # every earlier version used. NaN when not computed or not applicable.
    baseline_loo_rmse: float = float("nan")

    @property
    def order_summary(self) -> str:
        parts = [
            f"{variable_label(key)}: {order_label(self.orders.get(key, 1))}"
            for key in self.terms
            if key != TERM_CONST.key
        ]
        return ", ".join(parts) if parts else "offset only"

    @property
    def variable(self) -> str:
        """The single variable fitted against, or the engineered-pair marker."""
        return variable_of(self.terms)

    @property
    def variable_label(self) -> str:
        return variable_label(self.variable)

    @property
    def form_label(self) -> str:
        return FORM_LABELS.get(self.form, self.form)

    @property
    def is_ratio(self) -> bool:
        return self.form == FORM_RATIO

    # -------------------------------------------------------------- design

    def _design_row(self, features: dict, analytic_value: float) -> np.ndarray:
        columns = self.columns or tuple((key, 0 if key == TERM_CONST.key else 1) for key in self.terms)
        row = np.empty(len(columns), dtype=float)
        for i, (key, power) in enumerate(columns):
            if key == TERM_CONST.key:
                row[i] = 1.0
                continue
            if key == TERM_ANALYTIC_LEVEL.key:
                raw = float(analytic_value)
            else:
                value = features.get(key, np.nan)
                if value is None or not np.isfinite(float(value)):
                    raise CalibrationError(
                        f"The calibration for {self.channel} needs "
                        f"'{key}', which could not be computed from the inputs."
                    )
                raw = float(value)
            row[i] = (raw - self.centers[key]) ** power
        return row

    def predict(
        self, features: dict, analytic_value: float, alpha: float | None = None
    ) -> "ChannelPrediction":
        alpha = self.alpha if alpha is None else alpha
        analytic = float(analytic_value)
        if not np.isfinite(analytic):
            raise CalibrationError(
                f"An analytic estimate for {self.channel} is required; the "
                "calibration corrects it rather than replacing it."
            )
        x = self._design_row(features, analytic)
        fitted_target = float(x @ self.theta)
        var_target = max(float(x @ self.cov @ x), 0.0)
        # A new block also scatters about the fitted line by roughly as much as
        # the anchors did, so a genuine prediction interval carries both.
        var_total = var_target + self.weighted_residual_rms**2

        # Both variances live in target units. For a ratio that is
        # dimensionless, and the step into the channel's units is a
        # multiplication by the analytic estimate - so the standard errors scale
        # by |analytic| too. Skipping that would report a ratio's uncertainty as
        # though it were degrees C.
        if self.form == FORM_RATIO:
            value = analytic * fitted_target
            jacobian = abs(analytic)
        else:
            value = analytic + fitted_target
            jacobian = 1.0
        correction = value - analytic
        se_correction = float(np.sqrt(var_target) * jacobian)
        se_total = float(np.sqrt(var_total) * jacobian)
        t_crit = (
            float(stats.t.ppf(1.0 - alpha / 2.0, self.dof)) if self.dof > 0 else np.nan
        )
        return ChannelPrediction(
            channel=self.channel,
            unit=self.unit,
            analytic=analytic,
            correction=correction,
            value=value,
            se_correction=se_correction,
            ci_lo=value - t_crit * se_correction,
            ci_hi=value + t_crit * se_correction,
            pi_lo=value - t_crit * se_total,
            pi_hi=value + t_crit * se_total,
            n_anchor=self.n_anchor,
            confidence=1.0 - alpha,
            form=self.form,
            factor=fitted_target if self.form == FORM_RATIO else float("nan"),
        )

    def describe(self) -> str:
        return (
            f"{self.channel}: {len(self.estimates)} parameter(s) on "
            f"{self.n_anchor} anchor(s), {self.dof} dof, {self.form} form vs "
            f"{self.variable_label}, RMS residual "
            f"{self.residual_rms:.4g} {self.unit}"
        )


@dataclass(frozen=True)
class ChannelPrediction:
    channel: str
    unit: str
    analytic: float
    correction: float
    value: float
    se_correction: float
    ci_lo: float
    ci_hi: float
    pi_lo: float
    pi_hi: float
    n_anchor: int
    confidence: float
    form: str = FORM_ADDITIVE
    # The fitted multiplier for a ratio-form channel, NaN otherwise. Carried so
    # the Predict tab can say "analytic x 1.121" rather than only showing the
    # difference that multiplier happened to produce.
    factor: float = float("nan")


def _se_floor(actual_se: np.ndarray, actual: np.ndarray, fraction: float) -> float:
    """A floor for the combined standard error, in the channel's own units.

    Set from the anchor-to-anchor spread of the channel rather than from an
    absolute constant, so it carries over to a channel measured in kPa and a
    channel measured in degrees C without retuning. A channel that genuinely
    does not vary across the anchors falls back to the largest reported
    standard error, which keeps the weights finite and roughly equal.
    """
    values = actual[np.isfinite(actual)]
    spread = 0.0
    if values.size >= 2:
        median = float(np.median(values))
        spread = 1.4826 * float(np.median(np.abs(values - median)))
        if spread == 0.0:
            spread = float(np.ptp(values))
    floor = fraction * spread
    if floor > 0.0:
        return float(floor)
    finite_se = actual_se[np.isfinite(actual_se) & (actual_se > 0)]
    if finite_se.size:
        return float(finite_se.max())
    return 1.0e-9


def fit_channel(
    anchors: pd.DataFrame,
    channel: str,
    terms: tuple[str, ...] = DEFAULT_TERMS,
    alpha: float = DEFAULT_ALPHA,
    se_floor_fraction: float = DEFAULT_SE_FLOOR_FRACTION,
    orders: Mapping[str, int] | None = None,
    form: str | None = None,
) -> ChannelCalibration:
    """Fit one channel's correction by inverse-variance weighted least squares.

    `form` selects what is fitted: the additive residual `actual - analytic`, or
    the ratio `actual / analytic`. Left as None it is the channel's default, and
    asking for a ratio on a channel that may not have one raises rather than
    quietly computing a meaningless multiplier.
    """
    form = resolve_form(channel, form)
    terms = tuple(dict.fromkeys((TERM_CONST.key, *terms)))
    unknown = [t for t in terms if t not in TERM_BY_KEY]
    if unknown:
        raise CalibrationError(f"Unknown feature term(s): {', '.join(unknown)}.")
    resolved_orders = {
        key: resolve_order(key, (orders or {}).get(key, 1))
        for key in terms
        if key != TERM_CONST.key
    }
    design_columns = expand_terms(terms, resolved_orders)

    analytic_column = schema.analytic_name(channel)
    ok, reasons = usable_mask(anchors, channel, terms, form)
    used = anchors[ok]
    n, p = len(used), len(design_columns)
    if n == 0:
        raise CalibrationError(
            f"No usable anchor remains for {channel}. " + " ".join(reasons)
        )
    if n <= p:
        raise CalibrationError(
            f"{channel} has {n} usable anchor(s) but {p} parameter(s) to fit "
            f"({', '.join(f'{k} {order_label(v)}' for k, v in resolved_orders.items()) or 'offset only'}). "
            "At least one more anchor than parameters is needed for the fit to "
            "have any residual degrees of freedom - lower an order, turn a "
            "feature term off, or extract more blocks."
        )

    actual = pd.to_numeric(used[f"actual_{channel}"], errors="coerce").to_numpy(float)
    actual_se = pd.to_numeric(
        used[f"actual_{channel}_se"], errors="coerce"
    ).to_numpy(float)
    analytic = pd.to_numeric(used[analytic_column], errors="coerce").to_numpy(float)
    analytic_se = pd.to_numeric(
        used.get(f"{analytic_column}_se", pd.Series(np.nan, index=used.index)),
        errors="coerce",
    ).to_numpy(float)

    actual_se = np.where(np.isfinite(actual_se), actual_se, 0.0)
    analytic_se = np.where(np.isfinite(analytic_se), analytic_se, 0.0)

    # What is being explained, and how uncertain it is.
    #
    # Additive: the target is a difference of two measured medians, so the two
    # standard errors add in quadrature.
    #
    # Ratio: the target is a quotient, so it is the *relative* errors that add
    # in quadrature - se(r)/r = sqrt((se_a/a)^2 + (se_b/b)^2). Reusing the
    # additive standard error here would weight the anchors by the wrong
    # quantity and, on a channel like furnace temperature where the values are
    # near 2000, be wrong by three orders of magnitude.
    if form == FORM_RATIO:
        residual_target = actual / analytic
        relative = np.sqrt(
            np.divide(actual_se, actual, out=np.zeros_like(actual_se), where=actual != 0)
            ** 2
            + np.divide(
                analytic_se, analytic, out=np.zeros_like(analytic_se), where=analytic != 0
            )
            ** 2
        )
        target_se = np.abs(residual_target) * relative
        # Scaled to the spread of the ratio, which is order 1, rather than to
        # the channel's own units - a floor of 2% of ~2000 degC would swamp
        # every real error on a dimensionless target.
        floor = _se_floor(target_se, residual_target, se_floor_fraction)
    else:
        residual_target = actual - analytic
        target_se = np.sqrt(actual_se**2 + analytic_se**2)
        # Unchanged from v1.0-v1.2: scaled to the channel's own spread, which is
        # what the additive results already in hand were fitted with.
        floor = _se_floor(actual_se, actual, se_floor_fraction)

    se_total = np.sqrt(target_se**2 + floor**2)
    weights = 1.0 / se_total**2

    # Build the design in centred, unit-variance columns. Centring makes the
    # offset term mean "the correction at the middle of the anchor set" rather
    # than "the correction extrapolated to zero wall ratio", and scaling keeps
    # the normal equations well conditioned when one feature varies over 0.6
    # and another over 0.01. Both are undone before anything is reported, so
    # the coefficients below are per unit of the raw dimensionless feature.
    # Higher powers are taken of the *centred* feature, so adding a quadratic
    # term leaves the linear coefficient meaning what it meant without it, and
    # the two columns are far less collinear than f and f^2 would be.
    weight_sum = weights.sum()
    raw_by_key: dict[str, np.ndarray] = {}
    centers: dict[str, float] = {}
    for key in terms:
        if key == TERM_CONST.key:
            centers[key] = 0.0
            continue
        raw = (
            analytic
            if key == TERM_ANALYTIC_LEVEL.key
            else pd.to_numeric(used[key], errors="coerce").to_numpy(float)
        )
        raw_by_key[key] = raw
        centers[key] = float((weights * raw).sum() / weight_sum)

    columns: list[np.ndarray] = []
    spreads: list[float] = []
    for key, power in design_columns:
        if key == TERM_CONST.key:
            columns.append(np.ones(n))
            spreads.append(1.0)
            continue
        values = (raw_by_key[key] - centers[key]) ** power
        spread = float(np.sqrt((weights * values**2).sum() / weight_sum))
        if not np.isfinite(spread) or spread == 0.0:
            spread = 1.0
        columns.append(values / spread)
        spreads.append(spread)

    design = np.column_stack(columns)
    sqrt_w = np.sqrt(weights)
    design_w = design * sqrt_w[:, None]
    target_w = residual_target * sqrt_w

    theta_scaled, *_ = np.linalg.lstsq(design_w, target_w, rcond=None)
    gram = design_w.T @ design_w
    gram_inv = np.linalg.pinv(gram)
    condition = float(np.linalg.cond(design_w))

    fitted_target = design @ theta_scaled
    residual = residual_target - fitted_target
    # The fitted value in the channel's own units, whichever form was used, so
    # every consumer - the scatter plot, the report, the LOO score - speaks
    # degrees C and kPa rather than sometimes speaking dimensionless ratio.
    fitted_actual = (
        analytic * fitted_target if form == FORM_RATIO else analytic + fitted_target
    )
    dof = n - p
    weighted_ssr = float((weights * residual**2).sum())
    scale = weighted_ssr / dof
    cov_scaled = scale * gram_inv

    # Undo the column scaling so theta and cov are in per-raw-feature units.
    unscale = np.array([1.0 / spread for spread in spreads])
    theta = theta_scaled * unscale
    cov = cov_scaled * np.outer(unscale, unscale)

    se_theta = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, dof)) if dof > 0 else np.nan

    estimates = tuple(
        TermEstimate(
            key=key,
            label=_term_label(key, power),
            value=float(theta[i]),
            se=float(se_theta[i]),
            ci_lo=float(theta[i] - t_crit * se_theta[i]),
            ci_hi=float(theta[i] + t_crit * se_theta[i]),
            center=centers[key],
            spread=spreads[i],
            power=power,
        )
        for i, (key, power) in enumerate(design_columns)
    )

    # Reported in the channel's own units - the miss on the measurement, not on
    # whatever intermediate quantity was regressed. For the additive form these
    # are the same number; for the ratio form they differ by a factor of the
    # analytic estimate, and quoting a dimensionless 0.002 as though it were
    # degrees C would badly misrepresent the fit.
    residual_rms = float(np.sqrt(np.mean((actual - fitted_actual) ** 2)))
    # In target units, because this is what the prediction interval is built
    # from and it has to match the covariance it is added to.
    weighted_residual_rms = float(
        np.sqrt((weights * residual**2).sum() / weights.sum())
    )

    warnings: list[str] = []
    if dof < 3:
        warnings.append(
            f"Only {dof} residual degree(s) of freedom. Intervals are wide by "
            "construction and the fit cannot be checked against itself."
        )
    if condition > 30.0:
        warnings.append(
            f"Design condition number {condition:.0f}: the feature terms are "
            "close to collinear over this anchor set, so their coefficients "
            "trade off against each other and are not separately identified."
        )
    for estimate in estimates:
        if estimate.key == TERM_CONST.key:
            continue
        if estimate.spans_zero:
            warnings.append(
                f"{estimate.label}: the interval spans zero, so this anchor set "
                "does not show the correction depending on it. Consider turning "
                "the term off rather than paying a parameter for it."
            )
    if scale > 4.0:
        warnings.append(
            f"Weighted residual variance is {scale:.1f}x what the reported "
            "block standard errors predict. Either the model is missing "
            "structure, or the standard errors understate the real "
            "block-to-block repeatability."
        )
    elif scale < 0.05:
        warnings.append(
            f"Weighted residual variance is {scale:.3g}x what the reported "
            "block standard errors predict - the fit tracks the anchors far "
            "more closely than their own noise, which at this sample size "
            "usually means the parameters are absorbing that noise."
        )

    return ChannelCalibration(
        channel=channel,
        unit=schema.unit_of(channel),
        terms=terms,
        theta=theta,
        cov=cov,
        centers=centers,
        n_anchor=n,
        dof=dof,
        alpha=alpha,
        scale=float(scale),
        residual_rms=residual_rms,
        weighted_residual_rms=weighted_residual_rms,
        condition_number=condition,
        se_floor=floor,
        estimates=estimates,
        block_ids=tuple(int(b) for b in used["block_id"].to_numpy()),
        actual=actual,
        actual_se=actual_se,
        analytic=analytic,
        fitted=fitted_actual,
        weights=weights,
        warnings=tuple(warnings),
        excluded_notes=tuple(reasons),
        orders=resolved_orders,
        columns=design_columns,
        form=form,
    )


# --------------------------------------------------------------------------
# Choosing the shape of each term
# --------------------------------------------------------------------------
#
# Training error cannot decide this. Every extra parameter lowers the residual
# on the points it was fitted to - at five residual degrees of freedom a cubic
# will always look better than a linear, and none of that is evidence it
# describes anything real. So the criterion is leave-one-block-out
# cross-validation: drop an anchor, refit without it, and see how far the
# refitted correction misses the anchor it never saw. Eight anchors means eight
# refits of a three-parameter problem, which is instant.
#
# Training RMS is still computed and shown, precisely so the gap between it and
# the cross-validated error is visible. That gap is the overfitting.


def orders_label(orders: Mapping[str, int]) -> str:
    if not orders:
        return "offset only"
    return ", ".join(
        f"{TERM_BY_KEY[key].label if key in TERM_BY_KEY else key} {order_label(order)}"
        for key, order in orders.items()
    )


@dataclass(frozen=True)
class OrderCandidate:
    """One combination of per-term polynomial orders, and how it scored."""

    orders: dict[str, int]
    n_parameters: int
    loo_rmse: float
    training_rmse: float
    aic: float
    bic: float
    feasible: bool
    note: str = ""
    # Which variable and form this candidate was fitted with. Constant across
    # one `OrderScan`; the whole point of them in the v1.5 full search is that
    # candidates from different scans end up in one ranked list and have to
    # carry where they came from.
    variable: str = ENGINEERED_PAIR_KEY
    form: str = FORM_ADDITIVE

    @property
    def label(self) -> str:
        return orders_label(self.orders)

    @property
    def full_label(self) -> str:
        """Variable, shape and form, for a list that mixes all three."""
        shape = ", ".join(
            order_label(order) for order in self.orders.values()
        ) or "offset only"
        form = " ratio" if self.form == FORM_RATIO else ""
        return f"{variable_label(self.variable)} ({shape}){form}"

    @property
    def is_linear(self) -> bool:
        return all(order == 1 for order in self.orders.values())


@dataclass
class OrderScan:
    """Every candidate order combination for one channel, scored and ranked."""

    channel: str
    unit: str
    n_anchor: int
    candidates: tuple[OrderCandidate, ...]
    loo_best: dict[str, int]
    suggested: dict[str, int]
    guardrail_applied: bool
    guardrail_note: str = ""
    form: str = FORM_ADDITIVE
    variable: str = ENGINEERED_PAIR_KEY
    # What Auto picks: the simplest shape within `auto_tolerance` of the best
    # cross-validated error, before the small-n guardrail is applied on top.
    auto_best: dict[str, int] = field(default_factory=dict)
    auto_note: str = ""
    auto_tolerance: float = AUTO_ORDER_TOLERANCE

    @property
    def auto_orders(self) -> dict[str, int]:
        """What Auto resolves to, guardrail included.

        The guardrail is not a separate opinion Auto may overrule - below
        `MIN_ANCHORS_FOR_QUADRATIC` the order choice is itself unstable,
        so Auto is linear there whatever the tolerance rule found.
        """
        if self.guardrail_applied:
            return dict(self.linear)
        return dict(self.auto_best) if self.auto_best else dict(self.linear)

    @property
    def auto_explanation(self) -> str:
        if self.guardrail_applied:
            return (
                f"Auto selected linear: {self.n_anchor} anchors is below the "
                f"{MIN_ANCHORS_FOR_QUADRATIC} needed before a linear-vs-quadratic "
                "ranking is stable enough to act on, so the tolerance rule is "
                "not consulted. Quadratic can still be selected by hand, and "
                "dev-training's held-out split is the way to check it."
            )
        return self.auto_note

    @property
    def linear(self) -> dict[str, int]:
        return {key: 1 for key in self.loo_best}

    def candidate_for(self, orders: Mapping[str, int]) -> OrderCandidate | None:
        wanted = {key: int(value) for key, value in orders.items()}
        for candidate in self.candidates:
            if candidate.orders == wanted:
                return candidate
        return None

    def table(self) -> pd.DataFrame:
        """Side-by-side scores, so the evidence for a higher order is visible.

        Sorted by cross-validated error, with the training error alongside: a
        candidate that wins on training and loses on LOO-CV is fitting noise,
        and that has to be readable at a glance rather than inferred.
        """
        rows = []
        best = min(
            (c.loo_rmse for c in self.candidates if c.feasible and np.isfinite(c.loo_rmse)),
            default=np.nan,
        )
        for candidate in sorted(
            self.candidates,
            key=lambda c: (not c.feasible, c.loo_rmse, c.n_parameters),
        ):
            rows.append(
                {
                    "orders": candidate.label,
                    "parameters": candidate.n_parameters,
                    "LOO-CV RMSE": candidate.loo_rmse,
                    "vs best": (
                        candidate.loo_rmse / best
                        if np.isfinite(best) and best > 0 and np.isfinite(candidate.loo_rmse)
                        else np.nan
                    ),
                    "training RMSE": candidate.training_rmse,
                    "AIC": candidate.aic,
                    "BIC": candidate.bic,
                    "usable": "yes" if candidate.feasible else "no",
                    "note": candidate.note,
                }
            )
        return pd.DataFrame(rows)


@dataclass
class OrderSelection:
    scans: dict[str, OrderScan]

    def chosen_orders(self) -> dict[str, dict[str, int]]:
        return {channel: dict(scan.suggested) for channel, scan in self.scans.items()}

    def guardrail_channels(self) -> tuple[str, ...]:
        return tuple(
            channel for channel, scan in self.scans.items() if scan.guardrail_applied
        )


def _features_from_row(row: pd.Series, terms: tuple[str, ...]) -> dict:
    """Every feature a fit needs, taken off one anchor row.

    Driven by the fit's own term list rather than a fixed pair, so a raw input
    column is picked up the same way an engineered feature is.
    """
    return {
        key: row.get(key, np.nan)
        for key in terms
        if key not in (TERM_CONST.key, TERM_ANALYTIC_LEVEL.key)
    }


def _predict_value(fit: ChannelCalibration, row: pd.Series) -> float:
    """Corrected value for one anchor row, without the interval machinery.

    Always returns the reconstructed *measurement*, whichever form was fitted,
    so a cross-validation score is in the channel's own units and a ratio fit
    and an additive fit can be compared against each other directly.
    """
    analytic = float(row[schema.analytic_name(fit.channel)])
    design_row = fit._design_row(_features_from_row(row, fit.terms), analytic)
    target = float(design_row @ fit.theta)
    return analytic * target if fit.form == FORM_RATIO else analytic + target


def leave_one_out_rmse(
    anchors: pd.DataFrame,
    channel: str,
    terms: tuple[str, ...] = DEFAULT_TERMS,
    orders: Mapping[str, int] | None = None,
    alpha: float = DEFAULT_ALPHA,
    se_floor_fraction: float = DEFAULT_SE_FLOOR_FRACTION,
    form: str | None = None,
) -> float:
    """Weighted RMS error, in the channel's units, on anchors never seen.

    Each anchor is held out in turn, the correction is refitted on the rest, and
    the held-out block is predicted. The weighting is the same inverse-variance
    weighting the fit uses, computed once on the full set so the score is
    comparable across candidate orders - and, because the error is always taken
    on the measurement, across forms and across variables too.
    """
    form = resolve_form(channel, form)
    ok, _ = usable_mask(anchors, channel, terms, form)
    used = anchors[ok].reset_index(drop=True)
    n = len(used)
    if n < 3:
        return float("inf")

    reference = fit_channel(
        used, channel, terms, alpha, se_floor_fraction, orders, form
    )
    if n - 1 <= len(reference.columns):
        # One fewer anchor would leave the refit with no residual degrees of
        # freedom, so the score would be meaningless rather than merely poor.
        return float("inf")

    # The fit weights are 1 / se(target)^2. The errors below are on the
    # measurement, so for a ratio fit the weights need the same step into the
    # channel's units: se(actual) = |analytic| * se(ratio), hence w / analytic^2.
    # Without this a ratio score would weight kPa errors by a dimensionless
    # precision and would not be comparable with an additive one.
    weights = reference.weights
    if form == FORM_RATIO:
        analytic = pd.to_numeric(
            used[schema.analytic_name(channel)], errors="coerce"
        ).to_numpy(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            weights = np.where(analytic != 0, weights / analytic**2, weights)
        weights = np.where(np.isfinite(weights), weights, 0.0)

    squared, weight_sum = 0.0, 0.0
    for i in range(n):
        train = used.drop(index=i)
        try:
            fold = fit_channel(
                train, channel, terms, alpha, se_floor_fraction, orders, form
            )
        except CalibrationError:
            return float("inf")
        row = used.iloc[i]
        try:
            predicted = _predict_value(fold, row)
        except CalibrationError:
            return float("inf")
        error = float(row[f"actual_{channel}"]) - predicted
        if not np.isfinite(error):
            return float("inf")
        squared += weights[i] * error**2
        weight_sum += weights[i]
    if weight_sum <= 0:
        return float("inf")
    return float(np.sqrt(squared / weight_sum))


def scan_orders(
    anchors: pd.DataFrame,
    channel: str,
    terms: tuple[str, ...] = DEFAULT_TERMS,
    alpha: float = DEFAULT_ALPHA,
    se_floor_fraction: float = DEFAULT_SE_FLOOR_FRACTION,
    max_order: int = MAX_ORDER,
    min_anchors: int = MIN_ANCHORS_FOR_QUADRATIC,
    form: str | None = None,
    tolerance: float = AUTO_ORDER_TOLERANCE,
) -> OrderScan:
    """Score every per-term order combination for one channel.

    `terms` is whatever the channel is being fitted against - the two
    engineered features, or a single chosen variable. The scan machinery does
    not care which: it builds a design from however many columns it is given.

    `max_order` may lower the ceiling but never raise it above `MAX_ORDER`. A
    caller asking for a cubic scan gets a quadratic one rather than a candidate
    list containing a shape the app will not fit - the enumeration and the fit
    have to agree about what exists, or the search ranks options that cannot be
    adopted.
    """
    form = resolve_form(channel, form)
    max_order = max(1, min(int(max_order), MAX_ORDER))
    scan_variable = variable_of(terms)
    active = [key for key in terms if key != TERM_CONST.key]
    ok, _ = usable_mask(anchors, channel, terms, form)
    n_anchor = int(ok.sum())

    # A feature that does not vary across these anchors cannot have a
    # coefficient identified against it, and the rank-deficient fit that
    # results scores *better* on cross-validation than any honest candidate.
    # Refused here, once, so no shape of it reaches the ranking.
    used_rows = anchors[ok]
    degenerate = [
        key
        for key in active
        if key in used_rows.columns
        and feature_is_degenerate(
            pd.to_numeric(used_rows[key], errors="coerce").to_numpy(float)
        )
    ]

    candidates: list[OrderCandidate] = []
    combinations = (
        list(itertools.product(range(1, max_order + 1), repeat=len(active)))
        if active
        else [()]
    )
    for combo in combinations:
        orders = dict(zip(active, combo))
        n_parameters = 1 + sum(orders.values())
        if degenerate:
            candidates.append(
                OrderCandidate(
                    orders=orders,
                    n_parameters=n_parameters,
                    loo_rmse=float("inf"),
                    training_rmse=float("nan"),
                    aic=float("nan"),
                    bic=float("nan"),
                    feasible=False,
                    variable=scan_variable,
                    form=form,
                    note=(
                        f"{', '.join(degenerate)} does not vary across these "
                        "anchors, so no coefficient can be identified against it."
                    ),
                )
            )
            continue
        feasible = n_anchor >= n_parameters + 2
        if not feasible:
            candidates.append(
                OrderCandidate(
                    orders=orders,
                    n_parameters=n_parameters,
                    loo_rmse=float("inf"),
                    training_rmse=float("nan"),
                    aic=float("nan"),
                    bic=float("nan"),
                    feasible=False,
                    variable=scan_variable,
                    form=form,
                    note=(
                        f"{n_parameters} parameters needs at least "
                        f"{n_parameters + 2} anchors; there are {n_anchor}."
                    ),
                )
            )
            continue
        try:
            fit = fit_channel(
                anchors, channel, terms, alpha, se_floor_fraction, orders, form
            )
        except CalibrationError as exc:
            candidates.append(
                OrderCandidate(
                    orders=orders,
                    n_parameters=n_parameters,
                    loo_rmse=float("inf"),
                    training_rmse=float("nan"),
                    aic=float("nan"),
                    bic=float("nan"),
                    feasible=False,
                    variable=scan_variable,
                    form=form,
                    note=str(exc),
                )
            )
            continue
        if fit.condition_number > MAX_SEARCH_CONDITION_NUMBER:
            # Not caught by the spread test - each feature moves, but not
            # independently enough for the design to have an inverse worth the
            # name. Same treatment: excluded from the ranking, reason recorded.
            candidates.append(
                OrderCandidate(
                    orders=orders,
                    n_parameters=n_parameters,
                    loo_rmse=float("inf"),
                    training_rmse=fit.residual_rms,
                    aic=float("nan"),
                    bic=float("nan"),
                    feasible=False,
                    variable=scan_variable,
                    form=form,
                    note=(
                        f"design condition number {fit.condition_number:.3g}: the "
                        "coefficients are not separately identifiable on these "
                        "anchors."
                    ),
                )
            )
            continue
        loo = leave_one_out_rmse(
            anchors, channel, terms, orders, alpha, se_floor_fraction, form
        )
        # Gaussian-likelihood AIC/BIC on the weighted residual sum of squares.
        # Comparable between candidates for one channel; not across channels,
        # since the weights carry each channel's own units.
        weighted_ssr = float((fit.weights * (fit.actual - fit.fitted) ** 2).sum())
        n = fit.n_anchor
        log_likelihood_term = n * np.log(weighted_ssr / n) if weighted_ssr > 0 else -np.inf
        candidates.append(
            OrderCandidate(
                orders=orders,
                n_parameters=n_parameters,
                loo_rmse=loo,
                training_rmse=fit.residual_rms,
                aic=float(log_likelihood_term + 2 * n_parameters),
                bic=float(log_likelihood_term + n_parameters * np.log(n)),
                feasible=True,
                variable=scan_variable,
                form=form,
                note="",
            )
        )

    usable = [c for c in candidates if c.feasible and np.isfinite(c.loo_rmse)]
    linear = {key: 1 for key in active}
    if usable:
        # Ties go to the simpler model: identical cross-validated error is not a
        # reason to spend a degree of freedom.
        best = min(usable, key=lambda c: (round(c.loo_rmse, 12), c.n_parameters))
        loo_best = dict(best.orders)
    else:
        loo_best = dict(linear)

    auto_best, auto_note = _auto_choice(usable, linear, tolerance)

    guardrail = n_anchor < min_anchors
    note = ""
    if guardrail:
        chosen = dict(linear)
        if loo_best != linear:
            note = (
                f"{n_anchor} anchor(s) is below the {min_anchors} this app asks "
                "for before it will trust an order choice. Cross-validation "
                f"prefers {orders_label(loo_best)}, "
                "but at this sample size that ranking can turn on a single "
                "block - so linear is selected by default. Override it above if "
                "you have a physical reason to."
            )
        else:
            note = (
                f"{n_anchor} anchor(s) is below the {min_anchors} this app asks "
                "for before it will trust an order choice. Cross-validation "
                "happens to agree with linear here, which is the safe default "
                "anyway."
            )
    else:
        # The default selection is Auto's, not argmin's.
        chosen = dict(auto_best)

    return OrderScan(
        channel=channel,
        unit=schema.unit_of(channel),
        n_anchor=n_anchor,
        candidates=tuple(candidates),
        loo_best=loo_best,
        suggested=chosen,
        guardrail_applied=guardrail,
        guardrail_note=note,
        form=form,
        variable=variable_of(terms),
        auto_best=auto_best,
        auto_note=auto_note,
        auto_tolerance=tolerance,
    )


def _auto_choice(
    usable: list[OrderCandidate],
    linear: dict[str, int],
    tolerance: float,
) -> tuple[dict[str, int], str]:
    """The simplest shape whose CV error is within `tolerance` of the best.

    This is the whole of Auto. It is not argmin: among candidates that the
    cross-validation cannot meaningfully separate, the one with the fewest
    parameters wins, because at these anchor counts a few percent of CV error
    is noise and a parameter is not.
    """
    if not usable:
        return dict(linear), "No candidate shape was usable; falling back to linear."

    best = min(usable, key=lambda c: c.loo_rmse)
    if not np.isfinite(best.loo_rmse) or best.loo_rmse <= 0:
        return dict(linear), "Cross-validated error was not usable; linear selected."

    ceiling = best.loo_rmse * (1.0 + tolerance)
    within = [c for c in usable if c.loo_rmse <= ceiling]
    # Fewest parameters first, and among equals the lower CV error.
    pick = min(within, key=lambda c: (c.n_parameters, c.loo_rmse))

    if pick.orders == best.orders:
        note = (
            f"Auto selected {pick.label}: it has both the lowest "
            f"cross-validated error ({format_number(pick.loo_rmse)}) and the "
            "fewest parameters among the candidates within tolerance."
        )
    else:
        improvement = (pick.loo_rmse - best.loo_rmse) / pick.loo_rmse * 100.0
        note = (
            f"Auto selected {pick.label}: {best.label}'s CV error was only "
            f"{improvement:.0f}% lower, within the {tolerance:.0%} tolerance "
            "for preferring the simpler shape "
            f"({pick.n_parameters} parameters against {best.n_parameters})."
        )
    return dict(pick.orders), note


# --------------------------------------------------------------------------
# The full Auto search: variable and shape together (v1.5)
# --------------------------------------------------------------------------
#
# v1.4's Auto chose a shape for a variable someone else had picked. v1.5 lets it
# choose the variable too, which is a larger search - up to nine variables times
# three shapes, times whichever forms the channel is allowed - and a larger
# search is a larger opportunity to win by chance.
#
# Two things hold that in check, and neither is new machinery:
#
#   * The selection rule is still the simplicity tolerance, now read against
#     parameter count rather than polynomial order. A two-feature fit costs
#     three parameters against a single-variable linear's two, so it has to beat
#     it by more than the tolerance to be worth the third.
#   * The anchor guardrail now governs the whole search rather than just the
#     shape axis: below `MIN_ANCHORS_FOR_QUADRATIC`, only linear
#     candidates are in the running, across every variable.
#
# Neither of those detects the risk that the *search itself* is tuned to one
# run's geometry sweep. Nothing computed inside the anchor set can: LOO-CV
# proves a candidate generalises to the other blocks of the same run, which is
# a different claim. That is what `dev_training.py`'s held-out split is for.


@dataclass
class AutoSearch:
    """Every (variable, shape, form) candidate for one channel, ranked."""

    channel: str
    unit: str
    n_anchor: int
    candidates: tuple[OrderCandidate, ...]
    best: OrderCandidate | None
    chosen: OrderCandidate | None
    guardrail_applied: bool
    tolerance: float
    note: str = ""
    # The per-variable scans the candidates came from, kept so the existing
    # per-variable detail views still have something to show.
    scans: dict[str, OrderScan] = field(default_factory=dict)

    @property
    def chosen_variable(self) -> str:
        return self.chosen.variable if self.chosen else ENGINEERED_PAIR_KEY

    @property
    def chosen_orders(self) -> dict[str, int]:
        return dict(self.chosen.orders) if self.chosen else {}

    @property
    def chosen_form(self) -> str:
        return self.chosen.form if self.chosen else FORM_ADDITIVE

    @property
    def chosen_terms(self) -> tuple[str, ...]:
        return terms_for_variable(self.chosen_variable)

    @property
    def usable(self) -> list[OrderCandidate]:
        """Candidates that fitted and produced a finite score."""
        return [
            c for c in self.candidates if c.feasible and np.isfinite(c.loo_rmse)
        ]

    @property
    def eligible(self) -> list[OrderCandidate]:
        """Candidates the selection was actually allowed to choose from.

        Below the anchor guardrail that is the linear subset - which is why the
        table has to mark eligibility separately from usability, or the top of
        a ranked list looks like it was passed over for no reason.
        """
        usable = self.usable
        if self.guardrail_applied:
            return [c for c in usable if c.is_linear]
        return usable

    def table(self, limit: int | None = None) -> pd.DataFrame:
        """Every candidate across every variable, ranked by cross-validated error.

        Sorted so the winner's margin over the rest is visible - with two or
        three dozen candidates, "did this win by a nose?" is the question worth
        being able to answer at a glance, and `vs best` answers it.
        """
        rows = []
        eligible = {id(c) for c in self.eligible}
        best = min((c.loo_rmse for c in self.eligible), default=np.nan)
        ordered = sorted(
            self.candidates,
            key=lambda c: (not c.feasible, c.loo_rmse, c.n_parameters),
        )
        if limit is not None:
            ordered = ordered[:limit]
        for candidate in ordered:
            if not candidate.feasible:
                status = "no - did not fit"
            elif id(candidate) not in eligible:
                status = "no - above linear, below the anchor guardrail"
            else:
                status = "yes"
            rows.append(
                {
                    "candidate": candidate.full_label,
                    "variable": variable_label(candidate.variable),
                    "parameters": candidate.n_parameters,
                    "LOO-CV RMSE": candidate.loo_rmse,
                    "vs best eligible": (
                        candidate.loo_rmse / best
                        if np.isfinite(best) and best > 0 and np.isfinite(candidate.loo_rmse)
                        else np.nan
                    ),
                    "training RMSE": candidate.training_rmse,
                    "in use": "yes" if candidate is self.chosen else "",
                    "eligible": status,
                    "note": candidate.note,
                }
            )
        return pd.DataFrame(rows)


def search_auto(
    anchors: pd.DataFrame,
    channel: str,
    variables: tuple[str, ...] | None = None,
    alpha: float = DEFAULT_ALPHA,
    se_floor_fraction: float = DEFAULT_SE_FLOOR_FRACTION,
    max_order: int = MAX_ORDER,
    min_anchors: int = MIN_ANCHORS_FOR_QUADRATIC,
    tolerance: float = AUTO_ORDER_TOLERANCE,
    search_forms: bool = True,
) -> AutoSearch:
    """Search every variable and shape for one channel, and pick one.

    `search_forms` enumerates the forms the channel is *allowed* - which for a
    gauge pressure channel is additive only, taken from `allowed_forms` rather
    than re-derived here, so the search cannot offer something the manual
    dropdown refuses. `max_order` is clamped to `MAX_ORDER` for the same reason:
    one ceiling, applied wherever candidates are generated.
    """
    max_order = max(1, min(int(max_order), MAX_ORDER))
    options = tuple(variables) if variables else (
        *SINGLE_VARIABLE_KEYS,
        ENGINEERED_PAIR_KEY,
    )
    forms = allowed_forms(channel) if search_forms else (
        DEFAULT_FORMS.get(channel, FORM_ADDITIVE),
    )

    candidates: list[OrderCandidate] = []
    scans: dict[str, OrderScan] = {}
    n_anchor = 0
    for variable in options:
        terms = terms_for_variable(variable)
        for form in forms:
            try:
                scan = scan_orders(
                    anchors,
                    channel,
                    terms,
                    alpha,
                    se_floor_fraction,
                    max_order,
                    min_anchors,
                    form,
                    tolerance,
                )
            except CalibrationError:
                continue
            n_anchor = max(n_anchor, scan.n_anchor)
            # One scan per variable is kept for the detail views; when both
            # forms were searched the channel's default form is the one shown.
            if variable not in scans or form == DEFAULT_FORMS.get(channel):
                scans[variable] = scan
            candidates.extend(scan.candidates)

    guardrail = n_anchor < min_anchors
    usable = [c for c in candidates if c.feasible and np.isfinite(c.loo_rmse)]
    if guardrail:
        # The guardrail governs the whole search now, not only the shape axis:
        # every variable is still in the running, but only at linear.
        usable = [c for c in usable if c.is_linear]

    best = min(usable, key=lambda c: c.loo_rmse) if usable else None
    chosen, note = _auto_pick(usable, best, tolerance, guardrail, n_anchor, min_anchors)

    return AutoSearch(
        channel=channel,
        unit=schema.unit_of(channel),
        n_anchor=n_anchor,
        candidates=tuple(candidates),
        best=best,
        chosen=chosen,
        guardrail_applied=guardrail,
        tolerance=tolerance,
        note=note,
        scans=scans,
    )


def _auto_pick(
    usable: list[OrderCandidate],
    best: OrderCandidate | None,
    tolerance: float,
    guardrail: bool,
    n_anchor: int,
    min_anchors: int,
) -> tuple[OrderCandidate | None, str]:
    """The simplest candidate within `tolerance` of the best, and why.

    Identical in spirit to v1.4's shape-only rule, with "simplest" now meaning
    fewest parameters across the whole search rather than lowest polynomial
    order within one variable. That is what stops a three-parameter two-feature
    fit taking the selection off a two-parameter single-variable one it beats
    by a couple of percent.
    """
    if not usable or best is None:
        return None, "No candidate was usable on these anchors."
    if not np.isfinite(best.loo_rmse) or best.loo_rmse <= 0:
        return best, "Cross-validated error was not usable; the best fit is shown."

    ceiling = best.loo_rmse * (1.0 + tolerance)
    within = [c for c in usable if c.loo_rmse <= ceiling]
    pick = min(within, key=lambda c: (c.n_parameters, c.loo_rmse))

    guard = (
        f" Only linear shapes were considered: {n_anchor} anchors is below the "
        f"{min_anchors} needed before a shape choice is stable enough to act on."
        if guardrail
        else ""
    )
    if pick is best:
        note = (
            f"Auto selected {pick.full_label}: lowest cross-validated error "
            f"({format_number(pick.loo_rmse)}) of {len(usable)} candidate(s), "
            f"and nothing simpler came within {tolerance:.0%} of it.{guard}"
        )
    else:
        margin = (pick.loo_rmse - best.loo_rmse) / pick.loo_rmse * 100.0
        note = (
            f"Auto selected {pick.full_label}: {best.full_label}'s CV error was "
            f"only {margin:.0f}% lower, within the {tolerance:.0%} tolerance for "
            f"preferring the simpler fit ({pick.n_parameters} parameters "
            f"against {best.n_parameters}).{guard}"
        )
    return pick, note


def search_auto_all(
    anchors: Mapping[str, pd.DataFrame],
    alpha: float = DEFAULT_ALPHA,
    se_floor_fraction: float = DEFAULT_SE_FLOOR_FRACTION,
    min_anchors: int = MIN_ANCHORS_FOR_QUADRATIC,
    tolerance: float = AUTO_ORDER_TOLERANCE,
    search_forms: bool = True,
) -> dict[str, AutoSearch]:
    """Run the full search for every channel that has anchors."""
    out: dict[str, AutoSearch] = {}
    for channel in ordered_channels(anchors):
        table = anchors.get(channel)
        if table is None or table.empty:
            continue
        try:
            out[channel] = search_auto(
                table,
                channel,
                alpha=alpha,
                se_floor_fraction=se_floor_fraction,
                min_anchors=min_anchors,
                tolerance=tolerance,
                search_forms=search_forms,
            )
        except CalibrationError:
            continue
    return out


def select_orders(
    anchors: pd.DataFrame | Mapping[str, pd.DataFrame],
    terms: tuple[str, ...] = DEFAULT_TERMS,
    alpha: float = DEFAULT_ALPHA,
    se_floor_fraction: float = DEFAULT_SE_FLOOR_FRACTION,
    min_anchors: int = MIN_ANCHORS_FOR_QUADRATIC,
    app_version: str = "",
    variables: Mapping[str, str] | None = None,
    forms: Mapping[str, str] | None = None,
    preform_schema: schema.PreformSchema | None = None,
) -> OrderSelection:
    """Scan orders for every channel that has an anchor table.

    `variables` gives each channel the single variable to fit against; a channel
    absent from it falls back to `terms`. `forms` likewise overrides the
    additive/ratio choice per channel.
    """
    if isinstance(anchors, pd.DataFrame):
        base = preform_schema or schema.NONNESTED_SCHEMA
        tables = {channel: anchors for channel in base.setpoint_names}
    else:
        tables = dict(anchors)

    scans: dict[str, OrderScan] = {}
    for channel in ordered_channels(tables, preform_schema):
        table = tables.get(channel)
        if table is None or table.empty:
            continue
        channel_terms = terms
        if variables and channel in variables:
            channel_terms = terms_for_variable(variables[channel])
        try:
            scans[channel] = scan_orders(
                table,
                channel,
                channel_terms,
                alpha,
                se_floor_fraction,
                MAX_ORDER,
                min_anchors,
                (forms or {}).get(channel),
            )
        except CalibrationError:
            continue
    return OrderSelection(scans=scans)


# --------------------------------------------------------------------------
# The whole calibration
# --------------------------------------------------------------------------


@dataclass
class CalibrationSet:
    """Every channel's calibration, plus what it was fitted on."""

    preform_id: str
    created_utc: str
    n_anchor: int
    terms: tuple[str, ...]
    drawdown_mode: str
    preform_OD_mm: float | None
    alpha: float
    se_floor_fraction: float
    channels: dict[str, ChannelCalibration]
    anchor_table: pd.DataFrame
    # Since v1.1, each channel has its own steady-state blocks and therefore its
    # own anchors. `anchor_table` above is these stacked, kept for range checks
    # and for the human-readable CSV written beside the calibration.
    anchor_tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    source_files: dict[str, str] = field(default_factory=dict)
    notes: tuple[str, ...] = ()
    app_version: str = ""
    format_version: int = CALIBRATION_FORMAT

    @property
    def anchor_counts(self) -> dict[str, int]:
        return {name: fit.n_anchor for name, fit in self.channels.items()}

    @property
    def anchor_label(self) -> str:
        """Short badge for the UI, e.g. "Anchor (n=8)" or "Anchor (n=7-10)".

        A range rather than a single number whenever the channels rest on
        different numbers of blocks, which per-mapping extraction makes the
        normal case. Collapsing that to one figure would overstate the thinnest
        channel's evidence.
        """
        counts = sorted(set(self.anchor_counts.values()))
        if not counts:
            return f"Anchor (n={self.n_anchor})"
        if len(counts) == 1:
            return f"Anchor (n={counts[0]})"
        return f"Anchor (n={counts[0]}-{counts[-1]})"

    def anchor_label_for(self, channel: str) -> str:
        fit = self.channels.get(channel)
        return f"Anchor (n={fit.n_anchor})" if fit else self.anchor_label

    def anchors_for(self, channel: str) -> pd.DataFrame:
        """The anchor rows behind one channel, falling back to the stacked set."""
        table = self.anchor_tables.get(channel)
        if table is not None and not table.empty:
            return table
        return self.anchor_table

    def describe(self) -> str:
        when = self.created_utc.replace("T", " ")[:19]
        mode = (
            "geometric draw-down"
            if self.drawdown_mode == DRAWDOWN_GEOMETRIC
            else "kinematic draw-down"
        )
        # Reports the variables actually fitted against rather than the global
        # term list, which since v1.3 is only a fallback - each channel now
        # chooses its own.
        shapes = ", ".join(
            f"{name} vs {fit.variable_label}" for name, fit in self.channels.items()
        )
        return (
            f"{self.anchor_label} - {self.preform_id}, {mode}, "
            f"fitted {when} UTC ({shapes})"
        )

    @property
    def forms(self) -> dict[str, str]:
        return {name: fit.form for name, fit in self.channels.items()}

    @property
    def variables(self) -> dict[str, str]:
        return {name: fit.variable for name, fit in self.channels.items()}

    @property
    def preform_schema(self) -> schema.PreformSchema:
        """The column bundle this calibration was fitted against.

        Resolved from `preform_id` rather than stored, so no persisted file
        needs migrating - see `schema.schema_for_preform`.
        """
        return schema.schema_for_preform(self.preform_id)

    def features_for(self, inputs: dict) -> dict:
        """Compute the fit features for a target geometry.

        Uses whichever draw-down definition the calibration was fitted with, so
        a prediction can never be built from a differently-defined feature than
        the one the coefficient means.
        """
        # The raw inputs pass straight through: since v1.3 a channel can be
        # fitted against one of them directly, and the Predict tab already
        # collects every one of them from the operator. Which inputs those are
        # is a property of the geometry, resolved from the preform id this
        # calibration was saved under - so a pre-v1.9 file still resolves to
        # the non-nested schema and reconstructs exactly the features it did.
        preform_schema = self.preform_schema
        features = {
            spec.name: inputs.get(spec.name, np.nan)
            for spec in preform_schema.features
        }
        # One wall ratio for the single-capillary geometry, three for the
        # nested one. The non-nested key is still `cap_wall_ratio`, which is
        # what stored coefficients are indexed by.
        features.update(preform_schema.compute_wall_ratios(inputs))
        if self.drawdown_mode == DRAWDOWN_GEOMETRIC:
            preform_od = inputs.get("preform_OD_mm", self.preform_OD_mm)
            if preform_od is None or not np.isfinite(float(preform_od)):
                raise CalibrationError(
                    "This calibration was fitted with the geometric draw-down "
                    "ratio, so a preform outer diameter is required to predict "
                    "with it."
                )
            ratio = schema.geometric_drawdown_ratio(
                float(preform_od), inputs.get("fibre_OD_um", np.nan)
            )
        else:
            ratio = schema.kinematic_drawdown_ratio(
                inputs.get(schema.analytic_name("draw_speed_m_min"), np.nan),
                inputs.get("feed_speed_mm_min", np.nan),
            )
        features["drawdown_ratio"] = ratio
        features[TERM_DRAWDOWN.key] = (
            float(np.log(ratio)) if np.isfinite(ratio) and ratio > 0 else np.nan
        )
        return features

    def predict(self, inputs: dict) -> dict[str, ChannelPrediction]:
        """Calibrated setpoints for one target.

        `inputs` carries the target geometry and process inputs plus one
        `analytic_*` value per channel - the operator's own run of the fast
        estimator. The calibration corrects those values; it does not replace
        them and cannot produce a setpoint without them.
        """
        features = self.features_for(inputs)
        out: dict[str, ChannelPrediction] = {}
        for channel, calibration in self.channels.items():
            analytic = inputs.get(schema.analytic_name(channel), np.nan)
            out[channel] = calibration.predict(features, analytic, self.alpha)
        return out


def fit_calibration(
    anchors: pd.DataFrame | Mapping[str, pd.DataFrame],
    preform_id: str,
    drawdown_mode: str,
    preform_OD_mm: float | None = None,
    terms: tuple[str, ...] = DEFAULT_TERMS,
    alpha: float = DEFAULT_ALPHA,
    se_floor_fraction: float = DEFAULT_SE_FLOOR_FRACTION,
    source_files: dict[str, str] | None = None,
    notes: tuple[str, ...] = (),
    app_version: str = "",
    orders: Mapping[str, Mapping[str, int]] | None = None,
    variables: Mapping[str, str] | None = None,
    forms: Mapping[str, str] | None = None,
    preform_schema: schema.PreformSchema | None = None,
) -> CalibrationSet:
    """Fit every setpoint channel.

    `anchors` is either one table shared by every channel (v1.0 behaviour, and
    what the tests and the fallback path use) or a mapping of channel to its
    own anchor table, which is what per-mapping extraction produces.

    `orders` optionally gives, per channel, the polynomial order to use for each
    term. Channels absent from it fall back to linear.

    `variables` gives each channel the single variable to fit against - a raw
    input, an engineered feature, or `ENGINEERED_PAIR_KEY` for the original
    two-feature shape. `forms` overrides additive/ratio per channel.
    """
    if isinstance(anchors, pd.DataFrame):
        base = preform_schema or schema.NONNESTED_SCHEMA
        tables = {channel: anchors for channel in base.setpoint_names}
    else:
        tables = dict(anchors)

    # Validated before any fitting starts. A channel whose data defeats it is a
    # result, recorded per channel below; a channel asked for a form it may
    # never have is a mistake in the request, and silently returning a set with
    # that channel missing would look like the former.
    for channel, requested in (forms or {}).items():
        resolve_form(channel, requested)
    # Orders get the same treatment, and for the same reason. Since v1.6 an
    # order above `MAX_ORDER` is not a shape this app has, so asking for one is
    # a bad request rather than a channel that happened not to fit - and being
    # handed a calibration set that is quietly missing that channel is exactly
    # how a removed option looks like it half-worked.
    for channel, requested_orders in (orders or {}).items():
        for key, order in (requested_orders or {}).items():
            resolve_order(key, order)

    channels: dict[str, ChannelCalibration] = {}
    failures: list[str] = []
    for channel in ordered_channels(tables, preform_schema):
        table = tables.get(channel)
        if table is None or table.empty:
            failures.append(f"{channel}: no anchor table was produced for it.")
            continue
        channel_orders = (orders or {}).get(channel)
        channel_terms = terms
        if variables and channel in variables:
            channel_terms = terms_for_variable(variables[channel])
        channel_form = (forms or {}).get(channel)
        try:
            fit = fit_channel(
                table,
                channel,
                channel_terms,
                alpha,
                se_floor_fraction,
                orders=channel_orders,
                form=channel_form,
            )
            # The cross-validated error of the shape actually used, carried on
            # the fit so the report never has to quote a training figure alone.
            loo = leave_one_out_rmse(
                table, channel, channel_terms, fit.orders, alpha, se_floor_fraction,
                fit.form,
            )
            # And the same score for the two-feature engineered shape, under the
            # same form, so a single-variable fit can be reported as better or
            # worse than the thing it replaced rather than in isolation. This is
            # a simplification for interpretability - it is not guaranteed to
            # fit better, and the report must not imply that it does.
            baseline = float("nan")
            if variable_of(channel_terms) != ENGINEERED_PAIR_KEY:
                try:
                    baseline = leave_one_out_rmse(
                        table, channel, DEFAULT_TERMS, None, alpha,
                        se_floor_fraction, fit.form,
                    )
                except CalibrationError:
                    baseline = float("nan")
            channels[channel] = replace(
                fit, loo_rmse=loo, baseline_loo_rmse=baseline
            )
        except CalibrationError as exc:
            failures.append(f"{channel}: {exc}")

    if not channels:
        raise CalibrationError(
            "No channel could be fitted.\n" + "\n".join(failures)
        )

    n_anchor = max(c.n_anchor for c in channels.values())
    per_channel = {
        channel: table.reset_index(drop=True)
        for channel, table in tables.items()
        if channel in channels
    }
    stacked = combined_anchor_table(per_channel)
    return CalibrationSet(
        preform_id=preform_id,
        created_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        n_anchor=n_anchor,
        terms=tuple(dict.fromkeys((TERM_CONST.key, *terms))),
        drawdown_mode=drawdown_mode,
        preform_OD_mm=preform_OD_mm,
        alpha=alpha,
        se_floor_fraction=se_floor_fraction,
        channels=channels,
        anchor_table=stacked,
        anchor_tables=per_channel,
        source_files=dict(source_files or {}),
        notes=tuple(notes) + tuple(failures),
        app_version=app_version,
    )


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------


def save_calibration(
    calibration: CalibrationSet, path: Path, anchors_csv: Path | None = None
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibration, path)
    if anchors_csv is not None:
        # The anchor set is the whole evidential basis for every prediction the
        # app will make, so it is also written as plain CSV: readable without
        # this app, and reviewable by whoever inherits the calibration.
        calibration.anchor_table.to_csv(anchors_csv, index=False)


def load_calibration(path: Path) -> CalibrationSet | None:
    """Load a saved calibration, or None if there is not one yet."""
    if not path.exists():
        return None
    loaded = joblib.load(path)
    if not isinstance(loaded, CalibrationSet):
        raise CalibrationError(
            f"{path.name} does not contain a calibration written by this app."
        )
    if getattr(loaded, "format_version", 0) != CALIBRATION_FORMAT:
        raise CalibrationError(
            f"{path.name} was written in calibration format "
            f"{getattr(loaded, 'format_version', 'unknown')}, but this app "
            f"reads format {CALIBRATION_FORMAT}. Refit from the anchor blocks."
        )
    return loaded


# --------------------------------------------------------------------------
# Saying what was fitted, in words
# --------------------------------------------------------------------------
#
# The fit is carried out on centred features, because that is what keeps the
# offset interpretable and the normal equations well conditioned. Nobody reads
# an equation that way. `uncentred_polynomial` undoes the centring exactly - it
# is a change of basis, not an approximation - so the equation on screen is one
# an operator can substitute a raw measurement into and get the same number the
# app would.


def _binomial(n: int, k: int) -> float:
    from math import comb

    return float(comb(n, k))


def uncentred_polynomial(
    fit: ChannelCalibration, include: set[tuple[str, int]] | None = None
) -> tuple[float, dict[str, dict[int, float]]]:
    """Re-express the fit in raw, uncentred variables.

    Returns the constant term and, per variable, {power: coefficient}. The
    centring shows up as a contribution to the constant: expanding
    `theta_p * (x - c)^p` scatters `theta_p * C(p,j) * (-c)^(p-j)` across the
    lower powers, and the `j = 0` share of that lands on the intercept.

    `include` restricts which (variable, power) columns are expanded. Anything
    left out contributes *nothing at all* - not even to the constant - which is
    exactly equivalent to holding that variable at its centre, since
    `(x - c)^p` is zero there. That equivalence is the whole reason the fit is
    centred: an omitted term leaves the rest of the equation still reading
    correctly for a typical anchor. Folding an omitted term's intercept share in
    while dropping its slope would produce an equation that is wrong everywhere.
    """
    constant = 0.0
    by_variable: dict[str, dict[int, float]] = {}
    for index, (key, power) in enumerate(fit.columns):
        coefficient = float(fit.theta[index])
        if key == TERM_CONST.key:
            constant += coefficient
            continue
        if include is not None and (key, power) not in include:
            continue
        centre = float(fit.centers.get(key, 0.0))
        target = by_variable.setdefault(key, {})
        for j in range(power + 1):
            share = coefficient * _binomial(power, j) * (-centre) ** (power - j)
            if j == 0:
                constant += share
            else:
                target[j] = target.get(j, 0.0) + share
    return constant, by_variable


def format_number(value: float, significant: int = 4, strip: bool = True) -> str:
    """A number a human reads off a screen, not a float repr.

    Fixed notation while the magnitude is sane, because `0.00031` is far easier
    to act on at a draw tower than `3.1e-04`; scientific only once fixed
    notation would be absurd. `strip=False` keeps trailing zeros, which matters
    for a ratio near unity - "1" and "1.000" say different things about how
    close to no-correction the fit actually came.
    """
    if value is None or not np.isfinite(value):
        return "-"
    if value == 0:
        return "0"
    magnitude = abs(value)
    if magnitude >= 1e6 or magnitude < 1e-4:
        return f"{value:.{significant - 1}e}"
    decimals = max(0, significant - int(np.floor(np.log10(magnitude))) - 1)
    text = f"{value:.{decimals}f}"
    if strip and "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _monomial_text(key: str, power: int, coefficient: float) -> str:
    name = variable_label(key)
    body = name if power == 1 else f"{name}^{power}"
    sign = "-" if coefficient < 0 else "+"
    return f" {sign} {format_number(abs(coefficient))} x {body}"


@dataclass(frozen=True)
class EquationText:
    """The fitted function as a line of prose, plus what was left out of it."""

    equation: str
    excluded_note: str = ""
    form_note: str = ""

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.equation


def equation_text(fit: ChannelCalibration) -> EquationText:
    """The fitted correction, written out for someone who has to act on it.

    Terms whose confidence interval spans zero are left out of the line: this
    anchor set does not show the correction depending on them, and printing
    `+ 0.0 x log(draw-down ratio)` implies a dependence that was not found. What
    was dropped is named underneath rather than silently vanishing, and the
    prediction still uses the full fit - the line is a summary of the evidence,
    not a second model.
    """
    # Which (variable, power) coefficients survived their interval.
    kept: set[tuple[str, int]] = set()
    dropped: list[str] = []
    for estimate in fit.estimates:
        if estimate.key == TERM_CONST.key:
            continue
        if estimate.spans_zero:
            dropped.append(estimate.label)
        else:
            kept.add((estimate.key, estimate.power))

    # Expanded over the kept columns only, so the constant is the correction at
    # a typical anchor rather than an intercept extrapolated back to a variable
    # of zero that was then stripped of the slope leading to it.
    constant, by_variable = uncentred_polynomial(fit, include=kept)

    pieces: list[str] = []
    for key, powers in by_variable.items():
        for power in sorted(powers):
            coefficient = powers[power]
            if coefficient == 0.0:
                continue
            pieces.append(_monomial_text(key, power, coefficient))

    if fit.is_ratio:
        # A factor sits near 1, where four significant figures would round a
        # measured 1.0003 to a flat "1" and read as "no correction at all".
        constant_text = format_number(constant, significant=5, strip=False)
        body = constant_text + "".join(pieces)
        # Stated as the multiplication it is, with the direction written into
        # the sentence rather than left to a caption elsewhere.
        equation = (
            f"{fit.channel}:  analytic x ({body})  =  actual"
            if pieces
            else f"{fit.channel}:  analytic x {body}  =  actual"
        )
        form_note = (
            f"Ratio form, fitted as {RATIO_DIRECTION}: multiply the analytic "
            "estimate by the factor above to get the predicted measurement."
        )
    else:
        body = format_number(constant) + "".join(pieces)
        equation = f"{fit.channel} correction = {body}  {fit.unit}".rstrip()
        form_note = (
            "Additive form, fitted as actual - analytic: add the correction "
            "above to the analytic estimate."
        )

    note = ""
    if dropped:
        note = (
            ", ".join(dropped)
            + ": not distinguishable from zero, excluded above. The prediction "
            "still uses the full fit; the line above holds them at their "
            "typical anchor value."
        )
    return EquationText(equation=equation, excluded_note=note, form_note=form_note)


def variable_comparison_note(fit: ChannelCalibration, tolerance: float = 1.10) -> str:
    """Say plainly when the simpler single-variable fit predicts worse.

    Simplifying to one variable buys interpretability, not accuracy, and the
    report should not imply otherwise. `tolerance` is how much worse the
    cross-validated error may be before it is called out - 10%, so ordinary
    noise between two comparable fits does not trigger a warning on every run.
    """
    if fit.variable == ENGINEERED_PAIR_KEY:
        return ""
    if not np.isfinite(fit.baseline_loo_rmse) or not np.isfinite(fit.loo_rmse):
        return ""
    if fit.baseline_loo_rmse <= 0:
        return ""
    ratio = fit.loo_rmse / fit.baseline_loo_rmse
    if ratio > tolerance:
        return (
            f"Cross-validated error against {fit.variable_label} alone is "
            f"{format_number(fit.loo_rmse)} {fit.unit}, {ratio:.2f}x the "
            f"{format_number(fit.baseline_loo_rmse)} {fit.unit} of the "
            "two-feature fit. This single variable is the more interpretable "
            "shape, not the more accurate one."
        )
    if ratio < 1.0 / tolerance:
        return (
            f"Cross-validated error against {fit.variable_label} alone is "
            f"{format_number(fit.loo_rmse)} {fit.unit}, better than the "
            f"{format_number(fit.baseline_loo_rmse)} {fit.unit} of the "
            "two-feature fit - the simpler shape also predicts held-out blocks "
            "more closely here."
        )
    return (
        f"Cross-validated error against {fit.variable_label} alone is "
        f"{format_number(fit.loo_rmse)} {fit.unit}, within 10% of the "
        f"two-feature fit's {format_number(fit.baseline_loo_rmse)} {fit.unit} - "
        "no material accuracy cost for the simpler shape."
    )


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def parameter_table(calibration: CalibrationSet) -> pd.DataFrame:
    """Fitted parameters with confidence intervals, one row per term."""
    rows = []
    for channel, fit in calibration.channels.items():
        for estimate in fit.estimates:
            rows.append(
                {
                    "channel": channel,
                    "unit": fit.unit,
                    "term": estimate.label,
                    "estimate": estimate.value,
                    "std error": estimate.se,
                    f"CI lo ({1 - fit.alpha:.0%})": estimate.ci_lo,
                    f"CI hi ({1 - fit.alpha:.0%})": estimate.ci_hi,
                    "spans zero": "yes" if estimate.spans_zero else "no",
                }
            )
    return pd.DataFrame(rows)


def fit_summary_table(calibration: CalibrationSet) -> pd.DataFrame:
    """One row per channel: how well and how firmly it was fitted."""
    rows = []
    for channel, fit in calibration.channels.items():
        rows.append(
            {
                "channel": channel,
                "unit": fit.unit,
                "anchors": fit.n_anchor,
                "parameters": len(fit.terms),
                "dof": fit.dof,
                "RMS residual": fit.residual_rms,
                "weighted RMS residual": fit.weighted_residual_rms,
                "residual / reported noise": fit.scale,
                "condition number": fit.condition_number,
                "SE floor": fit.se_floor,
            }
        )
    return pd.DataFrame(rows)
