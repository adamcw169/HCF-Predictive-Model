"""Steady-state block extraction from a raw draw timeseries.

A draw run sampled at 1 Hz is not thousands of independent experiments. It is a
handful of settled operating points separated by ramps, adjustments and
transients. This module finds those settled stretches and reduces each one to a
single point with an uncertainty - which is what an anchor for the calibration
actually is.

The method is the one already proven against real data in
``steady_state_extraction.py``, generalised so the thresholds, window and block
rules are parameters rather than constants:

1. Mark every sample that fails a QC flag as unusable.
2. For each monitored channel, compute a centred rolling standard deviation and
   require it to sit below a per-channel threshold. A sample is *flat* only
   when every channel in the criteria set is simultaneously below its
   threshold.
3. Bridge short gaps in the flat mask, so one stray sample does not split an
   otherwise continuous block in two.
4. Trim the edges of each surviving run (the rolling window straddles the
   boundary there, so the edges are the least trustworthy part) and drop
   anything left that is too short to be a real operating point.
5. Summarise each block with a median and a standard error per channel.

The median and the MAD-based standard error are used rather than a mean and an
SD because a settled block can still contain the occasional spike, and a
handful of anchors cannot afford to have one of them dragged off by an outlier.

Per-mapping criteria sets (v1.1)
--------------------------------
Requiring *every* monitored channel to be simultaneously flat is stricter than
the physics demands, and it is the main reason only eight blocks came out of a
111-minute run. A furnace temperature blip says nothing about whether the
capillary pressure and the geometry it produced were measured under the same
settled condition.

So steadiness is defined **per correction pair**, not globally: each setpoint
channel gets its own criteria set - itself, the channels physically coupled to
it, and the channels its predictive features are built from - and its own block
boundaries. See `DEFAULT_STABILITY_GROUPS`.

This is deliberately *not* per-channel decoupling. Pressure controls wall
thinning, which sets the ID; detecting a pressure window and an ID window
independently would pair a pressure median with an ID median that were never
measured under the same condition, and the correspondence the whole calibration
rests on would be gone. Each group is therefore *joint over its own members* -
it is the irrelevant channels that are dropped, never the coupled ones.

Stability and segmentation are different questions
--------------------------------------------------
Dropping a channel from a criteria set stops it *vetoing* a sample. It must not
also stop it *separating* one operating point from the next. Take the furnace
group, which excludes capillary pressure: two stretches at different pressures
have the same furnace behaviour and would merge into one long block - but they
have different capillary geometry, and `cap_wall_ratio` is a feature in the
furnace correction. Their merged median would describe a geometry that never
existed.

So a channel outside a criteria set still splits a block when its *level*
moves, even though its noise no longer rejects a sample. Without this the
relaxation trades block count for block length and makes the anchor set worse;
with it, the run is cut into every distinct operating point that each mapping
can actually distinguish. `segment_tolerance_factor` sets how large a level
shift has to be to count.

Segmentation must not depend on how much of the run is loaded (v1.8.2)
------------------------------------------------------------------------
The percent criterion's segmentation used to convert its percent threshold
into the channel's own units by multiplying against `np.ptp` (max - min) of
the *whole loaded frame*: `tolerance = percent/100 * span`. That span is not a
property of the sensor - it is a property of how many distinct operating
points happen to be present in whatever was loaded. A short file covering only
a handful of similar blocks has a small span; the same channel, in a run that
also visits several other process points, has a much larger one. Loading a
time slice of a run therefore does not merely narrow the *time range*
`level_tolerance` acts on - it silently retunes the *entire threshold* the
segmentation of every block in that slice is measured against, because the
span it divides by has changed.

Measured on the reference run against a slice of its last ~38 minutes: the
span of `outer_dP_kPa` drops from 15.3 kPa (the full run, which visits several
pressure levels from 0 kPa upward) to 0.34 kPa (the slice, which stays within
one late-run pressure band) - a 45x difference produced entirely by which
rows happened to be loaded, not by anything about the instrument. The
resulting absolute tolerance shrinks by the same 45x, which is far below the
sensor's own sample-to-sample noise - so segmentation stops distinguishing a
real step from ordinary jitter and starts cutting on every noisy sample
instead, shredding a genuine multi-minute block into fragments too short to
survive the minimum-duration filter. The channel's *reference drift*
(`reference_percent`, already a percentage of the channel's own current
value rather than of the loaded frame's total range) moves far less between
the same two loads - single-digit percent, not 45x - which is what pointed at
`np.ptp` specifically rather than at the reference computation in general.

The fix evaluates a level shift the same way the *steadiness* criterion
already does: as a percentage of the channel's own current value, using
`channel_floor` as the same near-zero backstop `percent_change` uses, rather
than as a percentage of a global range snapshot. "Has this channel moved to a
different operating point" becomes a question with a local, per-sample
answer instead of one borrowed from however much surrounding data was in the
file - which is what makes it invariant to slicing. See
`_split_on_percent_level_changes` and
`tests/test_steady_state.py::test_slice_extraction_matches_full_run_filtered_
to_the_same_window`.

This also corrected a second, independent bug the same code path had: the
percent-mode caller pre-multiplied by `segment_tolerance_factor` before
calling into the (now-removed) conversion helper, which *itself* multiplied
by `segment_tolerance_factor` again - squaring the factor's effect (36x at
the default of 6) instead of applying it once. The absolute-criterion path
was never affected: it always applied the factor exactly once, which is how
the double application was found - by comparing the two paths rather than
assuming either was correct.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping

import numpy as np
import pandas as pd

import schema

# How steadiness is measured. Both are kept: the percent criterion is what the
# app uses, and the absolute one is the shape the original 8-block extraction
# was validated against - retaining it is what lets the change be *measured*
# against that result rather than asserted to be equivalent. See
# `compare_criteria`.
CRITERION_PERCENT = "percent"
CRITERION_ABSOLUTE_SD = "absolute_sd"

# Per-channel rolling-SD thresholds, in each channel's own units. Used only by
# the absolute criterion now, but they still name the six channels this app
# monitors, and their relative sizes are the record of what "settled" meant
# before v1.4.
DEFAULT_CHANNEL_THRESHOLDS: dict[str, float] = {
    "fibre_ID_um": 3.0,
    "feed_speed_mm_min": 0.04,
    "draw_speed_m_min": 0.03,
    "furnace_temp_C": 0.15,
    "core_dP_kPa": 0.08,
    "outer_dP_kPa": 0.12,
}

MONITORED_CHANNELS: tuple[str, ...] = tuple(DEFAULT_CHANNEL_THRESHOLDS)

# --------------------------------------------------------------------------
# The percent-variation criterion (v1.4)
# --------------------------------------------------------------------------
#
#     pct_change(t) = |A(t) - A(t-B)| / max(|A(t)|, floor) * 100
#
# One number for every channel instead of six hand-tuned absolutes, which is
# what makes a single slider possible.
#
# What it measures, and what it does not
# --------------------------------------
# This is a *drift* test: how far the channel has moved over the lag B. It is
# not a spread test - a channel oscillating fast around a fixed level can
# return to where it was and score zero. That is a deliberate simplification,
# not an oversight, and it is why the absolute rolling-SD criterion is kept
# selectable rather than deleted: the two answer different questions, and the
# comparison between them is reported rather than hidden.
#
# Two things the raw formula needs
# --------------------------------
# 1. A denominator floor. `core_dP_kPa` sits near zero for much of a run and
#    reads slightly negative from sensor noise; dividing by A(t) there turns a
#    physically trivial wobble into a percentage of thousands, or flips its
#    sign. The floor is derived per channel from that channel's own range in
#    the loaded data, so it carries across units without retuning.
# 2. A minimum contiguous span. A single sample-to-sample comparison is one
#    noisy difference; requiring the criterion to hold across a real stretch of
#    time is what makes it a statement about the process rather than about one
#    pair of samples.

# The denominator floor, as a fraction of the channel's full observed range.
# Two percent of the range is far below any level the channel meaningfully
# operates at, so it never softens a real percentage - it only stops the
# division exploding as A(t) passes through zero.
ZERO_CROSSING_FLOOR_FRACTION = 0.02

# Absolute backstop for a channel that does not vary at all in the loaded data,
# where a fraction of the range would itself be zero.
ZERO_CROSSING_FLOOR_MINIMUM = 1.0e-9

# Why the slider is a multiplier and not a raw percentage
# -------------------------------------------------------
# A single raw percent threshold cannot work here, and that is a property of
# the instruments rather than of the tuning. Measured on the reference run,
# *inside* stretches the validated extraction already called steady, the 95th
# percentile of |pct_change| over a 30 s lag is:
#
#     fibre_ID_um        9.30 %      draw_speed_m_min    0.21 %
#     feed_speed_mm_min  1.33 %      furnace_temp_C      0.010 %
#     outer_dP_kPa       0.85 %      core_dP_kPa         0.61 %
#
# Three orders of magnitude between an optical diameter gauge reading a 48 um
# feature and a thermocouple on a furnace. Any single threshold high enough to
# let `fibre_ID_um` through during genuine steadiness is ~900x looser than
# `furnace_temp_C` ever needs, so that channel stops vetoing anything; any
# threshold tight enough to constrain the furnace deletes every block that
# contains the fiber ID - which is exactly what the pressure mappings are made
# of, and exactly what a sweep of raw thresholds produces (zero blocks for
# `core_dP_kPa` and `Pocap_kPa` at every sensitivity worth having).
#
# So the slider scales each channel against *its own* quiet behaviour, measured
# from the loaded run rather than hand-entered. The operator still turns one
# knob and never types a per-channel number; what the knob means is "how many
# multiples of this channel's normal drift count as still settled". The derived
# reference per channel is shown in the UI, so the scaling is visible rather
# than hidden.

# Quantile of a channel's own |pct_change| distribution used as its reference.
# High enough to be meaningful for a channel that holds an exactly constant
# setpoint for most of a run (where the median is ~0), low enough to stay well
# inside quiet behaviour rather than being dragged up by the ramps.
REFERENCE_QUANTILE = 0.75

# Floor on the derived reference, as a percentage. Stops a perfectly constant
# channel from producing a zero reference and therefore an unsatisfiable
# threshold.
REFERENCE_PERCENT_MINIMUM = 0.005

# Slider default, in multiples of each channel's own reference drift. Picked by
# sweeping the reference run against the validated absolute-threshold result:
# at 2.5 the two kinematic mappings land on exactly the block counts that
# result produced (9 and 10) while the two pressure mappings gain roughly
# double the anchors, for comparable total settled time.
# `tests/test_steady_state.py::test_percent_default_reproduces_the_validated_
# kinematic_counts` pins that, so the default cannot drift unnoticed.
DEFAULT_SENSITIVITY = 2.5

# The lag B. Long enough that a slow ramp shows up as movement, short enough
# that it does not smear the boundary between two adjacent operating points.
DEFAULT_LAG_WINDOW_S = 20.0

# Boolean columns where True means "this sample is not trustworthy". A sample
# is excluded if any of them is set. Listed explicitly rather than discovered by
# prefix: `cap_id_invalid` also looks like a QC flag but must not exclude the
# whole sample, because a missing capillary measurement says nothing about
# whether the furnace and the capstan were steady.
DEFAULT_QC_EXCLUDE_COLUMNS: tuple[str, ...] = (
    "fibre_id_invalid",
    "qc_zero_values",
    "qc_ramp_up",
    "qc_outlier",
)

# QC flags that only invalidate a sample for *some* criteria sets. A flag
# listed here applies only when one of the named channels is being judged; a
# flag absent from this map applies to every extraction.
#
# `fibre_id_invalid` marks a bad fiber-ID measurement. That disqualifies the
# sample from a block whose whole point is to pin down the ID, and says nothing
# at all about whether the capstan held its speed - the same reasoning that
# motivates per-mapping criteria sets in the first place.
DEFAULT_QC_CHANNEL_SCOPE: dict[str, tuple[str, ...]] = {
    "fibre_id_invalid": ("fibre_ID_um",),
}

# Columns whose *values* are masked by a flag, without the sample itself being
# thrown away. {value column: flag column, True meaning "drop this value"}.
DEFAULT_VALUE_MASKS: dict[str, str] = {
    "cap_OD_um": "cap_id_invalid",
    "cap_ID_um": "cap_id_invalid",
}

# Fraction-of-samples column reported per block, if present.
QC_PASS_COLUMN = "qc_pass"

# Counts how many capillary measurements in a block were really measured rather
# than interpolated, if present.
CAP_REAL_COLUMN = "cap_OD_was_real"

# Name used for the all-channels-at-once extraction that the per-mapping counts
# are compared against.
GLOBAL_CRITERIA_LABEL = "all channels (v1.0 behaviour)"


@dataclass(frozen=True)
class StabilityGroup:
    """Which raw channels must be *jointly* steady to trust one setpoint channel.

    `members` always includes the setpoint's own measured channel plus every
    channel it is physically coupled to and every channel its predictive
    features are built from. Everything else in the file is irrelevant to this
    correction and is not allowed to veto a block.
    """

    channel: str
    members: tuple[str, ...]
    rationale: str


# The four correction pairs. Each lists only what has to hold still for *that*
# mapping to be physically meaningful.
#
# Common to all four: feed and draw speed. They set the draw-down ratio, which
# is a feature in every channel's correction, and a moving capstan means the
# geometry in the hot zone is still changing.
DEFAULT_STABILITY_GROUPS: dict[str, StabilityGroup] = {
    "furnace_temp_C": StabilityGroup(
        channel="furnace_temp_C",
        members=("furnace_temp_C", "feed_speed_mm_min", "draw_speed_m_min"),
        rationale=(
            "The furnace setpoint itself plus the kinematics that set how long "
            "glass spends in the hot zone. Capillary pressure does not enter: a "
            "pressure step changes the microstructure, not the temperature the "
            "correction is about."
        ),
    ),
    "draw_speed_m_min": StabilityGroup(
        channel="draw_speed_m_min",
        members=("draw_speed_m_min", "feed_speed_mm_min"),
        rationale=(
            "Capstan and feed only. This is the kinematic pair, and it is the "
            "narrowest group because nothing else in the file constrains "
            "whether the draw ratio was held."
        ),
    ),
    "core_dP_kPa": StabilityGroup(
        channel="core_dP_kPa",
        members=(
            "core_dP_kPa",
            "fibre_ID_um",
            "feed_speed_mm_min",
            "draw_speed_m_min",
        ),
        rationale=(
            "Core pressure and the fiber ID together. The ID is what the core "
            "pressure produces, so pairing a settled pressure with an ID still "
            "responding to an earlier step would break the correspondence the "
            "calibration depends on. Furnace temperature is excluded: a "
            "transient there does not invalidate an otherwise-good "
            "pressure/geometry window."
        ),
    ),
    "Pocap_kPa": StabilityGroup(
        channel="Pocap_kPa",
        members=(
            "core_dP_kPa",
            "outer_dP_kPa",
            "fibre_ID_um",
            "feed_speed_mm_min",
            "draw_speed_m_min",
        ),
        rationale=(
            "Pocap is the sum of the core and outer differentials, so both must "
            "be steady, and the ID comes with them for the same reason as core "
            "pressure: it is the geometry those pressures produced."
        ),
    ),
}


# --- nested geometry (v1.9) ----------------------------------------------
#
# Same reasoning as the four above, extended to three capillary layers. The
# principle that decides membership has not changed: a channel's criteria set
# is itself, the channels physically coupled to it, and the geometry it
# directly governs - and nothing else, because an irrelevant channel vetoing a
# sample is what made the anchor set worse before v1.1.
#
# What that means per layer: a capillary pressure governs *its own* layer's
# bore, so `Pmcap_kPa` takes `cap_ID_middle_um` and not the outer or inner
# layer's dimension. Pairing a middle-layer pressure with an outer-layer
# diameter would assert a correspondence the physics does not have, which is
# the same error the joint-over-own-members rule exists to prevent.
#
# The kinematic pair (feed and draw speed) is common to every group for the
# reason it always was: it sets the draw-down ratio, a feature in every
# channel's correction.

NESTED_MONITORED_CHANNELS: tuple[str, ...] = (
    "fibre_ID_um",
    "feed_speed_mm_min",
    "draw_speed_m_min",
    "furnace_temp_C",
    "core_dP_kPa",
    "outer_dP_kPa",
    "cap_ID_outer_um",
    "cap_ID_middle_um",
    "cap_ID_inner_um",
)

# Rolling-SD thresholds for the nested-only channels, in their own units. The
# three capillary bores are the same kind of optical measurement as the fiber
# ID, on a feature an order of magnitude smaller, so they inherit a
# proportionally tighter absolute threshold. Only the absolute criterion uses
# these; the percent criterion derives its own per-channel reference.
NESTED_CHANNEL_THRESHOLDS: dict[str, float] = {
    **DEFAULT_CHANNEL_THRESHOLDS,
    "cap_ID_outer_um": 0.3,
    "cap_ID_middle_um": 0.3,
    "cap_ID_inner_um": 0.3,
}

_KINEMATIC = ("feed_speed_mm_min", "draw_speed_m_min")

NESTED_STABILITY_GROUPS: dict[str, StabilityGroup] = {
    "furnace_temp_C": StabilityGroup(
        channel="furnace_temp_C",
        members=("furnace_temp_C", *_KINEMATIC),
        rationale=(
            "Unchanged from the non-nested geometry: the furnace setpoint plus "
            "the kinematics that set how long glass spends in the hot zone. No "
            "capillary pressure enters, at any layer - a pressure step changes "
            "the microstructure, not the temperature the correction is about."
        ),
    ),
    "draw_speed_m_min": StabilityGroup(
        channel="draw_speed_m_min",
        members=("draw_speed_m_min", "feed_speed_mm_min"),
        rationale=(
            "Capstan and feed only, exactly as in the non-nested geometry. "
            "Nothing about a third capillary layer constrains whether the draw "
            "ratio was held."
        ),
    ),
    "core_dP_kPa": StabilityGroup(
        channel="core_dP_kPa",
        members=("core_dP_kPa", "fibre_ID_um", *_KINEMATIC),
        rationale=(
            "Core pressure and the fiber ID together - the core pressure sets "
            "the core bore regardless of how many capillary layers surround it, "
            "so this group is the same as the non-nested one."
        ),
    ),
    "Pocap_kPa": StabilityGroup(
        channel="Pocap_kPa",
        members=(
            "core_dP_kPa",
            "outer_dP_kPa",
            "cap_ID_outer_um",
            *_KINEMATIC,
        ),
        rationale=(
            "The outer capillary pressure and the bore it produces. Pocap is "
            "still the sum of the core and outer differentials, so both must be "
            "steady; the geometry paired with it is the *outer* layer's bore, "
            "not the fiber ID, because on a nested preform the outer capillary "
            "pressure governs its own layer rather than the core."
        ),
    ),
    "Pmcap_kPa": StabilityGroup(
        channel="Pmcap_kPa",
        members=("Pmcap_kPa", "cap_ID_middle_um", *_KINEMATIC),
        rationale=(
            "The middle capillary pressure and the middle bore it produces. "
            "Deliberately not the outer or inner layer's dimension: pairing a "
            "settled middle-layer pressure with another layer's geometry would "
            "assert a correspondence the physics does not have."
        ),
    ),
    "Picap_kPa": StabilityGroup(
        channel="Picap_kPa",
        members=("Picap_kPa", "cap_ID_inner_um", *_KINEMATIC),
        rationale=(
            "The inner capillary pressure and the inner bore it produces, by "
            "the same reasoning as the middle layer."
        ),
    ),
}


@dataclass(frozen=True)
class ExtractionSettings:
    """Everything that decides what counts as a steady-state block.

    Durations are in seconds and converted to samples using the detected
    sampling period, so the same settings behave the same way on data logged at
    a different rate.
    """

    window_s: float = 61.0
    max_gap_bridge_s: float = 5.0
    min_block_duration_s: float = 60.0
    edge_trim_s: float = 5.0
    # Which of the two steadiness criteria to apply. The percent one is the
    # v1.4 default; the absolute one is what v1.0-v1.3 used and is kept so the
    # change stays measurable against it.
    criterion: str = CRITERION_PERCENT
    # The single slider - multiples of each channel's own reference drift -
    # and the lag it is evaluated over.
    sensitivity: float = DEFAULT_SENSITIVITY
    lag_window_s: float = DEFAULT_LAG_WINDOW_S
    reference_quantile: float = REFERENCE_QUANTILE
    # Which channels are watched at all. Under the percent criterion this is the
    # whole per-channel configuration - one threshold covers all of them.
    monitored_channels: tuple[str, ...] = MONITORED_CHANNELS
    # Only used by the absolute criterion.
    channel_thresholds: Mapping[str, float] = field(
        default_factory=lambda: dict(DEFAULT_CHANNEL_THRESHOLDS)
    )
    qc_exclude_columns: tuple[str, ...] = DEFAULT_QC_EXCLUDE_COLUMNS
    qc_channel_scope: Mapping[str, tuple[str, ...]] = field(
        default_factory=lambda: dict(DEFAULT_QC_CHANNEL_SCOPE)
    )
    value_masks: Mapping[str, str] = field(
        default_factory=lambda: dict(DEFAULT_VALUE_MASKS)
    )
    stability_groups: Mapping[str, StabilityGroup] = field(
        default_factory=lambda: dict(DEFAULT_STABILITY_GROUPS)
    )
    # How far a channel outside the criteria set has to move before it is
    # treated as a new operating point rather than noise, in multiples of
    # whichever steadiness threshold is in force. Six is comfortably outside
    # anything a settled channel does and well inside a real setpoint step.
    segment_tolerance_factor: float = 6.0
    # A block whose furnace temperature sits this far from the run's modal
    # temperature is flagged: it is probably a different process point rather
    # than a repeat of the same one.
    nominal_temp_tolerance_C: float = 20.0

    def with_thresholds(self, thresholds: Mapping[str, float]) -> ExtractionSettings:
        return replace(self, channel_thresholds=dict(thresholds))


@dataclass(frozen=True)
class ExtractionProfile:
    """One geometry's criteria sets: what is watched, and jointly with what.

    Separate from `ExtractionSettings` because the two are different kinds of
    thing. A profile is a *geometry* fact - a nested preform has three
    capillary bores whether or not anyone has tuned anything. The settings
    around it (sensitivity, window, minimum duration) are *process* judgements
    the operator makes and keeps across a preform change.
    """

    monitored_channels: tuple[str, ...]
    channel_thresholds: dict[str, float]
    stability_groups: dict[str, StabilityGroup]


# Keyed by `PreformDefinition.id`. A registry rather than a branch: adding a
# geometry means adding an entry here beside its `PreformDefinition`, which is
# the same "append, don't edit" shape the preform registry itself has.
# NANF: outer and inner bores only. Same membership reasoning as DNANF, minus
# the layer it does not have - each capillary pressure is joint with its own
# layer's bore and the kinematic pair, never another layer's dimension.
NANF_MONITORED_CHANNELS: tuple[str, ...] = (
    "fibre_ID_um",
    "feed_speed_mm_min",
    "draw_speed_m_min",
    "furnace_temp_C",
    "core_dP_kPa",
    "outer_dP_kPa",
    "cap_ID_outer_um",
    "cap_ID_inner_um",
)

NANF_CHANNEL_THRESHOLDS: dict[str, float] = {
    **DEFAULT_CHANNEL_THRESHOLDS,
    "cap_ID_outer_um": 0.3,
    "cap_ID_inner_um": 0.3,
}

NANF_STABILITY_GROUPS: dict[str, StabilityGroup] = {
    "furnace_temp_C": NESTED_STABILITY_GROUPS["furnace_temp_C"],
    "draw_speed_m_min": NESTED_STABILITY_GROUPS["draw_speed_m_min"],
    "core_dP_kPa": NESTED_STABILITY_GROUPS["core_dP_kPa"],
    "Pocap_kPa": NESTED_STABILITY_GROUPS["Pocap_kPa"],
    "Picap_kPa": StabilityGroup(
        channel="Picap_kPa",
        members=("Picap_kPa", "cap_ID_inner_um", *_KINEMATIC),
        rationale=(
            "The inner capillary pressure and the inner bore it produces. On "
            "this geometry the inner layer sits directly inside the outer one, "
            "but the group is still joint over its own members only - the "
            "outer layer's bore is not what this pressure governs."
        ),
    ),
}


EXTRACTION_PROFILES: dict[str, ExtractionProfile] = {
    "tubular": ExtractionProfile(
        monitored_channels=MONITORED_CHANNELS,
        channel_thresholds=dict(DEFAULT_CHANNEL_THRESHOLDS),
        stability_groups=dict(DEFAULT_STABILITY_GROUPS),
    ),
    "nanf": ExtractionProfile(
        monitored_channels=NANF_MONITORED_CHANNELS,
        channel_thresholds=dict(NANF_CHANNEL_THRESHOLDS),
        stability_groups=dict(NANF_STABILITY_GROUPS),
    ),
    "dnanf": ExtractionProfile(
        monitored_channels=NESTED_MONITORED_CHANNELS,
        channel_thresholds=dict(NESTED_CHANNEL_THRESHOLDS),
        stability_groups=dict(NESTED_STABILITY_GROUPS),
    ),
}


def settings_for_preform(
    preform_id: str, base: ExtractionSettings | None = None
) -> ExtractionSettings:
    """Extraction settings carrying that preform's criteria sets.

    The one place a geometry's stability groups, monitored channels and
    absolute thresholds are selected, so no caller has to know which preform
    maps to which set. `base` lets an operator's tuning survive a preform
    change, for the reason given on `ExtractionProfile`.

    An unknown id falls back to the shipped defaults rather than raising:
    extraction is reachable before a preform has been chosen, and those
    defaults are exactly what every pre-v1.9 caller already got.
    """
    settings = base or ExtractionSettings()
    profile = EXTRACTION_PROFILES.get(preform_id)
    if profile is None:
        # The ids were renamed in the tubular/NANF/DNANF release, and a stored
        # calibration or an older caller may still be passing the old one.
        # Silently returning the shipped defaults would give a nested run the
        # tubular criteria sets - four groups where it needs six - which does
        # not raise, it just quietly extracts the wrong thing.
        try:
            import preform as _preform_registry

            profile = EXTRACTION_PROFILES.get(
                _preform_registry.canonical_preform_id(preform_id)
            )
        except Exception:  # noqa: BLE001 - never fail a lookup on an import
            profile = None
    if profile is None:
        return settings
    return replace(
        settings,
        monitored_channels=profile.monitored_channels,
        channel_thresholds=dict(profile.channel_thresholds),
        stability_groups=dict(profile.stability_groups),
    )


@dataclass
class ExtractionResult:
    """Blocks, the annotated timeseries they came from, and how it went."""

    blocks: pd.DataFrame
    frame: pd.DataFrame
    # (start_index, end_index) into `frame`, inclusive, one per kept block.
    spans: list[tuple[int, int]]
    settings: ExtractionSettings
    sample_period_s: float
    channels_used: tuple[str, ...]
    channels_missing: tuple[str, ...]
    n_candidate_blocks: int
    notes: list[str]
    # Which setpoint channel this extraction is for, and the label shown in the
    # UI. `target_channel` is None for the all-channels-at-once extraction.
    target_channel: str | None = None
    criteria_label: str = GLOBAL_CRITERIA_LABEL
    # Per-sample: which criteria channel was furthest over its threshold, and by
    # what factor. Empty string / NaN where the sample was steady.
    limiting_channel: pd.Series | None = None
    limiting_ratio: pd.Series | None = None
    # Contiguous runs of the steadiness verdict, accepted and rejected alike.
    windows: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def n_blocks(self) -> int:
        return len(self.blocks)

    @property
    def total_steady_s(self) -> float:
        if self.blocks.empty or "duration_s" not in self.blocks.columns:
            return 0.0
        return float(self.blocks["duration_s"].sum()) * self.sample_period_s

    def describe(self) -> str:
        return (
            f"{self.n_blocks} steady-state block(s) from "
            f"{len(self.frame)} samples at {self.sample_period_s:g} s "
            f"({self.n_candidate_blocks} candidate run(s) before the "
            "duration filter)"
        )

    def limiting_factor_counts(self) -> pd.DataFrame:
        """How often each criteria channel was the binding constraint.

        Answers "why is this channel's block count what it is" in one table:
        the channel at the top is the one to loosen, or to question.
        """
        if self.limiting_channel is None:
            return pd.DataFrame()
        blocked = self.limiting_channel[self.limiting_channel != ""]
        if blocked.empty:
            return pd.DataFrame(
                columns=["limiting channel", "samples", "share of unsteady"]
            )
        counts = blocked.value_counts()
        total = int(counts.sum())
        return pd.DataFrame(
            {
                "limiting channel": counts.index.astype(str),
                "samples": counts.to_numpy(),
                "share of unsteady": counts.to_numpy() / total,
            }
        ).reset_index(drop=True)

    def explain_at(self, index: int) -> str:
        """Why one sample was or was not counted as steady."""
        if not 0 <= index < len(self.frame):
            return "Outside the loaded run."
        when = self.frame[schema.TIME_COLUMN].iloc[index]
        block = int(self.frame["steady_block_id"].iloc[index])
        if block > 0:
            return f"{when:%H:%M:%S} - inside block {block}."
        if self.limiting_channel is None:
            return f"{when:%H:%M:%S} - not steady."
        who = str(self.limiting_channel.iloc[index])
        if not who:
            return (
                f"{when:%H:%M:%S} - every criteria channel was within threshold "
                "here, but the run around it was too short to keep as a block "
                "(or was trimmed as an edge)."
            )
        ratio = float(self.limiting_ratio.iloc[index])
        if self.settings.criterion == CRITERION_PERCENT:
            threshold = self.settings.percent_threshold
            return (
                f"{when:%H:%M:%S} - blocked by {who}: it moved "
                f"{ratio * threshold:.3g}% over the last "
                f"{self.settings.lag_window_s:g} s, {ratio:.2g}x the "
                f"{threshold:g}% sensitivity threshold."
            )
        threshold = self.settings.channel_thresholds.get(who, float("nan"))
        return (
            f"{when:%H:%M:%S} - blocked by {who}: its rolling SD is "
            f"{ratio:.2g}x its threshold of {threshold:g} "
            f"{schema.unit_of(who) or ''}".strip()
            + "."
        )


@dataclass
class PerChannelExtraction:
    """One extraction per setpoint channel, plus the global one to compare to.

    Block boundaries and block counts differ per channel by design - that is the
    whole point of per-mapping criteria - so there is no single block table
    here, and downstream code must carry the channel along with the blocks.
    """

    results: dict[str, ExtractionResult]
    global_result: ExtractionResult
    groups: dict[str, StabilityGroup]
    notes: list[str] = field(default_factory=list)

    @property
    def channels(self) -> tuple[str, ...]:
        return tuple(self.results)

    def comparison_table(self) -> pd.DataFrame:
        """Before/after block counts, so the gain is measured and not assumed."""
        before = self.global_result.n_blocks
        before_s = self.global_result.total_steady_s
        rows = []
        for channel, result in self.results.items():
            group = self.groups[channel]
            rows.append(
                {
                    "channel": channel,
                    "criteria channels": ", ".join(group.members),
                    "blocks (all-channel)": before,
                    "blocks (per-mapping)": result.n_blocks,
                    "change": result.n_blocks - before,
                    "steady seconds (all-channel)": before_s,
                    "steady seconds (per-mapping)": result.total_steady_s,
                    "limiting channel": _dominant_limiter(result),
                }
            )
        return pd.DataFrame(rows)

    def describe(self) -> str:
        counts = ", ".join(
            f"{channel} {result.n_blocks}"
            for channel, result in self.results.items()
        )
        return (
            f"Per-mapping extraction: {counts} "
            f"(all-channel baseline: {self.global_result.n_blocks})"
        )


def _dominant_limiter(result: ExtractionResult) -> str:
    counts = result.limiting_factor_counts()
    if counts.empty:
        return "-"
    return str(counts.iloc[0]["limiting channel"])


class ExtractionError(ValueError):
    """Raised when the input file cannot be extracted from at all."""


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def load_raw_timeseries(
    path: str, preform_schema: "schema.PreformSchema | None" = None
) -> tuple[pd.DataFrame, list[str]]:
    """Read a raw 1 Hz timeseries CSV and put it in a known-good shape.

    Sorting by time and reconstructing each layer's absolute pressure happen
    here rather than in `extract_blocks` so the same prepared frame can be
    reused across repeated extractions while the operator tunes thresholds.

    `preform_schema` selects the pressure chain: the tower logs `core_dP_kPa`
    as a true absolute and every capillary layer as a differential against the
    layer outside it, so the absolutes the model is fitted on are reconstructed
    by walking that chain. Defaulting to the tubular schema makes this exactly
    the single `Pocap = core_dP + outer_dP` sum it has always been.
    """
    notes: list[str] = []
    df = pd.read_csv(path)
    if schema.TIME_COLUMN not in df.columns:
        raise ExtractionError(
            f"The raw timeseries file needs a '{schema.TIME_COLUMN}' column: "
            "steady-state blocks are defined by a time window, and that window "
            "is what the analytic estimates are later matched against."
        )

    df[schema.TIME_COLUMN] = pd.to_datetime(df[schema.TIME_COLUMN], errors="coerce")
    bad_time = int(df[schema.TIME_COLUMN].isna().sum())
    if bad_time:
        df = df[df[schema.TIME_COLUMN].notna()]
        notes.append(f"Dropped {bad_time} row(s) with an unparseable timestamp.")
    if df.empty:
        raise ExtractionError("No rows with a usable timestamp were found.")

    df = df.sort_values(schema.TIME_COLUMN).reset_index(drop=True)

    for column in schema.ALL_BLOCK_VALUE_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    active = preform_schema or schema.TUBULAR_SCHEMA
    df, pressure_notes = active.derive_pressures(df)
    notes.extend(pressure_notes)
    return df, notes


def detect_sample_period_s(times: pd.Series) -> float:
    """Median spacing between samples, in seconds."""
    deltas = times.diff().dt.total_seconds().dropna()
    deltas = deltas[deltas > 0]
    if deltas.empty:
        return 1.0
    return float(deltas.median())


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------


def _to_samples(seconds: float, period_s: float, minimum: int = 1) -> int:
    return max(minimum, int(round(float(seconds) / period_s)))


def _bool_column(df: pd.DataFrame, name: str) -> pd.Series:
    """Read a flag column as booleans, tolerating text and blanks."""
    raw = df[name]
    if raw.dtype == bool:
        return raw
    mapped = (
        raw.astype("string")
        .str.strip()
        .str.lower()
        .map({"true": True, "1": True, "yes": True, "false": False, "0": False, "no": False})
    )
    return mapped.fillna(False).astype(bool)


def _bridge_short_gaps(flag: np.ndarray, max_gap: int) -> np.ndarray:
    """Fill runs of False no longer than `max_gap` that sit between two Trues.

    A single sample of jitter in the middle of a settled stretch should not
    turn one long block into two short ones - short blocks carry a worse
    standard error and some fall below the minimum duration entirely.
    """
    out = flag.copy()
    if max_gap <= 0:
        return out
    idx = np.where(~out)[0]
    if idx.size == 0:
        return out
    gap_start = idx[np.r_[True, np.diff(idx) > 1]]
    gap_end = idx[np.r_[np.diff(idx) > 1, True]]
    for start, end in zip(gap_start, gap_end):
        inside = start > 0 and end < len(out) - 1
        if inside and (end - start + 1) <= max_gap and out[start - 1] and out[end + 1]:
            out[start : end + 1] = True
    return out


def _split_on_level_changes(
    run: tuple[int, int],
    frame: pd.DataFrame,
    channels: Mapping[str, float],
    tolerance_factor: float,
) -> list[tuple[int, int]]:
    """Cut one candidate run wherever a non-criteria channel changes level.

    The absolute-criterion path only: `channels` here carries each column's
    fixed, hand-tuned `channel_thresholds` value, in the channel's own units -
    a constant that does not depend on the loaded frame at all, so this
    function has no slice-invariance problem to begin with. The percent
    criterion uses `_split_on_percent_level_changes` instead; see that
    function's docstring and the module docstring's v1.8.2 note for why the
    two are not the same function with a different threshold source.

    Only channels *outside* the criteria set are passed in. A criteria channel
    cannot shift level without its rolling SD breaching the threshold, so the
    run would already have ended - segmenting on it would be redundant, and in
    the all-channel case it would make this function a no-op by construction,
    which is what keeps the v1.0 reference extraction reproducible.

    Level tracking is greedy: hold the level from the start of the current
    segment and cut at the first sample that departs from it by more than the
    tolerance. Setpoint steps in this data are square, so a greedy tracker cuts
    exactly at the step and does not accumulate drift.
    """
    start, end = run
    if not channels or tolerance_factor <= 0:
        return [run]

    cuts: set[int] = set()
    for column, threshold in channels.items():
        if column not in frame.columns:
            continue
        values = pd.to_numeric(
            frame[column].iloc[start : end + 1], errors="coerce"
        ).to_numpy(dtype=float)
        if values.size == 0:
            continue
        tolerance = abs(float(threshold)) * tolerance_factor
        if not np.isfinite(tolerance) or tolerance <= 0:
            continue
        level = np.nan
        for offset, value in enumerate(values):
            if not np.isfinite(value):
                continue
            if not np.isfinite(level):
                level = value
                continue
            if abs(value - level) > tolerance:
                cuts.add(start + offset)
                level = value

    if not cuts:
        return [run]
    segments: list[tuple[int, int]] = []
    edges = [start, *sorted(cuts), end + 1]
    for left, right in zip(edges[:-1], edges[1:]):
        if right - 1 >= left:
            segments.append((left, right - 1))
    return segments


def _split_on_percent_level_changes(
    run: tuple[int, int],
    frame: pd.DataFrame,
    channels_percent: Mapping[str, float],
    floors: Mapping[str, float],
) -> list[tuple[int, int]]:
    """Cut one candidate run wherever a non-criteria channel moves by more
    than a percentage of *its own current value* - the percent-criterion
    counterpart to `_split_on_level_changes`.

    Why this is a different function rather than a different threshold
    ----------------------------------------------------------------------
    Before v1.8.2 the percent criterion reused `_split_on_level_changes` by
    first converting its percent threshold into an absolute one via
    `percent/100 * np.ptp(loaded frame)`. That conversion is where the bug
    was: `np.ptp` is the total range of whatever happens to be loaded, which
    shrinks whenever a shorter time slice - or one covering fewer distinct
    operating points - is loaded, for reasons that have nothing to do with
    the sensor's own noise. A tolerance built from it collapses right along
    with the file, until it sits below ordinary sample-to-sample jitter and
    the segmentation starts cutting on noise instead of on genuine steps. See
    the module docstring for the numbers that pinned this down.

    This function instead measures a level shift exactly the way the
    steadiness criterion already measures drift: `percent_change`'s own
    formula, `|value - level| / max(|level|, floor) * 100`, evaluated against
    the tracked level rather than against a fixed lag. `floors` is expected
    to be `channel_floor` per column - the same near-zero backstop
    `percent_change` uses - so a channel that legitimately operates near zero
    is judged the same way here as it is everywhere else in this module.
    Because the comparison is against the channel's own *current* value
    rather than a range computed once from the whole loaded frame, it does
    not move when the amount of surrounding data does.

    `channels_percent[column]` is expected to already be the full threshold
    (`sensitivity * segment_tolerance_factor * reference_percent`) - applied
    once, here, and nowhere else. `_split_on_level_changes` applies
    `tolerance_factor` itself because its caller's `channels` values are raw
    per-channel thresholds with nothing else folded in; this function's
    caller folds the factor into `channels_percent` up front instead, so
    unlike that function this one must not multiply by it again.
    """
    start, end = run
    if not channels_percent:
        return [run]

    cuts: set[int] = set()
    for column, percent in channels_percent.items():
        if column not in frame.columns:
            continue
        if not np.isfinite(percent) or percent <= 0:
            continue
        values = pd.to_numeric(
            frame[column].iloc[start : end + 1], errors="coerce"
        ).to_numpy(dtype=float)
        if values.size == 0:
            continue
        floor = max(
            float(floors.get(column, ZERO_CROSSING_FLOOR_MINIMUM)),
            ZERO_CROSSING_FLOOR_MINIMUM,
        )
        level = np.nan
        for offset, value in enumerate(values):
            if not np.isfinite(value):
                continue
            if not np.isfinite(level):
                level = value
                continue
            denominator = max(abs(level), floor)
            change_percent = abs(value - level) / denominator * 100.0
            if change_percent > percent:
                cuts.add(start + offset)
                level = value

    if not cuts:
        return [run]
    segments: list[tuple[int, int]] = []
    edges = [start, *sorted(cuts), end + 1]
    for left, right in zip(edges[:-1], edges[1:]):
        if right - 1 >= left:
            segments.append((left, right - 1))
    return segments


def channel_floor(values: pd.Series | np.ndarray) -> float:
    """Denominator floor for one channel, from its own range in this run.

    Derived rather than fixed so it carries from kPa to degrees C without
    retuning, and so a channel that legitimately operates near zero is judged
    against the scale it actually spans rather than against its own noise.
    """
    array = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return ZERO_CROSSING_FLOOR_MINIMUM
    span = float(np.ptp(array))
    if span <= 0:
        # A channel that never moved: fall back to its own magnitude, so the
        # floor is still on the channel's scale rather than an absolute epsilon.
        span = float(np.max(np.abs(array)))
    return max(ZERO_CROSSING_FLOOR_FRACTION * span, ZERO_CROSSING_FLOOR_MINIMUM)


def reference_percent(
    values: pd.Series, lag_samples: int, quantile: float = REFERENCE_QUANTILE
) -> float:
    """One channel's normal drift, as a percentage, from its own data.

    This is what the sensitivity slider multiplies. Taken from the loaded run
    so it needs no per-channel entry and adapts to a different instrument or a
    different sampling rate on its own.
    """
    numeric = pd.to_numeric(values, errors="coerce")
    metric = percent_change(numeric, lag_samples, channel_floor(numeric))
    finite = metric.to_numpy(float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return REFERENCE_PERCENT_MINIMUM
    return max(float(np.quantile(finite, quantile)), REFERENCE_PERCENT_MINIMUM)


def percent_change(
    values: pd.Series, lag_samples: int, floor: float
) -> pd.Series:
    """|A(t) - A(t-B)| / max(|A(t)|, floor) * 100, per sample.

    NaN for the first `lag_samples`, where there is no A(t-B) to compare
    against. Callers treat that as "not steady": the run has not been observed
    for long enough to say it held.
    """
    numeric = pd.to_numeric(values, errors="coerce")
    lagged = numeric.shift(max(1, int(lag_samples)))
    denominator = numeric.abs().clip(lower=max(float(floor), ZERO_CROSSING_FLOOR_MINIMUM))
    return (numeric - lagged).abs() / denominator * 100.0


def _drop_short_runs(flag: np.ndarray, min_length: int) -> np.ndarray:
    """Erase runs of True shorter than `min_length`.

    This is the minimum-contiguous-span rule: a lone passing sample in the
    middle of a ramp is one favourable difference, not evidence the process
    settled. Applied after gap bridging, never before - eroding first would
    delete the two halves of a genuinely steady stretch that a single bad
    sample had split, which bridging exists to rejoin.
    """
    out = flag.copy()
    if min_length <= 1:
        return out
    for start, end in _runs_of_true(out):
        if end - start + 1 < min_length:
            out[start : end + 1] = False
    return out


def _runs_of_true(flag: np.ndarray) -> list[tuple[int, int]]:
    if flag.size == 0:
        return []
    change = np.diff(flag.astype(np.int8))
    starts = np.where(change == 1)[0] + 1
    ends = np.where(change == -1)[0]
    if flag[0]:
        starts = np.r_[0, starts]
    if flag[-1]:
        ends = np.r_[ends, len(flag) - 1]
    return list(zip(starts.tolist(), ends.tolist()))


def robust_center_spread(values) -> tuple[float, float, int]:
    """Median, standard error of that median, and the count behind it.

    The spread is a MAD scaled to a Gaussian-equivalent SD, divided by sqrt(n).
    A block whose channel never moved gives an SE of exactly zero, which is
    honest about the samples but not about the instrument - `calibration.py`
    applies a floor before turning these into regression weights.
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan"), 0
    median = float(np.median(x))
    mad = 1.4826 * float(np.median(np.abs(x - median)))
    return median, mad / np.sqrt(x.size), int(x.size)


def extract_blocks(
    df: pd.DataFrame,
    settings: ExtractionSettings | None = None,
    criteria: tuple[str, ...] | None = None,
    target_channel: str | None = None,
    criteria_label: str | None = None,
) -> ExtractionResult:
    """Find steady-state blocks in a prepared raw timeseries.

    `criteria` names the channels that must be *jointly* steady. Left as None it
    is every channel with a threshold, which is the v1.0 behaviour and what the
    reference extraction was validated against - the reproduction test depends
    on that default not moving. Pass a subset to judge steadiness for one
    correction pair; see `extract_per_channel`.
    """
    settings = settings or ExtractionSettings()
    notes: list[str] = []

    if schema.TIME_COLUMN not in df.columns:
        raise ExtractionError(f"'{schema.TIME_COLUMN}' column is missing.")

    frame = df.reset_index(drop=True).copy()
    times = frame[schema.TIME_COLUMN]
    period = detect_sample_period_s(times)
    if period <= 0:
        period = 1.0

    window = _to_samples(settings.window_s, period, minimum=3)
    max_gap = _to_samples(settings.max_gap_bridge_s, period, minimum=0)
    min_block = _to_samples(settings.min_block_duration_s, period, minimum=1)
    edge_trim = _to_samples(settings.edge_trim_s, period, minimum=0)
    if abs(period - 1.0) > 1e-6:
        notes.append(
            f"Sampling period detected as {period:g} s; the window and block "
            f"durations were converted to {window}, {max_gap}, {min_block} and "
            f"{edge_trim} samples respectively."
        )

    percent_mode = settings.criterion == CRITERION_PERCENT
    available = (
        tuple(settings.monitored_channels)
        if percent_mode
        else tuple(settings.channel_thresholds)
    )
    requested = (
        available if criteria is None else tuple(dict.fromkeys(criteria))
    )
    lag = _to_samples(settings.lag_window_s, period, minimum=1)
    min_span = _to_samples(settings.min_block_duration_s, period, minimum=1)
    label = criteria_label or (
        GLOBAL_CRITERIA_LABEL if criteria is None else ", ".join(requested)
    )

    # --- QC mask -------------------------------------------------------
    # A scoped flag only applies when one of the channels it speaks about is
    # actually part of this criteria set.
    clean = pd.Series(True, index=frame.index)
    qc_used, qc_absent, qc_skipped = [], [], []
    for column in settings.qc_exclude_columns:
        if column not in frame.columns:
            qc_absent.append(column)
            continue
        scope = settings.qc_channel_scope.get(column)
        if scope is not None and not any(name in requested for name in scope):
            qc_skipped.append(column)
            continue
        clean &= ~_bool_column(frame, column)
        qc_used.append(column)
    if qc_absent:
        notes.append(
            "QC flag column(s) not present and therefore not applied: "
            + ", ".join(qc_absent)
            + "."
        )
    if qc_skipped:
        notes.append(
            "QC flag(s) not applied because they speak only about channels "
            "outside this criteria set: " + ", ".join(qc_skipped) + "."
        )
    if not qc_used:
        notes.append(
            "No QC flag columns applied here, so every sample is treated as "
            "trustworthy. Blocks may include transients the flags would have "
            "removed."
        )

    # --- flatness mask -------------------------------------------------
    flat = pd.Series(True, index=frame.index)
    used, missing = [], []
    span_needed = lag if percent_mode else window
    if len(frame) < span_needed:
        raise ExtractionError(
            f"The file has {len(frame)} sample(s), fewer than the "
            f"{span_needed}-sample "
            + ("lag window" if percent_mode else "rolling window")
            + ". Either load a longer run or reduce the window."
        )

    # Tracks which criteria channel is furthest over its own threshold at each
    # sample, so a rejected window can name the channel responsible instead of
    # just saying "not steady".
    worst_ratio = np.zeros(len(frame), dtype=float)
    worst_name = np.full(len(frame), "", dtype=object)
    # {channel: its own reference drift in percent}, reported so the scaling
    # behind the single slider is visible rather than implicit.
    #
    # Computed for every monitored channel, not only the ones in this criteria
    # set: the channels *outside* the set are the ones segmentation uses, and
    # falling back to the floor there would give them a level tolerance orders
    # of magnitude too tight and shred every run into single samples.
    references: dict[str, float] = {}
    if percent_mode:
        for column in settings.monitored_channels:
            if column in frame.columns:
                references[column] = reference_percent(
                    frame[column], lag, settings.reference_quantile
                )

    for column in requested:
        if column not in frame.columns:
            missing.append(column)
            continue
        values = pd.to_numeric(frame[column], errors="coerce")

        if percent_mode:
            if column not in settings.monitored_channels:
                missing.append(f"{column} (not monitored)")
                continue
            floor = channel_floor(values)
            metric = percent_change(values, lag, floor)
            frame[f"{column}_pct_change"] = metric
            threshold = float(settings.sensitivity) * references[column]
        else:
            threshold = settings.channel_thresholds.get(column)
            if threshold is None:
                missing.append(f"{column} (no threshold set)")
                continue
            metric = values.rolling(window, center=True, min_periods=window).std()
            frame[f"{column}_rolling_sd"] = metric

        flat &= (metric < threshold).fillna(False)
        used.append(column)

        # A missing metric (the window runs off the end of the run) is an
        # infinite ratio: it can never be inside the threshold.
        ratio = (metric / threshold).to_numpy(dtype=float)
        ratio = np.where(np.isfinite(ratio), ratio, np.inf)
        replace_here = ratio > worst_ratio
        worst_ratio = np.where(replace_here, ratio, worst_ratio)
        worst_name = np.where(replace_here, column, worst_name)

    if missing:
        notes.append(
            "Channel(s) absent from the file and therefore not used as a "
            "steadiness criterion: " + ", ".join(missing) + "."
        )
    if not used:
        raise ExtractionError(
            "None of the requested criteria channels are usable here. Wanted: "
            + ", ".join(requested)
            + "."
        )

    frame["steady_flag"] = (flat & clean).to_numpy()

    # Only name a limiter where the channel actually failed; a sample rejected
    # purely by a QC flag has no limiting channel, and saying otherwise would
    # send the operator to loosen a threshold that was never the problem.
    over_threshold = worst_ratio >= 1.0
    limiting_channel = pd.Series(
        np.where(over_threshold, worst_name, ""), index=frame.index, dtype=object
    )
    limiting_ratio = pd.Series(
        np.where(over_threshold, worst_ratio, np.nan), index=frame.index, dtype=float
    )
    qc_only = (~clean.to_numpy()) & (~over_threshold)
    limiting_channel[qc_only] = "QC flag"
    frame["limiting_channel"] = limiting_channel
    frame["limiting_ratio"] = limiting_ratio

    # --- blocks --------------------------------------------------------
    bridged = _bridge_short_gaps(frame["steady_flag"].to_numpy().copy(), max_gap)
    if percent_mode:
        # The minimum-contiguous-span rule, applied to the criterion itself
        # rather than left to the block-length filter downstream: a percent
        # change is one difference between two samples, and one favourable
        # difference is not evidence the process settled.
        before = int(bridged.sum())
        bridged = _drop_short_runs(bridged, min_span)
        erased = before - int(bridged.sum())
        if erased:
            notes.append(
                f"{erased} sample(s) passed the percent test only in stretches "
                f"shorter than {settings.min_block_duration_s:g} s and were "
                "not counted as steady."
            )

    # Channels that no longer veto a sample can still mark the boundary between
    # two operating points. Criteria channels are excluded: they cannot change
    # level without already having ended the run.
    if percent_mode:
        # Expressed as a percentage of the channel's own *current value*,
        # matching the criterion in force, so one slider governs both what
        # counts as steady and what counts as a new operating point. This is
        # the full threshold - sensitivity and segment_tolerance_factor both
        # folded in here, once - because `_split_on_percent_level_changes`
        # applies it directly rather than multiplying by tolerance_factor a
        # second time the way the absolute-criterion path does. See the
        # v1.8.2 module docstring note for why "once" needed saying at all.
        segment_channels = {
            column: settings.sensitivity
            * settings.segment_tolerance_factor
            * references[column]
            for column in settings.monitored_channels
            if column not in requested and column in frame.columns
        }
        segment_floors = {
            column: channel_floor(frame[column]) for column in segment_channels
        }
    else:
        segment_channels = {
            column: threshold
            for column, threshold in settings.channel_thresholds.items()
            if column not in requested
        }
        segment_floors = {}
    candidates: list[tuple[int, int]] = []
    n_merged_runs = 0
    for run in _runs_of_true(bridged):
        if percent_mode:
            pieces = _split_on_percent_level_changes(
                run, frame, segment_channels, segment_floors
            )
        else:
            pieces = _split_on_level_changes(
                run, frame, segment_channels, settings.segment_tolerance_factor
            )
        if len(pieces) > 1:
            n_merged_runs += 1
        candidates.extend(pieces)
    if n_merged_runs:
        notes.append(
            f"{n_merged_runs} steady stretch(es) spanned a level change in a "
            "channel outside this criteria set and were split into separate "
            "operating points rather than averaged together."
        )

    spans: list[tuple[int, int]] = []
    rejected: list[tuple[int, int]] = []
    for start, end in candidates:
        s2, e2 = start + edge_trim, end - edge_trim
        if e2 - s2 + 1 >= min_block:
            spans.append((s2, e2))
        else:
            rejected.append((start, end))

    block_id = np.full(len(frame), -1, dtype=int)
    for i, (start, end) in enumerate(spans, start=1):
        block_id[start : end + 1] = i
    frame["steady_block_id"] = block_id

    if not spans:
        notes.append(
            "No block survived. Every candidate run was shorter than "
            f"{settings.min_block_duration_s:g} s after edge trimming - try "
            "raising the extraction sensitivity or shortening the minimum "
            "block duration."
        )

    blocks = _summarise_blocks(frame, spans, settings, clean, notes)
    windows = _window_table(frame, bridged, candidates, rejected, settings, period)

    return ExtractionResult(
        blocks=blocks,
        frame=frame,
        spans=spans,
        settings=settings,
        sample_period_s=period,
        channels_used=tuple(used),
        channels_missing=tuple(missing),
        n_candidate_blocks=len(candidates),
        notes=notes,
        target_channel=target_channel,
        criteria_label=label,
        limiting_channel=limiting_channel,
        limiting_ratio=limiting_ratio,
        windows=windows,
    )


def _window_table(
    frame: pd.DataFrame,
    bridged: np.ndarray,
    candidates: list[tuple[int, int]],
    rejected: list[tuple[int, int]],
    settings: ExtractionSettings,
    period_s: float,
) -> pd.DataFrame:
    """Every contiguous stretch of the run, accepted and rejected alike.

    The block table only shows what survived. An operator asking "why wasn't
    that flat-looking bit at 40 minutes kept?" needs the rejected stretches
    too, each carrying the channel that vetoed it.
    """
    times = frame[schema.TIME_COLUMN]
    rejected_starts = {start for start, _ in rejected}
    rows: list[dict] = []

    def add(start: int, end: int, verdict: str, reason: str) -> None:
        segment_limiters = frame["limiting_channel"].iloc[start : end + 1]
        blocking = segment_limiters[segment_limiters != ""]
        limiter = blocking.mode().iloc[0] if not blocking.empty else "-"
        rows.append(
            {
                "start_time": times.iloc[start],
                "end_time": times.iloc[end],
                "duration_s": (end - start + 1) * period_s,
                "verdict": verdict,
                "limiting channel": limiter,
                "reason": reason,
            }
        )

    for start, end in candidates:
        length = (end - start + 1) * period_s
        if start in rejected_starts:
            trimmed = length - 2 * settings.edge_trim_s
            add(
                start,
                end,
                "rejected",
                f"Steady for {length:g} s, but only {max(trimmed, 0):g} s after "
                f"edge trimming - under the {settings.min_block_duration_s:g} s "
                "minimum.",
            )
        else:
            add(start, end, "accepted", "Every criteria channel held its threshold.")
    for start, end in _runs_of_true(~bridged):
        add(
            start,
            end,
            "rejected",
            "Not steady: see the limiting channel.",
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "start_time",
                "end_time",
                "duration_s",
                "verdict",
                "limiting channel",
                "reason",
            ]
        )
    return (
        pd.DataFrame(rows).sort_values("start_time").reset_index(drop=True)
    )


def extract_per_channel(
    df: pd.DataFrame,
    settings: ExtractionSettings | None = None,
) -> PerChannelExtraction:
    """Run one extraction per setpoint channel, using only its own criteria set.

    Also runs the all-channels-at-once extraction, so the before/after block
    count per channel is a measurement rather than a claim.
    """
    settings = settings or ExtractionSettings()
    notes: list[str] = []

    global_result = extract_blocks(df, settings)

    # Which channels are available to a criteria set depends on which criterion
    # is in force: the percent one watches `monitored_channels`, the absolute
    # one watches whatever still has a threshold.
    available = (
        set(settings.monitored_channels)
        if settings.criterion == CRITERION_PERCENT
        else set(settings.channel_thresholds)
    )

    results: dict[str, ExtractionResult] = {}
    groups: dict[str, StabilityGroup] = {}
    for channel, group in settings.stability_groups.items():
        members = tuple(name for name in group.members if name in available)
        dropped = [name for name in group.members if name not in available]
        if dropped:
            notes.append(
                f"{channel}: criteria channel(s) {', '.join(dropped)} have no "
                "threshold set (switched off in the UI, or absent from the "
                "file) and were left out of this channel's criteria set."
            )
        if not members:
            notes.append(
                f"{channel}: none of its criteria channels are available, so it "
                "falls back to the all-channel extraction."
            )
            results[channel] = global_result
            groups[channel] = group
            continue
        try:
            results[channel] = extract_blocks(
                df,
                settings,
                criteria=members,
                target_channel=channel,
                criteria_label=", ".join(members),
            )
        except ExtractionError as exc:
            notes.append(f"{channel}: {exc} Falling back to the all-channel blocks.")
            results[channel] = global_result
        groups[channel] = group

    return PerChannelExtraction(
        results=results,
        global_result=global_result,
        groups=groups,
        notes=notes,
    )


def _summarise_blocks(
    frame: pd.DataFrame,
    spans: list[tuple[int, int]],
    settings: ExtractionSettings,
    clean: pd.Series,
    notes: list[str],
) -> pd.DataFrame:
    """Reduce each span to one row of medians, standard errors and counts."""
    value_columns = [c for c in schema.ALL_BLOCK_VALUE_COLUMNS if c in frame.columns]

    nominal_temp = float("nan")
    if "furnace_temp_C" in frame.columns:
        settled = frame.loc[clean.to_numpy(), "furnace_temp_C"].round(0).mode()
        if not settled.empty:
            nominal_temp = float(settled.iloc[0])

    rows: list[dict] = []
    for i, (start, end) in enumerate(spans, start=1):
        segment = frame.iloc[start : end + 1]
        row: dict = {
            "block_id": i,
            "start_time": segment[schema.TIME_COLUMN].iloc[0],
            "end_time": segment[schema.TIME_COLUMN].iloc[-1],
            "duration_s": len(segment),
            "start_index": int(start),
            "end_index": int(end),
        }
        for column in value_columns:
            source = segment
            mask_column = settings.value_masks.get(column)
            if mask_column and mask_column in segment.columns:
                source = segment[~_bool_column(segment, mask_column)]
            median, se, n = robust_center_spread(source[column])
            row[f"{column}_median"] = median
            row[f"{column}_se"] = se
            row[f"{column}_n"] = n

        if CAP_REAL_COLUMN in segment.columns:
            mask_column = settings.value_masks.get("cap_OD_um")
            source = segment
            if mask_column and mask_column in segment.columns:
                source = segment[~_bool_column(segment, mask_column)]
            row["cap_OD_n_real"] = int(_bool_column(source, CAP_REAL_COLUMN).sum())

        if QC_PASS_COLUMN in segment.columns:
            row["frac_qc_pass"] = float(_bool_column(segment, QC_PASS_COLUMN).mean())
        else:
            row["frac_qc_pass"] = float("nan")

        if np.isfinite(nominal_temp) and "furnace_temp_C_median" in row:
            row["temp_flag_nonnominal"] = bool(
                abs(row["furnace_temp_C_median"] - nominal_temp)
                > settings.nominal_temp_tolerance_C
            )
        else:
            row["temp_flag_nonnominal"] = False
        rows.append(row)

    if not rows:
        return pd.DataFrame(
            columns=["block_id", "start_time", "end_time", "duration_s"]
        )

    blocks = pd.DataFrame(rows).sort_values("start_time").reset_index(drop=True)
    flagged = int(blocks["temp_flag_nonnominal"].sum())
    if flagged:
        notes.append(
            f"{flagged} block(s) sit more than "
            f"{settings.nominal_temp_tolerance_C:g} degC from the run's modal "
            f"furnace temperature of {nominal_temp:g} degC. They are kept, but "
            "they are a different process point rather than a repeat of the "
            "same one."
        )
    return blocks


# --------------------------------------------------------------------------
# Presentation
# --------------------------------------------------------------------------

# What the block table shows by default: one median / SE / n triple per channel
# is a lot of columns, so the table is built from this ordering rather than
# from the raw frame's column order.
def block_table(blocks: pd.DataFrame) -> pd.DataFrame:
    """Reorder the block summary into a readable, operator-facing table."""
    if blocks.empty:
        return blocks
    front = ["block_id", "start_time", "end_time", "duration_s"]
    channels = [
        c
        for c in schema.ALL_BLOCK_VALUE_COLUMNS
        if f"{c}_median" in blocks.columns
    ]
    ordered = list(front)
    for column in channels:
        ordered += [f"{column}_median", f"{column}_se", f"{column}_n"]
    tail = [c for c in ("cap_OD_n_real", "frac_qc_pass", "temp_flag_nonnominal") if c in blocks.columns]
    ordered += tail
    return blocks[[c for c in ordered if c in blocks.columns]]


# --------------------------------------------------------------------------
# Measuring the criterion change rather than asserting it
# --------------------------------------------------------------------------


def compare_criteria(
    df: pd.DataFrame, settings: ExtractionSettings | None = None
) -> pd.DataFrame:
    """Old absolute-threshold result against the new percent one, per mapping.

    v1.0-v1.3 validated an extraction bit-for-bit against a hand-checked
    8-block result. v1.4 changes the criterion itself, so that test can no
    longer mean what it meant. This replaces it with a measurement: run both
    criteria over the same data and report what actually differs, so the change
    is quantified rather than assumed to be an improvement.
    """
    settings = settings or ExtractionSettings()
    absolute = extract_per_channel(
        df, replace(settings, criterion=CRITERION_ABSOLUTE_SD)
    )
    percent = extract_per_channel(
        df, replace(settings, criterion=CRITERION_PERCENT)
    )
    rows = []
    for channel in absolute.results:
        before = absolute.results[channel]
        after = percent.results.get(channel)
        if after is None:
            continue
        rows.append(
            {
                "channel": channel,
                "blocks (absolute SD)": before.n_blocks,
                "blocks (percent)": after.n_blocks,
                "change": after.n_blocks - before.n_blocks,
                "settled s (absolute SD)": before.total_steady_s,
                "settled s (percent)": after.total_steady_s,
            }
        )
    return pd.DataFrame(rows)


def reference_table(
    df: pd.DataFrame, settings: ExtractionSettings | None = None
) -> pd.DataFrame:
    """Each channel's derived reference drift and the threshold it produces.

    The single slider hides a per-channel scaling; this is what makes that
    scaling visible instead of magic.
    """
    settings = settings or ExtractionSettings()
    period = detect_sample_period_s(df[schema.TIME_COLUMN])
    lag = _to_samples(settings.lag_window_s, period, minimum=1)
    rows = []
    for column in settings.monitored_channels:
        if column not in df.columns:
            continue
        reference = reference_percent(df[column], lag, settings.reference_quantile)
        rows.append(
            {
                "channel": column,
                "unit": schema.unit_of(column),
                "reference drift %": reference,
                "threshold %": settings.sensitivity * reference,
                "denominator floor": channel_floor(df[column]),
            }
        )
    return pd.DataFrame(rows)
