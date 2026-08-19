"""Preform type registry.

A new preform geometry is added by appending a `PreformDefinition` here. No
application or UI code needs to change: the selector reads the registry, and
calibrations are stored per `PreformDefinition.id`.

Only geometries with real anchor draws behind them are marked
`is_implemented`. Unimplemented entries still appear in the UI, greyed out, so
the roadmap is visible without offering anything that does not work.

This mirrors the pattern used by the HCF Draw Predictor, reimplemented here -
the two apps share no code.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from schema import (
    ANALYTIC_COLUMNS,
    DNANF_SCHEMA,
    NANF_ANALYTIC_COLUMNS,
    NANF_FEATURE_COLUMNS,
    NANF_SCHEMA,
    NANF_SETPOINT_COLUMNS,
    TUBULAR_SCHEMA,
    NESTED_ANALYTIC_COLUMNS,
    NESTED_FEATURE_COLUMNS,
    NESTED_SCHEMA,
    NESTED_SETPOINT_COLUMNS,
    NONNESTED_SCHEMA,
    REQUIRED_FEATURE_COLUMNS,
    SETPOINT_COLUMNS,
    ColumnSpec,
    PreformSchema,
)


def _schema_dict(columns: tuple[ColumnSpec, ...]) -> dict[str, dict[str, str]]:
    """Convert column specs into the {name: {unit, description}} schema shape."""
    return {
        spec.name: {"unit": spec.unit, "description": spec.description}
        for spec in columns
    }


@dataclass(frozen=True)
class PreformDefinition:
    """One preform geometry the app can calibrate and predict for."""

    id: str
    label: str
    feature_schema: dict[str, dict[str, str]]
    target_schema: dict[str, dict[str, str]]
    is_implemented: bool
    # Shown in the UI next to a disabled entry to explain why it is unavailable.
    unavailable_reason: str = ""
    analytic_schema: dict[str, dict[str, str]] = field(default_factory=dict)
    # Nominal preform outer diameter in mm, when it is a fixed property of the
    # geometry. `None` means the operator must supply it if they want the
    # geometric draw-down ratio; the kinematic ratio is used otherwise.
    nominal_preform_OD_mm: float | None = None
    # The column bundle behind the three `*_schema` dicts above. Those dicts are
    # the UI's display shape ({name: {unit, description}}); this is the same
    # information in the form the pipeline needs - typed specs, the geometric
    # ratios this geometry defines, and its pressure chain. Since v1.9 every
    # stage takes one of these rather than reading module globals, which is what
    # lets a second geometry exist without the first one's numbers moving.
    schema: PreformSchema | None = None
    # Ids this geometry was previously registered under. The id is the storage
    # key for a calibration, so renaming one without carrying its history would
    # silently orphan every file saved under the old name.
    legacy_ids: tuple[str, ...] = ()

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(self.feature_schema)

    @property
    def target_names(self) -> tuple[str, ...]:
        return tuple(self.target_schema)


# The three geometries, under the names the fabricators use.
#
# The ids changed in this release (`hc_10cap_nonnested` -> `tubular`,
# `hc_nested_3layer` -> `dnanf`) and the id is the storage key: calibrations
# live in `models/<id>/`. A bare rename would therefore orphan every stored
# calibration, so each entry carries the ids it used to be known by and
# `get_preform` accepts them. Nothing has to be migrated and no saved file has
# to be refitted - see `legacy_ids` and `paths.calibration_path`.

TUBULAR = PreformDefinition(
    id="tubular",
    label="Tubular (single capillary layer)",
    feature_schema=_schema_dict(REQUIRED_FEATURE_COLUMNS),
    target_schema=_schema_dict(SETPOINT_COLUMNS),
    analytic_schema=_schema_dict(ANALYTIC_COLUMNS),
    is_implemented=True,
    schema=TUBULAR_SCHEMA,
    legacy_ids=("hc_10cap_nonnested",),
)

# NANF: outer and inner capillary layers, no middle. New in this release.
#
# Not DNANF with a layer left empty - it has no middle geometry at all, so its
# inner layer's differential is measured against the outer layer and its
# pressure chain has two links rather than three. Ships with zero anchors, and
# `is_implemented=True` means what it has always meant: the app can work with
# this geometry, not that it is calibrated.
NANF = PreformDefinition(
    id="nanf",
    label="NANF (outer / inner capillaries, no middle)",
    feature_schema=_schema_dict(NANF_FEATURE_COLUMNS),
    target_schema=_schema_dict(NANF_SETPOINT_COLUMNS),
    analytic_schema=_schema_dict(NANF_ANALYTIC_COLUMNS),
    is_implemented=True,
    schema=NANF_SCHEMA,
)

# DNANF: the three-layer nested geometry added in v1.9, renamed. Its schema,
# criteria sets and fitted behaviour are unchanged - only the label and id.
DNANF = PreformDefinition(
    id="dnanf",
    label="DNANF (outer / middle / inner capillaries)",
    feature_schema=_schema_dict(NESTED_FEATURE_COLUMNS),
    target_schema=_schema_dict(NESTED_SETPOINT_COLUMNS),
    analytic_schema=_schema_dict(NESTED_ANALYTIC_COLUMNS),
    is_implemented=True,
    schema=DNANF_SCHEMA,
    legacy_ids=("hc_nested_3layer",),
)

REGISTRY: tuple[PreformDefinition, ...] = (TUBULAR, NANF, DNANF)

DEFAULT_PREFORM_ID = TUBULAR.id


def get_preform(preform_id: str) -> PreformDefinition:
    """Look up a geometry by its current id, or by any id it used to have.

    Accepting the old ids is what lets this release rename them without
    orphaning a stored calibration: a file saved under `hc_10cap_nonnested`
    still resolves to the tubular preform it was fitted for.
    """
    for entry in REGISTRY:
        if entry.id == preform_id:
            return entry
    for entry in REGISTRY:
        if preform_id in entry.legacy_ids:
            return entry
    known = ", ".join(p.id for p in REGISTRY)
    raise KeyError(f"Unknown preform id {preform_id!r}. Known ids: {known}")


def canonical_preform_id(preform_id: str) -> str:
    """The current id for a possibly-legacy one, or the input if unknown."""
    try:
        return get_preform(preform_id).id
    except KeyError:
        return preform_id


def implemented_preforms() -> tuple[PreformDefinition, ...]:
    return tuple(p for p in REGISTRY if p.is_implemented)
