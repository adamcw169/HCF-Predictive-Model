"""Write the input spreadsheet the supervisor's estimator consumes.

One row per surviving steady-state block, carrying that block's *median*
values - not per raw sample. The estimator is being asked for one analytic
estimate per anchor, and an anchor is a block; handing it 6,663 raw rows would
ask a different question and return an answer this app has nowhere to put.

The round trip is deliberately external and stays that way: this writes a CSV,
the operator runs the estimator themselves, and its output comes back through
the existing "Load analytic estimates..." picker. Nothing here opens a
connection to anything.

Column format
-------------
The 13-column tubular layout below is **confirmed working** - it is taken
directly from a file the supervisor's estimator has already accepted, and
`tests/test_analytic_export.py` pins it against that exact header.

The multi-layer extension for NANF and DNANF is **not confirmed**. No working
multi-layer example exists yet, so the extra column names here are this app's
best guess, chosen to match the naming the single-layer format already uses
(`cap_OD_um`/`cap_ID_um`/`outer_dP_kPa` -> `cap_OD_middle_um`/
`cap_ID_middle_um`/`mid_dP_kPa`, and likewise `inner`). They must be checked
against the estimator's actual multi-layer input requirements before anyone
relies on the result. `NANF_DNANF_FORMAT_IS_UNCONFIRMED` and the warning
`export_blocks` returns exist so that caveat reaches the operator rather than
living only in this docstring.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import schema

# The confirmed tubular layout, in order. Do not reorder or rename: this is a
# file format another program parses, not a report.
TUBULAR_EXPORT_COLUMNS: tuple[str, ...] = (
    "sample",
    "time_utc",
    "feed_speed_mm_min",
    "draw_speed_m_min",
    "tension_g",
    "furnace_temp_C",
    "fibre_OD_um",
    "fibre_ID_um",
    "cap_OD_um",
    "cap_ID_um",
    "core_dP_kPa",
    "outer_dP_kPa",
    "atm_P_kPa",
)

# Whether the multi-layer column names have been checked against a file the
# estimator actually accepted. Flip this only alongside a real example.
NANF_DNANF_FORMAT_IS_UNCONFIRMED = True

UNCONFIRMED_WARNING = (
    "The extra capillary-layer columns in this export are a best guess. Only "
    "the 13-column single-layer format has been confirmed against a file the "
    "estimator accepted; the multi-layer column names have not. Check them "
    "against the estimator's actual NANF/DNANF input requirements before "
    "relying on the result."
)

# Which raw differential and geometry columns each extra layer contributes,
# appended after the confirmed 13 in outward-in order. `outer` is already in
# the confirmed format, so it is not repeated here.
_EXTRA_LAYER_COLUMNS: dict[str, tuple[str, ...]] = {
    "middle": ("cap_OD_middle_um", "cap_ID_middle_um", schema.MID_DP_COLUMN),
    "inner": ("cap_OD_inner_um", "cap_ID_inner_um", schema.INNER_DP_COLUMN),
}


@dataclass(frozen=True)
class ExportResult:
    """What was written, and anything the operator needs to know about it."""

    path: Path
    frame: pd.DataFrame
    n_rows: int
    warnings: tuple[str, ...] = ()
    missing_columns: tuple[str, ...] = ()

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(self.frame.columns)


def export_columns_for(preform_schema: schema.PreformSchema) -> tuple[str, ...]:
    """The column order for one geometry.

    Tubular is exactly the confirmed 13. A multi-layer geometry appends only
    the layers it actually has - NANF gets the inner columns and no middle
    ones, because it has no middle layer and an empty column would invite the
    estimator to read a blank as a measurement.
    """
    columns = list(TUBULAR_EXPORT_COLUMNS)
    layers = {link.layer for link in preform_schema.pressure_chain}
    for layer, extra in _EXTRA_LAYER_COLUMNS.items():
        if layer in layers:
            columns.extend(extra)
    return tuple(columns)


# The confirmed format names the single capillary layer `cap_OD_um` /
# `cap_ID_um`. A multi-layer geometry calls that same layer `cap_OD_outer_um` /
# `cap_ID_outer_um`, so the confirmed columns are filled from the outer layer
# rather than left empty - it is the same physical measurement under the name
# the estimator already expects.
_SOURCE_ALIASES: dict[str, tuple[str, ...]] = {
    "cap_OD_um": ("cap_OD_um", "cap_OD_outer_um"),
    "cap_ID_um": ("cap_ID_um", "cap_ID_outer_um"),
}


def _block_value(row: pd.Series, column: str):
    """One block's value for an export column, from its median.

    The block table stores summaries as `<column>_median`; the estimator wants
    the bare column name. A column the run did not carry comes back as NaN
    rather than raising, so a partially-instrumented run still exports and the
    gap is visible in the file instead of blocking the whole export.
    """
    for source in _SOURCE_ALIASES.get(column, (column,)):
        for candidate in (f"{source}_median", source):
            if candidate in row.index:
                value = row[candidate]
                if pd.notna(value):
                    return value
    return np.nan


def build_export_frame(
    blocks: pd.DataFrame,
    preform_schema: schema.PreformSchema | None = None,
    timestamp: str = "start",
) -> tuple[pd.DataFrame, list[str]]:
    """One row per block, in the estimator's column order.

    `timestamp` chooses what `time_utc` carries. The default is the block's
    **start time**, which is what the confirmed example file uses and what
    makes a row traceable back to a moment in the run; `"midpoint"` is offered
    because a block's midpoint is arguably more representative of the settled
    stretch as a whole. Documented rather than silent either way.
    """
    preform_schema = preform_schema or schema.TUBULAR_SCHEMA
    columns = export_columns_for(preform_schema)
    missing: list[str] = []

    rows: list[dict] = []
    for _, block in blocks.iterrows():
        if timestamp == "midpoint" and {"start_time", "end_time"} <= set(block.index):
            when = block["start_time"] + (block["end_time"] - block["start_time"]) / 2
        else:
            when = block.get("start_time")

        row: dict = {}
        for column in columns:
            if column == "sample":
                # The block id, so a row in the returned estimate file can be
                # matched back to the block it came from.
                row[column] = int(block["block_id"])
            elif column == "time_utc":
                row[column] = when
            else:
                value = _block_value(block, column)
                if isinstance(value, float) and np.isnan(value) and column not in missing:
                    missing.append(column)
                row[column] = value
        rows.append(row)

    frame = pd.DataFrame(rows, columns=list(columns))
    return frame, missing


def export_blocks(
    blocks: pd.DataFrame,
    path: str | Path,
    preform_schema: schema.PreformSchema | None = None,
    timestamp: str = "start",
) -> ExportResult:
    """Write the estimator input spreadsheet for a set of surviving blocks.

    `blocks` is expected to be the block table *after* any exclusions - the
    export should describe what the calibration will actually rest on, not
    every block the extraction originally found.
    """
    preform_schema = preform_schema or schema.TUBULAR_SCHEMA
    if blocks is None or blocks.empty:
        raise ValueError(
            "There are no surviving blocks to export. Extract steady-state "
            "blocks first, and check that exclusions have not removed all of "
            "them."
        )

    frame, missing = build_export_frame(blocks, preform_schema, timestamp)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)

    warnings: list[str] = []
    if preform_schema is not schema.TUBULAR_SCHEMA and NANF_DNANF_FORMAT_IS_UNCONFIRMED:
        warnings.append(UNCONFIRMED_WARNING)
    if missing:
        warnings.append(
            "This run carried no data for: "
            + ", ".join(missing)
            + ". Those columns are written empty - the estimator will see a "
            "blank, not a zero."
        )
    return ExportResult(
        path=path,
        frame=frame,
        n_rows=len(frame),
        warnings=tuple(warnings),
        missing_columns=tuple(missing),
    )
