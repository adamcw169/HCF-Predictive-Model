"""Where each steady-state block's analytic (physics-model) estimate comes from.

The calibration corrects an analytic estimate towards reality, so every anchor
block needs an analytic estimate to sit beside it. Today that estimate is read
out of a file the supervisor's fast estimator has already been run over.
Eventually the estimator itself will be callable from Python and the estimate
will be computed on demand.

Both cases are the same question - "what does the physics model say for this
block?" - so the rest of the app depends only on the `AnalyticSource`
interface and never on where the answer came from.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd

import schema
from steady_state import robust_center_spread

# Fewer contributing rows than this and the block's estimate is reported as
# thin rather than presented as if it were solid. It is not an error - a short
# block legitimately overlaps few analytic rows - but the operator should see
# it before it becomes an anchor.
DEFAULT_MIN_ROWS = 5

STATUS_OK = "ok"
STATUS_SPARSE = "sparse"
STATUS_EMPTY = "empty"


class AnalyticSource(ABC):
    """Supplies the analytic estimate for one steady-state block."""

    @abstractmethod
    def estimate_for_block(self, block: dict) -> dict:
        """Return analytic furnace_temp_C, draw_speed_m_min, core_dP_kPa,
        Pocap_kPa estimates for one steady-state block.

        `block` is one row of the steady-state block summary, as a dict; only
        `start_time` and `end_time` are required of it.

        The returned dict carries the four `analytic_*` values, a `_se` for
        each, the number of underlying rows that contributed (`n_rows`), a
        `status` of "ok", "sparse" or "empty", and a human-readable `note`.
        A caller must never treat a returned number as usable without looking
        at `status`.
        """


@dataclass
class StaticDatasetAnalyticSource(AnalyticSource):
    """Analytic estimates read from a dataset the estimator was already run over.

    A block's estimate is the median of every row whose timestamp falls inside
    the block's ``[start_time, end_time]`` window. The median matches how the
    block's own measured values are reduced, so the two sides of the
    calibration are summarised the same way.

    `tolerance_s` widens the window at both ends. It defaults to zero - the
    literal window - and exists because analytic files are often stamped at
    coarser resolution than the raw data. `timestamp_resolution_s` reports the
    resolution that was detected so the UI can suggest a tolerance instead of
    silently applying one.
    """

    frame: pd.DataFrame
    tolerance_s: float = 0.0
    min_rows: int = DEFAULT_MIN_ROWS
    label: str = ""

    def __post_init__(self) -> None:
        missing = [c for c in schema.ANALYTIC_NAMES if c not in self.frame.columns]
        if missing:
            raise ValueError(
                "The analytic file is missing required column(s): "
                + ", ".join(missing)
            )
        if schema.TIME_COLUMN not in self.frame.columns:
            raise ValueError(
                f"The analytic file needs a '{schema.TIME_COLUMN}' column so "
                "its rows can be matched to a block's time window."
            )
        self._times = pd.to_datetime(self.frame[schema.TIME_COLUMN], errors="coerce")
        self._values = {
            name: pd.to_numeric(self.frame[name], errors="coerce")
            for name in schema.ANALYTIC_NAMES
        }

    # ------------------------------------------------------------------

    @property
    def n_rows(self) -> int:
        return int(self._times.notna().sum())

    @property
    def time_span(self) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
        valid = self._times.dropna()
        if valid.empty:
            return None, None
        return valid.min(), valid.max()

    @property
    def timestamp_resolution_s(self) -> float:
        """Smallest non-zero spacing between consecutive timestamps.

        A file stamped once a minute over 1 Hz data reports 60 here even though
        it holds ~60 rows per stamp, which is exactly the case where a naive
        window match can come up empty for a short block.
        """
        deltas = self._times.dropna().sort_values().diff().dt.total_seconds()
        deltas = deltas[deltas > 0]
        if deltas.empty:
            return 0.0
        return float(deltas.min())

    # ------------------------------------------------------------------

    def estimate_for_block(self, block: dict) -> dict:
        start = pd.to_datetime(block["start_time"])
        end = pd.to_datetime(block["end_time"])
        pad = pd.Timedelta(seconds=float(self.tolerance_s))
        mask = (self._times >= start - pad) & (self._times <= end + pad)
        n_rows = int(mask.sum())

        result: dict = {"n_rows": n_rows}
        for name in schema.ANALYTIC_NAMES:
            median, se, count = robust_center_spread(self._values[name][mask])
            result[name] = median
            result[f"{name}_se"] = se
            result[f"{name}_n"] = count

        if n_rows == 0:
            result["status"] = STATUS_EMPTY
            result["note"] = (
                f"No analytic row falls in {start:%H:%M:%S}-{end:%H:%M:%S}. "
                "This block cannot be an anchor until the window is widened or "
                "a matching analytic file is loaded."
            )
            # An empty match must not look like a number. Blank the values so a
            # downstream bug cannot quietly fit against them.
            for name in schema.ANALYTIC_NAMES:
                result[name] = float("nan")
        elif n_rows < self.min_rows:
            result["status"] = STATUS_SPARSE
            result["note"] = (
                f"Only {n_rows} analytic row(s) contributed. The estimate is "
                "usable but its own spread is barely measured."
            )
        else:
            result["status"] = STATUS_OK
            result["note"] = f"{n_rows} analytic row(s) contributed."

        # A channel can be individually unusable even when the block matched
        # plenty of rows, if that one column is blank for those rows.
        blank = [
            name
            for name in schema.ANALYTIC_NAMES
            if result[f"{name}_n"] == 0 and n_rows > 0
        ]
        if blank:
            result["status"] = STATUS_SPARSE
            result["note"] += (
                " No usable value for: " + ", ".join(blank) + "."
            )
        return result

    def estimates_for_blocks(self, blocks: pd.DataFrame) -> pd.DataFrame:
        """Run `estimate_for_block` over a whole block summary."""
        if blocks.empty:
            return pd.DataFrame()
        rows = [
            {"block_id": row["block_id"], **self.estimate_for_block(row)}
            for row in blocks.to_dict("records")
        ]
        return pd.DataFrame(rows)


class LiveEstimatorAnalyticSource(AnalyticSource):
    """Seam for calling the supervisor's fast estimator directly.

    Deliberately unimplemented. When the estimator becomes importable, this is
    the only class that needs writing: it computes an analytic estimate from a
    block's own geometry and process inputs instead of looking one up. Nothing
    else in the app changes, because everything else depends on
    `AnalyticSource` rather than on this class.

    Until then the app uses `StaticDatasetAnalyticSource` on Tab 1 and typed-in
    values on Tab 2. Neither path constructs this class.
    """

    def estimate_for_block(self, block: dict) -> dict:
        raise NotImplementedError(
            "The fast estimator is not callable from this app yet. Run it "
            "yourself and supply its output as a file on the 'Extract & "
            "calibrate' tab, or type it into the 'Predict' tab. Implementing "
            "this class is the only change needed to make the estimator live."
        )


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def load_analytic_dataset(path: str) -> tuple[pd.DataFrame, list[str]]:
    """Read an analytic estimates file and check it can be matched to blocks.

    Accepts both shapes described in the schema: a full experimental dataset
    that happens to carry `analytic_*` columns, and a file holding only those
    columns plus a timestamp.
    """
    notes: list[str] = []
    df = pd.read_csv(path)

    if schema.TIME_COLUMN not in df.columns:
        raise ValueError(
            f"The analytic file needs a '{schema.TIME_COLUMN}' column. Blocks "
            "are matched to analytic rows by time window, so without it there "
            "is no way to say which rows belong to which block."
        )
    df[schema.TIME_COLUMN] = pd.to_datetime(df[schema.TIME_COLUMN], errors="coerce")
    unparsed = int(df[schema.TIME_COLUMN].isna().sum())
    if unparsed:
        df = df[df[schema.TIME_COLUMN].notna()]
        notes.append(f"Dropped {unparsed} row(s) with an unparseable timestamp.")

    pocap = schema.analytic_name(schema.POCAP_COLUMN)
    core = schema.analytic_name(schema.CORE_DP_COLUMN)
    outer = f"{schema.ANALYTIC_PREFIX}{schema.OUTER_DP_COLUMN}"
    if pocap not in df.columns and core in df.columns and outer in df.columns:
        df[pocap] = pd.to_numeric(df[core], errors="coerce") + pd.to_numeric(
            df[outer], errors="coerce"
        )
        notes.append(
            f"{pocap} derived as {core} + {outer} (the file carried only the "
            "analytic differential)."
        )

    missing = [c for c in schema.ANALYTIC_NAMES if c not in df.columns]
    if missing:
        raise ValueError(
            "The analytic file is missing required column(s): "
            + ", ".join(missing)
            + ". Expected the fast estimator's output columns: "
            + ", ".join(schema.ANALYTIC_NAMES)
            + "."
        )

    for name in schema.ANALYTIC_NAMES:
        df[name] = pd.to_numeric(df[name], errors="coerce")

    if df.empty:
        raise ValueError("The analytic file contains no usable rows.")
    return df.reset_index(drop=True), notes


def coverage_summary(estimates: pd.DataFrame) -> str:
    """One line describing how well the analytic file covered the blocks."""
    if estimates.empty:
        return "No blocks to match."
    counts = estimates["status"].value_counts()
    empty = int(counts.get(STATUS_EMPTY, 0))
    sparse = int(counts.get(STATUS_SPARSE, 0))
    total = len(estimates)
    if empty == total:
        return (
            f"None of the {total} block(s) matched any analytic row. Check that "
            "the analytic file covers the same run as the raw timeseries."
        )
    parts = [f"{total - empty - sparse} of {total} block(s) matched cleanly"]
    if sparse:
        parts.append(f"{sparse} thinly")
    if empty:
        parts.append(f"{empty} not at all")
    median_rows = int(np.nanmedian(estimates["n_rows"].to_numpy(dtype=float)))
    return "; ".join(parts) + f". Median {median_rows} analytic row(s) per block."
