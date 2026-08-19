"""Getting several raw CSVs onto one 1 Hz grid, in the shape extraction expects.

Why this exists
---------------
Until v1.8 the app took one experimental CSV that carried every column. Real
acquisition does not work that way: the capillary OD comes off a different
instrument than the furnace and capstan channels, writes its own file, and does
so on its own clock at its own rate. Asking the operator to pre-merge those
files by hand is asking them to do the one step most likely to go wrong quietly.

What this module does and does not do
-------------------------------------
It does exactly one thing: turn N files into the single dataframe
`steady_state.extract_blocks` already consumes. It does not extract, does not
fit, and does not decide anything about steadiness. Everything downstream of the
returned frame is untouched by v1.8.

The single-file path is deliberately unchanged
----------------------------------------------
One file with no offset is delegated straight to `steady_state.load_raw_timeseries`
and never resampled. That is a correctness decision, not laziness: resampling
a file onto 1-second bins moves values, and every calibration and prediction in
this app's history was fitted on the unresampled read. There is also nothing to
gain - resampling exists to put *several* clocks on a common grid, and with one
file there is no second clock. So a lone file behaves in v1.8 exactly as it did
in v1.7, and the tests assert that frame-for-frame.

The consequence is worth stating plainly, because it is real: the same file
loaded alone and loaded alongside a second file can produce slightly different
block statistics, because in the second case it has been binned. `MergeReport`
says so rather than leaving it to be discovered.

Binning must not launder a QC flag
----------------------------------
Averaging is right for a measurement and wrong for a flag. `qc_outlier` averaged
over three samples where one was set gives 0.333, which is not a boolean; worse,
`steady_state._bool_column` reads an unparseable value as False, so a bin
containing a bad sample would come out looking clean. Flags are therefore
aggregated by their meaning rather than by mean:

    "something is wrong here"  ->  any()   one bad sample taints the bin
    "this sample is good"      ->  all()   one bad sample spoils the bin

Both directions are conservative in the same direction: toward not trusting a
second that contains something untrustworthy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import schema
import steady_state as ss

# One second. The grid the whole app already assumes - block durations, minimum
# lengths and lag windows are all expressed in seconds against a ~1 Hz stream.
RESAMPLE_RULE = "1s"

# Flags whose True means "this sample is not trustworthy". Taken from
# `steady_state` rather than restated, so a flag added there cannot be silently
# averaged here.
FLAGS_ANY: frozenset[str] = frozenset(
    set(ss.DEFAULT_QC_EXCLUDE_COLUMNS) | set(ss.DEFAULT_VALUE_MASKS.values())
)

# Flags whose True means "this sample is good".
FLAGS_ALL: frozenset[str] = frozenset({ss.QC_PASS_COLUMN, ss.CAP_REAL_COLUMN})

# How close two files' copies of the same column have to agree before the
# disagreement is reported. Relative to the column's own spread, so it means the
# same thing for a temperature near 2000 and a ratio near 1.
CONFLICT_TOLERANCE_FRACTION = 1.0e-6


class IngestError(ValueError):
    """Raised when the supplied files cannot be turned into one frame."""


@dataclass(frozen=True)
class SourceFile:
    """One experimental CSV, and any known clock error against the first file.

    `offset_s` is added to this file's timestamps before binning. It is an
    operator-supplied correction for a *known* instrument clock discrepancy -
    the app does not estimate it, because cross-correlating two instruments that
    measure different quantities would be guessing, and a wrong guess here
    silently misaligns every block.
    """

    path: str
    offset_s: float = 0.0

    @property
    def name(self) -> str:
        return Path(self.path).name


@dataclass(frozen=True)
class ColumnCoverage:
    """How well one merged column is populated across the final 1-second grid."""

    column: str
    source: str
    n_bins: int
    n_averaged: int  # bins built from more than one raw sample
    n_single: int  # bins built from exactly one raw sample
    n_empty: int  # bins with no raw sample at all

    def _fraction(self, count: int) -> float:
        return (count / self.n_bins) if self.n_bins else float("nan")

    @property
    def frac_averaged(self) -> float:
        return self._fraction(self.n_averaged)

    @property
    def frac_single(self) -> float:
        return self._fraction(self.n_single)

    @property
    def frac_empty(self) -> float:
        return self._fraction(self.n_empty)


@dataclass
class MergeReport:
    """What the merge did, in enough detail to catch a bad file.

    Deliberately verbose about provenance. The failure this is built against is
    an operator seeing "cap_OD_um is missing" and concluding their instrument is
    broken, when in fact they simply did not select the second file - so the
    report distinguishes "no file had this column" from "this file did not, but
    that one did".
    """

    sources: tuple[SourceFile, ...] = ()
    column_source: dict[str, str] = field(default_factory=dict)
    coverage: dict[str, ColumnCoverage] = field(default_factory=dict)
    conflicts: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    resampled: bool = False
    n_bins: int = 0
    # Columns on the merged frame, including any derived after the merge. Kept
    # so "missing" can be answered against what came out rather than against
    # what each file put in - `Pocap_kPa` is often neither file's column and is
    # nonetheless present.
    columns: tuple[str, ...] = ()

    @property
    def columns_by_source(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {source.name: [] for source in self.sources}
        for column, source in sorted(self.column_source.items()):
            out.setdefault(source, []).append(column)
        return out

    def missing_columns(
        self, preform_schema: schema.PreformSchema | None = None
    ) -> list[str]:
        """Columns the pipeline consumes that no supplied file carried.

        The genuinely-absent set, as opposed to the merely-in-another-file set.
        Checked against everything extraction summarises per block, not against
        the six steadiness-criteria channels: `cap_OD_um` is not a criterion but
        every anchor needs it, so a run missing it is missing something real and
        must not be waved through because the criteria list happens not to
        mention it.

        Per-preform, and deliberately *not* the union of every geometry's
        columns: this is a statement about what the loaded run ought to have
        had, so measuring a non-nested file against the nested layers would
        report six absences that are not absences at all. That is the one place
        the union list would be wrong - everywhere else it is filtered against
        the columns actually present, where a wider list is harmless.

        Answered against the merged frame, so a column derived after the merge -
        `Pocap_kPa` from the two differentials - counts as present.
        """
        preform_schema = preform_schema or schema.NONNESTED_SCHEMA
        available = set(self.columns) or set(self.column_source)
        return [
            column
            for column in preform_schema.block_value_columns
            if column not in available
        ]

    def missing_criteria_channels(self) -> list[str]:
        """Steadiness-criteria channels absent from the merge.

        Weaker than `missing_columns`: a missing criterion relaxes the
        extraction rather than breaking it, which is why the two are reported
        as different things.
        """
        available = set(self.columns) or set(self.column_source)
        return [
            column
            for column in ss.DEFAULT_CHANNEL_THRESHOLDS
            if column not in available
        ]

    def coverage_table(self) -> pd.DataFrame:
        rows = []
        for column in sorted(self.coverage):
            entry = self.coverage[column]
            rows.append(
                {
                    "column": column,
                    "from file": entry.source,
                    "1 s bins": entry.n_bins,
                    "averaged (>1 sample)": entry.n_averaged,
                    "single sample": entry.n_single,
                    "no sample": entry.n_empty,
                    "% averaged": entry.frac_averaged * 100.0,
                    "% single": entry.frac_single * 100.0,
                    "% empty": entry.frac_empty * 100.0,
                }
            )
        return pd.DataFrame(rows)

    def source_table(self) -> pd.DataFrame:
        rows = []
        for source in self.sources:
            contributed = self.columns_by_source.get(source.name, [])
            rows.append(
                {
                    "file": source.name,
                    "time offset (s)": source.offset_s,
                    "columns contributed": len(contributed),
                    "columns": ", ".join(contributed),
                }
            )
        return pd.DataFrame(rows)

    def describe(self) -> str:
        if not self.resampled:
            return f"Loaded {self.sources[0].name} unchanged (single file)."
        parts = [
            f"Merged {len(self.sources)} file(s) onto {self.n_bins:,} "
            f"1-second bins."
        ]
        for source in self.sources:
            contributed = self.columns_by_source.get(source.name, [])
            offset = (
                f", offset {source.offset_s:+g} s" if source.offset_s else ""
            )
            parts.append(
                f"{source.name}{offset}: {len(contributed)} column(s)."
            )
        return " ".join(parts)


def _read_one(source: SourceFile) -> pd.DataFrame:
    """Read one CSV and index it by a corrected, parsed timestamp."""
    try:
        frame = pd.read_csv(source.path)
    except OSError as exc:
        raise IngestError(f"{source.name} could not be read: {exc}") from exc
    if schema.TIME_COLUMN not in frame.columns:
        raise IngestError(
            f"{source.name} has no '{schema.TIME_COLUMN}' column. Every "
            "experimental file needs one: the files are merged on time, so a "
            "file without a clock cannot be placed against the others."
        )
    times = pd.to_datetime(frame[schema.TIME_COLUMN], errors="coerce")
    frame = frame[times.notna()].copy()
    times = times[times.notna()]
    if frame.empty:
        raise IngestError(f"{source.name} has no rows with a usable timestamp.")
    if source.offset_s:
        times = times + pd.Timedelta(seconds=float(source.offset_s))
    frame = frame.drop(columns=[schema.TIME_COLUMN])
    frame.index = pd.DatetimeIndex(times.to_numpy(), name=schema.TIME_COLUMN)
    return frame.sort_index()


def _split_columns(frame: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """Partition a file's columns into measurements, bad-flags and good-flags."""
    flags_any = [c for c in frame.columns if c in FLAGS_ANY]
    flags_all = [c for c in frame.columns if c in FLAGS_ALL]
    known = set(flags_any) | set(flags_all)
    values = [c for c in frame.columns if c not in known]
    return values, flags_any, flags_all


def _resample_one(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bin one file to 1 s, returning the binned frame and per-column counts.

    The counts are of *raw samples that carried a value* per bin, which is what
    lets the report distinguish an averaged second from a single-sample second
    from an empty one. Counted per column rather than per row because a file can
    perfectly well log one channel faster than another.
    """
    values, flags_any, flags_all = _split_columns(frame)

    numeric = frame[values].apply(pd.to_numeric, errors="coerce")
    grouped = numeric.resample(RESAMPLE_RULE)
    binned = grouped.mean()
    counts = grouped.count()

    for column in flags_any:
        flags = ss._bool_column(frame, column)
        # any(): one untrustworthy sample makes the whole second untrustworthy.
        binned[column] = flags.resample(RESAMPLE_RULE).max().fillna(False).astype(bool)
        counts[column] = flags.resample(RESAMPLE_RULE).count()
    for column in flags_all:
        flags = ss._bool_column(frame, column)
        # all(): the second is only good if every sample in it was.
        binned[column] = flags.resample(RESAMPLE_RULE).min().fillna(False).astype(bool)
        counts[column] = flags.resample(RESAMPLE_RULE).count()

    return binned, counts


def _conflict_note(
    column: str, kept: pd.Series, other: pd.Series, kept_from: str, other_from: str
) -> str | None:
    """Describe a disagreement between two files' copies of one column.

    Returns None when they agree everywhere they overlap. A returned string is a
    warning, never a resolution: the merge keeps the first file's copy and says
    so, because picking the "better" one automatically would be inventing a
    judgement the data does not support.
    """
    both = kept.notna() & other.notna()
    if not bool(both.any()):
        return None
    a = kept[both].to_numpy(float)
    b = other[both].to_numpy(float)
    if a.dtype == bool or b.dtype == bool:
        differing = a != b
    else:
        scale = float(np.nanmax(np.abs(np.concatenate([a, b])))) or 1.0
        differing = np.abs(a - b) > CONFLICT_TOLERANCE_FRACTION * scale
    n_diff = int(differing.sum())
    if not n_diff:
        return None
    worst = float(np.nanmax(np.abs(a - b))) if a.dtype != bool else 1.0
    return (
        f"'{column}' appears in both {kept_from} and {other_from} and they "
        f"disagree on {n_diff:,} of {int(both.sum()):,} overlapping second(s) "
        f"(largest difference {worst:.6g}). {kept_from}'s copy was kept. This "
        "is reported rather than resolved: two instruments disagreeing about "
        "the same quantity is a data problem, not a preference."
    )


def load_sources(
    sources: list[SourceFile] | list[str],
    preform_schema: schema.PreformSchema | None = None,
) -> tuple[pd.DataFrame, MergeReport]:
    """Read, bin and merge experimental files into one frame for extraction.

    Returns the frame in exactly the shape `steady_state.load_raw_timeseries`
    returns it - `time_utc` as a column, sorted, numeric columns coerced, and
    `Pocap_kPa` derived if it was not supplied - so nothing downstream can tell
    how many files it came from.
    """
    entries = [
        source if isinstance(source, SourceFile) else SourceFile(str(source))
        for source in sources
    ]
    if not entries:
        raise IngestError("No experimental file was selected.")

    # One file and no clock correction is the v1.7 path, byte for byte. See the
    # module docstring: binning moves values, and there is nothing to align to.
    if len(entries) == 1 and not entries[0].offset_s:
        frame, notes = ss.load_raw_timeseries(entries[0].path, preform_schema)
        report = MergeReport(
            sources=tuple(entries),
            column_source={c: entries[0].name for c in frame.columns},
            columns=tuple(frame.columns),
            notes=list(notes)
            + [
                "Single file: read as-is, not resampled. Binning is only applied "
                "when several files have to be placed on a common clock."
            ],
            resampled=False,
            n_bins=len(frame),
        )
        return frame, report

    report = MergeReport(sources=tuple(entries), resampled=True)

    binned_frames: list[pd.DataFrame] = []
    count_frames: list[pd.DataFrame] = []
    for source in entries:
        raw = _read_one(source)
        binned, counts = _resample_one(raw)
        binned_frames.append(binned)
        count_frames.append(counts)
        if source.offset_s:
            report.notes.append(
                f"{source.name}: timestamps shifted by {source.offset_s:+g} s "
                "before binning, as instructed."
            )

    # The union of every file's bins, so a file that starts late or ends early
    # leaves empty seconds rather than truncating the others.
    index = binned_frames[0].index
    for other in binned_frames[1:]:
        index = index.union(other.index)
    index = pd.DatetimeIndex(index, name=schema.TIME_COLUMN).sort_values()

    merged = pd.DataFrame(index=index)
    merged_counts: dict[str, pd.Series] = {}
    for source, binned, counts in zip(entries, binned_frames, count_frames):
        aligned = binned.reindex(index)
        aligned_counts = counts.reindex(index).fillna(0)
        for column in binned.columns:
            if column in merged.columns:
                note = _conflict_note(
                    column,
                    merged[column],
                    aligned[column],
                    report.column_source[column],
                    source.name,
                )
                if note:
                    report.conflicts.append(note)
                continue
            merged[column] = aligned[column]
            merged_counts[column] = aligned_counts[column]
            report.column_source[column] = source.name

    for column, counts in merged_counts.items():
        values = counts.to_numpy(float)
        report.coverage[column] = ColumnCoverage(
            column=column,
            source=report.column_source[column],
            n_bins=len(index),
            n_averaged=int((values > 1).sum()),
            n_single=int((values == 1).sum()),
            n_empty=int((values == 0).sum()),
        )

    report.n_bins = len(index)

    # Reindexing onto the union puts NaN in every second a file did not cover,
    # which turns a bool column into object. `steady_state._bool_column` cannot
    # parse NaN and falls back to False, so the values would survive - but the
    # dtype would not, and a flag column that is sometimes bool and sometimes
    # object is a trap for the next person. Restored explicitly, with absence
    # meaning False in both senses: no evidence of a fault, and no evidence of a
    # pass. Both are the cautious reading.
    for column in merged.columns:
        if column in FLAGS_ANY or column in FLAGS_ALL:
            merged[column] = merged[column].fillna(False).astype(bool)

    # Back into the shape everything downstream already expects.
    frame = merged.reset_index()
    for column in schema.ALL_BLOCK_VALUE_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    active = preform_schema or schema.TUBULAR_SCHEMA
    frame, pressure_notes = active.derive_pressures(frame)
    report.columns = tuple(frame.columns)
    report.notes.extend(pressure_notes)
    report.notes.append(
        f"Every file was binned to {RESAMPLE_RULE} by averaging within each "
        "bin, then joined on the bin timestamp. A file loaded alone is not "
        "binned, so block statistics can differ slightly between loading this "
        "file by itself and loading it alongside another."
    )
    return frame, report
