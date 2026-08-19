"""Dev-mode training: run the Auto search on a split, score it on held-out data.

Why this exists
---------------
`calibration.search_auto` scores every candidate by leave-one-block-out
cross-validation. That is a real check, and it is not the check this module
performs.

LOO-CV asks: *given that I am searching this list of candidates, does the
winner generalise to the other blocks of this run?* It cannot ask whether the
list itself was a good idea. Every candidate is scored on the same anchors, the
winner is the one that scored best on them, and the score that made it the
winner is therefore not an unbiased estimate of how it will do next time - the
more candidates compared, the more of that score is the luck of this particular
geometry sweep. v1.5 took the search from three candidates to two or three
dozen, which makes the gap wider, not narrower.

So the search gets a subset, and the winner is scored once against blocks that
took no part in choosing it. Two numbers, never conflated:

    LOO-CV, from the search   - how the winner did on the data it was chosen on
    held-out, never seen      - how it did on data it was not

If the second is much worse than the first, the search found something about
this run rather than about the process. That is exactly the thing a single
number cannot tell you.

Chronological, not shuffled
---------------------------
Blocks are ordered in time and a draw run walks through its operating points in
sequence, so neighbouring blocks are more alike than distant ones. A random
split would put a block's near-twin on the other side of the partition and
report a held-out error that is really an interpolation error. The cut is by
time, and the held-out portion is the end of the run.

Extract first, split second (v1.6)
----------------------------------
The ordering is a contract, not an implementation detail:

    1. extraction runs once, on the whole raw series, and yields the block list
    2. the split divides *that list* by index

So the search fraction cannot change how many blocks exist, and the cut always
falls between two blocks rather than through one. `split_anchors` carries the
channel's total block count on the result and `TrainingReport.block_counts`
reports it per channel, so the invariant is visible on screen instead of being
a claim in a docstring.

v1.6 also removes the way the reported block count *could* fall as the search
fraction rose. It was never extraction: it was this module discarding a whole
channel when the fraction left fewer than `MIN_HELD_OUT_BLOCKS` on the far side
of the cut. On the reference run, 90% left one held-out block for
`furnace_temp_C` and `draw_speed_m_min`, both channels were skipped, and 48
blocks became 29 - which reads exactly like a smaller extraction and is not one.
The cut is now clamped so the held-out minimum is always reserved, and a channel
is skipped only when it genuinely cannot seat both minimums at any fraction.
Raising the search fraction can therefore move the boundary, but never removes a
channel that a lower fraction would have trained.

Nothing here runs on its own. It produces a recommendation; adopting it is an
explicit act in the UI.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import calibration as calib
import schema

# Fraction of blocks the search is allowed to see. The remainder is held out.
# 80/20 matches the split convention used elsewhere in this project.
DEFAULT_SEARCH_FRACTION = 0.80

# Below this many blocks on either side of the cut there is nothing to learn
# from the exercise: a search set too small to fit on, or a held-out set too
# small for its error to mean anything.
MIN_SEARCH_BLOCKS = 5
MIN_HELD_OUT_BLOCKS = 2


@dataclass
class SplitSpec:
    """How to divide one channel's blocks into search and held-out."""

    search_fraction: float = DEFAULT_SEARCH_FRACTION
    # {channel: {block_id}} forced into the held-out set whatever the
    # chronological cut says. For blocks an operator already treats as external
    # validation, a blind "last 20%" is the wrong instrument - it might well put
    # them in the search set.
    always_held_out: dict[str, set[int]] = field(default_factory=dict)

    def forced_for(self, channel: str) -> set[int]:
        return set(self.always_held_out.get(channel, set()))


@dataclass
class ChannelSplit:
    """One channel's blocks, divided.

    `n_blocks` is what extraction produced for this channel, before any cut. It
    is carried here so a caller can check the thing the v1.6 fix is about: the
    two sides always add back up to it, whatever fraction was asked for.
    """

    channel: str
    search: pd.DataFrame
    held_out: pd.DataFrame
    forced_ids: tuple[int, ...] = ()
    note: str = ""
    # Every block extraction produced for this channel.
    n_blocks: int = 0
    # What the operator asked for, and where the cut actually landed. They
    # differ only when the requested fraction would have left fewer than
    # `MIN_HELD_OUT_BLOCKS` on the held-out side.
    requested_fraction: float = DEFAULT_SEARCH_FRACTION
    effective_fraction: float = DEFAULT_SEARCH_FRACTION
    clamped: bool = False

    @property
    def search_ids(self) -> tuple[int, ...]:
        return tuple(int(b) for b in self.search.get("block_id", pd.Series(dtype=int)))

    @property
    def held_out_ids(self) -> tuple[int, ...]:
        return tuple(
            int(b) for b in self.held_out.get("block_id", pd.Series(dtype=int))
        )

    @property
    def is_partition(self) -> bool:
        """Whether the two sides account for every block exactly once.

        The property the split-then-extract ordering exists to guarantee: the
        boundary falls between blocks, so nothing is duplicated and nothing is
        lost. Asserted in the tests and in the shipped self-test rather than
        assumed.
        """
        search, held = list(self.search_ids), list(self.held_out_ids)
        both = search + held
        return len(set(both)) == len(both) == self.n_blocks


def split_anchors(
    anchors: pd.DataFrame, channel: str, spec: SplitSpec | None = None
) -> ChannelSplit:
    """Divide one channel's already-extracted blocks chronologically.

    Takes the block list extraction produced and cuts it by index. It never
    touches the raw series and never re-runs extraction, so the number of blocks
    is fixed before this function is called and cannot depend on the fraction.

    The forced ids come out first, then the chronological cut is applied to
    what remains - so marking a block as external validation removes it from
    the search whether or not the time cut would have.

    The cut is clamped so at least `MIN_HELD_OUT_BLOCKS` always remain held out.
    Without that, raising the fraction past the point where one block is left
    over made `train_auto_selection` drop the channel entirely, and the report's
    block count fell as the search set grew - the v1.6 bug. Clamping moves the
    boundary instead of discarding the channel, and says so in the note.
    """
    spec = spec or SplitSpec()
    fraction = float(spec.search_fraction)
    if anchors.empty:
        return ChannelSplit(
            channel,
            anchors,
            anchors,
            note="No anchors.",
            n_blocks=0,
            requested_fraction=fraction,
            effective_fraction=fraction,
        )

    ordered = anchors.copy()
    if "start_time" in ordered.columns:
        ordered = ordered.sort_values("start_time")
    elif "block_id" in ordered.columns:
        ordered = ordered.sort_values("block_id")
    ordered = ordered.reset_index(drop=True)

    n_blocks = len(ordered)
    forced = spec.forced_for(channel)
    is_forced = ordered["block_id"].isin(forced) if "block_id" in ordered else False
    candidates = ordered[~is_forced] if forced else ordered
    n_forced = n_blocks - len(candidates)

    n = len(candidates)
    n_search = int(np.floor(n * fraction))
    n_search = max(0, min(n_search, n))

    # Reserve the held-out minimum. Held-out ends up with (n - n_search) blocks
    # from the cut plus every forced one, so the search may take at most
    # n + n_forced - MIN_HELD_OUT_BLOCKS.
    ceiling = max(0, n + n_forced - MIN_HELD_OUT_BLOCKS)
    clamped = n_search > ceiling
    n_search = min(n_search, ceiling)

    search = candidates.iloc[:n_search]
    held_out = pd.concat(
        [candidates.iloc[n_search:], ordered[is_forced]] if forced else [candidates.iloc[n_search:]],
        ignore_index=True,
    )
    if "start_time" in held_out.columns and not held_out.empty:
        held_out = held_out.sort_values("start_time").reset_index(drop=True)

    effective = (n_search / n) if n else fraction
    note = (
        f"{len(search)} block(s) for the search, {len(held_out)} held out "
        f"({fraction:.0%} chronological cut"
        + (f", {len(forced)} forced" if forced else "")
        + ")."
    )
    if clamped:
        note += (
            f" The cut was moved back to {effective:.0%}: {fraction:.0%} would "
            f"have left fewer than {MIN_HELD_OUT_BLOCKS} blocks held out, and "
            "moving the boundary keeps the channel trainable rather than "
            "dropping it."
        )
    return ChannelSplit(
        channel=channel,
        search=search.reset_index(drop=True),
        held_out=held_out,
        forced_ids=tuple(sorted(forced)),
        note=note,
        n_blocks=n_blocks,
        requested_fraction=fraction,
        effective_fraction=effective,
        clamped=clamped,
    )


def held_out_rmse(
    fit: calib.ChannelCalibration,
    held_out: pd.DataFrame,
    channel: str,
) -> tuple[float, int]:
    """Weighted RMS error of a fitted correction on blocks it never saw.

    Deliberately the same reconstruction and the same weighting the LOO score
    uses, so the two numbers are comparable: both are inverse-variance weighted
    RMS errors on the measurement, in the channel's own units. The only
    difference between them is which blocks they were computed on, which is the
    difference the report exists to show.
    """
    if held_out.empty:
        return float("nan"), 0

    ok, _ = calib.usable_mask(held_out, channel, fit.terms, fit.form)
    used = held_out[ok]
    if used.empty:
        return float("nan"), 0

    actual = pd.to_numeric(used[f"actual_{channel}"], errors="coerce").to_numpy(float)
    actual_se = pd.to_numeric(
        used[f"actual_{channel}_se"], errors="coerce"
    ).to_numpy(float)
    actual_se = np.where(np.isfinite(actual_se), actual_se, 0.0)

    errors, weights = [], []
    for position, (_index, row) in enumerate(used.iterrows()):
        try:
            predicted = calib._predict_value(fit, row)
        except calib.CalibrationError:
            continue
        error = float(actual[position]) - predicted
        if not np.isfinite(error):
            continue
        # Inverse variance on the measurement, floored the same way the fit
        # floors its own weights so one perfectly-repeatable block cannot take
        # the whole score.
        floor = max(fit.se_floor, 1.0e-9)
        weight = 1.0 / (actual_se[position] ** 2 + floor**2)
        errors.append(error)
        weights.append(weight)

    if not errors:
        return float("nan"), 0
    errors_array = np.asarray(errors, dtype=float)
    weights_array = np.asarray(weights, dtype=float)
    score = float(
        np.sqrt((weights_array * errors_array**2).sum() / weights_array.sum())
    )
    return score, len(errors)


@dataclass
class ChannelTraining:
    """What the search chose for one channel, and how it did on held-out data."""

    channel: str
    unit: str
    split: ChannelSplit
    search: calib.AutoSearch | None
    fit: calib.ChannelCalibration | None
    loo_rmse: float = float("nan")
    held_out_rmse: float = float("nan")
    held_out_n: int = 0
    note: str = ""

    @property
    def variable(self) -> str:
        return self.search.chosen_variable if self.search else calib.ENGINEERED_PAIR_KEY

    @property
    def orders(self) -> dict[str, int]:
        return self.search.chosen_orders if self.search else {}

    @property
    def form(self) -> str:
        return self.search.chosen_form if self.search else calib.FORM_ADDITIVE

    @property
    def degradation(self) -> float:
        """Held-out error as a multiple of the searched error.

        Above 1 means the winner did worse on blocks it had not seen, which is
        the expected direction - the question is how much worse.
        """
        if not np.isfinite(self.loo_rmse) or self.loo_rmse <= 0:
            return float("nan")
        return self.held_out_rmse / self.loo_rmse

    # Below this many held-out blocks the score is a statement about two or
    # three blocks, and reading it as a verdict on the selection would repeat
    # the mistake the split exists to avoid.
    THIN_HELD_OUT = 4

    @property
    def verdict(self) -> str:
        ratio = self.degradation
        if not np.isfinite(ratio):
            return "no held-out score"
        if self.held_out_n < self.THIN_HELD_OUT:
            direction = "better" if ratio < 1 else f"{ratio:.1f}x worse"
            return (
                f"{direction}, but on only {self.held_out_n} block(s) - too few "
                "to conclude much either way"
            )
        if ratio <= 1.25:
            return "holds up"
        if ratio <= 2.0:
            return "somewhat worse"
        return "much worse - treat the selection as fitted to this run"


@dataclass
class TrainingReport:
    """One dev-training run across every channel."""

    results: dict[str, ChannelTraining]
    spec: SplitSpec
    notes: list[str] = field(default_factory=list)
    # {channel: blocks extraction produced}, for every channel that had an
    # anchor table - including channels the split then declined to train. This
    # is the number that must not move when the fraction does, so it is recorded
    # separately from anything the split touched.
    block_counts: dict[str, int] = field(default_factory=dict)

    @property
    def total_blocks(self) -> int:
        """Blocks extraction produced across every channel, split or not."""
        return sum(self.block_counts.values())

    def table(self) -> pd.DataFrame:
        """The two numbers side by side, labelled so they cannot be confused."""
        rows = []
        for channel, result in self.results.items():
            rows.append(
                {
                    "channel": channel,
                    "unit": result.unit,
                    "selected variable": calib.variable_label(result.variable),
                    "shape": calib.orders_label(result.orders) or "-",
                    "form": result.form,
                    # Extraction's own count, first, so a moved boundary is
                    # visibly a moved boundary and not a smaller extraction.
                    "blocks extracted": result.split.n_blocks,
                    "search blocks": len(result.split.search),
                    "held-out blocks": result.held_out_n,
                    "LOO-CV (from the search)": result.loo_rmse,
                    "held-out (never seen during search)": result.held_out_rmse,
                    "held-out / LOO-CV": result.degradation,
                    "verdict": result.verdict,
                }
            )
        return pd.DataFrame(rows)

    def adopted_variables(self) -> dict[str, str]:
        return {c: r.variable for c, r in self.results.items() if r.search}

    def adopted_orders(self) -> dict[str, dict[str, int]]:
        return {c: dict(r.orders) for c, r in self.results.items() if r.search}

    def adopted_forms(self) -> dict[str, str]:
        return {c: r.form for c, r in self.results.items() if r.search}

    def describe(self) -> str:
        if not self.results:
            return "No channel produced a trainable split."
        parts = []
        for channel, result in self.results.items():
            parts.append(
                f"{channel}: {calib.variable_label(result.variable)} "
                f"({calib.orders_label(result.orders) or '-'}), "
                f"LOO {calib.format_number(result.loo_rmse)} vs held-out "
                f"{calib.format_number(result.held_out_rmse)} {result.unit}"
            )
        return " | ".join(parts)


def train_auto_selection(
    anchor_tables: dict[str, pd.DataFrame],
    spec: SplitSpec | None = None,
    alpha: float = calib.DEFAULT_ALPHA,
    se_floor_fraction: float = calib.DEFAULT_SE_FLOOR_FRACTION,
    min_anchors: int = calib.MIN_ANCHORS_FOR_QUADRATIC,
    tolerance: float = calib.AUTO_ORDER_TOLERANCE,
) -> TrainingReport:
    """Search on part of the anchors, score the winner on the rest.

    `anchor_tables` is the *whole* extracted block list per channel. Extraction
    has already happened and is not re-run here, so nothing in this function can
    change how many blocks a channel has - only which side of the cut they fall.

    The search never receives the held-out frame - not as a scoring set, not as
    a tie-break, not at all. `ChannelSplit` carries both sides so the separation
    is inspectable rather than a promise in a docstring, and the tests assert
    the fitted block ids are a subset of the search ids.
    """
    spec = spec or SplitSpec()
    results: dict[str, ChannelTraining] = {}
    notes: list[str] = []
    block_counts: dict[str, int] = {}

    # Whichever channels the supplied anchor tables actually carry, in display
    # order. Iterating a module-level channel list would silently train only
    # the non-nested four on a nested dataset.
    for channel in calib.ordered_channels(anchor_tables):
        table = anchor_tables.get(channel)
        if table is None or table.empty:
            notes.append(f"{channel}: no anchors to train on.")
            continue

        block_counts[channel] = len(table)
        split = split_anchors(table, channel, spec)
        # A channel is refused only when its block list cannot seat both
        # minimums at *any* fraction. That makes the refusal a property of the
        # extraction rather than of the number in the spin box, which is what
        # stops a higher search fraction quietly costing a channel.
        if len(table) < MIN_SEARCH_BLOCKS + MIN_HELD_OUT_BLOCKS:
            notes.append(
                f"{channel}: {len(table)} extracted block(s) cannot seat both a "
                f"{MIN_SEARCH_BLOCKS}-block search set and a "
                f"{MIN_HELD_OUT_BLOCKS}-block held-out set at any split. "
                "Skipped - extract more blocks rather than moving the cut."
            )
            continue
        if len(split.search) < MIN_SEARCH_BLOCKS:
            notes.append(
                f"{channel}: only {len(split.search)} block(s) would remain for "
                f"the search at {split.requested_fraction:.0%}, below the "
                f"{MIN_SEARCH_BLOCKS} needed. Skipped - a selection made on "
                "fewer than that says more about the split than about the "
                "process. Raise the search fraction."
            )
            continue
        if split.clamped:
            notes.append(f"{channel}: {split.note}")

        # The search sees `split.search` and nothing else.
        search = calib.search_auto(
            split.search,
            channel,
            alpha=alpha,
            se_floor_fraction=se_floor_fraction,
            min_anchors=min_anchors,
            tolerance=tolerance,
        )
        if search.chosen is None:
            notes.append(f"{channel}: no candidate was usable on the search set.")
            continue

        try:
            fit = calib.fit_channel(
                split.search,
                channel,
                terms=search.chosen_terms,
                alpha=alpha,
                se_floor_fraction=se_floor_fraction,
                orders=search.chosen_orders,
                form=search.chosen_form,
            )
        except calib.CalibrationError as exc:
            notes.append(f"{channel}: the winning candidate would not refit ({exc}).")
            continue

        score, n_scored = held_out_rmse(fit, split.held_out, channel)
        results[channel] = ChannelTraining(
            channel=channel,
            unit=schema.unit_of(channel),
            split=split,
            search=search,
            fit=fit,
            loo_rmse=search.chosen.loo_rmse,
            held_out_rmse=score,
            held_out_n=n_scored,
            note=search.note,
        )

    return TrainingReport(
        results=results, spec=spec, notes=notes, block_counts=block_counts
    )


# --------------------------------------------------------------------------
# Refitting the winner on everything (v1.6)
# --------------------------------------------------------------------------
#
# The split exists to choose a configuration honestly, and it costs data to do
# it: the winning coefficients come off the search set alone, which on the
# reference run is seven to fourteen blocks out of nine to sixteen. Once the
# configuration has been chosen and scored, the held-out blocks have discharged
# their purpose, and there is no longer a reason for the *coefficients* to
# ignore them.
#
# So this refits the same (variable, shape, form) on 100% of the blocks. What it
# emphatically does not do is re-run the search. Re-searching on the full set
# would let the held-out blocks influence the selection, which is the one thing
# the split was run to prevent, and the held-out score printed above it would
# silently stop describing the configuration in hand.
#
# Nor is more data assumed to be better. A refit on more blocks can be worse in
# every way that matters - a coefficient that was significant on the search set
# can cross zero once the last blocks are in, and that is a finding about the
# selection, not a blemish to hide. Hence `FullDataRefit` carries its own RMS,
# its own degrees of freedom and its own parameter significance, and the UI
# shows them before anything is adopted.


@dataclass
class ChannelRefit:
    """One channel's winning configuration, refitted on every block."""

    channel: str
    unit: str
    variable: str
    orders: dict[str, int]
    form: str
    # The search-set fit the held-out score was computed for, and the same
    # configuration refitted on everything. Kept side by side because the
    # comparison is the point.
    search_fit: calib.ChannelCalibration
    full_fit: calib.ChannelCalibration

    @property
    def n_search(self) -> int:
        return self.search_fit.n_anchor

    @property
    def n_full(self) -> int:
        return self.full_fit.n_anchor

    @property
    def n_added(self) -> int:
        return self.n_full - self.n_search

    @property
    def newly_insignificant(self) -> tuple[str, ...]:
        """Coefficients that were clear of zero before the refit and are not now.

        The specific way a full-data refit can be worse. Reported rather than
        buried: a term that loses significance once the held-out blocks are
        included was being held up by the blocks the search happened to see.
        """
        before = {
            e.key: e
            for e in self.search_fit.estimates
            if e.key != calib.TERM_CONST.key
        }
        out = []
        for estimate in self.full_fit.estimates:
            if estimate.key == calib.TERM_CONST.key:
                continue
            was = before.get(estimate.key)
            if was is not None and not was.spans_zero and estimate.spans_zero:
                out.append(estimate.label)
        return tuple(out)

    @property
    def rms_change(self) -> float:
        """Full-data RMS as a multiple of the search-set RMS."""
        if not np.isfinite(self.search_fit.residual_rms) or (
            self.search_fit.residual_rms <= 0
        ):
            return float("nan")
        return self.full_fit.residual_rms / self.search_fit.residual_rms


@dataclass
class FullDataRefit:
    """Every channel's configuration refitted on the complete block list."""

    results: dict[str, ChannelRefit]
    notes: list[str] = field(default_factory=list)

    def table(self) -> pd.DataFrame:
        rows = []
        for channel, refit in self.results.items():
            rows.append(
                {
                    "channel": channel,
                    "unit": refit.unit,
                    "variable": calib.variable_label(refit.variable),
                    "shape": calib.orders_label(refit.orders) or "-",
                    "form": refit.form,
                    "search blocks": refit.n_search,
                    "all blocks": refit.n_full,
                    "added": refit.n_added,
                    "RMS (search-set fit)": refit.search_fit.residual_rms,
                    "RMS (full-data fit)": refit.full_fit.residual_rms,
                    "full / search RMS": refit.rms_change,
                    "dof": refit.full_fit.dof,
                    "terms spanning zero": sum(
                        1
                        for e in refit.full_fit.estimates
                        if e.key != calib.TERM_CONST.key and e.spans_zero
                    ),
                    "newly not significant": ", ".join(refit.newly_insignificant)
                    or "-",
                }
            )
        return pd.DataFrame(rows)

    def parameter_table(self) -> pd.DataFrame:
        """Every refitted coefficient with its interval, for inspection."""
        rows = []
        for channel, refit in self.results.items():
            for estimate in refit.full_fit.estimates:
                rows.append(
                    {
                        "channel": channel,
                        "term": estimate.label,
                        "estimate": estimate.value,
                        "std error": estimate.se,
                        "CI lo": estimate.ci_lo,
                        "CI hi": estimate.ci_hi,
                        "spans zero": "yes" if estimate.spans_zero else "no",
                    }
                )
        return pd.DataFrame(rows)

    def describe(self) -> str:
        if not self.results:
            return "No channel could be refitted on the full block list."
        return " | ".join(
            f"{channel}: {refit.n_search} -> {refit.n_full} blocks, RMS "
            f"{calib.format_number(refit.search_fit.residual_rms)} -> "
            f"{calib.format_number(refit.full_fit.residual_rms)} {refit.unit}"
            for channel, refit in self.results.items()
        )


def refit_on_full_data(
    report: TrainingReport,
    anchor_tables: dict[str, pd.DataFrame],
    alpha: float = calib.DEFAULT_ALPHA,
    se_floor_fraction: float = calib.DEFAULT_SE_FLOOR_FRACTION,
) -> FullDataRefit:
    """Refit each channel's chosen configuration on every available block.

    Coefficients only. The variable, the shape and the form are taken verbatim
    from what the split-validated search selected; nothing is re-ranked, and no
    candidate that lost on the search set gets a second hearing now that more
    data is available. `anchor_tables` is the same full-dataset mapping
    `train_auto_selection` was given, so "every available block" includes the
    held-out set and any block the split declined to use.
    """
    results: dict[str, ChannelRefit] = {}
    notes: list[str] = []

    for channel, result in report.results.items():
        if result.search is None or result.fit is None:
            continue
        table = anchor_tables.get(channel)
        if table is None or table.empty:
            notes.append(f"{channel}: no full anchor table to refit against.")
            continue
        try:
            full_fit = calib.fit_channel(
                table,
                channel,
                terms=result.search.chosen_terms,
                alpha=alpha,
                se_floor_fraction=se_floor_fraction,
                orders=result.search.chosen_orders,
                form=result.search.chosen_form,
            )
        except calib.CalibrationError as exc:
            notes.append(
                f"{channel}: the chosen configuration would not refit on the "
                f"full block list ({exc}). The search-set fit is unaffected."
            )
            continue
        results[channel] = ChannelRefit(
            channel=channel,
            unit=result.unit,
            variable=result.variable,
            orders=dict(result.orders),
            form=result.form,
            search_fit=result.fit,
            full_fit=full_fit,
        )

    for channel, refit in results.items():
        if refit.newly_insignificant:
            notes.append(
                f"{channel}: {', '.join(refit.newly_insignificant)} was clear of "
                "zero on the search set and is not once every block is "
                "included. The extra data did not confirm that term."
            )
        if np.isfinite(refit.rms_change) and refit.rms_change > 1.5:
            notes.append(
                f"{channel}: RMS residual is {refit.rms_change:.2f}x the "
                "search-set figure. More blocks is not automatically a better "
                "fit - the held-out blocks may sit somewhere the chosen shape "
                "does not describe."
            )
    return FullDataRefit(results=results, notes=notes)
