"""Where a channel's current coefficients came from, in one line.

Why this exists
---------------
By v1.6 there were three ways for a channel to end up with the shape it has -
the operator picked it, Auto picked it, or a dev-training run picked it and
someone pressed Adopt - and the screen looked identical in all three cases. The
shape was visible; the reason for it was not. An operator returning to the app
could read what was fitted and had no way to tell whether it had been chosen
deliberately, chosen by cross-validation on this run, or carried over from a
held-out-validated search two days ago.

That is a legibility problem, not a modelling one, and this module fixes only
that. Nothing here participates in a fit. It reads state the app already has and
turns it into a sentence.

The honest bit
--------------
The Fit & Inspect tab refits against its own anchor set whenever anything
changes, and it does that regardless of which Adopt button was pressed in the
dev-training dialog. So for an adopted channel, the *selection* came from the
split-validated search but the *coefficients on screen* were fitted here, on
every anchor this tab holds - which is not the fit the held-out score was
computed for.

Rather than print "Dev-trained" and let that be read as "these numbers were
validated against held-out data", `ChannelProvenance.line` says both halves: what
chose the shape, and what fitted the numbers. A label that is subtly untrue is
worse than no label, because it is the one an operator would rely on.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

import calibration as calib

# What last determined a channel's configuration.
SOURCE_MANUAL = "Manual"
SOURCE_AUTO = "Auto"
SOURCE_DEV = "Dev-trained"

# Which of the dev dialog's two Adopt actions was used. Canonical here rather
# than in the dialog, so the dialog and the tab cannot drift apart on the
# spelling of a value they pass between them.
SCOPE_SEARCH_SET = "search-set fit"
SCOPE_FULL_DATA = "full-data refit"


def _shape_label(orders: Mapping[str, int]) -> str:
    """The shape, short enough to sit in a one-line summary.

    `calib.orders_label` names every term ("Capillary wall ratio quadratic"),
    which is right for a report and too long here where the variable is already
    named alongside. This gives just the orders.
    """
    if not orders:
        return "offset only"
    return ", ".join(calib.order_label(int(order)) for order in orders.values())


def auto_reason(scan: calib.OrderScan | None) -> str:
    """Why Auto landed where it did for this channel, right now.

    Derived from the scan rather than parsed out of `auto_explanation`, so the
    numbers in the sentence are the numbers the decision was actually made on.
    Returns a clause, not a sentence - the caller supplies the shape it explains.

    The two cases are the two branches of the rule:

      * the anchor guardrail fired, so the tolerance rule was never consulted
      * the tolerance rule ran, and either the pick was outright best or it was
        the simpler candidate that the best one failed to beat by enough
    """
    if scan is None:
        return ""
    if scan.guardrail_applied:
        return (
            f"the anchor count of {scan.n_anchor} is below the quadratic floor "
            f"of {calib.MIN_ANCHORS_FOR_QUADRATIC}, so the cross-validated "
            "ranking was not consulted"
        )

    usable = [
        c for c in scan.candidates if c.feasible and np.isfinite(c.loo_rmse)
    ]
    if not usable:
        return "no candidate shape was usable, so linear was assumed"

    best = min(usable, key=lambda c: c.loo_rmse)
    pick = scan.candidate_for(scan.auto_orders)
    if pick is None or pick.orders == best.orders:
        reason = (
            f"it had the lowest cross-validated error of the {len(usable)} "
            "shape(s) tried"
        )
        # "and nothing simpler came within the tolerance" is only worth saying
        # when something simpler exists. For the simplest candidate on the list
        # it is vacuously true, and reads as though a comparison happened.
        simpler = [c for c in usable if c.n_parameters < (pick or best).n_parameters]
        if simpler:
            reason += (
                f", and nothing simpler came within {scan.auto_tolerance:.0%} of it"
            )
        return reason
    if not np.isfinite(pick.loo_rmse) or pick.loo_rmse <= 0:
        return "cross-validated error was not usable, so the simpler shape was kept"
    margin = (pick.loo_rmse - best.loo_rmse) / pick.loo_rmse * 100.0
    return (
        f"{best.label}'s cross-validated error was only {margin:.0f}% better, "
        f"within the {scan.auto_tolerance:.0%} tolerance for preferring the "
        "simpler shape"
    )


@dataclass(frozen=True)
class Adoption:
    """One dev-training adoption, recorded as it happened.

    Carries the configuration it adopted so the tab can tell, later, whether the
    operator has since overridden it. An adoption whose configuration no longer
    matches the dropdowns is stale and must stop claiming credit for them.
    """

    channel: str
    when: dt.datetime
    scope: str
    variable: str
    form: str
    orders: dict[str, int] = field(default_factory=dict)
    # Blocks behind the coefficients that were reviewed in the dialog: the
    # search set for a search-set fit, every block for a full-data refit.
    n_reviewed: int = 0
    n_search: int = 0
    n_held_out: int = 0

    @property
    def date_label(self) -> str:
        return self.when.strftime("%d/%m")

    def matches(
        self, variable: str | None, form: str | None, orders: Mapping[str, int] | None
    ) -> bool:
        if variable != self.variable or form != self.form:
            return False
        return {k: int(v) for k, v in (orders or {}).items()} == {
            k: int(v) for k, v in self.orders.items()
        }


@dataclass(frozen=True)
class ChannelProvenance:
    """What produced this channel's configuration, and what fitted its numbers."""

    channel: str
    source: str
    variable: str
    orders: dict[str, int] = field(default_factory=dict)
    form: str = calib.FORM_ADDITIVE
    n_anchor: int = 0
    auto_reason: str = ""
    adoption: Adoption | None = None
    # Set when the coefficients on screen were fitted somewhere other than where
    # the configuration was chosen. See the module docstring.
    refit_note: str = ""

    @property
    def source_label(self) -> str:
        """The Source field, including the adoption detail when there is any."""
        if self.source != SOURCE_DEV or self.adoption is None:
            return self.source
        adoption = self.adoption
        return (
            f"{SOURCE_DEV} (adopted {adoption.date_label}, "
            f"n={adoption.n_reviewed}, {adoption.scope})"
        )

    @property
    def shape_label(self) -> str:
        return _shape_label(self.orders)

    def line(self) -> str:
        """The always-visible provenance line for this channel.

        One shape throughout, so the channels stack into something
        scannable rather than four differently-worded paragraphs:

            Source: <who> - <config>, on <n> anchors from this run. <why>.

        The first clause answers "who chose this?" and the reader can stop
        there; the sentence after it answers "on what grounds?" for whoever
        does not stop. The shape is named once, in the config, rather than
        again in the justification.
        """
        config = (
            f"{calib.variable_label(self.variable)}, {self.shape_label}, "
            f"{'ratio' if self.form == calib.FORM_RATIO else 'additive'}"
        )
        body = (
            f"Source: {self.source_label} - {config}, on {self.n_anchor} "
            f"anchor(s) from this run."
        )

        if self.source == SOURCE_AUTO:
            reason = self.auto_reason
            if reason:
                shape = self.shape_label
                body += f" {shape[:1].upper()}{shape[1:]} was chosen because {reason}."
        elif self.source == SOURCE_MANUAL:
            body += " Chosen by hand; not checked against held-out data."

        if self.refit_note:
            body += " " + self.refit_note
        return body


def for_channel(
    channel: str,
    variable: str,
    orders: Mapping[str, int],
    form: str,
    n_anchor: int,
    uses_auto: bool,
    scan: calib.OrderScan | None = None,
    adoption: Adoption | None = None,
) -> ChannelProvenance:
    """Assemble one channel's provenance from the state the tab already holds.

    Precedence is deliberate. An adoption outranks Auto because a channel that
    was adopted has concrete orders rather than the Auto sentinel, and the fact
    that a held-out search chose them is the more informative statement. Auto
    outranks manual because the sentinel genuinely is what is in the box.
    """
    resolved = {key: int(value) for key, value in (orders or {}).items()}

    if adoption is not None:
        return ChannelProvenance(
            channel=channel,
            source=SOURCE_DEV,
            variable=variable,
            orders=resolved,
            form=form,
            n_anchor=n_anchor,
            adoption=adoption,
            refit_note=(
                "The search chose the shape; the coefficients above were "
                f"refitted here on all {n_anchor} of this tab's anchors, so "
                "they are not the fit the held-out score was computed for."
            ),
        )

    if uses_auto:
        return ChannelProvenance(
            channel=channel,
            source=SOURCE_AUTO,
            variable=variable,
            orders=resolved,
            form=form,
            n_anchor=n_anchor,
            auto_reason=auto_reason(scan),
        )

    return ChannelProvenance(
        channel=channel,
        source=SOURCE_MANUAL,
        variable=variable,
        orders=resolved,
        form=form,
        n_anchor=n_anchor,
    )


def search_line(
    channel: str,
    variable: str,
    orders: Mapping[str, int],
    form: str,
    n_search: int,
    n_held_out: int,
    verdict: str,
    adopted_scope: str = "",
) -> str:
    """The dev dialog's equivalent line: what the search chose, and on what.

    Deliberately says "selected on ... scored on ..." rather than "Source:".
    Nothing in this dialog is in use anywhere until Adopt is pressed, and a line
    that read like the tab's would imply otherwise.
    """
    config = (
        f"{calib.variable_label(variable)}, {_shape_label(orders)}, "
        f"{'ratio' if form == calib.FORM_RATIO else 'additive'}"
    )
    line = (
        f"{channel}: search selected {config} - chosen on {n_search} search "
        f"block(s), scored on {n_held_out} held-out block(s) ({verdict})."
    )
    if adopted_scope:
        line += f" Adopted as the {adopted_scope}."
    else:
        line += " Not adopted - nothing is in use until an Adopt button is pressed."
    return line
