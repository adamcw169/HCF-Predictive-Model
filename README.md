# HCF Anchor Predictor (v1.10 — tubular / NANF / DNANF)

A native Windows desktop app that turns a raw hollow-core fiber draw run into a
handful of steady-state **anchor points**, fits a small physically-motivated
correction between the fast estimator's analytic prediction and what the tower
actually did, and applies that correction to a new target geometry.

Fully offline. No network access anywhere, no server, no browser.

This is a **standalone project**. It shares no code with the HCF Draw Predictor
in the parent folder - that app is a design reference and a source of
conventions only. The two do share a data schema, deliberately: a CSV written
for one is readable by the other.

---

## Why this is a different model, not a refinement of the other one

A real steady-state extraction from one draw run yields about **eight**
independent points. The 3,465-row dataset the other app trains on is one run
sampled at 1 Hz - not 3,465 experiments. Consecutive rows one second apart are
not independent observations of anything.

At n≈8 across six input dimensions, a RandomForest or a Gaussian process can
interpolate the eight points it has and nothing else. It cannot identify real
structure, it cannot be checked against its own uncertainty in any meaningful
way, and it cannot transfer to a new preform geometry, because everything it
knows is memorised from this one.

What *can* be identified from eight points is two to four parameters, expressed
in dimensionless terms, weighted by how well each point is measured. That is
what this app fits. Every coefficient comes with a confidence interval, and the
app says plainly when a coefficient is not distinguishable from zero - which,
on the current single run, is true of the draw-down term for every channel.

## What changed in v1.10 (tubular / NANF / DNANF)

**What is fitted did not change.** Absolute pressures remain the native fitted
target for every capillary layer, as they have been for nine releases. The
ratio/additive branching, the order-selection tolerance, the anchor guardrails
and the dev-training held-out framework were all built and validated against
absolutes; switching the fitted target to differentials would invalidate that
validation to gain arithmetic a prediction can do afterwards anyway. Everything
below is either *ingestion* (turning what the tower logs into what the model
has always been fitted on) or *display*.

**Three named geometries.** The registry is now tubular / NANF / DNANF:

| Preform | Chain (fitted, absolute) | Capillary geometry |
| --- | --- | --- |
| Tubular | Pcore, Pocap | outer only |
| NANF | Pcore, Pocap, Picap | outer, inner - no middle |
| DNANF | Pcore, Pocap, Pmcap, Picap | outer, middle, inner |

Tubular is the former `hc_10cap_nonnested` and DNANF the former
`hc_nested_3layer`, renamed with no change to schema or fitted behaviour. NANF
is new and is **not** DNANF with a layer blanked out: it has no middle geometry
at all, so its inner differential is measured against the *outer* layer and its
chain has two links rather than three.

**The id is the storage key, so the rename carries its history.** Calibrations
live in `models/<id>/`. Each entry declares the ids it used to be known by;
`get_preform`, `schema_for_preform` and the extraction-profile lookup all
accept them, and `paths.calibration_path` falls back to the old folder when
nothing has been saved under the new one. No file is migrated, rewritten or
re-fitted - a calibration saved before this release simply still loads. New
saves go to the current id.

**Ingestion: absolutes are chained onto Pcore.** Per the supervisor, only
`core_dP_kPa` is a true absolute measurement; every capillary layer's raw
channel is a sequential differential against the layer outside it. The single
`Pocap = core_dP_kPa + outer_dP_kPa` sum generalises to a chain of arbitrary
length, selected by the active preform:

```
Pocap = core_dP_kPa + outer_dP_kPa     all three
Pmcap = Pocap       + mid_dP_kPa       DNANF only
Picap = Pmcap       + inner_dP_kPa     DNANF
Picap = Pocap       + inner_dP_kPa     NANF (skips the middle)
```

The chain is walked in order, and a broken link stops the layers inside it and
says so rather than computing a later layer against a base that was never
established. For tubular this reduces to exactly the sum it has always been -
asserted against the old `add_pocap` on the real reference file.

**Differentials are now the primary Predict output.** They lead the results
panel, because the step across each capillary wall is what a fabricator
actually sets at the tower. The absolutes stay directly beneath as reference:
they are what the model fits, they carry the intervals, and a differential is
only as trustworthy as the two predictions it is a difference of. Negatives are
still reported rather than clamped, and no interval is quoted on a delta - the
operands are correlated predictions sharing an anchor set.

**Analytic sample export.** Tab 1 gains "Export analytic input spreadsheet",
after extraction and exclusion, writing one row per *surviving* block from that
block's medians. `time_utc` is the block's **start time** (what the confirmed
example carries); `sample` is the block id, so the estimator's returned rows
map back. The 13-column tubular layout is **confirmed** - taken from a file the
estimator has already accepted, and pinned character-for-character in
`tests/test_analytic_export.py`.

> **The multi-layer columns are NOT confirmed.** No working NANF/DNANF example
> exists yet, so `cap_OD_middle_um`/`cap_ID_middle_um`/`mid_dP_kPa` and the
> `inner` equivalents are this app's best guess at the naming, chosen to match
> the confirmed single-layer format. Check them against the estimator's actual
> multi-layer requirements before relying on the output. The export itself
> warns the operator, and `analytic_export.NANF_DNANF_FORMAT_IS_UNCONFIRMED`
> is the flag to flip once a real example exists.

A geometry only gets the layer columns it actually has - NANF's export carries
inner columns and no middle ones, because an empty column invites the estimator
to read a blank as a measurement.

**Proof the tubular preform did not move.** The same mechanical bar as the
nested release: a full-precision capture of block boundaries, every fitted
coefficient and covariance entry, LOO-CV scores and the resulting prediction,
taken before the change and again after. Every fitted number is identical, and
`test_the_tubular_fitted_numbers_are_unmoved_by_this_release` pins them as
literals at 1e-12. One prose string changed deliberately - the derivation note
now says "the sequential differential" rather than "only the differential",
because the chain generally has more than one link - and that is pinned by its
own test rather than left as an unexplained diff.

## What changed in v1.9

**A second real preform: nested, three capillary layers.** The registry entry
that had been a greyed-out "coming soon" placeholder since v1.0 is now a real
`PreformDefinition` with its own schema, alongside the existing non-nested
preform rather than replacing it.

Ten geometry features (the shared fiber/tension/feed four, plus outer, middle
and inner capillary OD/ID), six setpoints (`furnace_temp_C`,
`draw_speed_m_min`, `core_dP_kPa` plus `Pocap_kPa`, `Pmcap_kPa`, `Picap_kPa`),
and six analytic estimates entered by hand exactly as the existing four are -
the estimator is still not callable from this app. Whether the supervisor's
estimator produces the middle and inner analytic values is not yet known; if it
does not, manual entry remains the input path either way.

**It ships with zero anchors, and says so.** `is_implemented=True` means *the
app can work with this geometry*, not *this geometry is calibrated* - which is
what the flag has always meant. Selecting it lands on the same "no calibration
saved yet" state a first run shows for any preform, because that is exactly the
situation. No placeholder data was fabricated to make it look trained.

**The chained pressure convention.** Each delta is the step across one capillary
wall - the pressure inside it minus the pressure immediately outside:

```
deltaPocap = Pocap_kPa - core_dP_kPa      (the existing delta_P, generalised)
deltaPmcap = Pmcap_kPa - Pocap_kPa
deltaPicap = Picap_kPa - Pmcap_kPa
```

Derived, never fitted - the same status `delta_P` has had since v1.8, and no
confidence interval on any of them for the same reason: the operands are
correlated predictions sharing an anchor set, and an independent-error interval
would overstate the confidence. **Negatives are reported, not clamped or
hidden**: a step the wrong way round is either a real physical surprise or a bad
estimate, and both are things the operator needs to see - the same treatment
this app already gives a negative analytic pressure. The non-nested preform
still shows its single `delta_P`, unchanged in name, formula, sign and position.

**Wall ratios per layer.** `cap_wall_ratio` generalises to `outer_wall_ratio`,
`middle_wall_ratio` and `inner_wall_ratio` - the same `(OD - ID) / OD` on each
layer - all offered in the fit-against dropdown. Each capillary pressure opens
on *its own* layer's ratio, which is the nested form of the v1.3 suggested-
defaults reasoning: `Pmcap_kPa` suggests `middle_wall_ratio`, never another
layer's dimension, because pairing a pressure with a geometry it does not
inflate asserts a coupling the physics does not have. `cap_wall_ratio` keeps its
name and key - it is what every stored coefficient is indexed by.

**The hardcoded-4-channel audit.** Several stages assumed exactly four setpoints
and exactly one capillary pressure. Each now derives from the active preform:

| Where | Was | Now |
| --- | --- | --- |
| `steady_state` stability groups | one hardcoded dict of four | `EXTRACTION_PROFILES`, keyed by preform id |
| `steady_state` watched channels / thresholds | module constants | per-geometry profile |
| `calibration` channel iteration (×5 sites) | `schema.SETPOINT_NAMES` | `ordered_channels()`, from the tables present |
| `calibration` anchor building | hardcoded `cap_wall_ratio` | `PreformSchema.compute_wall_ratios` |
| `calibration` prediction features | hardcoded six raw inputs | the preform's own feature list |
| `calibration` suggested variables | one dict | per-geometry, via `suggested_variable()` |
| `dev_training` search loop | `schema.SETPOINT_NAMES` | `ordered_channels()` |
| Fit & Inspect per-channel UI | `schema.SETPOINT_NAMES` | the tab's active schema |
| Predict tab inputs | hardcoded `INPUT_FIELDS` | `input_fields_for(schema)` |
| Predict tab analytic + results | module constants | rebuilt from the selected preform |

One honest correction to the brief's expectation: the Predict tab's preform
selector did **not** already work generically. It only toggled an availability
note - the inputs, analytic fields and results grid all stayed on the
non-nested columns whatever was selected. Selecting a geometry now rebuilds all
three, which is the registry's original design intent finally implemented.

**Proof the existing preform did not move.** A full-precision capture of the
non-nested pipeline - block boundaries for both criteria, anchor counts, every
fitted coefficient and covariance entry, LOO-CV scores, the prediction and its
intervals, and the schema surface itself - taken before the change and again
after, is **byte-identical**. The `--selftest` output is likewise unchanged
except for the new nested lines. No stored calibration needs migrating and the
calibration format was not bumped: the schema behind a saved file resolves from
the `preform_id` it already carries, so a pre-v1.9 file loads and predicts
exactly as before.

## What changed in v1.8.2

**Extraction, fixed: segmentation no longer depends on how much of a run is
loaded.** Reported bug: loading a genuine time slice of a run (a file that
starts and ends mid-run, containing several real steady blocks) could produce
wildly different extraction results than loading the full run and looking at
the same time range - at one tested sensitivity, every per-mapping channel
except `furnace_temp_C` found zero blocks in a 38-minute slice that
demonstrably contains four real, multi-minute steady periods, and
`furnace_temp_C` merged all four into one oversized block.

The end-of-run shutdown ramp in the untrimmed test file was tested first and
ruled out: a trimmed copy with the ramp removed failed the same way.

**Diagnosis, verified against printed numbers rather than assumed.** The
percent criterion's segmentation (v1.1) converts its percent threshold into
each channel's own units via `level_tolerance()`, which multiplied by
`np.ptp()` - the peak-to-peak range - of *whatever frame happened to be
loaded*. That range is not a sensor property; it shrinks whenever fewer
distinct operating points are present in the loaded data, which is exactly
what a shorter slice does. Measured directly: `outer_dP_kPa`'s span drops
from 15.3 kPa (the full 111-minute run, which visits several pressure
levels) to 0.34 kPa (a slice covering only the run's last four, similar-
pressure blocks) - a 45x difference from which rows happened to be loaded,
not from anything about the instrument. The resulting absolute segmentation
tolerance shrank by the same 45x, past the point of ordinary sample-to-sample
sensor noise, so segmentation stopped distinguishing genuine steps from
jitter and started shredding real multi-minute blocks into fragments too
short to survive the minimum-block-duration filter.

A competing hypothesis - that the *reference* computation itself (a quantile
of percent-change, already more robust than a raw max) was the culprit and
that trying a different, more robust quantile would fix it - was tested
directly and found insufficient: no single quantile choice stabilised every
channel (the median made `core_dP_kPa`'s full-vs-slice ratio *worse*, by
10x, while helping others). The instability was specifically in
`level_tolerance`'s range-based unit conversion, not in the reference
percentile it started from.

A second, independent bug was found in the same code path while comparing it
against the (unaffected) absolute-criterion path: `segment_tolerance_factor`
was applied twice - once folded into the percent handed to `level_tolerance`,
once again inside `_split_on_level_changes` - squaring its effect (36x at
the shipped default of 6) instead of applying it once.

**The fix.** `level_tolerance()` is removed. Percent-mode segmentation
(`_split_on_percent_level_changes`) now measures a level shift the same way
the steadiness criterion already measures drift: as a percentage of the
channel's own *current* value (`percent_change`'s own formula, with the same
`channel_floor` near-zero backstop), rather than as a percentage of a range
snapshot taken once from however much data was loaded. Evaluated locally
against the tracked level instead of globally against the loaded frame, it
does not move when the amount of surrounding data does. The
double-application of `segment_tolerance_factor` is fixed at the same time,
applied exactly once. The absolute-criterion path (`_split_on_level_changes`,
fixed per-channel thresholds that never depended on the loaded frame) is
untouched.

Consequence, verified rather than assumed away: fixing the mechanism also
makes it more sensitive on the *reference run itself*, not only on slices of
it - the old tolerance was loose enough to blur real, distinct pressure and
geometry transitions into oversized blocks there too. Per-mapping block
counts on the reference run move from 9/10/16/13 to 14/16/17/13
(`furnace_temp_C`/`draw_speed_m_min`/`core_dP_kPa`/`Pocap_kPa`); the extra
blocks were checked block-by-block against their `core_dP_kPa`/
`outer_dP_kPa`/`fibre_ID_um` medians and each one sits at a distinct,
physically real level the old tolerance had been silently averaging
together. The validated 8-block **absolute**-criterion reproduction
(`steady_state_points.csv`) is unaffected - that path never calls
segmentation with a run-derived tolerance and does not exercise either bug.

**New permanent regression test**:
`test_slice_extraction_matches_full_run_filtered_to_the_same_window` -
extracts the full reference run, filters the resulting per-mapping block
list to a real slice's time window, extracts that slice directly, and
asserts the two match (block count and boundaries, within a small edge
tolerance at the slice's own start/end). This is the property the bug
report named directly: a time slice of a run should reproduce the same
blocks as the full run filtered to that range, because the reverse-order
testing methodology used throughout this project (train on one temporal
slice, validate against another) depends on it.

## What changed in v1.8

**No modelling change, again.** `--selftest` output is line-for-line identical
to v1.7's apart from the version string, the timestamp and the install path.
Extraction criteria, calibration fitting, LOO-CV, the dev-training search and
the prediction mechanism are all untouched; v1.8 changes how data reaches the
pipeline and adds one derived display value.

**Several experimental files, merged on time.** Acquisition splits the channels
across instruments - the capillary OD arrives from a different device, on its
own clock, at its own rate - so the file picker now takes several files at once.
Each is binned to 1 s by averaging within the bin, then joined on the bin
timestamp, so files whose raw samples never share a timestamp still land on the
same rows. The schema is checked against the **union** of the files' columns,
and the report distinguishes the two failures that used to look identical: a
column no file carried, versus a column that is simply in the other file. Two
kinds of absence are reported separately, because they differ in kind - a
missing steadiness criterion relaxes the extraction, while a missing anchor
column such as `cap_OD_um` (which is not a criterion at all) means every anchor
is short of something the calibration needs.

Per merged column the report gives the fraction of 1-second bins that came from
an average of several raw samples, from a single sample, and from none at all -
so a column that is mostly empty seconds reads as thin evidence rather than as
data.

**Clock offsets are correctable.** Each file after the first gets a manual
offset control, applied before binning. The app does not estimate it:
cross-correlating two instruments that measure different quantities would be
guessing, and a wrong guess misaligns every block silently.

**A column in two files with different values is a warning, not a silent pick.**
The first file's copy is kept and the disagreement is reported with the number
of seconds affected and the largest difference. Two instruments disagreeing
about the same quantity is a data problem, and resolving it automatically would
be inventing a judgement the data does not support.

**Binning does not launder a QC flag.** Averaging is right for a measurement and
wrong for a flag: `qc_outlier` averaged over four samples with one set gives
0.25, which `steady_state._bool_column` cannot parse and therefore reads as
False - so a bad sample would vanish into a clean-looking second. Flags are
aggregated by meaning instead: "something is wrong here" is OR-ed across the
bin, "this sample is good" is AND-ed. The shipped self-test asserts every flag
column survives the merge as a genuine boolean.

**A lone file is not resampled.** One file with no offset is delegated straight
to the v1.7 reader and returned unchanged, asserted frame-for-frame against the
reference run. Binning moves values, and there is nothing to align a single file
to. The consequence is stated in the app rather than left to be discovered: the
same file loaded alone and loaded alongside a second one can give slightly
different block statistics, because in the second case it has been binned.

**delta_P in the prediction results.** `Pocap_kPa - core_dP_kPa`, shown under
the four setpoints on Tab 2 and written to the prediction log. Pure arithmetic
on values already predicted - nothing is fitted for it and nothing about the
prediction changed because of it. No interval is quoted: the two predictions are
correlated through the anchors they were fitted on, and combining their
intervals as though they were independent would report a confidence this app has
not earned. It is computed before the six-significant-figure display rounding,
not from the rounded text.

## What changed in v1.7

**No modelling change.** Nothing here touches the calibration maths, the
extraction criteria, or the dev-training search and split. The check on that is
mechanical rather than asserted: `--selftest` output is line-for-line identical
to v1.6's apart from the version string, the timestamp and the install path -
every block count, RMS, LOO-CV score, held-out score, coefficient and predicted
setpoint unchanged. What changed is that after six rounds of additions an
operator could no longer tell what the app was doing or why.

**The dev-training dialog scrolls.** It never had a scroll area at all -
contents were laid straight onto the dialog. Survivable while everything was
shut, and unusable the moment it was not: opening the candidate detail or the
full-data refit added several hundred pixels to a window that cannot grow past
the screen, and the Adopt buttons went off the bottom edge with no way back.
The body now scrolls and the action bar sits outside it, so no amount of
expanded detail can put the primary action out of reach. `CollapsibleSection`
also invalidates size hints up its whole parent chain on toggle: showing a child
invalidates only the immediate parent layout, which is not far enough for a
scroll area to recompute its scrollbar range, so the content would grow while
the range stayed put.

**Every channel says where its numbers came from.** A `Source:` line sits under
each equation, outside every disclosure, in one shape:

```
Source: Auto - cap_ID_um, quadratic, additive, on 13 anchor(s) from this run.
        Quadratic was chosen because it had the lowest cross-validated error of
        the 2 shape(s) tried, and nothing simpler came within 10% of it.
```

The reason is computed from the scan the decision was actually made on, not
paraphrased, so it names the real numbers - the anchor floor when the guardrail
fired, the percentage margin when the tolerance rule decided. It is wired to
whatever last touched the channel: adopting a dev-training configuration makes
the line say so, with the date, the block count behind the fit that was
reviewed, and which of the two Adopt actions produced it - and moving any of
that channel's dropdowns afterwards revokes the credit rather than leaving a
held-out search named against a shape it did not choose.

Where the architecture cannot honestly support the label, the line says so
rather than rounding up. The v1.6 report noted that the Fit & Inspect tab refits
against its own anchor set whichever Adopt button was pressed, which makes
"Dev-trained" true of the *selection* and false of the *coefficients*. So the
line carries both halves: *"The search chose the shape; the coefficients above
were refitted here on all 13 of this tab's anchors, so they are not the fit the
held-out score was computed for."* A label that is subtly untrue is worse than
none, because it is the one an operator would rely on.

**The v1.3 legibility bar, re-measured.** v1.3 required an unfamiliar reader to
get a channel's equation without opening anything, and checked it by walking the
widget tree. v1.4-v1.6 grew that back: three overlapping captions per channel
(the additive/ratio reading, the terms whose interval spans zero, the comparison
against the two-feature fit), a section header repeating the anchor count and
variable and shape, a fit banner listing every channel's *n* again, and a banner
announcing that nothing had gone wrong. Per channel the page is now an equation,
a Source line, and a quality summary - `RMS · residual dof · cautions`, which is
what the Source line does *not* say. Everything that merely justifies the fit
moved behind the existing `CollapsibleSection`; no new decluttering mechanism
was introduced. The widget-tree test now asserts both directions: equation and
Source reachable, justification not.

**Auto, manual and dev-training are related in the copy.** A permanent explainer
sits above the channel controls - not a tooltip, not dismissible, because the
distinction is which data a choice was checked against rather than a preference.
It replaces the old "each channel picks its own order..." caption rather than
joining it.

## What changed in v1.6

**The 80/90% split inconsistency, diagnosed and fixed.** Setting the dev-training
search fraction to 90% reported *fewer* blocks than 80% - 48 down to 29 on the
reference run - which is backwards: a larger search fraction cannot find less.

The suspected cause was that the percentage cut was being applied to the raw
time series before extraction, so a cut landing mid-steady-period would split
one long block into two fragments. **That is not what was happening.** Extraction
already ran once, on the whole series, and the split already operated on the
extracted block list; the block list was byte-identical at both fractions. The
actual cause was downstream, in `dev_training.train_auto_selection`: at 90% the
chronological cut left one block on the held-out side for `furnace_temp_C` (9
blocks) and `draw_speed_m_min` (10), below `MIN_HELD_OUT_BLOCKS = 2`, so both
channels were **skipped whole** - and two channels vanishing from the report
reads exactly like a smaller extraction.

The cut is now clamped so the held-out minimum is always reserved: the boundary
moves instead of the channel being discarded, and the note says by how much
("the cut was moved back to 78%"). A channel is refused only when its block list
cannot seat both minimums at *any* fraction, which makes the refusal a property
of the extraction rather than of the number in the spin box. `ChannelSplit` now
carries `n_blocks` and an `is_partition` check, the results table leads with a
**blocks extracted** column, and the self-test asserts across 50-90% that the
per-channel block counts are identical, that every split is a partition, and
that raising the fraction never trains fewer channels.

**Adopt and refit on full data.** A second action beside the original one. It
takes the winning `(variable, shape, form)` per channel *exactly* as the
split-validated search selected it - the search is **not** re-run, which would
let the held-out blocks influence the selection and silently invalidate the
held-out score printed above it - and refits only the coefficients on 100% of
the blocks: search set, held-out set, and anything the split did not use.

The refit reports its own RMS, degrees of freedom and parameter intervals before
anything is adopted, because more data is not the same thing as a better fit. On
the reference run it is not: `core_dP_kPa`'s `fibre_OD_um` term and three of
`Pocap_kPa`'s terms were clear of zero on the search set and are not once every
block is included, and the refit says so rather than leaving it to be noticed.

The original action stays as **Adopt (search-set fit)** - sometimes the held-out
portion should remain held out for reasons this dialog knows nothing about.

**Cubic removed; quadratic allowed to compete.** Cubic is gone from the manual
dropdown, from Auto, and from the dev-training search. It never had a physical
justification - nothing in the draw physics suggests an inflection in a small
residual correction - and at 4 parameters for a single variable or 7 for the
two-feature fit, the anchor counts in play cannot support it. Removed rather
than hidden: `MAX_ORDER` is 2, there is no label for a third power, and an
explicitly requested cubic **raises** instead of being quietly clamped down to
quadratic, which is how a removed option comes back as a silent downgrade.

The anchor threshold that forces linear-only is now
`MIN_ANCHORS_FOR_QUADRATIC = 9`, replacing the cubic-era
`MIN_ANCHORS_FOR_ORDER_SELECTION = 15`. Deliberately not the old value: 15 was
sized so the largest shape the search could reach still had room to be checked,
and that shape no longer exists. 9 comes from residual degrees of freedom at the
two quadratic shapes the search can now produce - a single-variable quadratic
costs 3 parameters and leaves 6, the widest two-feature quadratic costs 5 and
leaves 4 - both positive and non-trivial, with 4 about where a t interval stops
being too wide to say anything.

Crossing that line is now a **caution, not a wall**. Auto still stays linear
below it, but a manual quadratic is fitted and flagged: the caution names the
anchor count, says plainly that nothing on the tab has tested the choice, and
points at Dev > Train Auto selection. The real safeguard against a bad quadratic
is the held-out score, not a hard anchor-count wall - and since v1.5 that
mechanism exists, so the wall was standing in for something now available. On
the reference run `Pocap_kPa` (13 anchors) now earns a quadratic that the old
threshold forbade.

## What changed in v1.5

**Auto now chooses the variable too.** The search covers every single variable
at linear/quadratic (and, before v1.6, cubic) plus the two-feature fit, across
whichever forms the channel is allowed. Selection is the v1.4 simplicity
tolerance generalised from polynomial order to **parameter count**, so a
three-parameter two-feature fit only displaces a two-parameter single-variable
one by beating it by more than `AUTO_ORDER_TOLERANCE`.

Both existing guardrails now govern the whole search rather than the shape axis
alone: below the anchor threshold every variable is still in the running but
only at linear, and `allowed_forms()` - the same function the manual dropdown
greys out from - is what the search enumerates, so a ratio candidate cannot be
generated for a gauge pressure channel by a second check that drifted.

**Dev-mode training, behind a Dev menu.** A larger search is a larger chance of
winning by luck, and nothing computed inside the anchor set can detect that:
LOO-CV proves a candidate generalises to the other blocks of the same run,
which is a different claim from generalising to another run. So the dev dialog
splits the blocks **chronologically** (default 80/20, configurable; specific
block ids can be marked always-held-out for data already treated as external
validation), runs the whole search on the search portion only, refits the
winner there, and scores it once against blocks that took no part in choosing
it. Two columns, never merged: *LOO-CV (from the search)* and *held-out (never
seen during search)*. Nothing is applied until one of the Adopt actions is
pressed (v1.5 had one; v1.6 added the full-data refit beside it).

**The split immediately earned its keep.** On the reference run the search
picked `fibre_OD_um` for `furnace_temp_C` with the best cross-validated score of
any candidate - and a held-out error of **4.35e14**. The cause: `fibre_OD_um` is
*exactly constant* across the first seven blocks, so the fit is rank-deficient,
the slope and intercept become interchangeable, `lstsq` returns a coefficient of
-1.9e15 that cancels in-sample, and the degeneracy is present in every LOO fold -
so it scored better than every honest candidate. A feature that does not vary is
now refused entry to the search (`feature_is_degenerate`), with a
condition-number backstop for the near-degenerate case. Manual selection can
still fit one; the search will not choose it for you.

With that fixed, held-out against searched error on the reference run reads:
`draw_speed_m_min` and `Pocap_kPa` hold up, `core_dP_kPa` comes out 2.5x worse,
`furnace_temp_C` 2.9x - and where only two or three blocks were held out the
verdict says so rather than pretending the number settles anything.

## What changed in v1.4

**Extraction is one slider.** Six absolute rolling-SD thresholds, in six
different units, are replaced by the supervisor's percent-variation metric

```
pct_change(t) = |A(t) - A(t-B)| / max(|A(t)|, floor) * 100
```

with two safeguards the raw formula needs: a denominator floor derived per
channel from that channel's own range (`ZERO_CROSSING_FLOOR_FRACTION`), so a
gauge pressure passing through zero does not turn a trivial wobble into
thousands of percent; and a minimum contiguous span, reusing the existing
minimum-block duration, so one favourable sample-to-sample difference is not
mistaken for a settled process. Note what the metric measures: **drift over the
lag B, not spread** - a channel oscillating fast about a fixed level scores
zero. That is a deliberate simplification, which is why the old absolute
criterion is kept selectable rather than deleted.

*The slider is a multiplier, not a raw percentage, and that is not a
convenience.* Measured inside stretches the validated extraction already called
steady, the natural percent-change over a 30 s lag is:

| channel | | channel | |
| --- | --- | --- | --- |
| `fibre_ID_um` | 9.30 % | `draw_speed_m_min` | 0.21 % |
| `feed_speed_mm_min` | 1.33 % | `furnace_temp_C` | 0.010 % |
| `outer_dP_kPa` | 0.85 % | `core_dP_kPa` | 0.61 % |

Three orders of magnitude, between an optical gauge reading a 48 µm feature and
a thermocouple. A sweep of flat thresholds confirms the consequence: **zero
blocks for `core_dP_kPa` and `Pocap_kPa` at every sensitivity worth having.** So
the slider scales each channel against its own reference drift, measured from
the loaded run - one knob, no per-channel numbers, and the derived scaling shown
in a table rather than hidden.

**The reproduction test is replaced by a measurement, not deleted.** The
absolute criterion still reproduces the hand-checked 8-block result bit-for-bit
and is still tested. `compare_criteria` reports the difference, which the
selftest prints:

| mapping | blocks (absolute) | blocks (percent) | settled s |
| --- | --- | --- | --- |
| `furnace_temp_C` | 9 | **9** | 3568 → 4317 |
| `draw_speed_m_min` | 10 | **10** | 3686 → 4278 |
| `core_dP_kPa` | 8 | **16** | 3182 → 3627 |
| `Pocap_kPa` | 7 | **13** | 3035 → 2940 |

The two kinematic mappings land on exactly the validated counts; the two
pressure mappings roughly double their anchors.

> Superseded in v1.8.2: the "percent" column above measured a segmentation
> mechanism that turned out not to be invariant to how much of a run was
> loaded - see "Extraction, fixed: segmentation no longer depends on how
> much of a run is loaded" further down. The fix makes segmentation more
> sensitive on every mapping, not only the pressure pair; the kinematic
> counts landing on the absolute-criterion's numbers was a coincidence of
> the old mechanism, not a property worth having kept. Current counts:
> `furnace_temp_C` 14, `draw_speed_m_min` 16, `core_dP_kPa` 17, `Pocap_kPa`
> 13.

**Auto shape selection, which is not argmin.** A fourth option beside
Linear/Quadratic/Cubic, and the default. Auto picks the *simplest* shape whose
LOO-CV error is within `AUTO_ORDER_TOLERANCE` (10%) of the best - because at
these anchor counts a few percent of CV error is which block got held out, and
a parameter is not. `MIN_ANCHORS_FOR_ORDER_SELECTION` still overrides it: below
15 anchors Auto is linear whatever the tolerance rule found. It says which
shape it took and why, in one line. Two tests guard the rule that must not
regress - one that Auto prefers the simpler shape when a quadratic is 4% better,
one that it never exceeds linear below the guardrail - and the shipped exe
asserts the guardrail during `--selftest`.

> Superseded in v1.6: cubic is no longer an option, and the threshold is now
> `MIN_ANCHORS_FOR_QUADRATIC = 9`, below which a quadratic is cautioned rather
> than forbidden. The tolerance rule itself is unchanged.

**Blocks can be excluded per channel.** Select a row, exclude it; the block
stays in the table flagged `excluded` rather than vanishing, "Restore all" puts
it back, and the calibration refits immediately. Exclusion is a view over the
extraction, never an edit to it, so nothing needs re-running to undo. Excluding
a block for one mapping leaves the others alone. Exclusions are cleared when the
extraction re-runs, since block ids are assigned per extraction and a carried-over
id would silently remove a different window.

## What changed in v1.3

The correction is now fitted against **one chosen variable**, and the quantity
being fitted is chosen **per channel**. Extraction and the extraction
diagnostic are untouched.

**Ratio or additive, and the pressure channels may not have a ratio.** A ratio
says "the estimate is out by a factor", which only means anything when both
sides share a sign and stay away from zero:

| Channel | Form | Why |
| --- | --- | --- |
| `furnace_temp_C` | ratio, `actual / analytic` | absolute temperature, ~1781 analytic against ~1997 measured |
| `draw_speed_m_min` | ratio, `actual / analytic` | a speed, nowhere near zero |
| `core_dP_kPa` | additive | gauge pressure - the zero is a reference choice, so a multiplier is not a physical quantity |
| `Pocap_kPa` | additive | same, and its analytic estimate is observed at −17 kPa against a positive measurement, where no multiplier exists at all |

The refusal is a declared property of the channel, not a check on today's
numbers: `core_dP_kPa`'s analytic happens to stay positive on the anchors seen
so far, and a future all-positive anchor set must not quietly re-enable a
multiplier between two gauge pressures. The UI greys the option out and states
the reason — and states the *accurate* reason per channel, so it does not claim
a sign change that a given anchor set does not contain.

The direction is fixed at `actual / analytic` and written into the equation
itself, not left to a caption.

**One variable, chosen per channel.** A dropdown per channel offers the six raw
inputs, both engineered features, and the original two-feature fit. Changing it
refits, rescans the LOO-CV order comparison against that one variable, redraws
the correction-vs-variable scatter, and rewrites the equation — live. Each
channel opens on a physically-motivated default, stated as the hypothesis it is:

| Channel | Default | Why |
| --- | --- | --- |
| `core_dP_kPa` | `fibre_ID_um` | core pressure is the mechanism that sets fiber ID |
| `Pocap_kPa` | `cap_ID_um` | capillary pressure sets the capillary bore directly |
| `furnace_temp_C` | `tension_g` | temperature governs viscosity, which is what tension measures |
| `draw_speed_m_min` | `cap_wall_ratio` | v1.2's fit found no other predictor mattered here |

On the reference run every one of these predicts held-out blocks *better* than
the two-feature fit — `core_dP` 0.73 against 1.92 kPa, `Pocap` 1.41 against
2.21 kPa — but that is a result, not a guarantee, and the report says so either
way.

**The fitted equation, in words, without opening anything.** Each channel's
result leads with a line like

```
furnace_temp_C:  analytic x (1.0963 + 6.372e-05 x tension_g)  =  actual
Pocap_kPa correction = 10.37 - 0.9714 x cap_ID_um  kPa (gauge)
```

Coefficients are re-expressed in raw, uncentred variables — an exact change of
basis — so the line can be substituted into directly and gives the same number
the app does; a test asserts that for every channel. Terms whose interval spans
zero are left out and named underneath. Dropping one holds it at its centre
rather than folding its intercept share in, which would otherwise put a
`core_dP correction = 13.86 kPa` on screen for a fit whose correction is
actually 1.57 kPa.

**Calibration format 3.** The stored fit records the form and the variable per
channel; a format-2 file is refused with a clear message rather than loaded and
assumed additive.

## What changed in v1.2

Presentation only - no change to extraction logic, fit maths or order
selection, and the extraction diagnostic plot is byte-identical to v1.1's.

**Tab 1 is three steps instead of one scroll.** *Load & extract*, *Match
analytic*, *Fit & inspect*. Each page leads with its conclusion and folds the
evidence into a collapsed section: block counts are visible, per-block medians
and the window table are one click away; each channel shows `n=9 anchors ·
linear, linear · RMS 0.19 degC` with its candidate-order table and plot behind
its own **Advanced** disclosure. Fit options and the full coefficient report are
likewise collapsed. The tallest page is now roughly half the height of the v1.1
page.

**The order plot shows cross-validated error only.** The training curve was a
line nobody should be selecting from, and leaving it on screen invited exactly
that. It is still computed and still in the candidate table; the plot has a
`show training error` toggle, off by default, and a caption saying why.

**The raw-geometry diagnostic grid is gone.** The fitted scatter already shows
what it was there to show.

**The plot toolbar is readable.** It was white-on-white. Cause: matplotlib
chooses its icon colour from
`toolbar.palette().color(backgroundRole()).value() < 128`, and the v1.1
stylesheet set that background to `transparent`, which resolves to #000000 at
zero alpha - value 0 - so matplotlib concluded dark mode and drew white icons.
Fixed in both places: the stylesheet gives the toolbar an opaque light card
with hover, checked and pressed states, and `PlotPanel` pins the palette roles
directly so the icons cannot invert again if the stylesheet is edited. The
toolbar was also being squeezed to its minimum width by the hint label beside
it, collapsing nine buttons into an overflow chevron; it now has a fixed size
policy and the hint yields instead.

**Extraction settings start collapsed**, with a summary line - `61 s window,
60 s minimum block, 6 channel(s) monitored - all at the validated defaults` -
that says whether anything has been changed from the proven values, and the
Extract button now sits above the settings rather than below them.

## What changed in v1.1

**Steadiness is decided per correction pair, not globally.** v1.0 required every
monitored channel to be simultaneously flat before it would call a stretch
steady, which is stricter than the physics demands - a furnace blip does not
invalidate a pressure/geometry window. Each setpoint now has its own criteria
set (`DEFAULT_STABILITY_GROUPS` in `steady_state.py`), its own block
boundaries, and its own anchor count. Groups stay *joint over their own
members*: pressure controls wall thinning which sets the ID, so those are never
separated - only the irrelevant channels are dropped.

Relaxing a veto alone would have made the anchor set worse, not better: with
nothing to separate them, two stretches at different pressures merge into one
block whose median geometry never existed. So a channel outside a criteria set
still *segments* a block when its level moves, even though its noise no longer
rejects a sample. Measured on the reference run:

| mapping | blocks v1.0 | blocks v1.1 | settled seconds v1.0 | v1.1 |
| --- | --- | --- | --- | --- |
| `furnace_temp_C` | 8 | 9 | 2969 | 3568 |
| `draw_speed_m_min` | 8 | 10 | 2969 | 3686 |
| `core_dP_kPa` | 8 | 8 | 2969 | 3182 |
| `Pocap_kPa` | 8 | 7 | 2969 | 3035 |

The gain is modest and the app says why: for every mapping, the binding
constraint is that mapping's *own* output channel, which cannot be relaxed
because it is the quantity being measured. `Pocap_kPa` losing a block is not a
regression - two stretches at identical pressures, ID and speeds, separated
only by a transient, merged into one better-determined anchor.

**Function shape is chosen per channel per term, by cross-validation.** Each
term can be linear, quadratic or cubic, selected by leave-one-block-out CV -
never training RMS, which falls with every added parameter regardless of
whether the parameter means anything. Every candidate is scored side by side
with its training error next to its CV error, so the overfitting is visible
rather than argued about. Below `MIN_ANCHORS_FOR_ORDER_SELECTION` (15) anchors
the app selects linear whatever CV prefers, and says so: at these sample sizes
the order ranking itself can turn on one block. On the reference run all four
channels are below that threshold, and CV's preference for cubic and quadratic
shapes is shown but not acted on.

> Superseded in v1.6: a term is linear or quadratic, cubic having been removed
> entirely. The threshold is `MIN_ANCHORS_FOR_QUADRATIC = 9`, which three of the
> four channels on the reference run now clear.

**Raw-geometry diagnostics.** Section 6 plots each setpoint directly against
the raw geometry it is plausibly linked to, so the engineered features can be
sanity-checked against the thing they were engineered from. It is labelled
diagnostic-only throughout; the fitted scatter lives in section 7.

**Interactive plots.** Every plot carries matplotlib's Qt navigation toolbar
(box zoom, pan, reset) and hover tooltips. Hovering the extraction trace
reports the sample's value *and* why it was or was not counted as steady -
naming the channel that blocked it.

## The model

Since v1.3 each channel fits either an additive correction or a ratio, against
a single chosen variable — see above. The two-feature shape below is still
selectable per channel and is what v1.0–v1.2 used throughout:

```
actual  =  analytic  +  theta0
                     +  theta1 * (cap_wall_ratio     - centre)
                     +  theta2 * (log drawdown ratio - centre)
                    [+  theta3 * (analytic           - centre)]
```

* `cap_wall_ratio = (cap_OD_um - cap_ID_um) / cap_OD_um` - dimensionless.
* draw-down ratio - dimensionless, computed either **kinematically** from the
  analytic draw speed and the feed speed, or **geometrically** from a preform
  outer diameter over the fiber outer diameter when that is known. Whichever is
  used is recorded in the calibration and reused unchanged at prediction time.
  The *analytic* draw speed is used, never the measured one: the measured draw
  speed is one of the four things being predicted, so it does not exist at
  prediction time.
* The fourth (gain) term is off by default. At eight anchors a parameter is
  expensive.

Fitted by **inverse-variance weighted least squares**, weighting each anchor by
`1 / (se_actual^2 + se_analytic^2 + floor^2)`. Both standard errors enter
because the residual being fitted is a difference of two measured medians. The
floor is there because a channel that never moved during a block reports a
standard error of exactly zero - true of those samples, false of the instrument
- and without it that one block would take infinite weight.

Intervals come from a t distribution on `n - p` degrees of freedom, not a
normal one. At n≈8 the difference is large and it is in the direction of
honesty.

The weighted least squares is implemented on numpy and scipy in
`calibration.py` rather than pulled from statsmodels: at two to four parameters
it is a handful of lines, keeping it local puts the covariance and interval
conventions in the same file as the model, and it is one less large dependency
in the exe. `tests/test_calibration.py` cross-checks the coefficients, standard
errors, scale and degrees of freedom against `statsmodels.WLS` where that
package is installed.

## What the app does

### Tab 1 - Extract & calibrate

Three steps. Detail marked *(collapsed)* is present but folded away by default.

**Step 1 - Load & extract**

1. **Load raw data** - a 1 Hz experimental CSV (`hcf_timeseries_1s.csv` shape).
2. **Extract** - centred rolling standard deviation per channel against a
   per-channel threshold, QC-flag filtering, short-gap bridging, edge trimming
   and a minimum block duration, run once per correction pair. Defaults are the
   values already validated against a real draw run, and the all-channel rule
   still reproduces that reference extraction exactly. *(settings collapsed,
   with a summary line saying whether anything differs from the defaults)*
3. **Diagnostic plot** - the timeseries with the selected mapping's blocks
   shaded, so what was rejected is as visible as what was kept. Hover any
   sample for its value and the reason it was or was not kept.
4. **Blocks found per correction pair** - anchors and settled seconds per
   mapping against the all-channel rule, and the binding constraint.
   *(collapsed: per-block medians and errors, and a window table covering every
   stretch of the run, accepted and rejected alike)*

**Step 2 - Match analytic**

5. **Analytic estimate per block** - load a second file (either a full
   `model_ready_data_ready2.csv`-style dataset with `analytic_*` columns, or a
   file holding just those columns and a timestamp). Each block takes the
   median of the analytic rows falling inside its `[start_time, end_time]`
   window. The number of contributing rows is reported per block, and a block
   that matched few rows or none is flagged rather than silently given a
   number.

**Step 3 - Fit & inspect**

6. **Fit** - weighted least squares per channel. Visible by default: the
   guardrail caution, one line per channel giving its anchor count, chosen
   shape per term and residual RMS, and the calibrated-against-measured scatter
   with each anchor's own measurement error. *(collapsed, per channel: the
   order override boxes, what cross-validation preferred, the candidate table
   and the LOO-CV plot. Collapsed globally: fit options, and the full report of
   every coefficient, interval and caution.)* Saving persists the calibration
   with joblib, tagged with the preform id, the timestamp and the per-channel
   anchor counts, plus a plain-CSV copy of the anchor tables.

### Tab 2 - Predict

Target geometry in, calibrated setpoints out, each labelled with how many
anchors *that channel* rests on and the interval around the correction. Since
v1.1 the channels have different anchor counts, so the badge is per channel -
collapsing them to one figure would overstate the thinnest one's evidence.

**The analytic estimate is typed in, by design.** The tab has four required
fields - `analytic_furnace_temp_C`, `analytic_draw_speed_m_min`,
`analytic_core_dP_kPa`, `analytic_Pocap_kPa` - and *Run prediction* stays
disabled until all four are filled. The operator runs the supervisor's fast
estimator for the target themselves and enters what it said. This is permanent
for now, not a fallback: inferring or defaulting those numbers would produce a
prediction that looks exactly like a real one while resting on nothing.

Every prediction is appended to `data/prediction_log.csv`. Future anchors come
from real draws, and a real draw starts with a prediction.

## Extension points

* **New preform geometry** - append a `PreformDefinition` to `preform.py`.
  Calibrations are stored per preform id; nothing else changes.
* **Live analytic estimator** - implement `LiveEstimatorAnalyticSource` in
  `analytic_source.py`. It is the only class that needs writing: the rest of the
  app depends on the `AnalyticSource` interface, not on where an estimate came
  from. Nothing constructs it today, and Tab 2 is not wired to it.
* **Preform outer diameter** - the geometric draw-down ratio is already
  implemented and selected by entering a preform OD on Tab 1. Left at zero, the
  kinematic ratio is used.

## Non-goals

No RandomForest, no Gaussian process, no active-learning suggestion layer - the
sample size argument above has not changed, and per-mapping extraction moved it
from 8 points to 7-10, not to hundreds. No nested or custom-capillary
*physics* - v1.9 added a nested preform's schema, geometry and pressure
chain, but no new analytic model behind them.
No live connection to the estimator - only the interface seam and the
dataset-backed implementation. No runtime dependency on the reference app. The
WLS core is unchanged: numpy and scipy, still cross-checked against statsmodels
in the tests. Only block detection and feature shape moved in v1.1.

Added in v1.6: **no reintroduction of cubic anywhere.** The ratio/additive
branching and the zero-variance-feature refusal are unchanged from v1.5.
Nested and custom preforms remain out of scope and are gated on having real
anchor data for a new geometry - not on anything in this codebase.

Added in v1.7: **no new dimensionless features and no new modelling modes.**
v1.7 is a legibility release and changes no fitted number; the feature ideas
that came out of the literature review are a separate, future change and were
deliberately not folded in here.

Added in v1.9: **no nested-specific physics.** The nested preform adds a
schema, a pressure-chain convention and per-layer wall ratios - no new analytic
model, no change to the existing preform's behaviour or stored calibration, and
no fabricated anchor data. It starts untrained on purpose.

Added in v1.8: **no gap-filling for a missing `cap_ID_um`.** Interpolation,
imputation and any other way of inventing a capillary bore that was not measured
are explicitly deferred - the merge reports how many seconds are empty and
leaves them empty. v1.8 also changes nothing about extraction, fitting or
prediction; it is an input-plumbing release plus one derived display value.

---

## Running

```bash
pip install -r requirements.txt
python app.py
```

Tests (the reference-data tests skip if the parent folder's CSVs are absent):

```bash
python -m pytest tests -q
```

## Building the exe

```bash
pyinstaller app.spec
```

Produces `dist/HCFAnchorPredictor.exe` - single file, windowed, icon embedded.

Because a windowed build has no console, a bundling mistake would otherwise
show up only as a window that never appears. The exe can run the whole pipeline
inside the bundle and write down what happened:

```bash
dist/HCFAnchorPredictor.exe --selftest RAW.csv ANALYTIC.csv --out report.txt
```

Exit code 0 and `RESULT: PASS` in the report means raw CSV -> blocks ->
analytic match -> calibration -> save -> reload -> prediction all work in the
shipped artefact.

`app.spec` carries the packaging lessons from the reference app - the pandas
and scipy dynamic imports, the PySide6 Addons bloat, the torch false positive
that arrives through scikit-learn - plus two of its own: **pyarrow** is
excluded (pandas 3 declares it, nothing here reaches it, it is 60 MB), and
`scipy.optimize._highspy` **cannot** be excluded despite looking like dead
weight, because `scipy.stats` imports `scipy.optimize`, which imports
`_linprog` eagerly.

## Files

| File | What it is |
| --- | --- |
| `schema.py` | Column names, units, sanity bounds, derived dimensionless quantities |
| `preform.py` | Preform registry: one implemented geometry, one disabled placeholder |
| `paths.py` | Where calibrations and logs live, with a read-only-folder fallback |
| `steady_state.py` | Steady-state block extraction and per-block statistics |
| `analytic_source.py` | `AnalyticSource` interface, dataset-backed implementation, live stub |
| `calibration.py` | Weighted least squares, intervals, prediction, persistence |
| `ingest.py` | Several raw CSVs onto one 1 Hz grid. No extraction, no fitting |
| `analytic_export.py` | The estimator's input spreadsheet, one row per surviving block |
| `provenance.py` | What produced a channel's coefficients, as one line. No maths |
| `preform.py` | The geometry registry: one entry per preform, each with its own schema |
| `ui_common.py` | Shared widgets, table model, palette-matched matplotlib canvas |
| `ui_extract_tab.py` | Tab 1 |
| `ui_predict_tab.py` | Tab 2 |
| `app.py` | Window shell, startup calibration load, `--selftest` |
| `style.qss` | Adapted from the reference app's stylesheet |
| `app.spec` | PyInstaller build |
| `tests/` | Extraction reproduction, analytic matching, fit arithmetic, UI flow |
