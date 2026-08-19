"""Widgets, plotting helpers and palette shared by both tabs."""

from __future__ import annotations

import matplotlib

matplotlib.use("QtAgg")

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qtagg import (
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
from PySide6.QtCore import QAbstractTableModel, Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QSizePolicy,
    QTableView,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

try:  # pragma: no cover - exercised by whether the package is installed
    import mplcursors
except ImportError:  # pragma: no cover
    mplcursors = None

# Matches the QSS palette so charts do not look pasted in from elsewhere.
COLOR_BG = "#FFFFFF"
COLOR_TEXT = "#1F2933"
COLOR_MUTED = "#6B7684"
COLOR_GRID = "#EDEFF2"
COLOR_LINE = "#7C8896"
COLOR_ACCENT = "#2D6CDF"
COLOR_ACCENT_SOFT = "#C7D8F5"
COLOR_SECONDARY = "#E0A020"
COLOR_ERROR = "#C53030"
COLOR_OK = "#2F855A"


class DataFrameModel(QAbstractTableModel):
    """Read-only Qt table model over a pandas DataFrame."""

    def __init__(self, df: pd.DataFrame | None = None, float_format: str = "{:.4g}"):
        super().__init__()
        self._df = df if df is not None else pd.DataFrame()
        self._float_format = float_format

    def set_dataframe(self, df: pd.DataFrame) -> None:
        self.beginResetModel()
        self._df = df.reset_index(drop=True)
        self.endResetModel()

    def dataframe(self) -> pd.DataFrame:
        return self._df

    def rowCount(self, parent=None) -> int:  # noqa: N802 - Qt API
        return len(self._df)

    def columnCount(self, parent=None) -> int:  # noqa: N802 - Qt API
        return len(self._df.columns)

    def data(self, index, role=Qt.DisplayRole):  # noqa: N802 - Qt API
        if not index.isValid():
            return None
        value = self._df.iloc[index.row(), index.column()]
        if role == Qt.DisplayRole:
            if value is None or (isinstance(value, float) and pd.isna(value)):
                return ""
            if isinstance(value, (float, np.floating)):
                return self._float_format.format(float(value))
            if isinstance(value, pd.Timestamp):
                return value.strftime("%Y-%m-%d %H:%M:%S")
            return str(value)
        if role == Qt.TextAlignmentRole:
            if isinstance(value, (int, float, np.number)) and not isinstance(
                value, bool
            ):
                return int(Qt.AlignRight | Qt.AlignVCenter)
            return int(Qt.AlignLeft | Qt.AlignVCenter)
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):  # noqa: N802
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return str(self._df.columns[section])
        return str(self._df.index[section])


class Banner(QLabel):
    """Coloured message strip used for load results, coverage and warnings."""

    def __init__(self, text: str = "", kind: str = "info", parent=None):
        super().__init__(text, parent)
        self.setWordWrap(True)
        self.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        self.set_kind(kind)
        self.setVisible(bool(text))

    def set_kind(self, kind: str) -> None:
        self.setProperty("banner", kind)
        # Re-polish so the new property selector takes effect immediately.
        self.style().unpolish(self)
        self.style().polish(self)

    def show_message(self, text: str, kind: str = "info") -> None:
        self.setText(text)
        self.set_kind(kind)
        self.setVisible(bool(text))

    def clear_message(self) -> None:
        self.setText("")
        self.setVisible(False)


class MplCanvas(FigureCanvasQTAgg):
    """Matplotlib canvas styled to match the app palette."""

    def __init__(self, width: float = 5.0, height: float = 3.0, dpi: int = 100):
        self.figure = Figure(figsize=(width, height), dpi=dpi, facecolor=COLOR_BG)
        super().__init__(self.figure)
        self.setStyleSheet("background-color: transparent;")

    def style_axes(self, ax, grid: str = "y") -> None:
        ax.set_facecolor(COLOR_BG)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#DFE3E8")
        ax.tick_params(colors=COLOR_MUTED, labelsize=8, length=0)
        if grid in ("y", "both"):
            ax.yaxis.grid(True, color=COLOR_GRID, linewidth=1)
        if grid in ("x", "both"):
            ax.xaxis.grid(True, color=COLOR_GRID, linewidth=1)
        ax.set_axisbelow(True)
        ax.xaxis.label.set_color(COLOR_MUTED)
        ax.yaxis.label.set_color(COLOR_MUTED)
        ax.title.set_color(COLOR_TEXT)
        for text in (ax.title, ax.xaxis.label, ax.yaxis.label):
            text.set_fontsize(9)

    def show_placeholder(self, message: str) -> None:
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            message,
            ha="center",
            va="center",
            color=COLOR_MUTED,
            fontsize=9,
            wrap=True,
        )
        self.draw_idle()


# The toolbar background colour, kept next to the QSS rule that must agree
# with it. Matching the app's light card surface.
TOOLBAR_BG = "#F5F6F8"


def _force_light_toolbar_icons(toolbar) -> None:
    """Stop matplotlib drawing its toolbar icons white.

    matplotlib picks the icon colour by reading the toolbar's own palette:
    `palette().color(backgroundRole()).value() < 128` means "dark theme, draw
    white". A Qt stylesheet that sets `background-color: transparent` resolves
    that role to #000000 at zero alpha - value 0 - so the heuristic fires on a
    light app and every icon comes out white on near-white. That is the bug
    this fixes.

    Setting the palette role explicitly makes the check read the colour the
    toolbar is actually painted, independently of what the stylesheet says, so
    the icons cannot silently invert again if `style.qss` is edited later.
    """
    palette = toolbar.palette()
    light = QColor(TOOLBAR_BG)
    for role in (QPalette.Button, QPalette.Window, QPalette.Base):
        palette.setColor(role, light)
    palette.setColor(QPalette.WindowText, QColor(COLOR_TEXT))
    palette.setColor(QPalette.ButtonText, QColor(COLOR_TEXT))
    toolbar.setPalette(palette)


class PlotPanel(QWidget):
    """An `MplCanvas` with zoom/pan and hover-to-read-exact-value.

    Zoom and pan come from matplotlib's own Qt navigation toolbar, so they are
    the real thing rather than a reimplementation - rectangle zoom, drag to pan,
    home to reset, and save-image for free.

    Hover comes from mplcursors, driven by a per-plot formatter so the tooltip
    can say "block 4, measured 5.53 kPa" instead of a bare pair of coordinates.
    If mplcursors is not installed the plots still draw and still zoom; only the
    tooltip is missing, and `hover_available` reports that so the UI can say so
    rather than appearing broken.
    """

    def __init__(
        self,
        width: float = 5.0,
        height: float = 3.0,
        dpi: int = 100,
        toolbar: bool = True,
        parent=None,
    ):
        super().__init__(parent)
        self.canvas = MplCanvas(width=width, height=height, dpi=dpi)
        self._cursor = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        if toolbar:
            bar = QHBoxLayout()
            bar.setContentsMargins(0, 0, 0, 0)
            self.toolbar = NavigationToolbar(self.canvas, self)
            self.toolbar.setIconSize(self.toolbar.iconSize() * 0.8)
            _force_light_toolbar_icons(self.toolbar)
            # Fixed, so the hint beside it cannot squeeze the toolbar down to
            # its minimum - which collapses nine buttons into a single overflow
            # chevron and hides zoom and pan behind a menu.
            self.toolbar.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            bar.addWidget(self.toolbar)
            bar.addStretch(1)
            self.hint = QLabel(
                "drag to pan · box-zoom from the toolbar · hover for values"
                if self.hover_available
                else "drag to pan · box-zoom from the toolbar"
            )
            self.hint.setProperty("role", "subheading")
            # Yields space before the toolbar does when the panel is narrow.
            self.hint.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
            bar.addWidget(self.hint)
            layout.addLayout(bar)
        else:
            self.toolbar = None

        layout.addWidget(self.canvas)

    @property
    def hover_available(self) -> bool:
        return mplcursors is not None

    @property
    def figure(self):
        return self.canvas.figure

    def style_axes(self, ax, grid: str = "y") -> None:
        self.canvas.style_axes(ax, grid=grid)

    def show_placeholder(self, message: str) -> None:
        self._clear_cursor()
        self.canvas.show_placeholder(message)

    def draw_idle(self) -> None:
        self.canvas.draw_idle()

    def _clear_cursor(self) -> None:
        if self._cursor is not None:
            try:
                self._cursor.remove()
            except Exception:  # noqa: BLE001 - a stale cursor must never block a redraw
                pass
            self._cursor = None

    def set_hover(self, artists, formatter) -> None:
        """Attach hover tooltips to `artists`, labelled by `formatter`.

        `formatter` receives an mplcursors selection and returns the tooltip
        text. Called after every redraw; the previous cursor is dropped first,
        because a cursor holding artists from a cleared figure keeps them alive
        and pops tooltips for points that are no longer on screen.
        """
        self._clear_cursor()
        if mplcursors is None or not artists:
            return
        cursor = mplcursors.cursor(artists, hover=True, highlight=False)

        @cursor.connect("add")
        def _on_add(selection):  # pragma: no cover - interactive
            try:
                text = formatter(selection)
            except Exception:  # noqa: BLE001 - never break the plot over a label
                text = ""
            selection.annotation.set_text(text)
            selection.annotation.get_bbox_patch().set(
                facecolor=COLOR_TEXT, edgecolor="none", alpha=0.92
            )
            selection.annotation.set_color("#FFFFFF")
            selection.annotation.set_fontsize(8)
            if selection.annotation.arrow_patch is not None:
                selection.annotation.arrow_patch.set(
                    arrowstyle="-", color=COLOR_TEXT, alpha=0.6
                )

        self._cursor = cursor

    def reset_view(self) -> None:
        """Forget the zoom history so a new dataset starts framed correctly."""
        if self.toolbar is not None:
            self.toolbar.update()


class CollapsibleSection(QWidget):
    """A titled panel that starts closed and opens on click.

    The app has a lot of detail an operator occasionally needs and routinely
    does not: per-channel thresholds, candidate-order tables, block-by-block
    windows. Showing all of it at once buries the three or four numbers that
    matter. This keeps it one click away instead of one scroll away, and the
    summary line stays visible while it is shut so closing something never
    means losing track of what it says.
    """

    def __init__(self, title: str, summary: str = "", expanded: bool = False, parent=None):
        super().__init__(parent)
        self._title = title

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        self.toggle = QToolButton()
        self.toggle.setProperty("role", "collapse")
        self.toggle.setCheckable(True)
        self.toggle.setChecked(expanded)
        self.toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self.toggle.setText(title)
        self.toggle.setCursor(Qt.PointingHandCursor)
        self.toggle.toggled.connect(self._on_toggled)
        header.addWidget(self.toggle)

        self.summary = QLabel(summary)
        self.summary.setProperty("role", "subheading")
        self.summary.setWordWrap(True)
        header.addWidget(self.summary, 1)
        layout.addLayout(header)

        self.content = QWidget()
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(18, 4, 0, 4)
        self.content.setVisible(expanded)
        layout.addWidget(self.content)

    def _on_toggled(self, checked: bool) -> None:
        self.toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        self.content.setVisible(checked)
        self._resize_ancestors()

    def _resize_ancestors(self) -> None:
        """Tell every layout above this section that its size hint just moved.

        Showing or hiding a child invalidates the *immediate* parent layout, and
        for a section sitting directly in a window that is enough. It is not
        enough inside a `QScrollArea`: a resizable scroll area sizes its content
        widget from that widget's own size hint, and the scrollbar range is
        computed from the result. If the invalidation stops before it reaches
        the content widget, the section expands into space the scroll area does
        not know exists - the content grows, the scrollbar range does not, and
        everything below the section becomes unreachable.

        Walking to the top costs one pass over a handful of layouts, and only on
        a click, so it is cheaper than reasoning about how far up any particular
        caller needs the invalidation to travel.
        """
        widget = self.content
        while widget is not None:
            layout = widget.layout()
            if layout is not None:
                layout.invalidate()
            widget.updateGeometry()
            widget = widget.parentWidget()

    def add_widget(self, widget: QWidget) -> None:
        self.content_layout.addWidget(widget)

    def add_layout(self, layout) -> None:
        self.content_layout.addLayout(layout)

    def set_summary(self, text: str) -> None:
        self.summary.setText(text)

    def set_expanded(self, expanded: bool) -> None:
        self.toggle.setChecked(expanded)

    @property
    def is_expanded(self) -> bool:
        return self.toggle.isChecked()


def heading(text: str) -> QLabel:
    label = QLabel(text)
    label.setProperty("role", "heading")
    return label


def subheading(text: str) -> QLabel:
    label = QLabel(text)
    label.setProperty("role", "subheading")
    label.setWordWrap(True)
    return label


def mono(text: str) -> QLabel:
    label = QLabel(text)
    label.setProperty("role", "mono")
    label.setWordWrap(True)
    return label


def badge(text: str, thin: bool = False) -> QLabel:
    label = QLabel(text)
    label.setProperty("role", "badge-thin" if thin else "badge")
    label.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
    return label


def number_input(
    lo: float,
    hi: float,
    decimals: int = 3,
    step: float = 1.0,
    suffix: str = "",
    value: float | None = None,
) -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setRange(lo, hi)
    box.setDecimals(decimals)
    box.setSingleStep(step)
    box.setKeyboardTracking(True)
    if suffix:
        box.setSuffix(f" {suffix}")
    if value is not None:
        box.setValue(value)
    return box


def fit_table(view: QTableView, stretch_last: bool = False) -> None:
    """Common table view setup: read-only, tidy, sized to content."""
    view.setAlternatingRowColors(True)
    view.setSelectionBehavior(QTableView.SelectRows)
    view.setEditTriggers(QTableView.NoEditTriggers)
    view.verticalHeader().setVisible(False)
    view.horizontalHeader().setStretchLastSection(stretch_last)
    view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
    view.setWordWrap(False)


def format_measure(value: float, unit: str = "", digits: int = 4) -> str:
    if value is None or not np.isfinite(value):
        return "-"
    text = f"{value:.{digits}g}"
    return f"{text} {unit}".strip()
