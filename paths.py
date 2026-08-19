"""Local file locations.

All app state lives next to the executable: `./data/` for logs and `./models/`
for saved calibrations. When frozen by PyInstaller, "next to the executable"
means the folder holding the `.exe`, not the temporary extraction directory
(which is deleted on exit and would silently lose every saved calibration).

If that folder is read-only - the exe was dropped under Program Files, say -
storage falls back to %LOCALAPPDATA%\\HCFAnchorPredictor so the app still works
instead of failing at startup. `storage_note()` reports which one is in use so
the operator can always find their files.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

APP_NAME = "HCFAnchorPredictor"

_resolved_base: Path | None = None
_using_fallback = False


def _executable_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def _is_writable(directory: Path) -> bool:
    try:
        directory.mkdir(parents=True, exist_ok=True)
        probe = directory / ".write_test"
        probe.write_text("", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


def base_dir() -> Path:
    """Root folder holding ./data and ./models."""
    global _resolved_base, _using_fallback
    if _resolved_base is not None:
        return _resolved_base

    preferred = _executable_dir()
    if _is_writable(preferred):
        _resolved_base = preferred
    else:
        local_appdata = os.environ.get("LOCALAPPDATA")
        fallback = (
            Path(local_appdata) / APP_NAME
            if local_appdata
            else Path.home() / f".{APP_NAME}"
        )
        fallback.mkdir(parents=True, exist_ok=True)
        _resolved_base = fallback
        _using_fallback = True
    return _resolved_base


def data_dir() -> Path:
    path = base_dir() / "data"
    path.mkdir(parents=True, exist_ok=True)
    return path


def models_dir() -> Path:
    path = base_dir() / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _legacy_folder_with(preform_id: str, filename: str) -> Path | None:
    """An existing file under one of this preform's previous ids, if any.

    The preform ids were renamed in the tubular/NANF/DNANF release and the id
    is the storage key, so a calibration saved before it lives under the old
    folder name. Rather than migrating files on disk - which would rewrite a
    user's stored artefact as a side effect of upgrading - the old location is
    simply still readable. New saves go to the current id.
    """
    try:
        import preform as preform_registry

        entry = preform_registry.get_preform(preform_id)
    except Exception:  # noqa: BLE001 - a path helper must not depend on lookup
        return None
    for legacy in entry.legacy_ids:
        candidate = models_dir() / legacy / filename
        if candidate.exists():
            return candidate
    return None


def calibration_path(preform_id: str) -> Path:
    """Where the fitted calibration for one preform is persisted.

    Reads fall back to a previous id's folder when nothing has been saved under
    the current one, so the rename does not orphan an existing calibration.
    """
    folder = models_dir() / preform_id
    folder.mkdir(parents=True, exist_ok=True)
    current = folder / "calibration.joblib"
    if not current.exists():
        legacy = _legacy_folder_with(preform_id, "calibration.joblib")
        if legacy is not None:
            return legacy
    return current


def anchor_blocks_path(preform_id: str) -> Path:
    """Companion CSV of the anchor blocks a calibration was fitted on.

    Written alongside the joblib so the anchor set stays readable without the
    app, and so a later operator can see exactly which draws it rests on.
    """
    folder = models_dir() / preform_id
    folder.mkdir(parents=True, exist_ok=True)
    current = folder / "anchor_blocks.csv"
    if not current.exists():
        legacy = _legacy_folder_with(preform_id, "anchor_blocks.csv")
        if legacy is not None:
            return legacy
    return current


def prediction_log_path() -> Path:
    return data_dir() / "prediction_log.csv"


def storage_note() -> str:
    base = base_dir()
    if _using_fallback:
        return (
            f"Storage: {base} (the application folder was not writable, so "
            "local app data is used instead)"
        )
    return f"Storage: {base}"
