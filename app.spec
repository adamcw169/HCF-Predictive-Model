# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller build spec for HCF Anchor Predictor.

Build with:   pyinstaller app.spec
Output:       dist/HCFAnchorPredictor.exe  (single file, no console window)

Equivalent to --onefile --windowed --icon=app.ico, kept as a spec so the hidden
imports and exclusions below are reproducible.

The lists here start from the ones already worked out for the HCF Draw
Predictor, minus everything that was there for scikit-learn. This app has no
learner to bundle - the model is a few weighted-least-squares coefficients - so
the sklearn hidden imports are gone and sklearn itself is excluded outright.
Verify a build with:

    dist\\HCFAnchorPredictor.exe --selftest RAW.csv ANALYTIC.csv --out report.txt

which runs the whole pipeline inside the bundle and writes down what happened.
A windowed exe has no console, so a missing hidden import otherwise shows up
only as a window that never appears.
"""

block_cipher = None

hidden_imports = [
    # joblib persists the calibration; its default backend is imported by name.
    "joblib",
    "joblib.externals.loky",
    "joblib.externals.loky.backend",
    "joblib.externals.cloudpickle",
    # pandas' C parser and its datetime machinery. PyInstaller's static
    # analysis does not see these; without them the exe builds and then fails
    # on the first read_csv.
    "pandas._libs",
    "pandas._libs.tslibs",
    "pandas._libs.tslibs.base",
    # scipy.stats supplies the t distribution behind every confidence interval
    # in the app, and pulls its distributions in by name at import time.
    "scipy.stats",
    "scipy.stats._continuous_distns",
    "scipy.special",
    "scipy.special.cython_special",
    "scipy.linalg.cython_blas",
    "scipy.linalg.cython_lapack",
    "scipy._lib.messagestream",
    # matplotlib's Qt backend is selected at runtime by name, from ui_common.
    "matplotlib.backends.backend_qtagg",
    "matplotlib.backends.backend_agg",
    # mplcursors drives the hover tooltips. It is imported in a try/except so a
    # missing copy degrades to plots without tooltips rather than a crash -
    # which also means PyInstaller would happily ship a build with it silently
    # absent, so it is named here rather than left to static analysis.
    "mplcursors",
]

a = Analysis(
    ["app.py"],
    pathex=[],
    binaries=[],
    # Loaded at runtime via resource_path(), so they must ride along in the
    # bundle rather than being discovered as imports.
    datas=[
        ("style.qss", "."),
        ("app.ico", "."),
    ],
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # scikit-learn is the point of the exclusion list, not an afterthought.
        # This app deliberately has no forest and no Gaussian process; if
        # sklearn ever appears in a build it means something imported it by
        # accident. Excluding it also removes the chain that made the other
        # app's exe large: sklearn's optional array-API support references
        # torch, which is enough for PyInstaller to follow it and drag in CUDA
        # plumbing and the inductor compiler - over 160 MB of exe for code that
        # cannot execute here.
        "sklearn",
        "scikit-learn",
        "torch",
        "torchvision",
        "torchaudio",
        # Test-only cross-check of the weighted least squares. The app never
        # imports it, but it is installed in the development environment.
        "statsmodels",
        "patsy",
        # pandas 3 declares pyarrow as a dependency, but nothing this app does
        # reaches it: it reads and writes CSV, does arithmetic on float columns,
        # and never touches parquet, arrow-backed strings or the Flight
        # transport. Its DLLs are over 60 MB of the bundle, so it is excluded
        # and the exclusion is verified - `--selftest` runs the whole pipeline
        # inside the built exe. If a future change starts relying on an
        # arrow-backed dtype, remove this line and the size comes back.
        "pyarrow",
        # Not excludable, though it looks it: the HiGHS solver is 6.7 MB and
        # this app never solves a linear program, but scipy.optimize imports
        # _linprog eagerly and scipy.stats imports scipy.optimize - and
        # scipy.stats is where the t distribution behind every interval comes
        # from. Excluding "scipy.optimize._highspy" builds cleanly and then
        # dies on import. Left written down so the next person does not spend
        # a build cycle rediscovering it.
        #
        # matplotlib pulls Pillow in for raster image IO. This app draws vector
        # plots into a Qt canvas and never opens or writes an image file, so the
        # heavier codecs are dead weight. Pillow itself stays - matplotlib
        # imports PIL.Image at module level.
        "PIL._avif",
        "PIL._webp",
        "streamlit",
        "altair",
        "plotly",
        "pydeck",
        "sympy",
        "networkx",
        "jupyter_client",
        "ipykernel",
        "openpyxl",
        "uvicorn",
        "starlette",
        "watchdog",
        "MPh",
        "jpype",
        # PySide6 ships as PySide6_Essentials plus PySide6_Addons, and the
        # addons (WebEngine, 3D, Multimedia, Quick/QML, Charts...) are most of
        # the install by weight. This app uses QtCore, QtGui and QtWidgets and
        # nothing else. Removing an entry from this list only makes the exe
        # larger; it cannot make the app work better.
        "PySide6.Qt3DAnimation",
        "PySide6.Qt3DCore",
        "PySide6.Qt3DExtras",
        "PySide6.Qt3DInput",
        "PySide6.Qt3DLogic",
        "PySide6.Qt3DRender",
        "PySide6.QtBluetooth",
        "PySide6.QtCharts",
        "PySide6.QtDataVisualization",
        "PySide6.QtDesigner",
        "PySide6.QtGraphs",
        "PySide6.QtGraphsWidgets",
        "PySide6.QtHelp",
        "PySide6.QtHttpServer",
        "PySide6.QtLocation",
        "PySide6.QtMultimedia",
        "PySide6.QtMultimediaWidgets",
        "PySide6.QtNetworkAuth",
        "PySide6.QtNfc",
        "PySide6.QtPdf",
        "PySide6.QtPdfWidgets",
        "PySide6.QtPositioning",
        "PySide6.QtQml",
        "PySide6.QtQuick",
        "PySide6.QtQuick3D",
        "PySide6.QtQuickControls2",
        "PySide6.QtQuickWidgets",
        "PySide6.QtRemoteObjects",
        "PySide6.QtScxml",
        "PySide6.QtSensors",
        "PySide6.QtSerialBus",
        "PySide6.QtSerialPort",
        "PySide6.QtSpatialAudio",
        "PySide6.QtSql",
        "PySide6.QtStateMachine",
        "PySide6.QtTest",
        "PySide6.QtTextToSpeech",
        "PySide6.QtUiTools",
        "PySide6.QtWebChannel",
        "PySide6.QtWebEngineCore",
        "PySide6.QtWebEngineQuick",
        "PySide6.QtWebEngineWidgets",
        "PySide6.QtWebSockets",
        "tkinter",
        "PyQt5",
        "PyQt6",
        "PySide2",
        "wx",
        "IPython",
        "jupyter",
        "notebook",
        "nbconvert",
        "nbformat",
        "pytest",
        "sphinx",
        "docutils",
        "PIL.ImageQt",
        "sqlalchemy",
        "tornado",
        "zmq",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="HCFAnchorPredictor",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    # console=False is the --windowed half: a double-clicked exe must not open
    # a terminal behind the window. --selftest writes its report to a file for
    # exactly this reason.
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon="app.ico",
)
