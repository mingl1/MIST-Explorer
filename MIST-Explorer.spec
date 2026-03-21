# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules
from PyInstaller.utils.hooks import copy_metadata
from PyInstaller.utils.hooks import collect_data_files

datas = [('assets', 'assets')]
hiddenimports = ['metadata']
datas += copy_metadata('scanpy')
datas += copy_metadata('scikit-learn')
datas += copy_metadata('pandas')
datas += collect_data_files('matplotlib')
hiddenimports += collect_submodules('scikit-learn')
hiddenimports += collect_submodules('stardist')


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={
        "PyQt6": {
            "excluded_plugins": [
                "qtvirtualkeyboard",
                "qtwebengine",
                "qtpdf",
                "qtsql",
                "qtbluetooth",
                "qtnfc",
                "qt3dinput",
                "qt3dlogic",
                "qt3drender",
                "qt3dquick",
                "qt3dextras",
                "qtlocation",
                "qtpositioning",
            ]
        }
    },
    runtime_hooks=[],
    excludes=[
        # TensorFlow extras not needed at runtime
        'tensorboard',
        'tensorflow.python.debug',
        'tensorflow_estimator',
        'tensorflow.lite',
        # Jupyter/IPython - not needed in bundled app
        'IPython',
        'ipykernel',
        'jupyter',
        'notebook',
        'jupyterlab',
        # Other unused
        'tkinter',
        '_tkinter',
        'wx',
        'gi',
        'pygments',
        'jedi',
    ],
    noarchive=False,
    optimize=0,
)

# Filter out large binaries that are not needed at runtime
EXCLUDE_BINARIES = [
    'opencv_videoio_ffmpeg',  # video codec, not needed for image loading
    'opengl32sw',             # Qt software OpenGL fallback (drop if machines have GPU)
]
a.binaries = TOC([
    (name, path, typ)
    for name, path, typ in a.binaries
    if not any(excl.lower() in name.lower() for excl in EXCLUDE_BINARIES)
])

pyz = PYZ(a.pure)
splash = Splash(
    '.\\i2.png',
    binaries=a.binaries,
    datas=a.datas,
    text_pos=None,
    text_size=12,
    minify_script=True,
    always_on_top=True,
)

exe = EXE(
    pyz,
    a.scripts,
    splash,
    [],
    name='MIST-Explorer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=['vcruntime140.dll', 'msvcp140.dll', '_pywrap_tensorflow_internal.pyd'],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['assets\\final_icon.png'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    splash.binaries,
    strip=False,
    upx=True,
    upx_exclude=['vcruntime140.dll', 'msvcp140.dll', '_pywrap_tensorflow_internal.pyd'],
    name='MIST-Explorer',
)
