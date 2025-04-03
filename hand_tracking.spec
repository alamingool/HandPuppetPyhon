# -*- mode: python ; coding: utf-8 -*-

a = Analysis(
    ['hand_tracking.py'],
    pathex=['venv/Lib/site-packages'],  # Add your virtual environment's site-packages
    binaries=[],
    datas=[
        (r'd:\users\willi\downloads\handpuppetpython\venv\Lib\site-packages\mediapipe\modules\hand_landmark\hand_landmark_tracking_cpu.binarypb', 'mediapipe/modules/hand_landmark'),
        (r'd:\users\willi\downloads\handpuppetpython\venv\Lib\site-packages\mediapipe\modules\palm_detection\palm_detection_full.tflite', 'mediapipe/modules/palm_detection'),
        (r'd:\users\willi\downloads\handpuppetpython\venv\Lib\site-packages\mediapipe\modules\hand_landmark\hand_landmark_full.tflite', 'mediapipe/modules/hand_landmark'),
        (r'd:\users\willi\downloads\handpuppetpython\venv\Lib\site-packages\mediapipe\modules\hand_landmark\handedness.txt', 'mediapipe/modules/hand_landmark'),
    ],
    hiddenimports=['cv2', 'mediapipe', 'numpy', 'protobuf', 'av'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='hand_tracking',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # Set to True if you want a console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)