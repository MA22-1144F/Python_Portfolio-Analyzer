# -*- mode: python ; coding: utf-8 -*-
"""
Portfolio Analyzer - PyInstaller Specification File
このファイルはPyInstallerによるexeファイル生成の設定を定義します
"""

import os
from pathlib import Path

block_cipher = None

# プロジェクトルートディレクトリ
ROOT_DIR = Path(SPECPATH)

# 収集するデータファイル（assetsディレクトリ全体）
datas = [
    (str(ROOT_DIR / 'assets'), 'assets'),
    (str(ROOT_DIR / 'config' / 'default_settings.json'), 'config'),
]

# 隠れたインポート（明示的に指定が必要なモジュール）
hiddenimports = [
    'PySide6.QtCore',
    'PySide6.QtGui',
    'PySide6.QtWidgets',
    'PySide6.QtSvg',
    'PySide6.QtPrintSupport',
    'pandas',
    'numpy',
    'scipy',
    'scipy.optimize',
    'scipy.stats',
    'plotly',
    'plotly.graph_objs',
    'matplotlib',
    'matplotlib.backends.backend_qt5agg',
    'yfinance',
    'requests',
    'json',
    'pathlib',
    'logging',
]

a = Analysis(
    [str(ROOT_DIR / 'main.py')],
    pathex=[str(ROOT_DIR)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
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
    name='PortfolioAnalyzer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # GUIアプリケーションなのでコンソールを非表示
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ROOT_DIR / 'assets' / 'icons' / 'app_icon.ico'),  # アプリケーションアイコン
    version=str(ROOT_DIR / 'version_info.txt'),  # バージョン情報
)
