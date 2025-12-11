"""main.py"""

#!/usr/bin/env python3
import sys
import logging
from pathlib import Path
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QIcon

from ui.main_window import MainWindow
from config.app_config import initialize_config
from config.constants import (
    APP_NAME, APP_VERSION, APP_ID,
    ICON_FILE_EXTENSIONS, ASSETS_DIR, ICONS_DIR,
    LOG_FORMAT
)

logger = logging.getLogger(__name__)

class PortfolioAnalyzerApp:
    def __init__(self):
        self.app = None
        self.main_window = None
        self.config = None
    
    def initialize_app(self):
        """アプリケーションの初期化"""
        self.app = QApplication(sys.argv)
        self.config = initialize_config()
        
        # アプリケーション情報設定
        self.app.setApplicationName(self.config.get('app.name', APP_NAME))
        self.app.setApplicationVersion(self.config.get('app.version', APP_VERSION))
        
        # アイコン設定
        self._set_app_icon()
        
        # スタイルシート適用
        self._apply_stylesheet()
    
    def _set_app_icon(self):
        """アプリケーションアイコンを設定"""
        # ICO形式を優先，次にPNG
        icon_files = [f"app_icon{ext}" for ext in ICON_FILE_EXTENSIONS]
        
        project_root = Path(__file__).parent
        icon_path = None
        for filename in icon_files:
            path = project_root / ASSETS_DIR / ICONS_DIR / filename
            if path.exists():
                icon_path = path
                break
        
        if icon_path:
            icon = QIcon(str(icon_path))
            if not icon.isNull():
                self.app.setWindowIcon(icon)
                logger.info(f"アイコンを設定: {icon_path}")
                
                # Windowsのタスクバーアイコンを設定
                self._set_windows_taskbar_icon(str(icon_path))
            else:
                logger.warning(f"アイコンの読み込みに失敗: {icon_path}")
        else:
            logger.warning("アイコンファイルが見つかりません")
    
    def _set_windows_taskbar_icon(self, icon_path):
        """Windowsのタスクバーアイコンを設定"""
        try:
            if sys.platform == 'win32':
                import ctypes
                from ctypes import wintypes
                
                # AppUserModelIDを設定
                ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)
                
                # ICO形式の場合
                if icon_path.endswith('.ico'):
                    # アイコンをロード
                    icon_flags = 0x00000000  # LR_DEFAULTSIZE
                    hicon = ctypes.windll.user32.LoadImageW(
                        None,
                        icon_path,
                        1,  # IMAGE_ICON
                        0, 0,
                        0x00000010 | icon_flags  # LR_LOADFROMFILE
                    )
                    
                    if hicon:
                        logger.info("タスクバーアイコンを設定しました")
                    else:
                        logger.warning("アイコンのロードに失敗しました")
                else:
                    logger.info("ICO形式のアイコンを推奨します")
                    
        except Exception as e:
            logger.error(f"タスクバーアイコン設定エラー: {e}")
    
    def _apply_stylesheet(self):
        """スタイルシートを適用"""
        try:
            colors = self.config.get_ui_colors()
            style = self._generate_base_style(colors)
            
            # 外部スタイルファイルがあれば追加
            external_style = self._load_external_style()
            if external_style:
                style += "\n" + external_style
            
            self.app.setStyleSheet(style)
            
        except Exception as e:
            logger.error(f"スタイルシート適用エラー: {e}")
            # フォールバック: 外部ファイルのみ使用
            external_style = self._load_external_style()
            if external_style:
                self.app.setStyleSheet(external_style)
    
    def _generate_base_style(self, colors):
        """基本スタイルシートを生成"""
        bg_color = colors.get('background', '#2b2b2b')
        surface_color = colors.get('surface', '#3c3c3c')
        surface_hover_color = colors.get('surface_hover', '#4c4c4c')
        primary_color = colors.get('primary', '#0078d4')
        text_color = colors.get('text_primary', '#ffffff')
        border_color = colors.get('border', '#555555')
        
        return f"""
        QMainWindow, QWidget {{
            background-color: {bg_color};
            color: {text_color};
        }}
        
        QTabWidget::pane {{
            border: 1px solid {border_color};
            background-color: {surface_color};
        }}
        
        QTabBar::tab {{
            background-color: {surface_color};
            color: {text_color};
            padding: 8px 16px;
            margin-right: 2px;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
        }}
        
        QTabBar::tab:selected {{
            background-color: {primary_color};
        }}
        
        QTabBar::tab:hover {{
            background-color: {surface_hover_color};
        }}
        
        QFrame {{
            background-color: {surface_color};
            border: 1px solid {border_color};
            border-radius: 4px;
        }}
        
        QMenuBar {{
            background-color: {surface_color};
            color: {text_color};
            border-bottom: 1px solid {border_color};
        }}
        
        QMenuBar::item:selected, QMenu::item:selected {{
            background-color: {primary_color};
        }}
        
        QMenu {{
            background-color: {surface_color};
            color: {text_color};
            border: 1px solid {border_color};
        }}
        
        QStatusBar {{
            background-color: {surface_color};
            border-top: 1px solid {border_color};
        }}
        """
    
    def _load_external_style(self):
        """外部スタイルファイルを読み込み"""
        project_root = Path(__file__).parent
        style_path = project_root / "assets" / "style.qss"
        if style_path.exists():
            try:
                with open(style_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                logger.error(f"外部スタイルファイル読み込みエラー: {e}")
        return None
    
    def create_main_window(self):
        """メインウィンドウを作成"""
        self.main_window = MainWindow(self.config)
        
        # ウィンドウサイズ設定
        width = self.config.get('window.width', 1200)
        height = self.config.get('window.height', 700)
        self.main_window.resize(width, height)
        self.main_window.show()
    
    def run(self):
        """アプリケーション実行"""
        self.initialize_app()
        self.create_main_window()
        return self.app.exec()


def main():
    try:
        # ロギング設定
        logging.basicConfig(
            level=logging.INFO,
            format=LOG_FORMAT
        )

        app = PortfolioAnalyzerApp()
        exit_code = app.run()
        sys.exit(exit_code)
    except Exception as e:
        logger.error(f"Application startup error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()