"""main_window.py"""

from PySide6.QtWidgets import QMainWindow, QTabWidget, QMessageBox
from PySide6.QtCore import QTimer
from PySide6.QtGui import QAction

from ui.tab_analysis import AnalysisTab
from ui.tab_compare import CompareTab
from ui.tab_management import ManagementTab

import logging

class MainWindow(QMainWindow):
    def __init__(self, config=None):
        super().__init__()
        self.config = config
        self.analysis_tab = None
        self.compare_tab = None
        self.management_tab = None
        
        self._setup_ui()
        self._setup_menu()
        self._setup_status_bar()
        self._load_settings()
        self._connect_portfolio_export_signals() 
    
    def _setup_ui(self):
        """UIを初期化"""
        self.setWindowTitle("Portfolio Analyzer")
        
        # ウィンドウサイズ設定
        width = self.config.get('window.width', 1200) if self.config else 1200
        height = self.config.get('window.height', 700) if self.config else 700
        self.resize(width, height)
        
        # タブ作成
        self.tab_widget = QTabWidget()
        self.analysis_tab = AnalysisTab()
        self.compare_tab = CompareTab()
        self.management_tab = ManagementTab()
        
        self.tab_widget.addTab(self.analysis_tab, "個別資産分析")
        self.tab_widget.addTab(self.compare_tab, "ポートフォリオの比較分析")
        self.tab_widget.addTab(self.management_tab, "ポートフォリオ管理")
        
        self.setCentralWidget(self.tab_widget)
    
    def _connect_portfolio_export_signals(self):
        """ポートフォリオエクスポートシグナルを接続"""
        # 分析タブの結果タイルから分析ウィジェットを取得してシグナルを接続
        if self.analysis_tab and hasattr(self.analysis_tab, 'result_tile'):
            result_area = self.analysis_tab.result_tile.result_area
            
            # 既存の分析アイテムのシグナルを接続
            for item_widget in result_area.analysis_items:
                self._connect_analysis_widget_signal(item_widget.analysis_widget)
            
            # 今後追加される分析アイテムのために，result_areaを監視
            original_add = result_area.add_analysis_item
            
            def wrapped_add(item_type, item_name):
                original_add(item_type, item_name)
                # 最後に追加されたアイテムのシグナルを接続
                if result_area.analysis_items:
                    latest_item = result_area.analysis_items[-1]
                    self._connect_analysis_widget_signal(latest_item.analysis_widget)
            
            result_area.add_analysis_item = wrapped_add
    
    def _connect_analysis_widget_signal(self, analysis_widget):
        """分析ウィジェットのportfolio_export_requestedシグナルを接続"""
        if hasattr(analysis_widget, 'portfolio_export_requested'):
            try:
                # 既存の接続を切断（重複接続を防ぐ）
                analysis_widget.portfolio_export_requested.disconnect()
            except (RuntimeError, TypeError):
                # 接続されていない場合は無視
                pass
            
            # 新しい接続を作成
            analysis_widget.portfolio_export_requested.connect(
                self._on_portfolio_export_requested
            )
    
    def _on_portfolio_export_requested(self, portfolio):
        """ポートフォリオエクスポート要求を処理"""
        try:
            # 確認ダイアログを表示
            reply = QMessageBox.question(
                self,
                "ポートフォリオ管理タブへ移動",
                f"ポートフォリオ「{portfolio.name}」を管理タブで開きますか？\n\n"
                f"管理タブに移動し，ウエイトを確認・調整できます．",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            
            if reply == QMessageBox.Yes:
                # 管理タブに切り替え
                self.tab_widget.setCurrentIndex(2)
                
                # 管理タブにポートフォリオを渡す
                if self.management_tab:
                    self.management_tab.load_optimized_portfolio(portfolio)
                    self.status_bar.showMessage(
                        f"ポートフォリオ「{portfolio.name}」を管理タブで開きました"
                    )
        
        except Exception as e:
            QMessageBox.critical(
                self,
                "エラー",
                f"ポートフォリオの読み込み中にエラーが発生しました:\n{str(e)}"
            )
    
    def _setup_menu(self):
        """メニューバーを設定"""
        menubar = self.menuBar()
        
        self._create_file_menu(menubar)
        self._create_view_menu(menubar)
        self._create_tools_menu(menubar)
        self._create_help_menu(menubar)
    
    def _create_file_menu(self, menubar):
        """ファイルメニューを作成"""
        file_menu = menubar.addMenu("ファイル(&F)")
        
        actions = [
            ("新規ポートフォリオ(&N)", "Ctrl+N", self.new_portfolio),
            ("ポートフォリオを開く(&O)", "Ctrl+O", self.open_portfolio),
            ("保存(&S)", "Ctrl+S", self.save_portfolio),
            ("名前を付けて保存(&A)", "Ctrl+Shift+S", self.save_portfolio_as),
        ]
        
        for name, shortcut, slot in actions:
            action = QAction(name, self)
            action.setShortcut(shortcut)
            action.triggered.connect(slot)
            file_menu.addAction(action)
        
        file_menu.addSeparator()
        
        # ポートフォリオ操作
        portfolio_menu = file_menu.addMenu("ポートフォリオ操作")
        portfolio_actions = [
            ("ウエイトを正規化", self.normalize_portfolio_weights),
            ("全資産を削除", self.clear_portfolio),
        ]
        
        for name, slot in portfolio_actions:
            action = QAction(name, self)
            action.triggered.connect(slot)
            portfolio_menu.addAction(action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("終了(&X)", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
    
    def _create_view_menu(self, menubar):
        """表示メニューを作成"""
        view_menu = menubar.addMenu("表示(&V)")
        
        # タブ切り替え
        tab_menu = view_menu.addMenu("タブ")
        tab_actions = [
            ("個別資産分析(&A)", "Ctrl+1", 0),
            ("比較分析(&C)", "Ctrl+2", 1),
            ("ポートフォリオ管理(&M)", "Ctrl+3", 2),
        ]
        
        for name, shortcut, index in tab_actions:
            action = QAction(name, self)
            action.setShortcut(shortcut)
            action.triggered.connect(lambda checked, i=index: self.tab_widget.setCurrentIndex(i))
            tab_menu.addAction(action)
        
        view_menu.addSeparator()
        
        dark_mode_action = QAction("ダークモード(&D)", self)
        dark_mode_action.setCheckable(True)
        dark_mode_action.triggered.connect(self.toggle_dark_mode)
        view_menu.addAction(dark_mode_action)
    
    def _create_tools_menu(self, menubar):
        """ツールメニューを作成"""
        tools_menu = menubar.addMenu("ツール(&T)")
        
        fetch_rate_action = QAction("短期国債金利を取得(&R)", self)
        fetch_rate_action.triggered.connect(self.fetch_interest_rate)
        tools_menu.addAction(fetch_rate_action)
        
        tools_menu.addSeparator()
        
        validate_action = QAction("ポートフォリオを検証(&V)", self)
        validate_action.triggered.connect(self.validate_portfolio)
        tools_menu.addAction(validate_action)
    
    def _create_help_menu(self, menubar):
        """ヘルプメニューを作成"""
        help_menu = menubar.addMenu("ヘルプ(&H)")
        
        guide_menu = help_menu.addMenu("使い方ガイド")
        guides = [
            ("ポートフォリオ管理", self.show_portfolio_guide),
            ("ウエイト設定", self.show_weight_guide),
            ("レバレッジとキャッシュ", self.show_leverage_guide),
        ]
        
        for name, slot in guides:
            action = QAction(name, self)
            action.triggered.connect(slot)
            guide_menu.addAction(action)
        
        help_menu.addSeparator()
        
        about_action = QAction("バージョン情報(&A)", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def _setup_status_bar(self):
        """ステータスバーを設定"""
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("準備完了")
        
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self._update_status)
        self.status_timer.start(5000)
    
    def _load_settings(self):
        """設定を読み込み"""
        if not self.config:
            return
        
        try:
            import base64
            from PySide6.QtCore import QByteArray
            
            # ウィンドウジオメトリの復元
            geometry_str = self.config.get('window.geometry')
            if geometry_str:
                try:
                    geometry_bytes = base64.b64decode(geometry_str)
                    self.restoreGeometry(QByteArray(geometry_bytes))
                except Exception:
                    # ジオメトリの復元に失敗した場合は無視
                    pass
            
            # 最後に開いていたタブを復元
            last_tab = self.config.get('window.last_tab_index', 0)
            tab_index = min(last_tab, self.tab_widget.count() - 1)
            self.tab_widget.setCurrentIndex(tab_index)
        except Exception:
            # 設定読み込みに失敗した場合は無視
            pass

    def _save_settings(self):
        """設定を保存"""
        if not self.config:
            return
        
        try:
            import base64
            # QByteArrayをbase64文字列に変換して保存
            geometry = self.saveGeometry()
            geometry_str = base64.b64encode(geometry.data()).decode('utf-8')
            self.config.set('window.geometry', geometry_str)
            self.config.set('window.last_tab_index', self.tab_widget.currentIndex())
            self.config.save()
        except Exception as e:
            # 保存に失敗してもログに記録
            logging.warning(f"設定の保存に失敗しました: {e}")

    def closeEvent(self, event):
        """ウィンドウクローズ時の処理"""
        self._save_settings()
        event.accept()
    
    # ポートフォリオ操作
    def new_portfolio(self):
        """新規ポートフォリオ作成"""
        self.tab_widget.setCurrentIndex(2)
        if self.management_tab:
            self.management_tab.new_portfolio()
        self.status_bar.showMessage("新規ポートフォリオを作成しました")
    
    def open_portfolio(self):
        """ポートフォリオを開く"""
        self.tab_widget.setCurrentIndex(2)
        if self.management_tab:
            self.management_tab.tab_widget.setCurrentIndex(1)
        self.status_bar.showMessage("ポートフォリオを開く...")
    
    def save_portfolio(self):
        """ポートフォリオを保存"""
        if self.management_tab:
            self.management_tab.save_portfolio()
        self.status_bar.showMessage("ポートフォリオを保存しました")
    
    def save_portfolio_as(self):
        """名前を付けてポートフォリオを保存"""
        if self.management_tab:
            self.management_tab.save_portfolio_as()
        self.status_bar.showMessage("ポートフォリオを保存しました")
    
    def _get_management_edit_widget(self):
        """管理タブの編集ウィジェットを取得"""
        if self.tab_widget.currentIndex() != 2 or not self.management_tab:
            return None
        return self.management_tab.portfolio_edit_widget
    
    def normalize_portfolio_weights(self):
        """ポートフォリオのウエイトを正規化"""
        edit_widget = self._get_management_edit_widget()
        if edit_widget and edit_widget.assets_widget:
            edit_widget.assets_widget.normalize_weights()
        self.status_bar.showMessage("ウエイトを正規化しました")
    
    def clear_portfolio(self):
        """ポートフォリオをクリア"""
        edit_widget = self._get_management_edit_widget()
        if edit_widget and edit_widget.assets_widget:
            edit_widget.assets_widget.clear_all_assets()
        self.status_bar.showMessage("ポートフォリオをクリアしました")
    
    def fetch_interest_rate(self):
        """短期国債金利を取得"""
        edit_widget = self._get_management_edit_widget()
        if edit_widget and edit_widget.summary_widget:
            edit_widget.summary_widget.fetch_interest_rate()
        self.status_bar.showMessage("短期国債金利を取得中...")
    
    def validate_portfolio(self):
        """ポートフォリオを検証"""
        edit_widget = self._get_management_edit_widget()
        if edit_widget and edit_widget.portfolio:
            errors = edit_widget.portfolio.validate()
            if errors:
                QMessageBox.warning(self, "ポートフォリオ検証", 
                                   "以下の問題があります:\n\n" + "\n".join(errors))
            else:
                QMessageBox.information(self, "ポートフォリオ検証", 
                                       "ポートフォリオに問題はありません．")
        else:
            QMessageBox.information(self, "ポートフォリオ検証", 
                                   "ポートフォリオが作成されていません．")
        self.status_bar.showMessage("ポートフォリオを検証しました")
    
    def toggle_dark_mode(self, checked):
        """ダークモードを切り替え"""
        if self.config:
            theme = 'dark' if checked else 'light'
            self.config.apply_color_theme(theme)
        self.status_bar.showMessage("テーマを変更しました")
    
    # ガイド表示
    def show_portfolio_guide(self):
        """ポートフォリオ管理ガイド"""
        guide = """ポートフォリオ管理機能:

• 新規作成: 「新規作成」→名前・説明を入力
• 資産追加: 左側で検索→ダブルクリックで追加
• ウエイト設定: 各資産に%を入力（100%超過でレバレッジ）
• 保存・管理: 「保存」で保存，「管理」タブで既存管理
• 利子率: キャッシュ・借入の利率設定
• 一括操作: 正規化・全削除・均等配分

最適化結果の利用:
• 分析タブで最小分散フロンティア・下方偏差フロンティアを実行
• 分析サマリータブで「ポートフォリオとして保存」ボタンをクリック
• 最適化されたウエイトが管理タブに自動設定されます"""
        QMessageBox.information(self, "ポートフォリオ管理", guide)
    
    def show_weight_guide(self):
        """ウエイト設定ガイド"""
        guide = """ウエイト設定について:

• ウエイト = 投資比率（%）
• 合計 < 100%: 余剰資金をキャッシュ保有
• 合計 = 100%: 資金全額投資
• 合計 > 100%: 借入によるレバレッジ投資

操作：「正規化」で合計100%に調整"""
        QMessageBox.information(self, "ウエイト設定", guide)
    
    def show_leverage_guide(self):
        """レバレッジガイド"""
        guide = """レバレッジとキャッシュ:

• キャッシュ（< 100%）: 余剰資金を利子率で運用
• レバレッジ（> 100%）: 不足資金を利子率で借入

利子率：「取得」で短期国債金利自動取得

表示色：青=正常，黄=低投資，赤=レバレッジ
注意：レバレッジはリスクを高めます"""
        QMessageBox.information(self, "レバレッジ", guide)
    
    def show_about(self):
        """バージョン情報"""
        about = """Portfolio Analyzer v1.0.0

ポートフォリオ最適化分析ツール

主な機能:
• ポートフォリオ作成・管理・比較
• 資産検索・ウエイト設定
• レバレッジ・キャッシュ管理
• リスク・リターン分析
• 最小分散・下方偏差フロンティア分析
• 最適化結果の自動ポートフォリオ化
• 自動利子率取得

© 2025 Portfolio Analytics"""
        QMessageBox.about(self, "About", about)
    
    def _update_status(self):
        """ステータス更新"""
        tab_names = ["個別資産分析モード", "ポートフォリオ比較分析モード", "ポートフォリオ管理モード"]
        current_tab = self.tab_widget.currentIndex()
        if 0 <= current_tab < len(tab_names):
            self.status_bar.showMessage(tab_names[current_tab])