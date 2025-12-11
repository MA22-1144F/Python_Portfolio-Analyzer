"""tile_results.py"""

from PySide6.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QScrollArea, QWidget, QMessageBox
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont

from analysis.analysis_base_widget import AnalysisStyles
from analysis.individual_analysis import PriceSeriesWidget
from analysis.individual_analysis import ReturnRiskAnalysisWidget
from analysis.individual_analysis import CorrelationMatrixWidget
from analysis.individual_analysis import EfficientFrontierWidget
from analysis.individual_analysis import DownsideDeviationFrontierWidget
from analysis.individual_analysis import SecurityMarketLineWidget
from config.app_config import AppConfig


class AnalysisItemWidget(QFrame):
    remove_requested = Signal(object)
    
    # 分析ウィジェットマッピング
    WIDGET_MAPPING = {
        "price_series": PriceSeriesWidget,
        "return_risk_analysis": ReturnRiskAnalysisWidget,
        "correlation_matrix": CorrelationMatrixWidget,
        "efficient_frontier": EfficientFrontierWidget,
        "downside_deviation_frontier": DownsideDeviationFrontierWidget,
        "security_market_line": SecurityMarketLineWidget,
    }
    
    # デフォルトサイズ
    DEFAULT_SIZES = {
        "return_risk_analysis": 800,
        "price_series": 600,
        "correlation_matrix": 600,
        "efficient_frontier": 600,
        "downside_deviation_frontier": 600,
        "security_market_line": 600
    }
    
    def __init__(self, item_type: str, item_name: str, config=None):
        super().__init__()
        self.config = config or AppConfig()
        self.item_type = item_type
        self.item_name = item_name
        self.analysis_widget = None
        self.styles = AnalysisStyles(self.config)
        
        self._setup_widget()
    
    def _setup_widget(self):
        """ウィジェット設定"""
        self.setFrameShape(QFrame.StyledPanel)
        self._apply_sizing()
        self._apply_styles()
        self._setup_ui()
        self._create_analysis_widget()
    
    def _apply_sizing(self):
        """サイズ設定"""
        try:
            size_config = self.config.get(f'analysis.widget_sizes.{self.item_type}')
            if size_config:
                min_height = size_config.get('min_height', 400)
                min_width = size_config.get('min_width', 600)
            else:
                min_height = self.DEFAULT_SIZES.get(self.item_type, 400)
                min_width = 600
        except Exception:
            min_height = self.DEFAULT_SIZES.get(self.item_type, 400)
            min_width = 600
        
        self.setMinimumHeight(min_height)
        self.setMinimumWidth(min_width)
    
    def _apply_styles(self):
        """スタイル適用"""
        style = f"""
            AnalysisItemWidget {{
                border: 1px solid {self.styles.COLORS["border"]};
                border-radius: 6px;
                background-color: {self.styles.COLORS["surface"]};
                margin: 3px;
            }}
            AnalysisItemWidget:hover {{
                border-color: {self.styles.COLORS["primary"]};
                background-color: #404040;
            }}
        """
        self.setStyleSheet(style)
    
    def _setup_ui(self):
        """UIレイアウト設定"""
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        
        # ヘッダー
        header_layout = self._create_header()
        layout.addLayout(header_layout)
        
        # コンテンツエリア
        self.content_layout = QVBoxLayout()
        layout.addLayout(self.content_layout)
        
        self.setLayout(layout)
    
    def _create_header(self):
        """ヘッダー作成"""
        header_layout = QHBoxLayout()
        
        # タイトル
        title_label = QLabel(self.item_name)
        font = QFont()
        font.setPointSize(9)
        font.setBold(True)
        title_label.setFont(font)
        title_label.setStyleSheet(f"color: {self.styles.COLORS['text_primary']}; border: none;")
        header_layout.addWidget(title_label)
        
        header_layout.addStretch()
        
        # 削除ボタン
        remove_button = self._create_remove_button()
        header_layout.addWidget(remove_button)
        
        return header_layout
    
    def _create_remove_button(self):
        """削除ボタン作成"""
        button = QPushButton("×")
        button.setStyleSheet(self.styles.get_button_style_by_type("danger"))
        button.setMaximumSize(18, 18)
        # 小さいボタン用の追加スタイル
        current_style = button.styleSheet()
        button.setStyleSheet(
            current_style + " QPushButton { border-radius: 9px; font-weight: bold; }"
        )
        button.clicked.connect(lambda: self.remove_requested.emit(self))
        button.setToolTip("この分析項目を削除")
        return button
    
    def _create_analysis_widget(self):
        """分析ウィジェット作成"""
        widget_class = self.WIDGET_MAPPING.get(self.item_type)
        
        if widget_class:
            self.analysis_widget = widget_class(self.config)
        else:
            # プレースホルダー
            placeholder = QLabel(f"{self.item_name}の分析結果が\nここに表示されます．")
            placeholder.setAlignment(Qt.AlignCenter)
            placeholder.setStyleSheet(f"""
                QLabel {{
                    color: {self.styles.COLORS["text_secondary"]};
                    font-size: 10px; border: 1px dashed {self.styles.COLORS["border"]};
                    border-radius: 4px; padding: 15px;
                    background-color: {self.styles.COLORS["background"]};
                }}
            """)
            self.analysis_widget = placeholder
        
        self.content_layout.addWidget(self.analysis_widget)
    
    def run_analysis(self, assets, conditions):
        """分析実行"""
        if hasattr(self.analysis_widget, 'analyze'):
            self.analysis_widget.analyze(assets, conditions)
    
    def clear_analysis(self):
        """分析クリア"""
        if hasattr(self.analysis_widget, 'clear_data'):
            self.analysis_widget.clear_data()
    
    def set_price_data_source(self, price_widget):
        """価格データソース設定"""
        if hasattr(self.analysis_widget, 'set_price_data_source'):
            self.analysis_widget.set_price_data_source(price_widget)


class AnalysisResultArea(QScrollArea):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or AppConfig()
        self.styles = AnalysisStyles(self.config)
        self.analysis_items = []
        self.price_data_source = None
        
        self._setup_area()
        self._setup_content()
    
    def _setup_area(self):
        """エリア設定"""
        self.setAcceptDrops(True)
        self.setWidgetResizable(True)
        self.setStyleSheet(f"""
            QScrollArea {{
                border: 1px solid {self.styles.COLORS["border"]};
                border-radius: 4px; background-color: #323232;
            }}
        """)
    
    def _setup_content(self):
        """コンテンツ設定"""
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setAlignment(Qt.AlignTop)
        self.content_layout.setContentsMargins(5, 5, 5, 5)
        self.content_layout.setSpacing(8)
        
        # 空状態ラベル
        self.empty_label = QLabel("分析項目をここにドラッグ＆ドロップしてください．")
        self.empty_label.setAlignment(Qt.AlignCenter)
        self.empty_label.setStyleSheet(f"""
            QLabel {{
                color: {self.styles.COLORS["text_secondary"]}; font-size: 13px;
                background-color: {self.styles.COLORS["background"]};
                border: 2px dashed {self.styles.COLORS["border"]};
                border-radius: 8px; padding: 30px; margin: 10px;
            }}
        """)
        self.content_layout.addWidget(self.empty_label)
        
        self.setWidget(self.content_widget)
    
    def dragEnterEvent(self, event):
        """ドラッグ開始"""
        if event.mimeData().hasText():
            event.acceptProposedAction()
            self._set_drop_style(True)
    
    def dragLeaveEvent(self, event):
        """ドラッグ終了"""
        self._set_drop_style(False)
    
    def dropEvent(self, event):
        """ドロップ処理"""
        self._set_drop_style(False)
        
        if event.mimeData().hasText():
            text_data = event.mimeData().text()
            try:
                item_type, item_name = text_data.split('|', 1)
                self.add_analysis_item(item_type, item_name)
                event.acceptProposedAction()
            except ValueError:
                pass
    
    def _set_drop_style(self, dropping):
        """ドロップ時スタイル設定"""
        if dropping:
            style = f"""
                QScrollArea {{
                    border: 2px solid {self.styles.COLORS["primary"]};
                    border-radius: 4px; background-color: #1e3a52;
                }}
            """
        else:
            style = f"""
                QScrollArea {{
                    border: 1px solid {self.styles.COLORS["border"]};
                    border-radius: 4px; background-color: #323232;
                }}
            """
        self.setStyleSheet(style)
    
    def add_analysis_item(self, item_type: str, item_name: str):
        """分析項目追加"""
        # 重複チェック
        for existing_item in self.analysis_items:
            if existing_item.item_type == item_type:
                QMessageBox.information(
                    self, "重複エラー", 
                    f"'{item_name}' は既に追加されています．"
                )
                return
        
        # 空ラベル非表示
        if self.empty_label.isVisible():
            self.empty_label.hide()
        
        # アイテム作成
        item_widget = AnalysisItemWidget(item_type, item_name, self.config)
        item_widget.remove_requested.connect(self.remove_analysis_item)
        
        # 価格データソース設定
        if self.price_data_source:
            item_widget.set_price_data_source(self.price_data_source)
        
        self.content_layout.addWidget(item_widget)
        self.analysis_items.append(item_widget)
        
        # 価格系列の場合はデータソースとして設定
        if item_type == "price_series":
            self._update_price_data_source(item_widget.analysis_widget)
    
    def remove_analysis_item(self, item_widget: AnalysisItemWidget):
        """分析項目削除"""
        # 価格データソース更新
        if (item_widget.item_type == "price_series" and 
            self.price_data_source == item_widget.analysis_widget):
            self._update_price_data_source(None)
        
        # ウィジェット削除
        self.content_layout.removeWidget(item_widget)
        self.analysis_items.remove(item_widget)
        item_widget.deleteLater()
        
        # 空状態チェック
        if not self.analysis_items:
            self.empty_label.show()
    
    def _update_price_data_source(self, source):
        """価格データソース更新"""
        self.price_data_source = source
        for item in self.analysis_items:
            if item.analysis_widget != source:
                item.set_price_data_source(source)
    
    def clear_all_items(self):
        """全項目削除"""
        for item_widget in self.analysis_items[:]:
            self.remove_analysis_item(item_widget)
    
    def get_analysis_items(self):
        """分析項目リスト取得"""
        return [(item.item_type, item.item_name) for item in self.analysis_items]


class ResultTile(QFrame):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or AppConfig()
        self.styles = AnalysisStyles(self.config)
        self.setFrameShape(QFrame.StyledPanel)
        self._setup_ui()
    
    def _setup_ui(self):
        """UIレイアウト設定"""
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        
        # タイトル
        title_label = self._create_title_label()
        layout.addWidget(title_label)
        
        # コントロール
        control_layout = self._create_controls()
        layout.addLayout(control_layout)
        
        # 結果エリア
        self.result_area = AnalysisResultArea(self.config)
        layout.addWidget(self.result_area)
        
        self.setLayout(layout)
    
    def _create_title_label(self):
        """タイトルラベル作成"""
        label = QLabel("分析結果")
        font = QFont()
        font.setPointSize(11)
        font.setBold(True)
        label.setFont(font)
        label.setStyleSheet(f"color: {self.styles.COLORS['text_primary']}; margin-bottom: 5px;")
        return label
    
    def _create_controls(self):
        """コントロール作成"""
        control_layout = QHBoxLayout()
        
        # 分析開始ボタン
        self.analyze_button = QPushButton("分析開始")
        self.analyze_button.setStyleSheet(
            self.styles.get_button_style(
                font_size="10px", 
                min_width=f"{self.styles.SIZES['small_button_width']}px"
            ) + " font-weight: bold;"
        )
        self.analyze_button.clicked.connect(self._start_analysis)
        control_layout.addWidget(self.analyze_button)
        
        # 全削除ボタン
        self.clear_button = QPushButton("全削除")
        self.clear_button.setStyleSheet(
            self.styles.get_button_style(
                bg_color="#555555", hover_color="#666666",
                min_width=f"{self.styles.SIZES['button_min_width']}px"
            )
        )
        self.clear_button.clicked.connect(self._clear_all_results)
        control_layout.addWidget(self.clear_button)
        
        return control_layout
    
    def _start_analysis(self):
        """分析開始"""
        analysis_items = self.result_area.get_analysis_items()
        
        if not analysis_items:
            QMessageBox.information(
                self, "分析エラー", 
                "分析項目が選択されていません．\n"
                "分析項目タイルから分析項目をドラッグ＆ドロップしてください．"
            )
            return
        
        try:
            analysis_tab = self._get_analysis_tab()
            if not analysis_tab:
                QMessageBox.warning(self, "エラー", "分析条件を取得できませんでした．")
                return
            
            conditions = analysis_tab.get_analysis_conditions()
            assets = analysis_tab.get_selected_assets()
            
            if not assets:
                QMessageBox.warning(
                    self, "分析エラー", 
                    "分析対象資産が選択されていません．\n"
                    "分析対象資産タイルで資産を選択してください．"
                )
                return
            
            self._run_analysis_items(assets, conditions)
            
        except Exception as e:
            import traceback
            print(f"分析実行エラー:\n{traceback.format_exc()}")
            QMessageBox.critical(self, "エラー", f"分析中にエラーが発生しました:\n{str(e)}")
    
    def _run_analysis_items(self, assets, conditions):
        """分析項目実行"""
        # 価格系列を最初に実行
        price_series_widget = None
        other_widgets = []
        
        for item_widget in self.result_area.analysis_items:
            if item_widget.item_type == "price_series":
                price_series_widget = item_widget
            else:
                other_widgets.append(item_widget)
        
        if price_series_widget:
            price_series_widget.run_analysis(assets, conditions)
        
        for item_widget in other_widgets:
            item_widget.run_analysis(assets, conditions)
    
    def _get_analysis_tab(self):
        """分析タブ取得"""
        parent = self.parent()
        search_depth = 0
        max_depth = 10
        
        while parent and search_depth < max_depth:
            if (hasattr(parent, 'get_analysis_conditions') and 
                hasattr(parent, 'get_selected_assets')):
                return parent
            parent = parent.parent()
            search_depth += 1
        
        return None
    
    def _clear_all_results(self):
        """全結果削除"""
        if self.result_area.analysis_items:
            reply = QMessageBox.question(
                self, "確認", "全ての分析項目を削除しますか？",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.result_area.clear_all_items()
    
    def get_selected_analysis_items(self):
        """選択済み分析項目取得"""
        return self.result_area.get_analysis_items()
    
    # パブリックメソッド
    def start_analysis(self):
        """分析開始（外部呼び出し用）"""
        self._start_analysis()
    
    def clear_all_results(self):
        """全結果削除（外部呼び出し用）"""
        self._clear_all_results()