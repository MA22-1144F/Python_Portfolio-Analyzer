"""analysis_base_widget.py"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar, 
    QPushButton, QApplication, QComboBox, QTableWidget, QTabWidget,
    QSpinBox, QDoubleSpinBox, QHeaderView, QSizePolicy
)
from PySide6.QtCore import Qt
from typing import List, Dict, Any
from data.asset_info import AssetInfo
from config.app_config import get_config


class AnalysisStyles:
    """分析ウィジェット統一スタイル管理"""
    
    def __init__(self, config=None):
        
        if config is None:
            config = get_config()

        self.config = config
        
        # 設定から各種パラメータを取得
        self.COLORS = self.config.get_ui_colors()
        self.SIZES = self.config.get_ui_sizes()
        self.BUTTON_TYPES = self.config.get_button_types()
        self.LAYOUT = self.config.get_layout_settings()
    
    def get_progress_bar_style(self) -> str:
        return f"""
            QProgressBar {{
                border: 1px solid {self.COLORS["border"]};
                border-radius: 4px;
                background-color: {self.COLORS["background"]};
                color: {self.COLORS["text_primary"]};
                text-align: center;
                font-size: 10px;
                max-height: {self.SIZES["progress_bar_height"]}px;
            }}
            QProgressBar::chunk {{
                background-color: {self.COLORS["primary"]};
                border-radius: 3px;
            }}
        """
    
    def get_status_label_style(self) -> str:
        return f"""
            color: {self.COLORS["text_secondary"]};
            font-size: 10px;
            max-height: {self.SIZES["status_label_height"]}px;
        """
    
    def get_quality_info_style(self) -> str:
        return f"""
            color: {self.COLORS["text_accent"]};
            font-size: 9px;
            max-height: {self.SIZES["quality_info_height"]}px;
        """
    
    def get_empty_state_style(self) -> str:
        return f"""
            QLabel {{
                color: {self.COLORS["text_secondary"]};
                font-size: 11px;
                background-color: {self.COLORS["background"]};
                border: 1px dashed {self.COLORS["border"]};
                border-radius: 8px;
                padding: 8px;
                margin: 5px;
                min-height: {self.SIZES["empty_state_min_height"]}px;
                max-height: {self.SIZES["empty_state_max_height"]}px;
            }}
        """
    
    def get_button_style_by_type(self, button_type: str = "primary") -> str:
        """ボタン種類別のスタイルを取得"""
        # デフォルト設定を定義
        default_config = {
            "bg_color": self.COLORS.get("button_background", "#007ACC"),
            "text_color": self.COLORS.get("text_primary", "#FFFFFF"),
            "hover_color": self.COLORS.get("button_hover", "#005A9E"),
            "width": self.SIZES.get("button_width", 120)
        }
        
        # BUTTON_TYPESから設定を取得、なければデフォルトを使用
        config = self.BUTTON_TYPES.get(button_type, default_config)
        
        # configが空辞書の場合もデフォルトを使用
        if not config:
            config = default_config
        
        return f"""
            QPushButton {{
                background-color: {config.get("bg_color", "#007ACC")};
                color: {config.get("text_color", "#FFFFFF")};
                border: 1px solid {config.get("bg_color", "#007ACC")};
                border-radius: {self.SIZES.get("button_border_radius", 4)}px;
                padding: 4px 8px;
                min-width: {config.get("width", 120)}px;
                max-width: {config.get("width", 120) + 50}px;
                height: {self.SIZES.get("button_height", 28)}px;
                font-size: {self.SIZES.get("button_font_size", 12)}px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background-color: {config.get("hover_color", "#005A9E")};
                border-color: {config.get("hover_color", "#005A9E")};
            }}
            QPushButton:pressed {{
                background-color: {config.get("bg_color", "#007ACC")};
                border-color: {config.get("bg_color", "#007ACC")};
            }}
            QPushButton:disabled {{
                background-color: {self.COLORS.get("disabled_background", "#2D2D2D")};
                color: {self.COLORS.get("text_disabled", "#666666")};
                border-color: {self.COLORS.get("disabled_background", "#2D2D2D")};
            }}
        """
    
    def get_button_style(self, bg_color: str = None, hover_color: str = None, 
                        font_size: str = "9px", min_width: str = None) -> str:
        """互換性のための旧ボタンスタイル"""
        bg = bg_color or self.COLORS["primary"]
        hover = hover_color or self.COLORS["primary_hover"]
        width = min_width or f"{self.SIZES['button_min_width']}px"
        
        return f"""
            QPushButton {{
                background-color: {bg};
                color: {self.COLORS["text_primary"]};
                border: 1px solid {bg};
                border-radius: 4px;
                padding: 4px 8px;
                min-width: {width};
                font-size: {font_size};
            }}
            QPushButton:hover {{
                background-color: {hover};
            }}
            QPushButton:disabled {{
                background-color: {self.COLORS["disabled_background"]};
                color: {self.COLORS["text_disabled"]};
                border-color: {self.COLORS["disabled_background"]};
            }}
        """
    
    def get_table_style(self) -> str:
        return f"""
            QTableWidget {{
                border: 1px solid {self.COLORS["border"]};
                border-radius: 4px;
                background-color: {self.COLORS["background"]};
                color: {self.COLORS["text_primary"]};
                gridline-color: {self.COLORS["grid"]};
                selection-background-color: {self.COLORS["primary"]};
                font-size: 10px;
            }}
            QTableWidget::item {{
                padding: 2px 4px;
                border-bottom: 1px solid {self.COLORS["grid"]};
                border-right: 1px solid {self.COLORS["grid"]};
            }}
            QTableWidget::item:selected {{
                background-color: {self.COLORS["primary"]};
            }}
            QHeaderView::section {{
                background-color: {self.COLORS["surface"]};
                color: {self.COLORS["text_primary"]};
                border: 1px solid {self.COLORS["border"]};
                padding: 2px 4px;
                font-weight: bold;
                font-size: 9px;
            }}
        """
    
    def get_combo_style(self, min_width: str = "100px") -> str:
        return f"""
            QComboBox {{
                background-color: {self.COLORS["surface"]};
                color: {self.COLORS["text_primary"]};
                border: 1px solid {self.COLORS["border"]};
                border-radius: 4px;
                padding: 2px 8px;
                min-width: {min_width};
                font-size: 10px;
            }}
            QComboBox::drop-down {{ border: none; }}
            QComboBox::down-arrow {{
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 4px solid {self.COLORS["text_primary"]};
            }}
            QComboBox QAbstractItemView {{
                background-color: {self.COLORS["surface"]};
                color: {self.COLORS["text_primary"]};
                selection-background-color: {self.COLORS["primary"]};
            }}
        """
    
    def get_tab_style(self) -> str:
        return f"""
            QTabWidget::pane {{
                border: 1px solid {self.COLORS["border"]};
                background-color: {self.COLORS["background"]};
            }}
            QTabBar::tab {{
                background-color: {self.COLORS["surface"]};
                color: {self.COLORS["text_primary"]};
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }}
            QTabBar::tab:selected {{
                background-color: {self.COLORS["primary"]};
            }}
            QTabBar::tab:hover {{
                background-color: {self.COLORS["surface_hover"]};
            }}
        """
    
    def get_spinbox_style(self, min_width: str = "80px") -> str:
        return f"""
            QSpinBox, QDoubleSpinBox {{
                background-color: {self.COLORS["surface"]};
                color: {self.COLORS["text_primary"]};
                border: 1px solid {self.COLORS["border"]};
                border-radius: 4px;
                padding: 2px 8px;
                min-width: {min_width};
                font-size: 10px;
            }}
        """


class AnalysisBaseWidget(QWidget):
    """分析ウィジェット共通ベースクラス"""
    def __init__(self, config=None):
        super().__init__()
        self.price_data_source = None

        # 設定を取得
        if config is None:
            from config.app_config import AppConfig
            config = AppConfig()
        self.config = config
        
        self.styles = AnalysisStyles(config)
        self.setup_base_ui()
        
    def convert_risk_free_rate_to_span(self, annual_rate: float, span: str) -> float:
        """年率の無リスク利子率をスパンに応じて変換"""
        try:
            # 設定から変換係数を取得
            span_factors_config = self.config.get('analysis.time_conversion_factors.span_factors', {})
            time_factors = self.config.get('analysis.time_conversion_factors', {})
            
            # スパンに対応する設定キーまたは数値を取得
            factor_key = span_factors_config.get(span)
            
            if factor_key == 1:  # 年次の場合
                factor = 1
            elif isinstance(factor_key, str):  # 設定キーを参照する場合
                denominator = time_factors.get(factor_key, 365)
                factor = 1 / denominator
            else:
                # フォールバック：従来の値を使用
                fallback_factors = {
                    '日次': 1/365,
                    '週次': 1/52,
                    '月次': 1/12,
                    '年次': 1
                }
                factor = fallback_factors.get(span, 1/365)
            
            span_rate = annual_rate * factor
            
            if hasattr(self, 'logger'):
                self.logger.info(f"無リスク利子率変換: {annual_rate:.4f}(年率) -> {span_rate:.6f}({span})")
            
            return span_rate
            
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"無リスク利子率変換エラー: {e}")
            return 0.0
    
    def setup_base_ui(self):
        """共通UIレイアウトの設定"""
        self.main_layout = QVBoxLayout()
        margins = self.styles.LAYOUT["main_margins"]
        self.main_layout.setContentsMargins(*margins)
        self.main_layout.setSpacing(self.styles.LAYOUT["main_spacing"])
        
        # ヘッダーエリア
        self.header_layout = QHBoxLayout()
        self.setup_header_content()
        self.main_layout.addLayout(self.header_layout)
        
        # 共通プログレスエリア
        self.setup_progress_area()
        
        # メインコンテンツエリア（サブクラスで設定）
        self.content_layout = QVBoxLayout()
        self.content_layout.setSpacing(self.styles.LAYOUT["content_spacing"])
        self.main_layout.addLayout(self.content_layout)
        
        # 共通空状態エリア
        self.setup_empty_state()
        
        self.setLayout(self.main_layout)
    
    def setup_header_content(self):
        """ヘッダーコンテンツの設定"""
        self.header_layout.addStretch()
    
    def setup_progress_area(self):
        """共通プログレス表示エリア"""
        # 進捗バー
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximumHeight(self.styles.SIZES["progress_bar_height"])
        self.progress_bar.setStyleSheet(self.styles.get_progress_bar_style())
        self.main_layout.addWidget(self.progress_bar)
        
        # ステータスラベル
        self.status_label = QLabel("")
        self.status_label.setStyleSheet(self.styles.get_status_label_style())
        self.status_label.setVisible(False)
        self.status_label.setMaximumHeight(self.styles.SIZES["status_label_height"])
        self.main_layout.addWidget(self.status_label)
        
        # 品質情報ラベル
        self.quality_info_label = QLabel("")
        self.quality_info_label.setStyleSheet(self.styles.get_quality_info_style())
        self.quality_info_label.setVisible(False)
        self.quality_info_label.setWordWrap(True)
        self.quality_info_label.setMaximumHeight(self.styles.SIZES["quality_info_height"])
        self.main_layout.addWidget(self.quality_info_label)
    
    def setup_empty_state(self):
        """共通空状態表示"""
        self.empty_label = QLabel(self.get_empty_message())
        self.empty_label.setAlignment(Qt.AlignCenter)
        
        # 重要：ワードラップを有効にする
        self.empty_label.setWordWrap(True)
        
        # サイズポリシーを設定してレイアウトに適応させる
        self.empty_label.setSizePolicy(
            QSizePolicy.Expanding, 
            QSizePolicy.Minimum
        )
        
        # 最小・最大高さを設定
        self.empty_label.setMinimumHeight(self.styles.SIZES["empty_state_min_height"])
        self.empty_label.setMaximumHeight(self.styles.SIZES["empty_state_max_height"])
        
        self.empty_label.setStyleSheet(self.styles.get_empty_state_style())
        self.main_layout.addWidget(self.empty_label)
    
    def get_empty_message(self) -> str:
        """空状態メッセージ"""
        return "分析結果が表示されていません．\n価格時系列データの取得完了後，表示されます．"
    
    # 共通UI要素作成メソッド
    def create_button(self, text: str, button_type: str = "primary") -> QPushButton:
        """種類別スタイルのボタンを作成"""
        button = QPushButton(text)
        button.setStyleSheet(self.styles.get_button_style_by_type(button_type))
        return button
    
    def create_action_button(self, text: str, bg_color: str = None, 
                           hover_color: str = None, min_width: str = None) -> QPushButton:
        """旧式ボタン作成"""
        button = QPushButton(text)
        button.setStyleSheet(self.styles.get_button_style(bg_color, hover_color, min_width=min_width))
        return button
    
    def create_combo_box(self, items: List[str] = None, min_width: str = "100px") -> QComboBox:
        """統一スタイルのコンボボックスを作成"""
        combo = QComboBox()
        if items:
            combo.addItems(items)
        combo.setStyleSheet(self.styles.get_combo_style(min_width))
        return combo
    
    def create_table_widget(self) -> QTableWidget:
        """統一スタイルのテーブルを作成"""
        table = QTableWidget()
        table.setStyleSheet(self.styles.get_table_style())
        table.setSortingEnabled(False)
        table.setAlternatingRowColors(True)
        
        # ヘッダーの設定
        horizontal_header = table.horizontalHeader()
        horizontal_header.setSectionResizeMode(QHeaderView.ResizeToContents)
        
        vertical_header = table.verticalHeader()
        vertical_header.setSectionResizeMode(QHeaderView.ResizeToContents)
        
        return table
    
    def create_tab_widget(self) -> QTabWidget:
        """統一スタイルのタブウィジェットを作成"""
        tab_widget = QTabWidget()
        tab_widget.setStyleSheet(self.styles.get_tab_style())
        return tab_widget
    
    def create_spinbox(self, is_double: bool = False, min_width: str = "80px") -> QSpinBox:
        """統一スタイルのスピンボックスを作成"""
        if is_double:
            spinbox = QDoubleSpinBox()
        else:
            spinbox = QSpinBox()
        spinbox.setStyleSheet(self.styles.get_spinbox_style(min_width))
        return spinbox
    
    # 設定取得メソッド
    def get_analysis_widget_size(self, item_type: str) -> Dict[str, int]:
        """分析ウィジェットサイズを取得"""
        return self.config.get_analysis_widget_size(item_type)
    
    def get_min_data_points(self, span: str) -> int:
        """最小データポイント数を取得"""
        return self.config.get_min_data_points(span)
    
    def get_min_coverage_ratio(self) -> float:
        """最小カバレッジ率を取得"""
        return self.config.get_min_coverage_ratio()
    
    def convert_metric_with_config(self, value: float, metric_type: str, from_span: str, to_span: str) -> float:
        """設定を使用した指標換算"""
        if value is None or from_span == to_span:
            return value
        
        # リターン系指標（時間で調整）
        if metric_type in ['expected_return', 'min_return', 'max_return', 'excess_return']:
            factor = self.config.get_time_conversion_factor(from_span, to_span)
        # リスク系指標（時間の平方根で調整）
        elif metric_type in ['standard_deviation', 'downside_deviation', 'var_95', 'var_99', 'cvar_95', 'cvar_99']:
            factor = self.config.get_risk_conversion_factor(from_span, to_span)
        # 分散（時間で調整）
        elif metric_type == 'variance':
            factor = self.config.get_time_conversion_factor(from_span, to_span)
        # 効率性指標（時間の平方根で調整）
        elif metric_type in ['sharpe_ratio', 'sortino_ratio']:
            factor = self.config.get_risk_conversion_factor(from_span, to_span)
        else:
            # その他の指標は変換しない
            return value
        
        return value * factor
    
    # 共通制御メソッド
    def show_progress(self, visible: bool = True):
        """プログレス表示制御"""
        self.progress_bar.setVisible(visible)
        self.status_label.setVisible(visible)
    
    def update_progress(self, value: int, message: str = ""):
        """進捗更新"""
        self.progress_bar.setValue(value)
        if message:
            self.status_label.setText(message)
        QApplication.processEvents()
    
    def show_quality_info(self, message: str):
        """品質情報表示"""
        self.quality_info_label.setText(message)
        self.quality_info_label.setVisible(True)
    
    def hide_quality_info(self):
        """品質情報非表示"""
        self.quality_info_label.setVisible(False)
    
    def show_empty_state(self, show: bool = True):
        """空状態表示制御"""
        self.empty_label.setVisible(show)
    
    def show_main_content(self, show: bool = True):
        """メインコンテンツ表示制御（空状態と連動）"""
        self.show_empty_state(not show)
    
    # 抽象メソッド（サブクラスで実装）
    def analyze(self, assets: List[AssetInfo], conditions: Dict[str, Any]):
        """分析実行（サブクラスで実装）"""
        raise NotImplementedError("analyze method must be implemented by subclass")
    
    def clear_data(self):
        """データクリア（サブクラスで実装）"""
        raise NotImplementedError("clear_data method must be implemented by subclass")
    
    def set_price_data_source(self, price_widget):
        """価格データソース設定"""
        self.price_data_source = price_widget