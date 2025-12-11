"""tab_compare.py"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QGroupBox,
    QLabel, QPushButton, QTableWidget, QTableWidgetItem, QLineEdit,
    QComboBox, QRadioButton, QButtonGroup, QDateEdit, QDoubleSpinBox,
    QTabWidget, QHeaderView, QMessageBox, QCheckBox, QFormLayout, QApplication, QProgressDialog
)
from PySide6.QtCore import Qt, QDate, QTimer
from PySide6.QtGui import QColor
from typing import List, Dict, Optional

import logging
import numpy as np
import pandas as pd
import tempfile
import webbrowser
import os
from datetime import timedelta

from data.portfolio import Portfolio, PortfolioManager
from analysis.analysis_base_widget import AnalysisStyles
from config.app_config import get_config
from utils.ui_styles import get_checkbox_style
from utils.common_widgets import InterestRateThread

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    import matplotlib.figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from analysis.portfolio_comparison import (
    summary_stats,
    scatter_plot,
    heatmap,
    radar_chart,
    price_evolution,
    actual_vs_theoretical,
    frontier_comparison
)

class MatplotlibCanvas(FigureCanvas):
    """Matplotlib canvas for Qt integration"""
    
    def __init__(self, parent=None, width=10, height=8, dpi=100):
        self.figure = matplotlib.figure.Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.figure)
        self.setParent(parent)
        
        self.logger = logging.getLogger(__name__)

        # ダークテーマに合わせた背景色
        self.figure.patch.set_facecolor('#2b2b2b')
    
    def clear_plot(self):
        """プロットを完全にクリア"""
        self.figure.clear()
        self.draw()
    
    def display_figure(self, fig):
        """外部のfigureを表示"""
        from matplotlib.patches import Rectangle, Polygon
        from matplotlib.collections import PathCollection
        import matplotlib.pyplot as plt
        
        self.figure.clear()
        
        # 元のfigureの全axesを取得
        src_axes = fig.get_axes()
        
        if not src_axes:
            self.logger.warning("表示するaxesがありません")
            return
        
        # axesの合計widthを計算
        total_width = sum(ax.get_position().width for ax in src_axes)
        scale_factor = 1.0
        
        # 合計widthが0.95未満の場合，スケーリングして画面いっぱいに
        if total_width < 0.95:
            scale_factor = 0.95 / total_width
            self.logger.info(f"Scaling axes by {scale_factor:.2f} to fill space (total width was {total_width:.2f})")
        
        # 全てのaxesを位置情報を保持してコピー
        for src_ax in src_axes:
            # axesの位置情報を取得
            bbox = src_ax.get_position()
            
            # スケーリング適用
            if scale_factor != 1.0:
                new_width = bbox.width * scale_factor
                new_x0 = bbox.x0 * scale_factor
            else:
                new_width = bbox.width
                new_x0 = bbox.x0
            
            # projectionを確認
            projection = None
            if hasattr(src_ax, 'name') and src_ax.name == 'polar':
                projection = 'polar'
            
            # 新しいaxesをスケーリングした位置に作成
            dest_ax = self.figure.add_axes([new_x0, bbox.y0, new_width, bbox.height], 
                                        projection=projection)
            
            # 背景色設定
            dest_ax.set_facecolor(src_ax.get_facecolor())
            
            # カラーバーaxesかどうかをチェック
            is_colorbar = (bbox.width < 0.1)
            
            if is_colorbar:
                # カラーバーの場合
                dest_ax.set_ylabel(src_ax.get_ylabel(), color='white', fontsize=10)
                dest_ax.tick_params(colors='white')
                dest_ax.set_ylim(src_ax.get_ylim())
                try:
                    dest_ax.set_yticks(src_ax.get_yticks())
                    labels = [t.get_text() for t in src_ax.get_yticklabels()]
                    if labels:
                        dest_ax.set_yticklabels(labels, color='white', fontsize=9)
                except (AttributeError, ValueError, TypeError) as e:
                    logging.debug(f"Failed to copy y-axis labels: {e}")
                    pass
                
                # カラーバーのimageをコピー
                for image in src_ax.get_images():
                    try:
                        array = image.get_array()
                        extent = image.get_extent()
                        cmap = image.get_cmap()
                        alpha = image.get_alpha() if image.get_alpha() is not None else 1.0
                        dest_ax.imshow(array, cmap=cmap, aspect='auto', extent=extent, alpha=alpha)
                    except (AttributeError, ValueError, TypeError) as e:
                        logging.debug(f"Failed to copy image: {e}")
                        pass
                continue
            
            # メインaxesの処理
            # imagesをコピー
            for image in src_ax.get_images():
                try:
                    array = image.get_array()
                    extent = image.get_extent()
                    cmap = image.get_cmap()
                    alpha = image.get_alpha() if image.get_alpha() is not None else 1.0
                    dest_ax.imshow(array, cmap=cmap, aspect='auto', extent=extent, alpha=alpha)
                except Exception as e:
                    self.logger.warning(f"Image copy error: {e}")
            
            # 線をコピー
            for line in src_ax.get_lines():
                dest_ax.plot(line.get_xdata(), line.get_ydata(),
                        color=line.get_color(),
                        linewidth=line.get_linewidth(),
                        linestyle=line.get_linestyle(),
                        marker=line.get_marker(),
                        label=line.get_label())
            
            # collectionsをコピー
            for collection in src_ax.collections:
                try:
                    if isinstance(collection, PathCollection):
                        offsets = collection.get_offsets()
                        sizes = collection.get_sizes()
                        facecolors = collection.get_facecolors()
                        edgecolors = collection.get_edgecolors()
                        linewidths = collection.get_linewidths()
                        alpha = collection.get_alpha()
                        dest_ax.scatter(offsets[:, 0], offsets[:, 1], s=sizes, c=facecolors,
                                    edgecolors=edgecolors, linewidths=linewidths, alpha=alpha)
                    else:
                        paths = collection.get_paths() if hasattr(collection, 'get_paths') else None
                        if paths:
                            for path in paths:
                                vertices = path.vertices
                                if len(vertices) > 0:
                                    dest_ax.fill(vertices[:, 0], vertices[:, 1],
                                            facecolor=collection.get_facecolor()[0] if len(collection.get_facecolor()) > 0 else 'blue',
                                            alpha=collection.get_alpha() or 0.15,
                                            edgecolor=collection.get_edgecolor()[0] if len(collection.get_edgecolor()) > 0 else 'none')
                except Exception as e:
                    self.logger.warning(f"Collection copy error: {e}")
            
            # テキストをコピー
            for text in src_ax.texts:
                try:
                    dest_ax.text(text.get_position()[0], text.get_position()[1], text.get_text(),
                            color=text.get_color(), fontsize=text.get_fontsize(),
                            ha=text.get_ha(), va=text.get_va(), weight=text.get_weight())
                except (AttributeError, ValueError, TypeError) as e:
                    logging.debug(f"Failed to copy text: {e}")
                    pass
            
            # パッチをコピー
            for patch in src_ax.patches:
                try:
                    if isinstance(patch, Rectangle):
                        dest_ax.add_patch(Rectangle(patch.get_xy(), patch.get_width(), patch.get_height(),
                                                facecolor=patch.get_facecolor(), edgecolor=patch.get_edgecolor(),
                                                linewidth=patch.get_linewidth(), alpha=patch.get_alpha()))
                    elif isinstance(patch, Polygon):
                        dest_ax.add_patch(Polygon(patch.get_xy(), facecolor=patch.get_facecolor(),
                                                edgecolor=patch.get_edgecolor(), linewidth=patch.get_linewidth(),
                                                alpha=patch.get_alpha()))
                except (AttributeError, ValueError, TypeError) as e:
                    logging.debug(f"Failed to copy patch: {e}")
                    pass
            
            # タイトルとラベル
            dest_ax.set_title(src_ax.get_title(), color='white', fontsize=12, fontweight='bold')
            if projection != 'polar':
                dest_ax.set_xlabel(src_ax.get_xlabel(), color='white', fontsize=10)
                dest_ax.set_ylabel(src_ax.get_ylabel(), color='white', fontsize=10)
                try:
                    dest_ax.set_xlim(src_ax.get_xlim())
                    dest_ax.set_ylim(src_ax.get_ylim())
                except (AttributeError, ValueError, TypeError) as e:
                    logging.debug(f"Failed to set axis limits: {e}")
                    pass
            
            dest_ax.tick_params(colors='white')
            
            if projection == 'polar':
                try:
                    dest_ax.set_xticks(src_ax.get_xticks())
                    dest_ax.set_xticklabels([t.get_text() for t in src_ax.get_xticklabels()], color='white', fontsize=11)
                    dest_ax.set_ylim(src_ax.get_ylim())
                    dest_ax.set_yticks(src_ax.get_yticks())
                    dest_ax.set_yticklabels([t.get_text() for t in src_ax.get_yticklabels()], color='white', fontsize=9, alpha=0.7)
                except (AttributeError, ValueError, TypeError) as e:
                    logging.debug(f"Failed to set polar ticks: {e}")
                    pass
                dest_ax.grid(True, color='#444444', linestyle='--', linewidth=0.5, alpha=0.5)
            else:
                dest_ax.grid(True, alpha=0.2, color='#444444')
                if src_ax.get_xticklabels():
                    try:
                        dest_ax.set_xticks(src_ax.get_xticks())
                        labels = [t.get_text() for t in src_ax.get_xticklabels()]
                        if labels:
                            dest_ax.set_xticklabels(labels, rotation=0, color='white', fontsize=9)
                    except (AttributeError, ValueError, TypeError) as e:
                        logging.debug(f"Failed to set x-axis labels: {e}")
                        pass
            
            # 凡例
            legend = src_ax.get_legend()
            if legend:
                handles, labels = src_ax.get_legend_handles_labels()
                if handles:
                    if projection == 'polar':
                        dest_ax.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.3, 1.1),
                                    facecolor='#2b2b2b', edgecolor='white', labelcolor='white', fontsize=10)
                    else:
                        dest_ax.legend(handles, labels, facecolor='#2b2b2b', edgecolor='white',
                                    labelcolor='white', fontsize=9)
        
        # 全体タイトル
        if hasattr(fig, '_suptitle') and fig._suptitle:
            self.figure.suptitle(fig._suptitle.get_text(), color='white', fontsize=14, fontweight='bold', y=0.995)
        
        self.draw()

class CompareTab(QWidget):
    """ポートフォリオ比較分析タブ"""
    
    def __init__(self, config=None):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.config = config or get_config()
        self.styles = AnalysisStyles(self.config)
        self.portfolio_manager = PortfolioManager()
        
        # データ
        self.available_portfolios: List[tuple] = []  # (file_name, name, created_at)
        self.selected_portfolios: Dict[str, Portfolio] = {}  # {file_name: Portfolio}
        self.analysis_results: Optional[Dict] = None
        self.temp_html_files = []  # Plotlyの一時HTMLファイル
        
        self._setup_ui()
        self._load_portfolios()
        
        # 初期利子率取得
        self._fetch_initial_interest_rate()
    
    def _setup_ui(self):
        """UI構築"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # メインスプリッター (左右分割)
        self.main_splitter = QSplitter(Qt.Horizontal)
        
        # 左側: 選択・設定パネル
        left_panel = self._create_left_panel()
        self.main_splitter.addWidget(left_panel)
        
        # 右側: 分析結果表示
        right_panel = self._create_right_panel()
        self.main_splitter.addWidget(right_panel)
        
        # 初期スプリッター比率 (30:70)
        self.main_splitter.setSizes([300, 700])
        self.main_splitter.setCollapsible(0, False)
        self.main_splitter.setCollapsible(1, False)
        
        layout.addWidget(self.main_splitter)
    
    def _create_left_panel(self) -> QWidget:
        """左側パネル作成 (選択・設定)"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        # 上下スプリッター
        self.left_splitter = QSplitter(Qt.Vertical)
        
        # 上部: 分析設定セクション
        settings_group = self._create_settings_section()
        self.left_splitter.addWidget(settings_group)

        # 下部: ポートフォリオ選択セクション
        selection_group = self._create_selection_section()
        self.left_splitter.addWidget(selection_group)
        
        # 初期比率 (40:60)
        self.left_splitter.setSizes([240, 360])
        self.left_splitter.setCollapsible(0, False)
        self.left_splitter.setCollapsible(1, False)
        
        layout.addWidget(self.left_splitter)
        
        return panel
    
    def _create_selection_section(self) -> QGroupBox:
        """ポートフォリオ選択セクション"""
        group = QGroupBox("ポートフォリオ選択")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(5)
        
        # 検索バー
        search_layout = QHBoxLayout()
        search_label = QLabel("検索:")
        search_label.setFixedWidth(40)
        search_layout.addWidget(search_label)
        
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("名前で検索...")
        self.search_input.textChanged.connect(self._filter_portfolios)
        search_layout.addWidget(self.search_input)
        layout.addLayout(search_layout)
        
        # 全選択/全解除/更新ボタン
        button_layout = QHBoxLayout()
        self.select_all_button = QPushButton("全選択")
        self.select_all_button.setStyleSheet(self.styles.get_button_style_by_type("secondary"))
        self.select_all_button.clicked.connect(self._select_all_portfolios)
        button_layout.addWidget(self.select_all_button)
        
        self.deselect_all_button = QPushButton("全解除")
        self.deselect_all_button.setStyleSheet(self.styles.get_button_style_by_type("secondary"))
        self.deselect_all_button.clicked.connect(self._deselect_all_portfolios)
        button_layout.addWidget(self.deselect_all_button)
        
        self.refresh_button = QPushButton("更新")
        self.refresh_button.setStyleSheet(self.styles.get_button_style_by_type("secondary"))
        self.refresh_button.clicked.connect(self._refresh_portfolios)
        self.refresh_button.setToolTip("ポートフォリオリストを更新")
        button_layout.addWidget(self.refresh_button)

        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        # ソートコンボボックス
        sort_layout = QHBoxLayout()
        sort_label = QLabel("並び順:")
        sort_label.setFixedWidth(60)
        sort_layout.addWidget(sort_label)
        
        self.sort_combo = QComboBox()
        self.sort_combo.addItems(["作成日時(新)", "作成日時(古)", "名前(昇順)", "名前(降順)"])
        self.sort_combo.currentTextChanged.connect(self._sort_portfolios)
        sort_layout.addWidget(self.sort_combo)
        layout.addLayout(sort_layout)
        
        # ポートフォリオ一覧テーブル
        self.portfolio_table = QTableWidget()
        self.portfolio_table.setColumnCount(4)
        self.portfolio_table.setHorizontalHeaderLabels(["選択", "名前", "作成日時", "資産数"])
        
        # チェックボックスのスタイル
        colors = self.config.get_ui_colors()
        self.portfolio_table.setStyleSheet(f"""
            QTableWidget {{
                background-color: {colors.get('background', '#2b2b2b')};
                color: {colors.get('text_primary', '#ffffff')};
                border: 1px solid {colors.get('border', '#555555')};
                gridline-color: {colors.get('grid_line', '#444444')};
            }}
            QTableWidget::item {{
                padding: 2px;
            }}
            QTableWidget::item:selected {{
                background-color: {colors.get('primary', '#0078d4')};
            }}
            QHeaderView::section {{
                background-color: {colors.get('surface', '#3c3c3c')};
                color: {colors.get('text_primary', '#ffffff')};
                border: 1px solid {colors.get('border', '#555555')};
                padding: 4px;
            }}
        """)
        
        # カラム幅設定
        header = self.portfolio_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        
        self.portfolio_table.itemChanged.connect(self._on_selection_changed)
        layout.addWidget(self.portfolio_table)
        
        # 選択カウント表示
        self.selection_label = QLabel("選択: 0 個")
        self.selection_label.setStyleSheet("font-size: 10px; color: #888;")
        layout.addWidget(self.selection_label)
        
        return group
    
    def _create_settings_section(self) -> QGroupBox:
        """分析設定セクション"""
        group = QGroupBox("分析設定")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        
        # フォームレイアウト
        form_layout = QFormLayout()
        form_layout.setSpacing(6)
        
        # 分析期間
        today = QDate.currentDate()
        one_year_ago = today.addYears(-1)
        
        self.start_date_edit = QDateEdit()
        self.start_date_edit.setDate(one_year_ago)
        self.start_date_edit.setCalendarPopup(True)
        self.start_date_edit.setDisplayFormat("yyyy-MM-dd")
        form_layout.addRow("開始日:", self.start_date_edit)
        
        self.end_date_edit = QDateEdit()
        self.end_date_edit.setDate(today)
        self.end_date_edit.setCalendarPopup(True)
        self.end_date_edit.setDisplayFormat("yyyy-MM-dd")
        form_layout.addRow("終了日:", self.end_date_edit)
        
        # データスパン
        self.span_combo = QComboBox()
        self.span_combo.addItems(["日次", "週次", "月次"])
        form_layout.addRow("データスパン:", self.span_combo)
        
        # 無リスク利子率
        rate_layout = QHBoxLayout()
        self.interest_rate_spin = self._create_double_spinbox(" %", 3, (0.0, 20.0), 0.1, 0.5)
        rate_layout.addWidget(self.interest_rate_spin)
        
        self.fetch_rate_button = QPushButton("取得")
        self.fetch_rate_button.setMaximumWidth(50)
        self.fetch_rate_button.setStyleSheet(self.styles.get_button_style_by_type("secondary"))
        self.fetch_rate_button.clicked.connect(self._fetch_interest_rate)
        rate_layout.addWidget(self.fetch_rate_button)
        
        form_layout.addRow("利子率:", rate_layout)
        
        self.rate_status_label = QLabel("")
        self.rate_status_label.setStyleSheet("font-size: 9px; color: #888;")
        form_layout.addRow("", self.rate_status_label)
        
        # 市場ポートフォリオ
        self.market_combo = QComboBox()
        self.market_combo.addItems([
            "なし",
            "日経225 (^N225)",
            "S&P500 (^GSPC)",
            "TOPIX連動ETF (1306.T)"
        ])
        form_layout.addRow("市場:", self.market_combo)
        
        # 比較ベース
        self.basis_group = QButtonGroup()
        basis_widget = QWidget()
        basis_layout = QVBoxLayout(basis_widget)
        basis_layout.setContentsMargins(0, 0, 0, 0)
        basis_layout.setSpacing(3)
        
        # ラジオボタンのスタイル
        radio_style = """
            QRadioButton {
                color: #ffffff;
                spacing: 5px;
                background-color: transparent;
            }
            QRadioButton::indicator {
                width: 15px;
                height: 15px;
                border-radius: 7px;
                border: 2px solid #ffffff;
                background-color: #2b2b2b;
            }
            QRadioButton::indicator:checked {
                background-color: #2196F3;
                border: 2px solid #2196F3;
            }
            QRadioButton::indicator:hover {
                border: 2px solid #2196F3;
            }
        """
        for basis in ["実測値", "理論値 (CAPM)"]:
            radio = QRadioButton(basis)
            radio.setStyleSheet(radio_style)
            self.basis_group.addButton(radio)
            basis_layout.addWidget(radio)
            if basis == "実測値":
                radio.setChecked(True)
        
        form_layout.addRow("比較ベース:", basis_widget)
        
        layout.addLayout(form_layout)
        
        # 分析実行ボタンとクリアボタン
        button_layout = QHBoxLayout()
        
        self.execute_button = QPushButton("分析実行")
        self.execute_button.setStyleSheet(self.styles.get_button_style_by_type("primary"))
        self.execute_button.clicked.connect(self._execute_analysis)
        button_layout.addWidget(self.execute_button)
        
        # 分析結果クリアボタン
        self.clear_button = QPushButton("結果クリア")
        self.clear_button.setStyleSheet(self.styles.get_button_style_by_type("secondary"))
        self.clear_button.clicked.connect(self._clear_analysis_results)
        self.clear_button.setEnabled(False)  # 初期状態は無効
        button_layout.addWidget(self.clear_button)
        
        layout.addLayout(button_layout)
        
        return group
    
    def _create_double_spinbox(self, suffix: str, decimals: int, range_vals: tuple, 
                               step: float, default: float) -> QDoubleSpinBox:
        """DoubleSpinBox作成ヘルパー"""
        spinbox = QDoubleSpinBox()
        spinbox.setSuffix(suffix)
        spinbox.setDecimals(decimals)
        spinbox.setRange(range_vals[0], range_vals[1])
        spinbox.setSingleStep(step)
        spinbox.setValue(default)
        return spinbox
    
    def _create_right_panel(self) -> QWidget:
        """右側パネル作成 (分析結果表示)"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # メインタブウィジェット（直接summary_tabsを使用）
        self.summary_tabs = QTabWidget()
        colors = self.config.get_ui_colors()
        self.summary_tabs.setStyleSheet(f"""
            QTabWidget::pane {{
                border: 1px solid {colors.get('border', '#555555')};
                background-color: {colors.get('surface', '#3c3c3c')};
            }}
            QTabBar::tab {{
                background-color: {colors.get('surface', '#3c3c3c')};
                color: {colors.get('text_primary', '#ffffff')};
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }}
            QTabBar::tab:selected {{
                background-color: {colors.get('primary', '#0078d4')};
            }}
            QTabBar::tab:hover {{
                background-color: #4c4c4c;
            }}
        """)
        
        # 各分析項目のタブを直接追加
        # サブタブ1: サマリーテーブル
        self.summary_tabs.addTab(self._create_summary_table_tab(), "サマリー")
        
        # サブタブ2: リスク・リターン散布図
        self.summary_tabs.addTab(self._create_scatter_tab(), "散布図")
        
        # サブタブ3: パフォーマンスヒートマップ
        self.summary_tabs.addTab(self._create_heatmap_tab(), "ヒートマップ")
        
        # サブタブ4: レーダーチャート
        self.summary_tabs.addTab(self._create_radar_tab(), "レーダーチャート")
        
        # サブタブ5: 効率的フロンティア比較
        self.summary_tabs.addTab(self._create_frontier_comparison_tab(), "フロンティア比較")
        
        # サブタブ6: 市場との比較
        self.summary_tabs.addTab(self._create_market_comparison_tab(), "市場比較")
        
        # サブタブ7: 価格推移
        self.summary_tabs.addTab(self._create_price_evolution_tab(), "価格推移")
        
        # サブタブ8: 実測値 vs 理論値
        self.summary_tabs.addTab(self._create_actual_vs_theoretical_tab(), "実測vs理論")
        
        layout.addWidget(self.summary_tabs)
        
        # 初期状態: プレースホルダー表示
        self._show_placeholder()

        return panel
    
    def _create_summary_table_tab(self) -> QWidget:
        """サマリータブ（サブタブ構造）"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # サブタブウィジェット
        summary_subtabs = QTabWidget()
        
        # サブタブ1: テーブル
        table_tab = QWidget()
        table_layout = QVBoxLayout(table_tab)
        
        self.summary_table = QTableWidget()
        colors = self.config.get_ui_colors()
        self.summary_table.setStyleSheet(f"""
            QTableWidget {{
                background-color: {colors.get('background', '#2b2b2b')};
                color: {colors.get('text_primary', '#ffffff')};
                border: 1px solid {colors.get('border', '#555555')};
                gridline-color: {colors.get('grid_line', '#444444')};
            }}
            QHeaderView::section {{
                background-color: {colors.get('surface', '#3c3c3c')};
                color: {colors.get('text_primary', '#ffffff')};
                border: 1px solid {colors.get('border', '#555555')};
                padding: 4px;
            }}
        """)
        table_layout.addWidget(self.summary_table)
        summary_subtabs.addTab(table_tab, "テーブル")
        
        # サブタブ2: チャート
        chart_tab = QWidget()
        chart_layout = QVBoxLayout(chart_tab)
        chart_layout.setContentsMargins(5, 5, 5, 5)

        # Plotlyボタン配置
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.summary_plotly_browser_button = QPushButton("ブラウザで表示")
        self.summary_plotly_browser_button.setStyleSheet(self.styles.get_button_style_by_type("primary"))
        self.summary_plotly_browser_button.setEnabled(False)
        self.summary_plotly_browser_button.clicked.connect(lambda: self._open_chart_in_browser('summary'))
        button_layout.addWidget(self.summary_plotly_browser_button)
        
        self.summary_plotly_export_button = QPushButton("HTML保存")
        self.summary_plotly_export_button.setStyleSheet(self.styles.get_button_style_by_type("save"))
        self.summary_plotly_export_button.setEnabled(False)
        self.summary_plotly_export_button.clicked.connect(lambda: self._export_chart('summary'))
        button_layout.addWidget(self.summary_plotly_export_button)
        
        chart_layout.addLayout(button_layout)

        # Matplotlibキャンバス（アプリ内チャート表示）
        if MATPLOTLIB_AVAILABLE:
            self.summary_matplotlib_canvas = MatplotlibCanvas(parent=chart_tab, width=10, height=8)
            chart_layout.addWidget(self.summary_matplotlib_canvas)
        else:
            # Matplotlibが利用できない場合のプレースホルダー
            placeholder = QLabel("Matplotlibが利用できません")
            placeholder.setAlignment(Qt.AlignCenter)
            placeholder.setStyleSheet("color: #888; font-size: 12px;")
            chart_layout.addWidget(placeholder)

        summary_subtabs.addTab(chart_tab, "チャート")

        layout.addWidget(summary_subtabs)
        
        return tab


    def _create_scatter_tab(self) -> QWidget:
        """リスク・リターン散布図タブ"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # サブタブウィジェット
        scatter_subtabs = QTabWidget()
        
        # サブタブ1: テーブル
        table_tab = QWidget()
        table_layout = QVBoxLayout(table_tab)

        self.scatter_table = QTableWidget()
        colors = self.config.get_ui_colors()
        self.scatter_table.setStyleSheet(f"""
            QTableWidget {{
                background-color: {colors.get('background', '#2b2b2b')};
                color: {colors.get('text_primary', '#ffffff')};
                border: 1px solid {colors.get('border', '#555555')};
                gridline-color: {colors.get('grid_line', '#444444')};
            }}
            QHeaderView::section {{
                background-color: {colors.get('surface', '#3c3c3c')};
                color: {colors.get('text_primary', '#ffffff')};
                border: 1px solid {colors.get('border', '#555555')};
                padding: 4px;
            }}
        """)
        table_layout.addWidget(self.scatter_table)
        scatter_subtabs.addTab(table_tab, "テーブル")

        # サブタブ2: チャート
        chart_tab = QWidget()
        chart_layout = QVBoxLayout(chart_tab)
        chart_layout.setContentsMargins(5, 5, 5, 5)

        # Plotlyボタン配置
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.scatter_browser_button = QPushButton("ブラウザで表示")
        self.scatter_browser_button.setStyleSheet(self.styles.get_button_style_by_type("primary"))
        self.scatter_browser_button.setEnabled(False)
        self.scatter_browser_button.clicked.connect(lambda: self._open_chart_in_browser('scatter'))
        button_layout.addWidget(self.scatter_browser_button)
        
        self.scatter_export_button = QPushButton("HTML保存")
        self.scatter_export_button.setStyleSheet(self.styles.get_button_style_by_type("save"))
        self.scatter_export_button.setEnabled(False)
        self.scatter_export_button.clicked.connect(lambda: self._export_chart('scatter'))
        button_layout.addWidget(self.scatter_export_button)
        
        chart_layout.addLayout(button_layout)

        # Matplotlibキャンバス（アプリ内チャート表示）
        if MATPLOTLIB_AVAILABLE:
            self.scatter_matplotlib_canvas = MatplotlibCanvas(parent=chart_tab, width=10, height=7)
            chart_layout.addWidget(self.scatter_matplotlib_canvas)
        else:
            placeholder = QLabel("Matplotlibが利用できません")
            placeholder.setAlignment(Qt.AlignCenter)
            placeholder.setStyleSheet("color: #888; font-size: 12px;")
            chart_layout.addWidget(placeholder)

        scatter_subtabs.addTab(chart_tab, "チャート")
        
        layout.addWidget(scatter_subtabs)
        
        return tab


    def _create_heatmap_tab(self) -> QWidget:
        """ヒートマップタブ"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # サブタブウィジェット
        heatmap_subtabs = QTabWidget()
        
        # サブタブ1: テーブル
        table_tab = QWidget()
        table_layout = QVBoxLayout(table_tab)

        self.heatmap_table = QTableWidget()
        colors = self.config.get_ui_colors()
        self.heatmap_table.setStyleSheet(f"""
            QTableWidget {{
                background-color: {colors.get('background', '#2b2b2b')};
                color: {colors.get('text_primary', '#ffffff')};
                border: 1px solid {colors.get('border', '#555555')};
                gridline-color: {colors.get('grid_line', '#444444')};
            }}
            QHeaderView::section {{
                background-color: {colors.get('surface', '#3c3c3c')};
                color: {colors.get('text_primary', '#ffffff')};
                border: 1px solid {colors.get('border', '#555555')};
                padding: 4px;
            }}
        """)
        table_layout.addWidget(self.heatmap_table)
        heatmap_subtabs.addTab(table_tab, "テーブル")

        # サブタブ2: チャート
        chart_tab = QWidget()
        chart_layout = QVBoxLayout(chart_tab)
        chart_layout.setContentsMargins(5, 5, 5, 5)

        # Plotlyボタン配置
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.heatmap_browser_button = QPushButton("ブラウザで表示")
        self.heatmap_browser_button.setStyleSheet(self.styles.get_button_style_by_type("primary"))
        self.heatmap_browser_button.setEnabled(False)
        self.heatmap_browser_button.clicked.connect(lambda: self._open_chart_in_browser('heatmap'))
        button_layout.addWidget(self.heatmap_browser_button)
        
        self.heatmap_export_button = QPushButton("HTML保存")
        self.heatmap_export_button.setStyleSheet(self.styles.get_button_style_by_type("save"))
        self.heatmap_export_button.setEnabled(False)
        self.heatmap_export_button.clicked.connect(lambda: self._export_chart('heatmap'))
        button_layout.addWidget(self.heatmap_export_button)
        
        chart_layout.addLayout(button_layout)

        # Matplotlibキャンバス（アプリ内チャート表示）
        if MATPLOTLIB_AVAILABLE:
            self.heatmap_matplotlib_canvas = MatplotlibCanvas(parent=chart_tab, width=10, height=7)
            chart_layout.addWidget(self.heatmap_matplotlib_canvas)
        else:
            placeholder = QLabel("Matplotlibが利用できません")
            placeholder.setAlignment(Qt.AlignCenter)
            placeholder.setStyleSheet("color: #888; font-size: 12px;")
            chart_layout.addWidget(placeholder)

        heatmap_subtabs.addTab(chart_tab, "チャート")
        
        layout.addWidget(heatmap_subtabs)
        
        return tab


    def _create_radar_tab(self) -> QWidget:
        """レーダーチャートタブ"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # サブタブウィジェット
        radar_subtabs = QTabWidget()
        
        # サブタブ1: テーブル
        table_tab = QWidget()
        table_layout = QVBoxLayout(table_tab)

        self.radar_table = QTableWidget()
        colors = self.config.get_ui_colors()
        self.radar_table.setStyleSheet(f"""
            QTableWidget {{
                background-color: {colors.get('background', '#2b2b2b')};
                color: {colors.get('text_primary', '#ffffff')};
                border: 1px solid {colors.get('border', '#555555')};
                gridline-color: {colors.get('grid_line', '#444444')};
            }}
            QHeaderView::section {{
                background-color: {colors.get('surface', '#3c3c3c')};
                color: {colors.get('text_primary', '#ffffff')};
                border: 1px solid {colors.get('border', '#555555')};
                padding: 4px;
            }}
        """)
        table_layout.addWidget(self.radar_table)
        radar_subtabs.addTab(table_tab, "テーブル")

        # サブタブ2: チャート
        chart_tab = QWidget()
        chart_layout = QVBoxLayout(chart_tab)
        chart_layout.setContentsMargins(5, 5, 5, 5)

        # Plotlyボタン配置
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.radar_browser_button = QPushButton("ブラウザで表示")
        self.radar_browser_button.setStyleSheet(self.styles.get_button_style_by_type("primary"))
        self.radar_browser_button.setEnabled(False)
        self.radar_browser_button.clicked.connect(lambda: self._open_chart_in_browser('radar'))
        button_layout.addWidget(self.radar_browser_button)
        
        self.radar_export_button = QPushButton("HTML保存")
        self.radar_export_button.setStyleSheet(self.styles.get_button_style_by_type("save"))
        self.radar_export_button.setEnabled(False)
        self.radar_export_button.clicked.connect(lambda: self._export_chart('radar'))
        button_layout.addWidget(self.radar_export_button)
        
        chart_layout.addLayout(button_layout)

        # Matplotlibキャンバス（アプリ内チャート表示）
        if MATPLOTLIB_AVAILABLE:
            self.radar_matplotlib_canvas = MatplotlibCanvas(parent=chart_tab, width=10, height=8)
            chart_layout.addWidget(self.radar_matplotlib_canvas)
        else:
            placeholder = QLabel("Matplotlibが利用できません")
            placeholder.setAlignment(Qt.AlignCenter)
            placeholder.setStyleSheet("color: #888; font-size: 12px;")
            chart_layout.addWidget(placeholder)

        radar_subtabs.addTab(chart_tab, "チャート")
        
        layout.addWidget(radar_subtabs)
        
        return tab

    
    def _create_frontier_comparison_tab(self) -> QWidget:
        """効率的フロンティア比較タブ"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # ボタン配置
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.frontier_browser_button = QPushButton("ブラウザで表示")
        self.frontier_browser_button.setStyleSheet(self.styles.get_button_style_by_type("primary"))
        self.frontier_browser_button.setEnabled(False)
        self.frontier_browser_button.clicked.connect(lambda: self._open_chart_in_browser('frontier'))
        button_layout.addWidget(self.frontier_browser_button)

        self.frontier_export_button = QPushButton("HTML保存")
        self.frontier_export_button.setStyleSheet(self.styles.get_button_style_by_type("save"))
        self.frontier_export_button.setEnabled(False)
        self.frontier_export_button.clicked.connect(lambda: self._export_chart('frontier'))
        button_layout.addWidget(self.frontier_export_button)

        layout.addLayout(button_layout)

        # プレースホルダー
        self.frontier_placeholder = QLabel(
            "分析を実行すると，各ポートフォリオの効率的フロンティアが表示されます\n\n"
            "表示内容:\n"
            "- Minimum Variance Frontier (最小分散フロンティア)\n"
            "- Efficient Frontier (効率的フロンティア)\n"
            "- Global Minimum Variance Portfolio (全体最小分散ポートフォリオ)\n"
            "- Capital Allocation Line (資本配分線)\n"
            "- Tangency Portfolio (接点ポートフォリオ)\n"
            "- Risk-free Rate (無リスク利子率)\n"
            "- Current Portfolio Position (現在のポートフォリオ位置)"
        )
        self.frontier_placeholder.setAlignment(Qt.AlignCenter)
        self.frontier_placeholder.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self.frontier_placeholder)

        # Matplotlibキャンバス（アプリ内表示用）
        if MATPLOTLIB_AVAILABLE:
            self.frontier_canvas = MatplotlibCanvas(parent=tab, width=10, height=8)
            self.frontier_canvas.setVisible(False)
            layout.addWidget(self.frontier_canvas)

        return tab
    
    def _create_market_comparison_tab(self) -> QWidget:
        """市場ポートフォリオとの比較タブ"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        layout.setContentsMargins(5, 5, 5, 5)

        # サブタブウィジェット
        market_subtabs = QTabWidget()

        # サブタブ1: テーブル
        table_tab = QWidget()
        table_layout = QVBoxLayout(table_tab)

        self.market_table = QTableWidget()
        colors = self.config.get_ui_colors()
        self.market_table.setStyleSheet(f"""
            QTableWidget {{
                background-color: {colors.get('background', '#2b2b2b')};
                color: {colors.get('text_primary', '#ffffff')};
                border: 1px solid {colors.get('border', '#555555')};
                gridline-color: {colors.get('grid_line', '#444444')};
            }}
            QHeaderView::section {{
                background-color: {colors.get('surface', '#3c3c3c')};
                color: {colors.get('text_primary', '#ffffff')};
                border: 1px solid {colors.get('border', '#555555')};
                padding: 4px;
            }}
        """)
        table_layout.addWidget(self.market_table)
        market_subtabs.addTab(table_tab, "テーブル")

        # サブタブ2: チャート
        chart_tab = QWidget()
        chart_layout = QVBoxLayout(chart_tab)
        chart_layout.setContentsMargins(5, 5, 5, 5)

        # Plotlyボタン配置
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.market_browser_button = QPushButton("ブラウザで表示")
        self.market_browser_button.setStyleSheet(self.styles.get_button_style_by_type("primary"))
        self.market_browser_button.setEnabled(False)
        self.market_browser_button.clicked.connect(lambda: self._open_chart_in_browser('market'))
        button_layout.addWidget(self.market_browser_button)
        
        self.market_export_button = QPushButton("HTML保存")
        self.market_export_button.setStyleSheet(self.styles.get_button_style_by_type("save"))
        self.market_export_button.setEnabled(False)
        self.market_export_button.clicked.connect(lambda: self._export_chart('market'))
        button_layout.addWidget(self.market_export_button)
        
        chart_layout.addLayout(button_layout)

        # Matplotlibキャンバス（アプリ内チャート表示）
        if MATPLOTLIB_AVAILABLE:
            self.market_matplotlib_canvas = MatplotlibCanvas(parent=chart_tab, width=12, height=8)
            chart_layout.addWidget(self.market_matplotlib_canvas)
        else:
            placeholder = QLabel("Matplotlibが利用できません")
            placeholder.setAlignment(Qt.AlignCenter)
            placeholder.setStyleSheet("color: #888; font-size: 12px;")
            chart_layout.addWidget(placeholder)

        market_subtabs.addTab(chart_tab, "チャート")

        layout.addWidget(market_subtabs)
        
        return tab
    
    def _create_price_evolution_tab(self) -> QWidget:
        """価格推移タブ"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # サブタブウィジェット
        price_subtabs = QTabWidget()
        
        # サブタブ1: テーブル
        table_tab = QWidget()
        table_layout = QVBoxLayout(table_tab)

        self.price_table = QTableWidget()
        colors = self.config.get_ui_colors()
        self.price_table.setStyleSheet(f"""
            QTableWidget {{
                background-color: {colors.get('background', '#2b2b2b')};
                color: {colors.get('text_primary', '#ffffff')};
                border: 1px solid {colors.get('border', '#555555')};
                gridline-color: {colors.get('grid_line', '#444444')};
            }}
            QHeaderView::section {{
                background-color: {colors.get('surface', '#3c3c3c')};
                color: {colors.get('text_primary', '#ffffff')};
                border: 1px solid {colors.get('border', '#555555')};
                padding: 4px;
            }}
        """)
        table_layout.addWidget(self.price_table)
        price_subtabs.addTab(table_tab, "テーブル")

        # サブタブ2: チャート
        chart_tab = QWidget()
        chart_layout = QVBoxLayout(chart_tab)
        chart_layout.setContentsMargins(5, 5, 5, 5)

        # Plotlyボタン配置
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.price_browser_button = QPushButton("ブラウザで表示")
        self.price_browser_button.setStyleSheet(self.styles.get_button_style_by_type("primary"))
        self.price_browser_button.setEnabled(False)
        self.price_browser_button.clicked.connect(lambda: self._open_chart_in_browser('price'))
        button_layout.addWidget(self.price_browser_button)
        
        self.price_export_button = QPushButton("HTML保存")
        self.price_export_button.setStyleSheet(self.styles.get_button_style_by_type("save"))
        self.price_export_button.setEnabled(False)
        self.price_export_button.clicked.connect(lambda: self._export_chart('price'))
        button_layout.addWidget(self.price_export_button)
        
        chart_layout.addLayout(button_layout)

        # Matplotlibキャンバス（アプリ内チャート表示）
        if MATPLOTLIB_AVAILABLE:
            self.price_matplotlib_canvas = MatplotlibCanvas(parent=chart_tab, width=12, height=7)
            chart_layout.addWidget(self.price_matplotlib_canvas)
        else:
            placeholder = QLabel("Matplotlibが利用できません")
            placeholder.setAlignment(Qt.AlignCenter)
            placeholder.setStyleSheet("color: #888; font-size: 12px;")
            chart_layout.addWidget(placeholder)

        price_subtabs.addTab(chart_tab, "チャート")
        
        layout.addWidget(price_subtabs)
        
        return tab
    
    def _create_actual_vs_theoretical_tab(self) -> QWidget:
        """実測値vs理論値タブ"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # サブタブウィジェット
        capm_subtabs = QTabWidget()
        
        # サブタブ1: テーブル
        table_tab = QWidget()
        table_layout = QVBoxLayout(table_tab)

        self.capm_table = QTableWidget()
        colors = self.config.get_ui_colors()
        self.capm_table.setStyleSheet(f"""
            QTableWidget {{
                background-color: {colors.get('background', '#2b2b2b')};
                color: {colors.get('text_primary', '#ffffff')};
                border: 1px solid {colors.get('border', '#555555')};
                gridline-color: {colors.get('grid_line', '#444444')};
            }}
            QHeaderView::section {{
                background-color: {colors.get('surface', '#3c3c3c')};
                color: {colors.get('text_primary', '#ffffff')};
                border: 1px solid {colors.get('border', '#555555')};
                padding: 4px;
            }}
        """)
        table_layout.addWidget(self.capm_table)
        capm_subtabs.addTab(table_tab, "テーブル")

        # サブタブ2: チャート
        chart_tab = QWidget()
        chart_layout = QVBoxLayout(chart_tab)
        chart_layout.setContentsMargins(5, 5, 5, 5)

        # Plotlyボタン配置
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.capm_browser_button = QPushButton("ブラウザで表示")
        self.capm_browser_button.setStyleSheet(self.styles.get_button_style_by_type("primary"))
        self.capm_browser_button.setEnabled(False)
        self.capm_browser_button.clicked.connect(lambda: self._open_chart_in_browser('capm'))
        button_layout.addWidget(self.capm_browser_button)
        
        self.capm_export_button = QPushButton("HTML保存")
        self.capm_export_button.setStyleSheet(self.styles.get_button_style_by_type("save"))
        self.capm_export_button.setEnabled(False)
        self.capm_export_button.clicked.connect(lambda: self._export_chart('capm'))
        button_layout.addWidget(self.capm_export_button)
        
        chart_layout.addLayout(button_layout)

        # Matplotlibキャンバス（アプリ内チャート表示）
        if MATPLOTLIB_AVAILABLE:
            self.capm_matplotlib_canvas = MatplotlibCanvas(parent=chart_tab, width=14, height=6)
            chart_layout.addWidget(self.capm_matplotlib_canvas)
        else:
            placeholder = QLabel("Matplotlibが利用できません")
            placeholder.setAlignment(Qt.AlignCenter)
            placeholder.setStyleSheet("color: #888; font-size: 12px;")
            chart_layout.addWidget(placeholder)

        capm_subtabs.addTab(chart_tab, "チャート")
        
        layout.addWidget(capm_subtabs)
        
        return tab
    
    def _load_portfolios(self):
        """ポートフォリオ一覧を読み込み"""
        try:
            # 設定から最新のフォルダパスを取得してPortfolioManagerを再初期化
            custom_folder = self.config.get_portfolio_folder()
            if custom_folder:
                self.portfolio_manager = PortfolioManager(custom_folder)
            else:
                self.portfolio_manager = PortfolioManager()

            self.available_portfolios = self.portfolio_manager.list_portfolios()
            self._populate_portfolio_table()
        except Exception as e:
            self.logger.error(f"ポートフォリオ読み込みエラー: {e}")
            QMessageBox.warning(self, "エラー", f"ポートフォリオの読み込みに失敗しました:\n{e}")
    
    def _populate_portfolio_table(self):
        """ポートフォリオテーブルに表示"""
        self.portfolio_table.blockSignals(True)
        
        self.portfolio_table.setRowCount(len(self.available_portfolios))
        
        # チェックボックスのスタイル
        checkbox_style = get_checkbox_style()
        
        for row, (file_name, name, created_at) in enumerate(self.available_portfolios):
            # チェックボックスをQCheckBoxウィジェットとして追加
            checkbox = QCheckBox()
            checkbox.setStyleSheet(checkbox_style)
            checkbox.setChecked(False)
            checkbox.stateChanged.connect(self._on_selection_changed)
            
            # file_nameをチェックボックスに保存
            checkbox.setProperty("file_name", file_name)
            
            # チェックボックスを中央配置するためのウィジェット
            checkbox_widget = QWidget()
            checkbox_layout = QHBoxLayout(checkbox_widget)
            checkbox_layout.addWidget(checkbox)
            checkbox_layout.setAlignment(Qt.AlignCenter)
            checkbox_layout.setContentsMargins(0, 0, 0, 0)
            
            self.portfolio_table.setCellWidget(row, 0, checkbox_widget)
            
            # 名前
            name_item = QTableWidgetItem(name)
            name_item.setData(Qt.UserRole, file_name)  # file_nameを保存
            self.portfolio_table.setItem(row, 1, name_item)
            
            # 作成日時
            date_item = QTableWidgetItem(created_at)
            self.portfolio_table.setItem(row, 2, date_item)
            
            # 資産数
            try:
                portfolio = self.portfolio_manager.load_portfolio(file_name)
                asset_count = len(portfolio.positions) if portfolio else 0
            except Exception as e:
                logging.debug(f"Failed to load portfolio {file_name}: {e}")
                asset_count = 0
            
            count_item = QTableWidgetItem(str(asset_count))
            count_item.setTextAlignment(Qt.AlignCenter)
            self.portfolio_table.setItem(row, 3, count_item)
        
        self.portfolio_table.blockSignals(False)
    
    def _filter_portfolios(self, text: str):
        """ポートフォリオをフィルタリング"""
        for row in range(self.portfolio_table.rowCount()):
            name_item = self.portfolio_table.item(row, 1)
            if name_item:
                match = text.lower() in name_item.text().lower()
                self.portfolio_table.setRowHidden(row, not match)
    
    def _sort_portfolios(self, sort_type: str):
        """ポートフォリオをソート"""
        if "名前" in sort_type:
            col = 1
            ascending = "昇順" in sort_type
        else:  # 作成日時
            col = 2
            ascending = "古" in sort_type
        
        self.portfolio_table.sortItems(col, Qt.AscendingOrder if ascending else Qt.DescendingOrder)
    
    def _select_all_portfolios(self):
        """全選択"""
        self.portfolio_table.blockSignals(True)
        for row in range(self.portfolio_table.rowCount()):
            if not self.portfolio_table.isRowHidden(row):
                checkbox_widget = self.portfolio_table.cellWidget(row, 0)
                if checkbox_widget:
                    checkbox = checkbox_widget.findChild(QCheckBox)
                    if checkbox:
                        checkbox.setChecked(True)
        self.portfolio_table.blockSignals(False)
        # 選択状態を更新
        self._on_selection_changed()
    
    def _deselect_all_portfolios(self):
        """全解除"""
        self.portfolio_table.blockSignals(True)
        for row in range(self.portfolio_table.rowCount()):
            checkbox_widget = self.portfolio_table.cellWidget(row, 0)
            if checkbox_widget:
                checkbox = checkbox_widget.findChild(QCheckBox)
                if checkbox:
                    checkbox.setChecked(False)
        self.portfolio_table.blockSignals(False)
        # 選択状態を更新
        self._on_selection_changed()

    def _refresh_portfolios(self):
        """ポートフォリオリストを更新（選択状態を保持）"""
        # 現在選択されているポートフォリオのfile_nameを保存
        selected_file_names = set()
        for row in range(self.portfolio_table.rowCount()):
            checkbox_widget = self.portfolio_table.cellWidget(row, 0)
            if checkbox_widget:
                checkbox = checkbox_widget.findChild(QCheckBox)
                if checkbox and checkbox.isChecked():
                    file_name = checkbox.property("file_name")
                    if file_name:
                        selected_file_names.add(file_name)

        # ポートフォリオリストを再読み込み
        try:
            # 設定から最新のフォルダパスを取得してPortfolioManagerを再初期化
            custom_folder = self.config.get_portfolio_folder()
            if custom_folder:
                self.portfolio_manager = PortfolioManager(custom_folder)
            else:
                self.portfolio_manager = PortfolioManager()

            self.available_portfolios = self.portfolio_manager.list_portfolios()
            self._populate_portfolio_table()

            # 以前選択されていたポートフォリオを再選択
            if selected_file_names:
                self.portfolio_table.blockSignals(True)
                for row in range(self.portfolio_table.rowCount()):
                    checkbox_widget = self.portfolio_table.cellWidget(row, 0)
                    if checkbox_widget:
                        checkbox = checkbox_widget.findChild(QCheckBox)
                        if checkbox:
                            file_name = checkbox.property("file_name")
                            if file_name in selected_file_names:
                                checkbox.setChecked(True)
                self.portfolio_table.blockSignals(False)
                # 選択状態を更新
                self._on_selection_changed()

            self.logger.info(f"ポートフォリオリストを更新しました（{len(self.available_portfolios)}件）")

        except Exception as e:
            self.logger.error(f"ポートフォリオ更新エラー: {e}")
            QMessageBox.warning(self, "エラー", f"ポートフォリオの更新に失敗しました:\n{e}")

    def _on_selection_changed(self, item=None):
        """選択状態が変更されたとき"""
        self.selected_portfolios.clear()
        
        # 各行のチェックボックスの状態を確認
        for row in range(self.portfolio_table.rowCount()):
            checkbox_widget = self.portfolio_table.cellWidget(row, 0)
            if checkbox_widget:
                # ウィジェット内のチェックボックスを取得
                checkbox = checkbox_widget.findChild(QCheckBox)
                if checkbox and checkbox.isChecked():
                    file_name = checkbox.property("file_name")
                    if file_name:
                        try:
                            portfolio = self.portfolio_manager.load_portfolio(file_name)
                            if portfolio:
                                self.selected_portfolios[file_name] = portfolio
                        except Exception as e:
                            self.logger.error(f"ポートフォリオ読み込みエラー ({file_name}): {e}")
        
        # 選択数を更新
        count = len(self.selected_portfolios)
        self.selection_label.setText(f"選択中: {count} 個")
        
        # 分析ボタンの有効/無効
        self.execute_button.setEnabled(count >= 2)
    
    def _fetch_initial_interest_rate(self):
        """初期利子率を取得"""
        QTimer.singleShot(500, self._fetch_interest_rate)
    
    def _fetch_interest_rate(self):
        """利子率を取得"""
        self.fetch_rate_button.setEnabled(False)
        self.rate_status_label.setText("取得中...")
        
        self.rate_thread = InterestRateThread()
        self.rate_thread.rate_fetched.connect(self._on_rate_fetched)
        self.rate_thread.fetch_error.connect(self._on_rate_fetch_error)
        self.rate_thread.start()
    
    def _on_rate_fetched(self, rate: float):
        """利子率取得成功"""
        self.interest_rate_spin.setValue(rate)
        self.rate_status_label.setText(f"短期国債利回り: {rate:.3f}%")
        self.fetch_rate_button.setEnabled(True)
    
    def _on_rate_fetch_error(self, error: str):
        """利子率取得失敗"""
        self.rate_status_label.setText("取得失敗")
        self.fetch_rate_button.setEnabled(True)
    
    def _execute_analysis(self):
        """分析を実行"""
        if len(self.selected_portfolios) < 2:
            QMessageBox.warning(self, "選択不足", "比較には2つ以上のポートフォリオを選択してください．")
            return
        
        # 分析条件を取得
        start_date = self.start_date_edit.date().toPython()
        end_date = self.end_date_edit.date().toPython()
        interest_rate = self.interest_rate_spin.value() / 100.0
        span = self.span_combo.currentText()
        market_ticker = self._get_market_ticker()
        use_capm = self.basis_group.checkedButton().text() == "理論値 (CAPM)" if self.basis_group.checkedButton() else False
        
        self.logger.info(f"分析実行: {len(self.selected_portfolios)}個のポートフォリオ")
        self.logger.info(f"期間: {start_date} ~ {end_date}")
        self.logger.info(f"利子率: {interest_rate:.3%}")
        self.logger.info(f"比較ベース: {'CAPM' if use_capm else '実測値'}")
        
        # プログレスダイアログ
        progress = QProgressDialog("分析中...", "キャンセル", 0, 100, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()
        
        try:
            # Step 1: 価格データの取得
            progress.setValue(10)
            progress.setLabelText("価格データを取得中...")
            QApplication.processEvents()
            
            price_data = self._fetch_price_data(start_date, end_date, span)
            
            if not price_data:
                QMessageBox.warning(self, "エラー", "価格データの取得に失敗しました．")
                return
            
            # Step 2: 統計量の計算
            progress.setValue(30)
            progress.setLabelText("統計量を計算中...")
            QApplication.processEvents()
            
            from analysis.portfolio_comparison import calculate_all_portfolios_metrics
            
            portfolios = list(self.selected_portfolios.values())
            all_metrics = calculate_all_portfolios_metrics(
                portfolios,
                price_data,
                interest_rate
            )
            
            if not all_metrics:
                QMessageBox.warning(self, "エラー", "統計量の計算に失敗しました．")
                return
            
            # Step 3: 効率性指標の計算
            progress.setValue(60)
            progress.setLabelText("効率性指標を計算中...")
            QApplication.processEvents()
            
            from analysis.portfolio_comparison import calculate_all_portfolios_efficiency
            
            market_metrics = None
            if market_ticker and use_capm:
                market_metrics = self._fetch_market_metrics(
                    market_ticker, start_date, end_date, span, interest_rate
                )
            
            all_efficiency = calculate_all_portfolios_efficiency(
                all_metrics,
                market_metrics
            )
            
            # Step 4: 結果の保存
            progress.setValue(90)
            QApplication.processEvents()
            
            self.analysis_results = {
                'metrics': all_metrics,
                'efficiency': all_efficiency,
                'price_data': price_data,
                'conditions': {
                    'start_date': start_date,
                    'end_date': end_date,
                    'interest_rate': interest_rate,
                    'span': span,
                    'market_ticker': market_ticker,
                    'use_capm': use_capm
                }
            }
            
            # 結果を表示
            self._display_results()
            
            # Plotlyチャートを生成
            self._create_all_plotly_charts()
            
            # クリアボタンを有効化
            self.clear_button.setEnabled(True)
            
            progress.setValue(100)
            progress.close()
            
            QMessageBox.information(
                self,
                "分析完了",
                f"{len(self.selected_portfolios)}個のポートフォリオの比較分析が完了しました．"
            )
            
        except Exception as e:
            progress.close()
            self.logger.error(f"分析エラー: {e}", exc_info=True)
            QMessageBox.critical(self, "エラー", f"分析中にエラーが発生しました:\n{str(e)}")
    
    def _fetch_price_data(self, start_date, end_date, span):
        """価格データを取得"""
        try:
            import yfinance as yf
            
            # 全ポートフォリオの銘柄を集約
            all_symbols = set()
            for portfolio in self.selected_portfolios.values():
                for position in portfolio.positions:
                    all_symbols.add(position.asset.symbol)
            
            # yfinanceのインターバル変換
            interval_map = {
                '日次': '1d',
                '週次': '1wk',
                '月次': '1mo'
            }
            interval = interval_map.get(span, '1d')
            
            self.logger.info(f"価格データ取得開始: {len(all_symbols)}銘柄, 期間: {start_date} ~ {end_date}, 間隔: {interval}")
            
            price_data = {}
            success_count = 0
            failed_symbols = []
            
            for symbol in all_symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(
                        start=start_date,
                        end=end_date + timedelta(days=1),  # 終了日を含めるため+1日
                        interval=interval,
                        auto_adjust=False,  # 調整済み終値を取得
                        prepost=False,      # プレ・ポストマーケットを除外
                        repair=True         # データ修復を有効化
                    )
                    
                    if not hist.empty:
                        # 調整済み終値を優先，なければ終値を使用
                        if 'Adj Close' in hist.columns and not hist['Adj Close'].isna().all():
                            price_series = hist['Adj Close'].copy()
                        elif 'Close' in hist.columns:
                            price_series = hist['Close'].copy()
                        else:
                            self.logger.warning(f"{symbol}: 価格カラムが見つかりません")
                            failed_symbols.append(symbol)
                            continue
                        
                        # タイムゾーン処理
                        if hasattr(price_series.index, 'tz') and price_series.index.tz is not None:
                            price_series.index = price_series.index.tz_localize(None)
                        
                        # 日付のみに正規化
                        price_series.index = price_series.index.normalize()
                        
                        # NaN値を除去
                        price_series = price_series.dropna()
                        
                        if len(price_series) > 0:
                            price_data[symbol] = price_series
                            success_count += 1
                            self.logger.info(f"{symbol}: {len(price_series)}件のデータを取得")
                        else:
                            self.logger.warning(f"{symbol}: データが空です")
                            failed_symbols.append(symbol)
                    else:
                        self.logger.warning(f"{symbol}: データが取得できませんでした")
                        failed_symbols.append(symbol)
                        
                except Exception as e:
                    self.logger.error(f"{symbol} の価格データ取得エラー: {e}")
                    failed_symbols.append(symbol)
                    continue
            
            # 結果のサマリーをログ出力
            self.logger.info(f"価格データ取得完了: 成功 {success_count}/{len(all_symbols)} 銘柄")
            if failed_symbols:
                self.logger.warning(f"取得失敗銘柄: {', '.join(failed_symbols)}")
            
            return price_data
            
        except Exception as e:
            self.logger.error(f"価格データ取得エラー: {e}")
            import traceback
            self.logger.error(f"トレースバック:\n{traceback.format_exc()}")
            return {}
    
    def _fetch_market_metrics(self, market_ticker, start_date, end_date, span, risk_free_rate):
        """市場ポートフォリオの統計量を取得"""
        try:
            import yfinance as yf
            from analysis.portfolio_comparison import PortfolioMetricsCalculator
            
            # yfinanceのインターバル変換
            interval_map = {
                '日次': '1d',
                '週次': '1wk',
                '月次': '1mo'
            }
            interval = interval_map.get(span, '1d')
            
            # 市場データを取得
            ticker = yf.Ticker(market_ticker)
            hist = ticker.history(
                start=start_date,
                end=end_date + timedelta(days=1),
                interval=interval,
                auto_adjust=False,
                prepost=False,
                repair=True
            )
            
            if hist.empty:
                self.logger.warning(f"市場データが取得できませんでした: {market_ticker}")
                return None
            
            # 調整済み終値を取得
            if 'Adj Close' in hist.columns and not hist['Adj Close'].isna().all():
                market_data = hist['Adj Close'].copy()
            elif 'Close' in hist.columns:
                market_data = hist['Close'].copy()
            else:
                self.logger.warning(f"市場データに価格カラムが見つかりません: {market_ticker}")
                return None
            
            # タイムゾーン処理
            if hasattr(market_data.index, 'tz') and market_data.index.tz is not None:
                market_data.index = market_data.index.tz_localize(None)
            
            # 日付のみに正規化
            market_data.index = market_data.index.normalize()
            
            # NaN値を除去
            market_data = market_data.dropna()
            
            if market_data.empty:
                self.logger.warning(f"市場データが空です: {market_ticker}")
                return None
            
            # リターンを計算
            calculator = PortfolioMetricsCalculator()
            returns_data = calculator.calculate_returns_from_prices({
                market_ticker: market_data
            })
            
            market_returns = returns_data.get(market_ticker)
            if market_returns is None or market_returns.empty:
                self.logger.warning(f"市場リターンの計算に失敗しました: {market_ticker}")
                return None
            
            # 統計量を計算
            metrics = {
                'expected_return': market_returns.mean(),
                'portfolio_std': market_returns.std()
            }
            
            self.logger.info(f"市場データ取得成功: {market_ticker}, {len(market_returns)}件")
            return metrics
            
        except Exception as e:
            self.logger.error(f"市場統計量取得エラー: {e}")
            import traceback
            self.logger.error(f"トレースバック:\n{traceback.format_exc()}")
            return None
    
    def _get_market_ticker(self) -> str:
        """市場ティッカーを取得"""
        text = self.market_combo.currentText()
        if "^N225" in text:
            return "^N225"
        elif "^GSPC" in text:
            return "^GSPC"
        elif "1306.T" in text:
            return "1306.T"
        else:
            return ""
    
    def _update_scatter_matplotlib_chart(self, all_metrics, all_efficiency):
        """散布図のMatplotlibチャートを更新"""
        try:
            if not MATPLOTLIB_AVAILABLE:
                self.logger.warning("Matplotlibが利用できません")
                return
            
            if not hasattr(self, 'scatter_matplotlib_canvas'):
                self.logger.warning("散布図のMatplotlibキャンバスが初期化されていません")
                return
            
            # scatter_plotモジュールを使用してチャート生成
            fig = scatter_plot.create_scatter_matplotlib_chart(
                all_metrics,
                all_efficiency,
                self.config
            )
            
            if fig:
                # キャンバスに表示
                self.scatter_matplotlib_canvas.display_figure(fig)
                self.logger.info("散布図のMatplotlibチャートを更新しました")
            else:
                self.logger.warning("散布図チャート生成に失敗しました")
                
        except Exception as e:
            self.logger.error(f"散布図Matplotlibチャート更新エラー: {e}")
            import traceback
            self.logger.error(traceback.format_exc())


    def _update_heatmap_matplotlib_chart(self, all_metrics, all_efficiency):
        """ヒートマップのMatplotlibチャートを更新"""
        try:
            if not MATPLOTLIB_AVAILABLE:
                self.logger.warning("Matplotlibが利用できません")
                return
            
            if not hasattr(self, 'heatmap_matplotlib_canvas'):
                self.logger.warning("ヒートマップのMatplotlibキャンバスが初期化されていません")
                return
            
            # heatmapモジュールを使用してチャート生成
            fig = heatmap.create_heatmap_matplotlib_chart(
                all_metrics,
                all_efficiency,
                self.config
            )
            
            if fig:
                # キャンバスに表示
                self.heatmap_matplotlib_canvas.display_figure(fig)
                self.logger.info("ヒートマップのMatplotlibチャートを更新しました")
            else:
                self.logger.warning("ヒートマップチャート生成に失敗しました")
                
        except Exception as e:
            self.logger.error(f"ヒートマップMatplotlibチャート更新エラー: {e}")
            import traceback
            self.logger.error(traceback.format_exc())


    def _update_radar_matplotlib_chart(self, all_metrics, all_efficiency):
        """レーダーチャートのMatplotlibチャートを更新"""
        try:
            if not MATPLOTLIB_AVAILABLE:
                self.logger.warning("Matplotlibが利用できません")
                return
            
            if not hasattr(self, 'radar_matplotlib_canvas'):
                self.logger.warning("レーダーチャートのMatplotlibキャンバスが初期化されていません")
                return
            
            # radar_chartモジュールを使用してチャート生成
            fig = radar_chart.create_radar_matplotlib_chart(
                all_metrics,
                all_efficiency,
                self.config
            )
            
            if fig:
                # キャンバスに表示
                self.radar_matplotlib_canvas.display_figure(fig)
                self.logger.info("レーダーチャートのMatplotlibチャートを更新しました")
            else:
                self.logger.warning("レーダーチャート生成に失敗しました")
                
        except Exception as e:
            self.logger.error(f"レーダーチャートMatplotlibチャート更新エラー: {e}")
            import traceback
            self.logger.error(traceback.format_exc())


    def _update_price_evolution_matplotlib_chart(self):
        """価格推移のMatplotlibチャートを更新"""
        try:
            if not MATPLOTLIB_AVAILABLE:
                self.logger.warning("Matplotlibが利用できません")
                return
            
            if not hasattr(self, 'price_matplotlib_canvas'):
                self.logger.warning("価格推移のMatplotlibキャンバスが初期化されていません")
                return
            
            # データの準備
            portfolios = list(self.selected_portfolios.values())
            price_data = self.analysis_results.get('price_data', {})
            
            if not price_data:
                self.logger.warning("価格データがありません")
                return
            
            # price_evolutionモジュールを使用してチャート生成
            fig = price_evolution.create_price_evolution_matplotlib_chart(
                portfolios,
                price_data,
                self.config
            )
            
            if fig:
                # キャンバスに表示
                self.price_matplotlib_canvas.display_figure(fig)
                self.logger.info("価格推移のMatplotlibチャートを更新しました")
            else:
                self.logger.warning("価格推移チャート生成に失敗しました")
                
        except Exception as e:
            self.logger.error(f"価格推移Matplotlibチャート更新エラー: {e}")
            import traceback
            self.logger.error(traceback.format_exc())


    def _update_actual_vs_theoretical_matplotlib_chart(self):
        """実測vs理論値のMatplotlibチャートを更新"""
        try:
            if not MATPLOTLIB_AVAILABLE:
                self.logger.warning("Matplotlibが利用できません")
                return
            
            if not hasattr(self, 'capm_matplotlib_canvas'):
                self.logger.warning("CAPM分析のMatplotlibキャンバスが初期化されていません")
                return
            
            # データの準備
            # ベータ値データを取得
            beta_results = self.analysis_results.get('beta_results')
            market_expected_return = self.analysis_results.get('market_expected_return')
            span_risk_free_rate = self.analysis_results.get('span_risk_free_rate')

            if not beta_results:
                self.logger.info("実測vs理論値チャートにはベータ値データが必要です（市場比較タブで計算されます）")
                return

            # メトリクスデータを取得
            all_metrics = self.analysis_results.get('metrics', {})
            if not all_metrics:
                self.logger.warning("メトリクスデータがありません")
                return

            # ベータ値辞書を作成
            betas = {}
            for pf_name, beta_data in beta_results.items():
                if beta_data.get('success', False):
                    betas[pf_name] = beta_data['beta']

            if not betas:
                self.logger.warning("有効なベータ値がありません")
                return

            # Matplotlibチャート生成
            fig = actual_vs_theoretical.create_actual_vs_theoretical_matplotlib_chart(
                all_metrics,
                betas,
                span_risk_free_rate,
                market_expected_return,
                self.config
            )

            if fig:
                # キャンバスに表示
                self.capm_matplotlib_canvas.display_figure(fig)
                self.logger.info("実測vs理論値のMatplotlibチャートを更新しました")
            else:
                self.logger.warning("実測vs理論値チャート生成に失敗しました")
            
        except Exception as e:
            self.logger.error(f"実測vs理論値Matplotlibチャート更新エラー: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

    def _display_results(self):
        """結果を表示"""
        if not self.analysis_results:
            return
        
        try:
            all_metrics = self.analysis_results.get('metrics', {})
            all_efficiency = self.analysis_results.get('efficiency', {})
            
            # サマリーテーブルの更新
            self._update_summary_table(all_metrics, all_efficiency)
            self._update_scatter_table(all_metrics, all_efficiency)
            self._update_heatmap_table(all_metrics, all_efficiency)
            self._update_radar_table(all_metrics, all_efficiency)
            self._update_price_table()
            self._update_capm_table()

            if MATPLOTLIB_AVAILABLE:
                # サマリーチャート
                self._update_summary_matplotlib_chart(all_metrics, all_efficiency)
                
                # 散布図
                self._update_scatter_matplotlib_chart(all_metrics, all_efficiency)
                
                # ヒートマップ
                self._update_heatmap_matplotlib_chart(all_metrics, all_efficiency)
                
                # レーダーチャート
                self._update_radar_matplotlib_chart(all_metrics, all_efficiency)
                
                # 価格推移（データがある場合）
                if self.analysis_results.get('price_data'):
                    self._update_price_evolution_matplotlib_chart()
                
                # 実測vs理論値（ベータ値データがある場合）
                if self.analysis_results.get('beta_results'):
                    self._update_actual_vs_theoretical_matplotlib_chart()
            
            self.logger.info("分析結果の表示を完了しました")

        except Exception as e:
            self.logger.error(f"結果表示エラー: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            QMessageBox.warning(self, "表示エラー", f"結果の表示中にエラーが発生しました:\n{str(e)}")
    def _update_summary_matplotlib_chart(self, all_metrics, all_efficiency):
        """Matplotlibサマリーチャートを更新"""
        try:
            if not MATPLOTLIB_AVAILABLE:
                self.logger.warning("Matplotlibが利用できません")
                return
            
            if not hasattr(self, 'summary_matplotlib_canvas'):
                self.logger.warning("Matplotlibキャンバスが初期化されていません")
                return
            
            # summary_statsモジュールを使用してチャート生成
            fig = summary_stats.create_summary_matplotlib_chart(
                all_metrics,
                all_efficiency,
                self.config
            )
            
            if fig:
                # キャンバスに表示
                self.summary_matplotlib_canvas.display_figure(fig)
                self.logger.info("Matplotlibサマリーチャートを更新しました")
            else:
                self.logger.warning("チャート生成に失敗しました")
                
        except Exception as e:
            self.logger.error(f"Matplotlibチャート更新エラー: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    def _update_summary_table(self, all_metrics, all_efficiency):
        """サマリーテーブルを更新"""
        try:
            portfolios = list(all_metrics.keys())
            
            self.summary_table.setRowCount(len(portfolios))
            self.summary_table.setColumnCount(7)
            self.summary_table.setHorizontalHeaderLabels([
                "ポートフォリオ", "年率リターン", "年率リスク", "シャープレシオ", 
                "ソルティノレシオ", "最大DD", "資産数"
            ])
            
            for row, portfolio_name in enumerate(portfolios):
                metrics = all_metrics[portfolio_name]
                efficiency = all_efficiency.get(portfolio_name, {})
                
                # ポートフォリオ名
                name_item = QTableWidgetItem(portfolio_name)
                self.summary_table.setItem(row, 0, name_item)
                
                # 年率リターン
                annualized_return = metrics.get('annualized_return', np.nan)
                return_item = QTableWidgetItem(
                    f"{annualized_return * 100:.2f}%" if not np.isnan(annualized_return) else "N/A"
                )
                return_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.summary_table.setItem(row, 1, return_item)
                
                # 年率リスク
                annualized_std = metrics.get('annualized_std', np.nan)
                std_item = QTableWidgetItem(
                    f"{annualized_std * 100:.2f}%" if not np.isnan(annualized_std) else "N/A"
                )
                std_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.summary_table.setItem(row, 2, std_item)
                
                # シャープレシオ
                sharpe = efficiency.get('annualized_sharpe', np.nan)
                sharpe_item = QTableWidgetItem(
                    f"{sharpe:.3f}" if not np.isnan(sharpe) else "N/A"
                )
                sharpe_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.summary_table.setItem(row, 3, sharpe_item)
                
                # ソルティノレシオ
                sortino = efficiency.get('sortino_ratio', np.nan)
                sortino_item = QTableWidgetItem(
                    f"{sortino:.3f}" if not np.isnan(sortino) else "N/A"
                )
                sortino_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.summary_table.setItem(row, 4, sortino_item)
                
                # 最大ドローダウン
                max_dd = metrics.get('max_drawdown', np.nan)
                dd_item = QTableWidgetItem(
                    f"{max_dd * 100:.2f}%" if not np.isnan(max_dd) else "N/A"
                )
                dd_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.summary_table.setItem(row, 5, dd_item)
                
                # 資産数
                n_positions = metrics.get('n_positions', 0)
                count_item = QTableWidgetItem(str(n_positions))
                count_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.summary_table.setItem(row, 6, count_item)
            
            self.summary_table.resizeColumnsToContents()
            
        except Exception as e:
            self.logger.error(f"サマリーテーブル更新エラー: {e}")
    def _update_scatter_table(self, all_metrics, all_efficiency):
        """散布図テーブルを更新"""
        try:
            # 散布図のテーブルはサマリーと同じデータを表示
            self._populate_table_with_metrics(
                self.scatter_table,
                all_metrics,
                all_efficiency,
                ["ポートフォリオ", "年率リターン", "年率リスク", "シャープレシオ"]
            )
        except Exception as e:
            self.logger.error(f"散布図テーブル更新エラー: {e}")

    def _update_heatmap_table(self, all_metrics, all_efficiency):
        """ヒートマップテーブルを更新"""
        try:
            self._populate_table_with_metrics(
                self.heatmap_table,
                all_metrics,
                all_efficiency,
                ["ポートフォリオ", "年率リターン", "年率リスク", "シャープレシオ", "ソルティノレシオ", "最大DD"]
            )
        except Exception as e:
            self.logger.error(f"ヒートマップテーブル更新エラー: {e}")

    def _update_radar_table(self, all_metrics, all_efficiency):
        """レーダーチャートテーブルを更新"""
        try:
            self._populate_table_with_metrics(
                self.radar_table,
                all_metrics,
                all_efficiency,
                ["ポートフォリオ", "年率リターン", "年率リスク", "シャープレシオ", "ソルティノレシオ", "最大DD"]
            )
        except Exception as e:
            self.logger.error(f"レーダーチャートテーブル更新エラー: {e}")

    def _update_price_table(self):
        """価格推移テーブルを更新"""
        try:
            price_data = self.analysis_results.get('price_data', {})
            if not price_data:
                return

            # 価格推移の統計情報を表示
            portfolios = list(price_data.keys())
            self.price_table.setRowCount(len(portfolios))
            self.price_table.setColumnCount(5)
            self.price_table.setHorizontalHeaderLabels([
                "ポートフォリオ", "最終値", "最高値", "最低値", "累積リターン"
            ])

            for row, portfolio_name in enumerate(portfolios):
                prices = price_data[portfolio_name]
                if len(prices) == 0:
                    continue

                name_item = QTableWidgetItem(portfolio_name)
                self.price_table.setItem(row, 0, name_item)

                final_value = prices[-1]
                max_value = np.max(prices)
                min_value = np.min(prices)
                cumulative_return = (final_value / 100.0 - 1) * 100

                items = [
                    QTableWidgetItem(f"{final_value:.2f}"),
                    QTableWidgetItem(f"{max_value:.2f}"),
                    QTableWidgetItem(f"{min_value:.2f}"),
                    QTableWidgetItem(f"{cumulative_return:.2f}%")
                ]

                for col, item in enumerate(items, 1):
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    self.price_table.setItem(row, col, item)

            self.price_table.resizeColumnsToContents()

        except Exception as e:
            self.logger.error(f"価格推移テーブル更新エラー: {e}")

    def _update_capm_table(self):
        """実測vs理論値テーブルを更新"""
        try:
            # ベータ値データを取得
            beta_results = self.analysis_results.get('beta_results')
            market_expected_return = self.analysis_results.get('market_expected_return')
            span_risk_free_rate = self.analysis_results.get('span_risk_free_rate')

            if not beta_results:
                self.logger.info("実測vs理論テーブルにはベータ値データが必要です")
                return

            # メトリクスデータを取得
            all_metrics = self.analysis_results.get('metrics', {})

            if not all_metrics:
                self.logger.warning("メトリクスデータがありません")
                return

            # 有効なデータのみを取得
            valid_data = {name: data for name, data in beta_results.items()
                         if data.get('success', False)}

            if not valid_data:
                self.logger.warning("有効なベータ値データがありません")
                return

            # テーブルヘッダー
            headers = [
                "ポートフォリオ",
                "実測リターン (%)",
                "理論リターン (CAPM) (%)",
                "差異 (%)",
                "β値",
                "α値 (%)"
            ]

            # ポートフォリオ名リスト
            pf_names = sorted(valid_data.keys())

            # テーブル設定
            self.capm_table.setRowCount(len(pf_names))
            self.capm_table.setColumnCount(len(headers))
            self.capm_table.setHorizontalHeaderLabels(headers)

            # データ挿入
            for row, pf_name in enumerate(pf_names):
                beta_data = valid_data[pf_name]
                metrics = all_metrics.get(pf_name, {})

                # ポートフォリオ名
                name_item = QTableWidgetItem(pf_name)
                name_item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
                self.capm_table.setItem(row, 0, name_item)

                # 実測リターン（年率化済み）
                actual_return = metrics.get('annualized_return', np.nan)
                if not np.isnan(actual_return):
                    item = QTableWidgetItem(f"{actual_return * 100:.3f}")
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    self.capm_table.setItem(row, 1, item)
                else:
                    item = QTableWidgetItem("-")
                    item.setTextAlignment(Qt.AlignCenter)
                    self.capm_table.setItem(row, 1, item)

                # 理論リターン（CAPM）
                # CAPM: E(R) = Rf + β × (E(Rm) - Rf)
                beta = beta_data.get('beta', np.nan)
                if not np.isnan(beta) and market_expected_return is not None:
                    theoretical_return = span_risk_free_rate + beta * (market_expected_return - span_risk_free_rate)
                    # 年率化（スパンに応じて）
                    span = self.span_combo.currentText() if hasattr(self, 'span_combo') else '日次'
                    span_factors = {
                        '日次': 365,
                        '週次': 52,
                        '月次': 12,
                        '年次': 1
                    }
                    annualization_factor = span_factors.get(span, 365)
                    theoretical_return_annual = theoretical_return * annualization_factor

                    item = QTableWidgetItem(f"{theoretical_return_annual * 100:.3f}")
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    self.capm_table.setItem(row, 2, item)

                    # 差異（実測 - 理論）
                    if not np.isnan(actual_return):
                        difference = actual_return - theoretical_return_annual
                        item = QTableWidgetItem(f"{difference * 100:.3f}")
                        item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)

                        # 差異の色分け
                        if difference > 0:
                            item.setForeground(QColor("#00ff00"))  # 実測が理論より高い（良好）
                        elif difference < 0:
                            item.setForeground(QColor("#ff0000"))  # 実測が理論より低い（注意）
                        else:
                            item.setForeground(QColor("#ffffff"))

                        self.capm_table.setItem(row, 3, item)
                    else:
                        item = QTableWidgetItem("-")
                        item.setTextAlignment(Qt.AlignCenter)
                        self.capm_table.setItem(row, 3, item)
                else:
                    # 理論リターン
                    item = QTableWidgetItem("-")
                    item.setTextAlignment(Qt.AlignCenter)
                    self.capm_table.setItem(row, 2, item)
                    # 差異
                    item = QTableWidgetItem("-")
                    item.setTextAlignment(Qt.AlignCenter)
                    self.capm_table.setItem(row, 3, item)

                # β値
                if not np.isnan(beta):
                    item = QTableWidgetItem(f"{beta:.3f}")
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)

                    # β値の色分け
                    if beta > 1:
                        item.setForeground(QColor("#ff0000"))  # 市場より高リスク
                    elif beta < 1:
                        item.setForeground(QColor("#00ffff"))  # 市場より低リスク
                    else:
                        item.setForeground(QColor("#ffffff"))

                    self.capm_table.setItem(row, 4, item)
                else:
                    item = QTableWidgetItem("-")
                    item.setTextAlignment(Qt.AlignCenter)
                    self.capm_table.setItem(row, 4, item)

                # α値
                alpha = beta_data.get('alpha', np.nan)
                if not np.isnan(alpha):
                    # αも年率化
                    alpha_annual = alpha * annualization_factor
                    item = QTableWidgetItem(f"{alpha_annual * 100:.3f}")
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)

                    # α値の色分け
                    if alpha_annual > 0:
                        item.setForeground(QColor("#00ff00"))  # 正のα（超過収益）
                    elif alpha_annual < 0:
                        item.setForeground(QColor("#ff0000"))  # 負のα
                    else:
                        item.setForeground(QColor("#ffffff"))

                    self.capm_table.setItem(row, 5, item)
                else:
                    item = QTableWidgetItem("-")
                    item.setTextAlignment(Qt.AlignCenter)
                    self.capm_table.setItem(row, 5, item)

            # カラム幅を調整
            self.capm_table.resizeColumnsToContents()

            self.logger.info("実測vs理論テーブルを更新しました")

        except Exception as e:
            self.logger.error(f"CAPMテーブル更新エラー: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

    def _populate_table_with_metrics(self, table, all_metrics, all_efficiency, headers):
        """テーブルにメトリクスデータを設定するヘルパー関数"""
        try:
            portfolios = list(all_metrics.keys())
            table.setRowCount(len(portfolios))
            table.setColumnCount(len(headers))
            table.setHorizontalHeaderLabels(headers)

            for row, portfolio_name in enumerate(portfolios):
                metrics = all_metrics[portfolio_name]
                efficiency = all_efficiency.get(portfolio_name, {})

                # ポートフォリオ名
                name_item = QTableWidgetItem(portfolio_name)
                table.setItem(row, 0, name_item)

                # 各カラムのデータを設定
                col = 1
                if "年率リターン" in headers:
                    val = metrics.get('annualized_return', np.nan)
                    item = QTableWidgetItem(f"{val * 100:.2f}%" if not np.isnan(val) else "N/A")
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    table.setItem(row, col, item)
                    col += 1

                if "年率リスク" in headers:
                    val = metrics.get('annualized_std', np.nan)
                    item = QTableWidgetItem(f"{val * 100:.2f}%" if not np.isnan(val) else "N/A")
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    table.setItem(row, col, item)
                    col += 1

                if "シャープレシオ" in headers:
                    val = efficiency.get('annualized_sharpe', np.nan)
                    item = QTableWidgetItem(f"{val:.3f}" if not np.isnan(val) else "N/A")
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    table.setItem(row, col, item)
                    col += 1

                if "ソルティノレシオ" in headers:
                    val = efficiency.get('sortino_ratio', np.nan)
                    item = QTableWidgetItem(f"{val:.3f}" if not np.isnan(val) else "N/A")
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    table.setItem(row, col, item)
                    col += 1

                if "最大DD" in headers:
                    val = metrics.get('max_drawdown', np.nan)
                    item = QTableWidgetItem(f"{val * 100:.2f}%" if not np.isnan(val) else "N/A")
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    table.setItem(row, col, item)
                    col += 1

            table.resizeColumnsToContents()

        except Exception as e:
            self.logger.error(f"テーブルデータ設定エラー: {e}")

    def _show_placeholder(self):
        """プレースホルダーを表示"""
        pass
    
    def _clear_analysis_results(self):
        """分析結果をクリア"""
        try:
            # 確認ダイアログ
            reply = QMessageBox.question(
                self,
                "結果クリア確認",
                "分析結果をクリアしますか？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply != QMessageBox.Yes:
                return
            
            # 分析結果をクリア
            self.analysis_results = None
            
            # 一時ファイルを削除
            for temp_file in self.temp_html_files:
                try:
                    os.unlink(temp_file)
                except (OSError, FileNotFoundError) as e:
                    logging.debug(f"Failed to delete temp file {temp_file}: {e}")
                    pass
            self.temp_html_files.clear()
            
            if hasattr(self, 'chart_files'):
                self.chart_files.clear()
            
            # テーブルをクリア
            self.summary_table.setRowCount(0)
            self.summary_table.setColumnCount(0)
            
            # 全てのチャートボタンを無効化
            self.scatter_browser_button.setEnabled(False)
            self.scatter_export_button.setEnabled(False)
            
            self.heatmap_browser_button.setEnabled(False)
            self.heatmap_export_button.setEnabled(False)
            
            self.radar_browser_button.setEnabled(False)
            self.radar_export_button.setEnabled(False)
            
            if hasattr(self, 'frontier_browser_button'):
                self.frontier_browser_button.setEnabled(False)
                self.frontier_export_button.setEnabled(False)
            if hasattr(self, 'frontier_placeholder'):
                self.frontier_placeholder.setVisible(True)
            if hasattr(self, 'frontier_canvas'):
                self.frontier_canvas.clear_plot()
                self.frontier_canvas.setVisible(False)

            if hasattr(self, 'market_browser_button'):
                self.market_browser_button.setEnabled(False)
                self.market_export_button.setEnabled(False)
            if hasattr(self, 'market_placeholder'):
                self.market_placeholder.setVisible(True)
            
            self.price_browser_button.setEnabled(False)
            self.price_export_button.setEnabled(False)
            
            self.capm_browser_button.setEnabled(False)
            self.capm_export_button.setEnabled(False)
            
            # クリアボタンを無効化
            self.clear_button.setEnabled(False)
            
            self.logger.info("分析結果をクリアしました")
            
        except Exception as e:
            self.logger.error(f"結果クリアエラー: {e}")
            QMessageBox.warning(self, "エラー", f"結果のクリア中にエラーが発生しました:\n{str(e)}")
    
    def _create_all_plotly_charts(self):
        """全てのPlotlyチャートを生成"""
        if not PLOTLY_AVAILABLE:
            QMessageBox.warning(self, "警告", "Plotlyが利用できないため，インタラクティブチャートは表示できません．")
            return
        
        try:
            # 古い一時ファイルを削除
            for temp_file in self.temp_html_files:
                try:
                    os.unlink(temp_file)
                except (OSError, FileNotFoundError) as e:
                    logging.debug(f"Failed to delete temp file {temp_file}: {e}")
                    pass
            self.temp_html_files.clear()
            self._create_summary_plotly_chart()
            # 各チャートを生成
            self._create_scatter_chart()
            self._create_heatmap_chart()
            self._create_radar_chart()
            self._create_frontier_comparison_chart()
            self._create_market_comparison_chart()
            self._create_price_evolution_chart()
            self._create_actual_vs_theoretical_chart()
            
        except Exception as e:
            self.logger.error(f"Plotlyチャート生成エラー: {e}", exc_info=True)
            QMessageBox.warning(self, "警告", f"チャートの生成中にエラーが発生しました:\n{str(e)}")
    
    def _create_summary_plotly_chart(self):
        """サマリーPlotlyチャートを生成"""
        try:
            all_metrics = self.analysis_results['metrics']
            all_efficiency = self.analysis_results['efficiency']
            
            # summary_statsモジュールを使用してPlotlyチャート生成
            fig = summary_stats.create_summary_plotly_chart(
                all_metrics,
                all_efficiency,
                self.config
            )
            
            if fig:
                self._save_chart_to_temp_file(fig, 'summary')
                
                # ボタンを有効化
                if hasattr(self, 'summary_plotly_browser_button'):
                    self.summary_plotly_browser_button.setEnabled(True)
                if hasattr(self, 'summary_plotly_export_button'):
                    self.summary_plotly_export_button.setEnabled(True)
                if hasattr(self, 'summary_plotly_placeholder'):
                    self.summary_plotly_placeholder.setVisible(False)
                
                self.logger.info("Plotlyサマリーチャートを生成しました")
            
        except Exception as e:
            self.logger.error(f"Plotlyサマリーチャート生成エラー: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

    def _create_scatter_chart(self):
        """リスク・リターン散布図を生成"""
        try:
            all_metrics = self.analysis_results['metrics']
            all_efficiency = self.analysis_results['efficiency']
            
            # scatter_plotモジュールを使用してPlotlyチャート生成
            fig = scatter_plot.create_scatter_plotly_chart(
                all_metrics,
                all_efficiency,
                self.config
            )
            
            if fig:
                self._save_chart_to_temp_file(fig, 'scatter')
                
                # ボタンを有効化
                self.scatter_browser_button.setEnabled(True)
                self.scatter_export_button.setEnabled(True)
                
                self.logger.info("散布図のPlotlyチャートを生成しました")
            
        except Exception as e:
            self.logger.error(f"散布図生成エラー: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    def _create_heatmap_chart(self):
        """パフォーマンスヒートマップを生成"""
        try:
            all_metrics = self.analysis_results['metrics']
            all_efficiency = self.analysis_results['efficiency']
            
            # heatmapモジュールを使用してPlotlyチャート生成
            fig = heatmap.create_heatmap_plotly_chart(
                all_metrics,
                all_efficiency,
                self.config
            )
            
            if fig:
                self._save_chart_to_temp_file(fig, 'heatmap')
                
                # ボタンを有効化
                self.heatmap_browser_button.setEnabled(True)
                self.heatmap_export_button.setEnabled(True)
                
                self.logger.info("ヒートマップのPlotlyチャートを生成しました")
            
        except Exception as e:
            self.logger.error(f"ヒートマップ生成エラー: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    def _create_radar_chart(self):
        """レーダーチャートを生成"""
        try:
            all_metrics = self.analysis_results['metrics']
            all_efficiency = self.analysis_results['efficiency']
            
            # radar_chartモジュールを使用してPlotlyチャート生成
            fig = radar_chart.create_radar_plotly_chart(
                all_metrics,
                all_efficiency,
                self.config
            )
            
            if fig:
                self._save_chart_to_temp_file(fig, 'radar')
                
                # ボタンを有効化
                self.radar_browser_button.setEnabled(True)
                self.radar_export_button.setEnabled(True)
                
                self.logger.info("レーダーチャートのPlotlyチャートを生成しました")
            
        except Exception as e:
            self.logger.error(f"レーダーチャート生成エラー: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    def _create_frontier_comparison_chart(self):
        """効率的フロンティア比較チャートを生成"""
        try:
            if not PLOTLY_AVAILABLE:
                self.logger.warning("Plotlyが利用できません")
                return

            portfolios = list(self.selected_portfolios.values())
            price_data = self.analysis_results.get('price_data', {})
            all_metrics = self.analysis_results.get('metrics', {})

            if not price_data:
                self.logger.warning("価格データがありません")
                return

            if len(portfolios) == 0:
                self.logger.warning("選択されたポートフォリオがありません")
                return

            # 無リスク利子率を取得（分析条件から）
            risk_free_rate = 0.0
            if hasattr(self, 'interest_rate_spin'):
                risk_free_rate = self.interest_rate_spin.value() / 100.0
                self.logger.info(f"無リスク利子率（年率）: {risk_free_rate:.4f} ({risk_free_rate*100:.2f}%)")

            # スパン調整
            span = self.span_combo.currentText() if hasattr(self, 'span_combo') else '日次'
            span_risk_free_rate = self._convert_risk_free_rate_to_span(risk_free_rate, span)
            self.logger.info(f"無リスク利子率（{span}調整後）: {span_risk_free_rate:.6f} ({span_risk_free_rate*100:.4f}%)")

            # Plotlyチャートを作成（frontier_comparisonモジュールを使用）
            fig = frontier_comparison.create_frontier_comparison_plotly_chart(
                portfolios,
                price_data,
                all_metrics,
                span_risk_free_rate,
                self.config
            )

            if fig:
                # チャートを保存
                self._save_chart_to_temp_file(fig, 'frontier')

                # ボタンを有効化
                if hasattr(self, 'frontier_browser_button'):
                    self.frontier_browser_button.setEnabled(True)
                if hasattr(self, 'frontier_export_button'):
                    self.frontier_export_button.setEnabled(True)
                if hasattr(self, 'frontier_placeholder'):
                    self.frontier_placeholder.setVisible(False)

                self.logger.info("効率的フロンティア比較Plotlyチャートを生成しました")

            # Matplotlibチャートを作成（アプリ内表示用）
            if MATPLOTLIB_AVAILABLE and hasattr(self, 'frontier_canvas'):
                matplotlib_fig = frontier_comparison.create_frontier_comparison_matplotlib_chart(
                    portfolios,
                    price_data,
                    all_metrics,
                    span_risk_free_rate,
                    self.config
                )

                if matplotlib_fig:
                    self.frontier_canvas.display_figure(matplotlib_fig)
                    self.frontier_canvas.setVisible(True)
                    self.logger.info("効率的フロンティア比較Matplotlibチャートを生成しました")

        except Exception as e:
            self.logger.error(f"フロンティア比較チャート生成エラー: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

    def _convert_risk_free_rate_to_span(self, annual_rate: float, span: str) -> float:
        """年率の無リスク利子率をスパンに応じて変換"""
        try:
            span_factors = {
                '日次': 1/365,
                '週次': 1/52,
                '月次': 1/12,
                '年次': 1
            }
            factor = span_factors.get(span, 1/365)
            span_rate = annual_rate * factor
            self.logger.debug(f"無リスク利子率変換: {annual_rate:.4f} (年率) × {factor:.6f} ({span}) = {span_rate:.6f}")
            return span_rate
        except Exception as e:
            self.logger.error(f"無リスク利子率変換エラー: {e}")
            return 0.0
    
    def _create_market_comparison_chart(self):
        """市場ポートフォリオとの比較チャートを生成"""
        try:
            from analysis.portfolio_comparison import market_comparison
            from analysis.portfolio_comparison import PortfolioMetricsCalculator

            portfolios = list(self.selected_portfolios.values())
            price_data = self.analysis_results.get('price_data', {})

            if not price_data:
                self.logger.warning("価格データがありません")
                return

            # 分析条件を取得
            start_date = self.start_date_edit.date().toPython()
            end_date = self.end_date_edit.date().toPython()
            interest_rate = self.interest_rate_spin.value() / 100.0
            span = self.span_combo.currentText()
            market_ticker = self._get_market_ticker()

            if not market_ticker:
                self.logger.warning("市場ティッカーが指定されていません")
                return

            # スパンに応じた無リスク利子率の変換
            span_factors = {
                '日次': 1/365,
                '週次': 1/52,
                '月次': 1/12,
                '年次': 1
            }
            span_risk_free_rate = interest_rate * span_factors.get(span, 1/365)

            # 市場データを取得
            market_metrics = self._fetch_market_metrics(
                market_ticker, start_date, end_date, span, interest_rate
            )

            if not market_metrics:
                self.logger.warning("市場データの取得に失敗しました")
                return

            # 各ポートフォリオのリターンを計算
            calculator = PortfolioMetricsCalculator()
            portfolio_returns = {}

            for portfolio in portfolios:
                pf_return = self._calculate_portfolio_returns(portfolio, price_data)
                if pf_return is not None:
                    portfolio_returns[portfolio.name] = pf_return

            if not portfolio_returns:
                self.logger.warning("ポートフォリオリターンの計算に失敗しました")
                return

            # 市場リターンを取得
            import yfinance as yf

            interval_map = {
                '日次': '1d',
                '週次': '1wk',
                '月次': '1mo'
            }
            interval = interval_map.get(span, '1d')

            ticker = yf.Ticker(market_ticker)
            hist = ticker.history(
                start=start_date,
                end=end_date + timedelta(days=1),
                interval=interval,
                auto_adjust=False,
                prepost=False,
                repair=True
            )

            if hist.empty:
                self.logger.warning(f"市場データが取得できませんでした: {market_ticker}")
                return

            # 調整済み終値を取得
            if 'Adj Close' in hist.columns and not hist['Adj Close'].isna().all():
                market_data = hist['Adj Close'].copy()
            elif 'Close' in hist.columns:
                market_data = hist['Close'].copy()
            else:
                self.logger.warning(f"市場データに価格カラムが見つかりません")
                return

            # タイムゾーン処理
            if hasattr(market_data.index, 'tz') and market_data.index.tz is not None:
                market_data.index = market_data.index.tz_localize(None)

            # 日付のみに正規化
            market_data.index = market_data.index.normalize()
            market_data = market_data.dropna()

            if market_data.empty:
                self.logger.warning("市場データが空です")
                return

            # 市場リターンを計算
            returns_data = calculator.calculate_returns_from_prices({
                market_ticker: market_data
            })

            market_returns = returns_data.get(market_ticker)
            if market_returns is None or market_returns.empty:
                self.logger.warning("市場リターンの計算に失敗しました")
                return

            # 市場期待利益率
            market_expected_return = market_returns.mean()

            # ポートフォリオ比較分析
            comparator = market_comparison.PortfolioMarketComparison()
            portfolio_beta_results = comparator.calculate_portfolio_betas(
                portfolio_returns,
                market_returns,
                span_risk_free_rate
            )

            if not portfolio_beta_results:
                self.logger.warning("β値の計算に失敗しました")
                return

            # 証券市場線の計算
            sml_line = comparator.calculate_security_market_line(
                span_risk_free_rate,
                market_expected_return
            )

            # Plotlyチャート生成
            fig = market_comparison.create_market_comparison_plotly_chart(
                portfolio_beta_results,
                sml_line,
                market_expected_return,
                span_risk_free_rate,
                self.config
            )

            if fig:
                self._save_chart_to_temp_file(fig, 'market')

                # ボタンを有効化
                self.market_browser_button.setEnabled(True)
                self.market_export_button.setEnabled(True)

                self.logger.info("市場比較のPlotlyチャートを生成しました")

            # Matplotlibチャート生成（アプリ内表示用）
            if MATPLOTLIB_AVAILABLE and hasattr(self, 'market_matplotlib_canvas'):
                self._update_market_matplotlib_chart(
                    portfolio_beta_results,
                    sml_line,
                    market_expected_return,
                    span_risk_free_rate
                )

            # テーブル更新
            self._update_market_comparison_table(portfolio_beta_results, market_expected_return)

            # ベータ値データを分析結果に保存（実測vs理論値タブで使用）
            if not hasattr(self, 'analysis_results') or self.analysis_results is None:
                self.analysis_results = {}

            self.analysis_results['beta_results'] = portfolio_beta_results
            self.analysis_results['market_expected_return'] = market_expected_return
            self.analysis_results['span_risk_free_rate'] = span_risk_free_rate

            self.logger.info(f"ベータ値データを保存しました（{len(portfolio_beta_results)}ポートフォリオ）")

        except Exception as e:
            self.logger.error(f"市場比較チャート生成エラー: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    def _create_price_evolution_chart(self):
        """価格推移チャートを生成"""
        try:
            portfolios = list(self.selected_portfolios.values())
            price_data = self.analysis_results.get('price_data', {})
            
            if not price_data:
                self.logger.warning("価格データがありません")
                return
            
            # price_evolutionモジュールを使用してPlotlyチャート生成
            fig = price_evolution.create_price_evolution_plotly_chart(
                portfolios,
                price_data,
                self.config
            )
            
            if fig:
                self._save_chart_to_temp_file(fig, 'price')
                
                # ボタンを有効化
                self.price_browser_button.setEnabled(True)
                self.price_export_button.setEnabled(True)
                
                self.logger.info("価格推移のPlotlyチャートを生成しました")
            
        except Exception as e:
            self.logger.error(f"価格推移チャート生成エラー: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    def _calculate_portfolio_returns(self, portfolio, price_data):
        """ポートフォリオリターンを計算"""
        try:
            from analysis.portfolio_comparison import PortfolioMetricsCalculator
            
            calculator = PortfolioMetricsCalculator()
            
            # 価格データからリターンを計算
            returns_data = calculator.calculate_returns_from_prices(price_data)
            
            # ポートフォリオのウエイトベクトルを取得
            weights = {}
            for position in portfolio.positions:
                if position.asset.symbol in returns_data:
                    weights[position.asset.symbol] = position.weight
            
            # ウエイトを正規化
            total_weight = sum(weights.values())
            if total_weight == 0:
                return None
            
            for ticker in weights:
                weights[ticker] /= total_weight
            
            # ポートフォリオリターンを計算
            portfolio_returns = pd.Series(0.0, index=list(returns_data.values())[0].index)
            
            for ticker, weight in weights.items():
                if ticker in returns_data:
                    portfolio_returns += weight * returns_data[ticker]
            
            return portfolio_returns
            
        except Exception as e:
            self.logger.error(f"ポートフォリオリターン計算エラー: {e}")
            return None
    
    def _create_actual_vs_theoretical_chart(self):
        """実測値 vs 理論値チャートを生成"""
        try:
            # ベータ値データを取得
            beta_results = self.analysis_results.get('beta_results')
            market_expected_return = self.analysis_results.get('market_expected_return')
            span_risk_free_rate = self.analysis_results.get('span_risk_free_rate')

            if not beta_results:
                self.logger.info("実測vs理論チャートにはベータ値データが必要です（市場比較タブで計算されます）")
                return

            # メトリクスデータを取得
            all_metrics = self.analysis_results.get('metrics', {})

            if not all_metrics:
                self.logger.warning("メトリクスデータがありません")
                return

            # ベータ値辞書を作成（actual_vs_theoreticalモジュールで必要な形式）
            betas = {}
            for pf_name, beta_data in beta_results.items():
                if beta_data.get('success', False):
                    betas[pf_name] = beta_data['beta']

            if not betas:
                self.logger.warning("有効なベータ値がありません")
                return

            # Plotlyチャート生成
            fig = actual_vs_theoretical.create_actual_vs_theoretical_plotly_chart(
                all_metrics,
                betas,
                span_risk_free_rate,
                market_expected_return,
                self.config
            )

            if fig:
                self._save_chart_to_temp_file(fig, 'capm')

                # ボタンを有効化
                if hasattr(self, 'capm_browser_button'):
                    self.capm_browser_button.setEnabled(True)
                if hasattr(self, 'capm_export_button'):
                    self.capm_export_button.setEnabled(True)

                self.logger.info("実測vs理論のPlotlyチャートを生成しました")

        except Exception as e:
            self.logger.error(f"実測vs理論チャート生成エラー: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    def _update_market_matplotlib_chart(
        self,
        portfolio_beta_results,
        sml_line,
        market_expected_return,
        risk_free_rate
    ):
        """市場比較のMatplotlibチャートを更新"""
        try:
            from analysis.portfolio_comparison import market_comparison

            fig = market_comparison.create_market_comparison_matplotlib_chart(
                portfolio_beta_results,
                sml_line,
                market_expected_return,
                risk_free_rate,
                self.config
            )

            if fig and hasattr(self, 'market_matplotlib_canvas'):
                self.market_matplotlib_canvas.display_figure(fig)
                self.logger.info("市場比較のMatplotlibチャートを更新しました")

        except Exception as e:
            self.logger.error(f"Matplotlibチャート更新エラー: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

    def _update_market_comparison_table(self, portfolio_beta_results, market_expected_return):
        """市場比較テーブルを更新"""
        try:
            # 有効なデータのみを取得
            valid_data = {name: data for name, data in portfolio_beta_results.items()
                         if data.get('success', False)}

            if not valid_data:
                return

            # 指標リスト（行ヘッダー）
            metrics = [
                ("β値", "beta"),
                ("β標準誤差", "beta_std_error"),
                ("βp値", "beta_p_value"),
                ("β信頼区間下限", "beta_ci_lower"),
                ("β信頼区間上限", "beta_ci_upper"),
                ("α値", "alpha"),
                ("期待利益率", "expected_return"),
                ("CAPM期待収益率", "capm_expected_return"),
                ("相関係数", "correlation"),
                ("R²", "r_squared"),
                ("システマティックリスク", "systematic_risk"),
                ("アンシステマティックリスク", "unsystematic_risk"),
                ("総リスク", "total_risk"),
                ("残差標準誤差", "residual_std_error"),
                ("共通日数", "common_dates")
            ]

            # ポートフォリオ名リスト（列ヘッダー）
            pf_names = sorted(valid_data.keys())

            # テーブル設定
            self.market_table.setRowCount(len(metrics))
            self.market_table.setColumnCount(len(pf_names))
            self.market_table.setVerticalHeaderLabels([m[0] for m in metrics])
            self.market_table.setHorizontalHeaderLabels(pf_names)

            # データ挿入（行=指標，列=ポートフォリオ）
            for row, (metric_name, metric_key) in enumerate(metrics):
                for col, pf_name in enumerate(pf_names):
                    data = valid_data[pf_name]
                    value = data.get(metric_key, np.nan)

                    if pd.isna(value) or np.isinf(value):
                        item = QTableWidgetItem("-")
                        item.setForeground(QColor("#888888"))
                    else:
                        # フォーマット設定
                        if metric_key == "common_dates":
                            item = QTableWidgetItem(f"{int(value)}日")
                        elif metric_key in ["beta_p_value"]:
                            if value < 0.001:
                                item = QTableWidgetItem("< 0.001")
                            else:
                                item = QTableWidgetItem(f"{value:.3f}")

                            # p値の色分け
                            if value < 0.01:
                                item.setForeground(QColor("#00ff00"))  # 高い有意性
                            elif value < 0.05:
                                item.setForeground(QColor("#ffff00"))  # 有意
                            else:
                                item.setForeground(QColor("#ff0000"))  # 有意でない
                        elif metric_key in ["beta", "correlation", "r_squared", "beta_std_error",
                                          "beta_ci_lower", "beta_ci_upper", "residual_std_error"]:
                            item = QTableWidgetItem(f"{value:.3f}")

                            # 特別な色分け
                            if metric_key == "beta":
                                if value > 1:
                                    item.setForeground(QColor("#ff0000"))
                                elif value < 1:
                                    item.setForeground(QColor("#00ffff"))
                                else:
                                    item.setForeground(QColor("#ffffff"))
                            else:
                                item.setForeground(QColor("#ffffff"))
                        elif metric_key in ["alpha", "expected_return", "capm_expected_return",
                                          "systematic_risk", "unsystematic_risk", "total_risk"]:
                            item = QTableWidgetItem(f"{value*100:.3f}%")

                            # α値の色分け
                            if metric_key == "alpha":
                                if value > 0:
                                    item.setForeground(QColor("#00ff00"))
                                elif value < 0:
                                    item.setForeground(QColor("#ff0000"))
                                else:
                                    item.setForeground(QColor("#ffffff"))
                            else:
                                item.setForeground(QColor("#ffffff"))
                        else:
                            item = QTableWidgetItem(f"{value:.3f}")
                            item.setForeground(QColor("#ffffff"))

                    item.setTextAlignment(Qt.AlignCenter)
                    item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)

                    # ツールチップ設定
                    tooltip = f"{pf_name} の {metric_name}: "
                    if metric_key == "common_dates":
                        tooltip += f"{int(value)}日"
                    elif pd.isna(value) or np.isinf(value):
                        tooltip += "データなし"
                    elif metric_key in ["expected_return", "capm_expected_return", "alpha",
                                      "systematic_risk", "unsystematic_risk", "total_risk"]:
                        tooltip += f"{value*100:.4f}%"
                    else:
                        tooltip += f"{value:.4f}"

                    # 統計的解釈を追加
                    if metric_key == "beta_p_value" and not pd.isna(value):
                        if value < 0.01:
                            tooltip += " (高い有意性)"
                        elif value < 0.05:
                            tooltip += " (有意)"
                        else:
                            tooltip += " (有意でない)"

                    item.setToolTip(tooltip)
                    self.market_table.setItem(row, col, item)

            # カラム幅調整
            self.market_table.resizeColumnsToContents()
            self.market_table.resizeRowsToContents()

            # 最小/最大幅の設定
            for col in range(self.market_table.columnCount()):
                current_width = self.market_table.columnWidth(col)
                self.market_table.setColumnWidth(col, min(max(current_width, 80), 150))

            # 行ヘッダーの幅を適切に設定
            header = self.market_table.verticalHeader()
            header.setMinimumWidth(140)
            header.setMaximumWidth(180)

            self.logger.info("市場比較テーブルを更新しました")

        except Exception as e:
            self.logger.error(f"市場比較テーブル更新エラー: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

    def _save_chart_to_temp_file(self, fig, chart_type: str):
        """チャートを一時ファイルに保存"""
        try:
            temp_file = tempfile.NamedTemporaryFile(
                mode='w', suffix='.html', delete=False, encoding='utf-8'
            )
            
            html_content = fig.to_html(
                include_plotlyjs='cdn',
                config={
                    'displayModeBar': True,
                    'displaylogo': False,
                    'responsive': True
                }
            )
            
            temp_file.write(html_content)
            temp_file.close()
            
            # 一時ファイルリストに追加
            if not hasattr(self, 'chart_files'):
                self.chart_files = {}
            self.chart_files[chart_type] = temp_file.name
            self.temp_html_files.append(temp_file.name)
            
        except Exception as e:
            self.logger.error(f"チャート保存エラー ({chart_type}): {e}")
    
    def _open_chart_in_browser(self, chart_type: str):
        """チャートをブラウザで開く"""
        try:
            if not hasattr(self, 'chart_files') or chart_type not in self.chart_files:
                QMessageBox.warning(self, "エラー", "チャートファイルが見つかりません．")
                return
            
            file_path = self.chart_files[chart_type]
            webbrowser.open(f'file://{file_path}')
            
        except Exception as e:
            self.logger.error(f"ブラウザ表示エラー: {e}")
            QMessageBox.warning(self, "エラー", f"ブラウザでの表示に失敗しました:\n{str(e)}")
    
    def _export_chart(self, chart_type: str):
        """チャートをエクスポート"""
        try:
            if not hasattr(self, 'chart_files') or chart_type not in self.chart_files:
                QMessageBox.warning(self, "エラー", "チャートファイルが見つかりません．")
                return
            
            from PySide6.QtWidgets import QFileDialog
            
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "HTMLファイルを保存",
                f"portfolio_comparison_{chart_type}.html",
                "HTML Files (*.html)"
            )
            
            if file_path:
                import shutil
                shutil.copy(self.chart_files[chart_type], file_path)
                QMessageBox.information(self, "保存完了", f"チャートを保存しました:\n{file_path}")
                
        except Exception as e:
            self.logger.error(f"エクスポートエラー: {e}")
            QMessageBox.warning(self, "エラー", f"エクスポートに失敗しました:\n{str(e)}")