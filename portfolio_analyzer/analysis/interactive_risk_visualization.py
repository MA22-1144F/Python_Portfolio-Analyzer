"""interactive_risk_visualization.py"""

import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Optional
import tempfile
import os
import webbrowser

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QComboBox, QFileDialog, QMessageBox,
    QListWidget, QListWidgetItem
)
from PySide6.QtCore import Qt, QTimer, QThread, Signal
from analysis.analysis_base_widget import AnalysisStyles
from config.app_config import AppConfig

warnings.filterwarnings('ignore')


class BrowserLaunchThread(QThread):
    launch_completed = Signal(bool, str)
    
    def __init__(self, file_path: str):
        super().__init__()
        self.file_path = file_path
    
    def run(self):
        try:
            webbrowser.open(f"file:///{self.file_path}")
            self.launch_completed.emit(True, "ブラウザで表示しました")
        except Exception as e:
            self.launch_completed.emit(False, f"ブラウザ起動エラー: {str(e)}")


class AssetSelectionWidget(QWidget):
    selection_changed = Signal()
    
    def __init__(self, config=None):
        super().__init__()
        # 設定を取得
        if config is None:
            config = AppConfig()
        self.config = config
        
        self.all_symbols = []
        self.styles = AnalysisStyles(config)
        self.setup_ui()
    
    def get_parent_analysis_widget(self):
        """親のAnalysisBaseWidgetを取得"""
        parent = self.parent()
        while parent:
            if hasattr(parent, 'create_button') and hasattr(parent, 'styles'):
                return parent
            parent = parent.parent()
        return None
    
    def create_button(self, text: str, button_type: str = "primary") -> QPushButton:
        """親ウィジェットのcreate_buttonを利用，見つからない場合は自前実装"""
        parent_widget = self.get_parent_analysis_widget()
        if parent_widget:
            return parent_widget.create_button(text, button_type)
        else:
            # フォールバック：自前実装
            button = QPushButton(text)
            button.setStyleSheet(self.styles.get_button_style_by_type(button_type))
            return button

    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        
        title_label = QLabel("表示する資産を選択:")
        title_label.setStyleSheet(f"QLabel {{ color: {self.styles.COLORS['text_primary']}; font-weight: bold; font-size: 14px; padding: 5px 0px; }}")
        layout.addWidget(title_label)
        
        button_layout = QHBoxLayout()
        
        self.select_all_button = self.create_button("全選択", "secondary")
        self.select_all_button.clicked.connect(self.select_all)
        button_layout.addWidget(self.select_all_button)
        
        self.deselect_all_button = self.create_button("全解除", "neutral")
        self.deselect_all_button.clicked.connect(self.deselect_all)
        button_layout.addWidget(self.deselect_all_button)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        colors = self.config.get_ui_colors()
        self.asset_list = QListWidget()
        self.asset_list.setMinimumHeight(400)
        self.asset_list.setStyleSheet(f"""
            QListWidget {{
                background-color: {colors.get('background', '#2b2b2b')}; 
                color: {colors.get('text_primary', '#ffffff')}; 
                border: 1px solid {colors.get('border', '#555555')};
                border-radius: 4px; 
                font-size: 10px; 
                font-weight: bold;
                outline: 0; /* フォーカス時の点線枠を削除 */
            }}
            QListWidget::item {{
                padding: 5px; 
                border-bottom: 1px solid {colors.get('grid_line', '#444444')}; 
                min-height: 20px;
                color: {colors.get('text_primary', '#ffffff')};
            }}

            QListWidget::item:selected {{ 
                background-color: transparent; 
                color: {colors.get('text_primary', '#ffffff')};
            }}
            QListWidget::item:selected:hover {{
                background-color: {colors.get('surface', '#4c4c4c')};
            }}
            QListWidget::item:hover {{ 
                background-color: {colors.get('surface', '#4c4c4c')}; 
            }}

            QListWidget::indicator {{
                width: 14px;
                height: 14px;
                border-radius: 2px; /* 少し角を丸める */
                border: 1px solid {colors.get('border', '#555555')}; /* 未選択時の枠線の色 */
            }}

            /* 未選択時のスタイル（中は透明） */
            QListWidget::indicator:unchecked {{
                background-color: transparent;
            }}
            QListWidget::indicator:unchecked:hover {{
                border: 1px solid {colors.get('primary', '#0078d4')}; /* ホバー時に枠を少し強調 */
            }}

            /* 選択済みのスタイル（中を青で塗りつぶす） */
            QListWidget::indicator:checked {{
                background-color: {colors.get('primary', '#0078d4')}; /* ここで青く塗りつぶし */
                border: 1px solid {colors.get('primary', '#0078d4')}; /* 枠線も青に合わせる */
            }}
        """)
        layout.addWidget(self.asset_list)
        
        self.setLayout(layout)
    
    def update_assets(self, symbols: List[str]):
        self.all_symbols = symbols
        self.asset_list.clear()
        
        for symbol in symbols:
            item = QListWidgetItem(symbol)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            self.asset_list.addItem(item)
        
        self.asset_list.itemChanged.connect(lambda: self.selection_changed.emit())
    
    def get_selected_symbols(self) -> List[str]:
        selected = []
        for i in range(self.asset_list.count()):
            item = self.asset_list.item(i)
            if item.checkState() == Qt.Checked:
                selected.append(item.text())
        return selected
    
    def select_all(self):
        for i in range(self.asset_list.count()):
            self.asset_list.item(i).setCheckState(Qt.Checked)
        self.selection_changed.emit()
    
    def deselect_all(self):
        for i in range(self.asset_list.count()):
            self.asset_list.item(i).setCheckState(Qt.Unchecked)
        self.selection_changed.emit()


class InteractiveRiskVisualizationWidget(QWidget):
    def __init__(self, config=None):
        super().__init__()
        # 設定を取得
        if config is None:
            config = AppConfig()
        self.config = config
        
        self.risk_results = {}
        self.current_span = 'Daily'
        self.display_span = 'Daily'
        self.current_figure = None
        self.temp_html_files = []
        self.browser_thread = None
        self.asset_selection = None
        self.price_data_source = None
        self.analysis_conditions = None
        self.styles = AnalysisStyles(config)
        self.setup_ui()
    
    def get_parent_analysis_widget(self):
        """親のAnalysisBaseWidgetを取得"""
        parent = self.parent()
        while parent:
            if hasattr(parent, 'create_button') and hasattr(parent, 'styles'):
                return parent
            parent = parent.parent()
        return None
    
    def create_button(self, text: str, button_type: str = "primary") -> QPushButton:
        """親ウィジェットのcreate_buttonを利用，見つからない場合は自前実装"""
        parent_widget = self.get_parent_analysis_widget()
        if parent_widget:
            return parent_widget.create_button(text, button_type)
        else:
            # フォールバック：自前実装
            button = QPushButton(text)
            button.setStyleSheet(self.styles.get_button_style_by_type(button_type))
            return button
    
    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        
        if not PLOTLY_AVAILABLE:
            error_label = QLabel("インタラクティブな可視化にはPlotlyが必要です\n\npip install plotly\n\nでインストールしてください")
            error_label.setAlignment(Qt.AlignCenter)
            error_label.setStyleSheet(self.styles.get_empty_state_style().replace(
                self.styles.COLORS["text_secondary"], "#ff6b6b").replace(
                "dashed", "dashed").replace(
                self.styles.COLORS["border"], "#ff6b6b"))
            layout.addWidget(error_label)
            self.setLayout(layout)
            return
        
        control_layout = QHBoxLayout()
        
        self.chart_type_combo = QComboBox()
        self.chart_type_combo.addItems([
            "資産価格時系列",
            "ログリターン分布", 
            "期待利益率・標準偏差 散布図",
            "リスクメトリクス レーダーチャート", 
            "月次期待利益率時系列", 
            "月次標準偏差時系列",
            "月次下方偏差時系列", 
            "月次プラスリターン比率時系列", 
            "月次最大ドローダウン時系列"
        ])
        self.chart_type_combo.currentTextChanged.connect(self.update_visualization)
        self.chart_type_combo.setStyleSheet(self.styles.get_combo_style("250px"))
        
        control_layout.addWidget(QLabel("チャートタイプ:", 
                                       styleSheet=f"color: {self.styles.COLORS['text_primary']}; font-size: 12px;"))
        control_layout.addWidget(self.chart_type_combo)
        control_layout.addStretch()
        
        self.refresh_button = self.create_button("更新", "secondary")
        self.refresh_button.clicked.connect(self.update_visualization)
        control_layout.addWidget(self.refresh_button)
        
        self.browser_button = self.create_button("ブラウザで表示", "primary")
        self.browser_button.clicked.connect(self.open_in_browser)
        control_layout.addWidget(self.browser_button)
        
        self.export_button = self.create_button("HTML保存", "save")
        self.export_button.clicked.connect(self.export_chart)
        control_layout.addWidget(self.export_button)
        
        layout.addLayout(control_layout)
        
        self.asset_selection = AssetSelectionWidget()
        self.asset_selection.selection_changed.connect(self.update_visualization)
        layout.addWidget(self.asset_selection)
        
        self.status_label = QLabel("チャートタイプを選択して「ブラウザで表示」をクリックしてください．")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet(f"""
            QLabel {{
                background-color: {self.styles.COLORS["surface"]}; 
                color: {self.styles.COLORS["text_primary"]}; 
                border: 1px solid {self.styles.COLORS["border"]};
                border-radius: 4px; 
                padding: 10px; 
                margin: 5px 0px; 
                font-size: 12px;
            }}
        """)
        layout.addWidget(self.status_label)
        
        self.quality_info_label = QLabel("")
        self.quality_info_label.setStyleSheet("color: #17a2b8; font-size: 9px;")
        self.quality_info_label.setVisible(False)
        self.quality_info_label.setWordWrap(True)
        layout.addWidget(self.quality_info_label)
        
        self.browser_button.setEnabled(False)
        self.export_button.setEnabled(False)
        
        self.setLayout(layout)
        self.clear_plot()
    
    def update_data(self, risk_results: dict, current_span: str, display_span: str):
        self.risk_results = risk_results
        self.current_span = current_span
        self.display_span = display_span
        
        valid_symbols = [symbol for symbol, result in risk_results.items() 
                        if result.get('success', False)]
        self.asset_selection.update_assets(valid_symbols)
        
        self.update_quality_info()
        self.update_visualization()
    
    def update_quality_info(self):
        if self.price_data_source and hasattr(self.price_data_source, 'get_data_quality_stats'):
            quality_stats = self.price_data_source.get_data_quality_stats()
            if quality_stats:
                success_count = quality_stats.get('success_count', 0)
                total_count = quality_stats.get('total_count', 0)
                common_dates = quality_stats.get('common_dates', 0)
                
                self.quality_info_label.setText(
                    f"データ: 分析対象 {success_count}/{total_count} 資産, 共通営業日 {common_dates} 日"
                )
                self.quality_info_label.setVisible(True)
    
    def set_data_sources(self, price_data_source, analysis_conditions):
        self.price_data_source = price_data_source
        self.analysis_conditions = analysis_conditions
    
    def set_calculator(self, calculator):
        self.calculator = calculator
    
    def update_visualization(self):
        if not PLOTLY_AVAILABLE or not self.risk_results:
            self.clear_plot()
            return
        
        chart_type = self.chart_type_combo.currentText()
        
        try:
            chart_methods = {
                "資産価格時系列": self.create_asset_price_time_series,
                "ログリターン分布": self.create_log_return_distribution,
                "期待利益率・標準偏差 散布図": self.create_expected_return_vs_std,
                "リスクメトリクス レーダーチャート": self.create_risk_metrics_radar_chart,
                "月次期待利益率時系列": lambda: self.create_time_series_chart("Expected Return"),
                "月次標準偏差時系列": lambda: self.create_time_series_chart("Standard Deviation"),
                "月次下方偏差時系列": lambda: self.create_time_series_chart("Downside Deviation"),
                "月次プラスリターン比率時系列": lambda: self.create_time_series_chart("Positive Return Ratio"),
                "月次最大ドローダウン時系列": lambda: self.create_time_series_chart("Maximum Drawdown")
            }
            
            if chart_type in chart_methods:
                chart_methods[chart_type]()
            
        except Exception as e:
            print(f"可視化エラー: {e}")
            self.clear_plot()
            QMessageBox.warning(self, "可視化エラー", f"チャート作成中にエラーが発生しました:\n{str(e)}")
    
    def get_valid_data(self):
        selected_symbols = self.asset_selection.get_selected_symbols()
        return {symbol: self.risk_results[symbol] for symbol in selected_symbols 
                if symbol in self.risk_results and self.risk_results[symbol].get('success', False)}
    
    def get_price_data_for_selected_assets(self) -> Optional[pd.DataFrame]:
        if not self.price_data_source or not hasattr(self.price_data_source, 'get_analysis_dataframe'):
            return None
        
        price_df = self.price_data_source.get_analysis_dataframe()
        if price_df is None:
            return None
        
        selected_symbols = self.asset_selection.get_selected_symbols()
        available_symbols = [col for col in price_df.columns if col in selected_symbols]
        
        return price_df[available_symbols] if available_symbols else None
    
    def get_log_returns_data(self) -> Optional[Dict[str, pd.Series]]:
        if not self.calculator:
            return None
        
        try:
            log_returns_data = self.calculator.get_log_returns_data()
            selected_symbols = self.asset_selection.get_selected_symbols()
            return {symbol: data for symbol, data in log_returns_data.items() 
                   if symbol in selected_symbols}
        except Exception:
            return None
    
    def get_monthly_metrics_data(self) -> Optional[Dict[str, pd.DataFrame]]:
        if not self.calculator:
            return None
        
        try:
            monthly_data = self.calculator.get_monthly_metrics_data()
            selected_symbols = self.asset_selection.get_selected_symbols()
            return {symbol: data for symbol, data in monthly_data.items() 
                   if symbol in selected_symbols and not data.empty}
        except Exception:
            return None
    
    def create_asset_price_time_series(self):
        price_df = self.get_price_data_for_selected_assets()
        if price_df is None or price_df.empty:
            self.show_no_data_message("価格データが利用できません．")
            return
        
        fig = go.Figure()
        colors = px.colors.qualitative.Set1
        valid_assets = 0
        
        for i, symbol in enumerate(price_df.columns):
            prices = price_df[symbol].dropna()
            if len(prices) < 2:
                continue
            
            fig.add_trace(go.Scatter(
                x=prices.index, y=prices.values, mode='lines', name=symbol,
                line=dict(color=colors[i % len(colors)], width=2),
                hovertemplate=f'<b>{symbol}</b><br>Date: %{{x}}<br>Price: %{{y:.2f}}<extra></extra>'
            ))
            valid_assets += 1
        
        if valid_assets == 0:
            self.show_no_data_message("有効な価格データがありません．")
            return
        
        self._configure_price_chart(fig, valid_assets)
        self.display_figure(fig)
    
    def _configure_price_chart(self, fig, valid_assets):
        fig.update_layout(
            title={'text': f'Asset Price Time Series ({valid_assets} Assets)', 'x': 0.5, 'font': {'size': 18, 'color': 'white'}},
            xaxis=dict(
                title='Date', color='white', gridcolor='rgba(255,255,255,0.3)',
                rangeslider=dict(visible=True, bgcolor='rgba(70,70,70,0.8)', bordercolor='rgba(255,255,255,0.5)', borderwidth=1),
                rangeselector=dict(
                    buttons=[
                        dict(count=1, label="1M", step="month", stepmode="backward"),
                        dict(count=3, label="3M", step="month", stepmode="backward"),
                        dict(count=6, label="6M", step="month", stepmode="backward"),
                        dict(count=1, label="1Y", step="year", stepmode="backward"),
                        dict(step="all")
                    ],
                    bgcolor='rgba(70,70,70,0.8)', activecolor='rgba(0,120,212,0.8)', font=dict(color='white')
                ),
                type='date'
            ),
            yaxis=dict(title='Price', color='white', gridcolor='rgba(255,255,255,0.3)'),
            paper_bgcolor='rgba(43,43,43,1)', plot_bgcolor='rgba(50,50,50,1)',
            font=dict(color='white'), hovermode='x unified',
            legend=dict(bgcolor='rgba(0,0,0,0.7)', bordercolor='white', borderwidth=1)
        )
    
    def create_log_return_distribution(self):
        log_returns_data = self.get_log_returns_data()
        if not log_returns_data:
            self.show_no_data_message("ログリターンデータが利用できません．")
            return
        
        fig = go.Figure()
        colors = px.colors.qualitative.Set1
        valid_assets = 0
        
        for i, (symbol, log_returns) in enumerate(log_returns_data.items()):
            if len(log_returns) < 10:
                continue
            
            mean_return = log_returns.mean() * 100
            std_return = log_returns.std() * 100
            
            fig.add_trace(go.Violin(
                y=log_returns.values * 100, name=symbol, box_visible=True, meanline_visible=True,
                line_color=colors[i % len(colors)],
                fillcolor=colors[i % len(colors)].replace('rgb', 'rgba').replace(')', ', 0.4)'),
                hovertemplate=(f'<b>{symbol}</b><br>Log Return: %{{y:.3f}}%<br>'
                             f'Mean: {mean_return:.3f}%<br>Std: {std_return:.3f}%<extra></extra>'),
                showlegend=True
            ))
            valid_assets += 1
        
        if valid_assets == 0:
            self.show_no_data_message("有効なログリターンデータがありません．")
            return
        
        self._configure_distribution_chart(fig, valid_assets)
        self.display_figure(fig)
    
    def _configure_distribution_chart(self, fig, valid_assets):
        fig.update_layout(
            title={'text': f'Log Return Distribution ({valid_assets} Assets)', 'x': 0.5, 'font': {'size': 18, 'color': 'white'}},
            yaxis=dict(title='Log Return (%)', color='white', gridcolor='rgba(255,255,255,0.3)',
                      zeroline=True, zerolinecolor='rgba(255,255,255,0.5)', zerolinewidth=2),
            xaxis=dict(color='white', gridcolor='rgba(255,255,255,0.1)'),
            paper_bgcolor='rgba(43,43,43,1)', plot_bgcolor='rgba(50,50,50,1)',
            font=dict(color='white'), showlegend=True,
            legend=dict(bgcolor='rgba(0,0,0,0.7)', bordercolor='white', borderwidth=1)
        )
    
    def create_expected_return_vs_std(self):
        valid_data = self.get_valid_data()
        if not valid_data:
            self.show_no_data_message("有効なリスク・リターンデータがありません．")
            return
        
        symbols, returns, risks, sharpe_ratios = [], [], [], []
        
        for symbol, data in valid_data.items():
            mean_return = data.get('expected_return', np.nan)
            std_dev = data.get('standard_deviation', np.nan)
            sharpe = data.get('sharpe_ratio', np.nan)
            
            if pd.isna(mean_return) or pd.isna(std_dev) or pd.isna(sharpe):
                continue
                
            symbols.append(symbol)
            returns.append(mean_return * 100)
            risks.append(std_dev * 100)
            sharpe_ratios.append(sharpe)
        
        if not symbols:
            self.show_no_data_message("有効なリスク・リターンデータがありません．")
            return
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=risks, y=returns, mode='markers+text', text=symbols, textposition="top center",
            marker=dict(
                size=15, color=sharpe_ratios, colorscale='RdYlGn', showscale=True,
                colorbar=dict(title=dict(text="Sharpe Ratio", side="right", font=dict(color='white')),
                             tickfont=dict(color='white')),
                line=dict(width=2, color='white')
            ),
            textfont=dict(color='white', size=10),
            hovertemplate=('<b>%{text}</b><br>Standard Deviation: %{x:.2f}%<br>'
                          'Expected Return: %{y:.2f}%<br>Sharpe Ratio: %{marker.color:.3f}<br><extra></extra>'),
            name="", showlegend=False
        ))
        
        span_risk_free_rate = 0.0
        if valid_data:
            first_asset_data = next(iter(valid_data.values()))
            span_risk_free_rate = first_asset_data.get('risk_free_rate', 0.0)
        
        if span_risk_free_rate != 0.0:
            risk_free_rate_percent = span_risk_free_rate * 100
            fig.add_hline(y=risk_free_rate_percent, line_dash="dash", line_color="yellow", opacity=0.8,
                         annotation_text=f"Risk-free Rate ({risk_free_rate_percent:.3f}%)", 
                         annotation_position="bottom left", annotation_font_color="white")
        
        if risks:
            mean_risk = np.mean(risks)
            fig.add_vline(x=mean_risk, line_dash="dot", line_color="blue", opacity=0.7,
                         annotation_text=f"Average Risk ({mean_risk:.1f}%)", 
                         annotation_position="top left", annotation_font_color="white")
        
        self._configure_scatter_chart(fig, len(symbols))
        self.display_figure(fig)
    
    def _configure_scatter_chart(self, fig, num_assets):
        fig.update_layout(
            title={'text': f'Expected Return vs Standard Deviation ({num_assets} Assets)', 'x': 0.5, 'font': {'size': 18, 'color': 'white'}},
            xaxis=dict(title='Standard Deviation (%)', gridcolor='rgba(255,255,255,0.3)', color='white'),
            yaxis=dict(title='Expected Return (%)', gridcolor='rgba(255,255,255,0.3)', color='white'),
            plot_bgcolor='rgba(50,50,50,1)', paper_bgcolor='rgba(43,43,43,1)',
            font=dict(color='white'), hovermode='closest', showlegend=True,
            legend=dict(bgcolor='rgba(0,0,0,0.7)', bordercolor='white', borderwidth=1)
        )
    
    def create_risk_metrics_radar_chart(self):
        valid_data = self.get_valid_data()
        if not valid_data:
            self.show_no_data_message("有効なリスクメトリクスデータがありません．")
            return
        
        symbols = list(valid_data.keys())
        categories = ['Expected Return', 'Sortino Ratio', 'Max Drawdown', 'Positive %', 'CVaR(95%)']
        
        raw_metrics = {}
        for symbol in symbols:
            data = valid_data[symbol]
            raw_metrics[symbol] = {
                'Expected Return': data.get('expected_return', 0),
                'Sortino Ratio': data.get('sortino_ratio', 0),
                'Max Drawdown': data.get('max_drawdown', 0),
                'Positive %': data.get('positive_returns_ratio', 50),
                'CVaR(95%)': data.get('cvar_95', 0)
            }
        
        deviated_metrics = self._calculate_deviation_scores(raw_metrics, categories, symbols)
        
        fig = go.Figure()
        colors = px.colors.qualitative.Set1
        
        for i, symbol in enumerate(symbols):
            values = [deviated_metrics[symbol][cat] for cat in categories]
            values.append(values[0])
            
            color = colors[i % len(colors)]
            avg_score = np.mean([deviated_metrics[symbol][cat] for cat in categories])
            
            fig.add_trace(go.Scatterpolar(
                r=values, theta=categories + [categories[0]], fill='toself',
                name=f'{symbol} (avg: {avg_score:.1f})', line_color=color,
                fillcolor=color.replace('rgb', 'rgba').replace(')', ', 0.25)'),
                hovertemplate=(f'<b>{symbol}</b><br>%{{theta}}: %{{r:.1f}}<br>'
                             f'Overall Score: {avg_score:.1f}<extra></extra>')
            ))
        
        self._configure_radar_chart(fig, len(symbols))
        self.display_figure(fig)
    
    def _calculate_deviation_scores(self, raw_metrics, categories, symbols):
        deviated_metrics = {}
        
        for category in categories:
            values = [raw_metrics[symbol][category] for symbol in symbols]
            values_array = np.array(values)
            
            valid_values = values_array[~np.isnan(values_array)]
            if len(valid_values) == 0:
                for symbol in symbols:
                    if symbol not in deviated_metrics:
                        deviated_metrics[symbol] = {}
                    deviated_metrics[symbol][category] = 50.0
                continue
            
            mean_val = np.mean(valid_values)
            std_val = np.std(valid_values, ddof=0)
            
            if std_val == 0:
                deviation_scores = [50.0] * len(symbols)
            else:
                deviation_scores = []
                for i, symbol in enumerate(symbols):
                    value = values_array[i]
                    
                    if pd.isna(value):
                        deviation_scores.append(25.0)
                        continue
                    
                    if category in ['Max Drawdown', 'CVaR(95%)']:
                        adjusted_value = -value
                        adjusted_mean = -mean_val
                    else:
                        adjusted_value = value
                        adjusted_mean = mean_val
                    
                    z_score = (adjusted_value - adjusted_mean) / std_val
                    deviation_score = 50 + (z_score * 10)
                    deviation_score = max(10, min(90, deviation_score))
                    deviation_scores.append(deviation_score)
            
            for i, symbol in enumerate(symbols):
                if symbol not in deviated_metrics:
                    deviated_metrics[symbol] = {}
                deviated_metrics[symbol][category] = deviation_scores[i]
        
        return deviated_metrics
    
    def _configure_radar_chart(self, fig, num_assets):
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True, range=[0, 100], gridcolor='rgba(255,255,255,0.3)',
                    color='white', tickmode='linear', tick0=0, dtick=20
                ),
                angularaxis=dict(gridcolor='rgba(255,255,255,0.3)', color='white', linecolor='white'),
                bgcolor='rgba(50,50,50,1)'
            ),
            title={'text': f'Risk Metrics Radar Chart ({num_assets} Assets)', 'x': 0.5, 'font': {'size': 18, 'color': 'white'}},
            paper_bgcolor='rgba(43,43,43,1)', font=dict(color='white'),
            legend=dict(bgcolor='rgba(0,0,0,0.7)', bordercolor='white', borderwidth=1)
        )
    
    def create_time_series_chart(self, metric_name: str):
        monthly_data = self.get_monthly_metrics_data()
        if not monthly_data:
            self.show_no_data_message("月次メトリクスデータが利用できません．")
            return
        
        metric_map = {
            "Expected Return": ('expected_return', 'Expected Return', '%'),
            "Standard Deviation": ('volatility', 'Volatility', '%'),
            "Downside Deviation": ('downside_deviation', 'Downside Deviation', '%'),
            "Positive Return Ratio": ('positive_ratio', 'Positive Return Ratio', '%'),
            "Maximum Drawdown": ('max_drawdown', 'Maximum Drawdown', '%')
        }
        
        if metric_name not in metric_map:
            self.show_no_data_message(f"未対応のメトリクス: {metric_name}")
            return
        
        metric_key, display_name, unit = metric_map[metric_name]
        
        fig = go.Figure()
        colors = px.colors.qualitative.Set1
        valid_assets = 0
        
        for i, (symbol, data) in enumerate(monthly_data.items()):
            if metric_key not in data.columns or data.empty:
                continue
            
            values = data[metric_key].dropna()
            if len(values) < 2:
                continue
            
            display_values = values.values
            if unit == '%':
                display_values = display_values * 100
            
            mean_val = np.mean(display_values)
            std_val = np.std(display_values)
            
            fig.add_trace(go.Scatter(
                x=values.index, y=display_values, mode='lines+markers', name=symbol,
                line=dict(color=colors[i % len(colors)], width=2.5), marker=dict(size=5),
                hovertemplate=(f'<b>{symbol}</b><br>Date: %{{x}}<br>{display_name}: %{{y:.2f}}{unit}<br>'
                             f'Mean: {mean_val:.2f}{unit}<br>Std: {std_val:.2f}{unit}<extra></extra>')
            ))
            valid_assets += 1
        
        if valid_assets == 0:
            self.show_no_data_message(f"有効な{display_name}データがありません．")
            return
        
        self._configure_time_series_chart(fig, display_name, unit, valid_assets)
        self.display_figure(fig)
    
    def _configure_time_series_chart(self, fig, display_name, unit, valid_assets):
        fig.update_layout(
            title={'text': f'{display_name} Time Series ({valid_assets} Assets)', 'x': 0.5, 'font': {'size': 18, 'color': 'white'}},
            xaxis=dict(
                title='Date', color='white', gridcolor='rgba(255,255,255,0.3)',
                rangeslider=dict(visible=True, bgcolor='rgba(70,70,70,0.8)', bordercolor='rgba(255,255,255,0.5)', borderwidth=1),
                rangeselector=dict(
                    buttons=[
                        dict(count=3, label="3M", step="month", stepmode="backward"),
                        dict(count=6, label="6M", step="month", stepmode="backward"),
                        dict(count=1, label="1Y", step="year", stepmode="backward"),
                        dict(count=2, label="2Y", step="year", stepmode="backward"),
                        dict(step="all")
                    ],
                    bgcolor='rgba(70,70,70,0.8)', activecolor='rgba(0,120,212,0.8)', font=dict(color='white')
                ),
                type='date'
            ),
            yaxis=dict(title=f'{display_name} {unit}', color='white', gridcolor='rgba(255,255,255,0.3)'),
            paper_bgcolor='rgba(43,43,43,1)', plot_bgcolor='rgba(50,50,50,1)',
            font=dict(color='white'), hovermode='x unified',
            legend=dict(bgcolor='rgba(0,0,0,0.7)', bordercolor='white', borderwidth=1)
        )
    
    def display_figure(self, fig):
        try:
            self.current_figure = fig
            
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8')
            html_content = fig.to_html(
                include_plotlyjs='cdn',
                config={
                    'displayModeBar': True,
                    'modeBarButtonsToAdd': ['pan2d', 'select2d', 'lasso2d', 'resetScale2d'],
                    'modeBarButtonsToRemove': [],
                    'displaylogo': False,
                    'toImageButtonOptions': {
                        'format': 'png', 'filename': 'risk_chart',
                        'height': 600, 'width': 800, 'scale': 2
                    }
                }
            )
            temp_file.write(html_content)
            temp_file.close()
            
            self.temp_html_files.append(temp_file.name)
            
            chart_type = self.chart_type_combo.currentText()
            selected_count = len(self.asset_selection.get_selected_symbols())
            self.status_label.setText(f"チャート作成完了: {chart_type} ({selected_count}件の資産)")
            
            self.browser_button.setEnabled(True)
            self.export_button.setEnabled(True)
            
        except Exception as e:
            print(f"チャート表示エラー: {e}")
            self.show_no_data_message()
    
    def show_no_data_message(self, message="データが不足しています．資産を選択してください．"):
        self.status_label.setText(message)
        self.browser_button.setEnabled(False)
        self.export_button.setEnabled(False)
    
    def open_in_browser(self):
        if not self.current_figure or not self.temp_html_files:
            QMessageBox.information(self, "Information", "表示するチャートがありません")
            return
        
        try:
            latest_file = self.temp_html_files[-1]
            
            if os.path.exists(latest_file):
                self.browser_thread = BrowserLaunchThread(latest_file)
                self.browser_thread.launch_completed.connect(self.on_browser_launch_completed)
                self.browser_thread.start()
                
                self.browser_button.setEnabled(False)
                self.browser_button.setText("開いています...")
            else:
                QMessageBox.warning(self, "エラー", "HTMLファイルが見つかりません")
                
        except Exception as e:
            QMessageBox.critical(self, "エラー", f"ブラウザを開く際にエラーが発生しました:\n{str(e)}")
    
    def on_browser_launch_completed(self, success: bool, message: str):
        self.browser_button.setEnabled(True)
        self.browser_button.setText("ブラウザで表示")
        
        if success:
            original_text = self.browser_button.text()
            self.browser_button.setText("✓ 表示済み")
            QTimer.singleShot(2000, lambda: self.browser_button.setText(original_text))
        else:
            QMessageBox.warning(self, "ブラウザ起動エラー", message)
    
    def clear_plot(self):
        if PLOTLY_AVAILABLE:
            self.show_no_data_message()
    
    def export_chart(self):
        if not PLOTLY_AVAILABLE or not self.current_figure:
            QMessageBox.information(self, "Information", "保存するチャートがありません")
            return
        
        chart_type = self.chart_type_combo.currentText()
        default_filename = f"interactive_chart_{chart_type.replace(' ', '_').replace('-', '_')}.html"
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "チャートを保存", default_filename, "HTMLファイル (*.html)"
        )
        
        if file_path:
            try:
                html_content = self.current_figure.to_html(
                    include_plotlyjs=True,
                    config={
                        'displayModeBar': True,
                        'modeBarButtonsToAdd': ['pan2d', 'select2d', 'lasso2d', 'resetScale2d'],
                        'displaylogo': False,
                        'toImageButtonOptions': {
                            'format': 'png', 'filename': 'risk_chart',
                            'height': 800, 'width': 1200, 'scale': 2
                        }
                    }
                )
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                
                QMessageBox.information(self, "保存完了", f"チャートを保存しました:\n{file_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "エラー", f"保存中にエラーが発生しました:\n{str(e)}")
    
    def cleanup_temp_files(self):
        for temp_file in self.temp_html_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception:
                pass
        self.temp_html_files.clear()
    
    def closeEvent(self, event):
        self.cleanup_temp_files()
        super().closeEvent(event)
    
    def __del__(self):
        self.cleanup_temp_files()