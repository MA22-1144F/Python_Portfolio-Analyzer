"""correlation_matrix.py"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple
import logging
from PySide6.QtWidgets import (
    QTableWidgetItem, QHeaderView, QMessageBox,
    QApplication, QFileDialog, QTextEdit, QWidget, QVBoxLayout, QLabel
)
from PySide6.QtCore import QThread, Signal, Qt, QTimer
from PySide6.QtGui import QColor
from data.asset_info import AssetInfo
from analysis.analysis_base_widget import AnalysisBaseWidget
from config.app_config import get_config

import tempfile
import os
import webbrowser


class BrowserLaunchThread(QThread):
    launch_completed = Signal(bool, str)
    
    def __init__(self, file_path: str):
        super().__init__()
        self.file_path = file_path
    
    def run(self):
        try:
            webbrowser.open(f"file:///{self.file_path}")
            self.launch_completed.emit(True, "ブラウザで表示しました．")
        except Exception as e:
            self.launch_completed.emit(False, f"ブラウザ起動エラー: {str(e)}")


class CorrelationCalculationThread(QThread):
    """相関行列計算スレッド"""
    
    calculation_completed = Signal(dict, dict)  # correlation_matrix, quality_report
    calculation_error = Signal(str)
    progress_updated = Signal(int, str)
    
    def __init__(self, price_df: pd.DataFrame = None, log_returns_data: Dict[str, pd.Series] = None):
        super().__init__()
        self.price_df = price_df
        self.log_returns_data = log_returns_data
        self.logger = logging.getLogger(__name__)
        
        # データ品質基準を設定から取得
        config = get_config()
        corr_config = config.get('analysis.correlation_matrix', {})
        self.min_data_points = corr_config.get('min_data_points', 10)
        self.min_coverage_ratio = corr_config.get('min_coverage_ratio', 0.3)
    
    def run(self):
        """相関行列計算実行"""
        try:
            # ログリターンデータを取得または計算
            if self.log_returns_data:
                log_returns_data = self.log_returns_data
                self.progress_updated.emit(10, "ログリターンデータを使用中...")
            elif self.price_df is not None and not self.price_df.empty:
                self.progress_updated.emit(10, "価格データからログリターンを計算中...")
                log_returns_data = self._calculate_log_returns_from_prices()
            else:
                self.calculation_error.emit("価格データまたはログリターンデータが必要です．")
                return
            
            if not log_returns_data:
                self.calculation_error.emit("ログリターンデータの計算に失敗しました．")
                return
            
            self.progress_updated.emit(30, "データ品質チェック中...")
            self.msleep(50)
            
            # データ品質チェックと除外処理
            filtered_data, excluded_assets = self._filter_data_by_quality(log_returns_data)
            
            if not filtered_data:
                self.calculation_error.emit("品質基準を満たすデータがありません．")
                return
            
            self.progress_updated.emit(50, "共通日付データ統合中...")
            self.msleep(50)
            
            # 共通日付で統合
            common_df = self._create_common_dataframe(filtered_data)
            
            if common_df.empty or len(common_df) < 2:
                self.calculation_error.emit("共通日付のデータが不足しています（最低2日分必要）．")
                return
            
            self.progress_updated.emit(70, "相関係数計算中...")
            self.msleep(50)
            
            # 相関行列を計算
            correlation_matrix = common_df.corr()
            
            self.progress_updated.emit(90, "品質レポート作成中...")
            self.msleep(50)
            
            # 品質レポート作成
            quality_report = {
                'total_assets': len(log_returns_data),
                'analyzed_assets': len(correlation_matrix.columns),
                'excluded_assets': excluded_assets,
                'common_dates': len(common_df),
                'date_range': (common_df.index.min(), common_df.index.max()) if len(common_df) > 0 else None,
                'data_coverage': len(common_df) / max(len(series) for series in filtered_data.values()) if filtered_data else 0
            }
            
            self.progress_updated.emit(100, "計算完了")
            self.calculation_completed.emit(correlation_matrix.to_dict(), quality_report)
            
        except Exception as e:
            self.logger.error(f"相関行列計算エラー: {e}")
            self.calculation_error.emit(f"計算中にエラーが発生しました．: {str(e)}")
    
    def _calculate_log_returns_from_prices(self) -> Dict[str, pd.Series]:
        """価格データからログリターンを計算"""
        log_returns_data = {}
        
        try:
            # 価格データのカラム（Dateやメタデータ列を除外）
            price_columns = [col for col in self.price_df.columns if col != 'Date' and not col.startswith('_')]
            
            for symbol in price_columns:
                try:
                    # 価格データを取得（NaNを除去）
                    prices = self.price_df[symbol].dropna()
                    
                    if len(prices) < 2:
                        self.logger.warning(f"{symbol}: 価格データが不足（{len(prices)}件）")
                        continue
                    
                    # ログリターンを計算
                    log_returns = np.log(prices / prices.shift(1)).dropna()
                    
                    if len(log_returns) > 0:
                        log_returns.name = symbol
                        log_returns_data[symbol] = log_returns
                        self.logger.info(f"{symbol}: ログリターン {len(log_returns)}件を計算")
                    else:
                        self.logger.warning(f"{symbol}: ログリターンの計算に失敗")
                        
                except Exception as e:
                    self.logger.error(f"{symbol} のログリターン計算エラー: {e}")
                    continue
            
            return log_returns_data
            
        except Exception as e:
            self.logger.error(f"ログリターン計算エラー: {e}")
            return {}
    
    def _filter_data_by_quality(self, log_returns_data: Dict[str, pd.Series]) -> Tuple[Dict[str, pd.Series], List[Dict]]:
        """データ品質チェックとフィルタリング"""
        filtered_data = {}
        excluded_assets = []
        
        # 全データの最大長を取得
        max_length = max(len(series) for series in log_returns_data.values()) if log_returns_data else 0
        
        for symbol, series in log_returns_data.items():
            try:
                # データ数チェック
                if len(series) < self.min_data_points:
                    excluded_assets.append({
                        'symbol': symbol,
                        'reason': 'insufficient_data',
                        'data_points': len(series),
                        'threshold': self.min_data_points
                    })
                    continue
                
                # カバレッジ率チェック
                coverage_ratio = len(series) / max_length if max_length > 0 else 0
                if coverage_ratio < self.min_coverage_ratio:
                    excluded_assets.append({
                        'symbol': symbol,
                        'reason': 'low_coverage',
                        'coverage_ratio': coverage_ratio,
                        'threshold': self.min_coverage_ratio
                    })
                    continue
                
                # 統計的チェック（すべてNaNや同じ値でないか）
                if series.isna().all() or series.nunique() <= 1:
                    excluded_assets.append({
                        'symbol': symbol,
                        'reason': 'invalid_data',
                        'description': 'All NaN or constant values'
                    })
                    continue
                
                # 品質基準を満たす
                filtered_data[symbol] = series
                
            except Exception as e:
                self.logger.error(f"{symbol} の品質チェックエラー: {e}")
                excluded_assets.append({
                    'symbol': symbol,
                    'reason': 'quality_check_error',
                    'error': str(e)
                })
        
        return filtered_data, excluded_assets
    
    def _create_common_dataframe(self, log_returns_data: Dict[str, pd.Series]) -> pd.DataFrame:
        """共通日付でDataFrameを作成"""
        try:
            if not log_returns_data:
                return pd.DataFrame()
            
            # 最初のシリーズから開始
            first_symbol = list(log_returns_data.keys())[0]
            common_df = pd.DataFrame({first_symbol: log_returns_data[first_symbol]})
            
            # 他のシリーズを順次追加し，共通日付のみを保持
            for symbol, series in list(log_returns_data.items())[1:]:
                if len(common_df) > 0:
                    # 共通する日付のみを保持
                    common_dates = common_df.index.intersection(series.index)
                    if len(common_dates) > 0:
                        common_df = common_df.reindex(common_dates)
                        common_df[symbol] = series.reindex(common_dates)
                    else:
                        self.logger.warning(f"{symbol}: 共通日付がありません．")
                        break
            
            # NaNの除去
            common_df = common_df.dropna()
            
            return common_df
            
        except Exception as e:
            self.logger.error(f"共通DataFrameの作成エラー: {e}")
            return pd.DataFrame()


class CorrelationMatrixWidget(AnalysisBaseWidget):
    """相関行列分析ウィジェット"""
    
    def __init__(self, config=None):
        super().__init__(config)
        self.price_data_source = None
        self.correlation_matrix = None
        self.quality_report = {}
        self.log_returns_data = {}
        self.current_html_file = None
        self.temp_html_files = []
        self.browser_thread = None
        self.current_conditions = None
        self.logger = logging.getLogger(__name__)
        # 設定からサイズを取得
        min_height = self.config.get('analysis.widget_sizes.correlation_matrix.min_height', 600)
        min_width = self.config.get('analysis.widget_sizes.correlation_matrix.min_width', 600)
        self.setMinimumHeight(min_height)
        self.setMinimumWidth(min_width)
        
        self.setup_content()
    
    def analyze(self, assets: List[AssetInfo], conditions: Dict[str, Any]):
        """分析実行（AnalysisBaseWidgetの抽象メソッドを実装）"""
        if not self.price_data_source:
            QMessageBox.warning(self, "エラー", "価格データソースが設定されていません．")
            return
        
        if not self.price_data_source.is_ready():
            self.show_progress(True)
            self.update_progress(0, "価格データの取得完了を待機中...")
            self.price_data_source.data_ready.connect(self.on_price_data_ready)
            
            # 分析条件を保存
            self.current_conditions = conditions
            return
        
        # 分析条件を保存
        self.current_conditions = conditions
        self.start_calculation()
    
    def on_price_data_ready(self):
        """価格データ準備完了時の処理"""
        try:
            self.price_data_source.data_ready.disconnect(self.on_price_data_ready)
        except (RuntimeError, TypeError):
            # 接続されていない場合は無視
            pass
        
        if hasattr(self, 'current_conditions') and self.current_conditions:
            self.start_calculation()
    
    def setup_header_content(self):
        """ヘッダーコンテンツの設定（ベースクラスメソッドをオーバーライド）"""
        self.header_layout.addStretch()
        
        # ブラウザ表示ボタン
        self.browser_button = self.create_button(
            "ブラウザで表示", 
            "primary"
        )
        self.browser_button.setEnabled(False)
        self.browser_button.clicked.connect(self.open_in_browser)
        self.header_layout.addWidget(self.browser_button)
        
        # HTML保存ボタン
        self.export_html_button = self.create_button(
            "HTML保存",
            "save"
        )
        self.export_html_button.setEnabled(False)
        self.export_html_button.clicked.connect(self.export_html)
        self.header_layout.addWidget(self.export_html_button)
        
        # CSV出力ボタン
        self.export_button = self.create_button(
            "CSV出力", 
            "export"
        )
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export_to_csv)
        self.header_layout.addWidget(self.export_button)

    def setup_content(self):
        """メインコンテンツエリアの設定"""
        # 制約条件情報ラベル
        self.constraint_info_label = QLabel("")
        self.constraint_info_label.setStyleSheet("color: #0078d4; font-size: 9px;")
        self.constraint_info_label.setVisible(False)
        self.constraint_info_label.setWordWrap(True)
        self.constraint_info_label.setMaximumHeight(30)
        self.content_layout.addWidget(self.constraint_info_label)
        
        # 統一スタイルのタブウィジェットを作成
        self.tab_widget = self.create_tab_widget()
        
        # タブ1: 相関行列テーブル
        table_tab = QWidget()
        table_layout = QVBoxLayout(table_tab)
        table_layout.setContentsMargins(5, 5, 5, 5)
        
        # テーブルを作成
        self.correlation_table = self.create_table_widget()
        
        # テーブル固有のスタイルを設定
        self.correlation_table.setAlternatingRowColors(False)  # 交互行色を無効化
        self.correlation_table.setStyleSheet("""
            QTableWidget {
                gridline-color: #d0d0d0;
                border: 1px solid #d0d0d0;
            }
            QTableWidget::item {
                border: 1px solid #d0d0d0;
                border-radius: 0px;
                padding: 0px;
                margin: 0px;
            }
            QTableWidget::item:selected {
                background-color: rgba(0, 120, 215, 0.3);
                border: 2px solid #0078d4;
                border-radius: 0px;
            }
            QHeaderView::section {
                background-color: #f0f0f0;
                border: 1x solid #d0d0d0;
                border-radius: 0px;
                padding: 0px;
                margin: 0px;
                font-weight: bold;
            }
        """)

        # セルサイズの固定設定
        self.correlation_table.horizontalHeader().setDefaultSectionSize(70)
        self.correlation_table.verticalHeader().setDefaultSectionSize(30)
        self.correlation_table.horizontalHeader().setFixedHeight(30)
        self.correlation_table.verticalHeader().setFixedWidth(70)
        
        # リサイズモードを固定に設定
        self.correlation_table.horizontalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self.correlation_table.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
        
        table_layout.addWidget(self.correlation_table)
        self.tab_widget.addTab(table_tab, "相関行列")
        
        # タブ2: 分析結果サマリー
        summary_tab = QWidget()
        summary_layout = QVBoxLayout(summary_tab)
        summary_layout.setContentsMargins(5, 5, 5, 5)
        
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setStyleSheet(f"""
            QTextEdit {{
                background-color: {self.styles.COLORS["background"]};
                color: {self.styles.COLORS["text_primary"]};
                border: 1px solid {self.styles.COLORS["border"]};
                border-radius: 4px;
                padding: 5px;
                font-family: 'Courier New', monospace;
                font-size: 10px;
            }}
        """)
        summary_layout.addWidget(self.summary_text)
        
        self.tab_widget.addTab(summary_tab, "分析サマリー")
        
        self.content_layout.addWidget(self.tab_widget)
        
    def get_empty_message(self) -> str:
        """空状態メッセージ（オーバーライド）"""
        return (
            "相関行列が表示されていません．\n"
            "価格時系列データの取得完了後，または\n"
            "リターン・リスク指標の分析完了後に表示されます．"
        )

    def start_calculation(self):
        """計算開始"""
        # まずログリターンデータの取得を試行
        log_returns_data = self.get_log_returns_data()
        price_df = None
        
        # ログリターンデータが取得できない場合は価格データから計算
        if not log_returns_data:
            if not self.price_data_source or not self.price_data_source.is_ready():
                QMessageBox.warning(self, "エラー", 
                                  "データが不足しています．\n"
                                  "リターン・リスク指標の分析を先に実行するか，\n"
                                  "価格時系列データを取得してください．")
                return
            
            price_df = self.price_data_source.get_analysis_dataframe()
            if price_df is None or price_df.empty:
                QMessageBox.warning(self, "エラー", "価格データの取得に失敗しました．")
                return
        
        # UI状態更新
        self.show_progress(True)
        self.show_quality_info("計算中...")
        self.show_main_content(False)  # テーブルを非表示
        self.browser_button.setEnabled(False)
        self.export_html_button.setEnabled(False)
        self.export_button.setEnabled(False)
        
        # 分析資産数の表示
        if log_returns_data:
            n_assets = len(log_returns_data)
        elif price_df is not None:
            n_assets = len(price_df.columns)
        else:
            n_assets = 0
        
        self.constraint_info_label.setText(f"分析: {n_assets} 資産の相関行列を計算中...")
        self.constraint_info_label.setVisible(True)
        
        QApplication.processEvents()
        
        # 計算スレッド開始
        self.calculation_thread = CorrelationCalculationThread(price_df, log_returns_data)
        self.calculation_thread.progress_updated.connect(self.update_progress)
        self.calculation_thread.calculation_completed.connect(self.on_calculation_completed)
        self.calculation_thread.calculation_error.connect(self.on_calculation_error)
        self.calculation_thread.start()
    
    def get_log_returns_data(self) -> Dict[str, pd.Series]:
        """ログリターンデータを取得（まずリターン・リスク指標から，なければ価格データから計算）"""
        try:
            # 親ウィジェットを辿ってResultTileを見つける
            parent = self.parent()
            search_depth = 0
            max_depth = 10
            
            while parent and search_depth < max_depth:
                # ResultTileを見つける
                if hasattr(parent, 'analysis_items'):
                    # リターン・リスク指標のウィジェットを探す
                    for item_widget in parent.analysis_items:
                        if (item_widget.item_type == "return_risk_analysis" and 
                            hasattr(item_widget.analysis_widget, 'calculator') and 
                            item_widget.analysis_widget.calculator):
                            log_returns_data = item_widget.analysis_widget.calculator.get_log_returns_data()
                            if log_returns_data:
                                self.logger.info("リターン・リスク指標からログリターンデータを取得")
                                return log_returns_data
                
                parent = parent.parent()
                search_depth += 1
            
            # リターン・リスク指標がない場合は空の辞書を返す（価格データから計算する）
            self.logger.info("リターン・リスク指標が見つかりません．価格データから計算します．")
            return {}
            
        except Exception as e:
            logging.error(f"ログリターンデータ取得エラー: {e}")
            return {}
    
    def on_calculation_completed(self, correlation_dict: Dict, quality_report: Dict):
        """計算完了"""
        # 辞書をDataFrameに変換
        self.correlation_matrix = pd.DataFrame(correlation_dict)
        self.quality_report = quality_report
        
        # UI更新
        self.update_display()
        self.create_html_table()
        self.update_summary()
        
        # UI状態更新
        self.show_progress(False)
        self.hide_quality_info()  # 「計算中…」メッセージを非表示
        
        # データ品質情報を表示
        if quality_report:
            total = quality_report.get('total_assets', 0)
            analyzed = quality_report.get('analyzed_assets', 0)
            excluded_assets = quality_report.get('excluded_assets', [])
            excluded = len(excluded_assets)
            common_dates = quality_report.get('common_dates', 0)
            
            # テキスト形式
            quality_text = f"分析: {analyzed}/{total} 資産 | 除外: {excluded} 資産 | 共通日付: {common_dates} 日"
            
            if quality_report.get('date_range'):
                start_date, end_date = quality_report['date_range']
                quality_text += f" | 期間: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}"
            
            self.constraint_info_label.setText(quality_text)
        
        if analyzed > 0:
            self.browser_button.setEnabled(True)
            self.export_html_button.setEnabled(True)
            self.export_button.setEnabled(True)
    
    def on_calculation_error(self, error_message: str):
        """計算エラー"""
        self.show_progress(False)
        self.hide_quality_info()
        self.constraint_info_label.setVisible(False)
        QMessageBox.critical(self, "エラー", error_message)
        self.update_display()
    
    def get_correlation_color(self, correlation: float) -> QColor:
        """相関係数に基づく色を取得（グラデーション：赤-白-青）"""
        if pd.isna(correlation):
            return QColor(240, 240, 240)  # 薄いグレー
        
        # 相関係数を-1から1の範囲に制限
        correlation = max(-1.0, min(1.0, correlation))

        if correlation >= 0:
            # 正の相関: 白(255,255,255)から赤(220,20,20)へのグラデーション
            # correlationが0なら白，1なら赤
            r = 255 - int(35 * correlation)  # 255 -> 220
            g = 255 - int(235 * correlation)  # 255 -> 20
            b = 255 - int(235 * correlation)  # 255 -> 20
        else:
            # 負の相関: 白(255,255,255)から青(20,20,220)へのグラデーション
            abs_correlation = abs(correlation)
            r = 255 - int(235 * abs_correlation)  # 255 -> 20
            g = 255 - int(235 * abs_correlation)  # 255 -> 20  
            b = 255 - int(35 * abs_correlation)   # 255 -> 220
        
        return QColor(r, g, b)

    def get_text_color(self, correlation: float, background_color: QColor) -> QColor:
        """背景色に応じたテキスト色を決定"""
        if pd.isna(correlation):
            return QColor(100, 100, 100)
        
        # 背景色の明度を計算
        brightness = (background_color.red() * 0.299 + 
                    background_color.green() * 0.587 + 
                    background_color.blue() * 0.114)
        
        # 明度に応じてテキスト色を決定
        if brightness > 128:
            return QColor(0, 0, 0)      # 黒
        else:
            return QColor(255, 255, 255)  # 白
        
    def update_display(self):
        """表示更新"""
        if self.correlation_matrix is None or self.correlation_matrix.empty:
            self.show_main_content(False)  # テーブルを非表示，空状態を表示
            return
        
        self.show_main_content(True)  # 空状態を非表示，テーブルを表示
        
        symbols = list(self.correlation_matrix.columns)
        
        # テーブル設定
        self.correlation_table.setRowCount(len(symbols))
        self.correlation_table.setColumnCount(len(symbols))
        
        # ヘッダー設定
        self.correlation_table.setVerticalHeaderLabels(symbols)
        self.correlation_table.setHorizontalHeaderLabels(symbols)
        
        # テーブルのスタイルをリセット
        self.correlation_table.setStyleSheet("")
        
        # データ挿入と色分け
        for i, symbol_i in enumerate(symbols):
            for j, symbol_j in enumerate(symbols):
                correlation_value = self.correlation_matrix.loc[symbol_i, symbol_j]
                
                if pd.isna(correlation_value):
                    item = QTableWidgetItem("-")
                    item.setBackground(QColor(240, 240, 240))
                    item.setForeground(QColor(100, 100, 100))
                else:
                    item = QTableWidgetItem(f"{correlation_value:.3f}")
                    
                    # グラデーションによる相関係数の色分け（赤-白-青）
                    bg_color = self.get_correlation_color(correlation_value)
                    item.setBackground(bg_color)
                    
                    # テキスト色を設定
                    text_color = self.get_text_color(correlation_value, bg_color)
                    item.setForeground(text_color)
                    
                    # 対角線の要素（自己相関）は太字にする
                    if i == j:
                        font = item.font()
                        font.setBold(False)  # 太字を無効化
                        font.setPointSize(8)  # フォントサイズを小さく
                        item.setFont(font)
                        # 対角線は特別な背景色
                        item.setBackground(QColor(0, 0, 0))      # 黒
                        item.setForeground(QColor(255, 255, 255))  # 白

                item.setTextAlignment(Qt.AlignCenter)
                item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
                
                # ツールチップ設定
                if not pd.isna(correlation_value):
                    interpretation = self.interpret_correlation(correlation_value)
                    tooltip = (f"{symbol_i} vs {symbol_j}\n"
                            f"相関係数: {correlation_value:.4f}\n"
                            f"解釈: {interpretation}")
                    item.setToolTip(tooltip)
                
                self.correlation_table.setItem(i, j, item)
        
        # カラムとロウの幅を調整
        uniform_width = 70
        uniform_height = 30
        
        # 全てのカラムを統一幅に設定
        for col in range(self.correlation_table.columnCount()):
            self.correlation_table.setColumnWidth(col, uniform_width)
        # 全ての行を統一高さに設定
        for row in range(self.correlation_table.rowCount()):
            self.correlation_table.setRowHeight(row, uniform_height)
        
        # ヘッダーの高さも調整
        self.correlation_table.horizontalHeader().setDefaultSectionSize(uniform_width)
        self.correlation_table.verticalHeader().setDefaultSectionSize(uniform_height)
    
    def interpret_correlation(self, correlation: float) -> str:
        """相関係数の解釈を返す"""
        abs_corr = abs(correlation)
        
        if abs_corr >= 0.9:
            strength = "非常に強い"
        elif abs_corr >= 0.7:
            strength = "強い"
        elif abs_corr >= 0.5:
            strength = "中程度の"
        elif abs_corr >= 0.3:
            strength = "弱い"
        else:
            strength = "ほとんど無い"
        
        direction = "正の" if correlation >= 0 else "負の"
        
        if abs_corr < 0.1:
            return "相関はほとんど無い"
        else:
            return f"{strength}{direction}相関"
    
    def create_html_table(self):
        """HTMLテーブルを作成"""
        if self.correlation_matrix is None or self.correlation_matrix.empty:
            return
        
        try:
            symbols = list(self.correlation_matrix.columns)
            
            # HTMLコンテンツを生成
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>相関行列</title>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        background-color: #2b2b2b;
                        color: #ffffff;
                        margin: 20px;
                    }}
                    table {{
                        border-collapse: collapse;
                        margin: 0 auto;
                        background-color: #ffffff;
                        border-radius: 0px;
                        overflow: hidden;
                    }}
                    th, td {{
                        width: 70px;
                        height: 30px;
                        text-align: center;
                        border: 1px solid #d0d0d0;
                        font-size: 10px;
                        padding: 0;
                        margin: 0;
                    }}
                    th {{
                        background-color: #f0f0f0;
                        font-weight: bold;
                        color: #000000;
                    }}
                    .diagonal {{
                        background-color: #000000;
                        color: #ffffff;
                        font-weight: bold;
                    }}
                </style>
            </head>
            <body>
                <table>
                    <thead>
                        <tr>
                            <th>資産</th>
            """
            
            for symbol in symbols:
                html_content += f"<th>{symbol}</th>"
            
            html_content += """
                        </tr>
                    </thead>
                    <tbody>
            """
            
            for i, symbol_i in enumerate(symbols):
                html_content += f"<tr><th>{symbol_i}</th>"
                
                for j, symbol_j in enumerate(symbols):
                    correlation_value = self.correlation_matrix.loc[symbol_i, symbol_j]
                    
                    if pd.isna(correlation_value):
                        html_content += '<td style="background-color: #f0f0f0; color: #666;">-</td>'
                    else:
                        # 背景色を計算
                        bg_color = self.get_correlation_color(correlation_value)
                        text_color = self.get_text_color(correlation_value, bg_color)
                        
                        if i == j:  # 対角線
                            html_content += f'<td class="diagonal">{correlation_value:.3f}</td>'
                        else:
                            style = f"background-color: rgb({bg_color.red()}, {bg_color.green()}, {bg_color.blue()}); color: rgb({text_color.red()}, {text_color.green()}, {text_color.blue()});"
                            html_content += f'<td style="{style}" title="{symbol_i} vs {symbol_j}: {correlation_value:.4f} ({self.interpret_correlation(correlation_value)})">{correlation_value:.3f}</td>'
                
                html_content += "</tr>"
            
            html_content += """
                    </tbody>
                </table>
            </body>
            </html>
            """
            
            # 一時ファイルに保存
            temp_file = tempfile.NamedTemporaryFile(
                mode='w', suffix='.html', delete=False, encoding='utf-8'
            )
            temp_file.write(html_content)
            temp_file.close()
            
            self.current_html_file = temp_file.name
            self.temp_html_files.append(temp_file.name)
            
        except Exception as e:
            self.logger.error(f"HTMLテーブル作成エラー: {e}")
    
    def update_summary(self):
        """分析サマリーを更新"""
        if self.correlation_matrix is None or self.correlation_matrix.empty:
            self.summary_text.clear()
            return
        
        try:
            symbols = list(self.correlation_matrix.columns)
            
            # サマリーテキストの作成
            summary_lines = [
                "=== 相関行列分析結果 ===\n",
                f"分析日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"分析資産数: {len(symbols)} 資産",
                "",
                "=== データ品質情報 ===",
            ]
            
            if self.quality_report:
                total = self.quality_report.get('total_assets', 0)
                analyzed = self.quality_report.get('analyzed_assets', 0)
                excluded_assets = self.quality_report.get('excluded_assets', [])
                excluded = len(excluded_assets)
                common_dates = self.quality_report.get('common_dates', 0)
                
                summary_lines.extend([
                    f"総資産数: {total}",
                    f"分析対象資産数: {analyzed}",
                    f"除外資産数: {excluded}",
                    f"共通日付数: {common_dates}日",
                    f"データカバレッジ: {self.quality_report.get('data_coverage', 0):.1%}",
                ])
                
                if self.quality_report.get('date_range'):
                    start_date, end_date = self.quality_report['date_range']
                    summary_lines.append(f"分析期間: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")
                
                # 除外資産の詳細
                if excluded_assets:
                    summary_lines.extend([
                        "",
                        "=== 除外資産の詳細 ===",
                    ])
                    for asset_info in excluded_assets:
                        symbol = asset_info.get('symbol', 'Unknown')
                        reason = asset_info.get('reason', 'unknown')
                        if reason == 'insufficient_data':
                            summary_lines.append(f"{symbol}: データ不足 ({asset_info.get('data_points', 0)}件)")
                        elif reason == 'low_coverage':
                            summary_lines.append(f"{symbol}: カバレッジ不足 ({asset_info.get('coverage_ratio', 0):.1%})")
                        else:
                            summary_lines.append(f"{symbol}: {reason}")
            
            # 相関統計
            if len(symbols) > 1:
                # 対角線を除く相関係数を取得
                correlations = []
                for i in range(len(symbols)):
                    for j in range(i+1, len(symbols)):
                        corr = self.correlation_matrix.iloc[i, j]
                        if not pd.isna(corr):
                            correlations.append(corr)
                
                if correlations:
                    summary_lines.extend([
                        "",
                        "=== 相関統計 ===",
                        f"ペア数: {len(correlations)}",
                        f"最大相関: {np.max(correlations):.4f}",
                        f"最小相関: {np.min(correlations):.4f}",
                        f"平均相関: {np.mean(correlations):.4f}",
                        f"中央値: {np.median(correlations):.4f}",
                        f"標準偏差: {np.std(correlations):.4f}",
                    ])
                    
                    # 強い相関のペア
                    strong_positive = [(symbols[i], symbols[j], self.correlation_matrix.iloc[i, j])
                                     for i in range(len(symbols)) for j in range(i+1, len(symbols))
                                     if not pd.isna(self.correlation_matrix.iloc[i, j]) and self.correlation_matrix.iloc[i, j] >= 0.7]
                    
                    strong_negative = [(symbols[i], symbols[j], self.correlation_matrix.iloc[i, j])
                                     for i in range(len(symbols)) for j in range(i+1, len(symbols))
                                     if not pd.isna(self.correlation_matrix.iloc[i, j]) and self.correlation_matrix.iloc[i, j] <= -0.7]
                    
                    if strong_positive:
                        summary_lines.extend([
                            "",
                            "=== 強い正の相関 (≥0.7) ===",
                        ])
                        for asset1, asset2, corr in sorted(strong_positive, key=lambda x: x[2], reverse=True):
                            summary_lines.append(f"{asset1} - {asset2}: {corr:.4f}")
                    
                    if strong_negative:
                        summary_lines.extend([
                            "",
                            "=== 強い負の相関 (≤-0.7) ===",
                        ])
                        for asset1, asset2, corr in sorted(strong_negative, key=lambda x: x[2]):
                            summary_lines.append(f"{asset1} - {asset2}: {corr:.4f}")
            
            # 各資産の詳細
            summary_lines.extend([
                "",
                "=== 各資産の相関詳細 ===",
            ])
            
            for symbol in symbols:
                others = [s for s in symbols if s != symbol]
                correlations_with_others = [self.correlation_matrix.loc[symbol, other] 
                                          for other in others 
                                          if not pd.isna(self.correlation_matrix.loc[symbol, other])]
                
                if correlations_with_others:
                    avg_corr = np.mean(correlations_with_others)
                    max_corr = np.max(correlations_with_others)
                    min_corr = np.min(correlations_with_others)
                    
                    summary_lines.extend([
                        f"\n{symbol}:",
                        f"  他資産との平均相関: {avg_corr:.4f}",
                        f"  最大相関: {max_corr:.4f}",
                        f"  最小相関: {min_corr:.4f}",
                    ])
            
            self.summary_text.setPlainText("\n".join(summary_lines))
            
        except Exception as e:
            self.logger.error(f"サマリー更新エラー: {e}")
            self.summary_text.setPlainText(f"サマリー作成エラー: {str(e)}")
    
    def open_in_browser(self):
        """ブラウザで開く"""
        if not self.current_html_file:
            QMessageBox.information(self, "Information", "表示するHTMLファイルがありません．")
            return
        
        try:
            if os.path.exists(self.current_html_file):
                self.browser_thread = BrowserLaunchThread(self.current_html_file)
                self.browser_thread.launch_completed.connect(self.on_browser_launch_completed)
                self.browser_thread.start()
                
                self.browser_button.setEnabled(False)
                self.browser_button.setText("開いています...")
            else:
                QMessageBox.warning(self, "エラー", "HTMLファイルが見つかりません．")
                
        except Exception as e:
            QMessageBox.critical(self, "エラー", f"ブラウザを開く際にエラーが発生しました:\n{str(e)}")
    
    def on_browser_launch_completed(self, success: bool, message: str):
        """ブラウザ起動完了"""
        self.browser_button.setEnabled(True)
        self.browser_button.setText("ブラウザで表示")
        
        if success:
            original_text = self.browser_button.text()
            self.browser_button.setText("✓ 表示済み")
            QTimer.singleShot(2000, lambda: self.browser_button.setText(original_text))
        else:
            QMessageBox.warning(self, "ブラウザ起動エラー", message)
    
    def export_html(self):
        """HTMLファイルを保存"""
        if not self.current_html_file:
            QMessageBox.information(self, "Information", "保存するHTMLファイルがありません．")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "相関行列をHTML保存", "correlation_matrix.html", "HTML files (*.html)"
        )
        
        if file_path:
            try:
                import shutil
                shutil.copy2(self.current_html_file, file_path)
                QMessageBox.information(self, "保存完了", f"HTMLファイルを保存しました:\n{file_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "エラー", f"保存中にエラーが発生しました:\n{str(e)}")
    
    def export_to_csv(self):
        """CSV出力"""
        if self.correlation_matrix is None or self.correlation_matrix.empty:
            QMessageBox.information(self, "Information", "出力するデータがありません．")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "相関行列をCSV出力", "correlation_matrix.csv", "CSV ファイル (*.csv)"
        )
        
        if file_path:
            try:
                # CSV出力（行=資産，列=資産）
                with open(file_path, 'w', encoding='utf-8') as f:
                    # データ品質情報をヘッダーに追加
                    if self.quality_report:
                        f.write('# Correlation Matrix Report\n')
                        f.write(f'# Total Number of Assets: {self.quality_report.get("total_assets", 0)}\n')
                        f.write(f'# Number of Analyzed Assets: {self.quality_report.get("analyzed_assets", 0)}\n')
                        f.write(f'# Number of Excluded Assets: {len(self.quality_report.get("excluded_assets", []))}\n')
                        f.write(f'# Number of Common Dates: {self.quality_report.get("common_dates", 0)}\n')
                        f.write(f'# Data Coverage: {self.quality_report.get("data_coverage", 0):.1%}\n')
                        if self.quality_report.get('date_range'):
                            start_date, end_date = self.quality_report['date_range']
                            f.write(f'# Date Range: {start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")}\n')
                        
                        # 除外資産の詳細
                        excluded_assets = self.quality_report.get("excluded_assets", [])
                        if excluded_assets:
                            f.write('# Excluded Assets:\n')
                            for asset_info in excluded_assets:
                                symbol = asset_info.get('symbol', 'Unknown')
                                reason = asset_info.get('reason', 'unknown')
                                f.write(f'#   {symbol}: {reason}\n')
                        f.write('\n')
                    
                    # ヘッダー行（列ラベル）
                    symbols = list(self.correlation_matrix.columns)
                    f.write('Assets,' + ','.join(symbols) + '\n')
                    
                    # 各行のデータ
                    for symbol_i in symbols:
                        f.write(f'{symbol_i},')
                        values = []
                        for symbol_j in symbols:
                            correlation_value = self.correlation_matrix.loc[symbol_i, symbol_j]
                            if pd.isna(correlation_value):
                                values.append("")
                            else:
                                values.append(f"{correlation_value:.6f}")
                        f.write(','.join(values) + '\n')
                
                # 完了メッセージ
                quality_info = ""
                if self.quality_report:
                    analyzed = self.quality_report.get('analyzed_assets', 0)
                    excluded = len(self.quality_report.get('excluded_assets', []))
                    common_dates = self.quality_report.get('common_dates', 0)
                    quality_info = f"\n\n分析結果: {analyzed}資産の相関行列（除外: {excluded}資産，{common_dates}日間のデータ）"
                
                QMessageBox.information(self, "完了", f"CSVファイルを出力しました:\n{file_path}{quality_info}")
                
            except Exception as e:
                QMessageBox.critical(self, "エラー", f"CSV出力中にエラーが発生しました:\n{str(e)}")
    
    def clear_data(self):
        """データクリア"""
        self.correlation_matrix = None
        self.quality_report = {}
        self.log_returns_data = {}
        self.current_conditions = None
        self.cleanup_temp_files()
        
        self.update_display()
        self.summary_text.clear()
        
        self.browser_button.setEnabled(False)
        self.export_html_button.setEnabled(False)
        self.export_button.setEnabled(False)
        self.show_progress(False)
        self.hide_quality_info()
        self.constraint_info_label.setVisible(False)
    
    def cleanup_temp_files(self):
        """一時ファイルのクリーンアップ"""
        for temp_file in self.temp_html_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception:
                pass
        self.temp_html_files.clear()
        self.current_html_file = None
    
    def closeEvent(self, event):
        """ウィンドウクローズ時の処理"""
        self.cleanup_temp_files()
        super().closeEvent(event)
    
    def __del__(self):
        """デストラクタ"""
        self.cleanup_temp_files()
    
    def get_correlation_matrix(self) -> Optional[pd.DataFrame]:
        """相関行列を取得"""
        return self.correlation_matrix
    
    def get_quality_report(self) -> Dict[str, Any]:
        """品質レポートを取得"""
        return self.quality_report