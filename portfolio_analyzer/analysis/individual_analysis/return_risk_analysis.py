"""return_risk_analysis.py"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
import logging
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, 
    QTableWidgetItem, QMessageBox,
    QApplication, QFileDialog
)
from PySide6.QtCore import QThread, Signal, Qt
from data.asset_info import AssetInfo
from analysis.analysis_base_widget import AnalysisBaseWidget

try:
    from analysis.interactive_risk_visualization import InteractiveRiskVisualizationWidget
    INTERACTIVE_VIZ_AVAILABLE = True
except ImportError:
    InteractiveRiskVisualizationWidget = None
    INTERACTIVE_VIZ_AVAILABLE = False


class ReturnRiskCalculator:
    """統合リターン・リスク指標計算クラス"""
    
    def __init__(self, config=None):
        self.logger = logging.getLogger(__name__)
        # 設定を取得
        if config is None:
            from config.app_config import AppConfig
            config = AppConfig()
        self.config = config
        
        # データ品質基準を設定から取得
        self.min_data_points = self.config.get('analysis.min_data_points', {
            '日次': 30, '週次': 20, '月次': 12, '年次': 3
        })
        self.min_coverage_ratio = self.config.get('analysis.min_coverage_ratio', 0.7)
        
        # ログリターンデータを保存（可視化用）
        self.log_returns_data = {}
        self.monthly_metrics_data = {}
    
    def calculate_return_risk_metrics_from_dataframe(self, price_df: pd.DataFrame, span: str = '日次', risk_free_rate: float = 0.0) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
        """DataFrameから統合リターン・リスク指標を計算"""
        results = {}
        quality_report = {
            'total_assets': 0,
            'analyzed_assets': 0,
            'excluded_assets': [],
            'common_dates': 0,
            'date_range': None,
            'min_data_threshold': self.min_data_points.get(span, 30),
            'coverage_ratio': 0.0
        }
        
        # ログリターンデータと月次データをリセット
        self.log_returns_data = {}
        self.monthly_metrics_data = {}
        
        if price_df.empty:
            return results, quality_report
        
        # 価格データのカラムを取得（Date列やメタデータ列を除外）
        price_columns = [col for col in price_df.columns if col != 'Date' and not col.startswith('_')]
        quality_report['total_assets'] = len(price_columns)
        quality_report['common_dates'] = len(price_df)
        
        if len(price_df) > 0:
            quality_report['date_range'] = (price_df.index[0], price_df.index[-1])
        
        # 無リスク利子率をスパンに応じて変換
        span_risk_free_rate = self._convert_risk_free_rate_to_span(risk_free_rate, span)
        
        # データ品質チェック
        min_threshold = quality_report['min_data_threshold']
        
        for symbol in price_columns:
            try:
                # 価格データを取得（NaNを除去して実データのみ）
                prices = price_df[symbol].dropna()
                
                # データ数チェック
                if len(prices) < min_threshold:
                    self.logger.warning(f"{symbol}: データ不足（{len(prices)}件 < {min_threshold}件）")
                    quality_report['excluded_assets'].append({
                        'symbol': symbol,
                        'reason': 'insufficient_data',
                        'data_points': len(prices),
                        'threshold': min_threshold
                    })
                    results[symbol] = self._create_empty_result(len(prices), f'データ不足（{len(prices)}件 < {min_threshold}件）')
                    continue
                
                # カバレッジ率チェック
                coverage_ratio = len(prices) / len(price_df) if len(price_df) > 0 else 0
                if coverage_ratio < self.min_coverage_ratio:
                    self.logger.warning(f"{symbol}: カバレッジ不足（{coverage_ratio:.1%} < {self.min_coverage_ratio:.1%}）")
                    quality_report['excluded_assets'].append({
                        'symbol': symbol,
                        'reason': 'low_coverage',
                        'coverage_ratio': coverage_ratio,
                        'threshold': self.min_coverage_ratio
                    })
                    results[symbol] = self._create_empty_result(len(prices), f'カバレッジ不足（{coverage_ratio:.1%}）')
                    continue
                
                # ログリターンを計算
                log_returns = np.log(prices / prices.shift(1)).dropna()
                
                if len(log_returns) == 0:
                    self.logger.warning(f"{symbol}: ログリターンの計算に失敗")
                    quality_report['excluded_assets'].append({
                        'symbol': symbol,
                        'reason': 'log_return_calculation_failed',
                        'data_points': len(prices)
                    })
                    results[symbol] = self._create_empty_result(0, 'ログリターン計算失敗')
                    continue
                
                # ログリターンデータを保存（可視化用）
                self.log_returns_data[symbol] = log_returns
                
                # 月次メトリクスを計算・保存
                self.monthly_metrics_data[symbol] = self._calculate_monthly_metrics_for_symbol(log_returns, span_risk_free_rate)
                
                # 統合リターン・リスク指標を計算
                results[symbol] = self._calculate_all_return_risk_metrics(log_returns, symbol, span_risk_free_rate)
                quality_report['analyzed_assets'] += 1
                
            except Exception as e:
                self.logger.error(f"{symbol} のリターン・リスク指標計算エラー: {e}")
                quality_report['excluded_assets'].append({
                    'symbol': symbol,
                    'reason': 'calculation_error',
                    'error': str(e)
                })
                results[symbol] = self._create_empty_result(0, str(e))
        
        # 全体のカバレッジ率を計算
        if quality_report['total_assets'] > 0:
            quality_report['coverage_ratio'] = quality_report['analyzed_assets'] / quality_report['total_assets']
        
        return results, quality_report
    
    def _calculate_monthly_metrics_for_symbol(self, log_returns: pd.Series, risk_free_rate: float) -> pd.DataFrame:
        """シンボル用の月次メトリクスを計算"""
        try:
            if len(log_returns) < 10:  # 最低10日分のデータが必要
                return pd.DataFrame()
            
            # 月末でリサンプル
            monthly_resampled = log_returns.resample('M')
            
            monthly_metrics = []
            cumulative_returns = (1 + log_returns).cumprod()
            
            for name, group in monthly_resampled:
                if len(group) < 5:  # 最低5日分のデータが必要
                    continue
                
                # 月次リスク利子率（年率から月次に変換）
                monthly_risk_free = risk_free_rate / 12
                
                # 各指標を計算
                expected_return = group.mean()
                volatility = group.std()  # 標準偏差（ボラティリティ）
                downside_returns = group[group < monthly_risk_free]
                downside_deviation = downside_returns.std() if len(downside_returns) > 0 else 0.0
                positive_ratio = len(group[group > monthly_risk_free]) / len(group) * 100
                
                # 最大ドローダウン（月末時点での累積）
                month_end_cumulative = cumulative_returns.loc[:name]
                running_max = month_end_cumulative.expanding().max()
                drawdown = (month_end_cumulative - running_max) / running_max
                max_drawdown = drawdown.min()
                
                monthly_metrics.append({
                    'date': name,
                    'expected_return': expected_return,
                    'volatility': volatility,
                    'downside_deviation': downside_deviation,
                    'positive_ratio': positive_ratio,
                    'max_drawdown': max_drawdown
                })
            
            if monthly_metrics:
                return pd.DataFrame(monthly_metrics).set_index('date')
            else:
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"月次メトリクス計算エラー: {e}")
            return pd.DataFrame()
    
    def get_log_returns_data(self) -> Dict[str, pd.Series]:
        """ログリターンデータを取得（可視化用）"""
        return self.log_returns_data.copy()
    
    def get_monthly_metrics_data(self) -> Dict[str, pd.DataFrame]:
        """月次メトリクスデータを取得（時系列可視化用）"""
        return self.monthly_metrics_data.copy()
    
    def _convert_risk_free_rate_to_span(self, annual_risk_free_rate: float, span: str) -> float:
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
            
            span_risk_free_rate = annual_risk_free_rate * factor
            
            self.logger.info(f"無リスク利子率変換: {annual_risk_free_rate:.4f}(年率) -> {span_risk_free_rate:.6f}({span})")
            
            return span_risk_free_rate
            
        except Exception as e:
            self.logger.error(f"無リスク利子率変換エラー: {e}")
            return 0.0
    
    def _calculate_all_return_risk_metrics(self, log_returns: pd.Series, symbol: str, risk_free_rate: float) -> Dict[str, Any]:
        """全てのリターン・リスク指標を計算"""
        try:
            # 基本リターン統計量
            mean_return = log_returns.mean()          # 期待利益率（平均リターン）
            min_return = log_returns.min()            # 最小リターン
            max_return = log_returns.max()            # 最大リターン
            
            # 超過リターン（無リスク利子率を差し引いた）
            excess_returns = log_returns - risk_free_rate
            mean_excess_return = excess_returns.mean() # 超過期待利益率
            
            # リスク指標
            std_dev = log_returns.std()               # 標準偏差（ボラティリティ）
            variance = log_returns.var()              # 分散
            
            # 下方リスク
            downside_returns = log_returns[log_returns < risk_free_rate]  # 無リスク利子率を下回るリターン
            downside_std = downside_returns.std() if len(downside_returns) > 0 else 0.0
            
            # 最大ドローダウンの計算
            max_drawdown = self._calculate_max_drawdown(log_returns)
            
            # プラスリターン比率（無リスク利子率を上回る比率）
            positive_returns_ratio = len(log_returns[log_returns > risk_free_rate]) / len(log_returns) * 100
            
            # VaR（Value at Risk）の計算
            var_95 = log_returns.quantile(0.05)      # 5%VaR
            var_99 = log_returns.quantile(0.01)      # 1%VaR
            
            # CVaR（Conditional VaR）の計算
            cvar_95 = log_returns[log_returns <= var_95].mean() if len(log_returns[log_returns <= var_95]) > 0 else var_95
            cvar_99 = log_returns[log_returns <= var_99].mean() if len(log_returns[log_returns <= var_99]) > 0 else var_99
            
            # 分布の形状
            skewness = log_returns.skew()             # 歪度
            kurtosis = log_returns.kurtosis()         # 尖度
            
            # 効率性指標（無リスク利子率を考慮）
            sharpe_ratio = mean_excess_return / std_dev if std_dev > 0 else 0.0  # 修正済み
            sortino_ratio = mean_excess_return / downside_std if downside_std > 0 else 0.0  # 修正済み
            
            # ログ出力（デバッグ用）
            self.logger.debug(f"{symbol}: 無リスク利子率={risk_free_rate:.6f}, 平均リターン={mean_return:.6f}, 超過リターン={mean_excess_return:.6f}")
            self.logger.debug(f"{symbol}: シャープレシオ={sharpe_ratio:.4f}, ソルティノレシオ={sortino_ratio:.4f}")
            
            return {
                # リターン指標
                'expected_return': mean_return,              # 期待利益率
                'min_return': min_return,                    # 最小リターン
                'max_return': max_return,                    # 最大リターン
                'excess_return': mean_excess_return,         # 超過期待利益率（新規追加）
                
                # リスク指標
                'standard_deviation': std_dev,               # 標準偏差
                'variance': variance,                        # 分散
                'downside_deviation': downside_std,          # 下方偏差
                'max_drawdown': max_drawdown,                # 最大ドローダウン
                'positive_returns_ratio': positive_returns_ratio,  # プラスリターン比率（無リスク利子率基準）
                
                # VaR・CVaR
                'var_95': var_95,                           # VaR(95%)
                'var_99': var_99,                           # VaR(99%)
                'cvar_95': cvar_95,                         # CVaR(95%)
                'cvar_99': cvar_99,                         # CVaR(99%)
                
                # 分布統計
                'skewness': skewness,                       # 歪度
                'kurtosis': kurtosis,                       # 尖度
                
                # 効率性指標（無リスク利子率対応）
                'sharpe_ratio': sharpe_ratio,               # シャープレシオ（修正済み）
                'sortino_ratio': sortino_ratio,             # ソルティノレシオ（修正済み）
                
                # メタデータ
                'risk_free_rate': risk_free_rate,           # 使用した無リスク利子率
                'data_count': len(log_returns),             # データ数
                'success': True
            }
            
        except Exception as e:
            self.logger.error(f"リターン・リスク指標計算エラー: {e}")
            return self._create_empty_result(len(log_returns), str(e))
    
    def _calculate_max_drawdown(self, log_returns: pd.Series) -> float:
        """最大ドローダウンを計算"""
        try:
            # 累積リターンを計算
            cumulative_returns = (1 + log_returns).cumprod()
            
            # 各時点での最高値を計算
            running_max = cumulative_returns.expanding().max()
            
            # ドローダウンを計算
            drawdown = (cumulative_returns - running_max) / running_max
            
            # 最大ドローダウンを返す
            return drawdown.min()
            
        except Exception:
            return np.nan
    
    def _create_empty_result(self, count: int, error: str) -> Dict[str, Any]:
        """空の結果を作成"""
        return {
            'expected_return': np.nan,
            'min_return': np.nan,
            'max_return': np.nan,
            'excess_return': np.nan,
            'standard_deviation': np.nan,
            'variance': np.nan,
            'downside_deviation': np.nan,
            'max_drawdown': np.nan,
            'positive_returns_ratio': np.nan,
            'var_95': np.nan,
            'var_99': np.nan,
            'cvar_95': np.nan,
            'cvar_99': np.nan,
            'skewness': np.nan,
            'kurtosis': np.nan,
            'sharpe_ratio': np.nan,
            'sortino_ratio': np.nan,
            'risk_free_rate': np.nan,
            'data_count': count,
            'success': False,
            'error': error
        }


class ReturnRiskCalculationThread(QThread):
    """リターン・リスク指標計算スレッド"""
    
    calculation_completed = Signal(dict, dict, str)  # results, quality_report, span
    calculation_error = Signal(str)
    progress_updated = Signal(int, str)
    
    def __init__(self, price_df: pd.DataFrame, span: str, risk_free_rate: float = 0.0):
        super().__init__()
        self.price_df = price_df
        self.span = span
        self.risk_free_rate = risk_free_rate  # 無リスク利子率を追加
        self.calculator = ReturnRiskCalculator()
        self.logger = logging.getLogger(__name__)
    
    def run(self):
        """計算実行"""
        try:
            if self.price_df.empty:
                self.calculation_error.emit("計算対象データがありません．")
                return
            
            self.progress_updated.emit(0, "リターン・リスク指標計算を開始...")
            self.msleep(100)
            
            self.progress_updated.emit(20, "データ品質チェック中...")
            self.msleep(50)
            
            self.progress_updated.emit(40, "期待利益率・リターン統計を計算中...")
            self.msleep(50)
            
            self.progress_updated.emit(60, "リスク指標・標準偏差を計算中...")
            self.msleep(50)
            
            self.progress_updated.emit(80, "VaR・CVaR・効率性指標を計算中...")
            self.msleep(50)
            
            # リターン・リスク指標計算（データ品質管理対応版）
            results, quality_report = self.calculator.calculate_return_risk_metrics_from_dataframe(
                self.price_df, self.span, self.risk_free_rate
            )
            
            self.progress_updated.emit(100, "計算完了")
            self.calculation_completed.emit(results, quality_report, self.span)
            
        except Exception as e:
            self.logger.error(f"リターン・リスク指標計算エラー: {e}")
            self.calculation_error.emit(f"計算中にエラーが発生しました．: {str(e)}")


class ReturnRiskAnalysisWidget(AnalysisBaseWidget):
    """統合リターン・リスク指標分析ウィジェット"""
    
    def __init__(self, config=None):
        super().__init__(config)
        self.return_risk_results = {}
        self.quality_report = {}
        self.current_span = self.config.get('analysis.default_span', '日次')
        self.display_span = self.current_span
        self.current_risk_free_rate = self.config.get('analysis.default_risk_free_rate', 0.0)
        self.visualization_widget = None
        self.calculator = None

        # 設定からサイズを適用
        size_config = self.config.get_analysis_widget_size("return_risk_analysis")
        self.setMinimumHeight(size_config.get("min_height", 800))
        self.setMinimumWidth(size_config.get("min_width", 700))
        
        self.setup_content()
    
    def setup_header_content(self):
        """UIの設定"""
        # 表示単位選択
        self.unit_combo = self.create_combo_box(
            ["日次", "週次", "月次", "年次"],
            min_width="60px"
        )
        self.unit_combo.setCurrentText("日次")
        self.unit_combo.currentTextChanged.connect(self.update_display)
        
        self.header_layout.addWidget(QLabel("表示:", 
                                           styleSheet="color: #ffffff; font-size: 10px;"))
        self.header_layout.addWidget(self.unit_combo)
        
        self.header_layout.addStretch()
        
        # CSV出力ボタンを統一スタイルで作成
        self.export_button = self.create_button(
            "CSV出力",
            "export"
        )
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export_to_csv)
        self.header_layout.addWidget(self.export_button)
        
    def setup_content(self):
        """メインコンテンツエリアの設定"""
        # 統一スタイルのタブウィジェットを作成
        self.tab_widget = self.create_tab_widget()
        
        # タブ1：統合指標テーブル
        table_tab = QWidget()
        table_layout = QVBoxLayout(table_tab)
        table_layout.setContentsMargins(5, 5, 5, 5)
        
        # 統合リターン・リスク指標テーブル
        self.unified_table = self.create_table_widget()
        table_layout.addWidget(self.unified_table)
        
        self.tab_widget.addTab(table_tab, "指標テーブル")
        
        # タブ2：インタラクティブチャート
        if INTERACTIVE_VIZ_AVAILABLE:
            viz_tab = QWidget()
            viz_layout = QVBoxLayout(viz_tab)
            viz_layout.setContentsMargins(10, 10, 10, 10)
            
            # チャートウィジェット
            self.visualization_widget = InteractiveRiskVisualizationWidget()
            
            # チャートウィジェットのスタイル調整
            self.visualization_widget.setStyleSheet(f"""
                QWidget {{
                    background-color: {self.styles.COLORS["background"]};
                }}
                QLabel {{
                    font-size: 12px;
                    color: {self.styles.COLORS["text_primary"]};
                }}
                QListWidget {{
                    font-size: 14px;
                    min-height: 200px;
                    background-color: {self.styles.COLORS["surface"]};
                    color: {self.styles.COLORS["text_primary"]};
                    border: 1px solid {self.styles.COLORS["border"]};
                    border-radius: 4px;
                }}
                QListWidget::item {{
                    padding: 8px;
                    border-bottom: 1px solid {self.styles.COLORS["border"]};
                }}
                QListWidget::item:selected {{
                    background-color: {self.styles.COLORS["primary"]};
                }}
                QListWidget::item:hover {{
                    background-color: #4c4c4c;
                }}
            """)

            viz_layout.addWidget(self.visualization_widget)
            
            self.tab_widget.addTab(viz_tab, "チャート")
            
            # チャートステータス
            self.viz_status_label = QLabel("チャートデータ待機中...")
            self.viz_status_label.setStyleSheet(f"""
                QLabel {{
                    color: {self.styles.COLORS["text_secondary"]};
                    font-size: 9px;
                }}
            """)
            self.viz_status_label.setWordWrap(True)
            self.content_layout.addWidget(self.viz_status_label)
            
        else:
            # チャートが利用できない場合
            viz_error_tab = QWidget()
            viz_error_layout = QVBoxLayout(viz_error_tab)
            
            viz_error_label = QLabel("チャート機能が利用できません．")
            viz_error_label.setAlignment(Qt.AlignCenter)
            viz_error_label.setStyleSheet(self.styles.get_empty_state_style().replace(
                self.styles.COLORS["text_secondary"], "#ff6b6b").replace(
                "dashed", "dashed").replace(
                self.styles.COLORS["border"], "#ff6b6b"))
            viz_error_layout.addWidget(viz_error_label)
            
            self.tab_widget.addTab(viz_error_tab, "チャート作成に失敗 (要Plotly)")
        
        self.content_layout.addWidget(self.tab_widget)
    
    def get_empty_message(self) -> str:
        """空状態メッセージ（オーバーライド）"""
        return (
            "リターン・リスク指標が表示されていません．\n"
            "価格時系列データの取得完了後，表示されます．"
        )
    
    def convert_metric(self, value: float, metric_type: str, from_span: str, to_span: str) -> float:
        """指標のスパン変換"""
        if pd.isna(value) or from_span == to_span:
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
    
    def analyze(self, assets: List[AssetInfo], conditions: Dict[str, Any]):
        """分析実行"""
        if not self.price_data_source:
            QMessageBox.warning(self, "エラー", "価格データソースが設定されていません．")
            return
        
        if not self.price_data_source.is_ready():
            self.show_progress(True)
            self.update_progress(0, "価格データの取得完了を待機中...")
            self.price_data_source.data_ready.connect(self.on_price_data_ready)
            
            # 待機中でも無リスク利子率を保存
            self.current_risk_free_rate = conditions.get('risk_free_rate', 0.0)
            self.current_conditions = conditions  # 分析条件を保存
            return
        
        # 無リスク利子率を保存
        self.current_risk_free_rate = conditions.get('risk_free_rate', 0.0)
        self.current_conditions = conditions  # 分析条件を保存
        
        self.start_calculation()
    
    def on_price_data_ready(self):
        """価格データ準備完了時の処理"""
        try:
            self.price_data_source.data_ready.disconnect(self.on_price_data_ready)
        except (RuntimeError, TypeError):
            # 接続されていない場合は無視
            pass
        self.start_calculation()
    
    def start_calculation(self):
        """計算開始"""
        if not self.price_data_source:
            return
        
        price_df = self.price_data_source.get_analysis_dataframe()
        if price_df is None or price_df.empty:
            QMessageBox.warning(self, "エラー", "価格データの取得に失敗しました．")
            return
        
        self.current_span = self.price_data_source.get_current_span()
        self.display_span = self.current_span
        self.unit_combo.setCurrentText(self.current_span)
        
        # UI状態更新
        self.show_progress(True)
        self.show_quality_info("計算中...")
        self.show_main_content(False)
        self.export_button.setEnabled(False)
        
        # チャートステータスを更新
        if INTERACTIVE_VIZ_AVAILABLE:
            self.viz_status_label.setText("チャート計算中...")
        
        QApplication.processEvents()
        
        # 計算スレッド開始（無リスク利子率を渡す）
        self.calculation_thread = ReturnRiskCalculationThread(
            price_df, self.current_span, self.current_risk_free_rate
        )
        self.calculation_thread.progress_updated.connect(self.update_progress)
        self.calculation_thread.calculation_completed.connect(self.on_calculation_completed)
        self.calculation_thread.calculation_error.connect(self.on_calculation_error)
        self.calculation_thread.start()
    
    def get_analysis_conditions(self):
        """分析条件を取得（親から取得）"""
        try:
            # 保存された分析条件があればそれを使用
            if hasattr(self, 'current_conditions') and self.current_conditions:
                return self.current_conditions
            
            # 親ウィジェットを辿って分析条件を取得
            parent = self.parent()
            search_depth = 0
            max_depth = 10  # 無限ループ防止
            
            while parent and search_depth < max_depth:
                # 分析タブの特徴的なメソッドをチェック
                if hasattr(parent, 'get_analysis_conditions'):
                    conditions = parent.get_analysis_conditions()
                    if conditions:
                        return conditions
                
                # さらに詳細チェック：condition_tile があるか
                if hasattr(parent, 'condition_tile'):
                    if hasattr(parent.condition_tile, 'get_analysis_conditions'):
                        conditions = parent.condition_tile.get_analysis_conditions()
                        if conditions:
                            return conditions
                
                parent = parent.parent()
                search_depth += 1
            
            # デフォルト値を返す
            return {
                'risk_free_rate': 0.0,
                'start_date': datetime.now().date() - timedelta(days=365),
                'end_date': datetime.now().date() - timedelta(days=1),
                'span': '日次'
            }
            
        except Exception as e:
            return {
                'risk_free_rate': 0.0,
                'start_date': datetime.now().date() - timedelta(days=365),
                'end_date': datetime.now().date() - timedelta(days=1),
                'span': '日次'
            }
    
    def on_calculation_completed(self, results: Dict, quality_report: Dict, span: str):
        """計算完了"""
        self.return_risk_results = results
        self.quality_report = quality_report
        self.current_span = span
        
        # 計算機インスタンスを保存（可視化用データアクセス）
        self.calculator = self.calculation_thread.calculator
        
        # UI更新
        self.update_display()
        
        # UI状態更新
        self.show_progress(False)
        
        # データ品質情報を統合表示
        if quality_report:
            total = quality_report.get('total_assets', 0)
            analyzed = quality_report.get('analyzed_assets', 0)
            excluded = len(quality_report.get('excluded_assets', []))
            common_dates = quality_report.get('common_dates', 0)
            
            # 無リスク利子率情報
            risk_free_info = f"無リスク利子率: {self.current_risk_free_rate*100:.3f}%"
            
            # 統合表示（1行目）
            quality_text = f"分析: {analyzed}/{total} 資産 | 除外: {excluded} 資産 | 共通日付: {common_dates} 日 | {risk_free_info}"
            
            if quality_report.get('date_range'):
                start_date, end_date = quality_report['date_range']
                quality_text += f" | 期間: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}"

            # 除外理由がある場合は2行目に追加
            if excluded > 0:
                exclude_reasons = {}
                for asset in quality_report.get('excluded_assets', []):
                    reason = asset.get('reason', 'unknown')
                    if reason not in exclude_reasons:
                        exclude_reasons[reason] = 0
                    exclude_reasons[reason] += 1
                
                reason_text = ", ".join([f"{reason}: {count}件" for reason, count in exclude_reasons.items()])
                quality_text += f"\n除外理由: {reason_text}"
            
            self.show_quality_info(quality_text)
        
        if analyzed > 0:
            self.export_button.setEnabled(True)
            
            # チャートデータを更新
            if INTERACTIVE_VIZ_AVAILABLE and self.visualization_widget:
                try:
                    # 分析条件を取得
                    analysis_conditions = self.get_analysis_conditions()
                    
                    # データソースと分析条件を設定
                    self.visualization_widget.set_data_sources(
                        self.price_data_source, 
                        analysis_conditions
                    )
                    
                    # 計算機インスタンスを設定（拡張データアクセス用）
                    if hasattr(self.visualization_widget, 'set_calculator'):
                        self.visualization_widget.set_calculator(self.calculator)
                    
                    # データを更新
                    self.visualization_widget.update_data(
                        self.return_risk_results, 
                        self.current_span, 
                        self.display_span
                    )
                    self.viz_status_label.setText(f"チャート: {analyzed}/{total} 件")
                    
                except Exception as e:
                    self.viz_status_label.setText("チャート更新エラー")
    
    def on_calculation_error(self, error_message: str):
        """計算エラー"""
        self.show_progress(False)
        self.hide_quality_info()
        if INTERACTIVE_VIZ_AVAILABLE:
            self.viz_status_label.setText("チャート計算エラー")
        QMessageBox.critical(self, "エラー", error_message)
        self.update_display()
    
    def update_display(self):
        """表示更新（統合テーブル）"""
        if not self.return_risk_results:
            self.show_main_content(False)
            return
        
        self.show_main_content(True)
        
        self.display_span = self.unit_combo.currentText()
        
        # 成功したデータのみを取得
        valid_results = {k: v for k, v in self.return_risk_results.items() 
                        if v.get('success', False)}
        
        if self.price_data_source and self.price_data_source.processed_dataframe is not None:
            column_order = list(self.price_data_source.processed_dataframe.columns)
            ordered_symbols = [symbol for symbol in column_order if symbol in valid_results]
        else:
            ordered_symbols = sorted(valid_results.keys())
        
        if ordered_symbols:
            self.update_unified_table(ordered_symbols, valid_results)
        
        # チャートも更新（表示単位変更時）
        if INTERACTIVE_VIZ_AVAILABLE and self.visualization_widget:
            self.visualization_widget.update_data(
                self.return_risk_results,
                self.current_span,
                self.display_span
            )
    
    def update_unified_table(self, symbols: List[str], results: Dict):
        """統合リターン・リスク指標テーブル更新（18項目の指標を表示）"""
        table = self.unified_table
        
        # 全ての指標を定義（カテゴリ順）
        all_metrics = [
            # リターン指標
            ("期待利益率", "expected_return", "expected_return", "%", 100),
            ("超過期待利益率", "excess_return", "excess_return", "%", 100),
            ("最小リターン", "min_return", "min_return", "%", 100),
            ("最大リターン", "max_return", "max_return", "%", 100),
            
            # リスク指標
            ("標準偏差", "standard_deviation", "standard_deviation", "%", 100),
            ("分散", "variance", "variance", "", 10000),
            ("下方偏差", "downside_deviation", "downside_deviation", "%", 100),
            ("最大ドローダウン", "max_drawdown", "max_drawdown", "%", 100),
            ("プラスリターン比率", "positive_returns_ratio", "positive_returns_ratio", "%", 1),
            
            # VaR・CVaR
            ("VaR(95%)", "var_95", "var_95", "%", 100),
            ("VaR(99%)", "var_99", "var_99", "%", 100),
            ("CVaR(95%)", "cvar_95", "cvar_95", "%", 100),
            ("CVaR(99%)", "cvar_99", "cvar_99", "%", 100),
            
            # 分布統計
            ("歪度", "skewness", "skewness", "", 1),
            ("尖度", "kurtosis", "kurtosis", "", 1),
            
            # 効率性指標
            ("シャープレシオ", "sharpe_ratio", "sharpe_ratio", "", 1),
            ("ソルティノレシオ", "sortino_ratio", "sortino_ratio", "", 1),
            
            # メタデータ
            ("データ数", "data_count", "data_count", "件", 1),
        ]
        
        table.setRowCount(len(all_metrics))
        table.setColumnCount(len(symbols))
        
        # ヘッダー設定
        table.setVerticalHeaderLabels([metric[0] for metric in all_metrics])
        table.setHorizontalHeaderLabels(symbols)
        
        # データ挿入
        for row_idx, (name, key, metric_type, unit, multiplier) in enumerate(all_metrics):
            for col_idx, symbol in enumerate(symbols):
                result = results[symbol]
                
                # スパン変換の適用
                if key in ['expected_return', 'excess_return', 'min_return', 'max_return', 'standard_deviation', 'variance', 
                          'downside_deviation', 'var_95', 'var_99', 'cvar_95', 'cvar_99', 
                          'sharpe_ratio', 'sortino_ratio']:
                    original_value = result.get(key, np.nan)
                    converted_value = self.convert_metric(original_value, metric_type, self.current_span, self.display_span)
                else:
                    converted_value = result.get(key, np.nan)
                
                if pd.isna(converted_value):
                    item = QTableWidgetItem("-")
                    item.setForeground(Qt.gray)
                else:
                    if key == "data_count":
                        item = QTableWidgetItem(f"{int(converted_value)}{unit}")
                    else:
                        display_value = converted_value * multiplier
                        item = QTableWidgetItem(f"{display_value:.3f}{unit}")
                        
                        # 色分け
                        if key in ['expected_return', 'excess_return', 'max_return']:
                            if converted_value > 0:
                                item.setForeground(Qt.red)
                            elif converted_value < 0:
                                item.setForeground(Qt.green)
                            else:
                                item.setForeground(Qt.white)
                        elif key == 'min_return':
                            if converted_value < 0:
                                item.setForeground(Qt.green)
                            elif converted_value > 0:
                                item.setForeground(Qt.red)
                            else:
                                item.setForeground(Qt.white)
                        elif key in ['standard_deviation', 'variance', 'downside_deviation'] and converted_value > 0:
                            if display_value > 20:
                                item.setForeground(Qt.yellow)
                            elif display_value < 5:
                                item.setForeground(Qt.cyan)
                            else:
                                item.setForeground(Qt.white)
                        elif key == 'max_drawdown' and converted_value < 0:
                            abs_value = abs(display_value)
                            if abs_value > 30:
                                item.setForeground(Qt.yellow)
                            elif abs_value < 10:
                                item.setForeground(Qt.cyan)
                            else:
                                item.setForeground(Qt.white)
                        elif key in ['var_95', 'var_99', 'cvar_95', 'cvar_99']:
                            abs_value = abs(display_value)
                            if abs_value > 10:
                                item.setForeground(Qt.yellow)
                            elif abs_value < 3:
                                item.setForeground(Qt.cyan)
                            else:
                                item.setForeground(Qt.white)
                        elif key == "positive_returns_ratio":
                            if converted_value >= 50:
                                item.setForeground(Qt.red)
                            else:
                                item.setForeground(Qt.green)
                        elif key in ["sharpe_ratio", "sortino_ratio"]:
                            if converted_value > 1.0:
                                item.setForeground(Qt.red)
                            elif converted_value < 0:
                                item.setForeground(Qt.green)
                            else:
                                item.setForeground(Qt.white)
                        elif key == "skewness":
                            if converted_value > 0.5:
                                item.setForeground(Qt.magenta)
                            elif converted_value < -0.5:
                                item.setForeground(Qt.blue)
                            else:
                                item.setForeground(Qt.white)
                        elif key == "kurtosis":
                            if converted_value > 1.0:
                                item.setForeground(Qt.magenta)
                            elif converted_value < -1.0:
                                item.setForeground(Qt.blue)
                            else:
                                item.setForeground(Qt.white)
                        else:
                            item.setForeground(Qt.white)
                
                item.setTextAlignment(Qt.AlignCenter)
                item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
                
                # ツールチップ設定（データ品質情報を追加）
                if key in ['expected_return', 'excess_return', 'min_return', 'max_return', 'standard_deviation', 'variance', 
                          'downside_deviation', 'var_95', 'var_99', 'cvar_95', 'cvar_99', 
                          'sharpe_ratio', 'sortino_ratio']:
                    original_value = result.get(key, np.nan)
                    risk_free_info = f"無リスク利子率: {result.get('risk_free_rate', 0)*100:.3f}%({self.current_span})"
                    data_quality_info = f"データ数: {result.get('data_count', 0)}件（共通日付のみ）"
                    tooltip = (f"シンボル: {symbol}\n"
                              f"{name}({self.current_span}): {original_value*multiplier:.4f}{unit}\n"
                              f"{name}({self.display_span}): {display_value:.4f}{unit}\n"
                              f"{risk_free_info}\n"
                              f"{data_quality_info}")
                else:
                    tooltip = f"シンボル: {symbol}\n{name}: {converted_value:.4f}{unit}"
                item.setToolTip(tooltip)
                
                table.setItem(row_idx, col_idx, item)
        
        table.resizeColumnsToContents()
        table.resizeRowsToContents()
        
        # 列幅の調整
        for col in range(table.columnCount()):
            current_width = table.columnWidth(col)
            table.setColumnWidth(col, min(max(current_width, 60), 80))
        
        # 行ヘッダーの幅を適切に設定
        header = table.verticalHeader()
        header.setMinimumWidth(130)
        header.setMaximumWidth(160)
    
    def export_to_csv(self):
        """CSV出力"""
        if not self.return_risk_results:
            QMessageBox.information(self, "Information", "出力するデータがありません．")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "リターン・リスク指標をCSV出力", "return_risk_metrics.csv", "CSV ファイル (*.csv)"
        )
        
        if file_path:
            try:
                # 成功したデータのみを取得
                valid_results = {k: v for k, v in self.return_risk_results.items() 
                                if v.get('success', False)}
                
                # 価格データソースの列順序を取得
                if self.price_data_source and self.price_data_source.processed_dataframe is not None:
                    column_order = list(self.price_data_source.processed_dataframe.columns)
                    ordered_symbols = [symbol for symbol in column_order if symbol in valid_results]
                else:
                    ordered_symbols = sorted(valid_results.keys())
                
                if not ordered_symbols:
                    QMessageBox.information(self, "Information", "出力する有効なデータがありません．")
                    return
                
                # 全指標を定義（超過期待利益率を追加）
                all_metrics = [
                    ("Expected_Return", "expected_return"),
                    ("Excess_Return", "excess_return"),
                    ("Min_Return", "min_return"),
                    ("Max_Return", "max_return"),
                    ("Standard_Deviation", "standard_deviation"),
                    ("Variance", "variance"),
                    ("Downside_Deviation", "downside_deviation"),
                    ("Max_Drawdown", "max_drawdown"),
                    ("Positive_Returns_Ratio", "positive_returns_ratio"),
                    ("VaR_95", "var_95"),
                    ("VaR_99", "var_99"),
                    ("CVaR_95", "cvar_95"),
                    ("CVaR_99", "cvar_99"),
                    ("Skewness", "skewness"),
                    ("Kurtosis", "kurtosis"),
                    ("Sharpe_Ratio", "sharpe_ratio"),
                    ("Sortino_Ratio", "sortino_ratio"),
                    ("Risk_Free_Rate", "risk_free_rate"),
                    ("Data_Count", "data_count"),
                ]
                
                # CSVファイルに出力（行=項目，列=銘柄）
                with open(file_path, 'w', encoding='utf-8') as f:
                    # データ品質情報を最初に出力
                    if self.quality_report:
                        f.write('# Data Quality Report\n')
                        f.write(f'# Total Number of Assets: {self.quality_report.get("total_assets", 0)}\n')
                        f.write(f'# Number of Assets for Analysis: {self.quality_report.get("analyzed_assets", 0)}\n')
                        f.write(f'# Number of Excluded Assets: {len(self.quality_report.get("excluded_assets", []))}\n')
                        f.write(f'# Number of Common Dates: {self.quality_report.get("common_dates", 0)}\n')
                        f.write(f'# Minimum Data Threshold: {self.quality_report.get("min_data_threshold", 30)}\n')
                        f.write('\n')
                    
                    # ヘッダー行
                    f.write('Assets,' + ','.join(ordered_symbols) + '\n')
                    
                    # 各指標の行
                    for metric_name, metric_key in all_metrics:
                        f.write(f'{metric_name},')
                        values = []
                        for symbol in ordered_symbols:
                            result = valid_results[symbol]
                            value = result.get(metric_key, np.nan)
                            if pd.isna(value):
                                values.append("")
                            else:
                                values.append(f"{value:.6f}")
                        f.write(','.join(values) + '\n')
                
                # 完了メッセージにデータ品質情報を含める
                quality_info = ""
                if self.quality_report:
                    analyzed = self.quality_report.get('analyzed_assets', 0)
                    total = self.quality_report.get('total_assets', 0)
                    quality_info = f"\n\nデータ品質管理結果: {analyzed}/{total} 資産を分析対象としました．"
                
                QMessageBox.information(self, "完了", 
                                      f"CSVファイルを出力しました．:\n{file_path}{quality_info}")
                
            except Exception as e:
                QMessageBox.critical(self, "エラー", f"CSV出力中にエラーが発生しました:\n{str(e)}")
    
    def clear_data(self):
        """データクリア"""
        self.return_risk_results = {}
        self.quality_report = {}
        self.current_span = '日次'
        self.display_span = '日次'
        self.current_risk_free_rate = 0.0
        self.visualization_widget = None
        self.calculator = None
        # 設定からサイズを取得
        min_height = self.config.get('analysis.widget_sizes.return_risk_analysis.min_height', 800)
        min_width = self.config.get('analysis.widget_sizes.return_risk_analysis.min_width', 700)
        self.setMinimumHeight(min_height)
        self.setMinimumWidth(min_width)
        
        self.setup_content()
        self.update_display()
        self.export_button.setEnabled(False)
        self.show_progress(False)
        self.hide_quality_info()
        
        # チャートも無効化
        if INTERACTIVE_VIZ_AVAILABLE:
            self.viz_status_label.setText("チャートデータ待機中...")
    
    def get_results(self) -> Dict[str, Dict[str, Any]]:
        """リターン・リスク指標結果を取得"""
        return self.return_risk_results
    
    def get_quality_report(self) -> Dict[str, Any]:
        """データ品質レポートを取得"""
        return self.quality_report
    
    def get_log_returns_data(self) -> Dict[str, pd.Series]:
        """ログリターンデータを取得（可視化用）"""
        if self.calculator:
            return self.calculator.get_log_returns_data()
        else:
            return {}
    
    def get_monthly_metrics_data(self) -> Dict[str, pd.DataFrame]:
        """月次メトリクスデータを取得（時系列可視化用）"""
        if self.calculator:
            return self.calculator.get_monthly_metrics_data()
        else:
            return {}