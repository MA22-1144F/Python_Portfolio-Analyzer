"""security_market_line.py"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any
import logging
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QMessageBox, QApplication, QFileDialog,
    QTextEdit, QTableWidgetItem
)
from PySide6.QtCore import QThread, Signal, Qt, QTimer
from data.asset_info import AssetInfo
from analysis.analysis_base_widget import AnalysisBaseWidget
from analysis.market_data_fetcher import MarketDataFetcher

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import matplotlib.figure
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

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


class SecurityMarketLineCalculationThread(QThread):
    """証券市場線計算スレッド"""
    
    calculation_completed = Signal(dict)  # results
    calculation_error = Signal(str)
    progress_updated = Signal(int, str)
    
    def __init__(self, price_df: pd.DataFrame, conditions: Dict[str, Any], config=None):
        super().__init__()
        self.price_df = price_df
        self.conditions = conditions
        # 設定を取得
        if config is None:
            from config.app_config import AppConfig
            config = AppConfig()
        self.config = config
        
        self.market_fetcher = MarketDataFetcher()
        self.logger = logging.getLogger(__name__)
    
    def run(self):
        """計算実行"""
        try:
            if self.price_df.empty:
                self.calculation_error.emit("計算用データが利用できません．")
                return
            
            self.progress_updated.emit(5, "分析条件を確認中...")
            self.msleep(100)
            
            # 分析条件の取得
            start_date = self.conditions.get('start_date')
            end_date = self.conditions.get('end_date')
            span = self.conditions.get('span', '日次')
            risk_free_rate = self.conditions.get('risk_free_rate', 0.0)
            market_portfolio = self.conditions.get('market_portfolio', 'Nikkei 225 (^N225)')
            
            # 無リスク利子率のスパン変換
            span_risk_free_rate = self._convert_risk_free_rate_to_span(risk_free_rate, span)
            
            self.progress_updated.emit(10, "資産のログリターンを計算中...")
            self.msleep(50)
            
            # 各資産のログリターンを計算
            asset_returns = self._calculate_asset_log_returns(self.price_df)
            if not asset_returns:
                self.calculation_error.emit("資産のログリターンの計算に失敗しました．")
                return
            
            # 資産数チェック
            n_assets = len(asset_returns)
            if n_assets < 1:
                self.calculation_error.emit("証券市場線分析には最低1つの資産が必要です．")
                return
            
            self.progress_updated.emit(30, "市場ポートフォリオデータを取得中...")
            self.msleep(50)
            
            # 市場ポートフォリオデータの取得
            market_returns, market_status = self.market_fetcher.fetch_market_data(
                market_portfolio, start_date, end_date, span
            )
            
            if not market_status["success"]:
                self.calculation_error.emit(f"市場データの取得に失敗しました: {market_status['error_message']}")
                return
            
            self.progress_updated.emit(50, "回帰分析によるβ値を計算中...")
            self.msleep(50)
            
            # β値と期待利益率の計算（回帰分析）
            beta_results = self._calculate_beta_values_regression(asset_returns, market_returns, span_risk_free_rate)
            
            if not beta_results:
                self.calculation_error.emit("β値の計算に失敗しました．")
                return
            
            self.progress_updated.emit(70, "市場期待利益率を計算中...")
            self.msleep(50)
            
            # 市場期待利益率の計算
            market_expected_return = market_returns.mean()
            market_excess_return = market_expected_return - span_risk_free_rate
            
            self.progress_updated.emit(85, "証券市場線を計算中...")
            self.msleep(50)
            
            # 証券市場線の計算
            sml_line = self._calculate_security_market_line(span_risk_free_rate, market_excess_return)
            
            # 結果の構築
            results = {
                'asset_returns': asset_returns,
                'market_returns': market_returns,
                'market_status': market_status,
                'beta_results': beta_results,
                'market_expected_return': market_expected_return,
                'market_excess_return': market_excess_return,
                'span_risk_free_rate': span_risk_free_rate,
                'sml_line': sml_line,
                'conditions': self.conditions,
                'n_assets': n_assets
            }
            
            self.progress_updated.emit(100, "計算完了")
            self.calculation_completed.emit(results)
            
        except Exception as e:
            self.logger.error(f"証券市場線計算エラー: {e}")
            self.calculation_error.emit(f"計算中にエラーが発生しました: {str(e)}")
    
    def _convert_risk_free_rate_to_span(self, annual_rate: float, span: str) -> float:
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
            self.logger.info(f"無リスク利子率変換: {annual_rate:.4f}(年率) -> {span_rate:.6f}({span})")
            
            return span_rate
            
        except Exception as e:
            self.logger.error(f"無リスク利子率変換エラー: {e}")
            return 0.0
    
    def _calculate_asset_log_returns(self, price_df: pd.DataFrame) -> Dict[str, pd.Series]:
        """各資産のログリターンを計算"""
        asset_returns = {}
        
        try:
            for symbol in price_df.columns:
                try:
                    prices = price_df[symbol].dropna()
                    if len(prices) < 2:
                        self.logger.warning(f"{symbol}: 価格データが不足")
                        continue
                    
                    # ログリターンを計算
                    log_returns = np.log(prices / prices.shift(1)).dropna()
                    
                    if len(log_returns) > 0:
                        asset_returns[symbol] = log_returns
                        self.logger.info(f"{symbol}: ログリターン {len(log_returns)}件を計算")
                    else:
                        self.logger.warning(f"{symbol}: ログリターンの計算に失敗")
                        
                except Exception as e:
                    self.logger.error(f"{symbol} のログリターン計算エラー: {e}")
                    continue
            
            return asset_returns
            
        except Exception as e:
            self.logger.error(f"ログリターン計算エラー: {e}")
            return {}
    
    def _calculate_beta_values_regression(self, asset_returns: Dict[str, pd.Series], 
                                        market_returns: pd.Series, risk_free_rate: float) -> Dict[str, Dict]:
        """回帰分析による各資産のβ値と関連指標を計算"""
        beta_results = {}
        
        try:
            for symbol, returns in asset_returns.items():
                try:
                    # 共通日付でのデータを取得
                    common_dates = returns.index.intersection(market_returns.index)
                    
                    if len(common_dates) < 10:  # 最低10日分のデータが必要
                        self.logger.warning(f"{symbol}: 共通日付が不足（{len(common_dates)}日）")
                        continue
                    
                    asset_common = returns.reindex(common_dates)
                    market_common = market_returns.reindex(common_dates)
                    
                    # 超過リターンを計算
                    asset_excess = asset_common - risk_free_rate
                    market_excess = market_common - risk_free_rate
                    
                    # NaNを除去
                    valid_mask = ~(np.isnan(asset_excess) | np.isnan(market_excess))
                    asset_excess_clean = asset_excess[valid_mask]
                    market_excess_clean = market_excess[valid_mask]
                    
                    if len(asset_excess_clean) < 5:
                        self.logger.warning(f"{symbol}: 有効データが不足（{len(asset_excess_clean)}日）")
                        continue
                    
                    # 回帰分析の実行
                    if SCIPY_AVAILABLE:
                        # scipy.statsを使用した回帰分析
                        slope, intercept, r_value, p_value, std_err = stats.linregress(
                            market_excess_clean, asset_excess_clean
                        )
                        
                        # 回帰統計
                        beta = slope
                        alpha = intercept
                        correlation = r_value
                        r_squared = r_value ** 2
                        
                        # 信頼区間の計算（95%信頼区間）
                        n = len(asset_excess_clean)
                        dof = n - 2  # 自由度
                        t_val = stats.t.ppf(0.975, dof)  # 95%信頼区間のt値
                        beta_ci_lower = beta - t_val * std_err
                        beta_ci_upper = beta + t_val * std_err
                        
                        # 予測値と残差
                        predicted = alpha + beta * market_excess_clean
                        residuals = asset_excess_clean - predicted
                        
                        # 残差の統計
                        residual_std = np.std(residuals, ddof=2)
                        
                        # F統計量とp値の計算
                        mse_residual = np.sum(residuals**2) / dof
                        mse_total = np.sum((asset_excess_clean - np.mean(asset_excess_clean))**2) / (n-1)
                        f_statistic = (mse_total - mse_residual) / mse_residual if mse_residual > 0 else np.inf
                        f_p_value = 1 - stats.f.cdf(f_statistic, 1, dof) if np.isfinite(f_statistic) else 0
                        
                    else:
                        # numpyのみを使用した簡易回帰分析
                        X = np.column_stack([np.ones(len(market_excess_clean)), market_excess_clean])
                        y = asset_excess_clean.values
                        
                        # 最小二乗法
                        coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
                        alpha = coeffs[0]
                        beta = coeffs[1]
                        
                        # 基本統計
                        predicted = alpha + beta * market_excess_clean
                        residuals = y - predicted
                        correlation = np.corrcoef(market_excess_clean, asset_excess_clean)[0, 1]
                        r_squared = correlation ** 2
                        
                        # 標準誤差の近似計算
                        residual_variance = np.var(residuals, ddof=2)
                        market_variance = np.var(market_excess_clean, ddof=1)
                        std_err = np.sqrt(residual_variance / (len(market_excess_clean) * market_variance))
                        
                        # 統計量の近似
                        t_stat = beta / std_err if std_err > 0 else np.inf
                        p_value = 2 * (1 - stats.norm.cdf(abs(t_stat))) if 'stats' in globals() else np.nan
                        
                        # 信頼区間の近似
                        z_val = 1.96  # 95%信頼区間のz値（近似）
                        beta_ci_lower = beta - z_val * std_err
                        beta_ci_upper = beta + z_val * std_err
                        
                        residual_std = np.std(residuals, ddof=2)
                        f_statistic = np.nan
                        f_p_value = np.nan
                    
                    # 期待利益率の計算
                    expected_return = asset_common.mean()
                    
                    # CAPM期待利益率の計算
                    capm_expected_return = risk_free_rate + beta * (market_common.mean() - risk_free_rate)
                    
                    # システマティックリスクとアンシステマティックリスク
                    market_variance = np.var(market_excess_clean, ddof=1)
                    total_variance = np.var(asset_excess_clean, ddof=1)
                    systematic_variance = beta ** 2 * market_variance
                    unsystematic_variance = max(0, total_variance - systematic_variance)
                    
                    # 結果の格納
                    beta_results[symbol] = {
                        'beta': beta,
                        'alpha': alpha,
                        'expected_return': expected_return,
                        'capm_expected_return': capm_expected_return,
                        'correlation': correlation,
                        'r_squared': r_squared,
                        'systematic_risk': np.sqrt(systematic_variance) if systematic_variance >= 0 else 0,
                        'unsystematic_risk': np.sqrt(unsystematic_variance) if unsystematic_variance >= 0 else 0,
                        'total_risk': np.sqrt(total_variance),
                        'common_dates': len(asset_excess_clean),
                        # 回帰分析の詳細統計
                        'beta_std_error': std_err,
                        'beta_t_statistic': beta / std_err if std_err > 0 else np.inf,
                        'beta_p_value': p_value if 'p_value' in locals() else np.nan,
                        'beta_ci_lower': beta_ci_lower,
                        'beta_ci_upper': beta_ci_upper,
                        'residual_std_error': residual_std,
                        'f_statistic': f_statistic if 'f_statistic' in locals() else np.nan,
                        'f_p_value': f_p_value if 'f_p_value' in locals() else np.nan,
                        'durbin_watson': self._calculate_durbin_watson(residuals) if 'residuals' in locals() else np.nan,
                        'success': True
                    }
                    
                    self.logger.info(f"{symbol}: β={beta:.4f}±{std_err:.4f}, α={alpha:.4f}, R²={r_squared:.4f}")
                    
                except Exception as e:
                    self.logger.error(f"{symbol} の回帰分析エラー: {e}")
                    beta_results[symbol] = {
                        'beta': np.nan, 'alpha': np.nan, 'expected_return': np.nan, 
                        'capm_expected_return': np.nan, 'correlation': np.nan, 'r_squared': np.nan,
                        'systematic_risk': np.nan, 'unsystematic_risk': np.nan, 'total_risk': np.nan,
                        'common_dates': 0, 'beta_std_error': np.nan, 'beta_t_statistic': np.nan,
                        'beta_p_value': np.nan, 'beta_ci_lower': np.nan, 'beta_ci_upper': np.nan,
                        'residual_std_error': np.nan, 'f_statistic': np.nan, 'f_p_value': np.nan,
                        'durbin_watson': np.nan, 'success': False, 'error': str(e)
                    }
            
            return beta_results
            
        except Exception as e:
            self.logger.error(f"回帰分析エラー: {e}")
            return {}
    
    def _calculate_durbin_watson(self, residuals):
        """ダービン・ワトソン統計量の計算（系列相関の検定）"""
        try:
            if len(residuals) < 2:
                return np.nan
            
            diff_residuals = np.diff(residuals)
            dw = np.sum(diff_residuals**2) / np.sum(residuals**2)
            return dw
        except Exception:
            return np.nan
    
    def _calculate_security_market_line(self, risk_free_rate: float, market_excess_return: float) -> Dict:
        """証券市場線の計算"""
        try:
            # β値の範囲を設定（-0.5から2.5まで）
            beta_range = np.linspace(-0.5, 2.5, 100)
            
            # 証券市場線の期待利益率を計算
            # E(R) = Rf + β × (E(Rm) - Rf)
            sml_returns = risk_free_rate + beta_range * market_excess_return
            
            return {
                'beta_range': beta_range.tolist(),
                'sml_returns': sml_returns.tolist(),
                'slope': market_excess_return,
                'intercept': risk_free_rate
            }
            
        except Exception as e:
            self.logger.error(f"証券市場線計算エラー: {e}")
            return {'beta_range': [], 'sml_returns': [], 'slope': 0, 'intercept': 0}


class MatplotlibCanvas(FigureCanvas):
    """Matplotlib canvas for Qt integration"""
    
    def __init__(self, parent=None, width=8, height=6, dpi=100):
        self.figure = matplotlib.figure.Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.figure)
        self.setParent(parent)
        
        # Dark theme setup
        self.figure.patch.set_facecolor('#2b2b2b')
        
    def clear_plot(self):
        """プロットを完全にクリア"""
        self.figure.clear()
        self.draw()


class SecurityMarketLineWidget(AnalysisBaseWidget):
    """証券市場線分析ウィジェット"""
    
    def __init__(self, config=None):
        super().__init__(config)
        self.analysis_results = None
        self.current_figure = None
        self.temp_html_files = []
        self.browser_thread = None
        self.matplotlib_canvas = None
        self.current_conditions = None
        self.logger = logging.getLogger(__name__)
        # 設定からサイズを取得
        min_height = self.config.get('analysis.widget_sizes.security_market_line.min_height', 500)
        min_width = self.config.get('analysis.widget_sizes.security_market_line.min_width', 700)
        self.setMinimumHeight(min_height)
        self.setMinimumWidth(min_width)
        
        self.setup_content()

    def analyze(self, assets: List[AssetInfo], conditions: Dict[str, Any]):
        """分析実行"""
        if not self.price_data_source:
            QMessageBox.warning(self, "エラー", "価格データソースが設定されていません．")
            return
        
        if not self.price_data_source.is_ready():
            self.show_progress(True)
            self.update_progress(0, "価格データの取得完了を待機中...")
            self.price_data_source.data_ready.connect(self.on_price_data_ready)
            
            # 分析条件を保存
            self.current_conditions = conditions
            
            # tile_conditionsから市場ポートフォリオを取得してヘッダーに反映
            if 'market_portfolio' in conditions:
                market_text = conditions['market_portfolio']
                index = self.market_combo.findText(market_text)
                if index >= 0:
                    # シグナルを一時的に切断して無限ループを防ぐ
                    self.market_combo.currentTextChanged.disconnect()
                    self.market_combo.setCurrentIndex(index)
                    self.market_combo.currentTextChanged.connect(self.on_market_changed)
            
            return
        
        # 分析条件を保存
        self.current_conditions = conditions
        
        # tile_conditionsから市場ポートフォリオを取得してヘッダーに反映
        if 'market_portfolio' in conditions:
            market_text = conditions['market_portfolio']
            index = self.market_combo.findText(market_text)
            if index >= 0:
                # シグナルを一時的に切断して無限ループを防ぐ
                self.market_combo.currentTextChanged.disconnect()
                self.market_combo.setCurrentIndex(index)
                self.market_combo.currentTextChanged.connect(self.on_market_changed)
        
        self.start_calculation(conditions)
    
    def on_price_data_ready(self):
        """価格データ準備完了時の処理"""
        try:
            self.price_data_source.data_ready.disconnect(self.on_price_data_ready)
        except (RuntimeError, TypeError):
            # 接続されていない場合は無視
            pass
        
        if hasattr(self, 'current_conditions') and self.current_conditions:
            self.start_calculation(self.current_conditions)

    def setup_header_content(self):
        """UIの設定"""
        # 市場ポートフォリオ選択
        market_fetcher = MarketDataFetcher()
        available_markets = market_fetcher.get_available_markets()
        
        self.market_combo = self.create_combo_box(
            available_markets,
            min_width="200px"
        )
        self.market_combo.setCurrentText("Nikkei 225 (^N225)")
        # 市場変更時に再計算を実行
        self.market_combo.currentTextChanged.connect(self.on_market_changed)
        
        self.header_layout.addWidget(QLabel("市場:", 
                                           styleSheet="color: #ffffff; font-size: 10px;"))
        self.header_layout.addWidget(self.market_combo)
        
        self.header_layout.addStretch()

        # エクスポートボタン
        self.browser_button = self.create_button(
            "ブラウザで表示",
            "primary"
        )
        self.browser_button.setEnabled(False)
        self.browser_button.clicked.connect(self.open_in_browser)
        self.header_layout.addWidget(self.browser_button)

        self.export_chart_button = self.create_button(
            "HTML保存",
            "save"
        )
        self.export_chart_button.setEnabled(False)
        self.export_chart_button.clicked.connect(self.export_chart)
        self.header_layout.addWidget(self.export_chart_button)
        
        self.export_data_button = self.create_button(
            "CSV出力",
            "export"
        )
        self.export_data_button.setEnabled(False)
        self.export_data_button.clicked.connect(self.export_data)
        self.header_layout.addWidget(self.export_data_button)
    
    def on_market_changed(self):
        """市場ポートフォリオ変更時の処理"""
        # 分析結果がある場合のみ再計算を実行
        if self.analysis_results and self.current_conditions:
            # 現在の条件を更新
            self.current_conditions['market_portfolio'] = self.market_combo.currentText()
            
            # 親ウィジェットの条件タイルにも反映
            try:
                analysis_tab = self.get_analysis_tab()
                if analysis_tab and hasattr(analysis_tab, 'condition_tile'):
                    condition_tile = analysis_tab.condition_tile
                    # 市場ポートフォリオを更新
                    index = condition_tile.market_combo.findText(self.market_combo.currentText())
                    if index >= 0:
                        condition_tile.market_combo.setCurrentIndex(index)
            except Exception as e:
                self.logger.warning(f"条件タイルへの市場ポートフォリオ反映に失敗: {e}")
            
            # 再計算を実行
            self.start_calculation(self.current_conditions)
        
    def get_analysis_tab(self):
        """親の分析タブを取得"""
        parent = self.parent()
        search_depth = 0
        max_depth = 10
        
        while parent and search_depth < max_depth:
            if hasattr(parent, 'condition_tile'):
                return parent
            parent = parent.parent()
            search_depth += 1
        
        return None
    
    def setup_content(self):
        """メインコンテンツエリアの設定"""
        # 市場情報ラベル
        self.market_info_label = QLabel("")
        self.market_info_label.setStyleSheet("color: #0078d4; font-size: 9px;")
        self.market_info_label.setVisible(False)
        self.market_info_label.setWordWrap(True)
        self.market_info_label.setMaximumHeight(30)
        self.content_layout.addWidget(self.market_info_label)
        
        # 統一スタイルのタブウィジェットを作成
        self.tab_widget = self.create_tab_widget()
        
        # タブ1: チャート表示
        chart_tab = QWidget()
        chart_layout = QVBoxLayout(chart_tab)
        chart_layout.setContentsMargins(5, 5, 5, 5)
        
        missing_packages = []
        if not PLOTLY_AVAILABLE:
            missing_packages.append("plotly")
        if not MATPLOTLIB_AVAILABLE:
            missing_packages.append("matplotlib")
        if not SCIPY_AVAILABLE:
            missing_packages.append("scipy")
        
        if missing_packages:
            error_message = "証券市場線分析には以下のパッケージが必要です:\n"
            error_message += f"pip install {' '.join(missing_packages)}\n\nでインストールしてください\n\n"
            
            if not SCIPY_AVAILABLE:
                error_message += "※ scipyがない場合，簡易的な回帰分析を使用します"
            
            error_label = QLabel(error_message)
            error_label.setAlignment(Qt.AlignCenter)
            error_label.setStyleSheet(self.styles.get_empty_state_style().replace(
                self.styles.COLORS["text_secondary"], "#ff6b6b").replace(
                "dashed", "dashed").replace(
                self.styles.COLORS["border"], "#ff6b6b"))
            chart_layout.addWidget(error_label)
        
        if MATPLOTLIB_AVAILABLE:
            # Matplotlib canvas (メインチャートエリア)
            self.matplotlib_canvas = MatplotlibCanvas(parent=chart_tab, width=8, height=6)
            chart_layout.addWidget(self.matplotlib_canvas)

        self.tab_widget.addTab(chart_tab, "チャート")
        
        # タブ2: 統合指標テーブル（基本指標 + 回帰統計）
        table_tab = QWidget()
        table_layout = QVBoxLayout(table_tab)
        table_layout.setContentsMargins(5, 5, 5, 5)
        
        # 統合指標テーブル
        self.beta_table = self.create_table_widget()
        table_layout.addWidget(self.beta_table)
        
        self.tab_widget.addTab(table_tab, "指標・回帰統計")
        
        # タブ3: 分析結果サマリー
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
            "証券市場線チャートがここに表示されます．\n"
            "価格時系列データの準備後に分析を開始します．\n\n"
            "※ 回帰分析によるβ値計算を使用します"
        )
    
    def start_calculation(self, conditions: Dict[str, Any]):
        """計算開始"""
        if not self.price_data_source:
            return
        
        price_df = self.price_data_source.get_analysis_dataframe()
        if price_df is None or price_df.empty:
            QMessageBox.warning(self, "エラー", "価格データの取得に失敗しました．")
            return
        
        # 条件を保存し，市場ポートフォリオを追加
        self.current_conditions = conditions.copy()
        self.current_conditions['market_portfolio'] = self.market_combo.currentText()
        
        # 最小資産数チェック
        n_assets = len(price_df.columns)
        if n_assets < 1:
            QMessageBox.warning(
                self, "データエラー",
                "証券市場線分析には最低1つの資産が必要です．"
            )
            return
        
        # UI状態更新
        self.show_progress(True)
        self.market_info_label.setVisible(True)
        self.show_main_content(False)
        self.browser_button.setEnabled(False)
        self.export_chart_button.setEnabled(False)
        self.export_data_button.setEnabled(False)
        
        # 市場情報表示
        market_fetcher = MarketDataFetcher()
        market_description = market_fetcher.get_market_description(self.current_conditions['market_portfolio'])
        self.market_info_label.setText(f"市場: {market_description}")
        
        QApplication.processEvents()
        
        # 計算スレッド開始
        self.calculation_thread = SecurityMarketLineCalculationThread(price_df, self.current_conditions, self.config)
        self.calculation_thread.progress_updated.connect(self.update_progress)
        self.calculation_thread.calculation_completed.connect(self.on_calculation_completed)
        self.calculation_thread.calculation_error.connect(self.on_calculation_error)
        self.calculation_thread.start()
    
    def on_calculation_completed(self, results: Dict):
        """計算完了"""
        self.analysis_results = results
        
        # UI状態更新
        self.show_progress(False)
        self.hide_quality_info()
        self.show_main_content(True)

        # 市場情報ラベルを更新（分析資産数，市場，共通日数を表示）
        beta_results = results['beta_results']
        market_status = results['market_status']
        valid_data = {symbol: data for symbol, data in beta_results.items() 
                     if data.get('success', False)}
        
        total_assets = len(beta_results)
        analyzed_assets = len(valid_data)
        market_name = market_status['market_name']
        market_data_points = market_status.get('data_points', 0)
        
        # 共通日数を計算（有効な資産の最大共通日数）
        common_dates = 0
        if valid_data:
            common_dates = max(data.get('common_dates', 0) for data in valid_data.values())
        
        market_info_text = f"分析: {analyzed_assets}/{total_assets} 資産 | 市場: {market_name} | 共通日数: {common_dates} 日 | 回帰分析ベース"
        self.market_info_label.setText(market_info_text)
        
        # 結果表示
        self.update_display()
        self.create_interactive_chart()
        self.create_matplotlib_chart()
        self.update_beta_table()
        self.update_summary()
        
        # ボタン有効化
        self.browser_button.setEnabled(True)
        self.export_chart_button.setEnabled(True)
        self.export_data_button.setEnabled(True)
    
    def on_calculation_error(self, error_message: str):
        """計算エラー"""
        self.show_progress(False)
        self.hide_quality_info()
        self.market_info_label.setVisible(False)
        
        # エラー時は空状態を表示
        self.show_main_content(False)
        
        QMessageBox.critical(self, "エラー", error_message)
        self.update_display()
    
    def create_interactive_chart(self):
        """Plotlyでインタラクティブチャートを作成"""
        if not self.analysis_results or not PLOTLY_AVAILABLE:
            return
        
        try:
            results = self.analysis_results
            beta_results = results['beta_results']
            sml_line = results['sml_line']
            risk_free_rate = results['span_risk_free_rate']
            market_expected_return = results['market_expected_return']
            
            # 有効なβ値データのみを取得
            valid_data = {symbol: data for symbol, data in beta_results.items() 
                         if data.get('success', False)}
            
            if not valid_data:
                QMessageBox.warning(self, "エラー", "有効なβ値データがありません．")
                return
            
            # Plotlyグラフの作成
            fig = go.Figure()
            
            # 証券市場線
            if sml_line['beta_range'] and sml_line['sml_returns']:
                fig.add_trace(go.Scatter(
                    x=sml_line['beta_range'],
                    y=[r * 100 for r in sml_line['sml_returns']],  # パーセント変換
                    mode='lines',
                    name='Security Market Line',
                    line=dict(color='red', width=2),
                    hovertemplate='β: %{x:.2f}<br>Expected Return: %{y:.3f}%<extra></extra>'
                ))
            
            # 各資産のプロット
            symbols = list(valid_data.keys())
            betas = [valid_data[symbol]['beta'] for symbol in symbols]
            expected_returns = [valid_data[symbol]['expected_return'] * 100 for symbol in symbols]  # パーセント変換
            capm_returns = [valid_data[symbol]['capm_expected_return'] * 100 for symbol in symbols]
            alphas = [valid_data[symbol]['alpha'] * 100 for symbol in symbols]
            p_values = [valid_data[symbol].get('beta_p_value', np.nan) for symbol in symbols]
            r_squareds = [valid_data[symbol]['r_squared'] for symbol in symbols]
            
            # 資産の散布図
            colors = []
            for alpha in alphas:
                if pd.isna(alpha):
                    colors.append('gray')
                elif alpha >= 0:
                    colors.append('blue')
                else:
                    colors.append('orange')

            fig.add_trace(go.Scatter(
                x=betas,
                y=expected_returns,
                mode='markers+text',
                text=symbols,
                textposition="middle right",
                name='Assets',
                marker=dict(
                    size=8,
                    color=colors,
                    line=dict(width=2, color='white')
                ),
                textfont=dict(color='white', size=10),
                customdata=list(zip(capm_returns, alphas, p_values, r_squareds)),
                hovertemplate=('<b>%{text}</b><br>β: %{x:.3f}<br>'
                              'Expected Return: %{y:.3f}%<br>'
                              'CAPM Expected Return: %{customdata[0]:.3f}%<br>'
                              'α: %{customdata[1]:.3f}%<br>'
                              'β p-value: %{customdata[2]:.3f}<br>'
                              'R²: %{customdata[3]:.3f}<extra></extra>')
            ))

            # 市場ポートフォリオの点
            fig.add_trace(go.Scatter(
                x=[1.0],
                y=[market_expected_return * 100],
                mode='markers',
                name='Market Portfolio',
                marker=dict(size=10, color='green', symbol='circle'),
                hovertemplate='Market Portfolio<br>β: 1.0<br>Expected Return: %{y:.3f}%<extra></extra>'
            ))
            
            # Risk-free Rateを凡例として表示
            fig.add_trace(go.Scatter(
                x=[None],  # データポイントなし
                y=[None],
                mode='lines',
                name=f'Risk-free Rate ({risk_free_rate*100:.3f}%)',
                line=dict(color='pink', dash='dot'),
                showlegend=True
            ))
            
            # 実際のRisk-free Rate線（凡例には表示しない）
            max_beta = max(max(betas) if betas else 1, 1.5)
            min_beta = min(min(betas) if betas else -0.5, -0.5)
            
            fig.add_shape(
                type="line",
                x0=min_beta - 0.2, x1=max_beta + 0.2,
                y0=risk_free_rate * 100, y1=risk_free_rate * 100,
                line=dict(color="pink", width=1, dash="dot"),
            )
            
            # レイアウト設定（タイトル削除）
            fig.update_layout(
                xaxis=dict(
                    title='Beta (β)',
                    color='white',
                    gridcolor='rgba(255,255,255,0.3)',
                    showgrid=True,
                    range=[min_beta - 0.2, max_beta + 0.2]
                ),
                yaxis=dict(
                    title='Expected Return (%)',
                    color='white',
                    gridcolor='rgba(255,255,255,0.3)',
                    showgrid=True
                ),
                plot_bgcolor='rgba(50,50,50,1)',
                paper_bgcolor='rgba(43,43,43,1)',
                font=dict(color='white'),
                legend=dict(
                    bgcolor='rgba(0,0,0,0.7)',
                    bordercolor='white',
                    borderwidth=1
                ),
                autosize=True,
                margin=dict(l=50, r=80, t=30, b=50)
            )
            
            self.current_figure = fig
            
            # HTMLとして一時保存
            temp_file = tempfile.NamedTemporaryFile(
                mode='w', suffix='.html', delete=False, encoding='utf-8'
            )
            html_content = fig.to_html(
                include_plotlyjs='cdn',
                config={
                    'displayModeBar': True,
                    'displaylogo': False,
                    'responsive': True,
                    'toImageButtonOptions': {
                        'format': 'png',
                        'filename': 'security_market_line_regression',
                        'height': 700,
                        'width': 1000,
                        'scale': 2
                    }
                }
            )
            temp_file.write(html_content)
            temp_file.close()
            
            self.temp_html_files.append(temp_file.name)
            
        except Exception as e:
            self.logger.error(f"インタラクティブチャート作成エラー: {e}")
            QMessageBox.warning(self, "エラー", f"インタラクティブチャートの作成に失敗しました: {str(e)}")
    
    def create_matplotlib_chart(self):
        """Matplotlibで静的チャートを作成"""
        if not self.analysis_results or not MATPLOTLIB_AVAILABLE or not self.matplotlib_canvas:
            return
        
        try:
            results = self.analysis_results
            beta_results = results['beta_results']
            sml_line = results['sml_line']
            risk_free_rate = results['span_risk_free_rate']
            market_expected_return = results['market_expected_return']
            
            # 有効なデータのみを取得
            valid_data = {symbol: data for symbol, data in beta_results.items() 
                         if data.get('success', False)}
            
            if not valid_data:
                self.show_main_content(False)
                return
            
            # プロットのクリア
            self.matplotlib_canvas.figure.clear()
            ax = self.matplotlib_canvas.figure.add_subplot(111)
            
            # Dark theme
            ax.set_facecolor('#323232')
            
            # 証券市場線
            if sml_line['beta_range'] and sml_line['sml_returns']:
                sml_returns_pct = [r * 100 for r in sml_line['sml_returns']]
                ax.plot(sml_line['beta_range'], sml_returns_pct, 'r-', linewidth=2, 
                       label='Security Market Line', zorder=1)
            
            # 各資産のプロット
            symbols = list(valid_data.keys())
            betas = [valid_data[symbol]['beta'] for symbol in symbols]
            expected_returns_pct = [valid_data[symbol]['expected_return'] * 100 for symbol in symbols]
            alphas = [valid_data[symbol]['alpha'] * 100 for symbol in symbols]
            
            # α値に基づく色分け
            colors = []
            for alpha in alphas:
                if pd.isna(alpha):
                    colors.append('gray')
                elif alpha >= 0:
                    colors.append('blue')
                else:
                    colors.append('orange')
            
            for i, symbol in enumerate(symbols):
                ax.scatter(betas[i], expected_returns_pct[i], 
                          c=colors[i], s=30, alpha=0.8, edgecolors='white', zorder=2)
                ax.annotate(symbol, (betas[i], expected_returns_pct[i]), 
                           xytext=(5, 0), textcoords='offset points',
                           fontsize=8, color='white', zorder=3)
            
            # 市場ポートフォリオの点
            ax.scatter(1.0, market_expected_return * 100, 
                      c='green', s=40, marker='o', 
                      label='Market Portfolio', edgecolors='white', zorder=4)
            
            # 無リスク利子率の線
            max_beta = max(max(betas) if betas else 1, 1.5)
            min_beta = min(min(betas) if betas else -0.5, -0.5)
            ax.axhline(y=risk_free_rate * 100, color='pink', linewidth=1, 
                      linestyle=':', alpha=0.8,
                      label=f'Risk-free Rate ({risk_free_rate*100:.3f}%)', zorder=0)
            
            # ラベルと凡例（タイトル削除）
            ax.set_xlabel('Beta (β)', color='white')
            ax.set_ylabel('Expected Return (%)', color='white')
            ax.legend(facecolor='#2b2b2b', edgecolor='white', labelcolor='white', fontsize=8)
            ax.grid(True, alpha=0.3, zorder=0)
            ax.set_xlim(min_beta - 0.2, max_beta + 0.2)
            
            # 軸の色を設定
            ax.tick_params(colors='white', labelsize=8)
            ax.spines['bottom'].set_color('white')
            ax.spines['top'].set_color('white')
            ax.spines['right'].set_color('white')
            ax.spines['left'].set_color('white')
            
            # 描画更新
            self.matplotlib_canvas.draw()
            
        except Exception as e:
            self.logger.error(f"Matplotlibチャート作成エラー: {e}")
            # エラー時は空状態を表示
            self.show_main_content(False)
    
    def update_beta_table(self):
        """統合指標テーブルを更新（基本指標 + 回帰統計を統合）"""
        if not self.analysis_results:
            return
        
        try:
            beta_results = self.analysis_results['beta_results']
            valid_data = {symbol: data for symbol, data in beta_results.items() 
                         if data.get('success', False)}
            
            if not valid_data:
                return
            
            # 統合指標リスト（行ヘッダー）
            metrics = [
                # 基本指標
                ("β値", "beta"),
                ("β標準誤差", "beta_std_error"),
                ("βt統計量", "beta_t_statistic"),
                ("βp値", "beta_p_value"),
                ("β信頼区間下限", "beta_ci_lower"),
                ("β信頼区間上限", "beta_ci_upper"),
                ("α値", "alpha"),
                ("期待利益率", "expected_return"),
                ("均衡期待収益率", "capm_expected_return"),
                ("相関係数", "correlation"),
                ("R²", "r_squared"),
                ("システマティックリスク", "systematic_risk"),
                ("アンシステマティックリスク", "unsystematic_risk"),
                ("総リスク", "total_risk"),
                # 回帰統計
                ("残差標準誤差", "residual_std_error"),
                ("F統計量", "f_statistic"),
                ("F統計量p値", "f_p_value"),
                ("Durbin-Watson", "durbin_watson"),
                ("共通日数", "common_dates")
            ]
            
            # シンボルリスト（列ヘッダー）
            symbols = sorted(valid_data.keys())
            
            # テーブル設定
            self.beta_table.setRowCount(len(metrics))
            self.beta_table.setColumnCount(len(symbols))
            self.beta_table.setVerticalHeaderLabels([m[0] for m in metrics])
            self.beta_table.setHorizontalHeaderLabels(symbols)
            
            # データ挿入（行=指標，列=資産）
            for row, (metric_name, metric_key) in enumerate(metrics):
                for col, symbol in enumerate(symbols):
                    data = valid_data[symbol]
                    value = data.get(metric_key, np.nan)
                    
                    if pd.isna(value) or np.isinf(value):
                        item = QTableWidgetItem("-")
                        item.setForeground(Qt.gray)
                    else:
                        # フォーマット設定
                        if metric_key == "common_dates":
                            item = QTableWidgetItem(f"{int(value)}日")
                        elif metric_key in ["beta_p_value", "f_p_value"]:
                            if value < 0.001:
                                item = QTableWidgetItem("< 0.001")
                            else:
                                item = QTableWidgetItem(f"{value:.3f}")
                            
                            # p値の色分け
                            if value < 0.01:
                                item.setForeground(Qt.green)  # 高い有意性
                            elif value < 0.05:
                                item.setForeground(Qt.yellow)  # 有意
                            else:
                                item.setForeground(Qt.red)  # 有意でない
                        elif metric_key in ["beta", "correlation", "r_squared", "beta_std_error", 
                                          "beta_t_statistic", "beta_ci_lower", "beta_ci_upper",
                                          "residual_std_error", "f_statistic", "durbin_watson"]:
                            item = QTableWidgetItem(f"{value:.3f}")
                            
                            # 特別な色分け
                            if metric_key == "beta":
                                if value > 1:
                                    item.setForeground(Qt.red)
                                elif value < 1:
                                    item.setForeground(Qt.cyan)
                                else:
                                    item.setForeground(Qt.white)
                            elif metric_key == "durbin_watson":
                                if 1.5 <= value <= 2.5:
                                    item.setForeground(Qt.green)
                                else:
                                    item.setForeground(Qt.yellow)
                            else:
                                item.setForeground(Qt.white)
                        elif metric_key in ["alpha", "expected_return", "capm_expected_return", 
                                          "systematic_risk", "unsystematic_risk", "total_risk"]:
                            item = QTableWidgetItem(f"{value*100:.3f}%")
                            
                            # α値の色分け
                            if metric_key == "alpha":
                                if value > 0:
                                    item.setForeground(Qt.green)
                                elif value < 0:
                                    item.setForeground(Qt.red)
                                else:
                                    item.setForeground(Qt.white)
                            else:
                                item.setForeground(Qt.white)
                        else:
                            item = QTableWidgetItem(f"{value:.3f}")
                            item.setForeground(Qt.white)
                    
                    item.setTextAlignment(Qt.AlignCenter)
                    item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
                    
                    # ツールチップ設定
                    tooltip = f"{symbol} の {metric_name}: "
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
                    elif metric_key == "durbin_watson" and not pd.isna(value):
                        if 1.5 <= value <= 2.5:
                            tooltip += " (系列相関なし)"
                        else:
                            tooltip += " (系列相関の可能性)"
                    
                    item.setToolTip(tooltip)
                    self.beta_table.setItem(row, col, item)
            
            # カラム幅調整
            self.beta_table.resizeColumnsToContents()
            self.beta_table.resizeRowsToContents()
            
            # 最小/最大幅の設定
            for col in range(self.beta_table.columnCount()):
                current_width = self.beta_table.columnWidth(col)
                self.beta_table.setColumnWidth(col, min(max(current_width, 60), 120))
            
            # 行ヘッダーの幅を適切に設定
            header = self.beta_table.verticalHeader()
            header.setMinimumWidth(140)
            header.setMaximumWidth(180)
            
        except Exception as e:
            self.logger.error(f"統合指標テーブル更新エラー: {e}")
    
    def update_summary(self):
        """分析サマリーを更新"""
        if not self.analysis_results:
            self.summary_text.clear()
            return
        
        try:
            results = self.analysis_results
            beta_results = results['beta_results']
            market_status = results['market_status']
            conditions = results['conditions']
            
            # サマリーテキストの作成
            summary_lines = [
                "=== 証券市場線（SML）分析結果（回帰分析ベース） ===\n",
                f"分析日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"分析期間: {conditions.get('start_date')} ~ {conditions.get('end_date')}",
                f"スパン: {conditions.get('span', '日次')}",
                f"資産数: {int(results['n_assets'])} 資産",
                f"計算方法: 線形回帰分析（OLS）",
                "",
                "=== 分析条件 ===",
                f"市場ポートフォリオ: {market_status['market_name']}",
                f"無リスク利子率 (年率): {float(conditions.get('risk_free_rate', 0))*100:.3f}%",
                f"無リスク利子率 ({conditions.get('span', '日次')}): {float(results['span_risk_free_rate'])*100:.3f}%",
                "",
                "=== 市場データ ===",
                f"市場データ取得: {market_status.get('data_points', 0)}件",
                f"市場期待利益率: {float(results['market_expected_return'])*100:.3f}%",
                f"市場超過リターン: {float(results['market_excess_return'])*100:.3f}%",
            ]
            
            if market_status.get('date_range'):
                start_date, end_date = market_status['date_range']
                summary_lines.append(f"市場データ期間: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")
            
            # 回帰分析の説明を追加
            summary_lines.extend([
                "",
                "=== 回帰分析手法について ===",
                "・回帰式: 資産超過リターン = α + β × 市場超過リターン + ε",
                "・β値: 回帰係数として計算（市場に対する感応度）",
                "・α値: 回帰切片として計算（ジェンセン測度）",
                "・統計的有意性: t検定およびF検定により評価",
                "・信頼区間: 95%信頼区間を計算",
                "・残差診断: Durbin-Watson統計量で系列相関を検定",
                "",
                "=== 主要指標の説明 ===",
                "・β値(Beta): 市場ポートフォリオのリスクを1としたときの各資産のシステマティックリスク",
                "  - β=1: 市場と同じリスク量",
                "  - β>1: 市場より大きいリスク量（ハイリスク）",
                "  - β<1: 市場より小さいリスク量（ローリスク）",
                "",
                "・α値(Alpha，ジェンセン測度): CAPMによる均衡期待収益率からの超過リターン",
                "  - α>0: 市場を上回るパフォーマンス",
                "  - α<0: 市場を下回るパフォーマンス",
                "",
                "・統計的有意性:",
                "  - p値 < 0.01: 高い有意性 (***)",
                "  - p値 < 0.05: 有意 (**)",
                "  - p値 < 0.10: やや有意 (*)",
                "  - p値 ≥ 0.10: 有意でない",
                "",
                "・決定係数(R²): 市場要因で説明できる変動量の尺度",
                "・Durbin-Watson統計量: 系列相関の検定（1.5-2.5が理想的）",
                "・F統計量: 回帰式全体の有意性検定"
            ])

            # β値統計
            valid_data = {symbol: data for symbol, data in beta_results.items() 
                         if data.get('success', False)}
            
            if valid_data:
                betas = [data['beta'] for data in valid_data.values()]
                alphas = [data['alpha'] for data in valid_data.values()]
                p_values = [data.get('beta_p_value', np.nan) for data in valid_data.values()]
                r_squareds = [data['r_squared'] for data in valid_data.values()]
                std_errors = [data.get('beta_std_error', np.nan) for data in valid_data.values()]
                dw_stats = [data.get('durbin_watson', np.nan) for data in valid_data.values()]
                
                # 統計的有意性の集計
                significant_01 = sum(1 for p in p_values if not pd.isna(p) and p < 0.01)
                significant_05 = sum(1 for p in p_values if not pd.isna(p) and p < 0.05)
                significant_10 = sum(1 for p in p_values if not pd.isna(p) and p < 0.10)
                
                # 系列相関の評価
                good_dw = sum(1 for dw in dw_stats if not pd.isna(dw) and 1.5 <= dw <= 2.5)
                
                summary_lines.extend([
                    "",
                    "=== β値統計 ===",
                    f"有効資産数: {len(valid_data)}",
                    f"β値範囲: {np.min(betas):.3f} ~ {np.max(betas):.3f}",
                    f"β値平均: {np.mean(betas):.3f}",
                    f"β値中央値: {np.median(betas):.3f}",
                    f"β値標準偏差: {np.std(betas, ddof=1):.3f}",
                    "",
                    "=== α値統計 ===",
                    f"α値範囲: {np.min(alphas)*100:.3f}% ~ {np.max(alphas)*100:.3f}%",
                    f"α値平均: {np.mean(alphas)*100:.3f}%",
                    f"正のα値: {sum(1 for a in alphas if a > 0)} 資産",
                    f"負のα値: {sum(1 for a in alphas if a < 0)} 資産",
                    "",
                    "=== 回帰統計サマリー ===",
                    f"β統計的有意性 (1%水準): {significant_01} 資産",
                    f"β統計的有意性 (5%水準): {significant_05} 資産",
                    f"β統計的有意性 (10%水準): {significant_10} 資産",
                    f"有意でない: {len(valid_data) - significant_10} 資産",
                    f"平均R²: {np.mean(r_squareds):.3f}",
                    f"平均β標準誤差: {np.nanmean(std_errors):.4f}",
                    f"系列相関なし（DW 1.5-2.5）: {good_dw} 資産",
                    "",
                    "=== 各資産の詳細 ===",
                ])
                
                # 各資産の詳細
                for symbol, data in sorted(valid_data.items()):
                    p_val = data.get('beta_p_value', np.nan)
                    significance = ""
                    if not pd.isna(p_val):
                        if p_val < 0.01:
                            significance = " (***)"
                        elif p_val < 0.05:
                            significance = " (**)"
                        elif p_val < 0.1:
                            significance = " (*)"
                    
                    dw_val = data.get('durbin_watson', np.nan)
                    dw_status = ""
                    if not pd.isna(dw_val):
                        if 1.5 <= dw_val <= 2.5:
                            dw_status = " (OK)"
                        else:
                            dw_status = " (要注意)"
                    
                    summary_lines.extend([
                        f"\n{symbol}:",
                        f"  β値: {data['beta']:.3f} ± {data.get('beta_std_error', 0):.3f}{significance}",
                        f"  信頼区間: [{data.get('beta_ci_lower', np.nan):.3f}, {data.get('beta_ci_upper', np.nan):.3f}]",
                        f"  期待利益率: {data['expected_return']*100:.3f}%",
                        f"  α値: {data['alpha']*100:.3f}%",
                        f"  相関係数: {data['correlation']:.3f}",
                        f"  決定係数: {data['r_squared']:.3f}",
                        f"  β統計量p値: {p_val:.4f}" if not pd.isna(p_val) else "  β統計量p値: N/A",
                        f"  F統計量: {data.get('f_statistic', np.nan):.2f}",
                        f"  Durbin-Watson: {dw_val:.3f}{dw_status}" if not pd.isna(dw_val) else "  Durbin-Watson: N/A",
                        f"  共通データ数: {data['common_dates']}日"
                    ])
            
            self.summary_text.setPlainText("\n".join(summary_lines))
            
        except Exception as e:
            self.logger.error(f"サマリー更新エラー: {e}")
            self.summary_text.setPlainText(f"サマリー作成エラー: {str(e)}")
    
    def open_in_browser(self):
        """ブラウザで開く"""
        if not self.current_figure or not self.temp_html_files:
            QMessageBox.information(self, "Information", "表示するチャートがありません．")
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
    
    def export_chart(self):
        """チャートを保存"""
        if not self.current_figure:
            QMessageBox.information(self, "Information", "保存するチャートがありません．")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "証券市場線チャートを保存", "security_market_line_regression.html", "HTML files (*.html)"
        )
        
        if file_path:
            try:
                html_content = self.current_figure.to_html(
                    include_plotlyjs=True,
                    config={
                        'displayModeBar': True,
                        'displaylogo': False,
                        'responsive': True,
                        'toImageButtonOptions': {
                            'format': 'png',
                            'filename': 'security_market_line_regression',
                            'height': 800,
                            'width': 1200,
                            'scale': 2
                        }
                    }
                )
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                
                QMessageBox.information(self, "保存完了", f"チャートを保存しました:\n{file_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "エラー", f"保存中にエラーが発生しました:\n{str(e)}")
    
    def export_data(self):
        """分析データをCSV出力"""
        if not self.analysis_results:
            QMessageBox.information(self, "Information", "出力するデータがありません．")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "β値・SML分析データを保存", "security_market_line_regression_data.csv", "CSV files (*.csv)"
        )
        
        if file_path:
            try:
                results = self.analysis_results
                beta_results = results['beta_results']
                
                # 有効データのみを取得
                valid_data = {symbol: data for symbol, data in beta_results.items() 
                             if data.get('success', False)}
                
                if not valid_data:
                    QMessageBox.information(self, "Information", "出力する有効なデータがありません．")
                    return
                
                # スパンの英語変換
                span_english_map = {
                    '日次': 'Daily',
                    '週次': 'Weekly', 
                    '月次': 'Monthly',
                    '年次': 'Yearly'
                }
                
                # CSV出力
                with open(file_path, 'w', encoding='utf-8') as f:
                    # ヘッダー情報
                    market_status = results['market_status']
                    conditions = results['conditions']
                    span_english = span_english_map.get(conditions.get('span', 'Daily'), 'Daily')
                    
                    f.write('# Security Market Line Analysis Results (Regression Based)\n')
                    f.write(f'# Analysis Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
                    f.write(f'# Period: {conditions.get("start_date")} to {conditions.get("end_date")}\n')
                    f.write(f'# Span: {span_english}\n')
                    f.write(f'# Market Portfolio: {market_status["market_name"]}\n')
                    f.write(f'# Risk-free Rate: {float(conditions.get("risk_free_rate", 0))*100:.3f}% (Annual)\n')
                    f.write(f'# Market Expected Return: {float(results["market_expected_return"])*100:.3f}%\n')
                    f.write('# Method: Linear Regression (OLS)\n')
                    f.write('\n')
                    
                    # 指標名とデータキー
                    metrics = [
                        ("Beta", "beta"),
                        ("Beta_Std_Error", "beta_std_error"),
                        ("Beta_T_Statistic", "beta_t_statistic"),
                        ("Beta_P_Value", "beta_p_value"),
                        ("Beta_CI_Lower", "beta_ci_lower"),
                        ("Beta_CI_Upper", "beta_ci_upper"),
                        ("Alpha_%", "alpha", 100),
                        ("Expected_Return_%", "expected_return", 100),
                        ("CAPM_Expected_Return_%", "capm_expected_return", 100),
                        ("Correlation", "correlation"),
                        ("R_Squared", "r_squared"),
                        ("Systematic_Risk_%", "systematic_risk", 100),
                        ("Unsystematic_Risk_%", "unsystematic_risk", 100),
                        ("Total_Risk_%", "total_risk", 100),
                        ("Residual_Std_Error", "residual_std_error"),
                        ("F_Statistic", "f_statistic"),
                        ("F_P_Value", "f_p_value"),
                        ("Durbin_Watson", "durbin_watson"),
                        ("Common_Dates", "common_dates")
                    ]

                    symbols = sorted(valid_data.keys())

                    # 出力準備（1行目：資産名）
                    f.write('Metric,' + ','.join(symbols) + '\n')

                    # 各行：指標ごとに各資産の値を並べる
                    for metric in metrics:
                        if len(metric) == 3:
                            name, key, factor = metric
                        else:
                            name, key = metric
                            factor = 1.0
                        
                        row = [name]
                        for symbol in symbols:
                            value = valid_data[symbol].get(key, np.nan)
                            if pd.isna(value) or np.isinf(value):
                                row.append('N/A')
                            else:
                                row.append(f"{value * factor:.6f}")
                        f.write(','.join(row) + '\n')

                QMessageBox.information(self, "保存完了", f"分析データを保存しました:\n{file_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "エラー", f"保存中にエラーが発生しました:\n{str(e)}")
    
    def update_display(self):
        """表示更新"""
        if not self.analysis_results:
            self.show_main_content(False)
            return
        
        self.show_main_content(True)
    
    def clear_data(self):
        """データクリア"""
        self.analysis_results = None
        self.current_figure = None
        self.current_conditions = None
        self.cleanup_temp_files()
        
        self.summary_text.clear()
        
        # テーブルクリア
        self.beta_table.setRowCount(0)
        self.beta_table.setColumnCount(0)

        if MATPLOTLIB_AVAILABLE and self.matplotlib_canvas:
            self.matplotlib_canvas.clear_plot()
        
        # 空状態を表示
        self.show_main_content(False)

        self.browser_button.setEnabled(False)
        self.export_chart_button.setEnabled(False)
        self.export_data_button.setEnabled(False)
        self.show_progress(False)
        self.hide_quality_info()
        self.market_info_label.setVisible(False)
    
    def cleanup_temp_files(self):
        """一時ファイルのクリーンアップ"""
        for temp_file in self.temp_html_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception:
                pass
        self.temp_html_files.clear()
    
    def closeEvent(self, event):
        """ウィンドウクローズ時の処理"""
        self.cleanup_temp_files()
        super().closeEvent(event)
    
    def __del__(self):
        """デストラクタ"""
        self.cleanup_temp_files()