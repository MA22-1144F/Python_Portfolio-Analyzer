"""efficient_frontier.py"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Any
import logging
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QMessageBox,
    QPushButton, QApplication, QComboBox, QGroupBox, QFileDialog, QTextEdit
)
from PySide6.QtCore import QThread, Signal, Qt, QTimer
from data.asset_info import AssetInfo
from data.portfolio import Portfolio
from analysis.analysis_base_widget import AnalysisBaseWidget

try:
    import plotly.graph_objects as go
    from scipy.optimize import minimize
    PLOTLY_AVAILABLE = True
    SCIPY_AVAILABLE = True
except ImportError:
    go = None
    minimize = None
    PLOTLY_AVAILABLE = False
    SCIPY_AVAILABLE = False


try:
    from matplotlib.backends.backend_qtagg import FigureCanvas
    import matplotlib.figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    FigureCanvas = None
    matplotlib = None
    MATPLOTLIB_AVAILABLE = False

import tempfile
import os
from utils.common_widgets import BrowserLaunchThread

class EfficientFrontierCalculationThread(QThread):
    """最小分散フロンティア計算スレッド"""
    
    calculation_completed = Signal(dict)
    calculation_error = Signal(str)
    progress_updated = Signal(int, str)
    
    def __init__(self, price_df: pd.DataFrame, conditions: Dict[str, Any], config=None):
        super().__init__()
        self.price_df = price_df
        self.conditions = conditions
        if config is None:
            from config.app_config import AppConfig
            config = AppConfig()
        self.config = config
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
            min_weight = self.conditions.get('min_weight', 0.0)
            max_weight = self.conditions.get('max_weight', 1.0)
            steps = self.conditions.get('steps', 50)
            risk_free_rate = self.conditions.get('risk_free_rate', 0.0)
            start_date = self.conditions.get('start_date')
            end_date = self.conditions.get('end_date')
            span = self.conditions.get('span', '日次')
            
            # 無リスク利子率のスパン変換
            span_risk_free_rate = self._convert_risk_free_rate_to_span(risk_free_rate, span)
            
            self.progress_updated.emit(10, "ログリターンを計算中...")
            self.msleep(50)
            
            # ログリターンの計算
            log_returns = self._calculate_log_returns(self.price_df)
            if log_returns.empty:
                self.calculation_error.emit("ログリターンの計算に失敗しました．")
                return
            
            # 資産数チェック
            n_assets = log_returns.shape[1]
            if n_assets < 2:
                self.calculation_error.emit("最小分散フロンティア分析には最低2つの資産が必要です．")
                return
            
            # 投資割合制約の検証
            equal_weight = 1.0 / n_assets
            if min_weight > equal_weight:
                self.calculation_error.emit(
                    f"最小投資割合が大きすぎます．{n_assets}資産の場合，"
                    f"最小投資割合は ≤ {equal_weight:.4f} ({equal_weight*100:.2f}%) である必要があります．"
                )
                return
            
            if max_weight < equal_weight:
                self.calculation_error.emit(
                    f"最大投資割合が小さすぎます．{n_assets}資産の場合，"
                    f"最大投資割合は ≥ {equal_weight:.4f} ({equal_weight*100:.2f}%) である必要があります．"
                )
                return
            
            self.progress_updated.emit(30, "統計量を計算中...")
            self.msleep(50)
            
            # 統計量の計算
            mean_returns = log_returns.mean(axis=0).values
            cov_matrix = np.cov(log_returns.T.values)
            
            # 共分散行列の正則化
            cov_matrix += np.eye(len(cov_matrix)) * 1e-10
            
            self.progress_updated.emit(40, "期待利益率の範囲を計算中...")
            self.msleep(50)
            
            # 期待利益率範囲の計算
            sum_returns = np.sum(mean_returns)
            max_return_asset = np.max(mean_returns)
            min_return_asset = np.min(mean_returns)
            
            # 最大実効ウェイト
            max_effective_weight = min(max_weight, 1 - min_weight * (n_assets - 1))
            
            # ポートフォリオ期待利益率の範囲
            max_portfolio_return = (max_return_asset * max_effective_weight + 
                                  (sum_returns - max_return_asset) * min_weight)
            min_portfolio_return = (min_return_asset * max_effective_weight + 
                                  (sum_returns - min_return_asset) * min_weight)
            
            self.progress_updated.emit(50, "最小分散フロンティアを計算中...")
            
            # ターゲットリターンの設定
            epsilon = 1e-6
            target_returns = np.linspace(
                min_portfolio_return + epsilon, 
                max_portfolio_return - epsilon, 
                int(steps)
            )
            
            # 最小分散フロンティアの計算
            frontier_results = self._calculate_frontier_points(
                target_returns, mean_returns, cov_matrix, min_weight, max_weight, n_assets
            )
            
            if not frontier_results['volatilities'] or len(frontier_results['volatilities']) == 0:
                self.calculation_error.emit("最小分散フロンティアの計算に失敗しました．制約条件を確認してください．")
                return
            
            self.progress_updated.emit(80, "接点ポートフォリオを計算中...")
            self.msleep(50)
            
            # 最適リスク資産ポートフォリオ（シャープレシオ最大点）とリスクフリー資産の組み合わせ線の計算
            capital_allocation_line = self._calculate_capital_allocation_line(
                frontier_results, span_risk_free_rate
            )
            
            # 結果の構築
            results = {
                'log_returns': log_returns,
                'mean_returns': mean_returns,
                'cov_matrix': cov_matrix,
                'frontier_results': frontier_results,
                'capital_allocation_line': capital_allocation_line,
                'span_risk_free_rate': span_risk_free_rate,
                'conditions': self.conditions,
                'n_assets': n_assets,
                'max_effective_weight': max_effective_weight,
                'portfolio_return_range': (min_portfolio_return, max_portfolio_return)
            }
            
            self.progress_updated.emit(100, "計算完了")
            self.calculation_completed.emit(results)
            
        except Exception as e:
            self.logger.error(f"最小分散フロンティア計算エラー: {e}")
            self.calculation_error.emit(f"計算中にエラーが発生しました．: {str(e)}")
    
    def _convert_risk_free_rate_to_span(self, annual_rate: float, span: str) -> float:
        """年率の無リスク利子率をスパンに応じて変換"""
        try:
            span_factors_config = self.config.get('analysis.time_conversion_factors.span_factors', {})
            time_factors = self.config.get('analysis.time_conversion_factors', {})
            
            factor_key = span_factors_config.get(span)
            
            if factor_key == 1:
                factor = 1
            elif isinstance(factor_key, str):
                denominator = time_factors.get(factor_key, 365)
                factor = 1 / denominator
            else:
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
    
    def _calculate_log_returns(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """価格データからログリターンを計算"""
        try:
            if price_df.isnull().values.any():
                raise ValueError("価格データに欠損値が含まれています．")
            if (price_df <= 0).values.any():
                raise ValueError("価格データに非正の値が含まれています．")
            
            log_returns = np.log(price_df / price_df.shift(1)).dropna()
            return log_returns
            
        except Exception as e:
            self.logger.error(f"ログリターン計算エラー: {e}")
            return pd.DataFrame()
    
    def _calculate_portfolio_volatility(self, weights: np.ndarray, cov_matrix: np.ndarray) -> float:
        """ポートフォリオのボラティリティを計算"""
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    def _calculate_frontier_points(self, target_returns: np.ndarray, mean_returns: np.ndarray, 
                                 cov_matrix: np.ndarray, min_weight: float, max_weight: float, 
                                 n_assets: int) -> Dict:
        """最小分散フロンティアの各点を計算"""
        volatilities = []
        weights_list = []
        returns_list = []
        
        total_points = len(target_returns)
        
        for i, target in enumerate(target_returns):
            progress = 50 + int((i / total_points) * 25)
            self.progress_updated.emit(progress, f"最適化中... ({i+1}/{total_points})")
            
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                {'type': 'eq', 'fun': lambda w: np.dot(w, mean_returns) - target}
            ]
            
            bounds = tuple((min_weight, max_weight) for _ in range(n_assets))
            init_guess = np.array([1/n_assets] * n_assets)
            
            try:
                result = minimize(
                    self._calculate_portfolio_volatility,
                    init_guess,
                    args=(cov_matrix,),
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 1000, 'ftol': 1e-9}
                )
                
                if result.success:
                    volatilities.append(result.fun)
                    weights_list.append(result.x.copy())
                    returns_list.append(target)
                else:
                    self.logger.warning(f"最適化失敗: target={target:.6f}")
                    
            except Exception as e:
                self.logger.error(f"最適化エラー (target={target:.6f}): {e}")
                continue
            
            self.msleep(10)
        
        return {
            'volatilities': volatilities,
            'weights': weights_list,
            'returns': returns_list
        }
    
    def _calculate_capital_allocation_line(self, frontier_results: Dict, risk_free_rate: float) -> Dict:
        """最適リスク資産ポートフォリオとリスクフリー資産の組み合わせ線を計算"""
        try:
            if not frontier_results['volatilities'] or len(frontier_results['volatilities']) == 0:
                return {'cal_x': [], 'cal_y': [], 'optimal_risky_portfolio': None}
            
            volatilities = np.array(frontier_results['volatilities'])
            returns = np.array(frontier_results['returns'])
            
            excess_returns = returns - risk_free_rate
            
            non_zero_vol_mask = volatilities > 1e-10
            if not np.any(non_zero_vol_mask):
                return {'cal_x': [], 'cal_y': [], 'optimal_risky_portfolio': None}
            
            valid_volatilities = volatilities[non_zero_vol_mask]
            valid_returns = returns[non_zero_vol_mask]
            valid_excess_returns = excess_returns[non_zero_vol_mask]
            valid_weights = [frontier_results['weights'][i] for i, mask in enumerate(non_zero_vol_mask) if mask]
            
            sharpe_ratios = valid_excess_returns / valid_volatilities
            
            max_sharpe_idx = np.nanargmax(sharpe_ratios)
            optimal_vol = valid_volatilities[max_sharpe_idx]
            optimal_return = valid_returns[max_sharpe_idx]
            optimal_weights = valid_weights[max_sharpe_idx]
            
            max_vol = np.max(volatilities) * 1.2
            cal_x = np.linspace(0, max_vol, 100)
            slope = (optimal_return - risk_free_rate) / optimal_vol
            cal_y = risk_free_rate + slope * cal_x
            
            optimal_risky_portfolio = {
                'volatility': optimal_vol,
                'return': optimal_return,
                'weights': optimal_weights,
                'sharpe_ratio': sharpe_ratios[max_sharpe_idx]
            }
            
            return {
                'cal_x': cal_x.tolist(),
                'cal_y': cal_y.tolist(),
                'optimal_risky_portfolio': optimal_risky_portfolio
            }
            
        except Exception as e:
            self.logger.error(f"資本分配線計算エラー: {e}")
            return {'cal_x': [], 'cal_y': [], 'optimal_risky_portfolio': None}


class MatplotlibCanvas(FigureCanvas):
    """Matplotlib canvas for Qt integration"""
    
    def __init__(self, parent=None, width=8, height=6, dpi=100):
        self.figure = matplotlib.figure.Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.figure)
        self.setParent(parent)
        
        self.figure.patch.set_facecolor('#2b2b2b')
        
    def clear_plot(self):
        """プロットを完全にクリア"""
        self.figure.clear()
        self.draw()


class EfficientFrontierWidget(AnalysisBaseWidget):
    """最小分散フロンティア分析ウィジェット"""
    
    # ポートフォリオ保存シグナル
    portfolio_export_requested = Signal(Portfolio)
    
    def __init__(self, config=None):
        super().__init__(config)
        self.analysis_results = None
        self.current_figure = None
        self.temp_html_files = []
        self.browser_thread = None
        self.matplotlib_canvas = None
        self.current_conditions = None
        self.logger = logging.getLogger(__name__)
        
        min_height = self.config.get('analysis.widget_sizes.efficient_frontier.min_height', 600)
        min_width = self.config.get('analysis.widget_sizes.efficient_frontier.min_width', 800)
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
            
            self.current_conditions = conditions
            return
        
        self.current_conditions = conditions
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
        """ヘッダーコンテンツ設定"""
        self.header_layout.addStretch()

        self.browser_button = self.create_button("ブラウザで表示", "primary")
        self.browser_button.setEnabled(False)
        self.browser_button.clicked.connect(self.open_in_browser)
        self.header_layout.addWidget(self.browser_button)

        self.export_chart_button = self.create_button("HTML保存", "save")
        self.export_chart_button.setEnabled(False)
        self.export_chart_button.clicked.connect(self.export_chart)
        self.header_layout.addWidget(self.export_chart_button)
        
        self.export_data_button = self.create_button("CSV出力", "export")
        self.export_data_button.setEnabled(False)
        self.export_data_button.clicked.connect(self.export_data)
        self.header_layout.addWidget(self.export_data_button)
    
    def setup_content(self):
        """メインコンテンツエリアの設定"""
        self.constraint_info_label = QLabel("")
        self.constraint_info_label.setStyleSheet("color: #0078d4; font-size: 9px;")
        self.constraint_info_label.setVisible(False)
        self.constraint_info_label.setWordWrap(True)
        self.constraint_info_label.setMaximumHeight(30)
        self.content_layout.addWidget(self.constraint_info_label)
        
        self.tab_widget = self.create_tab_widget()
        
        # タブ1: チャート表示
        chart_tab = QWidget()
        chart_layout = QVBoxLayout(chart_tab)
        chart_layout.setContentsMargins(5, 5, 5, 5)
        
        if not PLOTLY_AVAILABLE or not SCIPY_AVAILABLE or not MATPLOTLIB_AVAILABLE:
            error_message = "最小分散フロンティア分析には以下のパッケージが必要です:\n"
            missing_packages = []
            if not PLOTLY_AVAILABLE:
                missing_packages.append("plotly")
            if not SCIPY_AVAILABLE:
                missing_packages.append("scipy")
            if not MATPLOTLIB_AVAILABLE:
                missing_packages.append("matplotlib")
            
            error_message += f"pip install {' '.join(missing_packages)}\n\nでインストールしてください"
            
            error_label = QLabel(error_message)
            error_label.setAlignment(Qt.AlignCenter)
            error_label.setStyleSheet(self.styles.get_empty_state_style().replace(
                self.styles.COLORS["text_secondary"], "#ff6b6b").replace(
                "dashed", "dashed").replace(
                self.styles.COLORS["border"], "#ff6b6b"))
            chart_layout.addWidget(error_label)
        else:
            self.matplotlib_canvas = MatplotlibCanvas(parent=chart_tab, width=8, height=6)
            chart_layout.addWidget(self.matplotlib_canvas)

        self.tab_widget.addTab(chart_tab, "チャート")
        
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
        
        # ポートフォリオ保存セクション
        portfolio_export_group = self._create_portfolio_export_section()
        summary_layout.addWidget(portfolio_export_group)
        
        self.tab_widget.addTab(summary_tab, "分析サマリー")
        
        self.content_layout.addWidget(self.tab_widget)
    
    def _create_portfolio_export_section(self):
        """ポートフォリオ保存セクションを作成"""
        group = QGroupBox("ポートフォリオとして保存")
        layout = QVBoxLayout(group)
        
        # 説明ラベル
        info_label = QLabel("最適化結果をポートフォリオとして保存し，管理タブで編集できます．")
        info_label.setStyleSheet("color: #888; font-size: 9px;")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        # 選択とボタンのレイアウト
        control_layout = QHBoxLayout()
        
        # ポートフォリオタイプ選択
        control_layout.addWidget(QLabel("タイプ:"))
        self.portfolio_type_combo = QComboBox()
        self.portfolio_type_combo.addItems([
            "最小分散ポートフォリオ",
            "接点ポートフォリオ（CAL）"
        ])
        self.portfolio_type_combo.setMinimumWidth(180)
        control_layout.addWidget(self.portfolio_type_combo)
        
        control_layout.addStretch()
        
        # 保存ボタン
        self.save_portfolio_button = QPushButton("ポートフォリオとして保存")
        self.save_portfolio_button.setStyleSheet(self.styles.get_button_style_by_type("primary"))
        self.save_portfolio_button.setEnabled(False)
        current_style = self.save_portfolio_button.styleSheet()
        self.save_portfolio_button.setStyleSheet(
            current_style + " QPushButton { padding: 8px 16px; font-weight: bold; }"
        )
        self.save_portfolio_button.clicked.connect(self._on_save_portfolio_clicked)
        control_layout.addWidget(self.save_portfolio_button)
        
        layout.addLayout(control_layout)
        
        return group
    
    def _on_save_portfolio_clicked(self):
        """ポートフォリオ保存ボタンクリック時の処理"""
        if not self.analysis_results:
            QMessageBox.warning(self, "エラー", "保存する分析結果がありません．")
            return
        
        portfolio_type = self.portfolio_type_combo.currentText()
        
        try:
            portfolio = self._create_portfolio_from_results(portfolio_type)
            
            if portfolio:
                # シグナルを発行してMainWindowに通知
                self.portfolio_export_requested.emit(portfolio)
                
                # 成功メッセージ
                QMessageBox.information(
                    self, 
                    "保存準備完了", 
                    f"「{portfolio.name}」を作成しました．\n"
                    f"ポートフォリオ管理タブで確認・編集できます．"
                )
            
        except Exception as e:
            self.logger.error(f"ポートフォリオ作成エラー: {e}")
            QMessageBox.critical(
                self, 
                "エラー", 
                f"ポートフォリオの作成に失敗しました:\n{str(e)}"
            )
    
    def _create_portfolio_from_results(self, portfolio_type: str) -> Optional[Portfolio]:
        """分析結果からポートフォリオオブジェクトを作成"""
        if not self.analysis_results:
            return None
        
        results = self.analysis_results
        frontier = results['frontier_results']
        cal = results['capital_allocation_line']
        conditions = results['conditions']
        symbols = list(results['log_returns'].columns)
        
        # ポートフォリオタイプに応じてウエイトと統計値を取得
        if portfolio_type == "最小分散ポートフォリオ":
            if not frontier['volatilities']:
                raise ValueError("最小分散ポートフォリオのデータがありません．")
            
            volatilities_array = np.array(frontier['volatilities'])
            returns_array = np.array(frontier['returns'])
            min_vol_idx = np.argmin(volatilities_array)
            
            weights = frontier['weights'][min_vol_idx]
            stats = {
                'volatility': float(volatilities_array[min_vol_idx]),
                'return': float(returns_array[min_vol_idx]),
                'sharpe_ratio': None
            }
            
        elif portfolio_type == "接点ポートフォリオ（CAL）":
            if not cal['optimal_risky_portfolio']:
                raise ValueError("接点ポートフォリオのデータがありません．")
            
            orp = cal['optimal_risky_portfolio']
            weights = orp['weights']
            stats = {
                'volatility': float(orp['volatility']),
                'return': float(orp['return']),
                'sharpe_ratio': float(orp['sharpe_ratio'])
            }
        else:
            raise ValueError(f"未知のポートフォリオタイプ: {portfolio_type}")
        
        # ポートフォリオ名と説明を生成
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
        
        if portfolio_type == "最小分散ポートフォリオ":
            portfolio_name = f"最小分散P_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            description = (
                f"最小分散フロンティア分析から作成\n"
                f"作成日時: {timestamp}\n"
                f"分析期間: {conditions.get('start_date')} ~ {conditions.get('end_date')}\n"
                f"スパン: {conditions.get('span', '日次')}\n"
                f"標準偏差: {stats['volatility']:.6f}\n"
                f"期待利益率: {stats['return']:.6f}"
            )
        else:  # 接点ポートフォリオ
            portfolio_name = f"接点P(Sharpe{stats['sharpe_ratio']:.2f})_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            description = (
                f"最小分散フロンティア分析から作成（CAL接点）\n"
                f"作成日時: {timestamp}\n"
                f"分析期間: {conditions.get('start_date')} ~ {conditions.get('end_date')}\n"
                f"スパン: {conditions.get('span', '日次')}\n"
                f"シャープレシオ: {stats['sharpe_ratio']:.4f}\n"
                f"標準偏差: {stats['volatility']:.6f}\n"
                f"期待利益率: {stats['return']:.6f}"
            )
        
        # Portfolioオブジェクトを作成
        portfolio = Portfolio(name=portfolio_name, description=description)
        
        # 資産情報を取得（price_data_sourceから）
        if self.price_data_source and hasattr(self.price_data_source, 'price_data'):
            price_data = self.price_data_source.price_data
            
            for symbol, weight in zip(symbols, weights):
                # AssetInfo を取得
                asset_data = price_data.get(symbol)
                if asset_data:
                    # price_dataからAssetInfoを再構築する必要がある
                    # 簡易的な方法として，symbolから新しいAssetInfoを作成
                    asset = AssetInfo(
                        symbol=symbol,
                        name=asset_data.get('name', symbol),
                        currency=asset_data.get('currency', 'USD')
                    )
                    portfolio.add_position(asset, float(weight))
        else:
            # フォールバック：symbolのみからAssetInfoを作成
            for symbol, weight in zip(symbols, weights):
                asset = AssetInfo(symbol=symbol, name=symbol)
                portfolio.add_position(asset, float(weight))
        
        # 日時を設定
        portfolio.created_at = datetime.now()
        portfolio.modified_at = datetime.now()
        
        return portfolio
    
    def get_empty_message(self) -> str:
        """空状態メッセージ"""
        return (
            "最小分散フロンティアチャートがここに表示されます．\n"
            "価格時系列データの準備後に分析を開始します．"
        )
    
    def start_calculation(self, conditions: Dict[str, Any]):
        """計算開始"""
        if not self.price_data_source:
            return
        
        price_df = self.price_data_source.get_analysis_dataframe()
        if price_df is None or price_df.empty:
            QMessageBox.warning(self, "エラー", "価格データの取得に失敗しました．")
            return
        
        self.current_conditions = conditions

        n_assets = len(price_df.columns)
        min_weight = conditions.get('min_weight', 0.0)
        max_weight = conditions.get('max_weight', 1.0)
        
        equal_weight_pct = 100.0 / n_assets
        min_weight_pct = min_weight * 100
        max_weight_pct = max_weight * 100
        
        if min_weight_pct > equal_weight_pct:
            QMessageBox.warning(
                self, "制約エラー",
                f"最小分散フロンティア分析の制約条件:\n\n"
                f"最小投資割合: 0.00% ≤ {min_weight_pct:.2f}% ≤ {equal_weight_pct:.2f}%\n\n"
                f"現在の最小投資割合設定が大きすぎます．\n"
                f"分析条件を調整してください．"
            )
            return
        
        if max_weight_pct < equal_weight_pct:
            QMessageBox.warning(
                self, "制約エラー",
                f"最小分散フロンティア分析の制約条件:\n\n"
                f"最大投資割合: {equal_weight_pct:.2f}% ≤ {max_weight_pct:.2f}% ≤ 100.00%\n\n"
                f"現在の最大投資割合設定が小さすぎます．\n"
                f"分析条件を調整してください．"
            )
            return
        
        self.show_progress(True)
        self.constraint_info_label.setVisible(True)
        self.show_main_content(False)
        self.browser_button.setEnabled(False)
        self.export_chart_button.setEnabled(False)
        self.export_data_button.setEnabled(False)
        self.save_portfolio_button.setEnabled(False)
        
        constraint_text = (f"分析: {n_assets} 資産 | "
                          f"制約: {min_weight_pct:.2f}% ≤ 投資割合 ≤ {max_weight_pct:.2f}% | "
                          f"段階数: {conditions.get('steps', 50)}")
        self.constraint_info_label.setText(constraint_text)
        
        QApplication.processEvents()
        
        self.calculation_thread = EfficientFrontierCalculationThread(price_df, conditions, self.config)
        self.calculation_thread.progress_updated.connect(self.update_progress)
        self.calculation_thread.calculation_completed.connect(self.on_calculation_completed)
        self.calculation_thread.calculation_error.connect(self.on_calculation_error)
        self.calculation_thread.start()
    
    def on_calculation_completed(self, results: Dict):
        """計算完了"""
        self.analysis_results = results
        
        self.show_progress(False)
        self.hide_quality_info()
        self.show_main_content(True)
        
        self.update_display()
        self.create_interactive_chart()
        self.create_matplotlib_chart()
        self.update_summary()
        
        self.browser_button.setEnabled(True)
        self.export_chart_button.setEnabled(True)
        self.export_data_button.setEnabled(True)
        self.save_portfolio_button.setEnabled(True)
    
    def on_calculation_error(self, error_message: str):
        """計算エラー"""
        self.show_progress(False)
        self.hide_quality_info()
        self.constraint_info_label.setVisible(False)
        
        self.show_main_content(False)
        
        QMessageBox.critical(self, "エラー", error_message)
        self.update_display()
    
    def create_interactive_chart(self):
        """Plotlyでインタラクティブチャートを作成"""
        if not self.analysis_results or not PLOTLY_AVAILABLE:
            return
        
        try:
            results = self.analysis_results
            frontier = results['frontier_results']
            cal = results['capital_allocation_line']
            
            if not frontier['volatilities'] or len(frontier['volatilities']) == 0:
                QMessageBox.warning(self, "エラー", "フロンティアデータが空です．計算を再実行してください．")
                return
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=frontier['volatilities'],
                y=frontier['returns'],
                mode='lines+markers',
                name='Minimum Variance Frontier',
                line=dict(color='maroon', width=1),
                marker=dict(size=2, color='maroon')
            ))
            
            if len(frontier['volatilities']) > 0:
                volatilities_array = np.array(frontier['volatilities'])
                returns_array = np.array(frontier['returns'])
                
                min_vol_idx = np.argmin(volatilities_array)
                efficient_vol = volatilities_array[min_vol_idx:]
                efficient_returns = returns_array[min_vol_idx:]
                
                fig.add_trace(go.Scatter(
                    x=efficient_vol,
                    y=efficient_returns,
                    mode='lines',
                    name='Efficient Frontier',
                    line=dict(color='red', width=2)
                ))
                
                fig.add_trace(go.Scatter(
                    x=[volatilities_array[min_vol_idx]],
                    y=[returns_array[min_vol_idx]],
                    mode='markers',
                    name='Global Minimum Variance Portfolio',
                    marker=dict(size=6, color='fuchsia', symbol='circle')
                ))
            
            if len(cal.get('cal_x', [])) > 0 and len(cal.get('cal_y', [])) > 0:
                fig.add_trace(go.Scatter(
                    x=cal['cal_x'],
                    y=cal['cal_y'],
                    mode='lines',
                    name='Capital Allocation Line',
                    line=dict(color='blue', width=2, dash='longdash')
                ))
                
                if cal['optimal_risky_portfolio'] is not None:
                    orp = cal['optimal_risky_portfolio']
                    fig.add_trace(go.Scatter(
                        x=[orp['volatility']],
                        y=[orp['return']],
                        mode='markers',
                        name=f'Tangency Portfolio (Sharpe: {orp["sharpe_ratio"]:.3f})',
                        marker=dict(size=6, color='aqua', symbol='circle')
                    ))
            
            risk_free_rate = results['span_risk_free_rate']
            max_x = np.max(frontier['volatilities']) * 1.1 if len(frontier['volatilities']) > 0 else 0.1
            fig.add_trace(go.Scatter(
                x=[0, max_x],
                y=[risk_free_rate, risk_free_rate],
                mode='lines',
                name=f'Risk-free Rate ({risk_free_rate:.3%})',
                line=dict(color='pink', width=1, dash='dot')
            ))
            
            fig.update_layout(
                xaxis=dict(
                    title='Standard Deviation (Risk)',
                    color='white',
                    gridcolor='rgba(255,255,255,0.3)',
                    showgrid=False
                ),
                yaxis=dict(
                    title='Expected Return',
                    color='white',
                    gridcolor='rgba(255,255,255,0.3)',
                    showgrid=False
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
                margin=dict(l=50, r=50, t=80, b=50)
            )
            
            self.current_figure = fig
            
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
                        'filename': 'efficient_frontier',
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
            frontier = results['frontier_results']
            cal = results['capital_allocation_line']
            
            if not frontier['volatilities'] or len(frontier['volatilities']) == 0:
                self.show_main_content(False)
                return
            
            self.matplotlib_canvas.figure.clear()
            ax = self.matplotlib_canvas.figure.add_subplot(111)
            
            ax.set_facecolor('#323232')
            
            volatilities = np.array(frontier['volatilities'])
            returns = np.array(frontier['returns'])
            
            ax.plot(volatilities, returns, 'c-', color='maroon', linewidth=1, label='Minimum Variance Frontier', marker='o', markersize=2)
            
            min_vol_idx = np.argmin(volatilities)
            efficient_vol = volatilities[min_vol_idx:]
            efficient_returns = returns[min_vol_idx:]
            ax.plot(efficient_vol, efficient_returns, 'y-', color='red', linewidth=2, label='Efficient Frontier')
            
            ax.plot(volatilities[min_vol_idx], returns[min_vol_idx], 'o', color='fuchsia', markersize=6, label='Global Minimum Variance Portfolio')
            
            if len(cal.get('cal_x', [])) > 0 and len(cal.get('cal_y', [])) > 0:
                ax.plot(cal['cal_x'], cal['cal_y'], 'blue', linewidth=2, linestyle='--', label='Capital Allocation Line')
                
                if cal['optimal_risky_portfolio'] is not None:
                    orp = cal['optimal_risky_portfolio']
                    ax.plot(orp['volatility'], orp['return'], 'o', color='aqua', markersize=6, 
                           label=f'Tangency Portfolio (Sharpe: {orp["sharpe_ratio"]:.3f})')
            
            risk_free_rate = results['span_risk_free_rate']
            max_x = np.max(volatilities) * 1.1
            ax.axhline(y=risk_free_rate, color='pink', linewidth=1, linestyle=':', 
                      label=f'Risk-free Rate ({risk_free_rate:.3%})')
            
            ax.set_xlabel('Standard Deviation (Risk)', color='white')
            ax.set_ylabel('Expected Return', color='white')
            ax.legend(facecolor='#2b2b2b', edgecolor='white', labelcolor='white', fontsize=8)
            ax.grid(True, alpha=0.3)
            
            ax.tick_params(colors='white', labelsize=8)
            ax.spines['bottom'].set_color('white')
            ax.spines['top'].set_color('white')
            ax.spines['right'].set_color('white')
            ax.spines['left'].set_color('white')
            
            self.matplotlib_canvas.draw()
            
        except Exception as e:
            self.logger.error(f"Matplotlibチャート作成エラー: {e}")
            self.show_main_content(False)
    
    def update_summary(self):
        """分析サマリーを更新"""
        if not self.analysis_results:
            self.summary_text.clear()
            return
        
        try:
            results = self.analysis_results
            frontier = results['frontier_results']
            cal = results['capital_allocation_line']
            conditions = results['conditions']
            
            summary_lines = [
                "=== 最小分散フロンティア分析結果 ===\n",
                f"分析日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"分析期間: {conditions.get('start_date')} ~ {conditions.get('end_date')}",
                f"スパン: {conditions.get('span', '日次')}",
                f"資産数: {int(results['n_assets'])} 資産",
                "",
                "=== 分析条件 ===",
                f"最小投資割合: {float(conditions.get('min_weight', 0))*100:.2f}%",
                f"最大投資割合: {float(conditions.get('max_weight', 1))*100:.2f}%",
                f"期待利益率段階数: {int(conditions.get('steps', 50))}",
                f"無リスク利子率 (年率): {float(conditions.get('risk_free_rate', 0))*100:.3f}%",
                f"無リスク利子率 ({conditions.get('span', '日次')}): {float(results['span_risk_free_rate'])*100:.3f}%",
                "",
                "=== フロンティア統計 ===",
                f"分析ポイント数: {len(frontier['volatilities'])} / {int(conditions.get('steps', 50))}",
            ]
            
            if len(frontier['volatilities']) > 0:
                volatilities_array = np.array(frontier['volatilities'])
                returns_array = np.array(frontier['returns'])
                
                min_vol = np.min(volatilities_array)
                max_vol = np.max(volatilities_array)
                min_ret = np.min(returns_array)
                max_ret = np.max(returns_array)
                
                summary_lines.extend([
                    f"標準偏差範囲: {float(min_vol):.4f} ~ {float(max_vol):.4f}",
                    f"期待利益率範囲: {float(min_ret):.4f} ~ {float(max_ret):.4f}",
                    "",
                    "=== 最小分散ポートフォリオ ===",
                ])
                
                min_vol_idx = np.argmin(volatilities_array)
                mvp_weights = frontier['weights'][min_vol_idx]
                mvp_symbols = list(results['log_returns'].columns)
                
                summary_lines.append(f"標準偏差: {float(min_vol):.4f}")
                summary_lines.append(f"期待利益率: {float(returns_array[min_vol_idx]):.4f}")
                summary_lines.append("投資割合:")
                for i, (symbol, weight) in enumerate(zip(mvp_symbols, mvp_weights)):
                    summary_lines.append(f"  {symbol}: {float(weight)*100:.2f}%")
            
            if cal['optimal_risky_portfolio'] is not None:
                orp = cal['optimal_risky_portfolio']
                summary_lines.extend([
                    "",
                    "=== CAL接点ポートフォリオ ===",
                    f"標準偏差: {float(orp['volatility']):.4f}",
                    f"期待利益率: {float(orp['return']):.4f}",
                    f"シャープレシオ: {float(orp['sharpe_ratio']):.4f}",
                    "投資割合:",
                ])
                
                mvp_symbols = list(results['log_returns'].columns)
                for symbol, weight in zip(mvp_symbols, orp['weights']):
                    summary_lines.append(f"  {symbol}: {float(weight)*100:.2f}%")
            
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
            self, "最小分散フロンティアチャートを保存", "efficient_frontier.html", "HTML files (*.html)"
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
                            'filename': 'efficient_frontier',
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
            self, "分析データを保存", "efficient_frontier_data.csv", "CSV files (*.csv)"
        )
        
        if file_path:
            try:
                results = self.analysis_results
                frontier = results['frontier_results']
                
                symbols = list(results['log_returns'].columns)
                data_rows = []
                
                for i, (vol, ret, weights) in enumerate(zip(
                    frontier['volatilities'], 
                    frontier['returns'], 
                    frontier['weights']
                )):
                    row = {
                        'Point': i + 1,
                        'Volatility': vol,
                        'Return': ret,
                        'Sharpe_Ratio': (ret - results['span_risk_free_rate']) / vol if vol > 0 else 0
                    }
                    
                    for j, symbol in enumerate(symbols):
                        row[f'{symbol}'] = weights[j]
                    
                    data_rows.append(row)
                
                df = pd.DataFrame(data_rows)
                df.to_csv(file_path, index=False, encoding='utf-8-sig')
                
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

        if MATPLOTLIB_AVAILABLE and self.matplotlib_canvas:
            self.matplotlib_canvas.clear_plot()
        
        self.show_main_content(False)

        self.browser_button.setEnabled(False)
        self.export_chart_button.setEnabled(False)
        self.export_data_button.setEnabled(False)
        self.save_portfolio_button.setEnabled(False)
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
    
    def closeEvent(self, event):
        """ウィンドウクローズ時の処理"""
        self.cleanup_temp_files()
        super().closeEvent(event)
    
    def __del__(self):
        """デストラクタ"""
        self.cleanup_temp_files()