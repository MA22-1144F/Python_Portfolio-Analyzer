"""downside_deviation_frontier_comparison.py"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import matplotlib.figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class PortfolioDownsideDeviationFrontierCalculator:
    """ポートフォリオの下方偏差フロンティアを計算するクラス"""

    def __init__(self, config=None):
        self.logger = logging.getLogger(__name__)
        if config is None:
            from config.app_config import AppConfig
            config = AppConfig()
        self.config = config

    def calculate_frontier(self, price_df: pd.DataFrame, portfolio_name: str,
                          risk_free_rate: float = 0.0, steps: int = 50) -> Optional[Dict]:
        """下方偏差フロンティアを計算"""
        try:
            if not SCIPY_AVAILABLE:
                self.logger.error("scipyが利用できません")
                return None

            # ログリターンを計算
            log_returns = np.log(price_df / price_df.shift(1)).dropna()

            if log_returns.empty or len(log_returns.columns) < 2:
                self.logger.warning(f"{portfolio_name}: フロンティア計算に十分なデータがありません")
                return None

            # 統計量の計算
            mean_returns = log_returns.mean(axis=0).values

            # 下方偏差共分散行列の計算
            downside_cov_matrix = self._calculate_downside_covariance_matrix(
                log_returns.values, risk_free_rate
            )

            # 共分散行列の正則化
            downside_cov_matrix += np.eye(len(downside_cov_matrix)) * 1e-10

            n_assets = len(mean_returns)

            # 期待リターンの範囲を計算
            max_return = np.max(mean_returns)
            min_return = np.min(mean_returns)

            # ターゲットリターンの設定
            epsilon = 1e-6
            target_returns = np.linspace(min_return + epsilon, max_return - epsilon, steps)

            # フロンティア点を計算
            frontier_points = self._calculate_frontier_points(
                target_returns, mean_returns, downside_cov_matrix, n_assets
            )

            if not frontier_points['downside_deviations']:
                return None

            # 最小下方偏差点を見つける
            downside_deviations_array = np.array(frontier_points['downside_deviations'])
            returns_array = np.array(frontier_points['returns'])
            min_dd_idx = np.argmin(downside_deviations_array)

            # 資本配分線（CAL）を計算（ソルティノレシオ使用）
            cal_data = self._calculate_capital_allocation_line(
                frontier_points, risk_free_rate
            )

            result = {
                'portfolio_name': portfolio_name,
                'symbols': list(price_df.columns),
                'downside_deviations': frontier_points['downside_deviations'],
                'returns': frontier_points['returns'],
                'weights': frontier_points['weights'],
                'min_dd_idx': min_dd_idx,
                'min_dd_point': {
                    'downside_deviation': downside_deviations_array[min_dd_idx],
                    'return': returns_array[min_dd_idx],
                    'weights': frontier_points['weights'][min_dd_idx]
                },
                'cal': cal_data,
                'risk_free_rate': risk_free_rate,
                'mean_returns': mean_returns,
                'downside_cov_matrix': downside_cov_matrix
            }

            return result

        except Exception as e:
            self.logger.error(f"{portfolio_name}: 下方偏差フロンティア計算エラー: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def _calculate_downside_covariance_matrix(self, returns_matrix: np.ndarray,
                                             target_return: float) -> np.ndarray:
        """下方偏差共分散行列を計算"""
        n_periods, n_assets = returns_matrix.shape
        downside_cov_matrix = np.zeros((n_assets, n_assets))

        for i in range(n_assets):
            for j in range(n_assets):
                # 各資産のdownsideリターン（target以下の部分のみ）を計算
                downside_returns_i = np.minimum(returns_matrix[:, i] - target_return, 0)
                downside_returns_j = np.minimum(returns_matrix[:, j] - target_return, 0)

                # 共分散を計算
                downside_cov_matrix[i, j] = np.mean(downside_returns_i * downside_returns_j)

        return downside_cov_matrix

    def _calculate_portfolio_downside_deviation(self, weights: np.ndarray,
                                              downside_cov_matrix: np.ndarray) -> float:
        """ポートフォリオの下方偏差を計算"""
        return np.sqrt(np.dot(weights.T, np.dot(downside_cov_matrix, weights)))

    def _calculate_frontier_points(self, target_returns: np.ndarray,
                                   mean_returns: np.ndarray,
                                   downside_cov_matrix: np.ndarray,
                                   n_assets: int) -> Dict:
        """フロンティアの各点を計算"""
        downside_deviations = []
        returns_list = []
        weights_list = []

        for target in target_returns:
            # 制約条件
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                {'type': 'eq', 'fun': lambda w: np.dot(w, mean_returns) - target}
            ]

            # 境界条件（0 <= weight <= 1）
            bounds = tuple((0.0, 1.0) for _ in range(n_assets))

            # 初期値（均等配分）
            init_guess = np.array([1/n_assets] * n_assets)

            try:
                # 最適化実行
                result = minimize(
                    self._calculate_portfolio_downside_deviation,
                    init_guess,
                    args=(downside_cov_matrix,),
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 1000, 'ftol': 1e-9}
                )

                if result.success:
                    downside_deviations.append(result.fun)
                    returns_list.append(target)
                    weights_list.append(result.x.copy())

            except Exception as e:
                self.logger.debug(f"最適化失敗 (target={target:.6f}): {e}")
                continue

        return {
            'downside_deviations': downside_deviations,
            'returns': returns_list,
            'weights': weights_list
        }

    def _calculate_capital_allocation_line(self, frontier_points: Dict,
                                          risk_free_rate: float) -> Dict:
        """資本配分線（CAL）を計算（ソルティノレシオ使用）"""
        try:
            if not frontier_points['downside_deviations']:
                return {'cal_x': [], 'cal_y': [], 'tangency_portfolio': None}

            downside_deviations = np.array(frontier_points['downside_deviations'])
            returns = np.array(frontier_points['returns'])

            # 超過リターンを計算
            excess_returns = returns - risk_free_rate

            # ゼロ除算を避ける
            non_zero_mask = downside_deviations > 1e-10
            if not np.any(non_zero_mask):
                return {'cal_x': [], 'cal_y': [], 'tangency_portfolio': None}

            valid_downside_deviations = downside_deviations[non_zero_mask]
            valid_returns = returns[non_zero_mask]
            valid_excess_returns = excess_returns[non_zero_mask]
            valid_weights = [frontier_points['weights'][i] for i, mask in enumerate(non_zero_mask) if mask]

            # ソルティノレシオを計算
            sortino_ratios = valid_excess_returns / valid_downside_deviations

            # ソルティノレシオ最大点（接点ポートフォリオ）
            max_sortino_idx = np.nanargmax(sortino_ratios)
            tangency_dd = valid_downside_deviations[max_sortino_idx]
            tangency_return = valid_returns[max_sortino_idx]
            tangency_weights = valid_weights[max_sortino_idx]
            tangency_sortino = sortino_ratios[max_sortino_idx]

            # 資本配分線の計算
            max_dd = np.max(downside_deviations) * 1.2
            cal_x = np.linspace(0, max_dd, 100)
            slope = (tangency_return - risk_free_rate) / tangency_dd
            cal_y = risk_free_rate + slope * cal_x

            return {
                'cal_x': cal_x.tolist(),
                'cal_y': cal_y.tolist(),
                'tangency_portfolio': {
                    'downside_deviation': tangency_dd,
                    'return': tangency_return,
                    'weights': tangency_weights,
                    'sortino_ratio': tangency_sortino
                }
            }

        except Exception as e:
            self.logger.error(f"CAL計算エラー: {e}")
            return {'cal_x': [], 'cal_y': [], 'tangency_portfolio': None}


def extract_portfolio_price_data(symbols: List[str], price_data: Dict) -> Optional[pd.DataFrame]:
    """ポートフォリオの資産の価格データを抽出してDataFrameを作成"""
    try:
        portfolio_prices = {}

        for symbol in symbols:
            if symbol in price_data:
                portfolio_prices[symbol] = price_data[symbol]

        if not portfolio_prices:
            return None

        # DataFrameに変換（共通の日付でアラインメント）
        df = pd.DataFrame(portfolio_prices)

        # NaNを含む行を削除
        df = df.dropna()

        return df

    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"価格データ抽出エラー: {e}")
        return None


def create_downside_deviation_frontier_comparison_plotly_chart(portfolios: List, price_data: Dict,
                                           all_metrics: Dict, risk_free_rate: float,
                                           config=None) -> Optional[go.Figure]:
    """複数ポートフォリオの下方偏差フロンティア比較Plotlyチャートを作成"""
    try:
        if not PLOTLY_AVAILABLE:
            logger = logging.getLogger(__name__)
            logger.warning("Plotlyが利用できません")
            return None

        logger = logging.getLogger(__name__)
        calculator = PortfolioDownsideDeviationFrontierCalculator(config)

        # 無リスク利子率のログ出力
        logger.info(f"下方偏差フロンティア比較チャート作成: risk_free_rate={risk_free_rate:.6f} ({risk_free_rate*100:.4f}%)")

        # Plotlyグラフの作成
        fig = go.Figure()

        # カラーパレット
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'brown']

        for idx, portfolio in enumerate(portfolios):
            color = colors[idx % len(colors)]
            portfolio_name = portfolio.name
            portfolio_num = idx + 1  # ポートフォリオ番号

            # ポートフォリオの資産リストを取得
            portfolio_symbols = [pos.asset.symbol for pos in portfolio.positions]

            # 価格データから該当する資産のデータを抽出
            portfolio_price_df = extract_portfolio_price_data(portfolio_symbols, price_data)

            if portfolio_price_df is None or portfolio_price_df.empty or len(portfolio_price_df.columns) < 2:
                logger.warning(f"{portfolio_name}: フロンティア計算に十分なデータがありません")
                continue

            # フロンティアを計算
            frontier_data = calculator.calculate_frontier(
                portfolio_price_df, portfolio_name, risk_free_rate, steps=50
            )

            if not frontier_data or len(frontier_data['downside_deviations']) == 0:
                logger.warning(f"{portfolio_name}: フロンティア計算結果が空です")
                continue

            downside_deviations = np.array(frontier_data['downside_deviations'])
            returns = np.array(frontier_data['returns'])
            min_dd_idx = frontier_data['min_dd_idx']

            # 1. Minimum Downside Deviation Frontier（全ての点）
            fig.add_trace(go.Scatter(
                x=downside_deviations,
                y=returns,
                mode='lines',
                name=f'P#{portfolio_num} - Min DD Frontier',
                line=dict(color=color, width=1, dash='dot'),
                opacity=0.5,
                hovertemplate=f'<b>Portfolio #{portfolio_num} - {portfolio_name}</b><br>' +
                            'Downside Deviation: %{x:.4f}<br>' +
                            'Return: %{y:.4f}<br>' +
                            '<extra></extra>'
            ))

            # 2. Efficient Downside Deviation Frontier（最小下方偏差点以降）
            efficient_dd = downside_deviations[min_dd_idx:]
            efficient_returns = returns[min_dd_idx:]
            fig.add_trace(go.Scatter(
                x=efficient_dd,
                y=efficient_returns,
                mode='lines',
                name=f'P#{portfolio_num} - Efficient DD Frontier',
                line=dict(color=color, width=1.5),
                hovertemplate=f'<b>Portfolio #{portfolio_num} - {portfolio_name}</b><br>' +
                            'Downside Deviation: %{x:.4f}<br>' +
                            'Return: %{y:.4f}<br>' +
                            '<extra></extra>'
            ))

            # 3. Global Minimum Downside Deviation Portfolio
            min_point = frontier_data['min_dd_point']
            fig.add_trace(go.Scatter(
                x=[min_point['downside_deviation']],
                y=[min_point['return']],
                mode='markers',
                name=f'P#{portfolio_num} - Global Min DD',
                marker=dict(size=5, color=color, symbol='diamond'),
                hovertemplate=f'<b>Portfolio #{portfolio_num} - {portfolio_name}</b><br>' +
                            '<b>Global Min Downside Deviation</b><br>' +
                            'Downside Deviation: %{x:.4f}<br>' +
                            'Return: %{y:.4f}<br>' +
                            '<extra></extra>'
            ))

            # 4. Capital Allocation Line（CAL - Sortino）
            cal = frontier_data['cal']
            if cal['cal_x'] and cal['cal_y']:
                fig.add_trace(go.Scatter(
                    x=cal['cal_x'],
                    y=cal['cal_y'],
                    mode='lines',
                    name=f'P#{portfolio_num} - CAL (Sortino)',
                    line=dict(color=color, width=1, dash='dash'),
                    opacity=0.7,
                    hovertemplate=f'<b>Portfolio #{portfolio_num} - {portfolio_name}</b><br>' +
                                '<b>Capital Allocation Line</b><br>' +
                                'Downside Deviation: %{x:.4f}<br>' +
                                'Return: %{y:.4f}<br>' +
                                '<extra></extra>'
                ))

                # 5. Tangency Portfolio（接点ポートフォリオ）
                if cal['tangency_portfolio']:
                    tp = cal['tangency_portfolio']
                    fig.add_trace(go.Scatter(
                        x=[tp['downside_deviation']],
                        y=[tp['return']],
                        mode='markers',
                        name=f'P#{portfolio_num} - Tangency (Sortino: {tp["sortino_ratio"]:.3f})',
                        marker=dict(size=7, color=color, symbol='circle'),
                        hovertemplate=f'<b>Portfolio #{portfolio_num} - {portfolio_name}</b><br>' +
                                    '<b>Tangency Portfolio</b><br>' +
                                    'Downside Deviation: %{x:.4f}<br>' +
                                    'Return: %{y:.4f}<br>' +
                                    'Sortino: ' + f'{tp["sortino_ratio"]:.4f}<br>' +
                                    '<extra></extra>'
                    ))

        # 6. Current Portfolio Weights（保存されているウエイト）
            portfolio_weights = portfolio.get_weights()
            mean_returns = frontier_data['mean_returns']
            downside_cov_matrix = frontier_data['downside_cov_matrix']

            # ウエイトを正規化（念のため）
            weights_array = np.array(portfolio_weights)
            if np.sum(weights_array) > 0:
                weights_array = weights_array / np.sum(weights_array)

                # 期待リターンと下方偏差を計算
                portfolio_return = np.dot(weights_array, mean_returns)
                portfolio_dd = calculator._calculate_portfolio_downside_deviation(
                    weights_array, downside_cov_matrix
                )

                fig.add_trace(go.Scatter(
                    x=[portfolio_dd],
                    y=[portfolio_return],
                    mode='markers',
                    name=f'P#{portfolio_num} - Current Weights',
                    marker=dict(size=5, color=color, symbol='x'),
                    hovertemplate=f'<b>Portfolio #{portfolio_num} - {portfolio_name}</b><br>' +
                                '<b>Current Portfolio Weights</b><br>' +
                                'Downside Deviation: %{x:.4f}<br>' +
                                'Return: %{y:.4f}<br>' +
                                '<extra></extra>'
                ))

        # 7. Risk-free Rate（無リスク利子率線）
        # チャートにデータがある場合のみ描画
        if len(fig.data) > 0:
            # x軸の範囲を取得
            all_x = []
            for trace in fig.data:
                if trace.x is not None and len(trace.x) > 0:
                    try:
                        all_x.extend([x for x in trace.x if x is not None and np.isfinite(x)])
                    except (TypeError, ValueError) as e:
                        logging.debug(f"Failed to process trace x values: {e}")
                        pass

            if all_x and risk_free_rate is not None and np.isfinite(risk_free_rate):
                max_x = max(all_x) * 1.1
                fig.add_trace(go.Scatter(
                    x=[0, max_x],
                    y=[risk_free_rate, risk_free_rate],
                    mode='lines',
                    name=f'Risk-free Rate ({risk_free_rate*100:.4f}%)',
                    line=dict(color='pink', width=1.5, dash='dot'),
                    hovertemplate='<b>Risk-free Rate</b><br>' +
                                'Rate: %{y:.4f}<br>' +
                                '<extra></extra>'
                ))

        # レイアウト設定
        fig.update_layout(
            title='Portfolio Downside Deviation Frontier Comparison',
            xaxis=dict(
                title='Downside Deviation (Downside Risk)',
                color='white',
                gridcolor='rgba(255,255,255,0.2)',
                showgrid=True
            ),
            yaxis=dict(
                title='Expected Return',
                color='white',
                gridcolor='rgba(255,255,255,0.2)',
                showgrid=True
            ),
            plot_bgcolor='rgba(50,50,50,1)',
            paper_bgcolor='rgba(43,43,43,1)',
            font=dict(color='white', size=10),
            legend=dict(
                bgcolor='rgba(0,0,0,0.7)',
                bordercolor='white',
                borderwidth=1,
                font=dict(size=9)
            ),
            hovermode='closest',
            autosize=True,
            margin=dict(l=80, r=80, t=100, b=80)
        )

        return fig

    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Plotly下方偏差フロンティア比較チャート作成エラー: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def create_downside_deviation_frontier_comparison_matplotlib_chart(portfolios: List, price_data: Dict,
                                               all_metrics: Dict, risk_free_rate: float,
                                               config=None) -> Optional[matplotlib.figure.Figure]:
    """複数ポートフォリオの下方偏差フロンティア比較Matplotlibチャートを作成（アプリ内表示用）"""
    try:
        if not MATPLOTLIB_AVAILABLE:
            logger = logging.getLogger(__name__)
            logger.warning("Matplotlibが利用できません")
            return None

        logger = logging.getLogger(__name__)
        calculator = PortfolioDownsideDeviationFrontierCalculator(config)

        # 無リスク利子率のログ出力
        logger.info(f"下方偏差フロンティア比較Matplotlibチャート作成: risk_free_rate={risk_free_rate:.6f} ({risk_free_rate*100:.4f}%)")

        # Figureを作成
        fig = matplotlib.figure.Figure(figsize=(10, 8), dpi=100)
        fig.patch.set_facecolor('#2b2b2b')
        ax = fig.add_subplot(111)
        ax.set_facecolor('#323232')

        # カラーパレット
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'brown']

        for idx, portfolio in enumerate(portfolios):
            color = colors[idx % len(colors)]
            portfolio_name = portfolio.name
            portfolio_num = idx + 1  # ポートフォリオ番号

            # ポートフォリオの資産リストを取得
            portfolio_symbols = [pos.asset.symbol for pos in portfolio.positions]

            # 価格データから該当する資産のデータを抽出
            portfolio_price_df = extract_portfolio_price_data(portfolio_symbols, price_data)

            if portfolio_price_df is None or portfolio_price_df.empty or len(portfolio_price_df.columns) < 2:
                logger.warning(f"{portfolio_name}: フロンティア計算に十分なデータがありません")
                continue

            # フロンティアを計算
            frontier_data = calculator.calculate_frontier(
                portfolio_price_df, portfolio_name, risk_free_rate, steps=50
            )

            if not frontier_data or len(frontier_data['downside_deviations']) == 0:
                logger.warning(f"{portfolio_name}: フロンティア計算結果が空です")
                continue

            downside_deviations = np.array(frontier_data['downside_deviations'])
            returns = np.array(frontier_data['returns'])
            min_dd_idx = frontier_data['min_dd_idx']

            # 1. Minimum Downside Deviation Frontier（全ての点）
            ax.plot(downside_deviations, returns, color=color, linewidth=1,
                   linestyle=':', alpha=0.5, label=f'P#{portfolio_num} - Min DD Frontier')

            # 2. Efficient Downside Deviation Frontier（最小下方偏差点以降）
            efficient_dd = downside_deviations[min_dd_idx:]
            efficient_returns = returns[min_dd_idx:]
            ax.plot(efficient_dd, efficient_returns, color=color, linewidth=2.5,
                   label=f'P#{portfolio_num} - Efficient DD Frontier')

            # 3. Global Minimum Downside Deviation Portfolio
            min_point = frontier_data['min_dd_point']
            ax.plot(min_point['downside_deviation'], min_point['return'], 'D',
                   color=color, markersize=8,
                   label=f'P#{portfolio_num} - Global Min DD')

            # 4. Capital Allocation Line（CAL）
            cal = frontier_data['cal']
            if cal['cal_x'] and cal['cal_y']:
                ax.plot(cal['cal_x'], cal['cal_y'], color=color, linewidth=2,
                       linestyle='--', alpha=0.7, label=f'P#{portfolio_num} - CAL (Sortino)')

                # 5. Tangency Portfolio（接点ポートフォリオ）
                if cal['tangency_portfolio']:
                    tp = cal['tangency_portfolio']
                    ax.plot(tp['downside_deviation'], tp['return'], 'o', color=color,
                           markersize=10,
                           label=f'P#{portfolio_num} - Tangency (Sortino: {tp["sortino_ratio"]:.3f})')

        # 6. Current Portfolio Weights（保存されているウエイト）
            portfolio_weights = portfolio.get_weights()
            mean_returns = frontier_data['mean_returns']
            downside_cov_matrix = frontier_data['downside_cov_matrix']

            # ウエイトを正規化（念のため）
            weights_array = np.array(portfolio_weights)
            if np.sum(weights_array) > 0:
                weights_array = weights_array / np.sum(weights_array)

                # 期待リターンと下方偏差を計算
                portfolio_return = np.dot(weights_array, mean_returns)
                portfolio_dd = calculator._calculate_portfolio_downside_deviation(
                    weights_array, downside_cov_matrix
                )

                ax.plot(portfolio_dd, portfolio_return, 'x', color=color,
                       markersize=14, markeredgecolor='white', markeredgewidth=2,
                       label=f'P#{portfolio_num} - Current Weights')

        # 7. Risk-free Rate（無リスク利子率線）
        if risk_free_rate is not None and np.isfinite(risk_free_rate) and risk_free_rate > 0:
            xlim = ax.get_xlim()
            ax.axhline(y=risk_free_rate, color='pink', linewidth=1.5,
                      linestyle=':', label=f'Risk-free Rate ({risk_free_rate*100:.4f}%)')

        # ラベルと凡例（英語表記）
        ax.set_xlabel('Downside Deviation (Downside Risk)', color='white', fontsize=11)
        ax.set_ylabel('Expected Return', color='white', fontsize=11)
        ax.set_title('Portfolio Downside Deviation Frontier Comparison', color='white', fontsize=13, pad=20)

        # 凡例を2列で表示
        ax.legend(facecolor='#2b2b2b', edgecolor='white', labelcolor='white',
                 fontsize=8, ncol=2, loc='best')
        ax.grid(True, alpha=0.3, color='white')

        # 軸の色を設定
        ax.tick_params(colors='white', labelsize=9)
        for spine in ax.spines.values():
            spine.set_color('white')

        fig.tight_layout()

        return fig

    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Matplotlib下方偏差フロンティア比較チャート作成エラー: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None