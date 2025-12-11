"""market_comparison.py"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Optional

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class PortfolioMarketComparison:
    """ポートフォリオと市場の比較分析クラス"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def calculate_portfolio_betas(
        self,
        portfolio_returns: Dict[str, pd.Series],
        market_returns: pd.Series,
        risk_free_rate: float
    ) -> Dict[str, Dict]:
        """各ポートフォリオのβ値とα値を計算"""
        results = {}

        for pf_name, pf_returns in portfolio_returns.items():
            try:
                # 共通日付でのデータを取得
                common_dates = pf_returns.index.intersection(market_returns.index)

                if len(common_dates) < 10:  # 最低10日分のデータが必要
                    self.logger.warning(f"{pf_name}: 共通日付が不足（{len(common_dates)}日）")
                    continue

                pf_common = pf_returns.reindex(common_dates)
                market_common = market_returns.reindex(common_dates)

                # 超過リターンを計算
                pf_excess = pf_common - risk_free_rate
                market_excess = market_common - risk_free_rate

                # NaNを除去
                valid_mask = ~(np.isnan(pf_excess) | np.isnan(market_excess))
                pf_excess_clean = pf_excess[valid_mask]
                market_excess_clean = market_excess[valid_mask]

                if len(pf_excess_clean) < 5:
                    self.logger.warning(f"{pf_name}: 有効データが不足（{len(pf_excess_clean)}日）")
                    continue

                # 回帰分析の実行
                if SCIPY_AVAILABLE:
                    # scipy.statsを使用した回帰分析
                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        market_excess_clean, pf_excess_clean
                    )

                    beta = slope
                    alpha = intercept
                    correlation = r_value
                    r_squared = r_value ** 2

                    # 信頼区間の計算（95%信頼区間）
                    n = len(pf_excess_clean)
                    dof = n - 2  # 自由度
                    t_val = stats.t.ppf(0.975, dof)  # 95%信頼区間のt値
                    beta_ci_lower = beta - t_val * std_err
                    beta_ci_upper = beta + t_val * std_err

                    # 予測値と残差
                    predicted = alpha + beta * market_excess_clean
                    residuals = pf_excess_clean - predicted

                    # 残差の統計
                    residual_std = np.std(residuals, ddof=2)

                else:
                    # numpyのみを使用した簡易回帰分析
                    X = np.column_stack([np.ones(len(market_excess_clean)), market_excess_clean])
                    y = pf_excess_clean.values

                    # 最小二乗法
                    coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
                    alpha = coeffs[0]
                    beta = coeffs[1]

                    # 基本統計
                    predicted = alpha + beta * market_excess_clean
                    residuals = y - predicted
                    correlation = np.corrcoef(market_excess_clean, pf_excess_clean)[0, 1]
                    r_squared = correlation ** 2

                    # 標準誤差の近似計算
                    residual_variance = np.var(residuals, ddof=2)
                    market_variance = np.var(market_excess_clean, ddof=1)
                    std_err = np.sqrt(residual_variance / (len(market_excess_clean) * market_variance))

                    # 信頼区間の近似
                    z_val = 1.96  # 95%信頼区間のz値（近似）
                    beta_ci_lower = beta - z_val * std_err
                    beta_ci_upper = beta + z_val * std_err

                    residual_std = np.std(residuals, ddof=2)
                    p_value = np.nan

                # 期待利益率の計算
                expected_return = pf_common.mean()

                # CAPM期待利益率の計算
                capm_expected_return = risk_free_rate + beta * (market_common.mean() - risk_free_rate)

                # システマティックリスクとアンシステマティックリスク
                market_variance = np.var(market_excess_clean, ddof=1)
                total_variance = np.var(pf_excess_clean, ddof=1)
                systematic_variance = beta ** 2 * market_variance
                unsystematic_variance = max(0, total_variance - systematic_variance)

                # 結果の格納
                results[pf_name] = {
                    'beta': beta,
                    'alpha': alpha,
                    'expected_return': expected_return,
                    'capm_expected_return': capm_expected_return,
                    'correlation': correlation,
                    'r_squared': r_squared,
                    'systematic_risk': np.sqrt(systematic_variance) if systematic_variance >= 0 else 0,
                    'unsystematic_risk': np.sqrt(unsystematic_variance) if unsystematic_variance >= 0 else 0,
                    'total_risk': np.sqrt(total_variance),
                    'common_dates': len(pf_excess_clean),
                    'beta_std_error': std_err,
                    'beta_p_value': p_value if SCIPY_AVAILABLE else np.nan,
                    'beta_ci_lower': beta_ci_lower,
                    'beta_ci_upper': beta_ci_upper,
                    'residual_std_error': residual_std,
                    'success': True
                }

                self.logger.info(f"{pf_name}: β={beta:.4f}±{std_err:.4f}, α={alpha:.4f}, R²={r_squared:.4f}")

            except Exception as e:
                self.logger.error(f"{pf_name} の回帰分析エラー: {e}")
                results[pf_name] = {
                    'beta': np.nan, 'alpha': np.nan, 'expected_return': np.nan,
                    'capm_expected_return': np.nan, 'correlation': np.nan, 'r_squared': np.nan,
                    'systematic_risk': np.nan, 'unsystematic_risk': np.nan, 'total_risk': np.nan,
                    'common_dates': 0, 'beta_std_error': np.nan, 'beta_p_value': np.nan,
                    'beta_ci_lower': np.nan, 'beta_ci_upper': np.nan,
                    'residual_std_error': np.nan, 'success': False, 'error': str(e)
                }

        return results

    def calculate_security_market_line(
        self,
        risk_free_rate: float,
        market_expected_return: float
    ) -> Dict:
        """証券市場線の計算"""
        try:
            # β値の範囲を設定（-0.5から2.5まで）
            beta_range = np.linspace(-0.5, 2.5, 100)

            # 市場超過リターン
            market_excess_return = market_expected_return - risk_free_rate

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


def create_market_comparison_plotly_chart(
    portfolio_beta_results: Dict[str, Dict],
    sml_line: Dict,
    market_expected_return: float,
    risk_free_rate: float,
    config=None
) -> Optional[go.Figure]:
    """市場比較のPlotlyチャートを作成"""
    if not PLOTLY_AVAILABLE:
        return None

    logger = logging.getLogger(__name__)

    try:
        # 有効なβ値データのみを取得
        valid_data = {name: data for name, data in portfolio_beta_results.items()
                     if data.get('success', False)}

        if not valid_data:
            logger.warning("有効なβ値データがありません")
            return None

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

        # 各ポートフォリオのプロット
        pf_names = list(valid_data.keys())
        betas = [valid_data[name]['beta'] for name in pf_names]
        expected_returns = [valid_data[name]['expected_return'] * 100 for name in pf_names]
        capm_returns = [valid_data[name]['capm_expected_return'] * 100 for name in pf_names]
        alphas = [valid_data[name]['alpha'] * 100 for name in pf_names]
        r_squareds = [valid_data[name]['r_squared'] for name in pf_names]

        # ポートフォリオの散布図
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
            text=pf_names,
            textposition="middle right",
            name='Portfolios',
            marker=dict(
                size=10,
                color=colors,
                line=dict(width=2, color='white')
            ),
            textfont=dict(color='white', size=10),
            customdata=list(zip(capm_returns, alphas, r_squareds)),
            hovertemplate=('<b>%{text}</b><br>β: %{x:.3f}<br>'
                          'Expected Return: %{y:.3f}%<br>'
                          'CAPM Expected Return: %{customdata[0]:.3f}%<br>'
                          'α: %{customdata[1]:.3f}%<br>'
                          'R²: %{customdata[2]:.3f}<extra></extra>')
        ))

        # 市場ポートフォリオの点
        fig.add_trace(go.Scatter(
            x=[1.0],
            y=[market_expected_return * 100],
            mode='markers',
            name='Market Portfolio',
            marker=dict(size=12, color='green', symbol='circle'),
            hovertemplate='Market Portfolio<br>β: 1.0<br>Expected Return: %{y:.3f}%<extra></extra>'
        ))

        # Risk-free Rateを凡例として表示
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode='lines',
            name=f'Risk-free Rate ({risk_free_rate*100:.3f}%)',
            line=dict(color='pink', dash='dot'),
            showlegend=True
        ))

        # 実際のRisk-free Rate線
        max_beta = max(max(betas) if betas else 1, 1.5)
        min_beta = min(min(betas) if betas else -0.5, -0.5)

        fig.add_shape(
            type="line",
            x0=min_beta - 0.2, x1=max_beta + 0.2,
            y0=risk_free_rate * 100, y1=risk_free_rate * 100,
            line=dict(color="pink", width=1, dash="dot"),
        )

        # レイアウト設定
        fig.update_layout(
            title='Portfolio vs Market - Security Market Line',
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
            margin=dict(l=50, r=80, t=50, b=50)
        )

        return fig

    except Exception as e:
        logger.error(f"Plotlyチャート作成エラー: {e}")
        return None


def create_market_comparison_matplotlib_chart(
    portfolio_beta_results: Dict[str, Dict],
    sml_line: Dict,
    market_expected_return: float,
    risk_free_rate: float,
    config=None
) -> Optional['matplotlib.figure.Figure']:
    """市場比較のMatplotlibチャートを作成"""
    if not MATPLOTLIB_AVAILABLE:
        return None

    logger = logging.getLogger(__name__)

    try:
        # 有効なデータのみを取得
        valid_data = {name: data for name, data in portfolio_beta_results.items()
                     if data.get('success', False)}

        if not valid_data:
            logger.warning("有効なβ値データがありません")
            return None

        # 図の作成
        fig = plt.figure(figsize=(10, 8), facecolor='#2b2b2b')
        ax = fig.add_subplot(111)
        ax.set_facecolor('#323232')

        # 証券市場線
        if sml_line['beta_range'] and sml_line['sml_returns']:
            sml_returns_pct = [r * 100 for r in sml_line['sml_returns']]
            ax.plot(sml_line['beta_range'], sml_returns_pct, 'r-', linewidth=2,
                   label='Security Market Line', zorder=1)

        # 各ポートフォリオのプロット
        pf_names = list(valid_data.keys())
        betas = [valid_data[name]['beta'] for name in pf_names]
        expected_returns_pct = [valid_data[name]['expected_return'] * 100 for name in pf_names]
        alphas = [valid_data[name]['alpha'] * 100 for name in pf_names]

        # α値に基づく色分け
        colors = []
        for alpha in alphas:
            if pd.isna(alpha):
                colors.append('gray')
            elif alpha >= 0:
                colors.append('blue')
            else:
                colors.append('orange')

        for i, name in enumerate(pf_names):
            ax.scatter(betas[i], expected_returns_pct[i],
                      c=colors[i], s=80, alpha=0.8, edgecolors='white', zorder=2)
            ax.annotate(name, (betas[i], expected_returns_pct[i]),
                       xytext=(5, 0), textcoords='offset points',
                       fontsize=9, color='white', zorder=3)

        # 市場ポートフォリオの点
        ax.scatter(1.0, market_expected_return * 100,
                  c='green', s=100, marker='o',
                  label='Market Portfolio', edgecolors='white', zorder=4)

        # 無リスク利子率の線
        max_beta = max(max(betas) if betas else 1, 1.5)
        min_beta = min(min(betas) if betas else -0.5, -0.5)
        ax.axhline(y=risk_free_rate * 100, color='pink', linewidth=1,
                  linestyle=':', alpha=0.8,
                  label=f'Risk-free Rate ({risk_free_rate*100:.3f}%)', zorder=0)

        # ラベルと凡例
        ax.set_xlabel('Beta (β)', color='white', fontsize=11)
        ax.set_ylabel('Expected Return (%)', color='white', fontsize=11)
        ax.set_title('Portfolio vs Market - Security Market Line', color='white', fontsize=13)
        ax.legend(facecolor='#2b2b2b', edgecolor='white', labelcolor='white', fontsize=9)
        ax.grid(True, alpha=0.3, zorder=0)
        ax.set_xlim(min_beta - 0.2, max_beta + 0.2)

        # 軸の色を設定
        ax.tick_params(colors='white', labelsize=9)
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['right'].set_color('white')
        ax.spines['left'].set_color('white')

        plt.tight_layout()

        return fig

    except Exception as e:
        logger.error(f"Matplotlibチャート作成エラー: {e}")
        return None