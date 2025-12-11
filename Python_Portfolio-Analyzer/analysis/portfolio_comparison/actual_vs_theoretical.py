"""actual_vs_theoretical.py"""

import numpy as np
from typing import Dict, Optional
import logging

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
    PlotlyFigure = go.Figure
except ImportError:
    PLOTLY_AVAILABLE = False
    PlotlyFigure = object


logger = logging.getLogger(__name__)


def _calculate_theoretical_return(
    beta: float,
    risk_free_rate: float,
    market_return: float
) -> float:
    """CAPM理論リターンを計算"""
    return risk_free_rate + beta * (market_return - risk_free_rate)


def create_actual_vs_theoretical_matplotlib_chart(
    metrics: Dict[str, Dict],
    betas: Dict[str, float],
    risk_free_rate: float,
    market_return: float,
    config: Optional[object] = None
) -> Optional[Figure]:
    """Matplotlibで実測vs理論値チャートを作成"""
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Matplotlibが利用できません")
        return None
    
    try:
        # カラー設定
        if config:
            colors = config.get_ui_colors()
            bg_color = colors.get('background', '#2b2b2b')
            text_color = colors.get('text_primary', '#ffffff')
            grid_color = colors.get('grid_line', '#444444')
        else:
            bg_color = '#2b2b2b'
            text_color = '#ffffff'
            grid_color = '#444444'
        
        # データ準備
        portfolios = []
        actual_returns = []
        theoretical_returns = []
        beta_values = []
        
        for portfolio_name in metrics.keys():
            if portfolio_name in betas:
                portfolios.append(portfolio_name)
                
                actual_ret = metrics[portfolio_name].get('annualized_return', 0) * 100
                actual_returns.append(actual_ret)
                
                beta = betas[portfolio_name]
                beta_values.append(beta)
                
                theoretical_ret = _calculate_theoretical_return(
                    beta, risk_free_rate, market_return
                ) * 100
                theoretical_returns.append(theoretical_ret)
        
        if not portfolios:
            logger.warning("比較可能なデータがありません")
            return None
        
        # ポートフォリオ名を番号に変換
        labels = [f"#{i+1:02d}" for i in range(len(portfolios))]
        
        # フィギュア作成（2つのサブプロット）
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.patch.set_facecolor(bg_color)
        
        # サブプロット1: 実測vs理論値の比較バー
        x_pos = np.arange(len(portfolios))
        width = 0.35
        
        bars1 = ax1.bar(x_pos - width/2, actual_returns, width,
                       label='Actual Return', color='#4CAF50', alpha=0.8)
        bars2 = ax1.bar(x_pos + width/2, theoretical_returns, width,
                       label='Theoretical (CAPM)', color='#2196F3', alpha=0.8)
        
        ax1.set_xlabel('Portfolio', color=text_color, fontsize=11)
        ax1.set_ylabel('Annual Return (%)', color=text_color, fontsize=11)
        ax1.set_title('Actual vs Theoretical Return',
                     color=text_color, fontsize=12, fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(labels, rotation=0, color=text_color, fontsize=10)
        ax1.tick_params(colors=text_color)
        ax1.legend(facecolor=bg_color, edgecolor=text_color,
                  labelcolor=text_color, fontsize=10)
        ax1.set_facecolor(bg_color)
        ax1.grid(True, alpha=0.3, color=grid_color, axis='y')
        ax1.axhline(y=0, color=text_color, linewidth=0.8, alpha=0.3)
        
        # サブプロット2: 散布図（実測 vs 理論値）
        ax2.scatter(theoretical_returns, actual_returns,
                   s=150, alpha=0.7, color='#FF9800', edgecolors=text_color, linewidth=1.5)
        
        # ラベル追加
        for i, label in enumerate(labels):
            ax2.annotate(label,
                        (theoretical_returns[i], actual_returns[i]),
                        textcoords="offset points",
                        xytext=(5, 5),
                        fontsize=9,
                        color=text_color)
        
        # 45度線（実測=理論値）
        min_val = min(min(theoretical_returns), min(actual_returns))
        max_val = max(max(theoretical_returns), max(actual_returns))
        margin = (max_val - min_val) * 0.1
        
        ax2.plot([min_val - margin, max_val + margin],
                [min_val - margin, max_val + margin],
                'r--', linewidth=2, alpha=0.5, label='Perfect Match')
        
        ax2.set_xlabel('Theoretical Return (CAPM) [%]', color=text_color, fontsize=11)
        ax2.set_ylabel('Actual Return [%]', color=text_color, fontsize=11)
        ax2.set_title('Actual vs Theoretical Scatter',
                     color=text_color, fontsize=12, fontweight='bold')
        ax2.tick_params(colors=text_color)
        ax2.legend(facecolor=bg_color, edgecolor=text_color,
                  labelcolor=text_color, fontsize=10)
        ax2.set_facecolor(bg_color)
        ax2.grid(True, alpha=0.3, color=grid_color)
        
        plt.tight_layout()
        
        return fig
        
    except Exception as e:
        logger.error(f"Matplotlibチャート作成エラー: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def create_actual_vs_theoretical_plotly_chart(
    metrics: Dict[str, Dict],
    betas: Dict[str, float],
    risk_free_rate: float,
    market_return: float,
    config: Optional[object] = None
) -> Optional[PlotlyFigure]:
    """Plotlyでインタラクティブな実測vs理論値チャートを作成"""
    if not PLOTLY_AVAILABLE:
        logger.warning("Plotlyが利用できません")
        return None
    
    try:
        from plotly.subplots import make_subplots
        
        # データ準備
        portfolios = []
        actual_returns = []
        theoretical_returns = []
        beta_values = []
        
        for portfolio_name in metrics.keys():
            if portfolio_name in betas:
                portfolios.append(portfolio_name)
                
                actual_ret = metrics[portfolio_name].get('annualized_return', 0) * 100
                actual_returns.append(actual_ret)
                
                beta = betas[portfolio_name]
                beta_values.append(beta)
                
                theoretical_ret = _calculate_theoretical_return(
                    beta, risk_free_rate, market_return
                ) * 100
                theoretical_returns.append(theoretical_ret)
        
        if not portfolios:
            logger.warning("比較可能なデータがありません")
            return None
        
        # サブプロット作成
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Actual vs Theoretical Return', 'Scatter Plot'),
            horizontal_spacing=0.12
        )
        
        # サブプロット1: バーチャート
        fig.add_trace(
            go.Bar(
                name='Actual Return',
                x=portfolios,
                y=actual_returns,
                marker_color='#4CAF50',
                hovertemplate='<b>%{x}</b><br>Actual: %{y:.2f}%<extra></extra>'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(
                name='Theoretical (CAPM)',
                x=portfolios,
                y=theoretical_returns,
                marker_color='#2196F3',
                hovertemplate='<b>%{x}</b><br>Theoretical: %{y:.2f}%<extra></extra>'
            ),
            row=1, col=1
        )
        
        # サブプロット2: 散布図
        fig.add_trace(
            go.Scatter(
                x=theoretical_returns,
                y=actual_returns,
                mode='markers+text',
                name='Portfolios',
                text=portfolios,
                textposition='top center',
                marker=dict(size=12, color='#FF9800', line=dict(width=2, color='white')),
                hovertemplate=(
                    '<b>%{text}</b><br>'
                    'Theoretical: %{x:.2f}%<br>'
                    'Actual: %{y:.2f}%<br>'
                    '<extra></extra>'
                )
            ),
            row=1, col=2
        )
        
        # 45度線
        min_val = min(min(theoretical_returns), min(actual_returns))
        max_val = max(max(theoretical_returns), max(actual_returns))
        margin = (max_val - min_val) * 0.1
        
        fig.add_trace(
            go.Scatter(
                x=[min_val - margin, max_val + margin],
                y=[min_val - margin, max_val + margin],
                mode='lines',
                name='Perfect Match',
                line=dict(color='red', width=2, dash='dash'),
                showlegend=True
            ),
            row=1, col=2
        )
        
        # レイアウト更新
        fig.update_xaxes(title_text='Portfolio', row=1, col=1, color='white')
        fig.update_yaxes(title_text='Annual Return (%)', row=1, col=1, color='white')
        
        fig.update_xaxes(title_text='Theoretical Return (CAPM) [%]', row=1, col=2, color='white')
        fig.update_yaxes(title_text='Actual Return [%]', row=1, col=2, color='white')
        
        fig.update_layout(
            title_text="Actual vs Theoretical (CAPM) Analysis",
            plot_bgcolor='rgba(50,50,50,1)',
            paper_bgcolor='rgba(43,43,43,1)',
            font=dict(color='white'),
            showlegend=True,
            legend=dict(
                bgcolor='rgba(0,0,0,0.7)',
                bordercolor='white',
                borderwidth=1
            ),
            barmode='group'
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Plotlyチャート作成エラー: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


# テスト用関数
def _generate_test_data():
    """テストデータ生成"""
    metrics = {
        'Portfolio A': {
            'annualized_return': 0.08,
        },
        'Portfolio B': {
            'annualized_return': 0.06,
        },
        'Portfolio C': {
            'annualized_return': 0.10,
        }
    }
    
    betas = {
        'Portfolio A': 1.2,
        'Portfolio B': 0.8,
        'Portfolio C': 1.5,
    }
    
    risk_free_rate = 0.02  # 2%
    market_return = 0.08  # 8%
    
    return metrics, betas, risk_free_rate, market_return


if __name__ == '__main__':
    # テスト実行
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("Actual vs Theoretical Module Test")
    print("=" * 60)
    
    metrics, betas, rf_rate, mkt_return = _generate_test_data()
    
    if MATPLOTLIB_AVAILABLE:
        print("\n1. Creating Matplotlib chart...")
        fig_mpl = create_actual_vs_theoretical_matplotlib_chart(
            metrics, betas, rf_rate, mkt_return
        )
        if fig_mpl:
            print("✓ Matplotlib chart created successfully")
        else:
            print("✗ Failed to create Matplotlib chart")
    
    if PLOTLY_AVAILABLE:
        print("\n2. Creating Plotly chart...")
        fig_plotly = create_actual_vs_theoretical_plotly_chart(
            metrics, betas, rf_rate, mkt_return
        )
        if fig_plotly:
            print("✓ Plotly chart created successfully")
        else:
            print("✗ Failed to create Plotly chart")
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)
