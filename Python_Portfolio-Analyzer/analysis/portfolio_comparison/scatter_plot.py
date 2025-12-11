"""scatter_plot.py"""

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


def create_scatter_matplotlib_chart(
    metrics: Dict[str, Dict],
    efficiency: Dict[str, Dict],
    config: Optional[object] = None
) -> Optional[Figure]:
    """Matplotlibでリスク・リターン散布図を作成"""
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
        
        # データ抽出
        portfolio_names = list(metrics.keys())
        risks = [metrics[name].get('annualized_std', 0) * 100 for name in portfolio_names]
        returns = [metrics[name].get('annualized_return', 0) * 100 for name in portfolio_names]
        sharpes = [efficiency.get(name, {}).get('annualized_sharpe', 0) for name in portfolio_names]
        
        # ポートフォリオ名を番号に変換
        labels = [f"#{i+1:02d}" for i in range(len(portfolio_names))]
        
        # フィギュア作成
        fig, ax = plt.subplots(figsize=(10, 7))
        fig.patch.set_facecolor(bg_color)
        
        # カラーマップ（シャープレシオベース）
        scatter = ax.scatter(
            risks, returns,
            s=200,
            c=sharpes,
            cmap='RdYlGn',
            alpha=0.7,
            edgecolors=text_color,
            linewidth=1.5
        )
        
        # ラベル追加
        for i, (x, y, label) in enumerate(zip(risks, returns, labels)):
            ax.annotate(
                label,
                (x, y),
                textcoords="offset points",
                xytext=(0, 10),
                ha='center',
                fontsize=10,
                color=text_color,
                weight='bold'
            )
        
        # カラーバー
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.15)
        cbar = plt.colorbar(scatter, cax=cax)
        cbar.set_label('Sharpe Ratio', color=text_color, fontsize=10)
        cbar.ax.tick_params(colors=text_color)
        
        # 軸設定
        ax.set_xlabel('Annual Risk (Std Dev) [%]', color=text_color, fontsize=11)
        ax.set_ylabel('Annual Return [%]', color=text_color, fontsize=11)
        ax.set_title('Risk-Return Scatter Plot', 
                    color=text_color, fontsize=14, fontweight='bold', pad=15)
        
        ax.set_facecolor(bg_color)
        ax.tick_params(colors=text_color)
        ax.grid(True, alpha=0.3, color=grid_color, linestyle='--')
        
        # 軸の範囲を調整（マージンを追加）
        x_margin = (max(risks) - min(risks)) * 0.1 if len(risks) > 1 else 1
        y_margin = (max(returns) - min(returns)) * 0.1 if len(returns) > 1 else 1
        ax.set_xlim(min(risks) - x_margin, max(risks) + x_margin)
        ax.set_ylim(min(returns) - y_margin, max(returns) + y_margin)
        
        # 原点線（リスク=0，リターン=0）
        ax.axhline(y=0, color=text_color, linewidth=0.8, linestyle=':', alpha=0.5)
        ax.axvline(x=0, color=text_color, linewidth=0.8, linestyle=':', alpha=0.5)
        
        fig.tight_layout()
        
        return fig
        
    except Exception as e:
        logger.error(f"Matplotlibチャート作成エラー: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def create_scatter_plotly_chart(
    metrics: Dict[str, Dict],
    efficiency: Dict[str, Dict],
    config: Optional[object] = None
) -> Optional[PlotlyFigure]:
    """Plotlyでインタラクティブなリスク・リターン散布図を作成"""
    if not PLOTLY_AVAILABLE:
        logger.warning("Plotlyが利用できません")
        return None
    
    try:
        fig = go.Figure()
        
        for portfolio_name, pf_metrics in metrics.items():
            pf_efficiency = efficiency.get(portfolio_name, {})
            
            x = pf_metrics.get('annualized_std', 0) * 100
            y = pf_metrics.get('annualized_return', 0) * 100
            sharpe = pf_efficiency.get('annualized_sharpe', 0)
            sortino = pf_efficiency.get('sortino_ratio', 0)
            max_dd = pf_metrics.get('max_drawdown', 0) * 100
            
            fig.add_trace(go.Scatter(
                x=[x],
                y=[y],
                mode='markers+text',
                name=portfolio_name,
                text=[portfolio_name],
                textposition="top center",
                marker=dict(
                    size=15,
                    line=dict(width=2, color='white')
                ),
                hovertemplate=(
                    f"<b>{portfolio_name}</b><br>"
                    f"Risk: {x:.2f}%<br>"
                    f"Return: {y:.2f}%<br>"
                    f"Sharpe Ratio: {sharpe:.3f}<br>"
                    f"Sortino Ratio: {sortino:.3f}<br>"
                    f"Max DD: {max_dd:.2f}%<br>"
                    "<extra></extra>"
                )
            ))
        
        fig.update_layout(
            title="Risk-Return Scatter Plot",
            xaxis=dict(
                title='Annual Risk (Std Dev) [%]',
                color='white',
                gridcolor='rgba(255,255,255,0.2)'
            ),
            yaxis=dict(
                title='Annual Return [%]',
                color='white',
                gridcolor='rgba(255,255,255,0.2)'
            ),
            plot_bgcolor='rgba(50,50,50,1)',
            paper_bgcolor='rgba(43,43,43,1)',
            font=dict(color='white'),
            legend=dict(
                bgcolor='rgba(0,0,0,0.7)',
                bordercolor='white',
                borderwidth=1
            ),
            hovermode='closest'
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
            'annualized_std': 0.12,
            'max_drawdown': -0.15,
        },
        'Portfolio B': {
            'annualized_return': 0.06,
            'annualized_std': 0.10,
            'max_drawdown': -0.08,
        },
        'Portfolio C': {
            'annualized_return': 0.10,
            'annualized_std': 0.20,
            'max_drawdown': -0.18,
        }
    }
    
    efficiency = {
        'Portfolio A': {
            'annualized_sharpe': 0.53,
            'sortino_ratio': 0.75
        },
        'Portfolio B': {
            'annualized_sharpe': 0.60,
            'sortino_ratio': 0.85
        },
        'Portfolio C': {
            'annualized_sharpe': 0.50,
            'sortino_ratio': 0.65
        }
    }
    
    return metrics, efficiency


if __name__ == '__main__':
    # テスト実行
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("Scatter Plot Module Test")
    print("=" * 60)
    
    metrics, efficiency = _generate_test_data()
    
    if MATPLOTLIB_AVAILABLE:
        print("\n1. Creating Matplotlib chart...")
        fig_mpl = create_scatter_matplotlib_chart(metrics, efficiency)
        if fig_mpl:
            print("✓ Matplotlib chart created successfully")
        else:
            print("✗ Failed to create Matplotlib chart")
    
    if PLOTLY_AVAILABLE:
        print("\n2. Creating Plotly chart...")
        fig_plotly = create_scatter_plotly_chart(metrics, efficiency)
        if fig_plotly:
            print("✓ Plotly chart created successfully")
        else:
            print("✗ Failed to create Plotly chart")
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)
