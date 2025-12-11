"""summary_stats.py"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import logging

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
    PlotlyFigure = go.Figure
except ImportError:
    PLOTLY_AVAILABLE = False
    PlotlyFigure = object


logger = logging.getLogger(__name__)


def create_summary_table_data(
    metrics: Dict[str, Dict],
    efficiency: Dict[str, Dict]
) -> pd.DataFrame:
    """比較サマリーテーブルのデータを作成"""
    try:
        data = []
        
        for portfolio_name in metrics.keys():
            pf_metrics = metrics[portfolio_name]
            pf_efficiency = efficiency.get(portfolio_name, {})
            
            row = {
                'Portfolio': portfolio_name,
                'Assets': pf_metrics.get('n_positions', 0),
                'Ann. Return (%)': pf_metrics.get('annualized_return', np.nan) * 100,
                'Ann. Risk (%)': pf_metrics.get('annualized_std', np.nan) * 100,
                'Sharpe Ratio': pf_efficiency.get('annualized_sharpe', np.nan),
                'Sortino Ratio': pf_efficiency.get('sortino_ratio', np.nan),
                'Max DD (%)': pf_metrics.get('max_drawdown', np.nan) * 100,
                'VaR 95% (%)': pf_metrics.get('var_95', np.nan) * 100,
                'CVaR 95% (%)': pf_metrics.get('cvar_95', np.nan) * 100,
            }
            
            data.append(row)
        
        df = pd.DataFrame(data)
        return df
        
    except Exception as e:
        logger.error(f"サマリーテーブルデータ作成エラー: {e}")
        return pd.DataFrame()


def create_summary_matplotlib_chart(
    metrics: Dict[str, Dict],
    efficiency: Dict[str, Dict],
    config: Optional[object] = None
) -> Optional[Figure]:
    """Matplotlibでサマリーチャートを作成"""
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Matplotlibが利用できません")
        return None
    
    try:
        # データフレームを作成
        df = create_summary_table_data(metrics, efficiency)
        
        if df.empty:
            logger.warning("データが空のため，チャートを作成できません")
            return None
        
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
        
        # ポートフォリオ名を番号に変換（#01, #02, ...）
        portfolio_labels = [f"#{i+1:02d}" for i in range(len(df))]
        
        # サブプロット作成（2x2グリッド）
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.patch.set_facecolor(bg_color)
        
        # 1. リターン vs リスク
        ax1 = axes[0, 0]
        x_pos = np.arange(len(df))
        width = 0.35
        
        ax1.bar(x_pos - width/2, df['Ann. Return (%)'], width, 
                label='Annual Return', color='#4CAF50', alpha=0.8)
        ax1.bar(x_pos + width/2, df['Ann. Risk (%)'], width, 
                label='Annual Risk', color='#FF5252', alpha=0.8)
        
        ax1.set_xlabel('Portfolio', color=text_color, fontsize=10)
        ax1.set_ylabel('Percentage (%)', color=text_color, fontsize=10)
        ax1.set_title('Return vs Risk', color=text_color, fontsize=12, fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(portfolio_labels, rotation=0, color=text_color, fontsize=9)
        ax1.tick_params(colors=text_color)
        ax1.legend(facecolor=bg_color, edgecolor=text_color, labelcolor=text_color, fontsize=9)
        ax1.set_facecolor(bg_color)
        ax1.grid(True, alpha=0.2, color=grid_color)
        ax1.axhline(y=0, color=text_color, linewidth=0.8, alpha=0.3)
        
        # 2. シャープレシオとソルティノレシオ
        ax2 = axes[0, 1]
        ax2.bar(x_pos - width/2, df['Sharpe Ratio'], width, 
                label='Sharpe Ratio', color='#2196F3', alpha=0.8)
        ax2.bar(x_pos + width/2, df['Sortino Ratio'], width, 
                label='Sortino Ratio', color='#9C27B0', alpha=0.8)
        
        ax2.set_xlabel('Portfolio', color=text_color, fontsize=10)
        ax2.set_ylabel('Ratio', color=text_color, fontsize=10)
        ax2.set_title('Risk-Adjusted Returns', color=text_color, fontsize=12, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(portfolio_labels, rotation=0, color=text_color, fontsize=9)
        ax2.tick_params(colors=text_color)
        ax2.legend(facecolor=bg_color, edgecolor=text_color, labelcolor=text_color, fontsize=9)
        ax2.set_facecolor(bg_color)
        ax2.grid(True, alpha=0.2, color=grid_color)
        ax2.axhline(y=0, color=text_color, linewidth=0.8, alpha=0.3)
        
        # 3. 最大ドローダウン
        ax3 = axes[1, 0]
        colors_dd = ['#FF6B6B' if val < -15 else '#FFA726' if val < -10 else '#FFD93D' 
                     for val in df['Max DD (%)']]
        ax3.bar(x_pos, df['Max DD (%)'], color=colors_dd, alpha=0.8)
        
        ax3.set_xlabel('Portfolio', color=text_color, fontsize=10)
        ax3.set_ylabel('Max Drawdown (%)', color=text_color, fontsize=10)
        ax3.set_title('Maximum Drawdown', color=text_color, fontsize=12, fontweight='bold')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(portfolio_labels, rotation=0, color=text_color, fontsize=9)
        ax3.tick_params(colors=text_color)
        ax3.set_facecolor(bg_color)
        ax3.grid(True, alpha=0.2, color=grid_color)
        ax3.axhline(y=0, color=text_color, linewidth=0.8, alpha=0.3)
        
        # 4. VaR と CVaR
        ax4 = axes[1, 1]
        ax4.bar(x_pos - width/2, df['VaR 95% (%)'], width, 
                label='VaR 95%', color='#FF9800', alpha=0.8)
        ax4.bar(x_pos + width/2, df['CVaR 95% (%)'], width, 
                label='CVaR 95%', color='#F44336', alpha=0.8)
        
        ax4.set_xlabel('Portfolio', color=text_color, fontsize=10)
        ax4.set_ylabel('Risk Measure (%)', color=text_color, fontsize=10)
        ax4.set_title('Value at Risk & Conditional VaR', color=text_color, fontsize=12, fontweight='bold')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(portfolio_labels, rotation=0, color=text_color, fontsize=9)
        ax4.tick_params(colors=text_color)
        ax4.legend(facecolor=bg_color, edgecolor=text_color, labelcolor=text_color, fontsize=9)
        ax4.set_facecolor(bg_color)
        ax4.grid(True, alpha=0.2, color=grid_color)
        ax4.axhline(y=0, color=text_color, linewidth=0.8, alpha=0.3)
        
        # 全体のタイトル
        fig.suptitle('Portfolio Comparison Summary', 
                    color=text_color, fontsize=14, fontweight='bold', y=0.995)
        
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        
        return fig
        
    except Exception as e:
        logger.error(f"Matplotlibチャート作成エラー: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def create_summary_plotly_chart(
    metrics: Dict[str, Dict],
    efficiency: Dict[str, Dict],
    config: Optional[object] = None
) -> Optional[PlotlyFigure]:
    """Plotlyでインタラクティブなサマリーチャートを作成"""
    if not PLOTLY_AVAILABLE:
        logger.warning("Plotlyが利用できません")
        return None
    
    try:
        # データフレームを作成
        df = create_summary_table_data(metrics, efficiency)
        
        if df.empty:
            logger.warning("データが空のため，チャートを作成できません")
            return None
        
        # ポートフォリオ名を番号に変換
        portfolio_labels = [f"#{i+1:02d}" for i in range(len(df))]
        
        # サブプロット作成（2x2グリッド）
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Return vs Risk', 
                'Risk-Adjusted Returns',
                'Maximum Drawdown',
                'Value at Risk Measures'
            ),
            vertical_spacing=0.12,
            horizontal_spacing=0.10
        )
        
        # 1. リターン vs リスク（バーチャート）
        fig.add_trace(
            go.Bar(
                name='Annual Return',
                x=portfolio_labels,
                y=df['Ann. Return (%)'],
                marker_color='#4CAF50',
                opacity=0.8,
                hovertemplate='%{x}<br>Return: %{y:.2f}%<extra></extra>'
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(
                name='Annual Risk',
                x=portfolio_labels,
                y=df['Ann. Risk (%)'],
                marker_color='#FF5252',
                opacity=0.8,
                hovertemplate='%{x}<br>Risk: %{y:.2f}%<extra></extra>'
            ),
            row=1, col=1
        )
        
        # 2. シャープ・ソルティノレシオ
        fig.add_trace(
            go.Bar(
                name='Sharpe Ratio',
                x=portfolio_labels,
                y=df['Sharpe Ratio'],
                marker_color='#2196F3',
                opacity=0.8,
                hovertemplate='%{x}<br>Sharpe: %{y:.3f}<extra></extra>'
            ),
            row=1, col=2
        )
        fig.add_trace(
            go.Bar(
                name='Sortino Ratio',
                x=portfolio_labels,
                y=df['Sortino Ratio'],
                marker_color='#9C27B0',
                opacity=0.8,
                hovertemplate='%{x}<br>Sortino: %{y:.3f}<extra></extra>'
            ),
            row=1, col=2
        )
        
        # 3. 最大ドローダウン
        dd_colors = ['#FF6B6B' if val < -15 else '#FFA726' if val < -10 else '#FFD93D' 
                     for val in df['Max DD (%)']]
        fig.add_trace(
            go.Bar(
                name='Max Drawdown',
                x=portfolio_labels,
                y=df['Max DD (%)'],
                marker_color=dd_colors,
                opacity=0.8,
                showlegend=False,
                hovertemplate='%{x}<br>Max DD: %{y:.2f}%<extra></extra>'
            ),
            row=2, col=1
        )
        
        # 4. VaR と CVaR
        fig.add_trace(
            go.Bar(
                name='VaR 95%',
                x=portfolio_labels,
                y=df['VaR 95% (%)'],
                marker_color='#FF9800',
                opacity=0.8,
                hovertemplate='%{x}<br>VaR: %{y:.2f}%<extra></extra>'
            ),
            row=2, col=2
        )
        fig.add_trace(
            go.Bar(
                name='CVaR 95%',
                x=portfolio_labels,
                y=df['CVaR 95% (%)'],
                marker_color='#F44336',
                opacity=0.8,
                hovertemplate='%{x}<br>CVaR: %{y:.2f}%<extra></extra>'
            ),
            row=2, col=2
        )
        
        # レイアウト設定
        fig.update_layout(
            title={
                'text': 'Portfolio Comparison Summary Dashboard',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'color': 'white'}
            },
            height=800,
            plot_bgcolor='rgba(50,50,50,1)',
            paper_bgcolor='rgba(43,43,43,1)',
            font=dict(color='white', size=11),
            showlegend=True,
            legend=dict(
                bgcolor='rgba(0,0,0,0.7)',
                bordercolor='white',
                borderwidth=1,
                x=1.02,
                y=0.5
            ),
            barmode='group'
        )
        
        # 軸の設定
        for i in range(1, 3):
            for j in range(1, 3):
                fig.update_xaxes(
                    showgrid=True,
                    gridcolor='rgba(255,255,255,0.2)',
                    color='white',
                    row=i, col=j
                )
                fig.update_yaxes(
                    showgrid=True,
                    gridcolor='rgba(255,255,255,0.2)',
                    color='white',
                    row=i, col=j
                )
        
        return fig
        
    except Exception as e:
        logger.error(f"Plotlyチャート作成エラー: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


# 使用例とテスト用のダミーデータ生成関数
def _generate_test_data() -> Tuple[Dict, Dict]:
    """テスト用のダミーデータを生成"""
    metrics = {
        'Portfolio A': {
            'n_positions': 5,
            'annualized_return': 0.08,
            'annualized_std': 0.15,
            'max_drawdown': -0.12,
            'var_95': -0.025,
            'cvar_95': -0.035
        },
        'Portfolio B': {
            'n_positions': 8,
            'annualized_return': 0.06,
            'annualized_std': 0.10,
            'max_drawdown': -0.08,
            'var_95': -0.018,
            'cvar_95': -0.025
        },
        'Portfolio C': {
            'n_positions': 10,
            'annualized_return': 0.10,
            'annualized_std': 0.20,
            'max_drawdown': -0.18,
            'var_95': -0.035,
            'cvar_95': -0.050
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
    print("Summary Stats Module Test")
    print("=" * 60)
    
    # テストデータ生成
    metrics, efficiency = _generate_test_data()
    
    # 1. テーブルデータ作成
    print("\n1. Creating summary table data...")
    df = create_summary_table_data(metrics, efficiency)
    print(df.to_string())
    
    # 2. Matplotlibチャート作成
    if MATPLOTLIB_AVAILABLE:
        print("\n2. Creating Matplotlib chart...")
        fig_mpl = create_summary_matplotlib_chart(metrics, efficiency)
        if fig_mpl:
            print("✓ Matplotlib chart created successfully")
            # fig_mpl.savefig('test_summary_matplotlib.png', dpi=100, facecolor='#2b2b2b')
        else:
            print("✗ Failed to create Matplotlib chart")
    
    # 3. Plotlyチャート作成
    if PLOTLY_AVAILABLE:
        print("\n3. Creating Plotly chart...")
        fig_plotly = create_summary_plotly_chart(metrics, efficiency)
        if fig_plotly:
            print("✓ Plotly chart created successfully")
            # fig_plotly.write_html('test_summary_plotly.html')
        else:
            print("✗ Failed to create Plotly chart")
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)
