"""heatmap.py"""

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


def create_heatmap_matplotlib_chart(
    metrics: Dict[str, Dict],
    efficiency: Dict[str, Dict],
    config: Optional[object] = None
) -> Optional[Figure]:
    """Matplotlibでパフォーマンスヒートマップを作成"""
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Matplotlibが利用できません")
        return None
    
    try:
        # カラー設定
        if config:
            colors = config.get_ui_colors()
            bg_color = colors.get('background', '#2b2b2b')
            text_color = colors.get('text_primary', '#ffffff')
        else:
            bg_color = '#2b2b2b'
            text_color = '#ffffff'
        
        # データマトリックス作成
        portfolios = list(metrics.keys())
        metrics_names = [
            'Ann. Return (%)',
            'Ann. Risk (%)',
            'Sharpe Ratio',
            'Sortino Ratio',
            'Max DD (%)'
        ]
        
        # ポートフォリオ名を番号に変換
        portfolio_labels = [f"#{i+1:02d}" for i in range(len(portfolios))]
        
        data_matrix = []
        raw_data_matrix = []  # 元の値を保持
        
        for portfolio_name in portfolios:
            pf_metrics = metrics[portfolio_name]
            pf_efficiency = efficiency.get(portfolio_name, {})
            
            row = [
                pf_metrics.get('annualized_return', 0) * 100,
                pf_metrics.get('annualized_std', 0) * 100,
                pf_efficiency.get('annualized_sharpe', 0),
                pf_efficiency.get('sortino_ratio', 0),
                pf_metrics.get('max_drawdown', 0) * 100
            ]
            raw_data_matrix.append(row)
            data_matrix.append(row)
        
        # 配列に変換
        data_array = np.array(data_matrix)
        raw_data_array = np.array(raw_data_matrix)
        
        # 正規化（0-1スケール）
        normalized_data = np.zeros_like(data_array)
        for j in range(data_array.shape[1]):
            col = data_array[:, j]
            min_val = np.min(col)
            max_val = np.max(col)
            if max_val - min_val > 1e-10:
                normalized_data[:, j] = (col - min_val) / (max_val - min_val)
            else:
                normalized_data[:, j] = 0.5  # すべて同じ値の場合は中間値
        
        # フィギュア作成
        fig, ax = plt.subplots(figsize=(10, 7))
        fig.patch.set_facecolor(bg_color)
        
        # ヒートマップ描画
        im = ax.imshow(normalized_data.T, cmap='RdYlGn', aspect='auto', alpha=0.8)
        
        # 軸設定
        ax.set_xticks(np.arange(len(portfolios)))
        ax.set_yticks(np.arange(len(metrics_names)))
        ax.set_xticklabels(portfolio_labels, color=text_color, fontsize=10)
        ax.set_yticklabels(metrics_names, color=text_color, fontsize=10)
        
        # グリッド
        ax.set_xticks(np.arange(len(portfolios)) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(metrics_names)) - 0.5, minor=True)
        ax.grid(which="minor", color=text_color, linestyle='-', linewidth=0.5, alpha=0.3)
        ax.tick_params(which="minor", size=0)
        
        # 値をテキストで表示
        for i in range(len(portfolios)):
            for j in range(len(metrics_names)):
                val = raw_data_array[i, j]
                text = ax.text(i, j, f'{val:.2f}',
                             ha="center", va="center",
                             color='black', fontsize=9, weight='bold')
        
        # カラーバー
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.15)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label('Normalized Score (0-1)', color=text_color, fontsize=10)
        cbar.ax.tick_params(colors=text_color)
        
        # タイトルと軸ラベル
        ax.set_title('Performance Heatmap (Normalized)',
                    color=text_color, fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Portfolio', color=text_color, fontsize=11)
        ax.set_ylabel('Metric', color=text_color, fontsize=11)
        
        ax.set_facecolor(bg_color)
        
        fig.tight_layout()
        
        return fig
        
    except Exception as e:
        logger.error(f"Matplotlibチャート作成エラー: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def create_heatmap_plotly_chart(
    metrics: Dict[str, Dict],
    efficiency: Dict[str, Dict],
    config: Optional[object] = None
) -> Optional[PlotlyFigure]: # type: ignore
    """Plotlyでインタラクティブなパフォーマンスヒートマップを作成 """
    if not PLOTLY_AVAILABLE:
        logger.warning("Plotlyが利用できません")
        return None
    
    try:
        portfolios = list(metrics.keys())
        metrics_names = [
            'Ann. Return (%)',
            'Ann. Risk (%)',
            'Sharpe Ratio',
            'Sortino Ratio',
            'Max DD (%)'
        ]
        
        # データマトリックス作成
        data_matrix = []
        for portfolio_name in portfolios:
            pf_metrics = metrics[portfolio_name]
            pf_efficiency = efficiency.get(portfolio_name, {})
            
            row = [
                pf_metrics.get('annualized_return', 0) * 100,
                pf_metrics.get('annualized_std', 0) * 100,
                pf_efficiency.get('annualized_sharpe', 0),
                pf_efficiency.get('sortino_ratio', 0),
                pf_metrics.get('max_drawdown', 0) * 100
            ]
            data_matrix.append(row)
        
        # 配列に変換
        data_array = np.array(data_matrix)
        
        # 正規化（0-1スケール）
        normalized_data = np.zeros_like(data_array)
        for j in range(data_array.shape[1]):
            col = data_array[:, j]
            min_val = np.min(col)
            max_val = np.max(col)
            if max_val - min_val > 1e-10:
                normalized_data[:, j] = (col - min_val) / (max_val - min_val)
            else:
                normalized_data[:, j] = 0.5
        
        # ヒートマップ作成
        fig = go.Figure(data=go.Heatmap(
            z=normalized_data.T,
            x=portfolios,
            y=metrics_names,
            colorscale='RdYlGn',
            text=[[f"{val:.2f}" for val in row] for row in data_array.T],
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate='%{y}<br>%{x}<br>Value: %{text}<br>Normalized: %{z:.3f}<extra></extra>',
            colorbar=dict(
                title=dict(
                    text='Normalized<br>Score',
                    font=dict(color='white')
                ),
                tickfont=dict(color='white')
            )
        ))
        
        fig.update_layout(
            title="Performance Heatmap (Normalized)",
            xaxis=dict(
                title='Portfolio',
                color='white',
                side='bottom'
            ),
            yaxis=dict(
                title='Metric',
                color='white'
            ),
            plot_bgcolor='rgba(50,50,50,1)',
            paper_bgcolor='rgba(43,43,43,1)',
            font=dict(color='white')
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
    print("Heatmap Module Test")
    print("=" * 60)
    
    metrics, efficiency = _generate_test_data()
    
    if MATPLOTLIB_AVAILABLE:
        print("\n1. Creating Matplotlib chart...")
        fig_mpl = create_heatmap_matplotlib_chart(metrics, efficiency)
        if fig_mpl:
            print("✓ Matplotlib chart created successfully")
        else:
            print("✗ Failed to create Matplotlib chart")
    
    if PLOTLY_AVAILABLE:
        print("\n2. Creating Plotly chart...")
        fig_plotly = create_heatmap_plotly_chart(metrics, efficiency)
        if fig_plotly:
            print("✓ Plotly chart created successfully")
        else:
            print("✗ Failed to create Plotly chart")
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)
