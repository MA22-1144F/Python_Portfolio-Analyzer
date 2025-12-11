"""radar_chart.py"""

import numpy as np
from typing import Dict, List, Optional
import logging
import math

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


def _normalize_values(metrics: Dict[str, Dict], efficiency: Dict[str, Dict]) -> Dict[str, List[float]]:
    """レーダーチャート用に値を正規化（0-100スケール）"""
    portfolios = list(metrics.keys())
    
    # 各指標の値を収集
    returns = [metrics[name].get('annualized_return', 0) * 100 for name in portfolios]
    risks = [metrics[name].get('annualized_std', 0) * 100 for name in portfolios]
    sharpes = [efficiency.get(name, {}).get('annualized_sharpe', 0) for name in portfolios]
    sortinos = [efficiency.get(name, {}).get('sortino_ratio', 0) for name in portfolios]
    max_dds = [metrics[name].get('max_drawdown', 0) * 100 for name in portfolios]
    
    # 正規化関数
    def normalize(values, invert=False):
        """0-100スケールに正規化"""
        min_val = min(values)
        max_val = max(values)
        if max_val - min_val < 1e-10:
            return [50.0] * len(values)  # すべて同じ値の場合は中間値
        
        normalized = [(v - min_val) / (max_val - min_val) * 100 for v in values]
        
        if invert:
            # 値が小さいほど良い指標の場合は反転
            normalized = [100 - v for v in normalized]
        
        return normalized
    
    # 各指標を正規化
    norm_returns = normalize(returns, invert=False)  # 高いほど良い
    norm_risks = normalize(risks, invert=True)  # 低いほど良い
    norm_sharpes = normalize(sharpes, invert=False)  # 高いほど良い
    norm_sortinos = normalize(sortinos, invert=False)  # 高いほど良い
    norm_max_dds = normalize(max_dds, invert=True)  # 低い(負の値の絶対値が小さい)ほど良い
    
    # ポートフォリオごとにまとめる
    result = {}
    for i, name in enumerate(portfolios):
        result[name] = [
            norm_returns[i],
            norm_risks[i],
            norm_sharpes[i],
            norm_sortinos[i],
            norm_max_dds[i]
        ]
    
    return result


def create_radar_matplotlib_chart(
    metrics: Dict[str, Dict],
    efficiency: Dict[str, Dict],
    config: Optional[object] = None
) -> Optional[Figure]:
    """Matplotlibでレーダーチャートを作成"""
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
        
        # カテゴリーラベル
        categories = ['Return', 'Risk', 'Sharpe', 'Sortino', 'Max DD']
        N = len(categories)
        
        # 正規化されたデータ取得
        normalized_data = _normalize_values(metrics, efficiency)
        portfolios = list(normalized_data.keys())
        
        # 角度計算
        angles = [n / float(N) * 2 * math.pi for n in range(N)]
        angles += angles[:1]  # 閉じるために最初の角度を追加
        
        # フィギュア作成
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
        fig.patch.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)
        
        # カラーパレット
        color_palette = plt.cm.Set2(np.linspace(0, 1, len(portfolios)))
        
        # 各ポートフォリオをプロット
        for idx, (portfolio_name, values) in enumerate(normalized_data.items()):
            values_plot = values + values[:1]  # 閉じるために最初の値を追加
            
            ax.plot(angles, values_plot,
                   linewidth=2,
                   linestyle='solid',
                   label=f"#{idx+1:02d}",
                   color=color_palette[idx])
            ax.fill(angles, values_plot,
                   alpha=0.15,
                   color=color_palette[idx])
        
        # 軸設定
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, color=text_color, fontsize=11)
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(['20', '40', '60', '80', '100'],
                          color=text_color, fontsize=9, alpha=0.7)
        
        # グリッド
        ax.grid(True, color=grid_color, linestyle='--', linewidth=0.5, alpha=0.5)
        
        # 凡例
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1),
                 facecolor=bg_color, edgecolor=text_color,
                 labelcolor=text_color, fontsize=10)
        
        # タイトル
        ax.set_title('Portfolio Comprehensive Evaluation',
                    color=text_color, fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        return fig
        
    except Exception as e:
        logger.error(f"Matplotlibチャート作成エラー: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def create_radar_plotly_chart(
    metrics: Dict[str, Dict],
    efficiency: Dict[str, Dict],
    config: Optional[object] = None
) -> Optional[PlotlyFigure]:
    """Plotlyでインタラクティブなレーダーチャートを作成"""
    if not PLOTLY_AVAILABLE:
        logger.warning("Plotlyが利用できません")
        return None
    
    try:
        categories = ['Return', 'Risk', 'Sharpe', 'Sortino', 'Max DD']
        
        # 正規化されたデータ取得
        normalized_data = _normalize_values(metrics, efficiency)
        
        fig = go.Figure()
        
        for portfolio_name, values in normalized_data.items():
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=portfolio_name,
                hovertemplate=(
                    f"<b>{portfolio_name}</b><br>"
                    "%{theta}: %{r:.1f}<br>"
                    "<extra></extra>"
                )
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    color='white',
                    gridcolor='rgba(255,255,255,0.3)'
                ),
                bgcolor='rgba(50,50,50,1)',
                angularaxis=dict(
                    color='white',
                    gridcolor='rgba(255,255,255,0.3)'
                )
            ),
            paper_bgcolor='rgba(43,43,43,1)',
            font=dict(color='white'),
            title="Portfolio Comprehensive Evaluation",
            legend=dict(
                bgcolor='rgba(0,0,0,0.7)',
                bordercolor='white',
                borderwidth=1
            )
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
    print("Radar Chart Module Test")
    print("=" * 60)
    
    metrics, efficiency = _generate_test_data()
    
    if MATPLOTLIB_AVAILABLE:
        print("\n1. Creating Matplotlib chart...")
        fig_mpl = create_radar_matplotlib_chart(metrics, efficiency)
        if fig_mpl:
            print("✓ Matplotlib chart created successfully")
        else:
            print("✗ Failed to create Matplotlib chart")
    
    if PLOTLY_AVAILABLE:
        print("\n2. Creating Plotly chart...")
        fig_plotly = create_radar_plotly_chart(metrics, efficiency)
        if fig_plotly:
            print("✓ Plotly chart created successfully")
        else:
            print("✗ Failed to create Plotly chart")
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)
