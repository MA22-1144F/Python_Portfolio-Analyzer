"""price_evolution.py"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List
import logging

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    import matplotlib.dates as mdates
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


def _calculate_portfolio_returns(
    portfolio,
    price_data: Dict[str, pd.Series],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Optional[pd.Series]:
    """ポートフォリオの累積リターンを計算"""
    try:
        # ポートフォリオのウェイトを取得
        weights = {}
        total_weight = 0.0
        
        for position in portfolio.positions:
            symbol = position.asset.symbol
            if symbol in price_data:
                weights[symbol] = position.weight
                total_weight += position.weight
        
        if total_weight == 0 or not weights:
            logger.warning(f"有効なウェイトがありません: {portfolio.name}")
            return None
        
        # ウェイトを正規化
        for symbol in weights:
            weights[symbol] /= total_weight
        
        # 共通の日付インデックスを取得
        all_dates = None
        for symbol in weights.keys():
            if all_dates is None:
                all_dates = price_data[symbol].index
            else:
                all_dates = all_dates.intersection(price_data[symbol].index)
        
        if all_dates is None or len(all_dates) == 0:
            logger.warning(f"共通の日付が見つかりません: {portfolio.name}")
            return None
        
        # 日付範囲でフィルタ
        if start_date:
            all_dates = all_dates[all_dates >= pd.to_datetime(start_date)]
        if end_date:
            all_dates = all_dates[all_dates <= pd.to_datetime(end_date)]
        
        if len(all_dates) == 0:
            return None
        
        # ポートフォリオリターンを計算
        portfolio_returns = pd.Series(0.0, index=all_dates)
        
        for symbol, weight in weights.items():
            series = price_data[symbol].loc[all_dates]
            returns = series.pct_change().fillna(0)
            portfolio_returns += weight * returns
        
        # 累積リターンを計算（初期値=1.0）
        cumulative_returns = (1 + portfolio_returns).cumprod()
        
        return cumulative_returns
        
    except Exception as e:
        logger.error(f"ポートフォリオリターン計算エラー: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def create_price_evolution_matplotlib_chart(
    portfolios: List,
    price_data: Dict[str, pd.Series],
    config: Optional[object] = None
) -> Optional[Figure]:
    """Matplotlibで価格推移チャートを作成"""
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
        
        # フィギュア作成
        fig, ax = plt.subplots(figsize=(12, 7))
        fig.patch.set_facecolor(bg_color)
        
        # カラーパレット
        color_palette = plt.cm.tab10(np.linspace(0, 1, len(portfolios)))
        
        # 各ポートフォリオの累積リターンをプロット
        has_data = False
        for idx, portfolio in enumerate(portfolios):
            cumulative_returns = _calculate_portfolio_returns(portfolio, price_data)
            
            if cumulative_returns is not None and len(cumulative_returns) > 0:
                # 初期値を100にスケール
                scaled_returns = cumulative_returns * 100
                
                ax.plot(cumulative_returns.index, scaled_returns,
                       linewidth=2,
                       label=f"#{idx+1:02d}",
                       color=color_palette[idx],
                       alpha=0.8)
                has_data = True
        
        if not has_data:
            logger.warning("プロット可能なデータがありません")
            return None
        
        # 軸設定
        ax.set_xlabel('Date', color=text_color, fontsize=11)
        ax.set_ylabel('Cumulative Value (Initial=100)', color=text_color, fontsize=11)
        ax.set_title('Portfolio Price Evolution',
                    color=text_color, fontsize=14, fontweight='bold', pad=15)
        
        ax.set_facecolor(bg_color)
        ax.tick_params(colors=text_color)
        ax.grid(True, alpha=0.3, color=grid_color, linestyle='--')
        
        # 基準線（初期値=100）
        ax.axhline(y=100, color=text_color, linewidth=1, linestyle=':', alpha=0.5)
        
        # 日付フォーマット
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)
        
        # 凡例
        ax.legend(facecolor=bg_color, edgecolor=text_color,
                 labelcolor=text_color, fontsize=10,
                 loc='best')
        
        plt.tight_layout()
        
        return fig
        
    except Exception as e:
        logger.error(f"Matplotlibチャート作成エラー: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def create_price_evolution_plotly_chart(
    portfolios: List,
    price_data: Dict[str, pd.Series],
    config: Optional[object] = None
) -> Optional[PlotlyFigure]:
    """Plotlyでインタラクティブな価格推移チャートを作成"""
    if not PLOTLY_AVAILABLE:
        logger.warning("Plotlyが利用できません")
        return None
    
    try:
        fig = go.Figure()
        
        # 各ポートフォリオの累積リターンをプロット
        has_data = False
        for idx, portfolio in enumerate(portfolios):
            cumulative_returns = _calculate_portfolio_returns(portfolio, price_data)
            
            if cumulative_returns is not None and len(cumulative_returns) > 0:
                # 初期値を100にスケール
                scaled_returns = cumulative_returns * 100
                
                fig.add_trace(go.Scatter(
                    x=cumulative_returns.index,
                    y=scaled_returns,
                    mode='lines',
                    name=portfolio.name,
                    line=dict(width=2),
                    hovertemplate=(
                        f"<b>{portfolio.name}</b><br>"
                        "Date: %{x|%Y-%m-%d}<br>"
                        "Value: %{y:.2f}<br>"
                        "<extra></extra>"
                    )
                ))
                has_data = True
        
        if not has_data:
            logger.warning("プロット可能なデータがありません")
            return None
        
        # 基準線（初期値=100）
        fig.add_hline(y=100, line_dash="dot", line_color="white",
                     opacity=0.5, annotation_text="Initial Value")
        
        fig.update_layout(
            title="Portfolio Price Evolution",
            xaxis=dict(
                title='Date',
                color='white',
                gridcolor='rgba(255,255,255,0.2)',
                tickformat='%Y-%m'
            ),
            yaxis=dict(
                title='Cumulative Value (Initial=100)',
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
            hovermode='x unified'
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Plotlyチャート作成エラー: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

if __name__ == '__main__':
    print("=" * 60)
    print("Price Evolution Module")
    print("=" * 60)
    print("このモジュールをテストするには，実際のPortfolioオブジェクトと")
    print("price_dataが必要です．")
    print("=" * 60)
