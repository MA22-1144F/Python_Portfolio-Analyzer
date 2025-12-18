"""
portfolio_comparison パッケージ
ポートフォリオ比較分析機能を提供
"""

from .portfolio_metrics_calculator import (
    PortfolioMetricsCalculator,
    calculate_all_portfolios_metrics
)

from .portfolio_efficiency import (
    PortfolioEfficiency,
    calculate_all_portfolios_efficiency
)

from .portfolio_ranking import (
    PortfolioRanking,
    create_comparison_summary
)

from .summary_stats import (
    create_summary_table_data,
    create_summary_matplotlib_chart,
    create_summary_plotly_chart
)

from .scatter_plot import (
    create_scatter_matplotlib_chart,
    create_scatter_plotly_chart
)

from .heatmap import (
    create_heatmap_matplotlib_chart,
    create_heatmap_plotly_chart
)

from .radar_chart import (
    create_radar_matplotlib_chart,
    create_radar_plotly_chart
)

from .price_evolution import (
    create_price_evolution_matplotlib_chart,
    create_price_evolution_plotly_chart
)

from .actual_vs_theoretical import (
    create_actual_vs_theoretical_matplotlib_chart,
    create_actual_vs_theoretical_plotly_chart
)

from .frontier_comparison import (
    PortfolioFrontierCalculator,
    extract_portfolio_price_data,
    create_frontier_comparison_plotly_chart,
    create_frontier_comparison_matplotlib_chart
)

from .downside_deviation_frontier_comparison import (
    PortfolioDownsideDeviationFrontierCalculator,
    create_downside_deviation_frontier_comparison_plotly_chart,
    create_downside_deviation_frontier_comparison_matplotlib_chart
)

from .market_comparison import (
    PortfolioMarketComparison,
    create_market_comparison_plotly_chart,
    create_market_comparison_matplotlib_chart
)

__all__ = [
    'PortfolioMetricsCalculator',
    'calculate_all_portfolios_metrics',
    'PortfolioEfficiency',
    'calculate_all_portfolios_efficiency',
    'PortfolioRanking',
    'create_comparison_summary',
    
    'create_summary_table_data',
    'create_summary_matplotlib_chart',
    'create_summary_plotly_chart',
    
    'create_scatter_matplotlib_chart',
    'create_scatter_plotly_chart',
    
    'create_heatmap_matplotlib_chart',
    'create_heatmap_plotly_chart',
    
    'create_radar_matplotlib_chart',
    'create_radar_plotly_chart',
    
    'create_price_evolution_matplotlib_chart',
    'create_price_evolution_plotly_chart',
    
    'create_actual_vs_theoretical_matplotlib_chart',
    'create_actual_vs_theoretical_plotly_chart',

    'PortfolioFrontierCalculator',
    'extract_portfolio_price_data',
    'create_frontier_comparison_plotly_chart',
    'create_frontier_comparison_matplotlib_chart',

    'PortfolioDownsideDeviationFrontierCalculator',
    'create_downside_deviation_frontier_comparison_plotly_chart',
    'create_downside_deviation_frontier_comparison_matplotlib_chart',

    'PortfolioMarketComparison',
    'create_market_comparison_plotly_chart',
    'create_market_comparison_matplotlib_chart',
]
