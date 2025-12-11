"""portfolio_ranking.py"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging


class PortfolioRanking:
    """ポートフォリオランキング・総合評価クラス"""
    
    def __init__(self, config=None):
        self.logger = logging.getLogger(__name__)
        
        if config is None:
            from config.app_config import AppConfig
            config = AppConfig()
        self.config = config
    
    def create_ranking(
        self,
        all_metrics: Dict[str, Dict],
        all_efficiency: Dict[str, Dict]
    ) -> Dict:
        """ランキングを作成"""
        try:
            # データフレームの作成
            df = self._create_dataframe(all_metrics, all_efficiency)
            
            if df.empty:
                return {'success': False, 'error': 'データなし'}
            
            # 単一指標ランキング
            rankings = self._calculate_single_metric_rankings(df)
            
            # 総合スコア
            composite_score = self._calculate_composite_score(df, rankings)
            
            # パレート最適性
            pareto_optimal = self._identify_pareto_optimal(df)
            
            return {
                'success': True,
                'dataframe': df,
                'rankings': rankings,
                'composite_score': composite_score,
                'pareto_optimal': pareto_optimal
            }
            
        except Exception as e:
            self.logger.error(f"ランキング作成エラー: {e}")
            return {'success': False, 'error': str(e)}
    
    def _create_dataframe(
        self,
        all_metrics: Dict[str, Dict],
        all_efficiency: Dict[str, Dict]
    ) -> pd.DataFrame:
        """統合データフレームを作成"""
        try:
            data = []
            
            for portfolio_name in all_metrics.keys():
                if portfolio_name not in all_efficiency:
                    continue
                
                metrics = all_metrics[portfolio_name]
                efficiency = all_efficiency[portfolio_name]
                
                row = {
                    'portfolio_name': portfolio_name,
                    # リターン指標
                    'expected_return': metrics.get('expected_return', np.nan),
                    'excess_return': metrics.get('excess_return', np.nan),
                    'annualized_return': metrics.get('annualized_return', np.nan),
                    # リスク指標
                    'portfolio_std': metrics.get('portfolio_std', np.nan),
                    'annualized_std': metrics.get('annualized_std', np.nan),
                    'downside_deviation': metrics.get('downside_deviation', np.nan),
                    'max_drawdown': metrics.get('max_drawdown', np.nan),
                    # VaR/CVaR
                    'var_95': metrics.get('var_95', np.nan),
                    'cvar_95': metrics.get('cvar_95', np.nan),
                    # 分散化
                    'diversification_ratio': metrics.get('diversification_ratio', np.nan),
                    'equivalent_n_assets': metrics.get('equivalent_n_assets', np.nan),
                    # 効率性指標
                    'sharpe_ratio': efficiency.get('sharpe_ratio', np.nan),
                    'annualized_sharpe': efficiency.get('annualized_sharpe', np.nan),
                    'sortino_ratio': efficiency.get('sortino_ratio', np.nan),
                    'calmar_ratio': efficiency.get('calmar_ratio', np.nan),
                    'return_risk_ratio': efficiency.get('return_risk_ratio', np.nan),
                    # メタデータ
                    'n_positions': metrics.get('n_positions', 0),
                    'total_weight': metrics.get('total_weight', 0.0)
                }
                
                data.append(row)
            
            df = pd.DataFrame(data)
            df = df.set_index('portfolio_name')
            
            return df
            
        except Exception as e:
            self.logger.error(f"データフレーム作成エラー: {e}")
            return pd.DataFrame()
    
    def _calculate_single_metric_rankings(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """単一指標でのランキングを計算"""
        try:
            rankings = {}
            
            # リターン系（降順: 高い方が良い）
            for metric in ['expected_return', 'annualized_return', 'excess_return']:
                if metric in df.columns:
                    rankings[f'{metric}_rank'] = df[metric].rank(ascending=False, method='min')
            
            # リスク系（昇順: 低い方が良い）
            for metric in ['portfolio_std', 'annualized_std', 'downside_deviation']:
                if metric in df.columns:
                    rankings[f'{metric}_rank'] = df[metric].rank(ascending=True, method='min')
            
            # 最大DD（昇順: 小さい負の値が良い = 絶対値が小さい方が良い）
            if 'max_drawdown' in df.columns:
                rankings['max_drawdown_rank'] = df['max_drawdown'].abs().rank(ascending=True, method='min')
            
            # VaR/CVaR（昇順: 小さい負の値が良い = 絶対値が小さい方が良い）
            for metric in ['var_95', 'cvar_95']:
                if metric in df.columns:
                    rankings[f'{metric}_rank'] = df[metric].abs().rank(ascending=True, method='min')
            
            # 効率性指標（降順: 高い方が良い）
            for metric in ['sharpe_ratio', 'annualized_sharpe', 'sortino_ratio', 'calmar_ratio']:
                if metric in df.columns:
                    rankings[f'{metric}_rank'] = df[metric].rank(ascending=False, method='min')
            
            # 分散化指標（降順: 高い方が良い）
            if 'diversification_ratio' in df.columns:
                rankings['diversification_ratio_rank'] = df['diversification_ratio'].rank(
                    ascending=False, method='min'
                )
            
            return rankings
            
        except Exception as e:
            self.logger.error(f"単一指標ランキング計算エラー: {e}")
            return {}
    
    def _calculate_composite_score(
        self,
        df: pd.DataFrame,
        rankings: Dict[str, pd.Series]
    ) -> pd.Series:
        """総合スコアを計算"""
        try:
            # 主要指標のランクを抽出
            key_metrics = [
                'annualized_return_rank',
                'annualized_std_rank',
                'max_drawdown_rank',
                'annualized_sharpe_rank'
            ]
            
            # 利用可能なランクのみを使用
            available_ranks = []
            for metric in key_metrics:
                if metric in rankings:
                    available_ranks.append(rankings[metric])
            
            if not available_ranks:
                return pd.Series(index=df.index, dtype=float)
            
            # 平均ランクを計算（低い方が良い）
            rank_df = pd.DataFrame(available_ranks).T
            composite_rank = rank_df.mean(axis=1)
            
            # スコア化（100点満点，高い方が良い）
            n_portfolios = len(df)
            composite_score = 100 * (1 - (composite_rank - 1) / (n_portfolios - 1)) if n_portfolios > 1 else 100
            
            return composite_score
            
        except Exception as e:
            self.logger.error(f"総合スコア計算エラー: {e}")
            return pd.Series(index=df.index, dtype=float)
    
    def _identify_pareto_optimal(self, df: pd.DataFrame) -> List[str]:
        """パレート最適なポートフォリオを特定"""
        try:
            # リターンとリスクの2軸で判定
            if 'expected_return' not in df.columns or 'portfolio_std' not in df.columns:
                return []
            
            returns = df['expected_return'].values
            risks = df['portfolio_std'].values
            names = df.index.tolist()
            
            pareto_optimal = []
            
            for i, (ret_i, risk_i, name_i) in enumerate(zip(returns, risks, names)):
                is_dominated = False
                
                # 他のポートフォリオと比較
                for j, (ret_j, risk_j) in enumerate(zip(returns, risks)):
                    if i == j:
                        continue
                    
                    # j が i を支配するか
                    # (リターンが同等以上 かつ リスクが同等以下，少なくとも一方で厳密に優れている)
                    if ret_j >= ret_i and risk_j <= risk_i:
                        if ret_j > ret_i or risk_j < risk_i:
                            is_dominated = True
                            break
                
                if not is_dominated:
                    pareto_optimal.append(name_i)
            
            return pareto_optimal
            
        except Exception as e:
            self.logger.error(f"パレート最適判定エラー: {e}")
            return []
    
    def get_top_portfolios(
        self,
        ranking_result: Dict,
        metric: str,
        top_n: int = 3
    ) -> List[Tuple[str, float]]:
        """指定指標でのトップNを取得"""
        try:
            df = ranking_result.get('dataframe')
            if df is None or df.empty:
                return []
            
            if metric not in df.columns:
                return []
            
            # リスク系指標は昇順，それ以外は降順
            risk_metrics = ['portfolio_std', 'annualized_std', 'downside_deviation', 
                          'max_drawdown', 'var_95', 'cvar_95']
            
            ascending = metric in risk_metrics
            
            # 最大DDとVaRは絶対値でソート
            if metric in ['max_drawdown', 'var_95', 'cvar_95']:
                sorted_df = df.sort_values(by=metric, key=lambda x: x.abs(), ascending=True)
            else:
                sorted_df = df.sort_values(by=metric, ascending=ascending)
            
            top_portfolios = []
            for idx in sorted_df.head(top_n).index:
                top_portfolios.append((idx, sorted_df.loc[idx, metric]))
            
            return top_portfolios
            
        except Exception as e:
            self.logger.error(f"トップポートフォリオ取得エラー: {e}")
            return []


def create_comparison_summary(
    all_metrics: Dict[str, Dict],
    all_efficiency: Dict[str, Dict]
) -> Dict:
    """比較サマリーを作成"""
    ranker = PortfolioRanking()
    ranking_result = ranker.create_ranking(all_metrics, all_efficiency)
    
    if not ranking_result.get('success', False):
        return ranking_result
    
    # 各指標でのトップ3を取得
    top_by_return = ranker.get_top_portfolios(ranking_result, 'annualized_return', 3)
    top_by_sharpe = ranker.get_top_portfolios(ranking_result, 'annualized_sharpe', 3)
    top_by_sortino = ranker.get_top_portfolios(ranking_result, 'sortino_ratio', 3)
    best_risk = ranker.get_top_portfolios(ranking_result, 'annualized_std', 1)
    
    summary = {
        'success': True,
        'ranking_result': ranking_result,
        'top_by_return': top_by_return,
        'top_by_sharpe': top_by_sharpe,
        'top_by_sortino': top_by_sortino,
        'best_risk': best_risk,
        'pareto_optimal': ranking_result.get('pareto_optimal', [])
    }
    
    return summary