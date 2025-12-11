"""portfolio_efficiency.py"""

import numpy as np
from typing import Dict, Optional
import logging


class PortfolioEfficiency:
    """ポートフォリオ効率性指標計算クラス"""
    
    def __init__(self, config=None):
        self.logger = logging.getLogger(__name__)
        
        if config is None:
            from config.app_config import AppConfig
            config = AppConfig()
        self.config = config
    
    def calculate_efficiency_metrics(
        self,
        portfolio_metrics: Dict,
        market_metrics: Optional[Dict] = None
    ) -> Dict:
        """効率性指標を計算"""
        try:
            # 基本効率性指標
            efficiency = {}
            efficiency.update(self._calculate_risk_adjusted_returns(portfolio_metrics))
            
            # 市場相対効率性指標（オプション）
            if market_metrics:
                efficiency.update(self._calculate_market_relative_efficiency(
                    portfolio_metrics, market_metrics
                ))
            
            efficiency['success'] = True
            return efficiency
            
        except Exception as e:
            self.logger.error(f"効率性指標計算エラー: {e}")
            return {'success': False, 'error': str(e)}
    
    def _calculate_risk_adjusted_returns(self, portfolio_metrics: Dict) -> Dict:
        """リスク調整リターン指標を計算"""
        try:
            # 必要な指標を取得
            expected_return = portfolio_metrics.get('expected_return', np.nan)
            excess_return = portfolio_metrics.get('excess_return', np.nan)
            portfolio_std = portfolio_metrics.get('portfolio_std', np.nan)
            downside_deviation = portfolio_metrics.get('downside_deviation', np.nan)
            max_drawdown = portfolio_metrics.get('max_drawdown', np.nan)
            annualized_return = portfolio_metrics.get('annualized_return', np.nan)
            annualized_std = portfolio_metrics.get('annualized_std', np.nan)
            
            # シャープレシオ（期間ベース）
            sharpe_ratio = (
                excess_return / portfolio_std 
                if portfolio_std > 0 and not np.isnan(excess_return) 
                else np.nan
            )
            
            # 年率換算シャープレシオ
            annualized_sharpe = (
                (annualized_return - portfolio_metrics.get('risk_free_rate', 0) * 252) / annualized_std
                if annualized_std > 0 and not np.isnan(annualized_return)
                else np.nan
            )
            
            # ソルティノレシオ
            sortino_ratio = (
                excess_return / downside_deviation
                if downside_deviation > 0 and not np.isnan(excess_return)
                else np.nan
            )
            
            # カルマーレシオ
            calmar_ratio = (
                annualized_return / abs(max_drawdown)
                if max_drawdown < 0 and not np.isnan(annualized_return)
                else np.nan
            )
            
            # リターン・リスク比率
            return_risk_ratio = (
                expected_return / portfolio_std
                if portfolio_std > 0 and not np.isnan(expected_return)
                else np.nan
            )
            
            return {
                'sharpe_ratio': sharpe_ratio,
                'annualized_sharpe': annualized_sharpe,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio,
                'return_risk_ratio': return_risk_ratio
            }
            
        except Exception as e:
            self.logger.error(f"リスク調整リターン計算エラー: {e}")
            return {}
    
    def _calculate_market_relative_efficiency(
        self,
        portfolio_metrics: Dict,
        market_metrics: Dict
    ) -> Dict:
        """市場相対効率性指標を計算"""
        try:
            portfolio_return = portfolio_metrics.get('expected_return', np.nan)
            market_return = market_metrics.get('expected_return', np.nan)
            
            # 超過リターン（市場に対する）
            excess_over_market = portfolio_return - market_return
            
            # トラッキングエラーの計算は別途必要
            # ここでは簡易的に計算
            tracking_error = portfolio_metrics.get('portfolio_std', np.nan)
            
            # インフォメーション比率
            information_ratio = (
                excess_over_market / tracking_error
                if tracking_error > 0 and not np.isnan(excess_over_market)
                else np.nan
            )
            
            return {
                'excess_over_market': excess_over_market,
                'tracking_error': tracking_error,
                'information_ratio': information_ratio
            }
            
        except Exception as e:
            self.logger.error(f"市場相対効率性計算エラー: {e}")
            return {}
    def _calculate_sharpe_ratio(self, portfolio_metrics: Dict) -> Dict:
        """シャープレシオを計算"""
        try:
            expected_return = portfolio_metrics.get('expected_return', np.nan)
            portfolio_std = portfolio_metrics.get('portfolio_std', np.nan)
            risk_free_rate = portfolio_metrics.get('risk_free_rate', 0.0)
            
            # シャープレシオ = (E[Rp] - Rf) / σp
            sharpe_ratio = (
                (expected_return - risk_free_rate) / portfolio_std
                if portfolio_std > 0 and not np.isnan(expected_return)
                else np.nan
            )
            
            # 年率換算シャープレシオ
            annualized_return = portfolio_metrics.get('annualized_return', np.nan)
            annualized_std = portfolio_metrics.get('annualized_std', np.nan)
            annualized_sharpe = (
                (annualized_return - risk_free_rate * 252) / annualized_std
                if annualized_std > 0 and not np.isnan(annualized_return)
                else np.nan
            )
            
            return {
                'sharpe_ratio': sharpe_ratio,
                'annualized_sharpe': annualized_sharpe
            }
            
        except Exception as e:
            self.logger.error(f"シャープレシオ計算エラー: {e}")
            return {
                'sharpe_ratio': np.nan,
                'annualized_sharpe': np.nan
            }


    def _calculate_sortino_ratio(self, portfolio_metrics: Dict) -> Dict:
        """ソルティノレシオを計算"""
        try:
            expected_return = portfolio_metrics.get('expected_return', np.nan)
            downside_deviation = portfolio_metrics.get('downside_deviation', np.nan)
            risk_free_rate = portfolio_metrics.get('risk_free_rate', 0.0)
            
            # ソルティノレシオ = (E[Rp] - Rf) / 下方偏差
            sortino_ratio = (
                (expected_return - risk_free_rate) / downside_deviation
                if downside_deviation > 0 and not np.isnan(expected_return)
                else np.nan
            )
            
            return {
                'sortino_ratio': sortino_ratio
            }
            
        except Exception as e:
            self.logger.error(f"ソルティノレシオ計算エラー: {e}")
            return {
                'sortino_ratio': np.nan
            }


    def _calculate_calmar_ratio(self, portfolio_metrics: Dict) -> Dict:
        """カルマーレシオを計算"""
        try:
            annualized_return = portfolio_metrics.get('annualized_return', np.nan)
            max_drawdown = portfolio_metrics.get('max_drawdown', np.nan)
            
            # カルマーレシオ = 年率リターン / |最大ドローダウン|
            calmar_ratio = (
                annualized_return / abs(max_drawdown)
                if max_drawdown != 0 and not np.isnan(annualized_return) and not np.isnan(max_drawdown)
                else np.nan
            )
            
            return {
                'calmar_ratio': calmar_ratio
            }
            
        except Exception as e:
            self.logger.error(f"カルマーレシオ計算エラー: {e}")
            return {
                'calmar_ratio': np.nan
            }


    def _calculate_return_risk_ratio(self, portfolio_metrics: Dict) -> Dict:
        """リターン/リスク比率を計算"""
        try:
            expected_return = portfolio_metrics.get('expected_return', np.nan)
            portfolio_std = portfolio_metrics.get('portfolio_std', np.nan)
            
            # リターン/リスク比率
            return_risk_ratio = (
                expected_return / portfolio_std
                if portfolio_std > 0 and not np.isnan(expected_return)
                else np.nan
            )
            
            return {
                'return_risk_ratio': return_risk_ratio
            }
            
        except Exception as e:
            self.logger.error(f"リターン/リスク比率計算エラー: {e}")
            return {
                'return_risk_ratio': np.nan
            }


    def calculate_efficiency_metrics(
        self,
        portfolio_metrics: Dict,
        market_metrics: Optional[Dict] = None
    ) -> Dict:
        """効率性指標を総合的に計算"""
        try:
            if not portfolio_metrics.get('success', False):
                return {'success': False, 'error': 'ポートフォリオ統計量が無効'}
            
            efficiency = {}
            
            # シャープレシオ
            sharpe = self._calculate_sharpe_ratio(portfolio_metrics)
            efficiency.update(sharpe)
            
            # ソルティノレシオ
            sortino = self._calculate_sortino_ratio(portfolio_metrics)
            efficiency.update(sortino)
            
            # カルマーレシオ
            calmar = self._calculate_calmar_ratio(portfolio_metrics)
            efficiency.update(calmar)
            
            # リターン/リスク比率
            return_risk = self._calculate_return_risk_ratio(portfolio_metrics)
            efficiency.update(return_risk)
            
            # 市場相対指標（オプション）
            if market_metrics:
                market_relative = self._calculate_market_relative_efficiency(
                    portfolio_metrics, market_metrics
                )
                efficiency.update(market_relative)
            
            efficiency['success'] = True
            return efficiency
            
        except Exception as e:
            self.logger.error(f"効率性指標計算エラー: {e}")
            return {'success': False, 'error': str(e)}

def calculate_all_portfolios_efficiency(
    all_metrics: Dict[str, Dict],
    market_metrics: Optional[Dict] = None
) -> Dict[str, Dict]:
    """複数ポートフォリオの効率性指標を一括計算"""
    calculator = PortfolioEfficiency()
    
    all_efficiency = {}
    
    for portfolio_name, metrics in all_metrics.items():
        if not metrics.get('success', False):
            continue
        
        efficiency = calculator.calculate_efficiency_metrics(
            metrics, market_metrics
        )
        
        if efficiency.get('success', False):
            all_efficiency[portfolio_name] = efficiency
    
    return all_efficiency