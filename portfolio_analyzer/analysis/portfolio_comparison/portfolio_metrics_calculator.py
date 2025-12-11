"""portfolio_metrics_calculator.py"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

from data.portfolio import Portfolio


class PortfolioMetricsCalculator:
    """ポートフォリオ基本統計量計算クラス"""
    
    def __init__(self, config=None):
        self.logger = logging.getLogger(__name__)
        
        if config is None:
            from config.app_config import AppConfig
            config = AppConfig()
        self.config = config
    
    def calculate_portfolio_metrics(
        self,
        portfolio: Portfolio,
        returns_data: Dict[str, pd.Series],
        covariance_matrix: pd.DataFrame,
        risk_free_rate: float = 0.0
    ) -> Dict:
        """ポートフォリオの基本統計量を計算"""
        try:
            # ポートフォリオに含まれる資産のシンボルを抽出
            portfolio_symbols = [pos.asset.symbol for pos in portfolio.positions]
            
            # 利用可能な資産のみをフィルタリング
            available_symbols = [s for s in portfolio_symbols if s in returns_data]
            
            if not available_symbols:
                return self._create_empty_result(f"ポートフォリオ {portfolio.name} の資産がデータに含まれていません")
            
            if len(available_symbols) < len(portfolio_symbols):
                missing = set(portfolio_symbols) - set(available_symbols)
                self.logger.warning(f"ポートフォリオ {portfolio.name}: {len(missing)}個の資産のデータが欠落: {missing}")
            
            # このポートフォリオの資産のみのリターンデータを抽出
            portfolio_returns_data = {s: returns_data[s] for s in available_symbols}
            
            # ウエイトベクトルの構築（利用可能な資産のみ）
            weights = self._build_weight_vector(portfolio, available_symbols)
            
            if weights is None or len(weights) == 0:
                return self._create_empty_result(f"ポートフォリオ {portfolio.name} のウエイト構築に失敗")
            
            # ウエイトの正規化（欠落資産がある場合）
            weights = weights / weights.sum()
            
            # このポートフォリオの資産間の共通日付を見つける
            portfolio_returns_df = self._create_common_dataframe(portfolio_returns_data)
            
            if portfolio_returns_df.empty:
                return self._create_empty_result(f"ポートフォリオ {portfolio.name} の資産間で共通日付が見つかりません")
            
            self.logger.info(f"ポートフォリオ {portfolio.name}: {len(portfolio_returns_df)}日分の共通データで分析")
            
            # ポートフォリオリターンの計算
            portfolio_returns = (portfolio_returns_df * weights).sum(axis=1)
            
            if portfolio_returns is None or len(portfolio_returns) == 0:
                return self._create_empty_result(f"ポートフォリオ {portfolio.name} のリターン計算に失敗")
            
            # このポートフォリオの共分散行列を計算
            portfolio_covariance_matrix = portfolio_returns_df.cov()
            
            # 基本統計量の計算
            metrics = {}
            
            # リターン指標
            metrics.update(self._calculate_return_metrics(
                portfolio_returns, risk_free_rate
            ))
            
            # リスク指標
            metrics.update(self._calculate_risk_metrics(
                weights, portfolio_covariance_matrix, portfolio_returns, risk_free_rate
            ))
            
            # エクストリームリスク
            metrics.update(self._calculate_extreme_risk_metrics(
                portfolio_returns, risk_free_rate
            ))
            
            # 分散化効果
            metrics.update(self._calculate_diversification_effect(
                weights, portfolio_returns_data, portfolio_covariance_matrix
            ))
            
            # メタデータ
            metrics['portfolio_name'] = portfolio.name
            metrics['n_positions'] = len(portfolio.positions)
            metrics['n_available_positions'] = len(available_symbols)
            metrics['total_weight'] = portfolio.total_weight
            metrics['data_points'] = len(portfolio_returns)
            metrics['risk_free_rate'] = risk_free_rate
            metrics['success'] = True
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"ポートフォリオ {portfolio.name} の統計量計算エラー: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return self._create_empty_result(str(e))
    
    def _create_common_dataframe(
        self,
        returns_data: Dict[str, pd.Series]
    ) -> pd.DataFrame:
        """共通日付でDataFrameを作成"""
        try:
            if not returns_data:
                self.logger.warning("returns_dataが空です")
                return pd.DataFrame()
            
            # シンボルをソート
            symbols = sorted(returns_data.keys())
            
            self.logger.info(f"=== 共通日付DataFrame作成開始 ===")
            self.logger.info(f"対象資産数: {len(symbols)}")
            
            # 各資産のデータ数を診断
            data_counts = [len(returns_data[s]) for s in symbols]
            self.logger.info(f"各資産のデータ数: {data_counts}")
            
            # price_series.pyと同じロジック：順次共通日付を見つける
            common_dataframe_list = []
            
            for symbol in symbols:
                series = returns_data[symbol]
                
                if not common_dataframe_list:
                    # 最初の資産
                    common_dataframe_list.append(series.copy())
                    self.logger.info(f"{symbol}（1番目）: {len(series)}日分")
                else:
                    # 既存のデータと共通する日付のみを保持
                    common_dates = common_dataframe_list[0].index.intersection(series.index)
                    
                    if len(common_dates) == 0:
                        # 共通日付がない場合の詳細診断
                        self.logger.error(f"{symbol}: 共通日付が0件！")
                        self.logger.error(f"  既存データの日付範囲: {common_dataframe_list[0].index.min()} ~ {common_dataframe_list[0].index.max()}")
                        self.logger.error(f"  {symbol}の日付範囲: {series.index.min()} ~ {series.index.max()}")
                        
                        # 日付の型を確認
                        self.logger.error(f"  既存データのindex型: {type(common_dataframe_list[0].index)}")
                        self.logger.error(f"  {symbol}のindex型: {type(series.index)}")
                        
                        # 最初の数個の日付を比較
                        self.logger.error(f"  既存データの最初の5日: {list(common_dataframe_list[0].index[:5])}")
                        self.logger.error(f"  {symbol}の最初の5日: {list(series.index[:5])}")
                        
                        # 共通日付が見つからない場合は空のDataFrameを返す
                        return pd.DataFrame()
                    
                    # 既存のDataFrame群を共通日付に絞り込む
                    common_dataframe_list = [df.reindex(common_dates) for df in common_dataframe_list]
                    # 新しいseriesも共通日付に絞り込んで追加
                    common_dataframe_list.append(series.reindex(common_dates))
                    
                    self.logger.info(f"{symbol}（{len(common_dataframe_list)}番目）: 共通日付 {len(common_dates)}日")
            
            # 最終的なDataFrameを作成
            if common_dataframe_list:
                common_df = pd.concat(common_dataframe_list, axis=1)
                common_df.columns = symbols
                
                # NaN値の確認と除去
                nan_count = common_df.isnull().sum().sum()
                if nan_count > 0:
                    self.logger.warning(f"NaN値が {nan_count} 個見つかりました - 除去します")
                    common_df = common_df.dropna()
                
                self.logger.info(f"=== 共通DataFrame作成完了 ===")
                self.logger.info(f"最終形状: {common_df.shape}")
                self.logger.info(f"期間: {common_df.index.min()} ~ {common_df.index.max()}")
                
                return common_df
            else:
                self.logger.error("common_dataframe_listが空です")
                return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"共通DataFrame作成エラー: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return pd.DataFrame()

    def _build_weight_vector(
        self,
        portfolio: Portfolio,
        available_symbols: List[str]
    ) -> Optional[np.ndarray]:
        """ウエイトベクトルを構築"""
        try:
            weights_dict = {}
            
            for position in portfolio.positions:
                symbol = position.asset.symbol
                if symbol in available_symbols:
                    # パーセントを小数に変換
                    weights_dict[symbol] = position.weight / 100.0
            
            if not weights_dict:
                return None
            
            # ソートして配列化（available_symbolsの順序に合わせる）
            symbols = sorted(weights_dict.keys())
            weights = np.array([weights_dict[s] for s in symbols])
            
            return weights
            
        except Exception as e:
            self.logger.error(f"ウエイトベクトル構築エラー: {e}")
            return None
    
    def _extract_portfolio_covariance(
        self,
        full_covariance_matrix: pd.DataFrame,
        portfolio_symbols: List[str]
    ) -> pd.DataFrame:
        """ポートフォリオに含まれる資産のみの共分散行列を抽出"""
        try:
            if full_covariance_matrix.empty:
                self.logger.error("共分散行列が空です")
                return pd.DataFrame()
            
            # ポートフォリオに含まれる資産のみを抽出
            available_symbols = [s for s in portfolio_symbols if s in full_covariance_matrix.index]
            
            if len(available_symbols) != len(portfolio_symbols):
                missing = set(portfolio_symbols) - set(available_symbols)
                self.logger.warning(f"共分散行列に存在しない資産: {missing}")
            
            if not available_symbols:
                self.logger.error("共分散行列に有効な資産が存在しません")
                return pd.DataFrame()
            
            # 行と列の両方から該当資産のみを抽出
            portfolio_cov = full_covariance_matrix.loc[available_symbols, available_symbols]
            
            self.logger.info(f"ポートフォリオ共分散行列を抽出: {portfolio_cov.shape}")
            
            return portfolio_cov
            
        except Exception as e:
            self.logger.error(f"共分散行列抽出エラー: {e}", exc_info=True)
            return pd.DataFrame()
    
    def _calculate_portfolio_returns(
        self,
        returns_data: Dict[str, pd.Series],
        weights: np.ndarray,
        symbols: List[str]
    ) -> Optional[pd.Series]:
        """ポートフォリオリターンを計算"""
        try:
            # 各資産のリターンデータの期間を確認
            self.logger.info("=== ポートフォリオリターン計算 ===")
            for symbol in symbols:
                data = returns_data[symbol]
                self.logger.info(f"{symbol}: {len(data)}点, {data.index.min()} ~ {data.index.max()}")
            
            # 共通日付を見つける
            common_dates = returns_data[symbols[0]].index
            for symbol in symbols[1:]:
                common_dates = common_dates.intersection(returns_data[symbol].index)
            
            self.logger.info(f"共通日付数: {len(common_dates)}")
            
            if len(common_dates) == 0:
                self.logger.warning("共通日付が存在しません")
                self.logger.warning("各資産のデータ期間が重複していない可能性があります")
                # 全体の期間を確認
                all_dates = set()
                for symbol in symbols:
                    all_dates.update(returns_data[symbol].index)
                self.logger.warning(f"全体の日付範囲: {min(all_dates)} ~ {max(all_dates)}")
                return None
            
            # 共通日付でDataFrameを構築
            returns_df = pd.DataFrame({
                symbol: returns_data[symbol].loc[common_dates]
                for symbol in symbols
            })
            
            self.logger.info(f"共通日付でのデータフレーム構築: {len(returns_df)} 行, {len(returns_df.columns)} 列")
            
            # NaN値がないことを確認
            if returns_df.isnull().any().any():
                self.logger.warning("NaN値が含まれています - 除去します")
                returns_df = returns_df.dropna()
            
            # DataFrameが空でないかチェック
            if returns_df.empty:
                self.logger.warning("共通日付のリターンデータが空です")
                return None
            
            self.logger.info(f"最終的なリターンデータ: {len(returns_df)} 行")
            self.logger.info(f"期間: {returns_df.index.min()} ~ {returns_df.index.max()}")
            
            # ポートフォリオリターンの計算
            # Rp = Σ(wi × Ri)
            portfolio_returns = (returns_df * weights).sum(axis=1)
            
            self.logger.info(f"ポートフォリオリターン計算完了: {len(portfolio_returns)}点")
            
            return portfolio_returns
            
        except Exception as e:
            self.logger.error(f"ポートフォリオリターン計算エラー: {e}", exc_info=True)
            return None
    
    def _calculate_return_metrics(
        self,
        portfolio_returns: pd.Series,
        risk_free_rate: float
    ) -> Dict:
        """リターン指標を計算"""
        try:
            # 期待リターン
            expected_return = portfolio_returns.mean()
            
            # 超過リターン
            excess_return = expected_return - risk_free_rate
            
            # 累積リターン
            cumulative_return = (1 + portfolio_returns).prod() - 1
            
            # 最小・最大リターン
            min_return = portfolio_returns.min()
            max_return = portfolio_returns.max()
            
            # 年率換算（日次データと仮定）
            annualized_return = expected_return * 252
            
            return {
                'expected_return': expected_return,
                'excess_return': excess_return,
                'cumulative_return': cumulative_return,
                'min_return': min_return,
                'max_return': max_return,
                'annualized_return': annualized_return
            }
            
        except Exception as e:
            self.logger.error(f"リターン指標計算エラー: {e}")
            return {}
    
    def _calculate_risk_metrics(
        self,
        weights: np.ndarray,
        covariance_matrix: pd.DataFrame,
        portfolio_returns: pd.Series,
        risk_free_rate: float
    ) -> Dict:
        """リスク指標を計算"""
        try:
            # ポートフォリオ分散: σp² = w^T Σ w
            portfolio_variance = np.dot(
                weights.T,
                np.dot(covariance_matrix.values, weights)
            )
            
            # ポートフォリオ標準偏差
            portfolio_std = np.sqrt(portfolio_variance)
            
            # 年率換算（日次データと仮定）
            annualized_std = portfolio_std * np.sqrt(252)
            
            # 下方偏差（Downside Deviation）
            # リスクフリーレート未満のリターンのみを考慮
            downside_returns = portfolio_returns[portfolio_returns < risk_free_rate]
            downside_deviation = downside_returns.std() if len(downside_returns) > 0 else 0.0
            
            # プラスリターン比率
            positive_returns_ratio = len(portfolio_returns[portfolio_returns > 0]) / len(portfolio_returns)
            
            # 半分散（Semi-variance）
            # 平均未満のリターンのみを考慮
            mean_return = portfolio_returns.mean()
            below_mean_returns = portfolio_returns[portfolio_returns < mean_return]
            semi_variance = ((below_mean_returns - mean_return) ** 2).mean() if len(below_mean_returns) > 0 else 0.0
            
            return {
                'portfolio_variance': portfolio_variance,
                'portfolio_std': portfolio_std,
                'annualized_std': annualized_std,
                'downside_deviation': downside_deviation,
                'positive_returns_ratio': positive_returns_ratio,
                'semi_variance': semi_variance
            }
            
        except Exception as e:
            self.logger.error(f"リスク指標計算エラー: {e}")
            return {
                'portfolio_variance': np.nan,
                'portfolio_std': np.nan,
                'annualized_std': np.nan,
                'downside_deviation': np.nan,
                'positive_returns_ratio': np.nan,
                'semi_variance': np.nan
            }
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """最大ドローダウンを計算"""
        try:
            # 累積リターン
            cumulative = (1 + returns).cumprod()
            
            # 各時点での最高値
            running_max = cumulative.expanding().max()
            
            # ドローダウン
            drawdown = (cumulative - running_max) / running_max
            
            # 最大ドローダウン
            return drawdown.min()
            
        except Exception:
            return np.nan
    
    def _calculate_extreme_risk_metrics(
        self,
        portfolio_returns: pd.Series,
        risk_free_rate: float
    ) -> Dict:
        """エクストリームリスク指標を計算"""
        try:
            # VaR (Value at Risk)
            var_95 = portfolio_returns.quantile(0.05)
            var_99 = portfolio_returns.quantile(0.01)
            
            # CVaR (Conditional VaR)
            cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean() if len(
                portfolio_returns[portfolio_returns <= var_95]
            ) > 0 else var_95
            
            cvar_99 = portfolio_returns[portfolio_returns <= var_99].mean() if len(
                portfolio_returns[portfolio_returns <= var_99]
            ) > 0 else var_99
            
            return {
                'var_95': var_95,
                'var_99': var_99,
                'cvar_95': cvar_95,
                'cvar_99': cvar_99
            }
            
        except Exception as e:
            self.logger.error(f"エクストリームリスク計算エラー: {e}")
            return {}
    
    def _calculate_diversification_effect(
        self,
        weights: np.ndarray,
        returns_data: Dict[str, pd.Series],
        covariance_matrix: pd.DataFrame
    ) -> Dict:
        """分散化効果を計算"""
        try:
            # シンボルをソート（weightsと順序を合わせる）
            symbols = sorted(returns_data.keys())
            
            # 共通日付のDataFrameを作成
            returns_df = pd.DataFrame({s: returns_data[s] for s in symbols})
            returns_df = returns_df.dropna()
            
            if returns_df.empty:
                return {
                    'diversification_ratio': np.nan,
                    'equivalent_n_assets': np.nan,
                    'weighted_avg_std': np.nan,
                    'herfindahl_index': np.nan,
                    'diversification_benefit': np.nan
                }
            
            # 個別資産の標準偏差（共通日付で計算）
            individual_stds = returns_df.std().values
            
            # 個別資産の加重平均リスク
            weighted_avg_std = np.sum(weights * individual_stds)
            
            # ポートフォリオのリスク
            portfolio_variance = np.dot(
                weights.T,
                np.dot(covariance_matrix.values, weights)
            )
            portfolio_std = np.sqrt(max(portfolio_variance, 0))  # 負の値を防ぐ
            
            # 分散化比率
            diversification_ratio = weighted_avg_std / portfolio_std if portfolio_std > 0 else 1.0
            
            # 等価資産数
            sum_weights_squared = np.sum(weights ** 2)
            equivalent_n_assets = 1.0 / sum_weights_squared if sum_weights_squared > 0 else len(weights)
            
            # ハーフィンダール指数
            herfindahl_index = sum_weights_squared
            
            # 分散化ベネフィット
            diversification_benefit = 1 - (portfolio_std / weighted_avg_std) if weighted_avg_std > 0 else 0.0
            
            return {
                'diversification_ratio': diversification_ratio,
                'equivalent_n_assets': equivalent_n_assets,
                'weighted_avg_std': weighted_avg_std,
                'herfindahl_index': herfindahl_index,
                'diversification_benefit': diversification_benefit
            }
            
        except Exception as e:
            self.logger.error(f"分散化効果計算エラー: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {
                'diversification_ratio': np.nan,
                'equivalent_n_assets': np.nan,
                'weighted_avg_std': np.nan,
                'herfindahl_index': np.nan,
                'diversification_benefit': np.nan
            }

    
    def _create_empty_result(self, error_message: str) -> Dict:
        """空の結果を作成"""
        return {
            'success': False,
            'error': error_message,
            'expected_return': np.nan,
            'portfolio_std': np.nan,
            'max_drawdown': np.nan
        }
    
    def calculate_returns_from_prices(
        self,
        price_data: Dict[str, pd.Series]
    ) -> Dict[str, pd.Series]:
        """価格データからリターンを計算"""
        try:
            returns_data = {}
            
            for symbol, prices in price_data.items():
                # 欠損値を除去
                prices = prices.dropna()
                
                if len(prices) < 2:
                    self.logger.warning(f"{symbol}: データ不足（{len(prices)}点）- スキップ")
                    continue
                
                # 日付の正規化処理を追加
                # タイムゾーン情報がある場合は削除
                if hasattr(prices.index, 'tz') and prices.index.tz is not None:
                    self.logger.info(f"{symbol}: タイムゾーン情報を削除 ({prices.index.tz})")
                    prices.index = prices.index.tz_localize(None)
                
                # 日付のみに正規化（時間部分を削除）
                if not all(t.time() == pd.Timestamp('00:00:00').time() for t in prices.index[:min(5, len(prices.index))]):
                    self.logger.info(f"{symbol}: 日付を正規化")
                    prices.index = prices.index.normalize()
                
                # 日付範囲の診断ログ
                self.logger.info(f"{symbol}: {len(prices)}点, 期間: {prices.index.min()} ~ {prices.index.max()}")
                
                # ログリターンを計算
                log_returns = np.log(prices / prices.shift(1)).dropna()
                
                if len(log_returns) > 0:
                    returns_data[symbol] = log_returns
                    self.logger.info(f"{symbol}: {len(log_returns)}点のリターンデータを計算")
                else:
                    self.logger.warning(f"{symbol}: リターン計算結果が空")
            
            self.logger.info(f"合計 {len(returns_data)} 資産のリターンデータを計算完了")
            return returns_data
            
        except Exception as e:
            self.logger.error(f"リターン計算エラー: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {}
    
    def calculate_covariance_matrix(
        self,
        returns_data: Dict[str, pd.Series]
    ) -> pd.DataFrame:
        """共分散行列を計算"""
        try:
            # シンボルをソート
            symbols = sorted(returns_data.keys())
            
            # DataFrameを作成
            returns_df = pd.DataFrame({symbol: returns_data[symbol] for symbol in symbols})
            
            # ペアワイズ共分散行列を計算（min_periods=10で最低10日の共通データがあれば計算）
            covariance_matrix = returns_df.cov(min_periods=10)
            
            return covariance_matrix
            
        except Exception as e:
            self.logger.error(f"共分散行列計算エラー: {e}")
            return pd.DataFrame()

    def _calculate_extreme_risk_metrics(
        self,
        portfolio_returns: pd.Series,
        risk_free_rate: float
    ) -> Dict:
        """エクストリームリスク指標を計算"""
        try:
            # VaR (Value at Risk) - 95%, 99%
            var_95 = portfolio_returns.quantile(0.05)
            var_99 = portfolio_returns.quantile(0.01)
            
            # CVaR (Conditional VaR / Expected Shortfall) - 95%, 99%
            cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
            cvar_99 = portfolio_returns[portfolio_returns <= var_99].mean()
            
            # 最大ドローダウンの計算
            cumulative_returns = (1 + portfolio_returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # 回復期間（オプション）
            is_drawdown = drawdown < 0
            recovery_periods = []
            current_period = 0
            
            for dd in is_drawdown:
                if dd:
                    current_period += 1
                else:
                    if current_period > 0:
                        recovery_periods.append(current_period)
                    current_period = 0
            
            max_recovery_period = max(recovery_periods) if recovery_periods else 0
            
            return {
                'var_95': var_95,
                'var_99': var_99,
                'cvar_95': cvar_95,
                'cvar_99': cvar_99,
                'max_drawdown': max_drawdown,
                'max_recovery_period': max_recovery_period
            }
            
        except Exception as e:
            self.logger.error(f"エクストリームリスク計算エラー: {e}")
            return {
                'var_95': np.nan,
                'var_99': np.nan,
                'cvar_95': np.nan,
                'cvar_99': np.nan,
                'max_drawdown': np.nan,
                'max_recovery_period': 0
            }


    def _calculate_diversification_effect(
        self,
        weights: np.ndarray,
        returns_data: Dict[str, pd.Series],
        covariance_matrix: pd.DataFrame
    ) -> Dict:
        """分散化効果を計算"""
        try:
            # シンボルをソート（weightsと順序を合わせる）
            symbols = sorted(returns_data.keys())
            
            # 個別資産の標準偏差
            individual_stds = np.array([returns_data[s].std() for s in symbols])
            
            # 個別資産の加重平均リスク
            weighted_avg_std = np.sum(weights * individual_stds)
            
            # ポートフォリオのリスク
            portfolio_variance = np.dot(
                weights.T,
                np.dot(covariance_matrix.values, weights)
            )
            portfolio_std = np.sqrt(portfolio_variance)
            
            # 分散化比率（高いほど分散効果が大きい）
            # DR = (個別リスクの加重平均) / (ポートフォリオリスク)
            diversification_ratio = weighted_avg_std / portfolio_std if portfolio_std > 0 else 1.0
            
            # 等価資産数（Effective Number of Assets）
            # ENB = 1 / Σ(wi²)
            sum_weights_squared = np.sum(weights ** 2)
            equivalent_n_assets = 1.0 / sum_weights_squared if sum_weights_squared > 0 else len(weights)
            
            # ハーフィンダール指数（集中度）
            # 0 = 完全分散, 1 = 完全集中
            herfindahl_index = sum_weights_squared
            
            # 分散化ベネフィット（リスク削減効果）
            # DB = 1 - (ポートフォリオリスク / 加重平均リスク)
            diversification_benefit = 1 - (portfolio_std / weighted_avg_std) if weighted_avg_std > 0 else 0.0
            
            return {
                'diversification_ratio': diversification_ratio,
                'equivalent_n_assets': equivalent_n_assets,
                'weighted_avg_std': weighted_avg_std,
                'herfindahl_index': herfindahl_index,
                'diversification_benefit': diversification_benefit
            }
            
        except Exception as e:
            self.logger.error(f"分散化効果計算エラー: {e}")
            return {
                'diversification_ratio': np.nan,
                'equivalent_n_assets': np.nan,
                'weighted_avg_std': np.nan,
                'herfindahl_index': np.nan,
                'diversification_benefit': np.nan
            }


    def _create_empty_result(self, reason: str = "計算失敗") -> Dict:
        """空の結果辞書を作成"""
        return {
            'success': False,
            'error': reason,
            # リターン指標
            'expected_return': np.nan,
            'excess_return': np.nan,
            'cumulative_return': np.nan,
            'min_return': np.nan,
            'max_return': np.nan,
            'annualized_return': np.nan,
            # リスク指標
            'portfolio_variance': np.nan,
            'portfolio_std': np.nan,
            'annualized_std': np.nan,
            'downside_deviation': np.nan,
            'positive_returns_ratio': np.nan,
            # エクストリームリスク
            'var_95': np.nan,
            'var_99': np.nan,
            'cvar_95': np.nan,
            'cvar_99': np.nan,
            'max_drawdown': np.nan,
            'max_recovery_period': 0,
            # 分散化効果
            'diversification_ratio': np.nan,
            'equivalent_n_assets': np.nan,
            'weighted_avg_std': np.nan,
            'herfindahl_index': np.nan,
            'diversification_benefit': np.nan,
            # メタデータ
            'portfolio_name': '',
            'n_positions': 0,
            'total_weight': 0.0,
            'data_points': 0,
            'risk_free_rate': 0.0
        }

def calculate_all_portfolios_metrics(
    portfolios: List[Portfolio],
    price_data: Dict[str, pd.Series],
    risk_free_rate: float = 0.0
) -> Dict[str, Dict]:
    """複数ポートフォリオの統計量を一括計算"""
    logger = logging.getLogger(__name__)
    calculator = PortfolioMetricsCalculator()
    
    logger.info(f"=== ポートフォリオ統計量の一括計算開始 ===")
    logger.info(f"対象ポートフォリオ数: {len(portfolios)}")
    logger.info(f"価格データ資産数: {len(price_data)}")
    
    # リターンと共分散行列を計算
    returns_data = calculator.calculate_returns_from_prices(price_data)
    
    if not returns_data:
        logger.error("リターンデータの計算に失敗しました")
        return {}
    
    covariance_matrix = calculator.calculate_covariance_matrix(returns_data)
    
    if covariance_matrix.empty:
        logger.error("共分散行列の計算に失敗しました")
        return {}
    
    # 各ポートフォリオの統計量を計算
    all_metrics = {}
    
    for portfolio in portfolios:
        logger.info(f"--- ポートフォリオ '{portfolio.name}' の計算開始 ---")
        
        metrics = calculator.calculate_portfolio_metrics(
            portfolio,
            returns_data,
            covariance_matrix,
            risk_free_rate
        )
        
        if metrics.get('success', False):
            all_metrics[portfolio.name] = metrics
            logger.info(f"ポートフォリオ '{portfolio.name}' の計算成功")
        else:
            error = metrics.get('error', '不明なエラー')
            logger.warning(f"ポートフォリオ '{portfolio.name}' の計算失敗: {error}")
    
    logger.info(f"=== 計算完了: {len(all_metrics)}/{len(portfolios)} 個成功 ===")
    
    return all_metrics