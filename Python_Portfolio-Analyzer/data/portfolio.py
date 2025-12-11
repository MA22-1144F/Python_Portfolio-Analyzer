"""portfolio.py"""

import json
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from data.asset_info import AssetInfo
from config.constants import (
    MIN_PORTFOLIO_WEIGHT, MAX_PORTFOLIO_WEIGHT, MAX_WEIGHT_WARNING_THRESHOLD,
    ERROR_EMPTY_PORTFOLIO, ERROR_NEGATIVE_WEIGHT, ERROR_EXCESSIVE_WEIGHT
)



@dataclass
class PortfolioPosition:
    """ポートフォリオ内の個別ポジション"""
    asset: AssetInfo
    weight: float
    
    def to_dict(self) -> Dict:
        return {
            'asset': self.asset.to_dict(),
            'weight': self.weight
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PortfolioPosition':
        return cls(
            asset=AssetInfo.from_dict(data['asset']),
            weight=data['weight']
        )


class Portfolio:
    """ポートフォリオクラス"""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.positions: List[PortfolioPosition] = []
        self.cash_interest_rate: float = 0.0
        self.created_at: Optional[datetime] = None
        self.modified_at: Optional[datetime] = None
        
        # 作成時に現在時刻を設定
        if self.created_at is None:
            self.created_at = datetime.now()
        self.modified_at = datetime.now()
    
    @property
    def total_weight(self) -> float:
        """総ウエイトを計算"""
        return sum(pos.weight for pos in self.positions)
    
    @property
    def cash_position(self) -> float:
        """キャッシュポジション（1 - 総ウエイト）"""
        return 1.0 - self.total_weight
    
    @property
    def is_leveraged(self) -> bool:
        """レバレッジポートフォリオかどうか"""
        return self.total_weight > 1.0
    
    @property
    def leverage_amount(self) -> float:
        """レバレッジ金額（総ウエイト - 1）"""
        return max(0.0, self.total_weight - 1.0)
    
    def add_position(self, asset: AssetInfo, weight: float) -> bool:
        """ポジションを追加"""
        # ウエイトの検証
        if weight < MIN_PORTFOLIO_WEIGHT:
            raise ValueError(f"{ERROR_NEGATIVE_WEIGHT}: {weight}")
        if weight > MAX_PORTFOLIO_WEIGHT:
            raise ValueError(f"{ERROR_EXCESSIVE_WEIGHT}: {weight}")
        
        # 同じ資産が既に存在するかチェック
        for pos in self.positions:
            if pos.asset.symbol == asset.symbol:
                return False
        
        self.positions.append(PortfolioPosition(asset, weight))
        self.modified_at = datetime.now()
        return True
    
    def update_position_weight(self, symbol: str, new_weight: float) -> bool:
        """ポジションのウエイトを更新"""
        # ウエイトの検証
        if new_weight < MIN_PORTFOLIO_WEIGHT:
            raise ValueError(f"{ERROR_NEGATIVE_WEIGHT}: {new_weight}")
        if new_weight > MAX_PORTFOLIO_WEIGHT:
            raise ValueError(f"{ERROR_EXCESSIVE_WEIGHT}: {new_weight}")
        
        for pos in self.positions:
            if pos.asset.symbol == symbol:
                pos.weight = new_weight
                self.modified_at = datetime.now()
                return True
        return False
    
    def remove_position(self, symbol: str) -> bool:
        """ポジションを削除"""
        for i, pos in enumerate(self.positions):
            if pos.asset.symbol == symbol:
                self.positions.pop(i)
                self.modified_at = datetime.now()
                return True
        return False
    
    def get_position(self, symbol: str) -> Optional[PortfolioPosition]:
        """特定のポジションを取得"""
        for pos in self.positions:
            if pos.asset.symbol == symbol:
                return pos
        return None
    
    def normalize_weights(self) -> None:
        """ウエイトを正規化（合計を1にする）"""
        total = self.total_weight
        if total > 0:
            for pos in self.positions:
                pos.weight = pos.weight / total
            self.modified_at = datetime.now()
    
    def clear_positions(self) -> None:
        """全ポジションをクリア"""
        self.positions.clear()
        self.modified_at = datetime.now()
    
    def get_assets(self) -> List[AssetInfo]:
        """すべての資産を取得"""
        return [pos.asset for pos in self.positions]
    
    def get_symbols(self) -> List[str]:
        """すべてのシンボルを取得"""
        return [pos.asset.symbol for pos in self.positions]
    
    def get_weights(self) -> List[float]:
        """すべてのウエイトを取得"""
        return [pos.weight for pos in self.positions]
    
    def get_weight_dict(self) -> Dict[str, float]:
        """シンボル -> ウエイトの辞書を取得"""
        return {pos.asset.symbol: pos.weight for pos in self.positions}
    
    def validate(self) -> List[str]:
        """ポートフォリオの妥当性チェック"""
        errors = []
        
        if not self.name.strip():
            errors.append("ポートフォリオ名が設定されていません")
        
        if not self.positions:
            errors.append(ERROR_EMPTY_PORTFOLIO)
        
        for pos in self.positions:
            if pos.weight < MIN_PORTFOLIO_WEIGHT:
                errors.append(f"{pos.asset.symbol}: {ERROR_NEGATIVE_WEIGHT}")
            if pos.weight > MAX_WEIGHT_WARNING_THRESHOLD:
                errors.append(f"{pos.asset.symbol}: {ERROR_EXCESSIVE_WEIGHT}")
        
        if self.total_weight <= 0:
            errors.append("総ウエイトが0以下です")
        
        # 重複チェック
        symbols = [pos.asset.symbol for pos in self.positions]
        if len(symbols) != len(set(symbols)):
            errors.append("同じ資産が複数選択されています")
        
        return errors
    
    def to_dict(self) -> Dict:
        """辞書形式に変換"""
        return {
            'name': self.name,
            'description': self.description,
            'positions': [pos.to_dict() for pos in self.positions],
            'cash_interest_rate': self.cash_interest_rate,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'modified_at': self.modified_at.isoformat() if self.modified_at else None,
            'total_weight': self.total_weight,
            'cash_position': self.cash_position,
            'is_leveraged': self.is_leveraged
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Portfolio':
        """辞書からポートフォリオを作成"""
        positions = [PortfolioPosition.from_dict(pos_data) for pos_data in data.get('positions', [])]
        
        # インスタンス作成
        portfolio = cls.__new__(cls)  # __init__を呼ばずにインスタンス作成
        portfolio.name = data['name']
        portfolio.description = data.get('description', '')
        portfolio.positions = positions
        portfolio.cash_interest_rate = data.get('cash_interest_rate', 0.0)
        
        # 作成日時の復元
        created_at_str = data.get('created_at')
        if created_at_str:
            try:
                portfolio.created_at = datetime.fromisoformat(created_at_str)
            except (ValueError, AttributeError):
                portfolio.created_at = datetime.now()
        else:
            portfolio.created_at = datetime.now()
        
        # 更新日時の復元
        modified_at_str = data.get('modified_at')
        if modified_at_str:
            try:
                portfolio.modified_at = datetime.fromisoformat(modified_at_str)
            except (ValueError, AttributeError):
                portfolio.modified_at = datetime.now()
        else:
            portfolio.modified_at = datetime.now()
        
        return portfolio
    
    def save_to_file(self, file_path: str) -> None:
        """ファイルに保存"""
        # 保存時に更新日時を更新
        self.modified_at = datetime.now()
        
        # JSONに変換して保存
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load_from_file(cls, file_path: str) -> 'Portfolio':
        """ファイルから読み込み"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def copy(self) -> 'Portfolio':
        """ポートフォリオのコピーを作成"""
        copied = Portfolio.from_dict(self.to_dict())
        # コピー時は新しい作成日時を設定
        copied.created_at = datetime.now()
        copied.modified_at = datetime.now()
        return copied


class PortfolioManager:
    """ポートフォリオ管理クラス"""
    
    def __init__(self, portfolio_dir: str = "portfolios"):
        # スクリプトファイルの場所を基準とした絶対パスを使用
        if not os.path.isabs(portfolio_dir):
            # 現在のファイル（portfolio.py）の場所を取得
            current_file_dir = Path(__file__).parent.parent  # data/portfolio.py -> プロジェクトルート
            self.portfolio_dir = current_file_dir / portfolio_dir
        else:
            self.portfolio_dir = Path(portfolio_dir)
        
        self._ensure_portfolio_dir()
    
    def _ensure_portfolio_dir(self):
        """portfolioディレクトリが存在することを確認し，なければ作成"""
        try:
            self.portfolio_dir.mkdir(parents=True, exist_ok=True)
            
        except (PermissionError, OSError) as e:
            # ディレクトリ作成に失敗した場合は，一時ディレクトリを使用
            import tempfile
            import warnings
            
            warnings.warn(f"portfoliosディレクトリの作成に失敗しました: {e}")
            warnings.warn("一時ディレクトリを使用します")
            
            # 一時ディレクトリにportfoliosフォルダを作成
            temp_dir = Path(tempfile.gettempdir()) / "portfolio_analyzer" / "portfolios"
            try:
                temp_dir.mkdir(parents=True, exist_ok=True)
                self.portfolio_dir = temp_dir
                warnings.warn(f"一時ディレクトリを使用: {self.portfolio_dir}")
            except Exception as e2:
                # 一時ディレクトリでも失敗した場合は現在のディレクトリを使用
                warnings.warn(f"一時ディレクトリの作成も失敗しました: {e2}")
                warnings.warn("現在のディレクトリを使用します")
                self.portfolio_dir = Path(".")
    
    def list_portfolios(self) -> List[Tuple[str, str, str]]:
        """保存されたポートフォリオのリストを取得"""
        portfolios = []
        
        # ディレクトリが存在することを確認
        self._ensure_portfolio_dir()
        
        try:
            for file_path in self.portfolio_dir.glob("*.json"):
                try:
                    # ポートフォリオファイルを読み込んで作成日時を取得
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    portfolio_name = data.get('name', file_path.stem)
                    
                    # 作成日時の取得と表示形式への変換
                    created_at_str = data.get('created_at')
                    if created_at_str:
                        try:
                            created_at = datetime.fromisoformat(created_at_str)
                            created_at_display = created_at.strftime('%Y-%m-%d %H:%M')
                        except (ValueError, AttributeError):
                            # ISO形式でない場合はファイルの更新日時をフォールバック
                            stat = file_path.stat()
                            created_at_display = datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M')
                    else:
                        # created_atがない場合はファイルの更新日時をフォールバック
                        stat = file_path.stat()
                        created_at_display = datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M')
                    
                    portfolios.append((file_path.name, portfolio_name, created_at_display))
                    
                except Exception as e:
                    # 個別ファイルの処理エラーは無視して続行
                    print(f"ポートフォリオファイル読み込みエラー: {file_path}, {e}")
                    continue
            
            # 作成日時で逆順ソート（新しいものが上に）
            portfolios.sort(key=lambda x: x[2], reverse=True)
            
        except Exception as e:
            import warnings
            warnings.warn(f"ポートフォリオ一覧の取得中にエラーが発生しました: {e}")
        
        return portfolios
    
    def load_portfolio(self, file_name: str) -> Optional[Portfolio]:
        """ポートフォリオを読み込み"""
        try:
            file_path = self.portfolio_dir / file_name
            if not file_path.exists():
                return None
            return Portfolio.load_from_file(str(file_path))
        except Exception as e:
            import warnings
            warnings.warn(f"ポートフォリオの読み込みに失敗しました: {e}")
            return None
    
    def save_portfolio(self, portfolio: Portfolio, file_name: Optional[str] = None) -> str:
            """ポートフォリオを保存"""
            # ディレクトリが存在することを確認
            self._ensure_portfolio_dir()
            
            # 保存時に更新日時を更新（念のため）
            portfolio.modified_at = datetime.now()
            
            # ファイル名の決定
            if file_name is None:
                file_name = self._generate_safe_filename(portfolio.name)
            
            # .json拡張子を確保
            if not file_name.endswith('.json'):
                file_name += '.json'
            
            # 完全なパスを構築
            file_path = self.portfolio_dir / file_name
            
            # ファイルに保存
            portfolio.save_to_file(str(file_path))
            
            return file_name
    
    def _generate_safe_filename(self, portfolio_name: str) -> str:
        """ポートフォリオ名から安全なファイル名を生成"""
        import re
        import unicodedata
        
        # 名前の正規化
        safe_name = portfolio_name.strip()
        if not safe_name:
            safe_name = "portfolio"
        
        # Unicode正規化
        safe_name = unicodedata.normalize('NFKC', safe_name)
        
        # 日本語から英語への変換
        char_map = {
            'ポートフォリオ': 'portfolio',
            'ポート': 'port',
            'フォリオ': 'folio',
            'プロジェクト': 'project',
            'テスト': 'test'
        }
        
        for jp, en in char_map.items():
            safe_name = safe_name.replace(jp, en)
        
        # 安全でない文字を処理
        safe_name = re.sub(r'[<>:"/\\|?*]', '_', safe_name)
        safe_name = re.sub(r'[^\w\s\-_.]', '_', safe_name)
        safe_name = re.sub(r'\s+', '_', safe_name)
        safe_name = safe_name.strip('._')
        
        # 長さ制限
        if len(safe_name) > 20:
            safe_name = safe_name[:20].rstrip('_')
        
        if not safe_name:
            safe_name = "portfolio"
        
        # タイムスタンプを追加
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"{safe_name}_{timestamp}"

    def delete_portfolio(self, file_name: str) -> bool:
        """ポートフォリオを削除"""
        try:
            file_path = self.portfolio_dir / file_name
            if file_path.exists():
                file_path.unlink()
                return True
            return False
        except Exception as e:
            import warnings
            warnings.warn(f"ポートフォリオの削除に失敗しました: {e}")
            return False
    
    def portfolio_exists(self, file_name: str) -> bool:
        """ポートフォリオファイルが存在するかチェック"""
        return (self.portfolio_dir / file_name).exists()
    
    def get_portfolio_dir(self) -> str:
        """portfolioディレクトリのパスを取得"""
        return str(self.portfolio_dir)