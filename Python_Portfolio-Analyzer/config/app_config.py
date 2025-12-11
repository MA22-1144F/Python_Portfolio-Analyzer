"""app_config.py"""

import json
from pathlib import Path
from typing import Dict, Any, Optional

class AppConfig:
    def __init__(self, config_file: Optional[str] = None):
        self.project_root = Path(__file__).parent.parent
        self.config_dir = self.project_root / "config"
        self.config_file = config_file or (self.config_dir / "user_settings.json")
        self.default_config_file = self.config_dir / "default_settings.json"
        
        self.config_dir.mkdir(exist_ok=True)
        self._settings: Dict[str, Any] = {}
        
        self.load_default_settings()
        self.load_user_settings()
    
    def load_default_settings(self):
        try:
            if self.default_config_file.exists():
                with open(self.default_config_file, 'r', encoding='utf-8') as f:
                    self._settings.update(json.load(f))
            else:
                self.create_default_config()
        except (json.JSONDecodeError, IOError):
            self.create_default_config()
    
    def create_default_config(self):
        """デフォルト設定ファイルが存在しない場合、空の辞書で初期化"""
        # default_settings.jsonは既に存在することを前提とする
        # 存在しない場合は空の設定で開始
        self._settings = {}
    
    def load_user_settings(self):
        try:
            if Path(self.config_file).exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    user_settings = json.load(f)
                    self._deep_update(self._settings, user_settings)
        except (json.JSONDecodeError, IOError):
            pass
    
    def save(self):
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self._settings, f, indent=2, ensure_ascii=False)
        except IOError:
            pass
    
    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split('.')
        value = self._settings
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def set(self, key: str, value: Any):
        keys = key.split('.')
        current = self._settings
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict):
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    # UI設定取得メソッド
    def get_ui_colors(self) -> Dict[str, str]:
        """UI色設定を取得"""
        theme = self.get('ui.theme', 'dark')
        return self.get(f'ui.colors.{theme}', {})
    
    def get_ui_sizes(self) -> Dict[str, int]:
        """UIサイズ設定を取得"""
        return self.get('ui.sizes', {})
    
    def get_button_types(self) -> Dict[str, Dict[str, Any]]:
        """ボタンタイプ設定を取得"""
        return self.get('ui.button_types', {})
    
    def get_layout_settings(self) -> Dict[str, Any]:
        """レイアウト設定を取得"""
        return self.get('ui.layout', {})
    
    def get_analysis_widget_size(self, item_type: str) -> Dict[str, int]:
        """分析ウィジェットサイズを取得"""
        sizes = self.get('ui.analysis_widget_sizes', {})
        return sizes.get(item_type, self.get('ui.default_analysis_size', {}))
    
    # 分析設定取得メソッド
    def get_min_data_points(self, span: str) -> int:
        """最小データポイント数を取得"""
        min_points = self.get('analysis.min_data_points', {})
        return min_points.get(span, 30)
    
    def get_min_coverage_ratio(self) -> float:
        """最小カバレッジ率を取得"""
        return self.get('analysis.min_coverage_ratio', 0.7)
    
    def get_time_conversion_factor(self, from_span: str, to_span: str) -> float:
        """時間換算係数を取得"""
        conversions = self.get('analysis.time_conversion', {})
        from_conversions = conversions.get(from_span, {})
        key = f"to_{to_span.replace('次', 'ly').replace('日', 'dai').replace('週', 'week').replace('月', 'month').replace('年', 'year')}"
        return from_conversions.get(key, 1.0)
    
    def get_risk_conversion_factor(self, from_span: str, to_span: str) -> float:
        """リスク換算係数を取得"""
        conversions = self.get('analysis.risk_conversion', {})
        from_conversions = conversions.get(from_span, {})
        key = f"to_{to_span.replace('次', 'ly').replace('日', 'dai').replace('週', 'week').replace('月', 'month').replace('年', 'year')}"
        return from_conversions.get(key, 1.0)
    
    # 市場設定取得メソッド
    def get_market_portfolios(self) -> Dict[str, Dict[str, str]]:
        """市場ポートフォリオ設定を取得"""
        return self.get('market.portfolios', {})
    
    def get_market_info(self, market_text: str) -> Optional[Dict[str, str]]:
        """特定の市場情報を取得"""
        portfolios = self.get_market_portfolios()
        return portfolios.get(market_text)
    
    # ポートフォリオフォルダ設定メソッド
    def get_portfolio_folder(self) -> Optional[str]:
        """カスタムポートフォリオフォルダパスを取得"""
        return self.get('portfolio.custom_folder', None)

    def set_portfolio_folder(self, folder_path: str):
        """カスタムポートフォリオフォルダパスを設定"""
        self.set('portfolio.custom_folder', folder_path)
        self.save()

    # 既存プロパティ
    @property
    def window_geometry(self):
        return self.get('window.geometry')
    
    @window_geometry.setter
    def window_geometry(self, value):
        self.set('window.geometry', value)
    
    @property
    def last_tab_index(self):
        return self.get('window.last_tab_index', 0)
    
    @last_tab_index.setter
    def last_tab_index(self, value):
        self.set('window.last_tab_index', value)
    
    # テーマ設定メソッド
    def apply_color_theme(self, theme_name: str):
        """カラーテーマを適用"""
        # テーマ設定はdefault_settings.jsonから読み込む
        theme_colors = self.get(f'ui.colors.{theme_name}')
        if theme_colors:
            self.set('ui.theme', theme_name)
            self.save()
    
    def scale_ui_sizes(self, scale_factor: float):
        """UIサイズを一括スケーリング"""
        sizes = self.get_ui_sizes()
        scaled_sizes = {key: int(value * scale_factor) for key, value in sizes.items()}
        self.set('ui.sizes', scaled_sizes)
        
        # 分析ウィジェットサイズもスケーリング
        widget_sizes = self.get('ui.analysis_widget_sizes', {})
        for widget_type, size_config in widget_sizes.items():
            scaled_config = {key: int(value * scale_factor) for key, value in size_config.items()}
            self.set(f'ui.analysis_widget_sizes.{widget_type}', scaled_config)
        
        # デフォルトサイズもスケーリング
        default_size = self.get('ui.default_analysis_size', {})
        scaled_default = {key: int(value * scale_factor) for key, value in default_size.items()}
        self.set('ui.default_analysis_size', scaled_default)
        
        self.save()


# グローバル設定インスタンス
_config_instance = None

def get_config() -> AppConfig:
    """グローバル設定インスタンスを取得"""
    global _config_instance
    if _config_instance is None:
        _config_instance = AppConfig()
    return _config_instance

def initialize_config(config_file: Optional[str] = None) -> AppConfig:
    """設定を初期化"""
    global _config_instance
    _config_instance = AppConfig(config_file)
    return _config_instance