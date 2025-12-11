"""analysis_size_config.py"""

from config.app_config import get_config


def get_analysis_widget_size(item_type: str, config=None) -> dict:
    """分析項目のサイズ設定を取得"""
    if config is None:
        config = get_config()
    
    return config.get_analysis_widget_size(item_type)

def get_min_height(item_type: str) -> int:
    """分析項目の最小高さを取得"""
    size_config = get_analysis_widget_size(item_type)
    return size_config.get("min_height", 400)


def get_preferred_height(item_type: str) -> int:
    """分析項目の推奨高さを取得"""
    size_config = get_analysis_widget_size(item_type)
    return size_config.get("preferred_height", 500)


def get_min_width(item_type: str) -> int:
    """分析項目の最小幅を取得"""
    size_config = get_analysis_widget_size(item_type)
    return size_config.get("min_width", 600)


def get_preferred_width(item_type: str) -> int:
    """分析項目の推奨幅を取得"""
    size_config = get_analysis_widget_size(item_type)
    return size_config.get("preferred_width", 700)


def calculate_total_container_height(item_type: str) -> int:
    """コンテナ全体の最小高さを計算"""
    widget_height = get_min_height(item_type)
    
    # 設定からコンテナのオーバーヘッドを取得
    config = get_config()
    layout_settings = config.get_layout_settings()
    ui_sizes = config.get_ui_sizes()
    
    # コンテナのオーバーヘッドを計算
    container_overhead = (
        35 +  # header_height
        50 +  # progress_area_height
        20 +  # footer_height
        layout_settings.get("main_margins", [8, 8, 8, 8])[0] * 2 +  # 上下マージン
        layout_settings.get("main_spacing", 8) * 3  # スペーシング
    )
    
    return widget_height + container_overhead


# 一括サイズ変更用の関数
def set_all_sizes(scale_factor: float = 1.0):
    """全ての分析項目のサイズを一括変更"""
    config = get_config()
    config.scale_ui_sizes(scale_factor)


# プリセット設定
SIZE_PRESETS = {
    "compact": 0.8,    # 小さめ表示
    "normal": 1.0,     # 標準表示
    "large": 1.2,      # 大きめ表示
    "xlarge": 1.4      # 特大表示
}


def apply_size_preset(preset_name: str):
    """プリセットサイズを適用"""
    if preset_name in SIZE_PRESETS:
        scale_factor = SIZE_PRESETS[preset_name]
        set_all_sizes(scale_factor)
    else:
        raise ValueError(f"Unknown preset: {preset_name}")


# 画面解像度に応じた自動調整
def auto_adjust_for_screen_resolution(screen_width: int, screen_height: int):
    """画面解像度に応じてサイズを自動調整"""
    # 基準解像度 (1920x1080)
    base_width = 1920
    base_height = 1080
    
    # スケールファクターを計算
    width_scale = screen_width / base_width
    height_scale = screen_height / base_height
    scale_factor = min(width_scale, height_scale)  # 小さい方を採用
    
    # 最小0.7倍，最大1.5倍に制限
    scale_factor = max(0.7, min(1.5, scale_factor))
    
    set_all_sizes(scale_factor)
    
    return scale_factor


# 設定更新用関数
def update_analysis_widget_size(item_type: str, size_config: dict):
    """特定の分析項目のサイズ設定を更新"""
    config = get_config()
    config.set(f'ui.analysis_widget_sizes.{item_type}', size_config)
    config.save()


def reset_to_default_sizes():
    """サイズ設定をデフォルトにリセット"""
    config = get_config()
    # デフォルト設定を強制的に再読み込み
    config.create_default_config()
    config.save()