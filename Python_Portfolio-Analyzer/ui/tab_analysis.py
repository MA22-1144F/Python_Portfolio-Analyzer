"""tab_analysis.py"""

from PySide6.QtWidgets import QWidget, QHBoxLayout, QSplitter
from PySide6.QtCore import Qt

from ui.tiles.tile_conditions import ConditionTile
from ui.tiles.tile_assets import AssetTile
from ui.tiles.tile_analysis_items import AnalysisItemTile
from ui.tiles.tile_results import ResultTile


class AnalysisTab(QWidget):
    def __init__(self):
        super().__init__()
        self.condition_tile = None
        self.asset_tile = None
        self.analysis_item_tile = None
        self.result_tile = None
        self._setup_ui()

    def _setup_ui(self):
        """UIレイアウトを設定"""
        layout = QHBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        
        # メインスプリッター（水平分割）
        main_splitter = QSplitter(Qt.Horizontal)
        
        # 左側スプリッター（垂直分割）
        left_splitter = QSplitter(Qt.Vertical)
        self.condition_tile = ConditionTile()
        self.asset_tile = AssetTile()
        left_splitter.addWidget(self.condition_tile)
        left_splitter.addWidget(self.asset_tile)
        left_splitter.setSizes([200, 400])
        
        # 中央・右側タイル
        self.analysis_item_tile = AnalysisItemTile()
        self.result_tile = ResultTile()
        
        # スプリッターに配置
        main_splitter.addWidget(left_splitter)
        main_splitter.addWidget(self.analysis_item_tile)
        main_splitter.addWidget(self.result_tile)
        main_splitter.setSizes([200, 125, 500])
        
        layout.addWidget(main_splitter)
        self.setLayout(layout)
    
    def get_analysis_conditions(self):
        """分析条件を取得"""
        return self.condition_tile.get_analysis_conditions() if self.condition_tile else {}
    
    def get_selected_assets(self):
        """選択された資産を取得"""
        return self.asset_tile.get_selected_assets() if self.asset_tile else []