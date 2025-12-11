"""tile_analysis_items.py"""

from PySide6.QtWidgets import (
    QFrame, QVBoxLayout, QLabel, QListWidget, QListWidgetItem, QMessageBox
)
from PySide6.QtCore import Qt, QMimeData, QPoint
from PySide6.QtGui import QFont, QDrag, QPixmap, QPainter


class DraggableAnalysisItem(QListWidgetItem):
    def __init__(self, item_type: str, item_name: str, description: str = ""):
        super().__init__(item_name)
        self.item_type = item_type
        self.item_name = item_name
        self.description = description
        self.setToolTip(description if description else item_name)
        self.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsDragEnabled)


class AnalysisItemList(QListWidget):
    # 実装済み分析項目
    IMPLEMENTED_ITEMS = {
        "price_series", "return_risk_analysis", "correlation_matrix", 
        "efficient_frontier", "downside_deviation_frontier", "security_market_line"
    }
    
    # 分析項目定義
    ANALYSIS_ITEMS = [
        ("price_series", "資産価格時系列表示", "分析期間における各資産の価格推移を表示します．"),
        ("return_risk_analysis", "リターン・リスク指標", 
         "期待利益率，標準偏差，VaR，シャープレシオなど17項目の分析を行います．"),
        ("correlation_matrix", "相関行列", "各資産間の相関係数を行列形式で表示します．"),
        ("efficient_frontier", "最小分散フロンティア", "最小分散フロンティアと資本分配線を描画します．"),
        ("downside_deviation_frontier", "下方偏差フロンティア", "下方偏差を最小化するフロンティアとソルティノ最適化線を描画します．"),
        ("security_market_line", "証券市場線", "証券市場線と各資産のβを描画します．")
    ]
    
    def __init__(self):
        super().__init__()
        self._setup_widget()
        self._populate_items()
    
    def _setup_widget(self):
        """ウィジェットの基本設定"""
        self.setDragDropMode(QListWidget.DragOnly)
        self.setDefaultDropAction(Qt.CopyAction)
        self.setStyleSheet("""
            QListWidget {
                border: 1px solid #555;
                border-radius: 4px;
                background-color: #2b2b2b;
                color: #ffffff;
                selection-background-color: #0078d4;
                outline: none;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #444;
            }
            QListWidget::item:hover {
                background-color: #404040;
            }
            QListWidget::item:selected {
                background-color: #0078d4;
            }
        """)
    
    def _populate_items(self):
        """分析項目を追加"""
        for item_type, item_name, description in self.ANALYSIS_ITEMS:
            item = DraggableAnalysisItem(item_type, item_name, description)
            
            if self._is_implemented(item_type):
                item.setForeground(Qt.white)
                item.setText(f"✓ {item_name}")
            else:
                item.setForeground(Qt.gray)
                item.setText(f"○ {item_name}")
                item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            
            self.addItem(item)
    
    def _is_implemented(self, item_type):
        """項目が実装済みかチェック"""
        return item_type in self.IMPLEMENTED_ITEMS
    
    def startDrag(self, supportedActions):
        """ドラッグ開始処理"""
        current_item = self.currentItem()
        if not current_item:
            return
        
        if not self._is_implemented(current_item.item_type):
            QMessageBox.information(
                self, "未実装", 
                f"'{current_item.item_name}' は現在開発中です．"
            )
            return
        
        self._execute_drag(current_item)
    
    def _execute_drag(self, item):
        """ドラッグ実行"""
        drag = QDrag(self)
        mime_data = QMimeData()
        mime_data.setText(f"{item.item_type}|{item.item_name}")
        drag.setMimeData(mime_data)
        
        # ドラッグ用画像作成
        pixmap = self._create_drag_pixmap(item)
        drag.setPixmap(pixmap)
        drag.setHotSpot(QPoint(pixmap.width() // 2, pixmap.height() // 2))
        
        drag.exec(Qt.CopyAction)
    
    def _create_drag_pixmap(self, item):
        """ドラッグ用画像作成"""
        pixmap = QPixmap(self.visualItemRect(item).size())
        pixmap.fill(Qt.transparent)
        
        painter = QPainter(pixmap)
        painter.fillRect(pixmap.rect(), Qt.darkGray)
        painter.setPen(Qt.white)
        painter.drawText(pixmap.rect(), Qt.AlignCenter, item.item_name)
        painter.end()
        
        return pixmap


class AnalysisItemTile(QFrame):
    def __init__(self):
        super().__init__()
        self.setFrameShape(QFrame.StyledPanel)
        self._setup_ui()
    
    def _setup_ui(self):
        """UIレイアウト設定"""
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        
        # タイトル
        title_label = self._create_title_label()
        layout.addWidget(title_label)
        
        # 説明
        description_label = self._create_description_label()
        layout.addWidget(description_label)
        
        # 分析項目リスト
        self.item_list = AnalysisItemList()
        layout.addWidget(self.item_list)
        
        self.setLayout(layout)
    
    def _create_title_label(self):
        """タイトルラベル作成"""
        label = QLabel("分析項目")
        font = QFont()
        font.setPointSize(11)
        font.setBold(True)
        label.setFont(font)
        label.setStyleSheet("color: #ffffff; margin-bottom: 5px;")
        return label
    
    def _create_description_label(self):
        """説明ラベル作成"""
        label = QLabel("分析項目を分析結果タイルにドラッグ")
        label.setWordWrap(True)
        label.setStyleSheet("""
            QLabel {
                color: #aaaaaa;
                font-size: 10px;
                padding: 5px;
                background-color: #333333;
                border-radius: 4px;
            }
        """)
        return label
    
    def get_analysis_items(self):
        """分析項目リストを取得"""
        items = []
        for i in range(self.item_list.count()):
            item = self.item_list.item(i)
            if isinstance(item, DraggableAnalysisItem):
                implemented = self.item_list._is_implemented(item.item_type)
                items.append((item.item_type, item.item_name, item.description, implemented))
        return items