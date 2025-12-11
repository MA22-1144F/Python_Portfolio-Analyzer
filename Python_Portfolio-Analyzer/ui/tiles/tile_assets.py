"""tile_assets.py"""

import os
import json
import logging
from PySide6.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QTreeWidget, QTreeWidgetItem, QMessageBox, QProgressBar,
    QFileDialog, QHeaderView, QMenu, QGroupBox
)
from PySide6.QtCore import QThread, Signal, QTimer, Qt
from PySide6.QtGui import QFont, QAction
from typing import List, Dict
from datetime import datetime

from data.asset_info import AssetInfo
from data.asset_searcher import AssetSearcher
from data.portfolio import Portfolio
from analysis.analysis_base_widget import AnalysisStyles

class AssetSearchThread(QThread):
    search_completed = Signal(list)
    search_error = Signal(str)
    
    def __init__(self, query: str, searcher: AssetSearcher):
        super().__init__()
        self.query = query
        self.searcher = searcher
    
    def run(self):
        try:
            results = self.searcher.search_assets(self.query)
            self.search_completed.emit(results)
        except Exception as e:
            logging.error(f"Asset search error: {e}")
            self.search_error.emit(str(e))


class AssetTile(QFrame):
    assets_changed = Signal()
    
    def __init__(self):
        super().__init__()
        self.setFrameShape(QFrame.StyledPanel)
        self.styles = AnalysisStyles()
        self.assets: Dict[str, AssetInfo] = {}
        self.searcher = AssetSearcher()
        self.search_thread = None
        self.search_timer = QTimer()
        self.search_results: List[AssetInfo] = []
        
        self._setup_ui()
        self._setup_connections()
    
    def _setup_ui(self):
        """UIレイアウト設定"""
        layout = QVBoxLayout()
        
        # タイトル
        title_label = self._create_title_label()
        layout.addWidget(title_label)
        
        # 検索セクション
        search_group = self._create_search_section()
        layout.addWidget(search_group, stretch=2)
        
        # 選択済みセクション
        selected_group = self._create_selected_section()
        layout.addWidget(selected_group, stretch=3)
        
        self.setLayout(layout)
        self._update_display()
    
    def _create_title_label(self):
        """タイトルラベル作成"""
        label = QLabel("分析対象資産")
        font = QFont()
        font.setPointSize(11)
        font.setBold(True)
        label.setFont(font)
        return label
    
    def _create_search_section(self):
        """検索セクション作成"""
        search_group = QGroupBox("資産検索")
        layout = QVBoxLayout(search_group)
        
        # 検索入力
        input_layout = QHBoxLayout()
        self.asset_input = QLineEdit()
        self.asset_input.setPlaceholderText("銘柄名・証券コード・ティッカーを入力")
        self.search_button = QPushButton("検索")
        self.search_button.setStyleSheet(self.styles.get_button_style_by_type("secondary"))
        self.search_button.setMaximumWidth(60)
        
        input_layout.addWidget(self.asset_input)
        input_layout.addWidget(self.search_button)
        layout.addLayout(input_layout)
        
        # プログレスバー
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumHeight(3)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # ステータス
        self.status_label = QLabel("例: NTT, AAPL, 8306")
        self.status_label.setStyleSheet("color: #666; font-size: 9px;")
        self.status_label.setMaximumHeight(12)
        layout.addWidget(self.status_label)
        
        # 検索結果ツリー
        self.search_tree = self._create_tree_widget(
            ["検索結果", "シンボル", "セクター/種別", "取引所"], 
            max_height=200
        )
        layout.addWidget(self.search_tree)
        
        return search_group
    
    def _create_selected_section(self):
        """選択済みセクション作成"""
        selected_group = QGroupBox("選択済み資産")
        layout = QVBoxLayout(selected_group)
        
        # ボタンレイアウト
        button_layout = QHBoxLayout()
        
        self.save_button = QPushButton("保存")
        self.save_button.setStyleSheet(self.styles.get_button_style_by_type("save"))
        self.save_button.setMaximumWidth(50)
        self.save_button.setEnabled(False)
        
        self.load_button = QPushButton("読込")
        self.load_button.setStyleSheet(self.styles.get_button_style_by_type("secondary"))
        self.load_button.setMaximumWidth(50)
        
        self.clear_button = QPushButton("全削除")
        self.clear_button.setStyleSheet(self.styles.get_button_style_by_type("danger"))
        self.clear_button.setMaximumWidth(60)
        self.clear_button.setEnabled(False)
        
        self.count_label = QLabel("0件選択")
        self.count_label.setStyleSheet("color: #666; font-size: 10px; font-weight: bold;")
        
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.load_button)
        button_layout.addWidget(self.clear_button)
        button_layout.addStretch()
        button_layout.addWidget(self.count_label)
        
        layout.addLayout(button_layout)
        
        # 選択済みツリー
        self.selected_tree = self._create_tree_widget(
            ["選択済み資産", "シンボル", "セクター/種別", "取引所", "操作"]
        )
        layout.addWidget(self.selected_tree)
        
        # 空状態ラベル
        self.empty_label = QLabel("資産が選択されていません．")
        self.empty_label.setAlignment(Qt.AlignCenter)
        self.empty_label.setStyleSheet("""
            QLabel {
                color: #999; font-size: 12px; background-color: #f8f9fa;
                border: 1px dashed #ccc; border-radius: 4px; padding: 20px; margin: 5px;
            }
        """)
        layout.addWidget(self.empty_label)
        
        return selected_group
    
    def _create_tree_widget(self, headers, max_height=None):
        """ツリーウィジェット作成"""
        tree = QTreeWidget()
        tree.setHeaderLabels(headers)
        tree.setAlternatingRowColors(True)
        tree.setRootIsDecorated(False)
        tree.setSortingEnabled(False)
        tree.setContextMenuPolicy(Qt.CustomContextMenu)
        
        if max_height:
            tree.setMaximumHeight(max_height)
        
        # ヘッダー設定
        header = tree.header()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        for i in range(1, len(headers)):
            header.setSectionResizeMode(i, QHeaderView.ResizeToContents)
        
        return tree
    
    def _setup_connections(self):
        """シグナル・スロット接続"""
        # 検索関連
        self.asset_input.textChanged.connect(self._on_search_text_changed)
        self.asset_input.returnPressed.connect(self._start_search)
        self.search_button.clicked.connect(self._start_search)
        
        # ボタン関連
        self.save_button.clicked.connect(self.save_asset_list)
        self.load_button.clicked.connect(self.load_asset_list)
        self.clear_button.clicked.connect(self.clear_all_assets)
        
        # ツリー関連
        self.search_tree.customContextMenuRequested.connect(self._show_search_context_menu)
        self.search_tree.itemDoubleClicked.connect(self._on_search_item_double_clicked)
        self.selected_tree.customContextMenuRequested.connect(self._show_selected_context_menu)
        self.selected_tree.itemDoubleClicked.connect(self._on_selected_item_double_clicked)
        
        # タイマー
        self.search_timer.setSingleShot(True)
        self.search_timer.timeout.connect(self._start_search)
    
    def _on_search_text_changed(self, text: str):
        """検索テキスト変更時処理"""
        if len(text.strip()) >= 2:
            self.search_timer.stop()
            self.search_timer.start(500)
        else:
            self.search_timer.stop()
            self._clear_search_results()
    
    def _start_search(self):
        """検索開始"""
        query = self.asset_input.text().strip()
        if not query or len(query) < 2:
            self._clear_search_results()
            return
        
        if self.search_thread and self.search_thread.isRunning():
            self.search_thread.quit()
            self.search_thread.wait()
        
        self._set_search_state(True)
        
        self.search_thread = AssetSearchThread(query, self.searcher)
        self.search_thread.search_completed.connect(self._on_search_completed)
        self.search_thread.search_error.connect(self._on_search_error)
        self.search_thread.start()
    
    def _set_search_state(self, searching: bool):
        """検索状態設定"""
        self.progress_bar.setVisible(searching)
        self.search_button.setEnabled(not searching)
        if searching:
            self.progress_bar.setRange(0, 0)
            self.status_label.setText("検索中...")
    
    def _on_search_completed(self, results: List[AssetInfo]):
        """検索完了時処理"""
        self._set_search_state(False)
        self.search_results = results
        
        if not results:
            self.status_label.setText("見つかりません．")
            self._clear_search_results()
            return
        
        self.status_label.setText(f"{len(results)}件 - ダブルクリックで追加")
        self._display_search_results(results)
    
    def _on_search_error(self, error_message: str):
        """検索エラー時処理"""
        self._set_search_state(False)
        self.status_label.setText("検索エラー")
        self._clear_search_results()
    
    def _display_search_results(self, results: List[AssetInfo]):
        """検索結果表示"""
        self.search_tree.clear()
        
        for asset in results:
            item = self._create_asset_tree_item(asset)
            
            # 既に追加済みの場合はグレーアウト
            if asset.symbol in self.assets:
                self._set_item_gray(item)
                item.setText(0, f"{asset.name} (追加済み)")
            
            self.search_tree.addTopLevelItem(item)
    
    def _create_asset_tree_item(self, asset: AssetInfo):
        """資産ツリーアイテム作成"""
        item = QTreeWidgetItem([
            asset.name, 
            asset.symbol,
            asset.get_sector_or_type(),
            asset.exchange or "-"
        ])
        item.setData(0, Qt.UserRole, asset)
        item.setToolTip(0, self._create_asset_tooltip(asset))
        return item
    
    def _create_asset_tooltip(self, asset: AssetInfo):
        """資産ツールチップ作成"""
        tooltip_parts = [f"名前: {asset.name}", f"シンボル: {asset.symbol}"]
        
        optional_fields = [
            ("セクター", asset.sector),
            ("業界", asset.industry),
            ("取引所", asset.exchange),
            ("通貨", asset.currency),
            ("国", asset.country)
        ]
        
        for label, value in optional_fields:
            if value:
                tooltip_parts.append(f"{label}: {value}")
        
        return "\n".join(tooltip_parts)
    
    def _set_item_gray(self, item):
        """アイテムをグレーアウト"""
        for i in range(item.columnCount()):
            item.setForeground(i, Qt.gray)
    
    def _clear_search_results(self):
        """検索結果クリア"""
        self.search_tree.clear()
        self.search_results = []
    
    def _on_search_item_double_clicked(self, item: QTreeWidgetItem, column: int):
        """検索結果ダブルクリック"""
        asset = item.data(0, Qt.UserRole)
        if isinstance(asset, AssetInfo):
            self.add_asset(asset)
    
    def _on_selected_item_double_clicked(self, item: QTreeWidgetItem, column: int):
        """選択済みアイテムダブルクリック"""
        asset = item.data(0, Qt.UserRole)
        if isinstance(asset, AssetInfo):
            self._show_asset_details(asset)
    
    def add_asset(self, asset: AssetInfo):
        """資産追加"""
        if asset.symbol in self.assets:
            self.status_label.setText(f"{asset.name} は既に追加されています．")
            return
        
        self.assets[asset.symbol] = asset
        self.status_label.setText(f"'{asset.name}' を追加しました．")
        self._update_display()
        self._refresh_search_results()
        self.assets_changed.emit()
    
    def remove_asset(self, symbol: str):
        """資産削除"""
        if symbol in self.assets:
            asset = self.assets.pop(symbol)
            self.status_label.setText(f"'{asset.name}' を削除しました．")
            self._update_display()
            self._refresh_search_results()
            self.assets_changed.emit()
    
    def clear_all_assets(self):
        """全資産削除"""
        if not self.assets:
            return
        
        reply = QMessageBox.question(
            self, "確認", "選択済みの全ての資産を削除しますか？",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.assets.clear()
            self.status_label.setText("全ての資産を削除しました．")
            self._update_display()
            self._refresh_search_results()
            self.assets_changed.emit()
    
    def _update_display(self):
        """表示更新"""
        self.selected_tree.clear()
        has_assets = len(self.assets) > 0
        
        self.empty_label.setVisible(not has_assets)
        self.selected_tree.setVisible(has_assets)
        
        if has_assets:
            for asset in sorted(self.assets.values(), key=lambda x: x.name):
                item = self._create_asset_tree_item(asset)
                item.setText(3, asset.exchange or "-")
                self.selected_tree.addTopLevelItem(item)
                
                # 削除ボタン追加
                delete_button = QPushButton("削除")
                delete_button.setStyleSheet(self.styles.get_button_style_by_type("danger"))
                delete_button.setMaximumWidth(50)
                delete_button.setMaximumHeight(20)
                delete_button.clicked.connect(
                    lambda checked, s=asset.symbol: self.remove_asset(s)
                )
                self.selected_tree.setItemWidget(item, 4, delete_button)
        
        self._update_controls()
    
    def _refresh_search_results(self):
        """検索結果更新"""
        if self.search_results:
            self._display_search_results(self.search_results)
    
    def _update_controls(self):
        """コントロール状態更新"""
        has_assets = len(self.assets) > 0
        self.save_button.setEnabled(has_assets)
        self.clear_button.setEnabled(has_assets)
        self.count_label.setText(f"{len(self.assets)}件選択")
    
    def _show_search_context_menu(self, position):
        """検索コンテキストメニュー表示"""
        item = self.search_tree.itemAt(position)
        if not item:
            return
        
        asset = item.data(0, Qt.UserRole)
        if not isinstance(asset, AssetInfo):
            return
        
        menu = QMenu(self)
        
        if asset.symbol not in self.assets:
            add_action = QAction("追加", self)
            add_action.triggered.connect(lambda: self.add_asset(asset))
            menu.addAction(add_action)
        
        details_action = QAction("詳細表示", self)
        details_action.triggered.connect(lambda: self._show_asset_details(asset))
        menu.addAction(details_action)
        
        if menu.actions():
            menu.exec(self.search_tree.mapToGlobal(position))
    
    def _show_selected_context_menu(self, position):
        """選択済みコンテキストメニュー表示"""
        item = self.selected_tree.itemAt(position)
        if not item:
            return
        
        asset = item.data(0, Qt.UserRole)
        if not isinstance(asset, AssetInfo):
            return
        
        menu = QMenu(self)
        
        remove_action = QAction("削除", self)
        remove_action.triggered.connect(lambda: self.remove_asset(asset.symbol))
        menu.addAction(remove_action)
        
        details_action = QAction("詳細表示", self)
        details_action.triggered.connect(lambda: self._show_asset_details(asset))
        menu.addAction(details_action)
        
        menu.exec(self.selected_tree.mapToGlobal(position))
    
    def _show_asset_details(self, asset: AssetInfo):
        """資産詳細表示"""
        details_parts = [f"資産名: {asset.name}", f"シンボル: {asset.symbol}"]
        
        optional_fields = [
            ("通貨", asset.currency),
            ("取引所", asset.exchange),
            ("国", asset.country),
            ("セクター", asset.sector),
            ("業界", asset.industry),
            ("種別", asset.legal_type)
        ]
        
        for label, value in optional_fields:
            if value:
                details_parts.append(f"{label}: {value}")
        
        if asset.description:
            details_parts.append(f"\n説明:\n{asset.description[:300]}...")
        
        QMessageBox.information(self, f"{asset.name} の詳細", "\n".join(details_parts))
    
    def save_asset_list(self):
        """資産リスト保存（統合ポートフォリオ形式）"""
        if not self.assets:
            QMessageBox.information(self, "保存", "保存する資産がありません．")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "ポートフォリオを保存", "portfolio.json", "JSON ファイル (*.json)"
        )
        
        if file_path:
            try:
                # Portfolioオブジェクトを作成
                portfolio = Portfolio(
                    name=os.path.splitext(os.path.basename(file_path))[0],
                    description="個別資産分析タブで作成"
                )
                
                # 資産を均等配分で追加（ウエイトは0%に設定）
                for asset in self.assets.values():
                    portfolio.add_position(asset, 0.0)
                
                # 日時を設定
                portfolio.created_at = datetime.now()
                portfolio.modified_at = datetime.now()
                
                # 保存
                portfolio.save_to_file(file_path)
                
                self.status_label.setText(f"保存完了: {os.path.basename(file_path)}")
                QMessageBox.information(
                    self, "保存完了", 
                    f"ポートフォリオを保存しました．\n"
                    f"ファイル: {os.path.basename(file_path)}\n"
                    f"資産数: {len(self.assets)}件\n\n"
                    f"※ウエイトは0%で保存されています．\n"
                    f"ポートフォリオ管理タブでウエイトを設定できます．"
                )
                
            except Exception as e:
                QMessageBox.critical(self, "保存エラー", f"保存エラー:\n{str(e)}")
    
    def load_asset_list(self):
        """資産リスト読み込み"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "ポートフォリオを読み込み", "", "JSON ファイル (*.json)"
        )
        
        if not file_path:
            return
        
        try:
            # JSONファイルを読み込み
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 読み込み前の資産数を記録
            original_count = len(self.assets)
            loaded_assets = {}
            duplicate_count = 0

            if 'positions' in data:
                for pos_data in data['positions']:
                    asset_data = pos_data.get('asset', {})
                    asset = AssetInfo.from_dict(asset_data)
                    
                    # 重複チェック
                    if asset.symbol in self.assets:
                        duplicate_count += 1
                    else:
                        loaded_assets[asset.symbol] = asset
                
                # 既存の資産に追加（置き換えではない）
                self.assets.update(loaded_assets)
                
                # ポートフォリオ情報を取得
                portfolio_name = data.get('name', 'Unknown')
                portfolio_desc = data.get('description', '')
                total_weight = data.get('total_weight', 0.0)
                
                # 詳細な結果メッセージを作成
                new_count = len(self.assets)
                added_count = len(loaded_assets)
                
                info_message = f"ポートフォリオを読み込みました．\n\n"
                info_message += f"名前: {portfolio_name}\n"
                if portfolio_desc:
                    info_message += f"説明: {portfolio_desc}\n"
                info_message += f"\n【読み込み結果】\n"
                info_message += f"追加された資産: {added_count}件\n"
                if duplicate_count > 0:
                    info_message += f"重複してスキップ: {duplicate_count}件\n"
                info_message += f"元の資産数: {original_count}件\n"
                info_message += f"現在の資産数: {new_count}件\n"
                
                if total_weight > 0:
                    info_message += f"\n総ウエイト: {total_weight * 100:.1f}%\n"
                    info_message += "※個別資産分析タブではウエイト情報は使用されません．"
                
                QMessageBox.information(self, "読み込み完了", info_message)
            
            # 旧形式（資産リストのみ）の読み込み
            elif 'assets' in data:
                for asset_data in data['assets']:
                    asset = AssetInfo.from_dict(asset_data)
                    
                    # 重複チェック
                    if asset.symbol in self.assets:
                        duplicate_count += 1
                    else:
                        loaded_assets[asset.symbol] = asset
                
                # 既存の資産に追加
                self.assets.update(loaded_assets)
                
                # 詳細な結果メッセージを作成
                new_count = len(self.assets)
                added_count = len(loaded_assets)
                
                info_message = f"資産リストを読み込みました．\n\n"
                info_message += f"追加された資産: {added_count}件\n"
                if duplicate_count > 0:
                    info_message += f"重複してスキップ: {duplicate_count}件\n"
                info_message += f"元の資産数: {original_count}件\n"
                info_message += f"現在の資産数: {new_count}件"
                
                QMessageBox.information(self, "読み込み完了", info_message)
            
            else:
                raise ValueError("認識できないファイル形式です．")
            
            # UI更新
            self._update_display()
            self._refresh_search_results()
            self.assets_changed.emit()
            
            # ステータスバーに詳細情報を表示
            self.status_label.setText(
                f"読込完了: {os.path.basename(file_path)} "
                f"(+{len(loaded_assets)}件, 重複{duplicate_count}件)"
            )
            
        except Exception as e:
            QMessageBox.critical(self, "読み込みエラー", f"読み込みエラー:\n{str(e)}")
    
    # パブリックメソッド
    def get_selected_assets(self) -> List[AssetInfo]:
        """選択済み資産取得"""
        return list(self.assets.values())
    
    def get_asset_count(self) -> int:
        """資産数取得"""
        return len(self.assets)
    
    def has_asset(self, symbol: str) -> bool:
        """資産存在チェック"""
        return symbol in self.assets