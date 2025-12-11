"""tab_management.py"""

import os
from typing import Dict, List, Optional
import json
import logging

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QTreeWidget, QTreeWidgetItem, QDoubleSpinBox, QTextEdit, QGroupBox,
    QSplitter, QFileDialog, QMessageBox, QProgressBar,
    QHeaderView, QCheckBox, QTabWidget, QDialog, QDialogButtonBox, QSizePolicy
)
from PySide6.QtCore import QThread, Signal, Qt, QTimer

from data.asset_info import AssetInfo
from data.asset_searcher import AssetSearcher
from data.scraper import get_latest_jgb_1year_rate
from data.portfolio import Portfolio, PortfolioManager
from datetime import datetime
from analysis.analysis_base_widget import AnalysisStyles
from ui.csv_import_dialog import CSVImportDialog
from utils.ui_styles import get_checkbox_style
from utils.common_widgets import InterestRateThread

class AssetSearchThread(QThread):
    search_completed = Signal(list)
    search_error = Signal(str)
    
    def __init__(self, query: str, searcher: AssetSearcher):
        super().__init__()
        self.query = query
        self.searcher = searcher
    
    def run(self):
        try:
            results = self.searcher.search_assets(self.query, max_results=20)
            self.search_completed.emit(results)
        except Exception as e:
            self.search_error.emit(str(e))

class WeightAllocationDialog(QDialog):
    def __init__(self, asset_names: List[str], parent=None):
        super().__init__(parent)
        self.asset_names = asset_names
        self.selected_assets = set(asset_names)
        self.allocation_weights = {}
        
        self._setup_ui()
        self._setup_connections()
    
    def _setup_ui(self):
        self.setWindowTitle("ウエイト一括設定")
        self.setFixedSize(450, 500)
        
        layout = QVBoxLayout()
        
        # 設定方法
        method_group = self._create_method_group()
        layout.addWidget(method_group)
        
        # 対象資産
        assets_group = self._create_assets_group()
        layout.addWidget(assets_group)
        
        # プレビュー
        preview_group = self._create_preview_group()
        layout.addWidget(preview_group)
        
        # ボタン
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addWidget(buttons)
        
        self.setLayout(layout)
        self._update_preview()
    
    def _create_method_group(self):
        group = QGroupBox("設定方法")
        layout = QVBoxLayout(group)
        
        # チェックボックスの共通スタイル
        checkbox_style = get_checkbox_style()

        self.equal_radio = QCheckBox("均等割り振り")
        self.equal_radio.setChecked(True)
        self.equal_radio.setStyleSheet(checkbox_style)
        layout.addWidget(self.equal_radio)
        
        # カスタム割合
        custom_layout = QHBoxLayout()
        self.custom_radio = QCheckBox("カスタム割合:")
        self.custom_radio.setStyleSheet(checkbox_style)
        self.custom_weight_spinbox = QDoubleSpinBox()
        self.custom_weight_spinbox.setRange(0.1, 100.0)
        self.custom_weight_spinbox.setValue(10.0)
        self.custom_weight_spinbox.setSuffix("%")
        self.custom_weight_spinbox.setEnabled(False)
        
        custom_layout.addWidget(self.custom_radio)
        custom_layout.addWidget(self.custom_weight_spinbox)
        custom_layout.addStretch()
        layout.addLayout(custom_layout)
        
        # 比例配分
        ratio_layout = QHBoxLayout()
        self.ratio_radio = QCheckBox("比例配分（総ウエイト:")
        self.ratio_radio.setStyleSheet(checkbox_style)
        self.total_weight_spinbox = QDoubleSpinBox()
        self.total_weight_spinbox.setRange(1.0, 500.0)
        self.total_weight_spinbox.setValue(100.0)
        self.total_weight_spinbox.setSuffix("%")
        self.total_weight_spinbox.setEnabled(False)
        
        ratio_layout.addWidget(self.ratio_radio)
        ratio_layout.addWidget(self.total_weight_spinbox)
        ratio_layout.addWidget(QLabel("）"))
        ratio_layout.addStretch()
        layout.addLayout(ratio_layout)
        
        return group
    
    def _create_assets_group(self):
        group = QGroupBox("対象資産")
        layout = QVBoxLayout(group)
        
        # チェックボックスの共通スタイル
        checkbox_style = get_checkbox_style()

        # 選択ボタン
        select_layout = QHBoxLayout()
        self.select_all_button = QPushButton("全選択")
        self.select_all_button.setMaximumWidth(60)
        self.deselect_all_button = QPushButton("全解除")
        self.deselect_all_button.setMaximumWidth(60)
        self.selected_count_label = QLabel(f"{len(self.selected_assets)}/{len(self.asset_names)}件選択")
        
        select_layout.addWidget(self.select_all_button)
        select_layout.addWidget(self.deselect_all_button)
        select_layout.addStretch()
        select_layout.addWidget(self.selected_count_label)
        layout.addLayout(select_layout)
        
        # 資産リスト
        self.assets_tree = QTreeWidget()
        self.assets_tree.setHeaderLabels(["選択", "資産名"])
        self.assets_tree.setMaximumHeight(200)
        self.assets_tree.setRootIsDecorated(False)
        
        for asset_name in self.asset_names:
            item = QTreeWidgetItem(["", asset_name])
            item.setData(0, Qt.UserRole, asset_name)
            checkbox = QCheckBox()
            checkbox.setChecked(True)
            checkbox.setStyleSheet(checkbox_style)
            checkbox.toggled.connect(lambda checked, name=asset_name: self._on_asset_toggled(name, checked))
            self.assets_tree.addTopLevelItem(item)
            self.assets_tree.setItemWidget(item, 0, checkbox)
        
        layout.addWidget(self.assets_tree)
        return group
    
    def _create_preview_group(self):
        group = QGroupBox("プレビュー")
        layout = QVBoxLayout(group)
        
        self.preview_label = QLabel()
        self.preview_label.setStyleSheet("font-family: monospace; font-size: 10px;")
        self.preview_label.setWordWrap(True)
        layout.addWidget(self.preview_label)
        
        return group
    
    def _setup_connections(self):
        # 方法変更
        for radio in [self.equal_radio, self.custom_radio, self.ratio_radio]:
            radio.toggled.connect(self._on_method_changed)
        
        # 数値変更
        self.custom_weight_spinbox.valueChanged.connect(self._update_preview)
        self.total_weight_spinbox.valueChanged.connect(self._update_preview)
        
        # 選択ボタン
        self.select_all_button.clicked.connect(self._select_all_assets)
        self.deselect_all_button.clicked.connect(self._deselect_all_assets)
        
        # ダイアログボタン
        buttons = self.findChild(QDialogButtonBox)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
    
    def _on_method_changed(self):
        sender = self.sender()
        if sender.isChecked():
            # 排他制御
            radios = [self.equal_radio, self.custom_radio, self.ratio_radio]
            for radio in radios:
                if radio != sender:
                    radio.setChecked(False)
            
            # スピンボックス制御
            self.custom_weight_spinbox.setEnabled(sender == self.custom_radio)
            self.total_weight_spinbox.setEnabled(sender == self.ratio_radio)
        
        self._update_preview()
    
    def _on_asset_toggled(self, asset_name: str, checked: bool):
        if checked:
            self.selected_assets.add(asset_name)
        else:
            self.selected_assets.discard(asset_name)
        
        self.selected_count_label.setText(f"{len(self.selected_assets)}/{len(self.asset_names)}件選択")
        self._update_preview()
    
    def _select_all_assets(self):
        self.selected_assets = set(self.asset_names)
        self._update_checkboxes()
        self._update_preview()
    
    def _deselect_all_assets(self):
        self.selected_assets.clear()
        self._update_checkboxes()
        self._update_preview()
    
    def _update_checkboxes(self):
        for i in range(self.assets_tree.topLevelItemCount()):
            item = self.assets_tree.topLevelItem(i)
            asset_name = item.data(0, Qt.UserRole)
            checkbox = self.assets_tree.itemWidget(item, 0)
            checkbox.blockSignals(True)
            checkbox.setChecked(asset_name in self.selected_assets)
            checkbox.blockSignals(False)
        
        self.selected_count_label.setText(f"{len(self.selected_assets)}/{len(self.asset_names)}件選択")
    
    def _update_preview(self):
        if not self.selected_assets:
            self.preview_label.setText("選択された資産がありません")
            return
        
        selected_count = len(self.selected_assets)
        
        if self.equal_radio.isChecked():
            weight_per_asset = 100.0 / selected_count
            preview_text = f"均等割り振り: 各資産 {weight_per_asset:.2f}%\n"
            self.allocation_weights = {name: weight_per_asset for name in self.selected_assets}
            
        elif self.custom_radio.isChecked():
            custom_weight = self.custom_weight_spinbox.value()
            preview_text = f"カスタム割合: 各資産 {custom_weight:.2f}%\n"
            self.allocation_weights = {name: custom_weight for name in self.selected_assets}
            
        elif self.ratio_radio.isChecked():
            total_target = self.total_weight_spinbox.value()
            weight_per_asset = total_target / selected_count
            preview_text = f"比例配分: 各資産 {weight_per_asset:.2f}%\n"
            self.allocation_weights = {name: weight_per_asset for name in self.selected_assets}
        
        preview_text += f"総ウエイト: {sum(self.allocation_weights.values()):.2f}%\n\n"
        for asset_name in sorted(self.selected_assets):
            preview_text += f"• {asset_name}: {self.allocation_weights[asset_name]:.2f}%\n"
        
        self.preview_label.setText(preview_text)
    
    def get_allocation_weights(self) -> Dict[str, float]:
        return self.allocation_weights.copy()


class AssetSearchWidget(QWidget):
    asset_selected = Signal(AssetInfo)
    
    def __init__(self):
        super().__init__()
        self.searcher = AssetSearcher()
        self.search_thread = None
        self.search_timer = QTimer()
        self.search_results = []
        self.styles = AnalysisStyles()
        self._setup_ui()
        self._setup_connections()
    
    def _setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        # タイトル
        title_label = QLabel("資産検索")
        title_label.setStyleSheet("font-weight: bold; font-size: 12px; margin-bottom: 5px;")
        title_label.setFixedHeight(20)
        title_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        layout.addWidget(title_label, 0, Qt.AlignTop)
        
        # 検索入力
        search_layout = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("銘柄名・証券コード・ティッカーを入力")
        self.search_input.setFixedHeight(30)
        search_layout.addWidget(self.search_input)
        
        self.search_button = QPushButton("検索")
        self.search_button.setStyleSheet(self.styles.get_button_style_by_type("secondary"))
        self.search_button.setMaximumWidth(60)
        search_layout.addWidget(self.search_button)
        
        # 検索レイアウトのコンテナ
        search_container = QWidget()
        search_container.setLayout(search_layout)
        search_container.setFixedHeight(40)
        layout.addWidget(search_container)
        
        # CSV読み込みボタン
        csv_layout = QHBoxLayout()
        csv_layout.setContentsMargins(0, 0, 0, 5)
        
        # 左側のスペーサー（検索入力欄と同じ幅）
        csv_layout.addStretch()

        self.csv_import_button = QPushButton("CSV読込")
        self.csv_import_button.setStyleSheet(self.styles.get_button_style_by_type("secondary"))
        self.csv_import_button.setMaximumWidth(60)
        self.csv_import_button.setToolTip("CSVファイルから証券コードを一括読み込み")
        self.csv_import_button.clicked.connect(self._open_csv_import_dialog)
        csv_layout.addWidget(self.csv_import_button)
        
        layout.addLayout(csv_layout)
        
        # プログレス・ステータス
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumHeight(3)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #666; font-size: 10px;")
        self.status_label.setFixedHeight(15)
        layout.addWidget(self.status_label)
        
        # 検索結果
        self.results_tree = QTreeWidget()
        self.results_tree.setHeaderLabels(["名前", "シンボル", "種別", "取引所"])
        self.results_tree.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.results_tree, 1)
        
        self.setLayout(layout)
    
    def _setup_connections(self):
        self.search_input.textChanged.connect(self._on_search_text_changed)
        self.search_input.returnPressed.connect(self._start_search)
        self.search_button.clicked.connect(self._start_search)
        self.results_tree.itemDoubleClicked.connect(self._on_item_double_clicked)
        
        self.search_timer.setSingleShot(True)
        self.search_timer.timeout.connect(self._start_search)
    
    def _on_search_text_changed(self, text: str):
        if len(text.strip()) >= 2:
            self.search_timer.stop()
            self.search_timer.start(500)
        else:
            self.search_timer.stop()
            self.results_tree.clear()
            self.status_label.setText("例: NTT, AAPL, 8306")
    
    def _start_search(self):
        query = self.search_input.text().strip()
        if not query or len(query) < 2:
            self.results_tree.clear()
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
        self.progress_bar.setVisible(searching)
        self.search_button.setEnabled(not searching)
        if searching:
            self.progress_bar.setRange(0, 0)
            self.status_label.setText("検索中...")
    
    def _on_search_completed(self, results: List[AssetInfo]):
        self._set_search_state(False)
        self.search_results = results
        
        if not results:
            self.status_label.setText("見つかりません")
            self.results_tree.clear()
            return
        
        self.status_label.setText(f"{len(results)}件 - ダブルクリックで選択")
        self._display_results(results)
    
    def _on_search_error(self, error_message: str):
        self._set_search_state(False)
        self.status_label.setText(f"検索エラー: {error_message}")
        self.results_tree.clear()
    
    def _display_results(self, results: List[AssetInfo]):
        self.results_tree.clear()
        
        for asset in results:
            item = QTreeWidgetItem([
                asset.name, asset.symbol,
                asset.get_sector_or_type(),
                asset.exchange or "-"
            ])
            item.setData(0, Qt.UserRole, asset)
            
            # ツールチップ
            tooltip_parts = [f"名前: {asset.name}", f"シンボル: {asset.symbol}"]
            for label, value in [("セクター", asset.sector), ("取引所", asset.exchange), 
                               ("通貨", asset.currency), ("国", asset.country)]:
                if value:
                    tooltip_parts.append(f"{label}: {value}")
            item.setToolTip(0, "\n".join(tooltip_parts))
            
            self.results_tree.addTopLevelItem(item)
    
    def _on_item_double_clicked(self, item: QTreeWidgetItem, column: int):
        asset = item.data(0, Qt.UserRole)
        if isinstance(asset, AssetInfo):
            self.asset_selected.emit(asset)
    
    def _open_csv_import_dialog(self):
        """CSV読み込みダイアログを開く"""
        dialog = CSVImportDialog(self)
        dialog.portfolio_created.connect(self._on_csv_portfolio_created)
        dialog.exec()
    
    def _on_csv_portfolio_created(self, portfolio: Portfolio):
        """CSVから作成されたポートフォリオを受け取る"""
        # 親ウィンドウ（ManagementTab）にポートフォリオを設定
        parent = self.parent()
        while parent and not isinstance(parent, ManagementTab):
            parent = parent.parent()
        
        if parent and isinstance(parent, ManagementTab):
            # load_optimized_portfolioメソッドを使用（CSV読み込み専用メッセージに変更）
            try:
                # 編集タブに切り替え
                parent.tab_widget.setCurrentIndex(0)
                
                # ポートフォリオを編集ウィジェットに設定
                parent.portfolio_edit_widget.set_portfolio(portfolio)
                
                # 情報ウィジェットのファイル名をクリア（新規として扱う）
                parent.portfolio_edit_widget.info_widget.current_file_name = None
                
                # ステータス表示
                parent.portfolio_edit_widget.info_widget.status_label.setText(
                    f"CSVから{len(portfolio.positions)}件の資産を読み込みました"
                )
                
                self.status_label.setText(
                    f"CSVから{len(portfolio.positions)}件の資産を読み込みました"
                )
            except Exception as e:
                import logging
                logging.error(f"CSV読み込みエラー: {e}")
                self.status_label.setText(f"エラー: {str(e)}")


class PortfolioInfoWidget(QWidget):
    info_changed = Signal()
    
    def __init__(self):
        super().__init__()
        self.portfolio: Optional[Portfolio] = None
        # 設定を読み込み
        from config.app_config import get_config
        self.config = get_config()

        # カスタムフォルダパスを取得
        custom_folder = self.config.get_portfolio_folder()
        if custom_folder:
            self.manager = PortfolioManager(custom_folder)
        else:
            self.manager = PortfolioManager()
        self.current_file_name: Optional[str] = None
        self.styles = AnalysisStyles()
        self._setup_ui()
        self._setup_connections()
    
    def _setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        # タイトル
        title_label = QLabel("ポートフォリオ情報")
        title_label.setStyleSheet("font-weight: bold; font-size: 12px; margin-bottom: 5px;")
        title_label.setFixedHeight(20)
        title_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        layout.addWidget(title_label, 0, Qt.AlignTop)
        
        # ボタン
        button_layout = QHBoxLayout()
        self.new_button = QPushButton("新規作成")
        self.new_button.setStyleSheet(self.styles.get_button_style_by_type("secondary"))
        self.new_button.setMaximumWidth(70)
        button_layout.addWidget(self.new_button)

        self.save_button = QPushButton("保存")
        self.save_button.setStyleSheet(self.styles.get_button_style_by_type("save"))
        self.save_button.setMaximumWidth(50)
        button_layout.addWidget(self.save_button)

        self.save_as_button = QPushButton("名前を付けて保存")
        self.save_as_button.setStyleSheet(self.styles.get_button_style_by_type("save"))
        self.save_as_button.setMaximumWidth(120)
        button_layout.addWidget(self.save_as_button)
        
        button_layout.addStretch()
        button_container = QWidget()
        button_container.setLayout(button_layout)
        button_container.setFixedHeight(40)
        button_container.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        layout.addWidget(button_container, 0, Qt.AlignTop)
        
        # 情報入力
        info_group = QGroupBox()
        info_layout = QVBoxLayout(info_group)
        
        # 名前
        name_layout = QHBoxLayout()
        name_label = QLabel("名前:")
        name_label.setFixedWidth(60)
        name_layout.addWidget(name_label)
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("ポートフォリオ名を入力")
        self.name_input.setFixedHeight(25)
        name_layout.addWidget(self.name_input)
        info_layout.addLayout(name_layout)
        
        # 作成日時と更新日時の表示
        dates_layout = QVBoxLayout()
        
        # 作成日時
        created_layout = QHBoxLayout()
        created_label = QLabel("作成日時:")
        created_label.setFixedHeight(18)
        created_layout.addWidget(created_label)
        self.created_at_label = QLabel("未設定")
        self.created_at_label.setStyleSheet("color: #666; font-size: 10px;")
        self.created_at_label.setFixedHeight(18)
        created_layout.addWidget(self.created_at_label)
        created_layout.addStretch()
        dates_layout.addLayout(created_layout)
        
        # 更新日時
        modified_layout = QHBoxLayout()
        modified_label = QLabel("更新日時:")
        modified_label.setFixedHeight(18)
        modified_layout.addWidget(modified_label)
        self.modified_at_label = QLabel("未設定")
        self.modified_at_label.setStyleSheet("color: #666; font-size: 10px;")
        self.modified_at_label.setFixedHeight(18)
        modified_layout.addWidget(self.modified_at_label)
        modified_layout.addStretch()
        dates_layout.addLayout(modified_layout)
        
        info_layout.addLayout(dates_layout)
        
        # 説明
        desc_layout = QVBoxLayout()
        desc_label = QLabel("説明:")
        desc_label.setFixedHeight(18)
        desc_layout.addWidget(desc_label)
        
        self.desc_input = QTextEdit()
        self.desc_input.setPlaceholderText("ポートフォリオの説明（任意）")
        self.desc_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.desc_input.setMinimumHeight(50)
        desc_layout.addWidget(self.desc_input)
        
        info_layout.addLayout(desc_layout)
        
        info_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(info_group, 1)
        
        # ステータス
        self.status_label = QLabel("準備完了")
        self.status_label.setStyleSheet("color: #666; font-size: 10px;")
        self.status_label.setFixedHeight(15)
        self.status_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        layout.addWidget(self.status_label, 0, Qt.AlignTop)
        
        self.setLayout(layout)
    
    def _setup_connections(self):
        self.name_input.textChanged.connect(self._on_info_changed)
        self.desc_input.textChanged.connect(self._on_info_changed)
        
        self.new_button.clicked.connect(self._new_portfolio)
        self.save_button.clicked.connect(self.save_portfolio)
        self.save_as_button.clicked.connect(self.save_portfolio_as)
    
    def set_portfolio(self, portfolio: Portfolio):
        """ポートフォリオを設定"""
        print(f"PortfolioInfoWidget.set_portfolio called")
        print(f"  Portfolio name: {portfolio.name}")
        print(f"  Portfolio description: {portfolio.description}")
        
        self.portfolio = portfolio
    
        # UIを更新（内部でシグナルがブロックされる）
        self._update_ui()
    
    def _update_ui(self):
        """UI表示を更新"""
        # シグナルを一時的にブロック（_update_ui内で常にブロック）
        self.name_input.blockSignals(True)
        self.desc_input.blockSignals(True)
        
        try:
            if not self.portfolio:
                # ポートフォリオがない場合は全てクリア
                self.name_input.clear()
                self.desc_input.clear()
                self.created_at_label.setText("未設定")
                self.modified_at_label.setText("未設定")
                return
            
            print(f"PortfolioInfoWidget._update_ui called")
            print(f"  Updating with portfolio: {self.portfolio.name}")
            print(f"  Description: '{self.portfolio.description}'")
            
            # 名前を設定
            self.name_input.setText(self.portfolio.name)
            
            # 説明を設定（確実にクリアしてから設定）
            self.desc_input.clear()
            if self.portfolio.description:
                self.desc_input.setPlainText(self.portfolio.description)
            
            print(f"  After setting description: '{self.desc_input.toPlainText()}'")
            
            # 作成日時の表示
            if hasattr(self.portfolio, 'created_at') and self.portfolio.created_at:
                if isinstance(self.portfolio.created_at, str):
                    try:
                        dt = datetime.fromisoformat(self.portfolio.created_at)
                        self.created_at_label.setText(dt.strftime('%Y-%m-%d %H:%M:%S'))
                    except (ValueError, TypeError) as e:
                        logging.debug(f"Failed to parse created_at: {e}")
                        self.created_at_label.setText(self.portfolio.created_at)
                else:
                    self.created_at_label.setText(self.portfolio.created_at.strftime('%Y-%m-%d %H:%M:%S'))
            else:
                self.created_at_label.setText("未設定")
            
            # 更新日時の表示
            if hasattr(self.portfolio, 'modified_at') and self.portfolio.modified_at:
                if isinstance(self.portfolio.modified_at, str):
                    try:
                        dt = datetime.fromisoformat(self.portfolio.modified_at)
                        self.modified_at_label.setText(dt.strftime('%Y-%m-%d %H:%M:%S'))
                    except (ValueError, TypeError) as e:
                        logging.debug(f"Failed to parse modified_at: {e}")
                        self.modified_at_label.setText(self.portfolio.modified_at)
                else:
                    self.modified_at_label.setText(self.portfolio.modified_at.strftime('%Y-%m-%d %H:%M:%S'))
            else:
                self.modified_at_label.setText("未設定")
        
        finally:
            # シグナルのブロックを解除
            self.name_input.blockSignals(False)
            self.desc_input.blockSignals(False)
    
    def _on_info_changed(self):
        """情報変更時の処理"""
        if not self.portfolio:
            return
        
        # ポートフォリオオブジェクトを更新
        self.portfolio.name = self.name_input.text()
        self.portfolio.description = self.desc_input.toPlainText()
        
        print(f"PortfolioInfoWidget._on_info_changed")
        print(f"  New description: '{self.portfolio.description}'")
        
        self.info_changed.emit()
    
    def _new_portfolio(self):
        from ui.tab_management import ManagementTab
        parent = self.parent()
        while parent and not isinstance(parent, ManagementTab):
            parent = parent.parent()
        
        if parent:
            parent.new_portfolio()
    
    def _check_duplicate_name(self, portfolio_name: str, exclude_file: str = None) -> bool:
        """ポートフォリオ名の重複をチェック
        
        Args:
            portfolio_name: チェックする名前
            exclude_file: 除外するファイル名（上書き保存時の自分自身）
        
        Returns:
            bool: 重複している場合True
        """
        try:
            # 既存のポートフォリオ一覧を取得
            portfolios = self.manager.list_portfolios()
            
            for file_name, existing_name, _ in portfolios:
                # 自分自身は除外
                if exclude_file and file_name == exclude_file:
                    continue
                
                # 名前が一致する場合は重複
                if existing_name == portfolio_name:
                    return True
            
            return False
        
        except Exception as e:
            print(f"重複チェックエラー: {e}")
            return False
    
    def save_portfolio(self):
        """ポートフォリオを保存"""
        if not self.portfolio:
            self.status_label.setText("保存するポートフォリオがありません")
            return
        
        # バリデーション
        errors = self.portfolio.validate()
        if errors:
            QMessageBox.warning(self, "入力エラー", "\n".join(errors))
            return
        
        # 名前の重複チェック（上書き保存時は自分自身を除外）
        if self._check_duplicate_name(self.portfolio.name, exclude_file=self.current_file_name):
            QMessageBox.warning(
                self,
                "名前の重複",
                f"ポートフォリオ名「{self.portfolio.name}」は既に使用されています．\n\n"
                f"別の名前を指定してください．"
            )
            return
        
        # 作成日時が未設定の場合は現在時刻を設定
        if not hasattr(self.portfolio, 'created_at') or not self.portfolio.created_at:
            self.portfolio.created_at = datetime.now()
        
        # 更新日時を明示的に設定
        self.portfolio.modified_at = datetime.now()
        
        print(f"Saving portfolio: {self.portfolio.name}")
        print(f"  Description: '{self.portfolio.description}'")
        print(f"  Modified at: {self.portfolio.modified_at}")
        
        try:
            if self.current_file_name:
                # 上書き保存
                result_file_name = self.manager.save_portfolio(self.portfolio, self.current_file_name)
                self.status_label.setText(f"保存完了: {result_file_name}")
                print(f"  Saved to existing file: {result_file_name}")
            else:
                # 新規保存
                result_file_name = self.manager.save_portfolio(self.portfolio)
                self.current_file_name = result_file_name
                self.status_label.setText(f"保存完了: {result_file_name}")
                print(f"  Saved to new file: {result_file_name}")
            
            # UI更新（日時表示を更新）
            self._update_ui()
            
            # 管理リストを更新
            self._refresh_management_list()
            
        except Exception as e:
            self.status_label.setText("保存失敗")
            QMessageBox.critical(self, "保存エラー", f"保存中にエラーが発生しました:\n{str(e)}")
    
    def save_portfolio_as(self):
        """名前を付けて保存"""
        if not self.portfolio:
            self.status_label.setText("保存するポートフォリオがありません")
            return
        
        # バリデーション
        errors = self.portfolio.validate()
        if errors:
            QMessageBox.warning(self, "入力エラー", "\n".join(errors))
            return
        
        # 名前の重複チェック（名前を付けて保存時は除外ファイルなし）
        if self._check_duplicate_name(self.portfolio.name):
            QMessageBox.warning(
                self,
                "名前の重複",
                f"ポートフォリオ名「{self.portfolio.name}」は既に使用されています．\n\n"
                f"別の名前を指定してください．"
            )
            return

        # 作成日時が未設定の場合は現在時刻を設定
        if not hasattr(self.portfolio, 'created_at') or not self.portfolio.created_at:
            self.portfolio.created_at = datetime.now()
        
        # 更新日時を明示的に設定
        self.portfolio.modified_at = datetime.now()
        
        # ファイル選択ダイアログ
        file_path, _ = QFileDialog.getSaveFileName(
            self, "ポートフォリオを保存", f"{self.portfolio.name}.json", "JSON ファイル (*.json)"
        )
        
        if file_path:
            try:
                self.portfolio.save_to_file(file_path)
                self.current_file_name = os.path.basename(file_path)
                self.status_label.setText(f"保存完了: {self.current_file_name}")
                
                # UI更新（日時表示を更新）
                self._update_ui()
                
                # 管理リストを更新
                self._refresh_management_list()
                
            except Exception as e:
                self.status_label.setText("保存失敗")
                QMessageBox.critical(self, "保存エラー", f"保存中にエラーが発生しました:\n{str(e)}")
    
    def _refresh_management_list(self):
        """管理リストを更新"""
        try:
            from ui.tab_management import ManagementTab
            parent = self.parent()
            while parent and not isinstance(parent, ManagementTab):
                parent = parent.parent()
            
            if parent and hasattr(parent, 'portfolio_list_widget'):
                print("Refreshing management list...")
                parent.portfolio_list_widget.refresh_list()
        except Exception as e:
            print(f"Failed to refresh management list: {e}")


class PortfolioAssetsWidget(QWidget):
    portfolio_changed = Signal()
    
    def __init__(self):
        super().__init__()
        self.portfolio: Optional[Portfolio] = None
        self.weight_widgets: Dict[str, QDoubleSpinBox] = {}
        self.styles = AnalysisStyles()
        self._setup_ui()
        self._setup_connections()
    
    def _setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        # タイトル
        title_label = QLabel("資産とウエイト")
        title_label.setStyleSheet("font-weight: bold; font-size: 12px; margin-bottom: 5px;")
        title_label.setFixedHeight(20)
        title_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        layout.addWidget(title_label, 0, Qt.AlignTop)
        
        # ツールバー
        toolbar_layout = QHBoxLayout()
        
        self.bulk_allocation_button = QPushButton("一括設定")
        self.bulk_allocation_button.setStyleSheet(self.styles.get_button_style_by_type("secondary"))
        self.bulk_allocation_button.setMaximumWidth(80)
        self.bulk_allocation_button.setToolTip("ウエイトを一括設定します")
        toolbar_layout.addWidget(self.bulk_allocation_button)

        self.normalize_button = QPushButton("正規化")
        self.normalize_button.setStyleSheet(self.styles.get_button_style_by_type("secondary"))
        self.normalize_button.setMaximumWidth(60)
        self.normalize_button.setToolTip("ウエイトの合計を100%に調整します")
        toolbar_layout.addWidget(self.normalize_button)

        self.clear_button = QPushButton("全削除")
        self.clear_button.setStyleSheet(self.styles.get_button_style_by_type("danger"))
        self.clear_button.setMaximumWidth(60)
        toolbar_layout.addWidget(self.clear_button)
        
        toolbar_layout.addStretch()
        
        self.weight_sum_label = QLabel("合計: 0.0%")
        self.weight_sum_label.setStyleSheet("font-weight: bold;")
        toolbar_layout.addWidget(self.weight_sum_label)
        
        toolbar_container = QWidget()
        toolbar_container.setLayout(toolbar_layout)
        toolbar_container.setFixedHeight(40)
        toolbar_container.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        layout.addWidget(toolbar_container, 0, Qt.AlignTop)
        
        # 資産リスト
        self.assets_tree = QTreeWidget()
        self.assets_tree.setHeaderLabels(["資産名", "シンボル", "ウエイト (%)", "削除"])
        self.assets_tree.setAlternatingRowColors(True)
        self.assets_tree.setRootIsDecorated(False)
        self.assets_tree.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        header = self.assets_tree.header()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        for i in range(1, 4):
            header.setSectionResizeMode(i, QHeaderView.ResizeToContents)

        layout.addWidget(self.assets_tree, 1)
        
        # 空状態表示
        self.empty_label = QLabel("資産が選択されていません")
        self.empty_label.setAlignment(Qt.AlignCenter)
        self.empty_label.setStyleSheet("""
            QLabel {
                color: #999; font-size: 12px; background-color: #f8f9fa;
                border: 1px dashed #ccc; border-radius: 4px; padding: 20px; margin: 5px;
            }
        """)
        self.empty_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.empty_label.setMinimumHeight(80)
        layout.addWidget(self.empty_label, 1)
        
        self.setLayout(layout)
    
    def _setup_connections(self):
        self.bulk_allocation_button.clicked.connect(self._show_bulk_allocation_dialog)
        self.normalize_button.clicked.connect(self.normalize_weights)
        self.clear_button.clicked.connect(self.clear_all_assets)
    
    def set_portfolio(self, portfolio: Portfolio):
        self.portfolio = portfolio
        self._update_ui()
    
    def _update_ui(self):
        if not self.portfolio:
            return
        
        self._update_assets_tree()
        self._update_weight_sum()
    
    def _update_assets_tree(self):
        self.assets_tree.clear()
        self.weight_widgets.clear()
        
        has_assets = bool(self.portfolio and self.portfolio.positions)
        self.empty_label.setVisible(not has_assets)
        self.assets_tree.setVisible(has_assets)
        self.bulk_allocation_button.setEnabled(has_assets)
        
        if not has_assets:
            return
        
        for position in self.portfolio.positions:
            item = QTreeWidgetItem([position.asset.name, position.asset.symbol, "", ""])
            item.setData(0, Qt.UserRole, position)
            
            # ツールチップ
            tooltip_parts = [f"名前: {position.asset.name}", f"シンボル: {position.asset.symbol}"]
            for label, value in [("セクター", position.asset.sector), ("取引所", position.asset.exchange)]:
                if value:
                    tooltip_parts.append(f"{label}: {value}")
            item.setToolTip(0, "\n".join(tooltip_parts))
            
            self.assets_tree.addTopLevelItem(item)
            
            # ウエイト入力
            weight_spinbox = QDoubleSpinBox()
            weight_spinbox.setRange(0.0, 1000.0)
            weight_spinbox.setSingleStep(1.0)
            weight_spinbox.setDecimals(2)
            weight_spinbox.setValue(position.weight * 100)
            weight_spinbox.valueChanged.connect(
                lambda value, symbol=position.asset.symbol: self._on_weight_changed(symbol, value)
            )
            self.weight_widgets[position.asset.symbol] = weight_spinbox
            self.assets_tree.setItemWidget(item, 2, weight_spinbox)
            
            # 削除ボタン
            delete_button = QPushButton("削除")
            delete_button.setMaximumWidth(50)
            delete_button.setMaximumHeight(25)
            delete_button.clicked.connect(
                lambda checked, symbol=position.asset.symbol: self.remove_asset(symbol)
            )
            self.assets_tree.setItemWidget(item, 3, delete_button)
    
    def _update_weight_sum(self):
        if not self.portfolio:
            total_weight = 0.0
        else:
            total_weight = self.portfolio.total_weight * 100
        
        # 色分け
        colors = {
            "high": "#ff6b6b",    # 赤（レバレッジ）
            "low": "#feca57",     # 黄（低投資）
            "normal": "#48dbfb"   # 青（正常）
        }
        
        if total_weight > 100:
            color = colors["high"]
        elif total_weight < 95:
            color = colors["low"]
        else:
            color = colors["normal"]
        
        self.weight_sum_label.setText(f"合計: {total_weight:.1f}%")
        self.weight_sum_label.setStyleSheet(f"color: {color}; font-weight: bold;")
    
    def add_asset(self, asset: AssetInfo, weight: float = 10.0):
        if not self.portfolio:
            return False
        
        if self.portfolio.add_position(asset, weight / 100.0):
            self._update_ui()
            self.portfolio_changed.emit()
            return True
        else:
            return False
    
    def remove_asset(self, symbol: str):
        if not self.portfolio:
            return
        
        if self.portfolio.remove_position(symbol):
            self._update_ui()
            self.portfolio_changed.emit()
    
    def _on_weight_changed(self, symbol: str, value: float):
        if not self.portfolio:
            return
        
        self.portfolio.update_position_weight(symbol, value / 100.0)
        self._update_weight_sum()
        self.portfolio_changed.emit()
    
    def _show_bulk_allocation_dialog(self):
        if not self.portfolio or not bool(self.portfolio.positions):
            return
        
        asset_names = [pos.asset.name for pos in self.portfolio.positions]
        dialog = WeightAllocationDialog(asset_names, self)
        
        if dialog.exec() == QDialog.Accepted:
            allocation_weights = dialog.get_allocation_weights()
            
            for position in self.portfolio.positions:
                if position.asset.name in allocation_weights:
                    new_weight = allocation_weights[position.asset.name] / 100.0
                    position.weight = new_weight
            
            self._update_ui()
            self.portfolio_changed.emit()
    
    def normalize_weights(self):
        if not self.portfolio or not bool(self.portfolio.positions):
            return
        
        reply = QMessageBox.question(
            self, "正規化確認",
            "ウエイトの合計を100%に調整しますか？\n（各ウエイトが比例的に調整されます）",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.portfolio.normalize_weights()
            self._update_ui()
            self.portfolio_changed.emit()
    
    def clear_all_assets(self):
        if not self.portfolio or not bool(self.portfolio.positions):
            return
        
        reply = QMessageBox.question(
            self, "全削除確認", "全ての資産を削除しますか？",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.portfolio.clear_positions()
            self._update_ui()
            self.portfolio_changed.emit()


class PortfolioSummaryWidget(QWidget):
    rate_changed = Signal()
    
    def __init__(self):
        super().__init__()
        self.portfolio: Optional[Portfolio] = None
        self.initial_rate_fetched = False
        self.styles = AnalysisStyles()
        self._setup_ui()
        self._setup_connections()
        self._fetch_initial_interest_rate()
    
    def _setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        # タイトル
        title_label = QLabel("ポートフォリオサマリー")
        title_label.setStyleSheet("font-weight: bold; font-size: 12px; margin-bottom: 5px;")
        title_label.setFixedHeight(20)
        title_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        layout.addWidget(title_label, 0, Qt.AlignTop)
        
        # 統計情報
        stats_group = QGroupBox()
        stats_group.setMaximumHeight(80)
        stats_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        stats_layout = QVBoxLayout(stats_group)
        
        # 基本情報
        basic_info_layout = QHBoxLayout()
        self.position_count_label = QLabel("銘柄数: 0")
        self.total_weight_label = QLabel("総ウエイト: 0.0%")
        basic_info_layout.addWidget(self.position_count_label)
        basic_info_layout.addWidget(self.total_weight_label)
        basic_info_layout.addStretch()
        stats_layout.addLayout(basic_info_layout)
        
        # ポジション情報
        position_info_layout = QHBoxLayout()
        self.cash_position_label = QLabel("キャッシュポジション: 100.0%")
        self.leverage_label = QLabel("レバレッジ: なし")
        position_info_layout.addWidget(self.cash_position_label)
        position_info_layout.addWidget(self.leverage_label)
        position_info_layout.addStretch()
        stats_layout.addLayout(position_info_layout)
        
        # 利子率設定
        rate_group = QGroupBox("利子率設定")
        rate_group.setMaximumHeight(90) 
        rate_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        layout.addWidget(rate_group, 0, Qt.AlignTop)
        rate_layout = QVBoxLayout(rate_group)
        
        rate_input_layout = QHBoxLayout()
        rate_input_layout.addWidget(QLabel("利子率:"))
        
        self.interest_rate_spinbox = QDoubleSpinBox()
        self.interest_rate_spinbox.setRange(-10.0, 20.0)
        self.interest_rate_spinbox.setSingleStep(0.01)
        self.interest_rate_spinbox.setDecimals(3)
        self.interest_rate_spinbox.setSuffix("%")
        self.interest_rate_spinbox.setMaximumWidth(100)
        rate_input_layout.addWidget(self.interest_rate_spinbox)
        
        self.fetch_rate_button = QPushButton("取得")
        self.fetch_rate_button.setStyleSheet(self.styles.get_button_style_by_type("secondary"))
        self.fetch_rate_button.setMaximumWidth(40)
        self.fetch_rate_button.setToolTip("短期国債金利を取得")
        rate_input_layout.addWidget(self.fetch_rate_button)

        rate_layout.addLayout(rate_input_layout)
        
        # ステータス
        rate_status_layout = QHBoxLayout()
        self.rate_status_label = QLabel("初期化中...")
        self.rate_status_label.setStyleSheet("font-size: 9px; color: #666;")
        rate_status_layout.addWidget(self.rate_status_label)

        rate_layout.addLayout(rate_status_layout)
        
        layout.addWidget(stats_group, 0, Qt.AlignTop)
        layout.addStretch(1)
        self.setLayout(layout)
    
    def _setup_connections(self):
        self.interest_rate_spinbox.valueChanged.connect(self._on_rate_changed)
        self.fetch_rate_button.clicked.connect(self._fetch_interest_rate)
    
    def set_portfolio(self, portfolio: Portfolio):
        self.portfolio = portfolio
        
        if self.initial_rate_fetched and self.portfolio.cash_interest_rate == 0:
            current_rate = self.interest_rate_spinbox.value()
            self.portfolio.cash_interest_rate = current_rate
        
        self._update_ui()
    
    def _update_ui(self):
        if not self.portfolio:
            return
        
        self.interest_rate_spinbox.setValue(self.portfolio.cash_interest_rate)
        self.update_summary()
    
    def update_summary(self):
        if not self.portfolio:
            self._set_empty_summary()
            return
        
        total_weight = self.portfolio.total_weight * 100
        cash_position = self.portfolio.cash_position * 100
        position_count = len(self.portfolio.positions)
        
        self.position_count_label.setText(f"銘柄数: {position_count}")
        
        # 重みの色分け
        color = self._get_weight_color(total_weight)
        self.total_weight_label.setText(f"総ウエイト: {total_weight:.2f}%")
        self.total_weight_label.setStyleSheet(f"color: {color}; font-weight: bold;")
        
        # ポジション情報
        if self.portfolio.is_leveraged:
            leverage_amount = self.portfolio.leverage_amount * 100
            self.cash_position_label.setText(f"借入: {leverage_amount:.2f}%")
            self.cash_position_label.setStyleSheet("color: #ff6b6b; font-weight: bold;")
            self.leverage_label.setText(f"レバレッジ: {self.portfolio.total_weight:.2f}x")
            self.leverage_label.setStyleSheet("color: #ff6b6b; font-weight: bold;")
        else:
            self.cash_position_label.setText(f"キャッシュ: {cash_position:.2f}%")
            self.cash_position_label.setStyleSheet("color: #54a0ff; font-weight: bold;")
            self.leverage_label.setText("レバレッジ: なし")
            self.leverage_label.setStyleSheet("color: #666;")
    
    def _set_empty_summary(self):
        self.total_weight_label.setText("総ウエイト: 0.0%")
        self.cash_position_label.setText("キャッシュポジション: 100.0%")
        self.leverage_label.setText("レバレッジ: なし")
        self.position_count_label.setText("銘柄数: 0")
    
    def _get_weight_color(self, total_weight):
        if total_weight > 100:
            return "#ff6b6b"  # 赤（レバレッジ）
        elif total_weight < 95:
            return "#feca57"  # 黄（低投資）
        else:
            return "#48dbfb"  # 青（正常）
    
    def _on_rate_changed(self):
        if not self.portfolio:
            return
        
        self.portfolio.cash_interest_rate = self.interest_rate_spinbox.value()
        self.rate_changed.emit()
    
    def _fetch_initial_interest_rate(self):
        self.fetch_rate_button.setEnabled(False)
        self.rate_status_label.setText("初期化中...")
        
        self.initial_rate_thread = InterestRateThread()
        self.initial_rate_thread.rate_fetched.connect(self._on_initial_rate_fetched)
        self.initial_rate_thread.fetch_error.connect(self._on_initial_rate_error)
        self.initial_rate_thread.start()
    
    def _fetch_interest_rate(self):
        self.fetch_rate_button.setEnabled(False)
        self.rate_status_label.setText("取得中...")
        
        self.rate_thread = InterestRateThread()
        self.rate_thread.rate_fetched.connect(self._on_rate_fetched)
        self.rate_thread.fetch_error.connect(self._on_rate_fetch_error)
        self.rate_thread.start()
    
    def _on_initial_rate_fetched(self, rate: float):
        self.interest_rate_spinbox.setValue(rate)
        self.rate_status_label.setText(f"短期国債金利: {rate:.3f}%")
        self.fetch_rate_button.setEnabled(True)
        self.initial_rate_fetched = True
        
        if self.portfolio and self.portfolio.cash_interest_rate == 0:
            self.portfolio.cash_interest_rate = rate
            self.rate_changed.emit()
    
    def _on_initial_rate_error(self, error: str):
        self.rate_status_label.setText("手動設定")
        self.fetch_rate_button.setEnabled(True)
        self.interest_rate_spinbox.setValue(0.1)
        self.initial_rate_fetched = True
    
    def _on_rate_fetched(self, rate: float):
        self.interest_rate_spinbox.setValue(rate)
        self.rate_status_label.setText(f"短期国債金利: {rate:.3f}%")
        self.fetch_rate_button.setEnabled(True)
        
        if self.portfolio:
            self.portfolio.cash_interest_rate = rate
            self.rate_changed.emit()
    
    def _on_rate_fetch_error(self, error: str):
        self.rate_status_label.setText("取得失敗")
        self.fetch_rate_button.setEnabled(True)
        QMessageBox.warning(self, "金利取得エラー", f"短期国債金利の取得に失敗しました:\n{error}")


class PortfolioEditWidget(QWidget):
    portfolio_changed = Signal()
    
    def __init__(self):
        super().__init__()
        self.portfolio: Optional[Portfolio] = None
        
        # 子ウィジェット
        self.asset_search_widget = AssetSearchWidget()
        self.assets_widget = PortfolioAssetsWidget()
        self.info_widget = PortfolioInfoWidget()
        self.summary_widget = PortfolioSummaryWidget()
        
        self._setup_ui()
        self._setup_connections()
    
    def _setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        
        # 田の字レイアウト（2x2グリッド）
        main_splitter = QSplitter(Qt.Vertical)
        
        # 上段（左右分割）
        top_splitter = QSplitter(Qt.Horizontal)
        top_splitter.addWidget(self.asset_search_widget)  # 左上：資産検索
        top_splitter.addWidget(self.assets_widget)        # 右上：資産とウエイト
        top_splitter.setSizes([400, 600])
        
        # 下段（左右分割）
        bottom_splitter = QSplitter(Qt.Horizontal)
        bottom_splitter.addWidget(self.info_widget)       # 左下：ポートフォリオ情報
        bottom_splitter.addWidget(self.summary_widget)    # 右下：サマリー
        bottom_splitter.setSizes([400, 600])
        
        main_splitter.addWidget(top_splitter)
        main_splitter.addWidget(bottom_splitter)
        main_splitter.setSizes([600, 400])
        
        layout.addWidget(main_splitter)
        self.setLayout(layout)
    
    def _setup_connections(self):
        # 資産選択
        self.asset_search_widget.asset_selected.connect(self.add_asset)
        
        # ポートフォリオ変更監視
        for widget in [self.info_widget, self.assets_widget, self.summary_widget]:
            if hasattr(widget, 'info_changed'):
                widget.info_changed.connect(self.on_portfolio_changed)
            if hasattr(widget, 'portfolio_changed'):
                widget.portfolio_changed.connect(self.on_portfolio_changed)
            if hasattr(widget, 'rate_changed'):
                widget.rate_changed.connect(self.on_portfolio_changed)
        
        # 資産変更時にサマリーも更新
        self.assets_widget.portfolio_changed.connect(self.summary_widget.update_summary)
    
    def set_portfolio(self, portfolio: Portfolio):
        self.portfolio = portfolio
        for widget in [self.info_widget, self.assets_widget, self.summary_widget]:
            widget.set_portfolio(portfolio)
        self.info_widget.current_file_name = None
    
    def create_new_portfolio(self):
        self.portfolio = Portfolio(name="新規ポートフォリオ")
        
        # 作成日時を設定
        if hasattr(self.portfolio, 'created_at'):
            self.portfolio.created_at = datetime.now()
        
        self.set_portfolio(self.portfolio)
        self.info_widget.status_label.setText("新規ポートフォリオを作成しました")
    
    def add_asset(self, asset: AssetInfo, weight: float = 10.0):
        if not self.portfolio:
            self.create_new_portfolio()
        
        if self.assets_widget.add_asset(asset, weight):
            self.summary_widget.update_summary()
            self.info_widget.status_label.setText(f"資産を追加: {asset.name}")
        else:
            QMessageBox.information(self, "追加できません", f"{asset.name} は既に追加されています")
    
    def on_portfolio_changed(self):
        self.portfolio_changed.emit()


class PortfolioListWidget(QWidget):
    portfolio_selected = Signal(Portfolio)
    
    def __init__(self):
        super().__init__()
        # 設定を読み込み
        from config.app_config import get_config
        self.config = get_config()

        # カスタムフォルダパスを取得
        custom_folder = self.config.get_portfolio_folder()
        if custom_folder:
            self.manager = PortfolioManager(custom_folder)
        else:
            self.manager = PortfolioManager()
        self.styles = AnalysisStyles()
        self._setup_ui()
        self._setup_connections()
        self.refresh_list()
    
    def _setup_ui(self):
        layout = QVBoxLayout()
        
        # ヘッダー
        header_layout = QHBoxLayout()
        
        title_label = QLabel("保存済みポートフォリオ")
        title_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        header_layout.addWidget(title_label)
        
        header_layout.addStretch()
        
        self.refresh_button = QPushButton("更新")
        self.refresh_button.setStyleSheet(self.styles.get_button_style_by_type("secondary"))
        self.refresh_button.setMaximumWidth(50)
        header_layout.addWidget(self.refresh_button)
        
        layout.addLayout(header_layout)
        
        # フォルダパス表示とボタン
        folder_layout = QHBoxLayout()
        folder_layout.setContentsMargins(0, 5, 0, 5)

        folder_label = QLabel("フォルダ:")
        folder_label.setStyleSheet("font-size: 10px;")
        folder_layout.addWidget(folder_label)

        self.folder_path_label = QLabel()
        self.folder_path_label.setStyleSheet("font-size: 10px; color: #666;")
        self.folder_path_label.setWordWrap(False)
        self.folder_path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._update_folder_path_display()
        folder_layout.addWidget(self.folder_path_label, 1)

        self.select_folder_button = QPushButton("フォルダ選択")
        self.select_folder_button.setStyleSheet(self.styles.get_button_style_by_type("secondary"))
        self.select_folder_button.setMaximumWidth(100)
        self.select_folder_button.setToolTip("保存済みポートフォリオを読み込むフォルダを選択")
        folder_layout.addWidget(self.select_folder_button)

        self.reset_folder_button = QPushButton("既定に戻す")
        self.reset_folder_button.setStyleSheet(self.styles.get_button_style_by_type("secondary"))
        self.reset_folder_button.setMaximumWidth(90)
        self.reset_folder_button.setToolTip("既定のフォルダに戻す")
        folder_layout.addWidget(self.reset_folder_button)

        layout.addLayout(folder_layout)

        # リスト
        self.portfolio_tree = QTreeWidget()
        self.portfolio_tree.setHeaderLabels(["名前", "説明", "更新日時", "操作"])
        self.portfolio_tree.setAlternatingRowColors(True)
        self.portfolio_tree.setRootIsDecorated(False)
        
        header = self.portfolio_tree.header()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        
        layout.addWidget(self.portfolio_tree)
        
        # 空状態表示
        self.empty_label = QLabel("保存されたポートフォリオがありません")
        self.empty_label.setAlignment(Qt.AlignCenter)
        self.empty_label.setStyleSheet("""
            QLabel {
                color: #999; font-size: 12px; background-color: #f8f9fa;
                border: 1px dashed #ccc; border-radius: 4px; padding: 20px; margin: 5px;
            }
        """)
        self.empty_label.setVisible(False)
        layout.addWidget(self.empty_label)
        
        self.setLayout(layout)
    
    def _setup_connections(self):
        self.refresh_button.clicked.connect(self.refresh_list)
        self.portfolio_tree.itemDoubleClicked.connect(self._on_item_double_clicked)
        self.select_folder_button.clicked.connect(self._select_custom_folder)
        self.reset_folder_button.clicked.connect(self._reset_to_default_folder)

    def refresh_list(self):
        """リストを更新"""
        print("PortfolioListWidget.refresh_list() called")
        
        self.portfolio_tree.clear()
        
        try:
            # PortfolioManagerから一覧を取得
            portfolios = self.manager.list_portfolios()
            print(f"取得したポートフォリオ数: {len(portfolios)}")
            
            if not portfolios:
                self.empty_label.setVisible(True)
                self.portfolio_tree.setVisible(False)
                return
            
            self.empty_label.setVisible(False)
            self.portfolio_tree.setVisible(True)
            
            for file_name, portfolio_name, modified_at_display in portfolios:
                print(f"追加中: {portfolio_name} - {modified_at_display} ({file_name})")
                
                # ポートフォリオファイルを直接読み込んで最新情報を取得
                description = ""
                actual_modified_at = modified_at_display  # デフォルトはlist_portfoliosから取得した値
                
                try:
                    # ファイルパスを構築
                    file_path = self.manager.portfolio_dir / file_name
                    
                    # JSONファイルを直接読み込む
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # 説明を取得
                    if data.get('description'):
                        desc_text = data['description']
                        if len(desc_text) > 30:
                            desc_text = desc_text[:27] + "..."
                        description = desc_text
                    
                    # 更新日時を取得（最新の情報）
                    if data.get('modified_at'):
                        try:
                            modified_dt = datetime.fromisoformat(data['modified_at'])
                            actual_modified_at = modified_dt.strftime('%Y-%m-%d %H:%M:%S')
                        except (ValueError, TypeError) as e:
                            logging.debug(f"Failed to parse modified_at from data: {e}")
                            pass
                    
                    # ウエイトが全て0%の場合は「未設定」と表示
                    total_weight = data.get('total_weight', 0.0)
                    if total_weight == 0:
                        if description:
                            description = f"{description} [ウエイト未設定]"
                        else:
                            description = "[ウエイト未設定]"
                
                except Exception as e:
                    print(f"ポートフォリオ情報読み込みエラー: {e}")
                    # エラー時はデフォルト値を使用
                
                # 最新の更新日時を使用してアイテムを作成
                item = QTreeWidgetItem([portfolio_name, description, actual_modified_at, ""])
                item.setData(0, Qt.UserRole, file_name)
                
                # ツールチップ設定
                tooltip_parts = [f"ファイル名: {file_name}"]
                if description and not description.startswith("["):
                    tooltip_parts.append(f"説明: {description}")
                tooltip_parts.append(f"更新日時: {actual_modified_at}")
                item.setToolTip(0, "\n".join(tooltip_parts))
                item.setToolTip(1, description if description else "説明なし")
                item.setToolTip(2, f"最終更新: {actual_modified_at}")
                
                self.portfolio_tree.addTopLevelItem(item)
                
                # 操作ボタン
                button_widget = QWidget()
                button_layout = QHBoxLayout(button_widget)
                button_layout.setContentsMargins(2, 2, 2, 2)
                button_layout.setSpacing(2)
                
                load_button = QPushButton("読込")
                load_button.setStyleSheet(self.styles.get_button_style_by_type("secondary"))
                load_button.setMaximumWidth(40)
                load_button.setMaximumHeight(24)
                load_button.setStyleSheet("font-size: 9px;")
                load_button.clicked.connect(lambda checked, fn=file_name: self._load_portfolio(fn))
                button_layout.addWidget(load_button)
                
                delete_button = QPushButton("削除")
                delete_button.setStyleSheet(self.styles.get_button_style_by_type("danger"))
                delete_button.setMaximumWidth(40)
                delete_button.setMaximumHeight(24)
                delete_button.setStyleSheet("font-size: 9px; color: #dc3545;")
                delete_button.clicked.connect(lambda checked, fn=file_name: self._delete_portfolio(fn))
                button_layout.addWidget(delete_button)
                
                self.portfolio_tree.setItemWidget(item, 3, button_widget)
            
            print(f"リスト更新完了: {len(portfolios)}件のポートフォリオを表示")
            
        except Exception as e:
            print(f"ポートフォリオリスト更新エラー: {e}")
            QMessageBox.critical(self, "エラー", f"ポートフォリオ一覧の取得に失敗しました:\n{str(e)}")
    
    def _on_item_double_clicked(self, item: QTreeWidgetItem, column: int):
        file_name = item.data(0, Qt.UserRole)
        self._load_portfolio(file_name)
    
    def _load_portfolio(self, file_name: str):
        """ポートフォリオを読み込み"""
        print(f"PortfolioListWidget._load_portfolio called with: '{file_name}'")
        
        try:
            portfolio = self.manager.load_portfolio(file_name)
            if portfolio:
                # ポートフォリオオブジェクトに元のファイル名を記録
                portfolio._source_file_name = file_name
                print(f"読み込み成功: {portfolio.name} (元ファイル: {file_name})")
                print(f"説明: {portfolio.description}")
                print(f"更新日時: {portfolio.modified_at}")
                
                # ウエイトが全て0%の場合は警告を表示
                if portfolio.total_weight == 0 and len(portfolio.positions) > 0:
                    reply = QMessageBox.question(
                        self, "ウエイト未設定",
                        f"このポートフォリオはウエイトが設定されていません．\n"
                        f"（おそらく個別資産分析タブで作成されました）\n\n"
                        f"このまま読み込みますか？\n"
                        f"読み込み後，ウエイトを設定できます．",
                        QMessageBox.Yes | QMessageBox.No,
                        QMessageBox.Yes
                    )
                    if reply != QMessageBox.Yes:
                        return
                
                self.portfolio_selected.emit(portfolio)
            else:
                print(f"読み込み失敗: {file_name}")
                QMessageBox.critical(self, "読み込みエラー", "ポートフォリオの読み込みに失敗しました")
        except Exception as e:
            print(f"読み込み例外: {e}")
            QMessageBox.critical(self, "読み込みエラー", f"ポートフォリオの読み込み中にエラーが発生しました:\n{str(e)}")
    
    def _delete_portfolio(self, file_name: str):
        """ポートフォリオを削除"""
        # ポートフォリオ名を取得
        portfolio_name = file_name
        for i in range(self.portfolio_tree.topLevelItemCount()):
            item = self.portfolio_tree.topLevelItem(i)
            if item.data(0, Qt.UserRole) == file_name:
                portfolio_name = item.text(0)
                break
        
        reply = QMessageBox.question(
            self, "削除確認",
            f"ポートフォリオ「{portfolio_name}」を削除しますか？\n"
            f"ファイル: {file_name}\n\n"
            f"※この操作は取り消せません．",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            try:
                if self.manager.delete_portfolio(file_name):
                    print(f"ポートフォリオ削除成功: {file_name}")
                    self.refresh_list()
                    QMessageBox.information(self, "削除完了", f"「{portfolio_name}」を削除しました．")
                else:
                    QMessageBox.critical(self, "削除エラー", "ポートフォリオの削除に失敗しました")
            except Exception as e:
                print(f"削除例外: {e}")
                QMessageBox.critical(self, "削除エラー", f"削除中にエラーが発生しました:\n{str(e)}")

    def _update_folder_path_display(self):
        """フォルダパス表示を更新"""
        folder_path = self.manager.get_portfolio_dir()
        # パスが長い場合は省略表示
        if len(folder_path) > 60:
            display_path = "..." + folder_path[-57:]
        else:
            display_path = folder_path
        self.folder_path_label.setText(display_path)
        self.folder_path_label.setToolTip(folder_path)

    def _select_custom_folder(self):
        """カスタムフォルダを選択"""
        current_folder = self.manager.get_portfolio_dir()

        # フォルダ選択ダイアログを表示
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "ポートフォリオフォルダを選択",
            current_folder,
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )

        if folder_path:
            # 選択されたフォルダパスを設定に保存
            self.config.set_portfolio_folder(folder_path)

            # PortfolioManagerを再初期化
            self.manager = PortfolioManager(folder_path)

            # UI更新
            self._update_folder_path_display()
            self.refresh_list()

            QMessageBox.information(
                self,
                "フォルダ変更完了",
                f"ポートフォリオフォルダを変更しました：\n{folder_path}"
            )

    def _reset_to_default_folder(self):
        """既定のフォルダに戻す"""
        reply = QMessageBox.question(
            self,
            "既定のフォルダに戻す",
            "ポートフォリオフォルダを既定の場所に戻しますか？\n\n"
            "※既存のファイルは削除されません．",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            # 設定をクリア
            self.config.set('portfolio.custom_folder', None)
            self.config.save()

            # PortfolioManagerを再初期化（既定のフォルダ）
            self.manager = PortfolioManager()

            # UI更新
            self._update_folder_path_display()
            self.refresh_list()

            QMessageBox.information(
                self,
                "フォルダリセット完了",
                f"ポートフォリオフォルダを既定の場所に戻しました：\n{self.manager.get_portfolio_dir()}"
            )
            
class ManagementTab(QWidget):
    def __init__(self):
        super().__init__()
        self._setup_ui()
        self._setup_connections()
        
        # 初期化
        self.portfolio_edit_widget.create_new_portfolio()
    
    def _setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        
        # タブウィジェット
        self.tab_widget = QTabWidget()
        
        # 編集タブ
        self.portfolio_edit_widget = PortfolioEditWidget()
        self.tab_widget.addTab(self.portfolio_edit_widget, "編集")
        
        # 管理タブ
        manage_tab = QWidget()
        manage_layout = QVBoxLayout(manage_tab)
        
        self.portfolio_list_widget = PortfolioListWidget()
        manage_layout.addWidget(self.portfolio_list_widget)
        
        self.tab_widget.addTab(manage_tab, "管理")
        
        layout.addWidget(self.tab_widget)
        self.setLayout(layout)
    
    def _setup_connections(self):
        self.portfolio_list_widget.portfolio_selected.connect(self._load_portfolio)
    
    def new_portfolio(self):
        """新規ポートフォリオ作成（外部から呼び出し用）"""
        self.tab_widget.setCurrentIndex(0)
        self.portfolio_edit_widget.create_new_portfolio()
    
    def save_portfolio(self):
        """ポートフォリオを保存（外部から呼び出し用）"""
        self.portfolio_edit_widget.info_widget.save_portfolio()
    
    def save_portfolio_as(self):
        """名前を付けて保存（外部から呼び出し用）"""
        self.portfolio_edit_widget.info_widget.save_portfolio_as()
    
    def _load_portfolio(self, portfolio: Portfolio):
        """ポートフォリオを読み込み"""
        self.portfolio_edit_widget.set_portfolio(portfolio)
        
        # 現在のfile_name設定
        if hasattr(portfolio, '_source_file_name'):
            file_name = portfolio._source_file_name
        else:
            # ポートフォリオ名から既存ファイルを検索
            portfolios = self.portfolio_list_widget.manager.list_portfolios()
            file_name = None
            for fn, pf_name, _ in portfolios:
                if pf_name == portfolio.name:
                    file_name = fn
                    break
        
        self.portfolio_edit_widget.info_widget.current_file_name = file_name
        self.tab_widget.setCurrentIndex(0)  # 編集タブに切り替え
    
    def load_optimized_portfolio(self, portfolio):
        """最適化されたポートフォリオを読み込み"""
        try:
            # 編集タブに切り替え
            self.tab_widget.setCurrentIndex(0)
            
            # ポートフォリオを編集ウィジェットに設定
            self.portfolio_edit_widget.set_portfolio(portfolio)
            
            # 情報ウィジェットのファイル名をクリア（新規として扱う）
            self.portfolio_edit_widget.info_widget.current_file_name = None
            
            # ステータス表示
            self.portfolio_edit_widget.info_widget.status_label.setText(
                f"最適化ポートフォリオ「{portfolio.name}」を読み込みました"
            )
            
            # 確認ダイアログを表示
            QMessageBox.information(
                self,
                "ポートフォリオ読み込み完了",
                f"「{portfolio.name}」を読み込みました．\n\n"
                f"資産数: {len(portfolio.positions)}件\n"
                f"総ウエイト: {portfolio.total_weight * 100:.2f}%\n\n"
                f"ウエイトを確認・調整して保存してください．"
            )
        
        except Exception as e:
            import logging
            logging.error(f"最適化ポートフォリオ読み込みエラー: {e}")
            QMessageBox.critical(
                self,
                "エラー",
                f"ポートフォリオの読み込み中にエラーが発生しました:\n{str(e)}"
            )