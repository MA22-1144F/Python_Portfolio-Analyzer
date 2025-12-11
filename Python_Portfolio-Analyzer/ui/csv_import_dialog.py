"""csv_import_dialog.py"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QProgressBar, QTextEdit, QGroupBox, QFileDialog, QMessageBox,
    QRadioButton, QButtonGroup, QFormLayout
)
from PySide6.QtCore import Signal, QThread
from typing import List, Optional
import logging
from datetime import datetime

from ui.csv_importer import CSVImporter, CSVAssetEntry
from data.asset_info import AssetInfo
from data.portfolio import Portfolio
from data.asset_searcher import AssetSearcher


class CSVProcessThread(QThread):
    """CSV処理スレッド"""
    
    progress_updated = Signal(int, int, str)  # (current, total, message)
    asset_found = Signal(AssetInfo, float)  # (asset, weight)
    finished = Signal(int, int, list)  # (success_count, failed_count, errors)
    
    def __init__(self, entries: List[CSVAssetEntry], equal_weight: bool = False):
        super().__init__()
        self.entries = entries
        self.equal_weight = equal_weight
        self.searcher = AssetSearcher()
        self.logger = logging.getLogger(__name__)
        self._is_cancelled = False
    
    def cancel(self):
        """処理をキャンセル"""
        self._is_cancelled = True
    
    def run(self):
        """CSV処理メインロジック"""
        success_count = 0
        failed_count = 0
        errors = []
        total = len(self.entries)
        
        # 均等配分の場合のウエイト計算
        equal_weight_value = 1.0 / total if self.equal_weight else None
        
        for idx, entry in enumerate(self.entries, start=1):
            if self._is_cancelled:
                self.finished.emit(success_count, failed_count, errors)
                return
            
            try:
                # 資産検索
                self.progress_updated.emit(idx, total, f"検索中: {entry.symbol}")
                
                results = self.searcher.search_assets(entry.symbol, max_results=5)
                
                if not results:
                    failed_count += 1
                    errors.append(f"行{entry.row_number}: {entry.symbol} が見つかりませんでした")
                    self.logger.warning(f"資産が見つかりません: {entry.symbol}")
                    continue
                
                # 最初の結果を使用
                asset = results[0]
                
                # ウエイトの決定
                if self.equal_weight:
                    weight = equal_weight_value
                elif entry.weight is not None:
                    weight = entry.weight
                else:
                    weight = 0.0  # ウエイト未指定の場合は0%
                
                # 成功通知
                self.asset_found.emit(asset, weight)
                success_count += 1
                
                self.logger.info(f"資産を追加: {asset.symbol} ({asset.name}) - {weight:.2f}%")
                
            except Exception as e:
                failed_count += 1
                errors.append(f"行{entry.row_number}: {entry.symbol} の処理エラー - {str(e)}")
                self.logger.error(f"資産処理エラー: {entry.symbol}, {e}", exc_info=True)
        
        # 完了通知
        self.finished.emit(success_count, failed_count, errors)


class CSVImportDialog(QDialog):
    """CSV読み込みダイアログ"""
    
    portfolio_created = Signal(Portfolio)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.importer = CSVImporter()
        self.process_thread: Optional[CSVProcessThread] = None
        self.entries: List[CSVAssetEntry] = []
        self.portfolio: Optional[Portfolio] = None
        self.logger = logging.getLogger(__name__)
        self.csv_file_name: Optional[str] = None
        
        self.setWindowTitle("CSVからポートフォリオを作成")
        self.setMinimumWidth(600)
        self.setMinimumHeight(500)
        
        # ダイアログ全体のスタイルを設定
        self.setStyleSheet("""
            QDialog {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QLabel {
                color: #ffffff;
            }
            QGroupBox {
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                color: #ffffff;
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        
        self._setup_ui()
    
    def _setup_ui(self):
        """UI構築"""
        layout = QVBoxLayout()
        layout.setSpacing(10)
        
        # ファイル選択セクション
        layout.addWidget(self._create_file_section())
        
        # ウエイト設定セクション
        layout.addWidget(self._create_weight_section())
        
        # ログ表示
        layout.addWidget(self._create_log_section())
        
        # プログレスバー
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # ボタン
        layout.addLayout(self._create_button_section())
        
        self.setLayout(layout)
    
    def _create_file_section(self) -> QGroupBox:
        """ファイル選択セクション"""
        group = QGroupBox("CSVファイル")
        layout = QVBoxLayout()
        
        # ファイルパス表示
        file_layout = QHBoxLayout()
        self.file_path_label = QLabel("ファイルが選択されていません")
        self.file_path_label.setStyleSheet("color: #ffffff;")
        file_layout.addWidget(self.file_path_label)
        
        # 選択ボタン
        self.select_file_button = QPushButton("ファイルを選択")
        self.select_file_button.clicked.connect(self._select_csv_file)
        file_layout.addWidget(self.select_file_button)
        
        layout.addLayout(file_layout)
        
        # CSVフォーマット説明
        format_label = QLabel(
            "CSV形式:\n"
            "• 必須列: symbol (証券コード)\n"
            "• オプション列: weight (ウエイト %), name (資産名)\n"
        )
        format_label.setStyleSheet("color: #cccccc; font-size: 10px;")
        layout.addWidget(format_label)
        
        group.setLayout(layout)
        return group
    
    def _create_weight_section(self) -> QGroupBox:
        """ウエイト設定セクション"""
        group = QGroupBox("ウエイト設定")
        layout = QFormLayout()
        
        self.weight_button_group = QButtonGroup()
        
        # オプションボタンのスタイル
        radio_style = """
            QRadioButton {
                color: #ffffff;
                spacing: 5px;
                background-color: transparent;
            }
            QRadioButton::indicator {
                width: 15px;
                height: 15px;
                border-radius: 7px;
                border: 2px solid #ffffff;
                background-color: #2b2b2b;
            }
            QRadioButton::indicator:checked {
                background-color: #2196F3;
                border: 2px solid #2196F3;
            }
            QRadioButton::indicator:hover {
                border: 2px solid #2196F3;
            }
        """
        
        # CSVのウエイトを使用
        self.use_csv_weight_radio = QRadioButton("CSVのウエイトを使用")
        self.use_csv_weight_radio.setChecked(True)
        self.use_csv_weight_radio.setToolTip("CSVファイルにweight列がある場合，その値を使用")
        self.use_csv_weight_radio.setStyleSheet(radio_style)
        self.weight_button_group.addButton(self.use_csv_weight_radio)
        layout.addRow(self.use_csv_weight_radio)
        
        # 均等配分
        self.equal_weight_radio = QRadioButton("均等配分")
        self.equal_weight_radio.setToolTip("全ての資産に均等にウエイトを配分")
        self.equal_weight_radio.setStyleSheet(radio_style)
        self.weight_button_group.addButton(self.equal_weight_radio)
        layout.addRow(self.equal_weight_radio)
        
        # ウエイト未設定（0%）
        self.zero_weight_radio = QRadioButton("ウエイト未設定 (0%)")
        self.zero_weight_radio.setToolTip("後でポートフォリオ管理タブでウエイトを設定")
        self.zero_weight_radio.setStyleSheet(radio_style)
        self.weight_button_group.addButton(self.zero_weight_radio)
        layout.addRow(self.zero_weight_radio)
        
        group.setLayout(layout)
        return group
    
    def _create_log_section(self) -> QGroupBox:
        """ログ表示セクション"""
        group = QGroupBox("処理ログ")
        layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(200)
        self.log_text.setStyleSheet("""
            QTextEdit {
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 10px;
                color: #ffffff;
                background-color: #1e1e1e;
                border: 1px solid #555555;
                padding: 5px;
            }
        """)
        layout.addWidget(self.log_text)
        
        group.setLayout(layout)
        return group
    
    def _create_button_section(self) -> QHBoxLayout:
        """ボタンセクション"""
        layout = QHBoxLayout()
        
        # テンプレート作成ボタン
        self.template_button = QPushButton("テンプレート作成")
        self.template_button.clicked.connect(self._create_template)
        layout.addWidget(self.template_button)
        
        layout.addStretch()
        
        # 実行ボタン
        self.execute_button = QPushButton("ポートフォリオ作成")
        self.execute_button.setEnabled(False)
        self.execute_button.clicked.connect(self._execute_import)
        layout.addWidget(self.execute_button)
        
        # キャンセルボタン
        self.cancel_button = QPushButton("閉じる")
        self.cancel_button.clicked.connect(self.reject)
        layout.addWidget(self.cancel_button)
        
        return layout
    
    def _select_csv_file(self):
        """CSVファイル選択"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "CSVファイルを選択", "", "CSV ファイル (*.csv);;すべてのファイル (*)"
        )
        
        if not file_path:
            return
        
        import os
        self.csv_file_name = os.path.splitext(os.path.basename(file_path))[0]
        
        self._log(f"ファイルを選択: {file_path}")
        self.file_path_label.setText(file_path)
        self.file_path_label.setStyleSheet("color: #ffffff;")
        
        # CSVを読み込み
        entries, errors = self.importer.load_from_csv(file_path)
        
        if errors:
            self._log("エラー:")
            for error in errors:
                self._log(f"{error}", is_error=True)
        
        if entries:
            self.entries = entries
            self._log(f"{len(entries)}件の資産を読み込みました")
            
            # バリデーション
            validation_errors = self.importer.validate_entries(entries)
            if validation_errors:
                self._log("警告:")
                for error in validation_errors:
                    self._log(f"  {error}", is_warning=True)
            
            self.execute_button.setEnabled(True)
        else:
            self.entries = []
            self.execute_button.setEnabled(False)
            self._log("有効な資産データが見つかりませんでした", is_error=True)
    
    def _create_template(self):
        """テンプレートCSV作成"""
        from ui.csv_importer import create_csv_template
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "テンプレートを保存", "portfolio_template.csv", "CSV ファイル (*.csv)"
        )
        
        if not file_path:
            return
        
        try:
            create_csv_template(file_path, template_type='standard')
            self._log(f"テンプレートを作成しました: {file_path}")
            QMessageBox.information(
                self, "テンプレート作成",
                f"CSVテンプレートを作成しました．\n\n{file_path}\n\nExcelやテキストエディタで編集してください．"
            )
        except Exception as e:
            self._log(f"テンプレート作成エラー: {e}", is_error=True)
            QMessageBox.critical(self, "エラー", f"テンプレート作成に失敗しました:\n{str(e)}")
    
    def _execute_import(self):
        """CSV読み込み実行"""
        if not self.entries:
            QMessageBox.warning(self, "警告", "CSVファイルを選択してください")
            return
        
        # ポートフォリオ作成
        default_name = self.csv_file_name if self.csv_file_name else f"CSV読込_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.portfolio = Portfolio(
            name=default_name,
            description=f"CSVから作成 ({len(self.entries)}資産)"
        )
        
        # UIの状態変更
        self.execute_button.setEnabled(False)
        self.select_file_button.setEnabled(False)
        self.template_button.setEnabled(False)
        self.cancel_button.setText("キャンセル")
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(len(self.entries))
        self.progress_bar.setValue(0)
        
        self._log(f"\n===== ポートフォリオ作成開始 =====")
        self._log(f"資産数: {len(self.entries)}")
        
        # ウエイト設定モード
        use_equal_weight = self.equal_weight_radio.isChecked()
        
        # 処理スレッド開始
        self.process_thread = CSVProcessThread(self.entries, use_equal_weight)
        self.process_thread.progress_updated.connect(self._on_progress_updated)
        self.process_thread.asset_found.connect(self._on_asset_found)
        self.process_thread.finished.connect(self._on_import_finished)
        self.process_thread.start()
    
    def _on_progress_updated(self, current: int, total: int, message: str):
        """進捗更新"""
        self.progress_bar.setValue(current)
        self._log(f"[{current}/{total}] {message}")
    
    def _on_asset_found(self, asset: AssetInfo, weight: float):
        """資産が見つかった時の処理"""
        self.portfolio.add_position(asset, weight)
        self._log(f"  追加: {asset.symbol} - {asset.name} ({weight:.2f}%)")
    
    def _on_import_finished(self, success_count: int, failed_count: int, errors: List[str]):
        """読み込み完了"""
        self.progress_bar.setVisible(False)
        self.cancel_button.setText("閉じる")
        
        self._log(f"\n===== 完了 =====")
        self._log(f"成功: {success_count}件")
        if failed_count > 0:
            self._log(f"失敗: {failed_count}件", is_error=True)
        
        if errors:
            self._log("\n失敗した資産:")
            for error in errors:
                self._log(f"  {error}", is_error=True)
        
        if success_count > 0:
            # ポートフォリオ名・説明を編集可能にする
            from PySide6.QtWidgets import QInputDialog
            
            name, ok1 = QInputDialog.getText(
                self, "ポートフォリオ名",
                "ポートフォリオ名を入力してください:",
                text=self.portfolio.name
            )
            
            if ok1 and name:
                self.portfolio.name = name
            else:
                self.portfolio.name = ""
            
            description, ok2 = QInputDialog.getText(
                self, "説明",
                "説明を入力してください（オプション）:",
                text=self.portfolio.description
            )
            
            if ok2:
                self.portfolio.description = description
            else:
                self.portfolio.description = ""
            
            # シグナル発行
            self.portfolio_created.emit(self.portfolio)
            
            QMessageBox.information(
                self, "完了",
                f"ポートフォリオを作成しました．\n\n"
                f"名前: {self.portfolio.name}\n"
                f"資産数: {success_count}件\n"
                f"総ウエイト: {self.portfolio.total_weight*100:.2f}%"
            )
            
            self.accept()
        else:
            QMessageBox.warning(
                self, "エラー",
                "有効な資産が1つも見つかりませんでした．\n"
                "CSVファイルの内容を確認してください．"
            )
    
    def _log(self, message: str, is_error: bool = False, is_warning: bool = False):
        """ログ出力"""
        if is_error:
            color = "#ff4444"  # 明るい赤
        elif is_warning:
            color = "#ffaa00"  # オレンジ
        else:
            color = "#ffffff"  # 白（デフォルト）
        
        self.log_text.append(f'<span style="color: {color};">{message}</span>')
    
    def reject(self):
        """ダイアログをキャンセル"""
        if self.process_thread and self.process_thread.isRunning():
            self.process_thread.cancel()
            self.process_thread.wait()
        super().reject()
