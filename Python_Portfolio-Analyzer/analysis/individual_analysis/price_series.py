"""price_series.py"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import logging
import concurrent.futures
import time
from PySide6.QtWidgets import (
    QLabel, QTableWidgetItem, QHeaderView, QProgressBar, QMessageBox,
    QApplication,QFileDialog
)
from PySide6.QtCore import QThread, Signal, Qt
from data.asset_info import AssetInfo
from analysis.analysis_base_widget import AnalysisBaseWidget
from analysis.analysis_size_config import get_min_height, get_min_width
from config.app_config import get_config

def fetch_single_asset_data(args):
    asset, start_date, end_date, interval = args
    logger = logging.getLogger(__name__)
    
    try:
        # yfinanceでデータ取得（auto_adjust=Falseに設定）
        ticker = yf.Ticker(asset.symbol)
        hist = ticker.history(
            start=start_date,
            end=end_date + timedelta(days=1),  # 終了日を含むため+1日
            interval=interval,
            auto_adjust=False,  # Adj Close取得に必須
            prepost=False,      # プレ・ポストマーケットを除外
            repair=True         # データの修復を有効化
        )
        
        if not hist.empty and 'Adj Close' in hist.columns:
            # 調整後終値を取得
            price_series = hist['Adj Close'].copy()
            price_series.name = asset.symbol
            
            # 日付データはyfinanceの元データをそのまま使用
            # タイムゾーン情報がある場合のみ削除
            if hasattr(price_series.index, 'tz') and price_series.index.tz is not None:
                price_series.index = price_series.index.tz_localize(None)
            
            # 日付のみに正規化（時間部分を削除）
            price_series.index = price_series.index.normalize()
            
            # NaN値を除去（補完は行わない）
            price_series = price_series.dropna()
            
            logger.info(f"{asset.symbol}: {len(price_series)}件の実データを取得")
            return {
                'symbol': asset.symbol,
                'name': asset.name,
                'currency': asset.currency or 'USD',  # 通貨情報を追加
                'data': price_series,
                'success': True
            }
        else:
            logger.warning(f"{asset.symbol}: データが取得できませんでした．")
            return {
                'symbol': asset.symbol,
                'name': asset.name,
                'currency': asset.currency or 'USD',
                'data': pd.Series(dtype=float),
                'success': False
            }
    
    except Exception as e:
        logger.error(f"{asset.symbol} のデータ取得エラー: {e}")
        return {
            'symbol': asset.symbol,
            'name': asset.name,
            'currency': asset.currency or 'USD',
            'data': pd.Series(dtype=float),
            'success': False,
            'error': str(e)
        }


class PriceDataFetcher(QThread):
    """価格データ取得スレッド - 応答性向上版"""
    
    progress_updated = Signal(int, str)  # 進捗率，メッセージ
    data_fetched = Signal(dict)  # 取得完了
    error_occurred = Signal(str)  # エラー発生
    
    def __init__(self, assets: List[AssetInfo], start_date: datetime, end_date: datetime, interval: str):
        super().__init__()
        self.assets = assets
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.logger = logging.getLogger(__name__)
    
    def run(self):
        """データ取得実行"""
        try:
            total_assets = len(self.assets)
            
            if total_assets == 0:
                self.error_occurred.emit("分析対象資産が選択されていません．")
                return
            
            # 進捗開始
            self.progress_updated.emit(0, "データ取得を開始しています...")
            
            # 少し待機してUIを更新させる
            self.msleep(100)
            
            # 並列処理用の引数リストを作成
            args_list = [
                (asset, self.start_date, self.end_date, self.interval) 
                for asset in self.assets
            ]
            
            price_data = {}
            completed_count = 0
            
            # 並列処理でデータ取得
            max_workers = min(4, total_assets)
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 全てのタスクを投入
                future_to_asset = {
                    executor.submit(fetch_single_asset_data, args): args[0] 
                    for args in args_list
                }
                
                # 完了したタスクから順次処理
                for future in concurrent.futures.as_completed(future_to_asset):
                    asset = future_to_asset[future]
                    completed_count += 1
                    
                    try:
                        result = future.result()
                        price_data[result['symbol']] = {
                            'name': result['name'],
                            'currency': result['currency'],
                            'data': result['data']
                        }
                        
                        # 進捗更新
                        progress = int((completed_count / total_assets) * 90)  # 90%まで
                        status = f"{result['symbol']} 完了 ({completed_count}/{total_assets})"
                        if not result['success']:
                            status += " - データなし"
                        
                        self.progress_updated.emit(progress, status)
                        
                        # UIの応答性を保つために少し待機
                        self.msleep(50)
                        
                    except Exception as e:
                        self.logger.error(f"{asset.symbol} の結果取得エラー: {e}")
                        price_data[asset.symbol] = {
                            'name': asset.name,
                            'currency': asset.currency or 'USD',
                            'data': pd.Series(dtype=float)
                        }
            
            # 完了
            self.progress_updated.emit(100, "データ取得完了")
            
            # 取得成功数をログに出力
            success_count = sum(1 for data in price_data.values() if not data['data'].empty)
            self.logger.info(f"データ取得完了: {success_count}/{total_assets} 件")
            
            self.data_fetched.emit(price_data)
            
        except Exception as e:
            self.logger.error(f"価格データ取得エラー: {e}")
            self.error_occurred.emit(f"データ取得中にエラーが発生しました．: {str(e)}")


class TableUpdateThread(QThread):
    """テーブル更新用スレッド"""
    
    progress_updated = Signal(int, str)
    update_completed = Signal()
    batch_ready = Signal(int, list)  # バッチ番号，アイテムリスト
    
    def __init__(self, df_to_display, currencies, start_row=0, max_rows=500):
        super().__init__()
        self.df_to_display = df_to_display
        self.currencies = currencies
        self.start_row = start_row
        self.max_rows = max_rows
        self.items_cache = []
    
    def run(self):
        """テーブルアイテム作成（バッチ処理）"""
        try:
            total_rows = min(len(self.df_to_display), self.max_rows)
            end_row = min(self.start_row + total_rows, len(self.df_to_display))
            
            # 設定からバッチサイズを取得
            config = get_config()
            batch_size = config.get('analysis.price_series.batch_size', 50)
            
            for batch_start in range(self.start_row, end_row, batch_size):
                batch_end = min(batch_start + batch_size, end_row)
                batch_items = []
                
                for row_idx in range(batch_start, batch_end):
                    row_items = []
                    date = self.df_to_display.index[row_idx]
                    row_data = self.df_to_display.iloc[row_idx]
                    
                    for col_idx, (symbol, price) in enumerate(row_data.items()):
                        currency = self.currencies.get(symbol, 'USD')
                        
                        if pd.isna(price):
                            item = QTableWidgetItem("N/A")
                            item.setForeground(Qt.gray)
                        else:
                            item = QTableWidgetItem(f"{float(price):.2f}")
                            item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                        
                        item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
                        # ツールチップは軽量化（必要最小限）
                        item.setToolTip(f"{symbol}: {date.strftime('%Y-%m-%d')}")
                        
                        row_items.append((row_idx, col_idx, item))
                    
                    batch_items.extend(row_items)
                
                # バッチ完了シグナル送信
                self.batch_ready.emit(batch_start, batch_items)
                
                # 進捗更新
                progress = int(((batch_end - self.start_row) / total_rows) * 100)
                self.progress_updated.emit(progress, f"表示準備中... {batch_end - self.start_row}/{total_rows}")
                
                # UIの応答性を保つため待機
                self.msleep(10)
            
            self.update_completed.emit()
            
        except Exception as e:
            self.progress_updated.emit(0, f"表示エラー: {str(e)}")


class PriceSeriesWidget(AnalysisBaseWidget):
    """資産価格時系列表示ウィジェット"""
    
    # データ取得完了シグナル（他のモジュールが待機するため）
    data_ready = Signal()
    
    def __init__(self, config=None):
        super().__init__(config)
        self.price_data = {}
        self.processed_dataframe = None  # 処理済みDataFrameをキャッシュ（全日付版）
        self.common_dataframe = None     # 共通日付版DataFrame（分析用）
        self.currencies_dict = {}  # 通貨情報を保存するための辞書
        self.is_data_ready = False  # データ準備完了フラグ
        self.current_span = '日次'  # 現在のスパン（期待利益率で順序を合わせるため）
        self.data_quality_stats = {}  # データ品質統計
        self.table_update_thread = None  # テーブル更新スレッド
        self.current_page = 0  # 現在のページ
        # 設定からページ当たりの行数を取得
        config = get_config()
        self.rows_per_page = config.get('analysis.price_series.rows_per_page', 200)
        self.setMinimumHeight(get_min_height("price_series"))
        self.setMinimumWidth(get_min_width("price_series"))
        self.setup_content()
    
    def setup_header_content(self):
        """UIの設定"""
        # 表示モード選択
        self.display_mode_combo = self.create_combo_box(
            ["全日付表示", "共通日付のみ"], 
            min_width="120px"
        )
        self.display_mode_combo.setCurrentText("全日付表示")
        self.display_mode_combo.currentTextChanged.connect(self.update_display)
        
        self.header_layout.addWidget(QLabel("表示:", styleSheet="color: #ffffff; font-size: 10px;"))
        self.header_layout.addWidget(self.display_mode_combo)
        
        # ページング制御
        self.rows_per_page_spin = self.create_spinbox()
        self.rows_per_page_spin.setRange(50, 1000)
        self.rows_per_page_spin.setValue(200)
        self.rows_per_page_spin.setSuffix(" 行")
        self.rows_per_page_spin.valueChanged.connect(self.on_rows_per_page_changed)

        self.header_layout.addWidget(QLabel("表示行数:", styleSheet="color: #ffffff; font-size: 10px;"))
        self.header_layout.addWidget(self.rows_per_page_spin)
        
        self.header_layout.addStretch()
        
        # ページングボタン
        self.prev_page_button = self.create_button(
            "◀ 前のページ", 
            "neutral"
        )
        self.prev_page_button.setEnabled(False)
        self.prev_page_button.clicked.connect(self.prev_page)
        self.header_layout.addWidget(self.prev_page_button)
        
        self.page_info_label = QLabel("ページ: - / -")
        self.page_info_label.setStyleSheet("color: #ffffff; font-size: 10px;")
        self.header_layout.addWidget(self.page_info_label)
        
        self.next_page_button = self.create_button(
            "次のページ ▶", 
            "neutral"
        )
        self.next_page_button.setEnabled(False)
        self.next_page_button.clicked.connect(self.next_page)
        self.header_layout.addWidget(self.next_page_button)
        
        # エクスポートボタン
        self.export_all_button = self.create_button(
            "CSV出力(全日付)",
            "export"
        )
        self.export_all_button.setEnabled(False)
        self.export_all_button.clicked.connect(lambda: self.export_to_csv("all"))
        self.header_layout.addWidget(self.export_all_button)
        
        self.export_common_button = self.create_button(
            "CSV出力(共通日付)",
            "save"
        )
        self.export_common_button.setEnabled(False)
        self.export_common_button.clicked.connect(lambda: self.export_to_csv("common"))
        self.header_layout.addWidget(self.export_common_button)
    
    def setup_content(self):
        """メインコンテンツエリアの設定"""
        # 表示進捗バー（テーブル更新用）
        self.display_progress_bar = QProgressBar()
        self.display_progress_bar.setVisible(False)
        self.display_progress_bar.setStyleSheet(
            self.styles.get_progress_bar_style().replace(
                self.styles.COLORS["primary"], 
                self.styles.COLORS["secondary"]
            )
        )
        self.display_progress_bar.setMaximumHeight(self.styles.SIZES["progress_bar_height"])
        self.content_layout.addWidget(self.display_progress_bar)
        
        # パフォーマンス情報ラベル
        self.performance_info_label = QLabel("")
        self.performance_info_label.setStyleSheet("color: #ff9800; font-size: 9px;")
        self.performance_info_label.setVisible(False)
        self.content_layout.addWidget(self.performance_info_label)
        
        # 価格データテーブル
        self.price_table = self.create_table_widget()
        
        # テーブルの設定
        self.price_table.setSortingEnabled(False)
        
        # ヘッダーの設定
        horizontal_header = self.price_table.horizontalHeader()
        horizontal_header.setSectionResizeMode(QHeaderView.Fixed)
        horizontal_header.setDefaultSectionSize(65)
        
        vertical_header = self.price_table.verticalHeader()
        vertical_header.setSectionResizeMode(QHeaderView.Fixed)
        vertical_header.setDefaultSectionSize(18)
        
        self.content_layout.addWidget(self.price_table)
    
    def get_empty_message(self) -> str:
        """空状態メッセージ（オーバーライド）"""
        return (
            "価格データが表示されていません．\n"
            "分析を開始すると，各資産の価格時系列が表示されます．"
        )
    
    def on_rows_per_page_changed(self, value):
        """表示行数変更時の処理"""
        self.rows_per_page = value
        self.current_page = 0  # 最初のページに戻る
        self.update_display()
    
    def prev_page(self):
        """前のページ"""
        if self.current_page > 0:
            self.current_page -= 1
            self.update_display()
    
    def next_page(self):
        """次のページ"""
        self.current_page += 1
        self.update_display()
    
    def update_page_controls(self, df_to_display):
        """ページング制御の更新"""
        if df_to_display is None or df_to_display.empty:
            self.prev_page_button.setEnabled(False)
            self.next_page_button.setEnabled(False)
            self.page_info_label.setText("ページ: - / -")
            return
        
        total_rows = len(df_to_display)
        total_pages = (total_rows + self.rows_per_page - 1) // self.rows_per_page
        
        # 現在のページが範囲外の場合は調整
        if self.current_page >= total_pages:
            self.current_page = max(0, total_pages - 1)
        
        self.prev_page_button.setEnabled(self.current_page > 0)
        self.next_page_button.setEnabled(self.current_page < total_pages - 1)
        self.page_info_label.setText(f"ページ: {self.current_page + 1} / {total_pages}")
    
    def analyze(self, assets: List[AssetInfo], conditions: Dict[str, Any]):
        """分析実行（価格データ取得）"""
        if not assets:
            QMessageBox.warning(self, "エラー", "分析対象資産が選択されていません．")
            return
        
        # 分析条件の取得
        start_date = conditions.get('start_date')
        end_date = conditions.get('end_date')
        span = conditions.get('span', '日次')
        
        if not start_date or not end_date:
            QMessageBox.warning(self, "エラー", "分析期間が設定されていません．")
            return
        
        # 現在のスパンを保存（期待利益率で順序を合わせるため）
        self.current_span = span
        
        # yfinanceのインターバル変換
        interval_map = {
            '日次': '1d',
            '週次': '1wk',
            '月次': '1mo'
        }
        interval = interval_map.get(span, '1d')
        
        # データ準備完了フラグをリセット
        self.is_data_ready = False
        self.current_page = 0  # ページをリセット
        
        # キャッシュをクリア
        self.processed_dataframe = None
        self.common_dataframe = None
        self.data_quality_stats = {}
        
        # UIの状態更新
        self.show_progress(True)
        self.show_quality_info("データ取得を開始しています...")
        self.show_main_content(False)  # テーブルを非表示
        self.export_all_button.setEnabled(False)
        self.export_common_button.setEnabled(False)
        
        # パフォーマンス情報表示
        self.performance_info_label.setText(f"表示設定: {self.rows_per_page}行/ページ")
        self.performance_info_label.setVisible(True)

        # アプリケーションイベントを処理してUIを更新
        QApplication.processEvents()
        
        # 開始時間記録
        self.start_time = time.time()
        
        # データ取得スレッド開始
        self.fetch_thread = PriceDataFetcher(assets, start_date, end_date, interval)
        self.fetch_thread.progress_updated.connect(self.update_progress)
        self.fetch_thread.data_fetched.connect(self.on_data_fetched)
        self.fetch_thread.error_occurred.connect(self.on_error_occurred)
        self.fetch_thread.start()
    
    def on_data_fetched(self, price_data: Dict):
        """データ取得完了"""
        # 処理時間計算
        elapsed_time = time.time() - self.start_time
        
        self.price_data = price_data
        
        # データ前処理を非同期で実行
        self.update_progress(95, "データを処理中...")
        QApplication.processEvents()
        
        # データ前処理
        self.preprocess_data()
        
        # UI状態更新
        self.show_progress(False)
        
        # データ品質結果表示
        if self.data_quality_stats:
            success_count = self.data_quality_stats['success_count']
            total_count = self.data_quality_stats['total_count']
            common_dates = self.data_quality_stats['common_dates']
            all_dates = self.data_quality_stats['all_dates']
            
            quality_text = (
                f"取得: {success_count}/{total_count} 件 "
                f"(全{all_dates}日, 共通{common_dates}日, 処理時間: {elapsed_time:.1f}秒)"
            )
            self.show_quality_info(quality_text)
        
        if price_data:
            self.export_all_button.setEnabled(True)
            self.export_common_button.setEnabled(True)
        
        # データ準備完了
        self.is_data_ready = True
        
        # 高速表示開始
        self.update_display()
        
        # データ準備完了シグナルを送信（他のモジュールが待機）
        self.data_ready.emit()
    
    def on_error_occurred(self, error_message: str):
        """エラー発生"""
        self.show_progress(False)
        self.display_progress_bar.setVisible(False)
        self.hide_quality_info()
        self.performance_info_label.setVisible(False)
        QMessageBox.critical(self, "エラー", error_message)
        self.update_display()
    
    def preprocess_data(self):
        """データ前処理"""
        if not self.price_data:
            self.processed_dataframe = None
            self.common_dataframe = None
            self.currencies_dict = {}
            self.data_quality_stats = {}
            return
        
        # 有効なデータのみを処理
        valid_series = {}
        currencies = {}
        
        for symbol, data in self.price_data.items():
            if not data['data'].empty:
                valid_series[symbol] = data['data']
                currencies[symbol] = data['currency']
        
        if not valid_series:
            self.processed_dataframe = None
            self.common_dataframe = None
            self.currencies_dict = {}
            self.data_quality_stats = {}
            return
        
        # 全日付のDataFrameを作成（行=日付，列=銘柄）
        # reindexを使用してNaNを明示的に表示
        all_dates = pd.Index([])
        for series in valid_series.values():
            all_dates = all_dates.union(series.index)
        all_dates = all_dates.sort_values()
        
        # 全日付版DataFrame（表示用）
        self.processed_dataframe = pd.DataFrame(index=all_dates)
        for symbol, series in valid_series.items():
            self.processed_dataframe[symbol] = series.reindex(all_dates)
        
        # 共通日付版DataFrame（分析用）
        common_dataframe_list = []
        for symbol, series in valid_series.items():
            if common_dataframe_list:
                # 既存のデータと共通する日付のみを保持
                common_dates = common_dataframe_list[0].index.intersection(series.index)
                common_dataframe_list = [df.reindex(common_dates) for df in common_dataframe_list]
                common_dataframe_list.append(series.reindex(common_dates))
            else:
                common_dataframe_list.append(series.copy())
        
        if common_dataframe_list:
            self.common_dataframe = pd.concat(common_dataframe_list, axis=1)
            self.common_dataframe.columns = list(valid_series.keys())
        else:
            self.common_dataframe = pd.DataFrame()
        
        # 通貨情報を属性として保存
        self.processed_dataframe.currencies = currencies
        if self.common_dataframe is not None and not self.common_dataframe.empty:
            self.common_dataframe.currencies = currencies
        self.currencies_dict = currencies
        
        # データ品質統計
        self.data_quality_stats = {
            'success_count': len(valid_series),
            'total_count': len(self.price_data),
            'all_dates': len(all_dates),
            'common_dates': len(self.common_dataframe) if self.common_dataframe is not None else 0
        }
    
    def update_display(self):
        """表示更新 - 高速ページング対応版"""
        display_mode = self.display_mode_combo.currentText()
        
        if display_mode == "全日付表示":
            df_to_display = self.processed_dataframe
        else:  # "共通日付のみ"
            df_to_display = self.common_dataframe
        
        if df_to_display is None or df_to_display.empty:
            self.show_main_content(False)
            self.update_page_controls(None)
            return
        
        self.show_main_content(True)
        
        # ページング制御を更新
        self.update_page_controls(df_to_display)
        
        # 現在のページのデータを取得
        start_row = self.current_page * self.rows_per_page
        end_row = min(start_row + self.rows_per_page, len(df_to_display))
        page_df = df_to_display.iloc[start_row:end_row]
        
        currencies = getattr(df_to_display, 'currencies', self.currencies_dict)
        
        # 高速表示開始
        self.display_progress_bar.setVisible(True)
        self.performance_info_label.setText(f"表示中: {start_row + 1}-{end_row}行目 / 全{len(df_to_display)}行")
        
        # テーブル設定: 行=ページ分のデータ，列=銘柄
        row_count = len(page_df)
        col_count = len(page_df.columns)
        
        self.price_table.setRowCount(row_count)
        self.price_table.setColumnCount(col_count)
        
        # 行ヘッダー設定（日付）
        date_labels = [date.strftime('%Y-%m-%d') for date in page_df.index]
        self.price_table.setVerticalHeaderLabels(date_labels)
        
        # 列ヘッダー設定（銘柄コードと通貨）
        column_headers = []
        for symbol in page_df.columns:
            currency = currencies.get(symbol, 'USD')
            column_headers.append(f"{symbol}\n{currency}")
        
        self.price_table.setHorizontalHeaderLabels(column_headers)
        
        # 非同期でテーブル更新
        self.start_table_update(page_df, currencies)
    
    def start_table_update(self, page_df, currencies):
        """非同期テーブル更新開始"""
        # 既存のスレッドを停止
        if self.table_update_thread and self.table_update_thread.isRunning():
            self.table_update_thread.quit()
            self.table_update_thread.wait()
        
        # 新しいスレッドで更新
        self.table_update_thread = TableUpdateThread(page_df, currencies)
        self.table_update_thread.progress_updated.connect(self.on_display_progress_updated)
        self.table_update_thread.batch_ready.connect(self.on_batch_ready)
        self.table_update_thread.update_completed.connect(self.on_display_completed)
        self.table_update_thread.start()
    
    def on_display_progress_updated(self, progress: int, message: str):
        """表示進捗更新"""
        self.display_progress_bar.setValue(progress)
        QApplication.processEvents()
    
    def on_batch_ready(self, batch_start: int, batch_items: list):
        """バッチ準備完了"""
        # バッチ単位でアイテムを設定
        for row_idx, col_idx, item in batch_items:
            # ページ内での相対位置に調整
            page_row = row_idx - (self.current_page * self.rows_per_page)
            if 0 <= page_row < self.price_table.rowCount():
                self.price_table.setItem(page_row, col_idx, item)
        
        # UIを更新
        QApplication.processEvents()
    
    def on_display_completed(self):
        """表示完了"""
        self.display_progress_bar.setVisible(False)
        
        # パフォーマンス情報更新
        display_mode = self.display_mode_combo.currentText()
        df_to_display = self.processed_dataframe if display_mode == "全日付表示" else self.common_dataframe
        
        if df_to_display is not None:
            start_row = self.current_page * self.rows_per_page
            end_row = min(start_row + self.rows_per_page, len(df_to_display))
            self.performance_info_label.setText(
                f"表示: {start_row + 1}-{end_row}行 / {len(df_to_display)}行 "
                f"({display_mode}, ページ{self.current_page + 1})"
            )
    
    def export_to_csv(self, export_type: str):
        """CSV出力"""
        if export_type == "all":
            df_to_export = self.processed_dataframe
            default_name = "price_series_all_dates.csv"
            title = "価格データをCSV出力（全日付版）"
        else:  # "common"
            df_to_export = self.common_dataframe
            default_name = "price_series_common_dates.csv"
            title = "価格データをCSV出力（共通日付版）"
        
        if df_to_export is None or df_to_export.empty:
            QMessageBox.information(self, "Information", "出力するデータがありません．")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, title, default_name, "CSV ファイル (*.csv)"
        )
        
        if file_path:
            try:
                # 通貨情報を取得
                currencies = getattr(df_to_export, 'currencies', self.currencies_dict)
                
                # CSVファイルに出力
                with open(file_path, 'w', encoding='utf-8') as f:
                    # 1行目: シンボル1,シンボル2,...
                    f.write('Assets,' + ','.join(df_to_export.columns) + '\n')
                    
                    # 2行目: Currency,通貨1,通貨2,...
                    currency_row = [currencies.get(symbol, 'USD') for symbol in df_to_export.columns]
                    f.write('Currency,' + ','.join(currency_row) + '\n')
                    
                    # 3行目: Type,データタイプ情報
                    if export_type == "all":
                        f.write('Type,' + ','.join(['All_Dates'] * len(df_to_export.columns)) + '\n')
                    else:
                        f.write('Type,' + ','.join(['Common_Dates'] * len(df_to_export.columns)) + '\n')
                    
                    # データを出力
                    for date, row in df_to_export.iterrows():
                        f.write(f"{date.strftime('%Y-%m-%d')},")
                        f.write(','.join([f"{value:.2f}" if pd.notna(value) else "N/A" 
                                         for value in row.values]))
                        f.write('\n')
                
                # 統計情報を追加
                stats_info = ""
                if self.data_quality_stats:
                    if export_type == "all":
                        stats_info = f"\n全日付版: {self.data_quality_stats['all_dates']}日分のデータ"
                    else:
                        stats_info = f"\n共通日付版: {self.data_quality_stats['common_dates']}日分のデータ"
                
                QMessageBox.information(self, "完了", 
                                      f"CSVファイルを出力しました:\n{file_path}{stats_info}")
                
            except Exception as e:
                QMessageBox.critical(self, "エラー", f"CSV出力中にエラーが発生しました．:\n{str(e)}")
    
    def clear_data(self):
        """データクリア"""
        self.price_data = {}
        self.processed_dataframe = None
        self.common_dataframe = None
        self.currencies_dict = {}
        self.is_data_ready = False
        self.current_span = '日次'
        self.data_quality_stats = {}
        self.table_update_thread = None
        self.current_page = 0
        # 設定からページ当たりの行数を取得
        config = get_config()
        self.rows_per_page = config.get('analysis.price_series.rows_per_page', 200)
        
        # 設定からサイズを取得
        min_height = self.config.get('analysis.widget_sizes.price_series.min_height', 600)
        min_width = self.config.get('analysis.widget_sizes.price_series.min_width', 800)
        self.setMinimumHeight(min_height)
        self.setMinimumWidth(min_width)
        
        self.setup_content()
        
        # スレッド停止
        if self.table_update_thread and self.table_update_thread.isRunning():
            self.table_update_thread.quit()
            self.table_update_thread.wait()
        
        self.update_display()
        self.export_all_button.setEnabled(False)
        self.export_common_button.setEnabled(False)
        self.hide_quality_info()
        self.performance_info_label.setVisible(False)
        self.show_progress(False)
        self.display_progress_bar.setVisible(False)
    
    def get_analysis_dataframe(self) -> Optional[pd.DataFrame]:
        """分析用DataFrameを取得（共通日付版，行=日付，列=銘柄）"""
        return self.common_dataframe if self.is_data_ready else None
    
    def get_display_dataframe(self) -> Optional[pd.DataFrame]:
        """表示用DataFrameを取得（全日付版，行=日付，列=銘柄）"""
        return self.processed_dataframe if self.is_data_ready else None
    
    def get_current_span(self) -> str:
        """現在のスパンを取得（期待利益率で順序を合わせるため）"""
        return self.current_span
    
    def is_ready(self) -> bool:
        """データ準備完了状態を確認"""
        return self.is_data_ready
    
    def get_data_quality_stats(self) -> Dict:
        """データ品質統計を取得"""
        return self.data_quality_stats