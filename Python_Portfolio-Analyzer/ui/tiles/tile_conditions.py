"""tile_conditions.py"""

from PySide6.QtWidgets import (
    QFrame, QVBoxLayout, QLabel, QFormLayout, QDateEdit, QComboBox, 
    QDoubleSpinBox, QSpinBox, QPushButton, QHBoxLayout, QMessageBox
)
from PySide6.QtCore import QDate, QPropertyAnimation, QEasingCurve, Signal, QTimer
from PySide6.QtGui import QFont
from datetime import datetime, timedelta

from data.scraper import get_latest_jgb_1year_rate
from analysis.market_data_fetcher import MarketDataFetcher
from config.app_config import AppConfig
from analysis.analysis_base_widget import AnalysisStyles

class ConditionTile(QFrame):
    collapsed_changed = Signal(bool)
    
    def __init__(self):
        super().__init__()
        self.setFrameShape(QFrame.StyledPanel)
        
        self.is_collapsed = False
        self.original_height = None
        self.collapsed_height = 50
        self.styles = AnalysisStyles()
        self._setup_ui()
        self._setup_animation()
        self._fetch_initial_rate()
    
    def _setup_ui(self):
        """UIレイアウト設定"""
        self.main_layout = QVBoxLayout()
        self._setup_header()
        self._setup_content()
        self.setLayout(self.main_layout)
    
    def _setup_header(self):
        """ヘッダー部分作成"""
        header_layout = QHBoxLayout()
        
        # タイトル
        self.title_label = QLabel("分析条件")
        font = QFont()
        font.setPointSize(11)
        font.setBold(True)
        self.title_label.setFont(font)
        header_layout.addWidget(self.title_label)
        
        header_layout.addStretch()
        
        # 折りたたみボタン
        self.collapse_button = QPushButton("⌃")
        self.collapse_button.setStyleSheet(self.styles.get_button_style_by_type("neutral"))
        self.collapse_button.setMaximumSize(25, 25)
        self.collapse_button.clicked.connect(self.toggle_collapse)
        self.collapse_button.setToolTip("条件設定を折りたたみ/展開")
        header_layout.addWidget(self.collapse_button)
        
        self.main_layout.addLayout(header_layout)
    
    def _setup_content(self):
        """コンテンツ部分作成"""
        self.content_frame = QFrame()
        content_layout = QVBoxLayout(self.content_frame)
        content_layout.setContentsMargins(0, 5, 0, 0)
        
        # フォームレイアウト
        form_layout = QFormLayout()
        self._create_form_fields(form_layout)
        content_layout.addLayout(form_layout)
        
        # 更新ボタン
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.update_rate_button = QPushButton("短期国債金利を更新")
        self.update_rate_button.setStyleSheet(self.styles.get_button_style_by_type("secondary"))
        self.update_rate_button.setMaximumWidth(160)
        self.update_rate_button.clicked.connect(self._fetch_interest_rate)
        self.update_rate_button.setToolTip("日本国財務省から最新の短期国債金利を取得")
        button_layout.addWidget(self.update_rate_button)
        
        content_layout.addLayout(button_layout)
        self.main_layout.addWidget(self.content_frame)
    
    def _create_form_fields(self, form_layout):
        """フォーム要素作成"""
        config = AppConfig()
        default_period = config.get('analysis.default_period_days', 365)
        
        # 日付設定
        yesterday = datetime.now().date() - timedelta(days=1)
        start_date = yesterday - timedelta(days=default_period)
        
        self.start_date_edit = QDateEdit()
        self.start_date_edit.setCalendarPopup(True)
        self.start_date_edit.setDate(QDate.fromString(start_date.isoformat(), "yyyy-MM-dd"))
        self.start_date_edit.setDisplayFormat("yyyy-MM-dd")
        form_layout.addRow("開始日", self.start_date_edit)

        self.end_date_edit = QDateEdit()
        self.end_date_edit.setCalendarPopup(True)
        self.end_date_edit.setDate(QDate.fromString(yesterday.isoformat(), "yyyy-MM-dd"))
        self.end_date_edit.setDisplayFormat("yyyy-MM-dd")
        form_layout.addRow("終了日", self.end_date_edit)
        
        # スパン選択
        self.span_combo = QComboBox()
        self.span_combo.addItems(["日次", "週次", "月次"])
        form_layout.addRow("スパン", self.span_combo)
        
        # 無リスク利子率
        self.risk_free = self._create_double_spinbox(" %", 3, (-10, 10), 0.001, 0.000)
        form_layout.addRow("無リスク利子率", self.risk_free)
        
        # 投資割合制約
        self.min_weight = self._create_double_spinbox(" %", 2, (0, 100), 0.001, 0)
        form_layout.addRow("投資割合下限", self.min_weight)
        
        self.max_weight = self._create_double_spinbox(" %", 2, (0, 100), 0.001, 100)
        form_layout.addRow("投資割合上限", self.max_weight)
        
        # 段階数
        self.steps = QSpinBox()
        self.steps.setRange(1, 100)
        self.steps.setValue(50)
        form_layout.addRow("期待利益率の段階数", self.steps)
        
        # 市場ポートフォリオ
        self.market_fetcher = MarketDataFetcher()
        self.market_combo = QComboBox()
        self.market_combo.addItems(self.market_fetcher.get_available_markets())
        self.market_combo.setCurrentText("Nikkei 225 (^N225)")
        self.market_combo.setToolTip("証券市場線分析で使用する市場ポートフォリオを選択")
        form_layout.addRow("市場ポートフォリオ", self.market_combo)
    
    def _create_double_spinbox(self, suffix, decimals, range_vals, step, default):
        """DoubleSpinBox作成ヘルパー"""
        spinbox = QDoubleSpinBox()
        spinbox.setSuffix(suffix)
        spinbox.setDecimals(decimals)
        spinbox.setRange(range_vals[0], range_vals[1])
        spinbox.setSingleStep(step)
        spinbox.setValue(default)
        return spinbox
    
    def _setup_animation(self):
        """アニメーション設定"""
        self.animation = QPropertyAnimation(self, b"maximumHeight")
        self.animation.setDuration(250)
        self.animation.setEasingCurve(QEasingCurve.InOutCubic)
        self.original_height = self.sizeHint().height()
    
    def toggle_collapse(self):
        """折りたたみ切り替え"""
        if self.is_collapsed:
            self._expand()
        else:
            self._collapse()
    
    def _collapse(self):
        """折りたたみ"""
        if self.is_collapsed:
            return
        
        self.is_collapsed = True
        self.collapse_button.setText("⌄")
        self.title_label.setText("分析条件 (折りたたみ中)")
        
        self.animation.setStartValue(self.height())
        self.animation.setEndValue(self.collapsed_height)
        self.animation.finished.connect(self._on_collapse_finished)
        self.animation.start()
    
    def _expand(self):
        """展開"""
        if not self.is_collapsed:
            return
        
        self.is_collapsed = False
        self.collapse_button.setText("⌃")
        self.title_label.setText("分析条件")
        
        self.content_frame.show()
        
        self.animation.setStartValue(self.collapsed_height)
        self.animation.setEndValue(self.original_height or self.sizeHint().height())
        
        self._disconnect_animation()
        self.animation.start()
        self.collapsed_changed.emit(False)
    
    def _on_collapse_finished(self):
        """折りたたみ完了時処理"""
        self.content_frame.hide()
        self.collapsed_changed.emit(True)
        self._disconnect_animation()
    
    def _disconnect_animation(self):
        """アニメーション接続解除"""
        try:
            self.animation.finished.disconnect()
        except TypeError:
            pass
    
    def _fetch_initial_rate(self):
        """初期金利取得"""
        self._fetch_interest_rate()
    
    def _fetch_interest_rate(self):
        """短期国債金利取得"""
        try:
            self._set_button_state(False, "取得中...")
            
            result = get_latest_jgb_1year_rate()
            if result is not None:
                _, _, latest_rate, _ = result
                self.risk_free.setValue(float(latest_rate))
                
                self._set_button_state(True, "更新完了")
                QTimer.singleShot(2000, lambda: self._set_button_state(True, "短期国債金利を更新"))
            else:
                self._handle_rate_fetch_error("短期国債金利の取得に失敗しました．")
                
        except (ValueError, TypeError, AttributeError) as e:
            self._handle_rate_fetch_error(f"短期国債金利取得エラー: {str(e)}")
        finally:
            if self.update_rate_button.text() == "取得中...":
                self._set_button_state(True, "短期国債金利を更新")
    
    def _set_button_state(self, enabled, text):
        """ボタン状態設定"""
        self.update_rate_button.setEnabled(enabled)
        self.update_rate_button.setText(text)
    
    def _handle_rate_fetch_error(self, message):
        """金利取得エラー処理"""
        self.risk_free.setValue(0.000)
        QMessageBox.warning(self, "短期国債金利取得エラー", f"{message}\n手動で入力してください．")
        self._set_button_state(True, "短期国債金利を更新")
    
    def get_analysis_conditions(self):
        """分析条件取得"""
        return {
            'start_date': self.start_date_edit.date().toPython(),
            'end_date': self.end_date_edit.date().toPython(),
            'span': self.span_combo.currentText(),
            'risk_free_rate': self.risk_free.value() / 100,
            'min_weight': self.min_weight.value() / 100,
            'max_weight': self.max_weight.value() / 100,
            'steps': self.steps.value(),
            'market_portfolio': self.market_combo.currentText()
        }
    
    def validate_investment_constraints_for_efficient_frontier(self, n_assets):
        """最小分散フロンティア用制約条件検証"""
        min_weight = self.min_weight.value() / 100
        max_weight = self.max_weight.value() / 100
        equal_weight = 1.0 / n_assets
        
        if min_weight > equal_weight:
            return False, f"最小投資割合は{equal_weight*100:.2f}%以下である必要があります"
        
        if max_weight < equal_weight:
            return False, f"最大投資割合は{equal_weight*100:.2f}%以上である必要があります"
        
        return True, ""
    
    def set_analysis_conditions(self, conditions):
        """分析条件設定"""
        condition_setters = [
            ('start_date', lambda d: self.start_date_edit.setDate(QDate(d.year, d.month, d.day))),
            ('end_date', lambda d: self.end_date_edit.setDate(QDate(d.year, d.month, d.day))),
            ('span', lambda s: self._set_combo_text(self.span_combo, s)),
            ('risk_free_rate', lambda r: self.risk_free.setValue(r * 100)),
            ('min_weight', lambda w: self.min_weight.setValue(w * 100)),
            ('max_weight', lambda w: self.max_weight.setValue(w * 100)),
            ('steps', lambda s: self.steps.setValue(s)),
            ('market_portfolio', lambda m: self._set_combo_text(self.market_combo, m))
        ]
        
        for key, setter in condition_setters:
            if key in conditions:
                setter(conditions[key])
    
    def _set_combo_text(self, combo, text):
        """コンボボックステキスト設定"""
        index = combo.findText(text)
        if index >= 0:
            combo.setCurrentIndex(index)
    
    def is_collapsed_state(self):
        """折りたたみ状態取得"""
        return self.is_collapsed
    
    def set_collapse_state(self, collapsed: bool):
        """折りたたみ状態設定"""
        if collapsed and not self.is_collapsed:
            self._collapse()
        elif not collapsed and self.is_collapsed:
            self._expand()