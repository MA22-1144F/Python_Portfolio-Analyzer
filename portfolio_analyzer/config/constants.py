"""constants.py
アプリケーション定数
プログラムロジックに必要な不変の定数を定義します．
ユーザーが変更可能な設定値は default_settings.json に定義されています．
"""

# ===========================
# アプリケーション情報
# ===========================
APP_NAME = "Portfolio Analyzer"
APP_VERSION = "1.0.0"
APP_AUTHOR = "MA22-1144F"
APP_ID = "portfolioanalyzer.app.1.0"

# ===========================
# ポートフォリオ検証
# ===========================
MIN_PORTFOLIO_WEIGHT = 0.0
MAX_PORTFOLIO_WEIGHT = 1.0
MAX_TOTAL_WEIGHT = 1.0
WEIGHT_TOLERANCE = 1e-6  # 浮動小数点誤差許容値
MAX_WEIGHT_WARNING_THRESHOLD = 10.0  # 警告を出す閾値

# ===========================
# データ取得設定
# ===========================
MAX_DOWNLOAD_RETRIES = 3
DEFAULT_RETRY_DELAY = 2  # 秒
EXPONENTIAL_BACKOFF_BASE = 2
DEFAULT_REQUEST_TIMEOUT = 10  # 秒
LONG_REQUEST_TIMEOUT = 30  # 秒
MIN_REQUEST_INTERVAL = 0.2  # 秒（連続リクエスト間隔）
SCRAPER_TIMEOUT = 10  # 秒（スクレイピングタイムアウト）

# ===========================
# ネットワーク設定
# ===========================
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
ENCODING_ATTEMPTS = ['shift_jis', 'utf-8', 'cp932']  # エンコーディング試行リスト
MOF_JGB_URL = "https://www.mof.go.jp/jgbs/reference/interest_rate/jgbcm.csv"

# ===========================
# 分析定数
# ===========================
TRADING_DAYS_PER_YEAR = 252  # 年間営業日数
WEEKLY_DATA_MIN_DAYS = 90  # 週次データ判定閾値（日数）
MONTHLY_DATA_MIN_DAYS = 730  # 月次データ判定閾値（日数）

# ===========================
# 日本株判定
# ===========================
JP_STOCK_CODE_LENGTH = 4  # 日本株コードの文字数

# ===========================
# 日付フォーマット
# ===========================
DATE_FORMAT_DISPLAY = "yyyy-MM-dd"  # Qt用
DATE_FORMAT_PYTHON = "%Y-%m-%d"  # Python用

# ===========================
# ファイルパス
# ===========================
PORTFOLIOS_DIR = "portfolios"
ASSETS_DIR = "assets"
ICONS_DIR = "icons"
ICON_FILE_EXTENSIONS = [".ico", ".png"]

# ===========================
# 検証メッセージ
# ===========================
ERROR_EMPTY_PORTFOLIO = "ポートフォリオが空です"
ERROR_INVALID_WEIGHT = "不正なウエイトが設定されています"
ERROR_WEIGHT_SUM_MISMATCH = "ウエイトの合計が1.0ではありません"
ERROR_NEGATIVE_WEIGHT = "負のウエイトは設定できません"
ERROR_EXCESSIVE_WEIGHT = "ウエイトが1.0を超えています"

# ===========================
# 分析メッセージ
# ===========================
MSG_NO_CALC_DATA = "計算用データが利用できません．"
MSG_CHECKING_CONDITIONS = "分析条件を確認中..."
MSG_CALCULATING_LOG_RETURNS = "ログリターンを計算中..."
MSG_LOG_RETURNS_FAILED = "ログリターンの計算に失敗しました．"
MSG_MIN_TWO_ASSETS = "最小分散フロンティア分析には最低2つの資産が必要です．"
MSG_EMPTY_ANALYSIS_STATE = "分析結果が表示されていません．\n価格時系列データの取得完了後，表示されます．"

# ===========================
# ログ設定
# ===========================
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_FILE_MAX_BYTES = 10 * 1024 * 1024  # 10MB
LOG_FILE_BACKUP_COUNT = 5

__all__ = [
    # アプリケーション情報
    'APP_NAME', 'APP_VERSION', 'APP_AUTHOR', 'APP_ID',

    # ポートフォリオ検証
    'MIN_PORTFOLIO_WEIGHT', 'MAX_PORTFOLIO_WEIGHT', 'MAX_TOTAL_WEIGHT',
    'WEIGHT_TOLERANCE', 'MAX_WEIGHT_WARNING_THRESHOLD',

    # データ取得設定
    'MAX_DOWNLOAD_RETRIES', 'DEFAULT_RETRY_DELAY', 'EXPONENTIAL_BACKOFF_BASE',
    'DEFAULT_REQUEST_TIMEOUT', 'LONG_REQUEST_TIMEOUT',
    'MIN_REQUEST_INTERVAL', 'SCRAPER_TIMEOUT',

    # ネットワーク設定
    'USER_AGENT', 'ENCODING_ATTEMPTS', 'MOF_JGB_URL',

    # 分析定数
    'TRADING_DAYS_PER_YEAR', 'WEEKLY_DATA_MIN_DAYS', 'MONTHLY_DATA_MIN_DAYS',

    # 日本株判定
    'JP_STOCK_CODE_LENGTH',

    # 日付フォーマット
    'DATE_FORMAT_DISPLAY', 'DATE_FORMAT_PYTHON',

    # ファイルパス
    'PORTFOLIOS_DIR', 'ASSETS_DIR', 'ICONS_DIR', 'ICON_FILE_EXTENSIONS',

    # メッセージ
    'ERROR_EMPTY_PORTFOLIO', 'ERROR_INVALID_WEIGHT', 'ERROR_WEIGHT_SUM_MISMATCH',
    'ERROR_NEGATIVE_WEIGHT', 'ERROR_EXCESSIVE_WEIGHT',
    'MSG_NO_CALC_DATA', 'MSG_CHECKING_CONDITIONS', 'MSG_CALCULATING_LOG_RETURNS',
    'MSG_LOG_RETURNS_FAILED', 'MSG_MIN_TWO_ASSETS', 'MSG_EMPTY_ANALYSIS_STATE',

    # ログ設定
    'LOG_FORMAT', 'LOG_DATE_FORMAT', 'LOG_FILE_MAX_BYTES', 'LOG_FILE_BACKUP_COUNT',
    
]