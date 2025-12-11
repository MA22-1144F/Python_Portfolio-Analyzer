"""scraper.py"""

import requests
import pandas as pd
from io import StringIO


def get_latest_jgb_1year_rate():
    """財務省から短期国債金利の最新データを取得"""
    try:
        csv_url = "https://www.mof.go.jp/jgbs/reference/interest_rate/jgbcm.csv"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        csv_response = requests.get(csv_url, headers=headers, timeout=10)
        csv_response.raise_for_status()
        
        # 複数のエンコーディングを試行
        for encoding in ['shift_jis', 'utf-8', 'cp932']:
            try:
                csv_content = csv_response.content.decode(encoding)
                df = pd.read_csv(StringIO(csv_content), header=None)
                break
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue
        else:
            return None
        
        # 有効なデータを持つ行を抽出
        valid_rows = df[df.iloc[:, 1].notna()]
        if valid_rows.empty:
            return None
        
        # 最新の有効データを取得
        last_valid_index = valid_rows.index[-1]
        latest_date = df.iloc[last_valid_index, 0]
        latest_rate = df.iloc[last_valid_index, 1]
        
        # 利率を数値に変換
        try:
            latest_rate = float(latest_rate)
        except (ValueError, TypeError):
            return None
        
        return csv_url, latest_date, latest_rate, df
        
    except Exception as e:
        # エラーログは呼び出し側で処理
        return None


if __name__ == "__main__":
    result = get_latest_jgb_1year_rate()
    if result:
        csv_url, latest_date, latest_rate, df = result
        print(f"基準日: {latest_date}")
        print(f"短期国債金利: {latest_rate}%")
    else:
        print("データの取得に失敗")