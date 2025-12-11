"""csv_importer.py"""

import csv
import logging
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class CSVAssetEntry:
    """CSV読み込み時の資産エントリ"""
    symbol: str
    weight: Optional[float] = None
    name: Optional[str] = None
    row_number: int = 0  # エラー報告用


class CSVImporter:
    """CSVファイルから証券コードを読み込むクラス"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def load_from_csv(self, file_path: str) -> Tuple[List[CSVAssetEntry], List[str]]:
        """CSVファイルを読み込み，資産エントリのリストとエラーリストを返す"""
        entries = []
        errors = []
        
        try:
            # ファイル存在チェック
            if not Path(file_path).exists():
                errors.append(f"ファイルが見つかりません: {file_path}")
                return entries, errors
            
            # UTF-8で読み込み試行，失敗したらShift-JIS
            encodings = ['utf-8', 'shift-jis', 'cp932']
            content = None
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding, newline='') as f:
                        content = f.read()
                    self.logger.info(f"CSVファイルを{encoding}で読み込みました")
                    break
                except UnicodeDecodeError:
                    continue
            
            if content is None:
                errors.append("CSVファイルのエンコーディングが不明です（UTF-8またはShift-JISを使用してください）")
                return entries, errors
            
            # CSV解析
            lines = content.strip().split('\n')
            if not lines:
                errors.append("CSVファイルが空です")
                return entries, errors
            
            # ヘッダー行を解析
            reader = csv.DictReader(lines)
            headers = reader.fieldnames
            
            if not headers:
                errors.append("CSVのヘッダー行が読み取れません")
                return entries, errors
            
            # ヘッダーを正規化（小文字化，空白削除）
            headers_normalized = {h.strip().lower(): h for h in headers if h}
            
            # 必須カラム: symbolまたはticker
            symbol_col = None
            if 'symbol' in headers_normalized:
                symbol_col = headers_normalized['symbol']
            elif 'ticker' in headers_normalized:
                symbol_col = headers_normalized['ticker']
            elif 'code' in headers_normalized:
                symbol_col = headers_normalized['code']
            elif '証券コード' in headers_normalized:
                symbol_col = headers_normalized['証券コード']
            
            if not symbol_col:
                errors.append("CSVに'symbol'列が見つかりません（'symbol', 'ticker', 'code', '証券コード'のいずれかが必要です）")
                return entries, errors
            
            # オプションカラム: weight, name
            weight_col = None
            if 'weight' in headers_normalized:
                weight_col = headers_normalized['weight']
            elif 'ウエイト' in headers_normalized:
                weight_col = headers_normalized['ウエイト']
            elif '比率' in headers_normalized:
                weight_col = headers_normalized['比率']
            
            name_col = None
            if 'name' in headers_normalized:
                name_col = headers_normalized['name']
            elif '名前' in headers_normalized:
                name_col = headers_normalized['名前']
            elif '銘柄名' in headers_normalized:
                name_col = headers_normalized['銘柄名']
            
            # データ行を読み込み
            for row_num, row in enumerate(reader, start=2):  # ヘッダーの次から
                try:
                    # シンボル取得（必須）
                    symbol = row.get(symbol_col, '').strip()
                    if not symbol:
                        errors.append(f"行{row_num}: 証券コードが空です")
                        continue
                    
                    # ウエイト取得（オプション）
                    weight = None
                    if weight_col and row.get(weight_col):
                        try:
                            weight_str = row[weight_col].strip().replace('%', '')
                            weight = float(weight_str)
                            if weight < 0:
                                errors.append(f"行{row_num}: ウエイトが負の値です ({symbol})")
                                continue
                            if weight > 1000:
                                errors.append(f"行{row_num}: ウエイトが過大です ({symbol}, 最大1000%)")
                                continue
                        except ValueError:
                            errors.append(f"行{row_num}: ウエイトが数値として解釈できません ({symbol})")
                            continue
                    
                    # 名前取得（オプション）
                    name = None
                    if name_col:
                        name = row.get(name_col, '').strip()
                    
                    # エントリ作成
                    entry = CSVAssetEntry(
                        symbol=symbol,
                        weight=weight,
                        name=name,
                        row_number=row_num
                    )
                    entries.append(entry)
                    
                except Exception as e:
                    errors.append(f"行{row_num}: 解析エラー - {str(e)}")
            
            if not entries:
                errors.append("有効な資産データが見つかりませんでした")
            
            self.logger.info(f"CSV読み込み完了: {len(entries)}件の資産，{len(errors)}件のエラー")
            
        except Exception as e:
            errors.append(f"CSV読み込みエラー: {str(e)}")
            self.logger.error(f"CSV読み込みエラー: {e}", exc_info=True)
        
        return entries, errors
    
    def validate_entries(self, entries: List[CSVAssetEntry]) -> List[str]:
        """エントリの妥当性をチェック"""
        errors = []
        
        if not entries:
            errors.append("資産データが空です")
            return errors
        
        # 重複チェック
        symbols = [e.symbol for e in entries]
        duplicates = set([s for s in symbols if symbols.count(s) > 1])
        if duplicates:
            errors.append(f"重複する証券コードがあります: {', '.join(duplicates)}")
        
        # ウエイト合計チェック（全てウエイトが指定されている場合のみ）
        if all(e.weight is not None for e in entries):
            total_weight = sum(e.weight for e in entries)*100
            if total_weight > 0:
                # 警告レベル（合計が1%未満や500%超は警告）
                if total_weight < 1.0:
                    errors.append(f"警告: ウエイトの合計が非常に小さいです ({total_weight:.2f}%)")
                elif total_weight > 500:
                    errors.append(f"警告: ウエイトの合計が非常に大きいです ({total_weight:.2f}%)")
        
        return errors


def create_csv_template(file_path: str, template_type: str = 'standard'):
    """CSVテンプレートファイルを作成"""
    try:
        with open(file_path, 'w', encoding='utf-8', newline='') as f:
            if template_type == 'simple':
                # 形式1: 証券コードのみ
                f.write('symbol\n')
                f.write('8306.T\n')
                f.write('7203.T\n')
                f.write('9432.T\n')
                
            elif template_type == 'standard':
                # 形式2: 証券コードとウエイト
                f.write('symbol,weight\n')
                f.write('8306.T,0.3\n')
                f.write('7203.T,0.5\n')
                f.write('9432.T,0.2\n')
                
            elif template_type == 'detailed':
                # 形式3: 証券コード，ウエイト，名前
                f.write('symbol,weight,name\n')
                f.write('8306.T,0.3,Mitsubishi UFJ Financial Group Inc.\n')
                f.write('7203.T,0.5,Toyota Motor Corporation\n')
                f.write('9432.T,0.2,NTT Inc.\n')
            
        logging.info(f"CSVテンプレート作成完了: {file_path}")
        
    except Exception as e:
        logging.error(f"CSVテンプレート作成エラー: {e}", exc_info=True)
        raise
