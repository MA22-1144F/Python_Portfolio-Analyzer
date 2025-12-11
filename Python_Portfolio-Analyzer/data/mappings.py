"""mappings.py"""

"""共通マッピング定義
資産情報で使用される国、通貨、取引所のマッピングを一元管理します。
"""

# 取引所と国のマッピング
EXCHANGE_COUNTRY_MAP = {
    'JPX': 'Japan', 'TSE': 'Japan', 'TYO': 'Japan',
    'NASDAQ': 'United States', 'NYSE': 'United States', 'NYQ': 'United States',
    'LSE': 'United Kingdom', 'LON': 'United Kingdom',
    'FRA': 'Germany', 'XETRA': 'Germany',
    'PAR': 'France', 'EPA': 'France',
    'SWX': 'Switzerland',
    'ASX': 'Australia',
    'HKG': 'Hong Kong', 'HKSE': 'Hong Kong',
    'SSE': 'China', 'SHG': 'China', 'SHZ': 'China',
    'KRX': 'South Korea', 'KSC': 'South Korea',
    'TPE': 'Taiwan', 'TWO': 'Taiwan',
    'BSE': 'India', 'NSE': 'India',
    'TSX': 'Canada',
    'BMV': 'Mexico',
    'BOVESPA': 'Brazil', 'SAO': 'Brazil',
}

# シンボルサフィックスと国のマッピング
SUFFIX_COUNTRY_MAP = {
    '.T': 'Japan',
    '.US': 'United States',
    '.L': 'United Kingdom',
    '.DE': 'Germany',
    '.PA': 'France',
    '.SW': 'Switzerland',
    '.AX': 'Australia',
    '.HK': 'Hong Kong',
    '.SS': 'China',
    '.SZ': 'China',
    '.KS': 'South Korea',
    '.TW': 'Taiwan',
    '.BO': 'India',
    '.NS': 'India',
    '.TO': 'Canada',
    '.MX': 'Mexico',
    '.SA': 'Brazil',
}

# 国と通貨のマッピング
COUNTRY_CURRENCY_MAP = {
    'Japan': 'JPY',
    'United States': 'USD',
    'United Kingdom': 'GBP',
    'Germany': 'EUR',
    'France': 'EUR',
    'Switzerland': 'CHF',
    'Australia': 'AUD',
    'Hong Kong': 'HKD',
    'China': 'CNY',
    'South Korea': 'KRW',
    'Taiwan': 'TWD',
    'India': 'INR',
    'Canada': 'CAD',
    'Mexico': 'MXN',
    'Brazil': 'BRL',
}

def infer_country_from_exchange(exchange: str) -> str:
    """取引所から国を推論"""
    return EXCHANGE_COUNTRY_MAP.get(exchange, 'United States')

def infer_exchange_from_symbol(symbol: str) -> str:
    """シンボルから取引所を推論"""
    if '.T' in symbol:
        return 'JPX'
    elif '.US' in symbol or not any(suffix in symbol for suffix in SUFFIX_COUNTRY_MAP.keys()):
        return 'NASDAQ'
    elif '.L' in symbol:
        return 'LSE'
    elif '.DE' in symbol:
        return 'XETRA'
    elif '.PA' in symbol:
        return 'EPA'
    elif '.AX' in symbol:
        return 'ASX'
    elif '.HK' in symbol:
        return 'HKSE'
    elif '.SS' in symbol or '.SZ' in symbol:
        return 'SSE'
    return 'NASDAQ'

def infer_country_from_symbol(symbol: str) -> str:
    """シンボルから国を推論"""
    for suffix, country in SUFFIX_COUNTRY_MAP.items():
        if suffix in symbol:
            return country
    return 'United States'

def infer_currency_from_country(country: str) -> str:
    """国から通貨を推論"""
    return COUNTRY_CURRENCY_MAP.get(country, 'USD')

__all__ = [
    'EXCHANGE_COUNTRY_MAP',
    'SUFFIX_COUNTRY_MAP',
    'COUNTRY_CURRENCY_MAP',
    'infer_country_from_exchange',
    'infer_exchange_from_symbol',
    'infer_country_from_symbol',
    'infer_currency_from_country',
]