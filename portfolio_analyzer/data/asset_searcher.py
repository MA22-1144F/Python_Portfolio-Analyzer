"""asset_searcher.py"""

import requests
import time
from typing import List, Optional, Dict, Any
import logging

from config.constants import USER_AGENT, MIN_REQUEST_INTERVAL
from data.asset_info import AssetInfo
from data.mappings import infer_country_from_symbol

class AssetSearcher:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.search_url = "https://query1.finance.yahoo.com/v1/finance/search"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': USER_AGENT,
            'Accept': 'application/json',
        })
        self.last_request_time = 0
        self.min_request_interval = MIN_REQUEST_INTERVAL
    
    def search_assets(self, query: str, max_results: int = 20) -> List[AssetInfo]:
        if not query or len(query.strip()) < 1:
            return []
        
        try:
            self._rate_limit()
            search_results = self._call_yahoo_search_api(query.strip())
            
            if not search_results:
                return []
            
            assets = []
            for result in search_results[:max_results]:
                asset = self._convert_to_asset_info(result)
                if asset:
                    assets.append(asset)
            
            return assets
            
        except Exception as e:
            self.logger.error(f"Search error for '{query}': {e}")
            return []
    
    def _rate_limit(self):
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        
        self.last_request_time = time.time()
    
    def _call_yahoo_search_api(self, query: str) -> List[Dict[str, Any]]:
        try:
            params = {'q': query, 'quotesCount': 15, 'newsCount': 0}
            response = self.session.get(self.search_url, params=params, timeout=3)
            response.raise_for_status()
            data = response.json()
            return data.get('quotes', [])
        except Exception as e:
            self.logger.error(f"API error: {e}")
            return []
    
    def _convert_to_asset_info(self, yahoo_result: Dict[str, Any]) -> Optional[AssetInfo]:
        try:
            symbol = yahoo_result.get('symbol')
            if not symbol:
                return None
            
            name = (yahoo_result.get('longname') or 
                   yahoo_result.get('shortname') or 
                   symbol)
            
            exchange = yahoo_result.get('exchange', '')
            yf_currency = yahoo_result.get('currency')
            inferred_currency = AssetInfo._static_infer_currency_from_symbol(symbol)
            
            currency = yf_currency
            if (symbol.startswith('^') or 
                symbol in AssetInfo._get_known_symbol_currencies() or
                not yf_currency or 
                (yf_currency == 'USD' and inferred_currency != 'USD')):
                currency = inferred_currency
            
            country = self._infer_country(exchange, symbol)
            
            return AssetInfo(
                symbol=symbol, name=name, exchange=exchange, currency=currency,
                country=country, sector=yahoo_result.get('sector'),
                industry=yahoo_result.get('industry'), 
                legal_type=yahoo_result.get('quoteType')
            )
            
        except Exception as e:
            self.logger.error(f"Conversion error: {e}")
            return None
    
    def _infer_country(self, exchange: str, symbol: str) -> str:
        return infer_country_from_symbol(symbol)
    
    def validate_symbol(self, symbol: str) -> bool:
        try:
            results = self.search_assets(symbol, max_results=1)
            return len(results) > 0
        except Exception as e:
            logging.debug(f"Symbol validation failed for {symbol}: {e}")
            return False