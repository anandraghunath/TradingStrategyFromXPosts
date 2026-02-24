import re
from typing import Dict, Any, Optional

class TweetParser:
    def __init__(self):
        # We can implement an LLM fallback here using google-genai
        # For now, regex mock to avoid using tokens/credits pointlessly.
        pass

    def parse(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Parses a tweet to extract option trade details.
        Extracts: ticker, strike, option_type (C/P), expiration, entry_price.
        """
        ticker_match = re.search(r'\$([A-Z]+)', text)
        if not ticker_match:
            return None
        ticker = ticker_match.group(1)

        strike_match = re.search(r'(\d+(?:\.\d+)?)([CP])\b', text, re.IGNORECASE)
        if not strike_match:
            return None
        strike = float(strike_match.group(1))
        option_type = strike_match.group(2).upper()

        exp_match = re.search(r'\b(\d{1,2}/\d{1,2}(?:/\d{2,4})?)\b', text)
        expiration = exp_match.group(1) if exp_match else None

        price_match = re.search(r'(?:(?:entry|at)\s*)?(\.*\d+\.\d+|\.\d+)(?:\s*entry)?', text, re.IGNORECASE)
        entry_price = float(price_match.group(1)) if price_match else None

        return {
            "ticker": ticker,
            "strike": strike,
            "option_type": "call" if option_type == 'C' else "put",
            "expiration": expiration,
            "entry_price": entry_price
        }
