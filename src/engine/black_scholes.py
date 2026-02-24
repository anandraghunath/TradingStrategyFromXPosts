import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

class BlackScholesModel:
    def __init__(self, risk_free_rate: float = 0.05):
        self.r = risk_free_rate

    def d1(self, S, K, T, sigma):
        # Avoid division by zero
        if sigma <= 0 or T <= 0:
            return 0.0
        return (np.log(S / K) + (self.r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))

    def d2(self, S, K, T, sigma):
        if sigma <= 0 or T <= 0:
            return 0.0
        return self.d1(S, K, T, sigma) - sigma * np.sqrt(T)

    def price(self, S, K, T, sigma, option_type="call"):
        if T <= 0:
            return max(0, S - K) if option_type == "call" else max(0, K - S)
        
        d1 = self.d1(S, K, T, sigma)
        d2 = self.d2(S, K, T, sigma)
        
        if option_type == "call":
            return S * norm.cdf(d1) - K * np.exp(-self.r * T) * norm.cdf(d2)
        else:
            return K * np.exp(-self.r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    def vega(self, S, K, T, sigma):
        if T <= 0:
            return 0.0
        d1 = self.d1(S, K, T, sigma)
        return S * norm.pdf(d1) * np.sqrt(T)

    def implied_volatility(self, S, K, T, market_price, option_type="call"):
        if market_price is None or market_price <= 0 or T <= 0:
            return 0.0
            
        def objective_function(sigma):
            return self.price(S, K, T, sigma, option_type) - market_price

        try:
            return brentq(objective_function, 1e-4, 5.0)
        except (ValueError, RuntimeError):
            return 0.0

    def delta(self, S, K, T, sigma, option_type="call"):
        if T <= 0 or sigma <= 0:
            if option_type == "call":
                return 1.0 if S > K else 0.0
            else:
                return -1.0 if S < K else 0.0
                
        d1 = self.d1(S, K, T, sigma)
        if option_type == "call":
            return norm.cdf(d1)
        else:
            return norm.cdf(d1) - 1.0

    def gamma(self, S, K, T, sigma):
        if T <= 0 or sigma <= 0:
            return 0.0
        d1 = self.d1(S, K, T, sigma)
        return norm.pdf(d1) / (S * sigma * np.sqrt(T))
