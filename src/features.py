import pandas as pd
import numpy as np

class FeatureEngineer:
    def __init__(self, data):
        self.df = data.copy()

    def add_volatility(self, window=21):
        # Calculate rolling standard deviation of log returns
        self.df['Volatility'] = self.df['Log_Returns'].rolling(window=window).std()
        return self
    
    def add_trend(self, window=200):
        # Calculate rolling mean of the actual price
        ma_name = f'MA{window}'
        self.df[ma_name] = self.df['Close'].rolling(window=window).mean()
        
        # Calculate % distance from that mean
        self.df['Trend_Signal'] = (self.df['Close'] / self.df[ma_name]) - 1
        return self
    
    def add_rsi(self, window=14):
        # Look at difference in price from yesterday
        delta = self.df['Close'].diff()

        # Separate wins from losses
        gain = (delta.where(delta > 0, 0))
        loss = (-delta.where(delta < 0, 0))

        # Calculate the avg win and loss over the last 14 days
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()

        # Convert to RSI
        rs = avg_gain / avg_loss
        self.df['RSI'] = 100 - (100 / (1 + rs))
        return self