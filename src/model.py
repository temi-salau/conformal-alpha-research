import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

class ConformalRidgePredictor:
    def __init__(self, alpha_penalty=1.0):
        """
        Initialises the Ridge Regression base model
        alpha_penalty is the L2 regularisation strength (lambda)
        """

        self.model = Ridge(alpha=alpha_penalty)

    def prepare_and_split_data(self, df, train_pct=0.6, cal_pct=0.2):
        """
        Shifts target variable (Log_Returns) by -1
        Drops the last row to clean up the NaN value
        Splits data sequentially into Train, Calibration, and Test blocks
        """

        df = df.copy()
        df['Target'] = df['Log_Returns'].shift(-1)
        df_clean = df.dropna(subset=['Target'])

        feature_cols = ['Volatility', 'RSI', 'Trend_Signal']
        X = df_clean[feature_cols]
        y = df_clean['Target']

        total_rows = len(df_clean)
        train_end = int(total_rows * train_pct)
        cal_end = train_end + int(total_rows * cal_pct)
        X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
        X_cal, y_cal = X.iloc[train_end:cal_end], y.iloc[train_end:cal_end]
        X_test, y_test = X.iloc[cal_end:], y.iloc[cal_end:]

        return X_train, y_train, X_cal, y_cal, X_test, y_test
    
    def fit(self, X_train, y_train):
        """
        Fits the Ridge Regression model on the training features and targets
        """
        return self.model.fit(X_train, y_train)

    def predict(self, X):
        """
        Generates point predictions for next-day log returns
        """
        return self.model,self.predict(X)