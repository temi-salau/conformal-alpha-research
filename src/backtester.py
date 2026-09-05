import numpy as np
import pandas as pd

class VectorisedBacktester:
    def __init__(self, data, predictions, lower_bounds, upper_bounds, transaction_cost=0.0005):
        """
        Initialises the vectorised backtester

        Parameters:
        - data: DataFrame containing historical price data (must include asset returns)
        - predictions: Model point predictions (expected log returns)
        - lower_bounds: Conformal lower prediction bounds
        - upper_bounds: Conformal upper prediction bounds
        - transaction_cost: Cost per unit of turnover (default 0.0005)
        """
        self.data = data
        self.predictions = predictions
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.tc = transaction_cost

    def generate_signals(self, method="directional", threshold=0.0):
        """
        Generates trading signals based on the selected method
        """
        if method == "directional":
            # Simple directional sign of point prediction
            signals = np.where(self.predictions > threshold, 1, -1)

        elif method == "conformal_filter":
            # High conviction, only go long if the lower bound is above 0 or short if upper < 0
            signals = np.zeros_like(self.predictions)
            signals[self.lower_bounds > threshold] = 1
            signals[self.upper_bounds < -threshold] = -1

        else:
            raise ValueError(f"Unknown signal generation method: {method}")

        return pd.Series(signals, index=self.data.index)

    def run_backtest(self, signals, asset_returns):
        """
        Simulates portfolio performance, accounting for signals, asset returns, and transaction costs
        """
        df = pd.DataFrame(index=self.data.index)
        df['signal'] = signals
        df['asset_return'] = asset_returns

        # Shift yesterday's signal to apply to today's return (avoids look ahead bias)
        df['strategy_return_raw'] = df['signal'].shift(1).fillna(0) * df['asset_return']

        # Calculate turnover (abs daily change in pos size)
        df['turnover'] = df['signal'].diff().abs().fillna(abs(df['signal'].iloc[0]))

        # Deduct transaction costs based off of turnover
        df['transaction_costs'] = df['turnover'] * self.tc
        df['strategy_return_net'] = df['strategy_return_raw'] - df['transaction_costs']

        # Calculate cumulative returns for both market and strategy
        df['cumulative_market_return'] = (1 + df['asset_return']).cumprod() - 1
        df['cumulative_strategy_return'] = (1 + df['strategy_return_net']).cumprod() - 1

        return df

    def evaluate_performance(self, results_df, periods_per_year=252):
        """
        Computes key performance metrics (Sharpe ratio, max drawdown, total return, volatility)
        """
        net_returns = results_df['strategy_return_net']

        total_return = results_df['cumulative_strategy_return'].iloc[-1]
        ann_return = net_returns.mean() * periods_per_year
        ann_vol = net_returns.std() * np.sqrt(periods_per_year)

        # Sharpe ratio assuming a 0% ris-free rate
        sharpe_ratio = ann_return / ann_vol if ann_vol != 0 else 0.0

        # Maximum drawdown calculation
        rolling_max = (1 + net_returns).cumprod().cummax()
        drawdown = ((1 + net_returns).cumprod() - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        return {
            "cumulative_return": total_return,
            "annualized_return": ann_return,
            "annualized_volatility": ann_vol,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown
        }