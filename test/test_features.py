import pandas as pd
import numpy as np
from src.features import FeatureEngineer

rng = np.random.default_rng(seed = 42)
prices = 100 + np.cumsum(rng.normal(0, 1, 300))

data = {
    'Close': prices,
    'Log_Returns': rng.normal(0.001, 0.02, 300) # Random daily bounces
}
df = pd.DataFrame(data)

fe = FeatureEngineer(df)

fe.add_volatility().add_trend().add_rsi()

print(fe.df.tail()) # Show the last 5 rows
print("\nCheck for empty spots (NaN):")
print(fe.df.isna().sum())

fe.clean_data()

fe.run_stationary_test('RSI')
fe.run_stationary_test('Volatility')