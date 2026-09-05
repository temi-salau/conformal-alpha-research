import numpy as np
import pandas as pd

class VectorisedBacktester:
    def __init__(self, data, predictions, lower_bounds, upper_bounds, transaction_cost=0.0005):
        self.data = data
        self.predictions = predictions
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.tc = transaction_cost

        def generate_signals(self, method="directional", threshold=0.0):
            if method == "directional":
                # Simple directional sign of point prediction
                signals = np.where(self.predictions > threshold, 1, -1)

            elif method == "conformal_filter":
                # High conviction, only go long if the lower bound is above 0 or short if upper < 0
                signals = np.zeros_like(self.predictions)
                signals[self.lower_bounds > threshold] = 1
                signals[self.upper_bounds < -threshold] = -1

            return pd.Series(signals, index=self.data.index)
