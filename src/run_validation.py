import pandas as pd
import numpy as np
from src.model import ConformalRidgePredictor
from src.features import FeatureEngineer
from src.evaluation import calculate_empirical_coverage, calculate_average_width

def main():
    df = pd.read_parquet("data/SPY_2018-01-01_2024-01-01.parquet")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    
    engineer = FeatureEngineer(df)
    engineer.add_volatility().add_trend().add_rsi().clean_data()
    df_with_features = engineer.df

    pipeline = ConformalRidgePredictor()

    X_train, y_train, X_cal, y_cal, X_test, y_test = pipeline.prepare_and_split_data(df_with_features)
    pipeline.fit(X_train, y_train)
    pipeline.calibrate(X_cal, y_cal)
    
    y_pred, lower_bounds, upper_bounds = pipeline.predict_intervals(X_test, confidence_level=0.95)
    
    coverage = calculate_empirical_coverage(y_test, lower_bounds, upper_bounds)
    width = calculate_average_width(lower_bounds, upper_bounds)

    print("\n--- Validation Results ---")
    print(f"Empirical Coverage: {coverage:.2%}")
    print(f"Average Interval Width: {width:.6f}\n")

if __name__ == "__main__":
    main()