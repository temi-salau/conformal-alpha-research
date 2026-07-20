import numpy as np

def calculate_empirical_coverage(y_true, lower_bounds, upper_bounds):
    """
    Calculates the percentage of actual values that fall within the predicted conformal prediction intervals
    """
    is_inside_interval = (y_true >= lower_bounds) & (y_true <= upper_bounds)
    coverage_fraction = np.mean(is_inside_interval)
    return coverage_fraction

def calculate_average_width(lower_bounds, upper_bounds):
    """
    Calculates the mean width of the predicted intervals
    """

    widths = upper_bounds - lower_bounds
    average_width = np.mean(widths)
    return average_width