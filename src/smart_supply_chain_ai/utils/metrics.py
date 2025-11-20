"""
Time Series Forecasting Evaluation Metrics and Utilities

This module provides functions for calculating various forecast accuracy metrics,
including standard error metrics (RMSE, MAE), the safe sMAPE, and a custom
business-oriented cost metric for inventory forecasting (overstock/understock costs).
It also includes utilities for comparing model performance in cross-validation
and extracting the best-performing model's forecasts.

Author: Roberto Rosário Balbinotti
Created: 2025
Version: 1.0
"""


import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error



def get_best_model_forecast(forecasts_df, evaluation_df):
    """
    Extracts the forecast values from the best-performing model for each time series,
    including point forecasts and optional prediction intervals.

    Parameters:
    ----------
    forecasts_df : pandas.DataFrame
        A DataFrame containing forecast results from multiple models.
        Must include columns for each model's forecast and optional interval suffixes:
        - 'unique_id': identifier for each time series
        - 'ds': date or timestamp
        - model columns: e.g., 'model_a', 'model_b', 'model_a-lo-90', etc.

    evaluation_df : pandas.DataFrame
        A DataFrame containing evaluation metrics and a 'best_model' column
        indicating the best model for each 'unique_id'.

    Returns:
    -------
    pandas.DataFrame
        A DataFrame with columns:
        - 'unique_id'
        - 'ds'
        - 'best_model': point forecast from the best model
        - 'best_model-lo-90': lower bound of prediction interval (if available)
        - 'best_model-hi-90': upper bound of prediction interval (if available)
    """

    # Merge forecast data with best model info based on 'unique_id'
    with_best = forecasts_df.merge(evaluation_df[['unique_id', 'best_model']])

    # Initialize result DataFrame with identifiers
    res = with_best[['unique_id', 'ds']].copy()

    # For each forecast type (point, lower bound, upper bound), extract values from the best model
    for suffix in ('', '-lo-90', '-hi-90'):
        res[f'best_model{suffix}'] = with_best.apply(
            lambda row: row[row['best_model'] + suffix], axis=1
        )

    # Return the final DataFrame with best model forecasts
    return res



def evaluate_cv(df, metric):
    """
    Evaluate cross-validation results for multiple forecasting models and identify the best-performing model.

    Parameters:
    ----------
    df : pandas.DataFrame
        A DataFrame containing cross-validation results. Must include columns:
        - 'unique_id': identifier for each time series
        - 'ds': date or timestamp
        - 'y': actual values
        - 'cutoff': forecast origin
        - model columns: predicted values from different models

    metric : callable
        A function that takes the DataFrame and a list of model column names, and returns a DataFrame
        with evaluation scores for each model.

    Returns:
    -------
    pandas.DataFrame
        A DataFrame with evaluation scores for each model and an additional column 'best_model'
        indicating the model with the lowest score for each row.
    """

    # Extract model column names by excluding metadata columns
    models = df.columns.drop(['unique_id', 'ds', 'y', 'cutoff']).tolist()

    # Apply the evaluation metric to the DataFrame using the selected models
    evals = metric(df, models=models)

    # Determine the best-performing model for each row (lowest score)
    evals['best_model'] = evals[models].idxmin(axis=1)

    # Return the evaluation results with the best model identified
    return evals




def drop_columns(X, cols_to_drop):
    """
    Drop specified columns from the input data.
    
    Parameters:
    -----------
    X : pandas.DataFrame or numpy.ndarray
        Input data
    cols_to_drop : list
        List of column names to drop
    
    Returns:
    --------
    pandas.DataFrame or numpy.ndarray
        Data with specified columns removed
    """
    # If X is already a numpy array, return it as is (no columns to drop)
    if hasattr(X, 'columns'):
        # It's a DataFrame
        cols = [col for col in cols_to_drop if col in X.columns]
        return X.drop(columns=cols)
    else:
        # It's already a numpy array, return as is
        return X

# Function to compute robust sMAPE
def safe_smape(y_true, y_pred):
    """Robust sMAPE that avoids division by zero"""
    denominator = (np.abs(y_true) + np.abs(y_pred))
    # Avoid division by zero – replace zeros with 1
    denominator = np.where(denominator == 0, 1, denominator)
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / denominator)



class CustomCostMetrics:
    """
    Class to calculate custom cost metrics based on business costs.
    
    This class allows weighting the cost of over-forecasting (excess inventory)
    and under-forecasting (out-of-stock).
    """
    def __init__(self, cost_per_overstock_unit=1.0, cost_per_understock_unit=3.0):
        """
        Initializes the class with unit costs.

        Args:
            cost_per_overstock_unit (float): Cost of having one extra unit in stock.
            cost_per_understock_unit (float): Cost of losing a sale due to being out of stock.
        """
        self.cost_per_overstock_unit = cost_per_overstock_unit
        self.cost_per_understock_unit = cost_per_understock_unit
        
    def calculate_cost(self, y_true, y_pred):
        """
        Calculates the total cost of the forecast.

        Args:
            y_true (np.array or list): True values.
            y_pred (np.array or list): Predicted values.

        Returns:
            float: The total cost.
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Forecasting errors
        errors = y_pred - y_true
        
        # Under-forecasting cost (forecast < actual)
        understock_errors = errors[errors < 0]
        understock_cost = np.sum(np.abs(understock_errors)) * self.cost_per_understock_unit
        
        # Over-forecasting cost (forecast > actual)
        overstock_errors = errors[errors > 0]
        overstock_cost = np.sum(overstock_errors) * self.cost_per_overstock_unit
        
        return float(overstock_cost + understock_cost)



def evaluate_forecast(y_true, y_pred, custom_cost_metrics=None):
    """
    Calculates and returns the evaluation metrics for a forecast.

    Args:
        y_true (np.array or list): True values.
        y_pred (np.array or list): Predicted values.
        custom_cost_metrics (CustomCostMetrics, optional): Instance of the
                                                            CustomCostMetrics class.
                                                            If provided,
                                                            the business cost will be calculated.

    Returns:
        dict: A dictionary with the calculated metrics.
    """
    
    metrics = {
        'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'MAE': mean_absolute_error(y_true, y_pred),    
    }
    
    if custom_cost_metrics:
        metrics['Custom_Cost'] = custom_cost_metrics.calculate_cost(y_true, y_pred)
        
    return metrics

if __name__ == '__main__':
    # Example usage
    y_true = [100, 110, 120, 130, 140]
    y_pred = [95, 115, 118, 135, 138]

    # Standard evaluation (RMSE and MAE)
    standard_metrics = evaluate_forecast(y_true, y_pred)
    print("Standard Evaluation Metrics:")
    print(standard_metrics)

    # Evaluation with a custom cost metric
    # Assuming the cost of being out of stock is 3x higher than the cost of overstock
    custom_evaluator = CustomCostMetrics(cost_per_overstock_unit=1.0, cost_per_understock_unit=3.0)
    full_metrics = evaluate_forecast(y_true, y_pred, custom_cost_metrics=custom_evaluator)
    
    print("\nEvaluation Metrics with Business Cost:")
    print(full_metrics)