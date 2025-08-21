import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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
        'r2_score': r2_score(y_true, y_pred)
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