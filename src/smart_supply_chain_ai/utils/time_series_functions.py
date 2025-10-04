import pandas as pd
import numpy as np

class TimeSeriesFeatureGenerator:
    """
    A class to generate time-series features from a DataFrame.

    This class provides methods for creating rolling window statistics and
    lagged features, designed specifically to handle data with repeated dates
    or hierarchical indices.

    Attributes:
        value_column (str): The name of the column containing the values for feature generation.
        date_column (str): The name of the column with date information.
        lags (list): A list of window sizes (in days) for rolling calculations.
        max_lag (int): The maximum number of lagged features to create.
        min_periods (int): The minimum number of observations required for a rolling window calculation.
        replace_NaN (Any): The value used for filling missing data (NaNs).
    """
    def __init__(self, value_column: str, date_column: str, lags: list = [7, 14, 21], max_lag: int = 3, min_periods: int = None, replace_NaN=None):
        self.value_column = value_column
        self.lags = lags
        self.min_periods = min_periods
        self.replace_NaN = replace_NaN
        self.max_lag = max_lag
        self.date_column = date_column

    def create_grouped_windows(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds rolling window features (mean, std, sum) to a DataFrame.

        The function first aggregates the data by date and then calculates rolling
        statistics based on the specified window sizes. This is useful for data
        where multiple entries exist per day.

        Args:
            df (pd.DataFrame): The input DataFrame. It must have a DatetimeIndex.

        Returns:
            pd.DataFrame: The original DataFrame with new rolling window features.
        """
        # Group by date index and calculate the daily mean to create a time-series reference.
        daily_ref = df.groupby(level=0)[self.value_column].mean().to_frame(f'daily_{self.value_column}')

        # Calculate rolling statistics (mean, std, sum) for each specified window size.
        for window in self.lags:
            daily_ref[f'{self.value_column}_mean_{window}d'] = daily_ref[f'daily_{self.value_column}'].rolling(window=window, min_periods=self.min_periods).mean()
            daily_ref[f'{self.value_column}_std_{window}d'] = daily_ref[f'daily_{self.value_column}'].rolling(window=window, min_periods=self.min_periods).std()
            daily_ref[f'{self.value_column}_sum_{window}d'] = daily_ref[f'daily_{self.value_column}'].rolling(window=window, min_periods=self.min_periods).sum()

        # Fill any missing values in the new columns if a replacement value is provided.
        if self.replace_NaN is not None:
            daily_ref.fillna(self.replace_NaN, inplace=True)

        # Join the new daily features back to the original DataFrame based on the date index.
        df = df.join(daily_ref, how='left')
        return df

    def create_multiindex_lags(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates lagged features while preserving temporal order for repeated dates.

        This method is essential for creating correct lags in datasets where
        multiple events can occur on the same day. It works by creating a temporary
        sequential ID for each row within a date, ensuring that the temporal
        sequence is respected.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: The original DataFrame with new lagged feature columns.
        """
        # Reset the index to work with date and unique ID columns.
        temp_df = df.reset_index(drop=True)

        # Create a temporary sequential ID to order rows with repeated dates.
        temp_df['__temp_order_id__'] = temp_df.groupby(self.date_column).cumcount()

        # Sort the DataFrame by date and the new temporary ID to ensure accurate lags.
        temp_df = temp_df.sort_values([self.date_column, '__temp_order_id__'])

        # Generate lagged columns by shifting the value column based on the temporal order.
        for lag in range(1, self.max_lag + 1):
            temp_df[f'{self.value_column}_lag_{lag}'] = temp_df.groupby('__temp_order_id__')[self.value_column].shift(lag)

        # Fill any missing values in the new lagged columns if a value is provided.
        if self.replace_NaN is not None:
            temp_df.fillna(self.replace_NaN, inplace=True)

        # Re-establish the MultiIndex with the date and the new temporary order ID.
        return temp_df.set_index([self.date_column, '__temp_order_id__'])