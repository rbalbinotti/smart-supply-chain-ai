import pandas as pd
import subprocess
import sys

from sklearn.base import BaseEstimator, TransformerMixin

# Class for Apply in stationary variable
class Differentiator(BaseEstimator, TransformerMixin):
    '''
        Original columns not removed.
        Create diff in columns

        Use e.g:
        diff = functions.Differentiator(columns=['Delivery_Lag', 'Days_For_Expiration'])
        new_df = diff.transform(df)
    '''
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col in self.columns:
            diff_col = f"{col}_diff"
            X_copy[diff_col] = X_copy[col].diff().fillna(0)
        
        # Removes the original date column
        X_copy = X_copy.drop(columns=self.columns)
        
        return X_copy




# Class for use with date
class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    '''
        Transform column in datetime and extract:
        Year, Month, Day, 
        Day of Year,
        Day of Week,
        Week of Year

        e.g:
        extractor = functions.DateFeatureExtractor(date_column='Date_Received')
        df_transformed = extractor.transform(df)
    '''
    def __init__(self, date_column):
        self.date_column = date_column

    def fit(self, X, y=None):
        # This transformer doesn't need to learn anything from the data, so fit does nothing.
        return self

    def transform(self, X):
        # Ensures the operation does not alter the original DataFrame
        X_copy = X.copy()
        
        # Converts the column to datetime format
        X_copy[self.date_column] = pd.to_datetime(X_copy[self.date_column])

        # Ascending date
        X_copy = X_copy.sort_values(by=self.date_column).reset_index(drop=True)
        
        # Creates new time-based features
        X_copy['year'] = X_copy[self.date_column].dt.year.astype('category')
        X_copy['month'] = X_copy[self.date_column].dt.month.astype('category')
        X_copy['day'] = X_copy[self.date_column].dt.day
        X_copy['dayofyear'] = X_copy[self.date_column].dt.dayofyear
        X_copy['dayofweek'] = X_copy[self.date_column].dt.dayofweek.astype('category')
        X_copy['weekofyear'] = X_copy[self.date_column].dt.isocalendar().week.astype(int)
        
        # Removes the original date column
        X_copy = X_copy.drop(columns=[self.date_column])
        
        return X_copy




# Function to dynamically get PDM dependencies
def get_pdm_requirements():
    """
    Exports PDM dependencies and returns a list of strings.
    """
    try:
        # Calls PDM to export the production environment requirements, without hashes
        process = subprocess.run(
            ['pdm', 'export', '--no-hashes'],
            capture_output=True,
            text=True,
            check=True
        )
        # Splits the output into lines and filters out empty lines
        requirements = [line.strip() for line in process.stdout.splitlines() if line.strip()][2:]
        return requirements
    except subprocess.CalledProcessError as e:
        print(f"Error exporting dependencies with PDM: {e.stderr}")
        return []



# Class for process data
class DataProcessor:
    """
    Class to detect outliers in Dataframe
    """
    def __init__(self, dataset: pd.DataFrame, column: str):
        self.dataset = dataset
        self.column = column


    def get_outliers(self) -> pd.DataFrame:
        """
        Returns a DataFrame containing the outlier rows from the specified column.
        """

        # Sale Volume Outliers
        Q1 = self.dataset[self.column].quantile(0.25)
        Q3 = self.dataset[self.column].quantile(0.75)
        IQR = Q3 - Q1
        upper_limit = Q3 + 1.5 * IQR
        under_limit = Q1 - 1.5 * IQR

        return self.dataset[(self.dataset[self.column] > upper_limit) | (self.dataset[self.column] < under_limit)]



if __name__ == '__main__':

    """
    Demonstrates a simple manual test for the get_outliers method.
    """
    # 1. Define a sample DataFrame with known outliers
    data_test = {'values': [-100, 1, 2, 3, 4, 5, 100]}
    df_test = pd.DataFrame(data_test)
    
    # 2. Instantiate the DataProcessor class
    processor = DataProcessor(df_test, 'values')
    
    # 3. Call the method to find outliers
    outliers_df = processor.get_outliers()

    print('DataFrame with outliers:')
    print(outliers_df)

    print('Function for requirements stract')
    print(get_pdm_requirements)
