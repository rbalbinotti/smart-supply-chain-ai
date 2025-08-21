import pandas as pd

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
