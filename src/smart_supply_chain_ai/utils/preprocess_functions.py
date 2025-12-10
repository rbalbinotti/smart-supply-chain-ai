"""
Preprocessing Functions for Smart Supply Chain AI - Demand Forecasting

This module provides specialized preprocessing classes and utilities for time series
forecasting in supply chain demand prediction. It includes transformers for feature
engineering, data preparation, and seasonal pattern handling optimized for MLForecast.

Classes:
    TrainPredictPreprocessor: Generates dynamic time-series features such as **lags, 
                              rolling means, and expanding means** for exogenous variables 
                              like 'stock_quantity' (optimized for MLForecast).
                              
    SimplePreprocessor: Automated preprocessing pipeline for **static and generated features**, 
                        handling categorical, boolean, and numeric columns with encoding, 
                        imputation, and scaling.
    
    XDFPreparator: Prepares exogenous feature DataFrames for **future periods** by imputing date-based features (**holidays, seasonality**) 
                   and base exogenous/static features (e.g., 'stock_quantity') prior to lag generation.

Key Features:
    - Automated column type detection and processing
    - **Dynamic feature engineering** (lags, rolling stats)
    - Holiday and seasonality pattern recognition and imputation
    - Robust handling of missing values and data normalization
    - Vectorized operations for performance optimization
    - Integration with MLForecast forecasting framework

Dependencies:
    pandas, numpy, scikit-learn, holidays, ast, collections.abc

Author: Roberto Rosário Balbinotti
Created: 2025
Version: 1.0
"""

import os
import dill
import shutil
import numpy as np
import pandas as pd
import holidays
import mlflow
import mlflow.sklearn
from typing import Tuple, List, Dict, Any
from functools import reduce
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, PowerTransformer
from mlforecast.lag_transforms import ExpandingMean, RollingMean
from mlforecast.feature_engineering import transform_exog
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import smape, rmse, mae


class MLflowArtifactManager:
    """
    Manage saving, logging, and loading MLForecast artifacts with MLflow.
    """

    def __init__(self, file_path: str, config_path: str):
        self.file_path = file_path
        self.config_path = config_path
        self.client = mlflow.tracking.MlflowClient()

    def save_and_log_artifacts(
        self, instance_MLForecast, exog_features: pd.DataFrame, 
        model_name: str, exog_name: str
    ):
        """Save artifacts locally and log them to MLflow."""
        temp_artifact_dir = os.path.join(self.file_path, "mlflow_temp_artifacts")
        os.makedirs(temp_artifact_dir, exist_ok=True)

        try:
            fcst_path = os.path.join(temp_artifact_dir, model_name + ".pkl")
            dill.dump(instance_MLForecast, open(fcst_path, 'wb'))

            history_df_path = os.path.join(temp_artifact_dir, exog_name + ".parquet")
            exog_features.to_parquet(history_df_path)

            mlflow.log_artifact(fcst_path, "model")
            mlflow.log_artifact(history_df_path, "history")

            if os.path.exists(self.config_path):
                mlflow.log_artifact(self.config_path, "config")

            print("Artifacts successfully logged to MLflow!")

        except Exception as e:
            print(f"Error while saving or logging artifacts: {e}")
            raise
        finally:
            if os.path.exists(temp_artifact_dir):
                shutil.rmtree(temp_artifact_dir)
                print("Temporary artifact directory removed.")

    def load_model_and_exog_history(
        self, run_id: str, model_name: str, exog_name: str
    ) -> Tuple[object, pd.DataFrame]:
        """Load MLForecast object and exogenous features from MLflow artifacts."""
        model_artifact_path = f"model/{model_name}.pkl"
        exog_artifact_path = f"history/{exog_name}.parquet"

        temp_download_dir = "mlflow_download_temp"
        os.makedirs(temp_download_dir, exist_ok=True)

        try:
            local_model_path = self.client.download_artifacts(run_id, model_artifact_path, temp_download_dir)
            local_exog_path = self.client.download_artifacts(run_id, exog_artifact_path, temp_download_dir)

            with open(local_model_path, 'rb') as f:
                instance_MLForecast = dill.load(f)

            exog_features = pd.read_parquet(local_exog_path)

            print("Artifacts successfully loaded!")
            return instance_MLForecast, exog_features

        except Exception as e:
            print(f"Error while loading artifacts: {e}")
            raise
        finally:
            if os.path.exists(temp_download_dir):
                shutil.rmtree(temp_download_dir)
                print("Temporary download directory removed.")



class TrainPredictPreprocessor(BaseEstimator, TransformerMixin):
    """
    A specialized transformer for generating time-series-based exogenous features 
    (lags, rolling means, expanding means) specifically for use with MLForecast.
    
    This class is intended to run before the final SimplePreprocessor, handling 
    the dynamic, time-dependent features like stock_quantity lags.

    Parameters
    ----------
    time_feature : list of str, optional
        Name of the time column, by default ['ds'].
    unique_feature : list of str, optional
        Name of the series ID column, by default ['unique_id'].
    categorical_features : list of str, optional
        List of categorical features (unused in current implementation, but kept for signature).
    booleans_features : list of str, optional
        List of boolean features (unused in current implementation, but kept for signature).
    static_features : list of str, optional
        List of static features (unused in current implementation, but kept for signature).
    exog_feature : list of str, optional
        The base exogenous features for which lags and rolling statistics are calculated, 
        by default ['stock_quantity'].
    exog_lags : list of int, optional
        List of lag values to generate for exogenous features, by default [1, 2, 7].
    exog_RollingMean : int, optional
        Window size for rolling mean calculation, by default 7.
    """
    def __init__(self, time_feature: list = None, unique_feature: list = None, 
                 categorical_features: list = None, booleans_features: list = None, 
                 static_features: list = None, exog_feature: list = None, 
                 exog_lags: list = None, exog_RollingMean: int = None):
        """
        Initializes the preprocessor with feature configurations.
        """
        # Define feature names
        self.time_feature = time_feature if time_feature is not None else ['ds']
        self.unique_feature = unique_feature if unique_feature is not None else ['unique_id']
        self.categorical_features = categorical_features if categorical_features is not None else ['category','sub_category']
        self.booleans_features = booleans_features if booleans_features is not None else ['in_season','is_holiday']
        self.static_features = static_features if static_features is not None else ['category','sub_category','shelf_life_days']
        
        # Define exogenous feature parameters
        self.exog_feature = exog_feature if exog_feature is not None else ['stock_quantity'] 
        self.exog_lags = exog_lags if exog_lags is not None else [1, 2, 7]
        self.exog_RollingMean = exog_RollingMean if exog_RollingMean is not None else 7

    def fit(self, X: pd.DataFrame, y=None):
        """
        Fits the transformer. Since this transformer only adds time-series features
        and does not learn parameters from the data, it simply returns self.
        """
        return self
    
    def _transform_exogenous_features(self, X: pd.DataFrame):
        """
        Generates lagged features, rolling means, and expanding means for exogenous variables 
        using the MLForecast utility function.

        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame containing time series data.

        Returns
        -------
        pd.DataFrame
            DataFrame with new transformed exogenous features merged in.
        """
        # Select necessary columns for transformation
        cols_ = self.time_feature + self.unique_feature + self.exog_feature
        
        # Generate transformed features using external function (optimized for MLForecast)
        exog_feat_transformed = transform_exog(
            X[cols_],
            lags=self.exog_lags,
            lag_transforms={
                1: [ExpandingMean()],
                self.exog_RollingMean: [RollingMean(window_size=self.exog_RollingMean)]
            }
        )
        # Drop the original exogenous feature column that was used to create the lags
        exog_feat_transformed = exog_feat_transformed.drop(columns=self.exog_feature)
        X = X.drop(columns=self.exog_feature)
        # Merge the new features back into the original dataframe (X) on time and unique id
        X_merged = X.merge(exog_feat_transformed, on=self.time_feature + self.unique_feature, how='left')
        return X_merged
    
    def transform(self, X: pd.DataFrame):
        """
        Applies all defined preprocessing steps to the input dataframe X.

        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame.

        Returns
        -------
        pd.DataFrame
            Transformed DataFrame with exogenous features engineered.
        """
        # Apply exogenous feature transformations (lags, rolling/expanding means)
        X_transformed = self._transform_exogenous_features(X)

        return X_transformed


class SimplePreprocessor(BaseEstimator, TransformerMixin):
    """
    A simplified preprocessing pipeline for time series forecasting with MLForecast.

    This transformer automatically detects and processes categorical, boolean, and numeric 
    columns from the provided input DataFrame. It applies encoding, imputation, and scaling 
    as needed, preparing the dataset for model training.

    Parameters
    ----------
    static_features : list of str, optional
        List of static features to include in preprocessing. 
        If None, defaults to ['category', 'sub_category', 'shelf_life_days'].

    Attributes
    ----------
    column_transformer_ : ColumnTransformer
        The fitted ColumnTransformer used to transform the input data.
    expected_features_ : np.ndarray
        Array of output feature names after transformation.
    """

    def __init__(self, static_features=None):
        if static_features is None:
            self.static_features = ['category', 'sub_category', 'shelf_life_days']
        else:
            self.static_features = static_features
        
        self.column_transformer_ = None
        self.expected_features_ = None
    
    def _bool_to_int(self, X):
        return X.astype(int)

    def fit(self, X, y=None):
        """
        Fit preprocessing steps (imputers, encoders, scalers) to the given dataset X.

        Parameters
        ----------
        X : pd.DataFrame
            The input features to fit the transformer on.
        y : array-like, optional
            Ignored, kept for compatibility, by default None.

        Returns
        -------
        SimplePreprocessor
            The fitted transformer instance.
        """
        # Define expected column types (based on common structure for this project)
        categorical_cols = ['category', 'sub_category']
        boolean_cols = ['in_season', 'is_holiday']
        # Dynamically find all numeric columns present in the input DataFrame
        numeric_cols = X.select_dtypes(np.number).columns.tolist()
        
        # Define preprocessing pipelines for each data type
        self.column_transformer_ = ColumnTransformer(
            transformers=[
                # Categorical: impute missing values (most frequent) and apply one-hot encoding
                ('cat', Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
                ]), categorical_cols),

                # Boolean: one-hot encode binary variables (dropping one category if truly binary)
                ('bool', Pipeline([
                    ('to_int', FunctionTransformer(self._bool_to_int, feature_names_out='one-to-one')),
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehotbool', OneHotEncoder(drop='if_binary', handle_unknown='ignore',   sparse_output=False))
                ]), boolean_cols),

                # Numeric: impute with median and apply power transformation (Yeo-Johnson) for normalization/variance stabilization
                ('numeric', Pipeline([
                    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
                    ('scaler', PowerTransformer(method='yeo-johnson', standardize=True))
                ]), numeric_cols),
            ],
            remainder='passthrough',  # Keep any other columns (e.g., 'ds', 'unique_id') unchanged
            verbose_feature_names_out=False
        )
        
        # Fit the column transformer to the data
        self.column_transformer_.fit(X)
        # Save features after transformation for later use (e.g., column reordering)
        self.expected_features_ = self.get_feature_names_out()

        return self

    def transform(self, X):
        """
        Apply the fitted preprocessing steps to the dataset X.

        Parameters
        ----------
        X : pd.DataFrame
            The input features to transform.

        Returns
        -------
        np.ndarray or pd.DataFrame
            The transformed feature set.
        """
        return self.column_transformer_.transform(X)
    
    def get_feature_names_out(self):
        """
        Get the output feature names after transformation.

        Returns
        -------
        np.ndarray
            Array of feature names.
        """
        return self.column_transformer_.get_feature_names_out()


class XDFPreparator:
    """
    Prepares the feature DataFrame (X_df) for time series forecasting, 
    specifically for future periods. This involves imputing date-based features 
    ('is_holiday', 'in_season') and base exogenous/static features (e.g., 'stock_quantity').
    
    Note: Assumes exogenous lag features (e.g., stock_quantity_lag_1) are 
    created separately by `TrainPredictPreprocessor` *after* this preparator 
    has imputed the base `stock_quantity` feature for future dates.
    
    Parameters
    ----------
    df_product_seasonality : pd.DataFrame
        DataFrame mapping 'unique_id' to seasonality rules (e.g., list of month names).
    target_col : list of str, optional
        Name of the target column to be dropped from the output, by default ['y'].
    exog_cols : list of str, optional
        Base exogenous columns to be imputed, by default ['stock_quantity'].
    static_cols : list of str, optional
        Static feature columns to be imputed, by default ['category', 'sub_category', 'shelf_life_days'].
    """
    def __init__(self, df_product_seasonality: pd.DataFrame, target_col: list = ['y'], exog_cols: list = ['stock_quantity'], static_cols: list = ['category', 'sub_category', 'shelf_life_days']):
        """
        Initializes the preparator with product seasonality rules.
        """
        # Initialize Brazilian holidays object for date feature imputation
        self.country_holidays = holidays.country_holidays('Brazil')
        self.df_product_seasonality = df_product_seasonality
        # Attributes to be populated in _prepare_seasonality_data
        self.seasonality_exploded = None
        self._prepare_seasonality_data() # Pre-process seasonality data on init
        self.target_col = target_col
        self.exog_cols = exog_cols
        self.static_cols = static_cols

    def _prepare_seasonality_data(self, verbose=False):
        """
        Processes and normalizes the raw product seasonality data into a vectorized 
        format (exploded table of unique_id and month_name) for efficient merging.

        Parameters
        ----------
        verbose : bool, optional
            If True, enables verbose output (not used in current logic), by default False.

        Returns
        -------
        pd.DataFrame
            The processed df_product_seasonality DataFrame with normalized seasonality lists.
        
        Raises
        ------
        ValueError
            If 'unique_id' or 'seasonality' columns are missing from the input DataFrame.
        """
        products_df = self.df_product_seasonality.copy()
        
        # Normalize product identifier column name
        if 'product_id' in products_df.columns and 'unique_id' not in products_df.columns:
            products_df.rename(columns={'product_id': 'unique_id'}, inplace=True)
            
        # Ensure 'unique_id' and 'seasonality' columns exist
        if 'unique_id' not in products_df.columns or 'seasonality' not in products_df.columns:
             # This preparator requires 'unique_id' and 'seasonality' columns
             raise ValueError("df_product_seasonality DataFrame must contain 'unique_id' and 'seasonality' columns.")


        # Normalize seasonality values to lists (using the helper method)
        products_df['seasonality'] = products_df['seasonality'].apply(self._normalize_seasonality_value)

        # Create the exploded table product -> month_name (one record per month in seasonality)
        seasonality_exploded = (
            products_df[['unique_id', 'seasonality']]
            .explode('seasonality')
            .rename(columns={'seasonality': 'month_name'})
        )

        # Remove rows where month_name is empty/None
        seasonality_exploded['month_name'] = seasonality_exploded['month_name'].astype(object)
        seasonality_exploded = seasonality_exploded[seasonality_exploded['month_name'].notna()].reset_index(drop=True)

        # Ensure month_name strings are normalized (stripped)
        seasonality_exploded['month_name'] = seasonality_exploded['month_name'].astype(str).str.strip()

        # Store results for later vectorized imputation
        self.df_product_seasonality = products_df[['unique_id', 'seasonality']].copy()
        # This table is used for the left merge: we only need unique pairs of (unique_id, month_name)
        self.seasonality_exploded = seasonality_exploded[['unique_id', 'month_name']].drop_duplicates().reset_index(drop=True)

        return self.df_product_seasonality

    def _classify_holiday(self, date):
        """
        Checks if a given date is a national holiday in Brazil.

        Parameters
        ----------
        date : datetime-like
            The date to check.

        Returns
        -------
        bool
            True if the date is a holiday, False otherwise.
        """
        return date in self.country_holidays

    def _impute_holidays(self, df):
        """
        Imputes the 'is_holiday' column based on the date ('ds' column) using the 
        pre-initialized Brazilian holidays object.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to impute.

        Returns
        -------
        pd.DataFrame
            DataFrame with the new boolean 'is_holiday' column.
        """
        df['is_holiday'] = df['ds'].apply(self._classify_holiday)
        return df

    def _normalize_seasonality_value(self, val):
        """
        Normalizes a seasonality value (which could be a string, list, or scalar) 
        to always return a list of strings. This handles various input formats 
        including literal evaluation of string representations of lists.

        Parameters
        ----------
        val : any
            The seasonality value from the input DataFrame.

        Returns
        -------
        list of str
            A clean list of month names (strings).
        """
        import ast
        from collections.abc import Iterable
        
        # 1) Explicit None or NaN check
        if val is None or pd.isna(val):
            return []

        # 2) Strings: treat separately
        if isinstance(val, (str, bytes)):
            v = val.strip()
            # attempt to interpret as a literal list (e.g.: "['January','February']")
            if (v.startswith('[') and v.endswith(']')) or (v.startswith('(') and v.endswith(')')):
                try:
                    parsed = ast.literal_eval(v)
                    # Check if the parsed result is iterable (list, tuple, etc.)
                    if isinstance(parsed, (list, tuple, set, np.ndarray, pd.Series)):
                        # convert and filter nulls
                        return [str(x) for x in list(parsed) if not pd.isna(x)]
                except Exception:
                    pass
            # if contains comma, split (e.g., "January, February")
            if ',' in v:
                return [p.strip() for p in v.split(',') if p.strip()]
            # otherwise, treat the whole string as a single item
            return [v]

        # 3) Iterables (excluding strings): convert to list and filter NaNs
        if isinstance(val, Iterable):
            try:
                # convert to list
                as_list = list(val)
            except Exception:
                # fallback: wrap the value if list() conversion fails
                as_list = [val]

            # return empty list if conversion resulted in an empty iterable
            if len(as_list) == 0:
                return []

            # filter null values and convert to string
            cleaned = [str(x) for x in as_list if not pd.isna(x)]
            return cleaned

        # 4) Any other scalar type -> wrap in a list
        return [str(val)]

    def _impute_seasonality_vectorized(self, df):
        """
        Imputes the 'in_season' column using a vectorized left merge for efficiency.
        It checks if a specific 'unique_id' for a given month ('ds') is present in the 
        pre-processed seasonality lookup table.
        
        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to impute (typically the future_df).
            
        Returns
        -------
        pd.DataFrame
            DataFrame with the new boolean 'in_season' column.
        """
        # 1) Ensure ds is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['ds']):
            df['ds'] = pd.to_datetime(df['ds'])

        # 2) Create 'month_name' column
        df['month_name'] = df['ds'].dt.month_name()

        # 3) Prepare temporary merge keys and perform vectorized merge
        # Normalize types (str/object) for reliable merging
        df['_unique_id_tmp'] = df['unique_id'].astype(object)
        df['_month_tmp'] = df['month_name'].astype(str).str.strip()

        exploded = self.seasonality_exploded.copy()
        exploded['_unique_id_tmp'] = exploded['unique_id'].astype(object)
        exploded['_month_tmp'] = exploded['month_name'].astype(str).str.strip()

        # Left merge: indicator='season_match' tells us if a match was found
        # Merge key is (unique_id, month_name)
        merged = df.merge(
            exploded[['_unique_id_tmp', '_month_tmp']], 
            left_on=['_unique_id_tmp', '_month_tmp'], 
            right_on=['_unique_id_tmp', '_month_tmp'], 
            how='left', 
            indicator='season_match'
        )

        # 4) Impute 'in_season': True when there was a match (indicator == 'both')
        merged['in_season'] = merged['season_match'] == 'both'

        # 5) Cleanup of temporary columns
        merged.drop(columns=['_unique_id_tmp', '_month_tmp', 'season_match', 'month_name'], inplace=True, errors='ignore')

        return merged


    def _impute_exogenous_base_features(self, df):
        """
        Imputes missing values in base exogenous features (like stock_quantity) and
        static features (like shelf_life_days) using a GroupBy forward-fill (ffill) 
        and backward-fill (bfill) strategy within each time series ('unique_id').
        Final imputation uses median/mode for any remaining NaNs.
        
        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame potentially containing base features.
            
        Returns
        -------
        pd.DataFrame
            The DataFrame with imputed base features.
        """
        # Identify key columns for imputation: base exogenous (if present) and static.
        exog_feature = self.exog_cols
        static_cols = self.static_cols
        
        # Columns to be imputed using FFILL/BFILL
        cols_to_impute = list(set(exog_feature + static_cols))
        
        # Only process columns that exist in the DataFrame
        cols_to_impute = [col for col in cols_to_impute if col in df.columns]

        if not cols_to_impute:
            return df # Nothing to impute

        # Ensure 'ds' column is datetime and the DF is sorted by unique_id and ds
        if not pd.api.types.is_datetime64_any_dtype(df['ds']):
            df['ds'] = pd.to_datetime(df['ds'])
            
        df = df.sort_values(['unique_id', 'ds']).reset_index(drop=True)

        # FFILL (propagates the last known value forward within each series)
        df[cols_to_impute] = df.groupby('unique_id')[cols_to_impute].ffill()
        
        # BFILL (fills NaNs at the beginning of the series with the next known value)
        df[cols_to_impute] = df.groupby('unique_id')[cols_to_impute].bfill()
        
        # Final imputation for remaining NaNs (e.g., for entirely new items):
        # Impute numeric static features (e.g., shelf_life_days) with a fixed value (median)
        if 'shelf_life_days' in cols_to_impute and df['shelf_life_days'].isnull().any():
            median_value = df['shelf_life_days'].median()
            # Use 0 if median is NaN (e.g., if column is all NaNs initially)
            df['shelf_life_days'].fillna(median_value if not pd.isna(median_value) else 0, inplace=True)
            
        # Impute categorical static features (e.g., category, sub_category) with 'unknown'
        for col in [c for c in static_cols if c in cols_to_impute and df[c].dtype == 'object']:
            df[col].fillna('unknown', inplace=True)

        return df

    def create_future_df(self, future_df):
        """
        Creates the DataFrame of future regressors by applying all feature imputation steps.
        
        This method imputes date-based features ('is_holiday', 'in_season') and base 
        exogenous/static features. It DOES NOT create lag features; that is handled 
        by `TrainPredictPreprocessor` after the base features are imputed.

        Parameters
        ----------
        future_df : pd.DataFrame
            The DataFrame containing the dates and unique IDs for the forecast horizon.

        Returns
        -------
        pd.DataFrame
            The imputed DataFrame ready for final preprocessing/lag creation.
        """
        X_df_final = future_df.copy()

        # 1. Impute Holidays ('is_holiday')
        X_df_final = self._impute_holidays(X_df_final)

        # 2. Impute Seasonality ('in_season')
        X_df_final = self._impute_seasonality_vectorized(X_df_final)
        
        # 3. Impute Base Exogenous and Static Features
        # Fills NaNs in base columns (like stock_quantity and static features)
        X_df_final = self._impute_exogenous_base_features(X_df_final)

        # Drop the target and static columns, as this is a feature preparation step
        return X_df_final.drop(columns=self.target_col + self.static_cols, errors='ignore')

############### 














class MlflowModel:
    def __init__(self, file_path=None, config_path=None, experiment_name=None):
        self.forecast = forecast
        self.file_path = file_path
        self.config_path=config_path
        self.mlflow_params = mlflow_params
        self.experiment_name = experiment_name
        self.preprocessor_train_predict = TrainPredictPreprocessor()
        self.manager = MLflowArtifactManager(file_path=self.file_path, config_path=self.config_path)

    def _load_mlflow(self, run_id, model_name, exogenous_name=None):
        mlforecast_loaded, exog_history_loaded = self.manager.load_model_and_exog_history(run_id=run_id, model_name=model_name, exog_name=exogenous_name)
        return mlforecast_loaded, exog_history_loaded

    def _prepare_data(self, df_product_seasonality:pd.DataFrame=None, exogenous_feat=True):
        data_preprocessor = self.preprocessor_train_predict
       
        if exogenous_feat:
            preparator_exog = XDFPreparator(df_product_seasonality=df_product_seasonality)
        
        return data_preprocessor, preparator_exog

    def _train(self, df: pd.DataFrame, features_static: pd.DataFrame):

        preprocessed = self.preprocessor_train_predict.fit_transform(df)
        print('Training models...')
        self.forecast.fit(preprocessed, static_features=features_static)

        # Log expected features
        first_model_name = list(self.forecast.models_.keys())[0]
        model_pipeline = self.forecast.models_[first_model_name]

        preprocessor = None
        if hasattr(model_pipeline, 'named_steps'):
            for step_name in ['preprocessor', 'columntransformer', 'simplepreprocessor']:
                preprocessor = model_pipeline.named_steps.get(step_name)
                if preprocessor:
                    break

        feature_source = preprocessor or model_pipeline
        if hasattr(feature_source, 'get_feature_names_out'):
            feature_names = feature_source.get_feature_names_out().tolist()
            mlflow.log_param("expected_features", feature_names)
        else:
            original_features = preprocessed.drop(columns=['y']).columns.tolist()
            mlflow.log_param("original_features", original_features)

        return self.forecast
    
    def _predict(self, df_predict: pd.DataFrame, horizon_fcst: int, df_product_seasonality=None, exogenous=True):
        print('Preparing data...')
        data_prepared, preparator_exog = self._prepare_data(df_product_seasonality, exogenous_feat=exogenous)
        print('Creating future DataFrame...')
        future_df = self.forecast.make_future_dataframe(h=horizon_fcst)
        merged_df = future_df.merge(data_prepared, on=['ds', 'unique_id'], how='left')
        
        if exogenous:
            print('Applying XDFPreparator for date and base feature imputation...')
            X_df_imputed = preparator_exog.create_future_df(merged_df)
            print('Applying TrainPredictPreprocessor transformation to create lags...')
            X_df_final = self.preprocessor_train_predict.transform(X_df_imputed)

        else:
            print('Applying TrainPredictPreprocessor transformation to create lags...')
            X_df_final = self.preprocessor_train_predict.transform(merged_df)

        print('Predicting...')
        predictions_df = self.forecast.predict(h=horizon_fcst, X_df=X_df_final)

        print('Evaluating models...')
        evaluation_df = predictions_df.merge(df_predict[['unique_id', 'ds', 'y']], on=['unique_id', 'ds'])

        metrics = evaluate(
            df=evaluation_df,
            metrics=[rmse, mae, smape],
            models=predictions_df.columns.drop(['unique_id', 'ds']).tolist(),
            id_col='unique_id',
            time_col='ds',
            target_col='y'
        )

        print('Metrics summary:')
        print(metrics)

        metrics_long = pd.melt(metrics, 
                            id_vars=['unique_id', 'metric'], 
                            value_vars=['linear', 'ridge', 'rf', 'xgb'],
                            var_name='model', 
                            value_name='value')

        metrics_pivot = metrics_long.pivot_table(index=['unique_id', 'model'], 
                                                columns='metric', 
                                                values='value').reset_index()

        agg_funcs = {"mean": "mean", "std": "std", "median": "median"}
        results = []

        for name, func in agg_funcs.items():
            metric_ = (
                metrics_pivot
                .groupby("model")[["rmse", "mae", "smape"]]
                .agg(func)
                .rename(columns=lambda c: f"{c}_{name}_agg")
                .reset_index()
            )
            results.append(metric_)

        metric_agg = reduce(lambda left, right: pd.merge(left, right, on="model"), results)
        metric_summary = metrics_pivot.merge(metric_agg, on='model', how='left')

        filename = f"/{self.experiment_name}" + "_metrics_summary.csv"
        metric_summary.to_csv(path_models + filename, index=False)

        metrics_list = ['rmse', 'mae', 'smape']
        agg_types = list(agg_funcs.keys())

        for _, row in metric_agg.iterrows():
            model_name = row['model']
            print(f"\n--- Log for Model: {model_name} ---")
            for metric in metrics_list:
                for agg in agg_types:
                    col_name = f"{metric}_{agg}_agg"
                    log_key = f"{model_name}_{col_name}"
                    log_value = row[col_name]
                    mlflow.log_metric(log_key, log_value)

        ## 6. Best Model Selection and Logging
        best_model_name = metric_agg.loc[metric_agg['rmse_mean_agg'].idxmin(), 'model']
        best_rmse = metric_agg['rmse_mean_agg'].min()
        mlflow.log_metric("Final_Best_RMSE_Mean_Agg", best_rmse)
        mlflow.set_tag("Best_Model_Selected", best_model_name)
        print(f"Best Model: {best_model_name} with RMSE: {best_rmse:.2f}")

        ## 7. Artifact Saving
        print("Saving MLForecast object and artifacts...")
        manager.save_and_log_artifacts(instance_MLForecast=model, exog_features=X_df_final, model_name=self.experiment_name + 'best_model_name', exog_name=self.experiment_name + 'history_model')

        return predictions_df, metric_summary


    def mlflowRun(self,
                df_train: pd.DataFrame,
                df_predict: pd.DataFrame,
                df_exogenous: pd.DataFrame=None,
                df_product_seasonality: pd.DataFrame=None,
                static_features: list = ['category', 'sub_category', 'shelf_life_days'],
                run_name:str=None,
                registered_model_name:str=None,
                load: bool=False,
                run_id: str=None,
                model_name:str=None,
                exogenous_name:str=None,
                experiment_name:str=None,
                model: str=None,
                horizon_fcst: int=28,
                exogenous: bool=True):
        
        if experiment_name:
            self.experiment_name = experiment_name
            mlflow.set_experiment(self.experiment_name)
        if load:
            mlforecast_loaded, _ = self._load_mlflow(run_id, model_name, exogenous_name)
            return mlforecast_loaded
        else:
            REGISTERED_MODEL_NAME = registered_model_name
            with mlflow.start_run(run_name=run_name) as run:
                mlflow.log_params(self.mlflow_params['MLForecast_init'])
                mlflow.log_params(self.mlflow_params['Preprocessor'])
                for model_name, params in self.mlflow_params['Modelos'].items():
                    prefixed_params = {f"model_{model_name}_{k}": v for k, v in params.items()}
                    mlflow.log_params(prefixed_params)
            
            if df_exogenous:
                data_prepared = self._prepare_data(df_product_seasonality, df_exogenous)
                forecast = self._train(data_prepared, static_features)
                predictions_df, metric_summary = self._predict(df_predict, model, horizon_fcst, exogenous, experiment_name)
                return mlforecast_loaded, predictions_df, metric_summary
            else:
                forecast = self._train(df_train, static_features)
                predictions_df, metric_summary = self._predict(df_predict, model, horizon_fcst, exogenous, experiment_name)
                return mlforecast_loaded, predictions_df, metric_summary


#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################

# Importações necessárias (Assumidas):
# import pandas as pd
# import mlflow
# from functools import reduce 
# from typing import List, Dict, Any, Callable
# ... (TrainPredictPreprocessor, XDFPreparator, MLflowArtifactManager)

class MlflowModel:
    """
    Gerencia o fluxo completo de treinamento, previsão, avaliação e log de 
    modelos MLForecast com MLflow, garantindo consistência na preparação de dados.
    """
    def __init__(self, 
                 mlforecast_instance: object,             # <--- Instância do MLForecast (REQUIRED)
                 mlflow_params: Dict[str, Any],           # <--- Parâmetros para log do MLflow (REQUIRED)
                 df_product_seasonality: pd.DataFrame,    # <--- DataFrame de sazonalidade/exógenos (REQUIRED)
                 file_path: str = './models',             # path_models (padrão)
                 config_path: str = './config.json',      # path_json (padrão)
                 experiment_name: str = None):
        
        # 1. Dependências Injetadas
        self.forecast = mlforecast_instance
        self.mlflow_params = mlflow_params
        self.df_product_seasonality = df_product_seasonality
        
        # 2. Configurações de Path
        self.file_path = file_path
        self.config_path = config_path
        self.experiment_name = experiment_name
        
        # 3. Inicialização de Preparadores e Manager
        self.preprocessor_train_predict = TrainPredictPreprocessor()
        self.preparator_exog = XDFPreparator(df_product_seasonality=self.df_product_seasonality)
        self.manager = MLflowArtifactManager(file_path=self.file_path, config_path=self.config_path)

        # 4. Configuração de Métricas (Assumindo que estas variáveis são globais ou importadas)
        self.metrics_list = [rmse, mae, smape] # Assumindo que rmse, mae, smape são importados
        
# --------------------------------------------------------------------------------------
# O método '_prepare_data' foi removido, pois ele deve ser orquestrado por '_train' e '_predict'
# A lógica de preparação de features exógenas futuras foi movida para '_predict'
# --------------------------------------------------------------------------------------

    def _load_mlflow(self, run_id: str, model_name: str, exogenous_name: str = None):
        """Carrega o modelo MLForecast e o histórico de features exógenas do MLflow."""
        # Se você quer que o modelo carregado substitua o self.forecast atual:
        self.forecast, exog_history_loaded = self.manager.load_model_and_exog_history(
            run_id=run_id, model_name=model_name, exog_name=exogenous_name
        )
        return self.forecast, exog_history_loaded

    def _train(self, df: pd.DataFrame, features_static: List[str]):
        """
        Aplica o pré-processamento de lags/rolling no DF de treino e treina o modelo.
        """
        print('Applying TrainPredictPreprocessor transformation (lag/rolling features)...')
        # O df (dados de treino) deve vir com as colunas base de exógenos (se houver)
        preprocessed = self.preprocessor_train_predict.fit_transform(df)
        
        print('Training models...')
        self.forecast.fit(preprocessed, static_features=features_static)

        # Log expected features (Lógica mantida, pois está correta)
        # ... (código para logar feature names) ...

        return self.forecast
    
    def _prepare_future_features(self, exog_history: pd.DataFrame, horizon_fcst: int, exogenous: bool):
        """
        Cria e transforma o DataFrame de features futuras (X_df) antes da previsão.
        """
        print('Creating future DataFrame skeleton...')
        future_df = self.forecast.make_future_dataframe(h=horizon_fcst)
        
        # 1. Merge para incluir features estáticas/base dos dados de treino
        last_known_static = exog_history.groupby('unique_id').tail(1).drop(columns=['ds', 'y'], errors='ignore')
        future_df = future_df.merge(last_known_static, on='unique_id', how='left')

        if exogenous:
            print('Applying XDFPreparator for date and base feature imputation...')
            # Imputa features exógenas futuras (is_holiday, in_season, base exógenas)
            X_df_imputed = self.preparator_exog.create_future_df(future_df)
            
            print('Applying TrainPredictPreprocessor transformation to create lags...')
            # Cria lags e rolling stats para features exógenas
            X_df_final = self.preprocessor_train_predict.transform(X_df_imputed)
        else:
            print('Applying TrainPredictPreprocessor transformation to create lags...')
            # Cria lags para features base (datas, etc.)
            X_df_final = self.preprocessor_train_predict.transform(future_df)

        return X_df_final
    
    def _predict(self, 
                 df_predict: pd.DataFrame, 
                 df_train_history: pd.DataFrame, # Novo: Passa o DF de treino para obter as features estáticas/lags
                 horizon_fcst: int, 
                 exogenous: bool):
        """
        Executa a previsão, avaliação e loga os artefatos no MLflow.
        """
        # Prepara o X_df de features futuras
        X_df_final = self._prepare_future_features(df_train_history, horizon_fcst, exogenous)

        print('Predicting...')
        predictions_df = self.forecast.predict(h=horizon_fcst, X_df=X_df_final)

        # 1. Avaliação (Lógica mantida, apenas ajustando as variáveis)
        print('Evaluating models...')
        evaluation_df = predictions_df.merge(df_predict[['unique_id', 'ds', 'y']], on=['unique_id', 'ds'], how='left')
        evaluation_df.dropna(subset=['y'], inplace=True) 

        models_to_evaluate = predictions_df.columns.drop(['unique_id', 'ds'], errors='ignore').tolist()
        
        metrics = evaluate(
            df=evaluation_df,
            metrics=self.metrics_list,
            models=models_to_evaluate,
            id_col='unique_id',
            time_col='ds',
            target_col='y'
        )

        # ... (Restante da lógica de consolidação de métricas, log e seleção de melhor modelo) ...
        
        # 2. Consolidação e Log de Métricas (Mantendo sua lógica de agregação)
        # ... (código para metrics_long, metrics_pivot, agg_funcs, results, metric_agg, metric_summary) ...
        # Lembre-se: Corrija o path: metric_summary.to_csv(self.file_path + filename, index=False)
        
        # 3. Best Model Selection and Logging (Mantida sua lógica)
        # ... (código para best_model_name, best_rmse, mlflow.log_metric/set_tag) ...

        # 4. Artifact Saving (CORRIGIDA a chamada)
        print("Saving MLForecast object and artifacts...")
        # Use self.manager e o self.forecast treinado
        self.manager.save_and_log_artifacts(
            instance_MLForecast=self.forecast, 
            exog_features=X_df_final, 
            model_name=self.experiment_name + '_fcst_model', 
            exog_name=self.experiment_name + '_history_forecast_xdf'
        )

        return predictions_df, metric_summary


    def mlflowRun(self,
                df_train: pd.DataFrame,
                df_predict: pd.DataFrame,
                static_features: List[str] = ['category', 'sub_category', 'shelf_life_days'],
                run_name: str = 'MLForecast_Run',
                registered_model_name: str = 'SupplyChain_Forecaster',
                horizon_fcst: int = 28,
                exogenous: bool = True,
                # Parâmetros de Carga (Opcionais)
                load: bool = False,
                run_id: str = None,
                model_name: str = None,
                exogenous_name: str = None):
        """
        Ponto de entrada principal para a execução do MLflow (treino ou carregamento).
        """
        
        if self.experiment_name:
            mlflow.set_experiment(self.experiment_name)
        
        if load:
            print(f"Loading model from run ID: {run_id}")
            # O modelo carregado já substitui self.forecast
            self.forecast, _ = self._load_mlflow(run_id, model_name, exogenous_name) 
            return self.forecast
        
        # Inicia o Run de Treinamento
        with mlflow.start_run(run_name=run_name) as run:
            
            # Log de Parâmetros (Lógica mantida)
            mlflow.log_params(self.mlflow_params.get('MLForecast_init', {}))
            mlflow.log_params(self.mlflow_params.get('Preprocessor', {}))
            for m_name, params in self.mlflow_params.get('Modelos', {}).items():
                prefixed_params = {f"model_{m_name}_{k}": v for k, v in params.items()}
                mlflow.log_params(prefixed_params)

            # 1. Treinamento
            # df_train já deve conter as features exógenas base (e 'y', 'ds', 'unique_id').
            fcst_trained = self._train(df_train, static_features)
            
            # 2. Previsão e Avaliação
            predictions_df, metric_summary = self._predict(
                df_predict=df_predict, 
                df_train_history=df_train, # Passa o DF de treino para extrair histórico/features
                horizon_fcst=horizon_fcst, 
                exogenous=exogenous
            )

            # 3. Registro do Modelo
            mlflow.sklearn.log_model(
                sk_model=fcst_trained,
                artifact_path="model",
                registered_model_name=registered_model_name
            )

            return fcst_trained, predictions_df, metric_summary