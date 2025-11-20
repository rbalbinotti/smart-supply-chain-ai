"""
Preprocessing Functions for Smart Supply Chain AI - Demand Forecasting

This module provides specialized preprocessing classes and utilities for time series
forecasting in supply chain demand prediction. It includes transformers for feature
engineering, data preparation, and seasonal pattern handling optimized for MLForecast.

Classes:
    SimplePreprocessor: Automated preprocessing pipeline for categorical, boolean, 
                       and numeric features with encoding, imputation, and scaling.
    
    XDFPreparator: Prepares exogenous feature DataFrames for MLForecast predictions,
                  handling holidays, seasonality imputation, and lag feature generation.

Key Features:
    - Automated column type detection and processing
    - Holiday and seasonality pattern recognition
    - Robust handling of missing values and data normalization
    - Vectorized operations for performance optimization
    - Integration with MLForecast forecasting framework

Dependencies:
    pandas, numpy, scikit-learn, holidays, ast, collections.abc

Author: Roberto Rosário Balbinotti
Created: 2025
Version: 1.0
"""

import pandas as pd
import numpy as np
import holidays
import os
import json
import ast
from collections.abc import Iterable
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer


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
    """

    def __init__(self, static_features=None):
        if static_features is None:
            self.static_features = ['category', 'sub_category', 'shelf_life_days']
        else:
            self.static_features = static_features
        
        self.column_transformer_ = None
        self.expected_features_ = None
    
    def fit(self, X, y=None):
        """Fit preprocessing steps to the given dataset X."""
        # Define expected column types
        categorical_cols = ['category', 'sub_category']
        boolean_cols = ['in_season', 'is_holiday']
        numeric_cols = X.select_dtypes(np.number).columns.tolist()
        
        # Define preprocessing pipelines for each data type
        self.column_transformer_ = ColumnTransformer(
            transformers=[
                # Categorical: impute missing values and apply one-hot encoding
                ('cat', Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
                ]), categorical_cols),

                # Boolean: encode binary variables (drop one category if binary)
                ('bool', OneHotEncoder(
                    drop='if_binary', handle_unknown='ignore', sparse_output=False
                ), boolean_cols),

                # Numeric: impute with median and apply power transformation for normalization
                ('numeric', Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', PowerTransformer(method='yeo-johnson', standardize=True))
                ]), numeric_cols),
            ],
            remainder='passthrough',  # Keep any other columns unchanged
            verbose_feature_names_out=False
        )
        
        # Fit the column transformer to the data
        self.column_transformer_.fit(X)
        # Save features after transformation
        self.expected_features_ = self.get_feature_names_out()

        return self

    def transform(self, X):
        """Apply the fitted preprocessing steps to the dataset X."""
        return self.column_transformer_.transform(X)
    
    def get_feature_names_out(self):
        """Get the output feature names after transformation"""
        return self.column_transformer_.get_feature_names_out()


class XDFPreparator:
    """
    Class to prepare the X_df_final DataFrame (regressors) for mlforecast.
    """
    def __init__(self, exog_df, json_path):
        """
        Initializes the preparator with validation data and the path to the JSONs.
        
        Args:
            exog_df (pd.DataFrame): The validation DataFrame containing already computed regressors
                                     (lags, rolling means, etc.).
            json_path (str): Path to the directory containing catalog JSON files 
                             (products, products_categories, etc.).
        """
        self.exog_df = exog_df
        self.json_path = json_path
        self.store_catalog = self._load_catalog_jsons()
        self.country_holidays = holidays.country_holidays('Brazil')
        self.products_seasonality = self._prepare_seasonality_data() # This will now also populate self.seasonality_exploded

    def _load_catalog_jsons(self):
        """Loads the catalog JSON files."""
        arch_json = ['products', 'products_categories', 'suppliers']
        catalog = {}
        for name in arch_json:
            file_path = os.path.join(self.json_path, f"{name}.json")
            with open(file_path, "r", encoding="utf-8") as f:
                catalog[name] = json.load(f)
        return catalog

    def _prepare_seasonality_data(self):
        """
        Prepares and returns two items:
         - products_seasonality: DataFrame with columns ['product','seasonality'] where seasonality is a list
         - seasonality_exploded: DataFrame with columns ['product','month_name'] with one row per month in seasonality
        """
        products_df = pd.DataFrame.from_dict(self.store_catalog['products']).T.reset_index().rename(columns={'index': 'product'})
        if 'id' in products_df.columns and 'product_id' not in products_df.columns:
            products_df.rename(columns={'id': 'product_id'}, inplace=True)

        # ensure seasonality column exists
        if 'seasonality' not in products_df.columns:
            products_df['seasonality'] = None

        # normalize seasonality to lists (vectorized)
        products_df['seasonality'] = products_df['seasonality'].apply(self._normalize_seasonality_value)

        # create the exploded table product -> month_name (one record per month in seasonality)
        seasonality_exploded = products_df[['product', 'seasonality']].explode('seasonality').rename(columns={'seasonality': 'month_name'})
        # remove rows where month_name is empty
        seasonality_exploded['month_name'] = seasonality_exploded['month_name'].astype(object)
        seasonality_exploded = seasonality_exploded[seasonality_exploded['month_name'].notna()].reset_index(drop=True)

        # ensure month_name strings are normalized (e.g., 'January' with consistent capitalization)
        seasonality_exploded['month_name'] = seasonality_exploded['month_name'].astype(str).str.strip()

        # store in attributes for later use
        self.products_seasonality = products_df[['product', 'seasonality']].copy()
        self.seasonality_exploded = seasonality_exploded[['product', 'month_name']].drop_duplicates().reset_index(drop=True)

        return self.products_seasonality

    def _classify_holiday(self, date):
        """Classifies whether a date is a national holiday."""
        return date in self.country_holidays

    def _impute_holidays(self, df):
        """Imputes the 'is_holiday' column."""
        df['is_holiday'] = df['ds'].apply(self._classify_holiday)
        return df
    
    def _normalize_seasonality_value(self, val):
        """
        Normalizes seasonality to always return a list (default: []).
        Covered cases:
         - None / NaN -> []
         - list/tuple/set/np.ndarray/pd.Series/pd.Index/range/... -> list (filtering NaNs)
         - string -> attempt ast.literal_eval (e.g.: "['January','February']")
                     otherwise split by comma, otherwise list with the string itself
         - other scalar types -> [val]
        """
        # 1) Explicit None
        if val is None:
            return []

        # 2) Strings: treat separately (strings are iterable but we want to treat them as scalars)
        if isinstance(val, (str, bytes)):
            v = val.strip()
            # attempt to interpret as a literal list
            if (v.startswith('[') and v.endswith(']')) or (v.startswith('(') and v.endswith(')')):
                try:
                    parsed = ast.literal_eval(v)
                    if isinstance(parsed, (list, tuple, set, np.ndarray, pd.Series)):
                        # convert and filter nulls
                        return [x for x in list(parsed) if not pd.isna(x)]
                except Exception:
                    pass
            # if contains comma, split
            if ',' in v:
                return [p.strip() for p in v.split(',') if p.strip()]
            return [v]

        # 3) Iterables (excluding strings): convert to list and filter NaNs
        #    This catches np.ndarray, pd.Series, pd.Index, list, tuple, set, range, etc.
        if isinstance(val, Iterable):
            try:
                # convert to list — many iterables support list()
                as_list = list(val)
            except Exception:
                # fallback: wrap the value (not very common)
                as_list = [val]

            # if after converting we have an empty list -> []
            if len(as_list) == 0:
                return []

            # filter null values (pd.isna on each element)
            cleaned = [x for x in as_list if not pd.isna(x)]
            return cleaned

        # 4) Safe check for NaN in scalars
        if pd.isna(val):
            return []

        # 5) Any other scalar type -> wrap
        return [val]


    def _check_seasonality(self, row):
        """Receives a row with 'month_name' and 'seasonality' already normalized (list)."""
        received_month = row['month_name']
        seasonality_list = row.get('seasonality', [])
        if not seasonality_list:
            return False
        # safety: ensure seasonality_list is iterable
        try:
            return received_month in seasonality_list
        except TypeError:
            return False

    def _impute_seasonality(self, df):
        """Imputes the 'in_season' column robustly — normalizes seasonality before checking."""
        # Ensure df has a 'product' column (fallback already handled in the previous version)
        # Merging with seasonality if necessary was done before, so here we assume the 'seasonality' column exists
        if 'seasonality' not in df.columns:
            df['seasonality'] = None

        # Normalize the seasonality column to lists
        df['seasonality'] = df['seasonality'].apply(self._normalize_seasonality_value)

        # Ensure ds column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['ds']):
            df['ds'] = pd.to_datetime(df['ds'])

        # Create column with month name
        df['month_name'] = df['ds'].dt.month_name()

        # Apply check (safe because seasonality is now always a list)
        df['in_season'] = df.apply(self._check_seasonality, axis=1)

        # Cleanup
        drop_cols = ['month_name', 'seasonality']
        df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
        return df


    def _impute_seasonality_vectorized(self, df):
        """
        Vectorized version for imputing in_season:
         1. Ensures ds is datetime and creates month_name
         2. If necessary, attaches 'product' column from unique_id -> product mapping
         3. Performs a left merge with seasonality_exploded on ['product','month_name']
         4. in_season = True where there was a match, False otherwise
        """
        # 0) Setup
        if not hasattr(self, 'seasonality_exploded') or self.seasonality_exploded is None:
            # initialize if not yet done (this also populates self.products_seasonality)
            self._prepare_seasonality_data()

        # 1) Ensure ds is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['ds']):
            df['ds'] = pd.to_datetime(df['ds'])

        # 2) Ensure 'product' column exists — if not, try to map from unique_id using products_seasonality/exog_df
        if 'product' not in df.columns:
            # first, try to map from exog_df (in case it contains unique_id->product)
            if {'unique_id', 'product'}.issubset(self.exog_df.columns):
                mapping = self.exog_df[['unique_id', 'product']].drop_duplicates()
                df = df.merge(mapping, on='unique_id', how='left')
            else:
                # try to map from seasonality_exploded itself if it has unique_id (rare)
                if 'unique_id' in self.seasonality_exploded.columns:
                    map_prod = self.seasonality_exploded[['unique_id', 'product']].drop_duplicates()
                    df = df.merge(map_prod, on='unique_id', how='left')
                else:
                    # ensure empty column to avoid later errors
                    df['product'] = None

        # 3) Create month_name column in df (same format as seasonality_exploded)
        df['month_name'] = df['ds'].dt.month_name()

        # 4) Perform vectorized merge on ['product','month_name']
        # Normalize types (str) to avoid mismatches
        df['_product_tmp'] = df['product'].astype(object)
        df['_month_tmp'] = df['month_name'].astype(str).str.strip()

        exploded = self.seasonality_exploded.copy()
        exploded['_product_tmp'] = exploded['product'].astype(object)
        exploded['_month_tmp'] = exploded['month_name'].astype(str).str.strip()

        merged = df.merge(exploded[['_product_tmp', '_month_tmp']], on=['_product_tmp', '_month_tmp'], how='left', indicator='season_match')

        # 5) in_season = True when there was a match (indicator == 'both')
        merged['in_season'] = merged['season_match'] == 'both'

        # 6) Cleanup of temporary columns
        merged.drop(columns=['_product_tmp', '_month_tmp', 'season_match', 'month_name'], inplace=True, errors='ignore')

        # If desired, keep product; if not, it can be dropped later (but keep until used)
        return merged


    def _impute_stock_quantity_lags(self, df):
        """Imputes lag and rolling mean columns for stock_quantity."""
        # Forward-fill for middle values
        df = df.groupby('unique_id').apply(lambda g: g.ffill()).reset_index(drop=True)
        # Backward-fill for initial values (that couldn't be filled)
        df = df.groupby('unique_id').apply(lambda g: g.bfill()).reset_index(drop=True)
        # Reorder at the end
        df = df.sort_values(['unique_id', 'ds'])
        return df

    def create_future_df(self, future_df):
        """
        Creates the final regressor DataFrame for mlforecast.

        Args:
            future_df (pd.DataFrame): Future DataFrame generated by fcst.make_future_dataframe(h=horizon).
            
        Returns:
            pd.DataFrame: The X_df_final with all regressors filled.
        """
        regressor_cols = [
            'ds', 'unique_id', 'product', 'is_holiday', 'in_season', 
            'stock_quantity_lag1', 'stock_quantity_lag2', 'stock_quantity_lag7', 
            'stock_quantity_expanding_mean_lag1', 'stock_quantity_rolling_mean_lag7_window_size7'
        ]
        
        # 1. Select and perform initial merge with existing regressors
        X_regressor = self.exog_df[regressor_cols].copy()
        X_df_final = future_df.merge(X_regressor, on=['unique_id', 'ds'], how='left')


        # 2. Impute Holidays
        X_df_final = self._impute_holidays(X_df_final)

        # 3. Impute Seasonality
        X_df_final = self._impute_seasonality_vectorized(X_df_final)
        
        # 4. Impute Lags and Rolling Means (stock_quantity)
        X_df_final = self._impute_stock_quantity_lags(X_df_final)

        # 5. Remove referential column product
        if 'product' in X_df_final.columns:
            X_df_final = X_df_final.drop(columns=['product'])
        
        # print(f"X_df_final completed. Remaining NaNs: {X_df_final.isnull().sum().sum()}")
        return X_df_final




