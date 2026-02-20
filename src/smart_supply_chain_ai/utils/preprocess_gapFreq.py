import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder, PowerTransformer
from sklearn.impute import SimpleImputer
from sktime.transformations.series.adapt import TabularToSeriesAdaptor


import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from typing import Optional
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from sklearn.base import BaseEstimator, TransformerMixin


class TimeSeriesIntegrityTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer for enforcing continuity and consistency in panel time series data.

    This class:
    - Aggregates duplicate timestamps per ID.
    - Fills missing date gaps for each identifier.
    - Ensures a fixed frequency across all series.
    - Applies sensible defaults for missing values.

    Attributes:
        id_col (str): Column name for unique identifiers (e.g., 'store_id').
        date_col (str): Column name for timestamps.
        target_col (str): Column name for the target variable to be summed.
        freq (str): Pandas frequency string (e.g., 'D' for daily).
        unique_ids_ (np.ndarray): Array of unique IDs learned during fitting.
    """

    def __init__(self, id_col: str, date_col: str, target_col: str, freq: str = 'D') -> None:
        self.id_col = id_col
        self.date_col = date_col
        self.target_col = target_col
        self.freq = freq
        self.unique_ids_: Optional[np.ndarray] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "TimeSeriesIntegrityTransformer":
        """
        Identify and store unique identifiers from the dataset.

        Args:
            X (pd.DataFrame): Input DataFrame containing id_col.
            y (pd.Series, optional): Ignored, present for compatibility.

        Returns:
            TimeSeriesIntegrityTransformer: Fitted transformer instance.
        """
        self.unique_ids_ = X[self.id_col].unique()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize time series data:
        - Convert timestamps to datetime.
        - Aggregate duplicate entries.
        - Fill missing dates for each ID.
        - Apply default fills for missing values.

        Args:
            X (pd.DataFrame): Raw input data.

        Returns:
            pd.DataFrame: Sanitized DataFrame with MultiIndex [id, date].
        """
        df = X.copy()

        # Ensure proper datetime format and optimize ID column
        df[self.date_col] = pd.to_datetime(df[self.date_col])
        if not isinstance(df[self.id_col].dtype, CategoricalDtype):
            df[self.id_col] = df[self.id_col].astype('category')

        # Define aggregation rules: sum for target, last known value for others
        agg_cols = {col: 'last' for col in df.columns if col not in [self.id_col, self.date_col]}
        if self.target_col in agg_cols:
            agg_cols[self.target_col] = 'sum'

        # Resolve duplicate timestamps by grouping
        df = df.groupby([self.id_col, self.date_col], observed=True).agg(agg_cols).reset_index()

        # Build complete date range across min/max
        start, end = df[self.date_col].min(), df[self.date_col].max()
        full_range = pd.date_range(start, end, freq=self.freq)

        # Construct MultiIndex for all IDs × full date range
        mux = pd.MultiIndex.from_product(
            [self.unique_ids_, full_range],
            names=[self.id_col, self.date_col]
        )

        # Reindex to enforce frequency and order
        df_clean = (
            df.set_index([self.id_col, self.date_col])
            .reindex(mux)
            .sort_index(level=[0, 1])
        )

        # Fill missing target values with 0 (no activity assumption)
        if self.target_col in df_clean.columns:
            df_clean[self.target_col] = df_clean[self.target_col].fillna(0)

        # Forward-fill other features within each ID
        df_clean = df_clean.groupby(level=self.id_col).ffill()

        # Apply default fills for remaining missing values
        df_clean[df_clean.select_dtypes(include='object').columns] = (
            df_clean.select_dtypes(include='object').fillna('missing')
        )
        df_clean[df_clean.select_dtypes(include='datetime').columns] = (
            df_clean.select_dtypes(include='datetime').fillna(0)
        )
        df_clean = df_clean.groupby(self.id_col).transform(lambda x: x.fillna(x.mode()[0]))

        return df_clean.reset_index()



class PrepareForecastingData(BaseEstimator, TransformerMixin):
    def __init__(self, 
                 date_col: str, 
                 id_col: str,
                 target_col: str, 
                 exog_features: list = None, 
                 agg_rules: dict = None,
                 frequency: str = "D"
                 ):

        self.date_col = date_col
        self.id_col = id_col
        self.target_col = target_col
        self.exog_features = exog_features or []
        self.frequency = frequency
        self.agg_rules = agg_rules or {target_col: 'sum'}
        
        self.cols_to_use = [id_col, date_col, target_col] + exog_features

    def _complete_index_fillna(self, X):
        all_unique_ids = X.index.get_level_values(self.id_col).unique()
        
        start_date = X.index.get_level_values(self.date_col).min()
        end_date = X.index.get_level_values(self.date_col).max()
        full_date_range = pd.date_range(start=start_date, end=end_date, freq=self.frequency)

        complete_index = pd.MultiIndex.from_product(
            [all_unique_ids, full_date_range], 
            names=[self.id_col, self.date_col]
        )

        df_reindexed = X.reindex(complete_index)
        
        if self.target_col in df_reindexed.columns:
            df_reindexed[self.target_col] = df_reindexed[self.target_col].fillna(0)

        for column in df_reindexed.columns:
            if column == self.target_col:
                continue
            
            if pd.api.types.is_numeric_dtype(df_reindexed[column]):
                df_reindexed[column] = df_reindexed.groupby(self.id_col, observed=False)[column].ffill().fillna(0)
            else:
                df_reindexed[column] = (
                    df_reindexed.groupby(self.id_col, observed=False)[column]
                    .ffill().fillna('missing')
                )

        return df_reindexed

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        dataframe = X[self.cols_to_use].copy()
        
        dataframe = dataframe.rename(columns={
            self.date_col: "date", 
            self.id_col: "unique_id", 
            self.target_col: "y"
        })
        dataframe[self.date_col] = pd.to_datetime(dataframe[self.date_col])
        dataframe = dataframe.drop_duplicates()
        
        agg_cols = [self.id_col, self.date_col]
        dataframe = dataframe.groupby(agg_cols, observed=False).agg(self.agg_rules).sort_index()
        
        dataframe = self._complete_index_fillna(dataframe)
        
        return dataframe

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)



class CategoricalNumericalPreprocessor(BaseEstimator, TransformerMixin):
    """
    Pré-processador customizado para lidar com variáveis categóricas e numéricas.
    Pode ser usado dentro de pipelines do sktime.
    """
    
    def __init__(self, 
                 categorical_cols=None,
                 numerical_cols=None,
                 cat_imputer_strategy='constant',
                 cat_imputer_fill_value='missing',
                 num_imputer_strategy='constant',
                 num_imputer_fill_value=0,
                 ordinal_unknown_value=-1,
                 ordinal_missing_value=-2,
                 use_power_transform=True):
        
        self.categorical_cols = categorical_cols or []
        self.numerical_cols = numerical_cols or []
        self.cat_imputer_strategy = cat_imputer_strategy
        self.cat_imputer_fill_value = cat_imputer_fill_value
        self.num_imputer_strategy = num_imputer_strategy
        self.num_imputer_fill_value = num_imputer_fill_value
        self.ordinal_unknown_value = ordinal_unknown_value
        self.ordinal_missing_value = ordinal_missing_value
        self.use_power_transform = use_power_transform
        
        # Inicializar componentes
        self.cat_imputer_ = None
        self.num_imputer_ = None
        self.ordinal_encoders_ = {}
        self.power_transformers_ = {}
        self.is_fitted_ = False
        
    def fit(self, X, y=None):
        """
        Fit do pré-processador.
        
        Parameters
        ----------
        X : pandas DataFrame
            Dados de entrada
        y : ignorado
            
        Returns
        -------
        self : objeto
            Pré-processador ajustado
        """
        X = X.copy()
        
        # 1. Imputação para categóricas
        if self.categorical_cols:
            # Verificar quais colunas categóricas existem em X
            available_cat_cols = [col for col in self.categorical_cols if col in X.columns]
            if available_cat_cols:
                # Imputar missing values
                X[available_cat_cols] = X[available_cat_cols].fillna(self.cat_imputer_fill_value)
                
                # Fit ordinal encoders
                for col in available_cat_cols:
                    encoder = OrdinalEncoder(
                        handle_unknown='use_encoded_value',
                        unknown_value=self.ordinal_unknown_value,
                        encoded_missing_value=self.ordinal_missing_value
                    )
                    encoder.fit(X[[col]])
                    self.ordinal_encoders_[col] = encoder
        
        # 2. Imputação e scaling para numéricas
        if self.numerical_cols:
            # Verificar quais colunas numéricas existem em X
            available_num_cols = [col for col in self.numerical_cols if col in X.columns]
            if available_num_cols:
                # Fit imputer
                self.num_imputer_ = SimpleImputer(
                    strategy=self.num_imputer_strategy,
                    fill_value=self.num_imputer_fill_value
                )
                X[available_num_cols] = self.num_imputer_.fit_transform(X[available_num_cols])
                
                # Fit power transformers
                if self.use_power_transform:
                    for col in available_num_cols:
                        transformer = PowerTransformer(method='yeo-johnson')
                        transformer.fit(X[[col]])
                        self.power_transformers_[col] = transformer
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        """
        Transforma os dados.
        
        Parameters
        ----------
        X : pandas DataFrame
            Dados de entrada
            
        Returns
        -------
        X_transformed : pandas DataFrame
            Dados transformados
        """
        if not self.is_fitted_:
            raise ValueError("O pré-processador deve ser ajustado antes de transformar")
        
        X = X.copy()
        
        # 1. Transformar categóricas
        if self.categorical_cols:
            available_cat_cols = [col for col in self.categorical_cols if col in X.columns]
            for col in available_cat_cols:
                if col in self.ordinal_encoders_:
                    # Imputar missing
                    X[col] = X[col].fillna(self.cat_imputer_fill_value)
                    # Transformar
                    X[col] = self.ordinal_encoders_[col].transform(X[[col]]).flatten()
        
        # 2. Transformar numéricas
        if self.numerical_cols:
            available_num_cols = [col for col in self.numerical_cols if col in X.columns]
            if available_num_cols:
                # Imputar
                if self.num_imputer_:
                    X[available_num_cols] = self.num_imputer_.transform(X[available_num_cols])
                
                # Transformar com power transformer
                if self.use_power_transform:
                    for col in available_num_cols:
                        if col in self.power_transformers_:
                            X[col] = self.power_transformers_[col].transform(X[[col]]).flatten()
        
        return X
    
    def fit_transform(self, X, y=None):
        """Fit e transform em uma única operação."""
        return self.fit(X, y).transform(X)
    
    def get_feature_names_out(self, input_features=None):
        """Retorna os nomes das features de saída."""
        if input_features is None:
            # Usar todas as colunas disponíveis
            output_cols = []
            if self.categorical_cols:
                output_cols.extend(self.categorical_cols)
            if self.numerical_cols:
                output_cols.extend(self.numerical_cols)
            return np.array(output_cols)
        else:
            return np.array(input_features)
    
    def get_params(self, deep=True):
        """Retorna parâmetros do pré-processador."""
        return {
            'categorical_cols': self.categorical_cols,
            'numerical_cols': self.numerical_cols,
            'cat_imputer_strategy': self.cat_imputer_strategy,
            'cat_imputer_fill_value': self.cat_imputer_fill_value,
            'num_imputer_strategy': self.num_imputer_strategy,
            'num_imputer_fill_value': self.num_imputer_fill_value,
            'ordinal_unknown_value': self.ordinal_unknown_value,
            'ordinal_missing_value': self.ordinal_missing_value,
            'use_power_transform': self.use_power_transform
        }
    
    def set_params(self, **parameters):
        """Define parâmetros do pré-processador."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    


class SktimePreprocessorAdapter(TabularToSeriesAdaptor):
    """
    Adaptador do pré-processador customizado para sktime.
    """
    def __init__(self, preprocessor):
        super().__init__(preprocessor)
        self.preprocessor = preprocessor
        
    def _check_X_y(self, X=None, y=None, return_metadata=False):
        """Override para evitar checagem de categorias."""
        # Bypass da verificação original
        if X is None:
            return None, None
        
        # Conversão básica
        from sktime.datatypes._panel._check import is_nested_dataframe
        
        if is_nested_dataframe(X):
            X_inner = X
        else:
            # Converter para DataFrame se necessário
            if not isinstance(X, pd.DataFrame):
                X_inner = pd.DataFrame(X)
            else:
                X_inner = X
        
        if return_metadata:
            metadata = {
                "scitype": "Panel",
                "mtype": "nested_univ",
                "is_univariate": X_inner.shape[1] == 1,
            }
            return X_inner, metadata
        else:
            return X_inner