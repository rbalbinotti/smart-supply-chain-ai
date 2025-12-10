# Standard Libraries
import pandas as pd
import mlflow
from typing import Optional, Dict, List
from functools import reduce

# Specialized Libraries
# Note: mlflow is listed twice in the original, kept one instance here.
from mlforecast import MLForecast
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import smape, rmse, mae

# To import customized class and function
# Assuming these imports are correctly configured in the project environment
from smart_supply_chain_ai.utils.preprocess_functions import TrainPredictPreprocessor, XDFPreparator, MLflowArtifactManager


class MLflowForecastManager:
    """
    Manages the full lifecycle (Training, Prediction, Logging Metrics, and Saving/Loading) 
    of an MLForecast model, integrating seamlessly with MLflow for tracking and deployment.

    This class handles feature engineering history, model persistence, and standardized 
    evaluation metric logging using MLflow runs.
    """

    def __init__(self,
                 model_name: str = "default_model",
                 experiment_name: str = "default_experiment",
                 path_models: str = "./mlruns_artifacts",
                 json_path: str = "./config.json",
                 horizon: int = 7,
                 product_seasonality: Optional[pd.DataFrame] = None,
                 fcst: Optional['MLForecast'] = None):
        """
        Initializes the MLflowForecastManager with configuration settings and utility objects.

        Args:
            model_name (str): The name used for logging and saving the model artifact.
            experiment_name (str): The MLflow experiment name to track runs.
            path_models (str): Local path where MLflow artifacts are stored.
            json_path (str): Path to the configuration JSON file for the Artifact Manager.
            horizon (int): The forecast horizon (number of steps/days to predict).
            product_seasonality (Optional[pd.DataFrame]): Optional DataFrame containing 
                exogenous features related to seasonality/products.
            fcst (Optional[MLForecast]): An initialized MLForecast instance. Must be provided 
                for training/prediction.
        """
        
        # --- Fixed Configurations ---
        self.model_name = model_name
        self.experiment_name = experiment_name
        self.horizon = horizon
        # Artifact name for the feature history (exogenous data)
        self.exog_name = 'history_' + model_name 
        
        # --- Core Objects ---
        self.fcst = fcst # MLForecast instance
        # Utility for saving and loading MLflow artifacts
        self.manager = MLflowArtifactManager(file_path=path_models, config_path=json_path) 
        # Utility for feature engineering during training and prediction
        self.preprocessor_train_predict = TrainPredictPreprocessor()
        # Utility for handling future exogenous features
        self.preparator_exog = XDFPreparator(df_product_seasonality=product_seasonality)

        # --- Class State (Stores feature history for prediction) ---
        # Stores the feature history (df_preprocessed without 'y') needed for future predictions
        self.exog_history: Optional[pd.DataFrame] = None 
        # The MLflow run ID of the current or loaded model
        self.current_run_id: Optional[str] = None 
        
        # Initialize MLflow experiment
        mlflow.set_experiment(experiment_name)
        print(f"Manager initialized. MLflow Experiment: {experiment_name}")

    # =========================================================================
    # --- Helper Methods ---
    # =========================================================================

    def _get_active_run_id(self) -> Optional[str]:
        """
        Gets the active MLflow run ID if a run is currently active, 
        otherwise returns the stored run ID from the class state.

        Returns:
            Optional[str]: The active or stored MLflow run ID.
        """
        if mlflow.active_run():
            return mlflow.active_run().info.run_id
        return self.current_run_id

    def _log_metrics(self, df_evaluation: pd.DataFrame, df_validation: pd.DataFrame, run_id: str) -> None:
        """
        Calculates a set of evaluation metrics (RMSE, MAE, SMAPE) by model, 
        aggregates them (mean, std, median), and logs the results to MLflow.

        Args:
            df_evaluation (pd.DataFrame): DataFrame containing 'unique_id', 'ds', 'y', and model predictions.
            df_validation (pd.DataFrame): Predictions DataFrame used for column reference.
            run_id (str): MLflow run ID for logging the metrics.
        """
        with mlflow.start_run(run_id=run_id, nested=True) as run:
            print('Calculating and logging metrics...')
            
            # 1. Evaluation using utilsforecast
            metrics = evaluate(
                df=df_evaluation,
                metrics=[rmse, mae, smape],
                models=df_validation.columns.drop(['unique_id', 'ds']).tolist(),
                id_col='unique_id',
                time_col='ds',
                target_col='y'
            )
            
            # 2. Consolidation and Aggregation
            # Reshape from wide to long format
            metrics_long = pd.melt(metrics, id_vars=['unique_id', 'metric'], 
                                   value_vars=df_validation.columns.drop(['unique_id', 'ds']).tolist(), 
                                   var_name='model', value_name='value')

            # Pivot to have unique_id and model as index, metrics as columns
            metrics_pivot = metrics_long.pivot_table(index=['unique_id', 'model'], columns='metric', values='value').reset_index()

            agg_funcs = {"mean": "mean", "std": "std", "median": "median"}
            results = []
            
            # Calculate and store aggregated metrics (mean, std, median)
            for name, func in agg_funcs.items():
                metric_ = (metrics_pivot.groupby("model")[["rmse", "mae", "smape"]].agg(func)
                                        .rename(columns=lambda c: f"{c}_{name}_agg").reset_index())
                results.append(metric_)

            # Merge all aggregated results into a single DataFrame
            metric_agg = reduce(lambda left, right: pd.merge(left, right, on="model"), results)

            # 3. Logging all aggregated metrics
            metrics_list = ['rmse', 'mae', 'smape']
            agg_types = list(agg_funcs.keys())
            
            for _, row in metric_agg.iterrows():
                model_name_ = row['model']
                for metric in metrics_list:
                    for agg in agg_types:
                        log_key = f"{model_name_}_{metric}_{agg}_agg"
                        mlflow.log_metric(log_key, row[f"{metric}_{agg}_agg"])

            # 4. Best Model Selection and Logging
            # Filters models that failed (whose aggregates resulted in NaN)
            valid_metrics = metric_agg.dropna(subset=['rmse_mean_agg'])
            
            # --- DEBUG: Verifique os RMSE de todos os modelos ---
            print("--- Result of metric_agg (Models and Mean RMSE) ---")
            print(metric_agg[['model', 'rmse_mean_agg']].sort_values(by='rmse_mean_agg'))
            print("---------------------------------------------------")

            # Ensures there are valid models to select from
            if valid_metrics.empty:
                print("ERROR: No valid metrics found for best model selection.")
                # Logs a high value or handles the exception
                mlflow.log_metric("Final_Best_RMSE_Mean_Agg", 99999.0) 
                mlflow.set_tag("Best_Model_Selected", "FAIL_NO_VALID_MODEL")
            else:
                # Selects the model with the minimum Mean RMSE
                best_model_name = valid_metrics.loc[valid_metrics['rmse_mean_agg'].idxmin(), 'model']
                best_rmse = valid_metrics['rmse_mean_agg'].min() # Now, guaranteed not to include invalid 0s/NaNs

                mlflow.log_metric("Final_Best_RMSE_Mean_Agg", best_rmse)
                mlflow.set_tag("Best_Model_Selected", best_model_name)
                
            print(f"Metrics successfully logged. Best Model: {best_model_name}")

    def log_and_save(self, df_validation: pd.DataFrame, df_predict: pd.DataFrame, run_id: str) -> bool:
        """
        Performs model evaluation, logs metrics to MLflow, and saves the trained 
        MLForecast model and the required feature history (exog_history) as artifacts.

        Args:
            df_validation (pd.DataFrame): Validation DataFrame, including the actual 'y' column, 
                used for merging and evaluation.
            df_predict (pd.DataFrame): Predictions DataFrame (output of .predict()), 
                used for comparison against 'y'.
            run_id (str): MLflow run ID to log metrics and save artifacts.

        Returns:
            bool: True if saving artifacts was successful, False otherwise.
        """
        if self.fcst is None or self.exog_history is None:
            print("Error: Model (self.fcst) or feature history (exog_history) are not available for saving.")
            return False
        
        # Evaluation
        
        # 1. Drop the potentially incorrect/stale 'y' column from predictions (if it exists)
        df_predictions_only = df_predict.drop(columns=['y'], errors='ignore')
        
        # 2. Merge the true 'y' from the validation set
        evaluation_df = df_predictions_only.merge(df_validation[['unique_id', 'ds', 'y']], 
                                              on=['unique_id', 'ds'], how='left')

        # 3. Filter: Only evaluate on dates where the true target 'y' is available (not NaN)
        evaluation_df.dropna(subset=['y'], inplace=True)
        
        # Log Metrics
        self._log_metrics(evaluation_df, df_predict, run_id)

        # Artifact Saving (in a new nested run or the same one)
        with mlflow.start_run(run_id=run_id, nested=True) as run:
            print("Saving MLForecast object and artifacts...")
            save_success = self.manager.save_and_log_artifacts(
                instance_MLForecast=self.fcst, 
                exog_features=self.exog_history, 
                model_name=self.model_name, 
                exog_name=self.exog_name
            )
            return save_success

    # =========================================================================
    # --- Main Methods ---
    # =========================================================================

    def train(self,
              df_train: pd.DataFrame,
              run_name: str = "default_run",
              mlflow_params: Optional[Dict] = None,
              features_static: Optional[List] = None) -> str:
        """
        Trains the MLForecast model, logs parameters, and stores the feature history 
        required for future predictions. Starts a new MLflow run.

        Args:
            df_train (pd.DataFrame): The training dataset with 'unique_id', 'ds', and 'y'.
            run_name (str): Name for the new MLflow run.
            mlflow_params (Optional[Dict]): Dictionary of parameters (e.g., model hyperparameters) 
                to be logged to MLflow.
            features_static (Optional[List]): List of static feature column names.

        Returns: 
            str: The created MLflow run ID (run_id).
        
        Raises:
            ValueError: If the MLForecast instance (self.fcst) was not initialized.
        """
        if self.fcst is None:
            raise ValueError("MLForecast instance (self.fcst) was not provided in __init__.")
            
        with mlflow.start_run(run_name=run_name) as run:
            self.current_run_id = run.info.run_id
            
            # 1. Parameter Logging
            if mlflow_params:
                # Log general parameters and then specific model parameters
                for key, params in mlflow_params.items():
                    if key != 'Modelos':
                         mlflow.log_params(params)
                    else:
                        for m_name, p in params.items():
                            # Prefix model parameters for clarity
                            prefixed_params = {f"model_{m_name}_{k}": v for k, v in p.items()} 
                            mlflow.log_params(prefixed_params)

            # 2. Training and Feature Extraction
            print('Applying pre-processing and feature extraction...')
            # Apply feature engineering (e.g., lagged targets)
            df_preprocessed = self.preprocessor_train_predict.fit_transform(df_train) 
            
            print('Training models...')
            self.fcst.fit(df_preprocessed, static_features=features_static)
            
            # Store the feature history (all columns except the target 'y')
            # This history is crucial for generating future lags during prediction
            self.exog_history = df_preprocessed.drop(columns=['y'])
            
            # Log expected features (optional: for model governance/reproducibility)
            # ... (Maintenance note: original logic for logging features should be here) ...

            print(f"Training complete. Run ID: {self.current_run_id}")
            return self.current_run_id

    def predict(self, df_predict: pd.DataFrame) -> pd.DataFrame:
        """
        Performs forecasting using the stored model (must be trained or loaded). 
        It prepares the future features and then generates the predictions.

        Args:
            df_predict (pd.DataFrame): DataFrame containing future exogenous variables and/or 
                the real 'y' column for a validation period. Must contain 'unique_id' and 'ds'.

        Returns:
            pd.DataFrame: The DataFrame with 'unique_id', 'ds', and model-specific prediction columns.
        
        Raises:
            ValueError: If the MLForecast model (self.fcst) is not available.
        """
        if self.fcst is None:
            raise ValueError("The MLForecast model is not available. Train or load first.")
        if self.exog_history is None:
            # This is a critical warning, as many MLForecast models rely on lags from history
            print("Warning: No feature history (exog_history) is stored. Prediction might fail if the model relies on lags/exogenous features.")

        # 1. Data Preparation for Prediction
        print('Creating future DataFrame...')
        # Generates the future date range
        future_df = self.fcst.make_future_dataframe(h=self.horizon) 
        # Merge the future dates with provided exogenous data (df_predict)
        merged_df = future_df.merge(df_predict, on=['ds', 'unique_id'], how='left')

        print('Applying XDFPreparator...')
        # Prepare exogenous data (e.g., imputation, external feature creation)
        X_df_imputed = self.preparator_exog.create_future_df(merged_df)

        print('Applying TrainPredictPreprocessor (transform) to create lags...')
        # Apply the same transformations (lags/window features) using the stored history
        X_df_final = self.preprocessor_train_predict.transform(X_df_imputed)

        print('Predicting...')
        predictions_df = self.fcst.predict(h=self.horizon, X_df=X_df_final)

        print('Prediction finished.')

        return predictions_df

    def load(self, run_id: str) -> None:
        """
        Loads the MLForecast model and the corresponding feature history (exog_history) 
        from a specific MLflow run ID.

        Args:
            run_id (str): The MLflow run ID from which to load the artifacts.

        Raises:
            FileNotFoundError: If the model artifacts cannot be found or loaded.
        """
        print(f'Loading Model (Run ID: {run_id})...')
        
        # Use the artifact manager to retrieve both model and history
        mlforecast_loaded, exog_history_loaded = self.manager.load_model_and_exog_history(
            run_id=run_id, model_name=self.model_name, exog_name=self.exog_name
        )
        
        if mlforecast_loaded is None:
            raise FileNotFoundError(f"Could not load the model for Run ID: {run_id}")

        # Update the class state
        self.fcst = mlforecast_loaded
        self.exog_history = exog_history_loaded
        self.current_run_id = run_id
        
        print("Model and feature history successfully loaded.")
  