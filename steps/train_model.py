import logging
import pandas as pd 
import mlflow
from zenml import step

from src.model import LRModel
from sklearn.base import RegressorMixin
from .config import ModelNameConfig

from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker.name)
def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    config: ModelNameConfig,
    ) -> RegressorMixin:

    """

    Args:
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.DataFrame,
        y_test: pd.DataFrame,
        config: ModelNameConfig

    """

    
    try:
        model = None
        if config.model_name == "LinearRegression":
            mlflow.sklearn.autolog()
            model = LRModel()
            
            trained_model = model.train(X_train, y_train)
            return trained_model # type: ignore
        else:
            raise ValueError("Model {} not supported".format(config.model_name))
    except Exception as ex:
        logging.error("Error in model training: {}".format(ex))
        raise ex
    
