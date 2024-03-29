import logging

import mlflow
import pandas as pd
from zenml import step
from sklearn.base import RegressorMixin
from typing import Tuple
from typing_extensions import Annotated

from src.model_evaluation import MSE, RMSE, R2
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker




@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(model: RegressorMixin, 
                   X_test: pd.DataFrame, 
                   y_test: pd.DataFrame
                   ) -> Tuple[
                       Annotated[float, "r2"],
                       Annotated[float, "rmse"],
                   ]:
    """
    Args:
        model: RegressorMixin
        x_test: pd.DataFrame
        y_test: pd.Series
    Returns:
        r2_score: float
        rmse: float
    """

    try:

        prediction = model.predict(X_test)
        mse = MSE().calculate_score(y_test, prediction )
        mlflow.log_metric("mse", mse)

        r2 = R2().calculate_score(y_test, prediction)
        mlflow.log_metric("r2", r2)

        rmse = RMSE().calculate_score(y_test, prediction)
        mlflow.log_metric("rmse", rmse)

        return r2, rmse
    except Exception as ex:
        logging.error("Error in Model evaluation {}".format(ex))
        raise(ex)