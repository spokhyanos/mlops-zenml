import logging
import numpy as np

from abc import ABC, abstractmethod
from sklearn.metrics import mean_squared_error, r2_score

class ModelEvaluation(ABC):

    """

    """

    @abstractmethod
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):

        pass

class MSE(ModelEvaluation):

    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):

        try:
            logging.info("Calculating MSE")
            mse = mean_squared_error(y_true, y_pred)
            logging.info("MSE: {}".format(mse))
            return mse
        except Exception as ex:
            logging.error("Error in MSE calculation {}".format(ex))
            raise ex

class R2(ModelEvaluation):

    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):

        try:
            logging.info("Calculating R2")
            r2 = r2_score(y_true, y_pred)
            logging.info("MSE: {}".format(r2))
            return r2
        except Exception as ex:
            logging.error("Error in MSE calculation {}".format(ex))
            raise ex

class RMSE(ModelEvaluation):


    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):

        try:
            logging.info("Calculating R2")
            rmse = mean_squared_error(y_true, y_pred, squared=False)
            logging.info("MSE: {}".format(rmse))
            return rmse
        except Exception as ex:
            logging.error("Error in MSE calculation {}".format(ex))
            raise ex