import logging

from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression


class Model(ABC):
    
    @abstractmethod
    def train (self, X_train, y_train):
        pass


class LRModel(Model):

    
    def train(self, X_train, y_train, **kwargs):
        
        try:
            reg_model = LinearRegression(**kwargs)
            reg_model.fit(X_train, y_train)
            logging.info("Model training completed")

            return reg_model
        except Exception as ex:
            logging.error("Error in training model: {}".format(ex))
    

    



