import logging

from abc import ABC, abstractmethod
from typing import Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class DataStrategy(ABC):
    
    """
        Template class to handle data 
    """

    @abstractmethod
    def manage_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass


class PreProcessDataStrategy(DataStrategy):

    def manage_data(self, data: pd.DataFrame) -> pd.DataFrame:

        try:
            # this is feature engineering - year and month features are hidden in feature date
            data['date'] = pd.to_datetime(data['date'])
            data['year'] = data['date'].apply(lambda date: date.year)
            data['month'] = data['date'].apply(lambda date: date.month)
            # drop date, id and zipcode fields
            data = data.drop(
                [
                    "date",
                    "zipcode",
                    'id'
                ], axis=1
            )
            return data
        except Exception as ex:
            logging.error("Error data processing and feature engineering: {}".format(ex))
            raise ex
        

class SplitDataStartegy(DataStrategy):
    """
        Base method to split the data

    """

    def manage_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        try:
            X = data.drop('price', axis=1)
            y = data['price']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
            return X_train, X_test, y_train, y_test
        except Exception as ex:
            logging.error("Error in splitting data: {}".format(ex))
            raise ex
        

class DataCleaning:
    """

    """
    def __init__(self, data: pd.DataFrame, strategy: DataStrategy):
        self.data = data
        self.strategy = strategy
        
    def process_data(self) -> Union[pd.DataFrame, pd.Series]:

        try:
            return self.strategy.manage_data(self.data)
        except Exception as ex:
            logging.error("Error in managing data")
            raise ex
        

if __name__ == "__main__":
    data = pd.read_csv("data\kc_house_data.csv")
    data_cleaning = DataCleaning(data, PreProcessDataStrategy())
    data_cleaning.process_data()

