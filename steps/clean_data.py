import logging

import pandas as pd 

from zenml import step
from src.data_cleaning import DataCleaning, PreProcessDataStrategy, SplitDataStartegy
from typing_extensions import Annotated
from typing import Tuple


@step
def clean_df(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train" ],
    Annotated[pd.Series, "y_test"],
    ]:
    try:
        process_strategy = PreProcessDataStrategy()
        data_cleaning = DataCleaning(df, process_strategy)
        processed_data = data_cleaning.process_data()

        split_strategy = SplitDataStartegy()
        data_cleaning = DataCleaning(processed_data, split_strategy)
        X_train, X_test, y_train, y_test = data_cleaning.process_data()

        logging.info("Data processing completed")
        return X_train, X_test, y_train, y_test
    except Exception as ex:
        logging.error("Error processing data: {}".format(ex))
        raise ex
    


