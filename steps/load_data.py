import logging

import pandas as pd
from zenml import step


class LoadData:

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        
    def get_data(self):
        logging.info(f"Loading data from {self.data_dir}")
        return pd.read_csv(self.data_dir)
    
@step
def load_df(data_dir: str) -> pd.DataFrame:
    """
    load the data from the data directory

    Args:
        data_dir: file location of the data files
    """
    try:
        load_data = LoadData(data_dir)
        return load_data.get_data()
    except Exception as ex:
        logging.error(f"Error while loding data: {ex}")
        raise ex
    
        