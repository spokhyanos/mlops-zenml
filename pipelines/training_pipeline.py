from zenml import pipeline
from steps.load_data import load_df
from steps.clean_data import clean_df
from steps.train_model import train_model
from steps.evaluation import evaluate_model



@pipeline()
def train_pipeline(data_dir:str):
    df = load_df(data_dir)

    X_train, X_test, y_train, y_test = clean_df(df)
    model = train_model(X_train, X_test, y_train, y_test)
    r2, rmse = evaluate_model(model, X_test, y_test)
