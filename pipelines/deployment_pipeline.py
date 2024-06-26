import numpy as np
import pandas as pd

from zenml import pipeline, step
from zenml.config import DockerSettings
# from materializer.custom_materializer import cs_materializer
from steps.clean_data import clean_df
from steps.evaluation import evaluate_model
from steps.load_data import load_df
from steps.train_model import train_model
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW, TENSORFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.steps import BaseParameters, Output

# from .utils import get_data_for_test

docker_settings = DockerSettings(required_integrations=[MLFLOW])

class DeploymentTriggerConfig(BaseParameters):
    min_accuracy: float = 0.60

@step
def deployment_trigger(
    accuracy: float,
    config: DeploymentTriggerConfig,
):
    return accuracy >= config.min_accuracy

class MLFlowDeploymentLoaderStepParameters(BaseParameters):


    pipeline_name: str
    step_name: str
    running: bool = True

@step(enable_cache=False)
def prediction_service_loader(
    pipeline_name: str,
    step_name: str,
    running: bool = True,
    model_name: str = "model",
) -> MLFlowDeploymentService:
    


    # get MLFlow deployer stack componenet 
    mlflow_model_deployer_component = MLFlowModelDeployer.get_active_model_deployer()
    # fetch existing services
    existing_services = mlflow_model_deployer_component.find_model_server(
        pipeline_name=pipeline_name,
        step_name = step_name,
        model_name= model_name,
        running=running,
    )

    if not existing_services:
        raise RuntimeError(
            f"No MLflow prediction service deployed by the "
            f"{pipeline_step_name} step in the {pipeline_name} "
            f"pipeline for the '{model_name}' model is currently "
            f"running."
        )
    print(existing_services)
    print(type(existing_services))
    return existing_services[0]


@pipeline(enable_cache=False, settings={"docker": docker_settings})
def continuous_deployment_pipeline(
    data_dir: str,
    min_accuracy: float = 0.6,
    workers: int = 1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT,

):
    df = load_df(data_dir)
    X_train, X_test, y_train, y_test = clean_df(df)
    model = train_model(X_train, X_test, y_train, y_test)
    r2, rmse = evaluate_model(model, X_test, y_test)
    deployment_decision = deployment_trigger(r2)
    mlflow_model_deployer_step(
        model=model,
        deploy_decision = deployment_decision,
        workers = workers,
        timeout = timeout,
    )

@pipeline(enable_cache=False, settings={"docker": docker_settings})
def inference_pipeline(pipeline_name: str, pipeline_step_name: str):
    # Link all the steps artifacts together
    # batch_data = dynamic_importer()
    # model_deployment_service = prediction_service_loader(
    #     pipeline_name=pipeline_name,
    #     pipeline_step_name=pipeline_step_name,
    #     running=False,
    # )
    print("in Inference pipeline")
