import optuna
from sklearn.model_selection import train_test_split
import mlflow
import numpy as np
from typing import Tuple
from typing_extensions import Annotated
import pandas as pd
from zenml import pipeline, step
import datetime
import sklearn.metrics as m
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate, StratifiedKFold
from zenml.logger import get_logger
import sagemaker
from catboost import CatBoostClassifier
from sagemaker import image_uris
from sagemaker import image_uris, model_uris, script_uris
import utils
import json
import boto3
from sagemaker.estimator import Estimator


sagemaker_session = sagemaker.Session()
logger = get_logger(__name__)

now = datetime.datetime.now().strftime("%Y-%m-%d")
MODEL_NAME = "Model_catboost_meli_fraud_detection"
EXP_NAME = f"Exp_catboost_meli_fraud-detection_{now}"
ROLE_SAGEMAKER = "arn:aws:iam::324430962407:role/service-role/AmazonSageMaker-ExecutionRole-20240202T053444"
TUNING_N_TRIALS = 50
MIN_ROC_AUC_SCORE = 0.82
VARS = ["a", "b", "c", "d", "e", "f", "g", "h", "k", "l", "m", "n", "o", "p", "monto"]
CAT_VARS = ["g", "o", "p"]


@step(experiment_tracker="mlflow_experiment_tracker")
def load_data() -> pd.DataFrame:
    """
    Load data from csv file.
    Returns:
        pd.DataFrame: Dataframe with the data.
    """
    data = pd.read_csv("dados.csv")
    return data


@step(experiment_tracker="mlflow_experiment_tracker")
def data_preparation(data: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:
    """
    Split data into train and test sets.
    Args:
        data (pd.DataFrame): Dataframe with the data.
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: Train and test sets.
    """

    X = utils.pre_cat(data[VARS])
    y = data["fraude"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    return X_train, X_test, y_train, y_test


@step(experiment_tracker="mlflow_experiment_tracker")
def sagemaker_training_data(
    X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series
) -> Tuple[
    Annotated[str, "sagemaker_training_data_path"],
    Annotated[str, "sagemaker_validation_data_path"],
]:
    """
    Create a Sagemaker training data path.
    Args:
        X_train (pd.DataFrame): Train set.
        y_train (pd.Series): Train labels.
        X_test (pd.DataFrame): Test set.
        y_test (pd.Series): Test labels.
    Returns:
        Tuple[str, str]: Sagemaker training and validation
         data path.

    """
    data_train = pd.concat(
        [pd.Series(y_train, name="Fraude", dtype=int), X_train], axis=1
    )
    data_val = pd.concat([pd.Series(y_test, name="Fraude", dtype=int), X_test], axis=1)

    bucket_name = "bbs-datalake"
    object_name = f"CuratedZone/Sagemaker_data/{MODEL_NAME}/{EXP_NAME}"

    cat_index_list = list(data_train.columns.get_indexer(CAT_VARS))
    # necessarry to catboost training.
    data = {"cat_index_list": [int(i) for i in cat_index_list]}
    json_data = json.dumps(data)
    json_file_name = "categorical_index.json"

    s3 = boto3.client("s3")
    # create and json object in s3
    json_s3_key = f"{object_name}/{json_file_name}"
    s3.put_object(Body=json_data, Bucket=bucket_name, Key=json_s3_key)

    # save data in s3 in a format than sagemaker understands
    path = f"s3://{bucket_name}/{object_name}"
    training_path = f"{path}/"
    validation_path = f"{path}_validation.csv"
    training_path_csv = f"{path}/train.csv"

    data_train.to_csv(training_path_csv, index=False, header=False)
    data_val.to_csv(validation_path, index=False, header=False)

    return training_path, validation_path


@step(enable_cache=False, experiment_tracker="mlflow_experiment_tracker")
def train_cat(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """Training a sklearn RF model.
    Args:
        X_train (pd.DataFrame): Train set.
        y_train (pd.Series): Train labels.
    Returns:
        ClassifierMixin: Trained model.
    """

    default_cat_params = {
        "task_type": "CPU",
        "scale_pos_weight": 1 / (y_train.mean()),
        "eval_metric": "Logloss",
        "cat_features": CAT_VARS,
        "silent": True,
        "iterations": 30,
    }

    def objective(trial):
        """
        Define objective function for optimization.
        Args:
            trial (optuna.Trial): Optuna trial object.
        Returns:
            float: Mean ROC AUC score.
        """
        # Define hyperparameters for optimization
        depth = trial.suggest_int("depth", 1, 16)
        learning_rate = trial.suggest_float("learning_rate", 0.001, 0.3, log=True)
        colsample_bylevel = trial.suggest_float("colsample_bylevel", 0.05, 1.0)
        min_data_in_leaf = trial.suggest_int("min_data_in_leaf", 1, 100)

        params = {
            "depth": depth,
            "learning_rate": learning_rate,
            "colsample_bylevel": colsample_bylevel,
            "min_data_in_leaf": min_data_in_leaf,
        }

        scoring = {
            "roc_auc": "roc_auc",
            "brier_score": make_scorer(m.brier_score_loss),
            "logloss": make_scorer(m.log_loss),
            "f1": "f1",
        }
        # Define cross-validation strategy
        stratified_kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        # Create model
        model = CatBoostClassifier(**params, **default_cat_params, random_state=42)

        cv_results = cross_validate(
            model,
            X_train,
            y_train,
            cv=stratified_kfold,
            scoring=scoring,
            return_train_score=False,
        )
        mean_roc_auc = np.mean(cv_results["test_roc_auc"])
        mean_brier_score = np.mean(cv_results["test_brier_score"])
        mean_logloss = np.mean(cv_results["test_logloss"])
        mean_f1 = np.mean(cv_results["test_f1"])
        mean_inference_time = np.mean(cv_results["score_time"])

        with mlflow.start_run(nested=True):
            # Log hyperparameters and metrics to MLflow
            mlflow.log_params(trial.params)
            mlflow.log_metric("roc_auc_score", mean_roc_auc)
            mlflow.log_metric("brier_score", mean_brier_score)
            mlflow.log_metric("logloss", mean_logloss)
            mlflow.log_metric("f1_score", mean_f1)
            mlflow.log_metric("inference_time", mean_inference_time)
            # mlflow.sklearn.log_model(sk_model=model, artifact_path="model")
        # Return the metric to be optimized
        return mean_roc_auc

    mlflow.set_experiment(EXP_NAME)

    # create a study to maximize using bayesian optimization (Tree-structured Parzen Estimator)
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler()
    )
    study.optimize(objective, n_trials=TUNING_N_TRIALS)
    print("Best trial:")
    best_trial = study.best_trial
    print("  ROC_AUC: ", best_trial.value)
    print("  Params: ")
    for key, value in best_trial.params.items():
        logger.info("    {}: {}".format(key, value))
    # Obtain the runs from the experiment and find the run with the highest accuracy


    params = {**default_cat_params, **best_trial.params}
    return params


@step(experiment_tracker="mlflow_experiment_tracker")
def evaluator(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict,
) -> dict:
    """
    Evaluate model performance.
    Args:
        X_test (pd.DataFrame): Test set.
        y_test (pd.Series): Test labels.
        clf (ClassifierMixin): Trained model.
    Returns:
        dict: Evaluation metrics.
    """
    clf = CatBoostClassifier(**params).fit(X_train, y_train)
    y_pred = clf.predict_proba(X_test)
    eval_dict = utils.evaluate(y_test, y_pred[:, 1])
    logger.info(eval_dict)
    return eval_dict


@step(experiment_tracker="mlflow_experiment_tracker")
def deployment_trigger(eval_dict: dict, min_metric_value: float):
    """
    Trigger for model performance.
    Args:
        eval_dict (dict): Evaluation metrics.
        min_metric_value (float): Minimum metric value.
    Returns:
        bool: True if the metric value is greater than the minimum value, False otherwise.
    """
    logger.info(f'SCORE VALUE {eval_dict["roc_auc_score"]}')
    logger.info(f"MINIMUM SCORE VALUE: {min_metric_value}")
    logger.info(eval_dict["roc_auc_score"] > min_metric_value)
    return eval_dict["roc_auc_score"] > min_metric_value


@step(experiment_tracker="mlflow_experiment_tracker")
def deploy_cat_to_sagemaker(
    params: dict,
    training_path: str,
    validation_path: str,
    deploy_decision: bool,
    instance_type: str = "ml.m4.xlarge",
) -> Annotated[str, "sagemaker_endpoint_name"]:
    """
    Deploy the model to SageMaker.
    Args:
        clf (ClassifierMixin): Trained model.
        training_path (str): Path to the training data.
        deploy_decision (bool): Decision to deploy the model.
        instance_type (str, optional): Instance type for the endpoint. Defaults to "ml.m4.xlarge".
        container_startup_health_check_timeout (int, optional): Timeout for the container startup health check. Defaults to 300.
    Returns:
        str: SageMaker endpoint name.
    """
    # If repo_id and revision are not provided, get them from the model version
    #  Otherwise, use the provided values.
    # Sagemaker
    params.pop("cat_features")
    if deploy_decision:
        # Obtain the runs from the experiment and find the run with the highest accuracy
        runs = mlflow.search_runs(
        experiment_ids=mlflow.get_experiment_by_name(EXP_NAME).experiment_id
        )

        best_run_id = runs.loc[runs["metrics.roc_auc_score"].idxmax()].run_id

        mlflow.register_model(
            model_uri=f"runs:/{best_run_id}/model",
            name=MODEL_NAME,
            tags={"stage": "production"},
        )
        logger.info(f"Model registered with name: {MODEL_NAME}")

        train_model_id, train_model_version, train_scope = (
            "catboost-classification-model",
            "*",
            "training",
        )

        # Retrieve the docker image
        train_image_uri = image_uris.retrieve(
            region=None,
            framework=None,
            model_id=train_model_id,
            model_version=train_model_version,
            image_scope=train_scope,
            instance_type=instance_type,
        )

        # Retrieve the training script
        train_source_uri = script_uris.retrieve(
            model_id=train_model_id,
            model_version=train_model_version,
            script_scope=train_scope,
        )

        train_model_uri = model_uris.retrieve(
            model_id=train_model_id,
            model_version=train_model_version,
            model_scope=train_scope,
        )

        # Create SageMaker Estimator instance
        tabular_estimator = Estimator(
            role=ROLE_SAGEMAKER,
            image_uri=train_image_uri,
            source_dir=train_source_uri,
            model_uri=train_model_uri,
            entry_point="transfer_learning.py",
            instance_count=1,
            instance_type=instance_type,
            max_run=360000,
            hyperparameters=params,
        )

        # Launch a SageMaker Training job by passing the S3 path of the training data
        tabular_estimator.fit(
            {
                "training": training_path,
                "validation": validation_path,
            },
            logs=True,
        )

        deploy_image_uri = image_uris.retrieve(
            region=None,
            framework=None,
            image_scope="inference",
            model_id=train_model_id,
            model_version=train_model_version,
            instance_type=instance_type,
        )
        # Retrieve the inference script uri
        deploy_source_uri = script_uris.retrieve(
            model_id=train_model_id,
            model_version=train_model_version,
            script_scope="inference",
        )
        predictor = tabular_estimator.deploy(
            initial_instance_count=1,
            instance_type=instance_type,
            entry_point="inference.py",
            image_uri=deploy_image_uri,
            source_dir=deploy_source_uri,
        )
        endpoint_name = predictor.endpoint_name

        logger.info(f"Model deployed to SageMaker: {endpoint_name}")
        return endpoint_name
    else:
        logger.warning(
            "Model not deployed to SageMaker. Perfomance lower than the necessary minimum."
        )
        return None


@pipeline()
def training_cat_pipeline():
    data = load_data()
    X_train, X_test, y_train, y_test = data_preparation(data)
    training_path, validation_path = sagemaker_training_data(
        X_train, y_train, X_test, y_test
    )
    params = train_cat(X_train, y_train)
    eval_dict = evaluator(X_test, y_test, X_train, y_train, params)
    deployment_decision = deployment_trigger(eval_dict, MIN_ROC_AUC_SCORE)
    deploy_cat_to_sagemaker(params, training_path, validation_path, deployment_decision)


if __name__ == "__main__":
    training_cat_pipeline()
