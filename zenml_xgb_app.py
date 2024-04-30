import optuna
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import mlflow
import numpy as np
from typing import Tuple
from typing_extensions import Annotated
import pandas as pd
from zenml import pipeline, step
from sklearn.base import ClassifierMixin
import datetime
import sklearn.metrics as m
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate, StratifiedKFold
from zenml.logger import get_logger
import sagemaker
from sagemaker.inputs import TrainingInput
from sagemaker.serializers import CSVSerializer
import utils


sagemaker_session = sagemaker.Session()
logger = get_logger(__name__)

now = datetime.datetime.now().strftime("%Y-%m-%d")
MODEL_NAME = "Model_xgboost_meli_fraud_detection"
EXP_NAME = f"Exp_xgboost_meli_fraud-detection_{now}"
ROLE_SAGEMAKER = "arn:aws:iam::324430962407:role/service-role/AmazonSageMaker-ExecutionRole-20240202T053444"
TUNING_N_TRIALS = 50
MIN_ROC_AUC_SCORE = 0.82
VARS = ["a", "b", "c", "d", "e", "f", "g", "h", "k", "l", "m", "n", "o", "p", "monto"]


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

    X = utils.pre_xgb(data[VARS])
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
        str: Sagemaker training data path.

    """
    data_train = utils.encode_categorical_to_integers(
        pd.concat([pd.Series(y_train, name="Fraude", dtype=int), X_train], axis=1)
    )
    data_val = utils.encode_categorical_to_integers(
        pd.concat([pd.Series(y_test, name="Fraude", dtype=int), X_test], axis=1)
    )

    bucket_name = "bbs-datalake"
    object_name = f"CuratedZone/Sagemaker_data/{MODEL_NAME}/{EXP_NAME}"

    # save data in s3 in a format than sagemaker understands
    training_path = f"s3://{bucket_name}/{object_name}.csv"
    validation_path = f"s3://{bucket_name}/{object_name}_validation.csv"

    data_train.to_csv(training_path, index=False, header=False)
    data_val.to_csv(validation_path, index=False, header=False)

    return training_path, validation_path


@step(enable_cache=False, experiment_tracker="mlflow_experiment_tracker")
def train_xgb(X_train: pd.DataFrame, y_train: pd.Series) -> ClassifierMixin:
    """Training a sklearn RF model.
    Args:
        X_train (pd.DataFrame): Train set.
        y_train (pd.Series): Train labels.
    Returns:
        ClassifierMixin: Trained model.
    """

    default_xgb_params = {
        "n_jobs": -1,
        "scale_pos_weight": 1 / (y_train.mean()),
        "enable_categorical": True,
        "use_label_encoder": False,
        "tree_method": "hist",
        "eval_metric": "logloss",
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
        max_depth = trial.suggest_int("max_depth", 2, 20)
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
        subsample = trial.suggest_float("subsample", 0.1, 1.0)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.1, 1.0)
        n_estimators = trial.suggest_int("n_estimators", 50, 1000)
        min_child_weight = trial.suggest_float("min_child_weight", 1, 20)
        gamma = trial.suggest_float("gamma", 0.1, 10.0)

        params = {
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "n_estimators": n_estimators,
            "min_child_weight": min_child_weight,
            "gamma": gamma,
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
        model = XGBClassifier(**params, **default_xgb_params, random_state=42)

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

    model = XGBClassifier(**best_trial.params, **default_xgb_params).fit(
        X_train, y_train
    )
    return model



@step(experiment_tracker="mlflow_experiment_tracker")
def evaluator(X_test: pd.DataFrame, y_test: pd.Series, clf: ClassifierMixin) -> dict:
    """
    Evaluate model performance.
    Args:
        X_test (pd.DataFrame): Test set.
        y_test (pd.Series): Test labels.
        clf (ClassifierMixin): Trained model.
    Returns:
        dict: Evaluation metrics.
    """

    y_pred = clf.predict_proba(X_test)
    eval_dict = utils.evaluate(y_test, y_pred[:, 1])
    logger.info(eval_dict)
    return eval_dict


@step(experiment_tracker="mlflow_experiment_tracker")
def deployment_trigger(eval_dict: dict, min_metric_value: float) -> bool:
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
def deploy_xgb_to_sagemaker(
    clf: ClassifierMixin,
    training_path: str,
    validation_path: str,
    deploy_decision: bool,
    instance_type: str = "ml.m4.xlarge",
    container_startup_health_check_timeout: int = 300,
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



        params = {k: v for k, v in clf.get_params().items() if v is not None}
        params["num_round"] = 50
        params.pop("enable_categorical")
        container = sagemaker.image_uris.retrieve("xgboost", sagemaker_session.boto_region_name, "latest")
        xgb = sagemaker.estimator.Estimator(image_uri=container, 
                                            hyperparameters=params,
                                            role=ROLE_SAGEMAKER,
                                            instance_count=1, 
                                            instance_type=instance_type,
                                            sagemaker_session=sagemaker_session)


        content_type = "csv"
        train_input = TrainingInput(training_path, content_type=content_type)
        validation_input = TrainingInput(validation_path, content_type=content_type)
        # Training the model
        xgb.fit({"train": train_input, "validation": validation_input})

        # Implement an endpoint in sagemaker
        predictor = xgb.deploy(
            initial_instance_count=1,
            instance_type="ml.m4.xlarge",
            container_startup_health_check_timeout=container_startup_health_check_timeout,
            serializer=CSVSerializer(),
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
def training_xgb_pipeline():
    data = load_data()
    X_train, X_test, y_train, y_test = data_preparation(data)
    training_path, validation_path = sagemaker_training_data(
        X_train, y_train, X_test, y_test
    )
    clf = train_xgb(X_train, y_train)
    eval_dict = evaluator(X_test, y_test, clf)
    deployment_decision = deployment_trigger(eval_dict, MIN_ROC_AUC_SCORE)
    deploy_xgb_to_sagemaker(clf, training_path, validation_path, deployment_decision)


if __name__ == "__main__":
    training_xgb_pipeline()
