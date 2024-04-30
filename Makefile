start_mlflow_server:
	mlflow server

setup_zenml:
	zenml   up --blocking

install_requirements:
	-pip install -r requirements.txt
	-zenml integration install mlflow -y
	-zenml integration install sklearn

register_experiment_tracker:
	-zenml   experiment-tracker register mlflow_experiment_tracker --flavor mlflow --tracking_uri=http://localhost:5000 --tracking_token=123
	-zenml   stack register mlflow_stack -e mlflow_experiment_tracker -a default -o default
	-zenml   stack set mlflow_stack

clean_zenml:
	zenml clean -y

execute_pipelines:
	-python zenml_cat_app.py
	-python zenml_xgb_app.py
