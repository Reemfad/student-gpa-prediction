import os
import mlflow
import dagshub

def setup_mlflow():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

    if tracking_uri:
        # GitHub Actions, local dev, generic MLflow
        mlflow.set_tracking_uri(tracking_uri)
    else:
        # Railway production (DAGsHub)
        dagshub.init(
            repo_owner=os.getenv("DAGSHUB_REPO_OWNER"),
            repo_name=os.getenv("DAGSHUB_REPO_NAME"),
            mlflow=True
        )
