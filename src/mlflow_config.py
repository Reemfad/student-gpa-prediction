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
        # dagshub.init(
        #     repo_owner=os.getenv("reemfad51"),
        #     repo_name=os.getenv("student-gpa-prediction"),
        #     mlflow=True
        # )
        dagshub.init(repo_owner='reemfad51', 
             repo_name='student-gpa-prediction', 
             mlflow=True)
