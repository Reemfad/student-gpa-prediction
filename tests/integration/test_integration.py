"""
Integration Tests - Test components working together
"""
import pytest
import sys
import os
import mlflow
import dagshub

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.preprocessing import FeaturePreprocessor

def test_model_loading_from_mlflow():
    """Test that model can be loaded from MLflow registry"""
    dagshub.auth.add_token(
    os.environ["DAGSHUB_TOKEN"],
    host="https://dagshub.com"
    )
    dagshub.init(
    repo_owner="reemfad51",
    repo_name="student-gpa-prediction",
    mlflow=True,
    
    )
    try:
        # Attempt to load model
        client = mlflow.tracking.MlflowClient()
        model_name = "gpa_predictor"
        
        latest_versions = client.get_latest_versions(model_name, stages=["None"])
        
        assert len(latest_versions) > 0, "No model versions found in registry"
        
        latest_version = latest_versions[0].version
        model_uri = f"models:/{model_name}/{latest_version}"
        
        model = mlflow.sklearn.load_model(model_uri)
        
        assert model is not None, "Model loaded but is None"
        print(f"✅ Integration test passed: Model version {latest_version} loaded successfully")
        
    except Exception as e:
        pytest.fail(f"Model loading failed: {str(e)}")

def test_preprocessing_and_prediction_pipeline():
    """Test complete pipeline: preprocess → predict"""
    dagshub.auth.add_token(
    os.environ["DAGSHUB_TOKEN"],
    host="https://dagshub.com"
    )
    dagshub.init(
    repo_owner="reemfad51",
    repo_name="student-gpa-prediction",
    mlflow=True,
    
    )
    # Load preprocessor
    preprocessor = FeaturePreprocessor('models/label_encoders.pkl')
    
    # Load model
    client = mlflow.tracking.MlflowClient()
    model_name = "gpa_predictor"
    latest_versions = client.get_latest_versions(model_name, stages=["None"])
    latest_version = latest_versions[0].version
    model_uri = f"models:/{model_name}/{latest_version}"
    model = mlflow.sklearn.load_model(model_uri)
    
    # Test data
    test_data = {
        "student_id": 12345,
        "uni_name": "American University",
        "major": "Computer Science",
        "disability": False,
        "dob": "2002-05-15",
        "academic_year": 3,
        "study_hours": 7.5,
        "athleticstatus": "Active",
        "countryoforigin": "Iraq",
        "countryofresidence": "Iraq",
        "dropout": False
    }
    
    # Preprocess
    X = preprocessor.preprocess(test_data)
    
    # Predict
    prediction = model.predict(X)[0]
    
    # Validate prediction
    assert prediction is not None, "Prediction is None"
    assert 0.0 <= prediction <= 4.0, f"GPA prediction {prediction} out of valid range (0-4)"
    print(f"✅ Integration test passed: Pipeline produced valid prediction: {prediction:.2f}")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])