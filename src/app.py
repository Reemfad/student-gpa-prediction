from flask import Flask, request, jsonify
import mlflow
import dagshub
import numpy as np
from src.preprocessing import FeaturePreprocessor
import os

app = Flask(__name__)

# Initialize DagsHub
# gpa_predictorP
# dagshub.init(repo_owner='reemfad51', 
#              repo_name='student-gpa-prediction', 
#              mlflow=True)
mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
print("connect railway")
# Global variables
model = None
preprocessor = None
model_version = None

def load_model_from_mlflow():
    """Load the latest model from MLflow registry"""
    global model, model_version
    
    try:
        # Get the latest version of the registered model
        client = mlflow.tracking.MlflowClient()
        model_name = "gpa_predictor"
        
        # Get latest version
        latest_versions = client.get_latest_versions(model_name, stages=["None"])
        
        if not latest_versions:
            raise Exception("No model versions found")
        
        latest_version = latest_versions[0].version
        model_uri = f"models:/{model_name}/{latest_version}"
        
        # Load model
        model = mlflow.sklearn.load_model(model_uri)
        model_version = latest_version
        
        print(f"✅ Loaded model version {model_version}")
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

# @app._got_first_request
def initialize():
    """Initialize model and preprocessor on startup"""
    global preprocessor
    
    print("🚀 Initializing ML Container...")
    
    # Load preprocessor
    preprocessor = FeaturePreprocessor('models/label_encoders.pkl')
    print("✅ Preprocessor loaded")
    print("and all is successful")
    print("ready to load the model")
    
    # Load model
    if load_model_from_mlflow():
        print("✅ ML Container ready!")
    else:
        print("⚠️ ML Container started but model loading failed")
initialize()

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_version': model_version
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    global model, preprocessor
    
    # Check if model is loaded
    if model is None:
        return jsonify({
            'error': 'Model not loaded'
        }), 503
    
    try:
        # Get input data
        backend_data = request.get_json()
        
        if not backend_data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Preprocess
        X = preprocessor.preprocess(backend_data)
        
        # Predict
        prediction = model.predict(X)[0]
        
        # Prepare response
        response = {
            'student_id': backend_data.get('student_id', None),
            'predicted_gpa': round(float(prediction), 2),
            'model_version': model_version,
            'features_used': X.shape[1]
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/', methods=['GET'])
def root():
    """Root endpoint"""
    return jsonify({
        'service': 'GPA Prediction ML Container',
        'status': 'online',
        'version': '1.0.0',
        'endpoints': {
            'health': '/health',
            'predict': '/predict (POST)'
        }
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
