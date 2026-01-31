"""
End-to-End Tests - Test complete system with Flask app
"""
import pytest
import sys
import os
import json
import threading
import time
import requests

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.app import app

@pytest.fixture
def client():
    """Create a test client for Flask app"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_health_endpoint(client):
    """Test that health endpoint returns correct status"""
    response = client.get('/health')
    
    assert response.status_code == 200, f"Health check failed with status {response.status_code}"
    
    data = json.loads(response.data)
    
    assert 'status' in data, "Health response missing 'status' field"
    assert data['status'] == 'healthy', f"Health status is {data['status']}, expected 'healthy'"
    assert data['model_loaded'] == True, "Model not loaded"
    
    print(f"✅ E2E test passed: Health endpoint working, model version {data['model_version']}")

def test_prediction_endpoint_full_flow(client):
    """Test complete prediction flow through Flask API"""
    
    test_payload = {
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
    
    response = client.post(
        '/predict',
        data=json.dumps(test_payload),
        content_type='application/json'
    )
    
    assert response.status_code == 200, f"Prediction failed with status {response.status_code}"
    
    data = json.loads(response.data)
    
    # Validate response structure
    assert 'predicted_gpa' in data, "Response missing 'predicted_gpa'"
    assert 'student_id' in data, "Response missing 'student_id'"
    assert 'model_version' in data, "Response missing 'model_version'"
    
    # Validate prediction value
    predicted_gpa = data['predicted_gpa']
    assert 0.0 <= predicted_gpa <= 4.0, f"Invalid GPA prediction: {predicted_gpa}"
    
    print(f"✅ E2E test passed: Full prediction flow working, predicted GPA: {predicted_gpa}")

def test_error_handling_missing_data(client):
    """Test that API handles missing data gracefully"""
    
    # Empty payload
    response = client.post(
        '/predict',
        data=json.dumps({}),
        content_type='application/json'
    )
    
    # Should still work (uses defaults) or return proper error
    assert response.status_code in [200, 400, 500], f"Unexpected status code: {response.status_code}"
    
    data = json.loads(response.data)
    
    # Either successful with defaults or proper error message
    if response.status_code == 200:
        assert 'predicted_gpa' in data
        print("✅ E2E test passed: Empty data handled with defaults")
    else:
        assert 'error' in data
        print("✅ E2E test passed: Empty data rejected with error message")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
