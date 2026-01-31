"""
Unit Tests - Test individual components in isolation
"""
import pytest
import sys
import os
import sys
import os

# Add project root to Python path
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.insert(0, project_root)

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# from preprocessing import FeaturePreprocessor
from src.preprocessing import FeaturePreprocessor
def test_preprocessing_output_shape():
    """Test that preprocessing returns correct shape"""
    preprocessor = FeaturePreprocessor('models/label_encoders.pkl')
    
    test_data = {
        "student_id": 12345,
        "uni_name": "Test University",
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
    
    X = preprocessor.preprocess(test_data)
    
    # Assert correct shape (1 sample, 12 features)
    assert X.shape == (1, 12), f"Expected shape (1, 12), got {X.shape}"
    print("✅ Unit test passed: Preprocessing output shape correct")

def test_unknown_category_handling():
    """Test that unknown categories don't crash the system"""
    preprocessor = FeaturePreprocessor('models/label_encoders.pkl')
    
    test_data = {
        "student_id": 99999,
        "uni_name": "Unknown University XYZ",
        "major": "Unknown Major",
        "disability": False,
        "dob": "2030-01-01",
        "academic_year": 3,
        "study_hours": 5.0,
        "athleticstatus": "Unknown Status",
        "countryoforigin": "Unknown Country",
        "countryofresidence": "Unknown Country",
        "dropout": False
    }
    
    # Should not raise exception
    X = preprocessor.preprocess(test_data)
    
    assert X is not None, "Preprocessing failed with unknown values"
    assert X.shape[1] == 12, "Feature count mismatch with unknown values"
    print("✅ Unit test passed: Unknown categories handled gracefully")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])