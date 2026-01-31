import pickle
import pandas as pd
import numpy as np
from pathlib import Path

class FeaturePreprocessor:
    """Handle feature engineering and encoding for predictions"""
    
    def __init__(self, encoders_path='models/label_encoders.pkl'):
        """Load saved label encoders"""
        with open(encoders_path, 'rb') as f:
            self.encoders = pickle.load(f)
        
        print(f"✅ Loaded {len(self.encoders)} label encoders")
    
    def prepare_features(self, backend_data):
        """
        Convert backend data format to model input format
        
        Args:
            backend_data: dict from backend API
            
        Returns:
            pd.DataFrame ready for model prediction
        """
        # Create full feature dict with defaults
        features = {
        'academicyear': backend_data.get('academic_year', 1),
        'athleticstatus': backend_data.get('athleticstatus', 'Inactive'),
        'countryoforigin': backend_data.get('countryoforigin', 'Unknown'),
        'countryofresidence': backend_data.get('countryofresidence', 'Unknown'),
        'disability': backend_data.get('disability', 'None'),
        'dob': backend_data.get('dob', '2000-01-01'),
        'gender': 'Unknown',  # Backend doesn't have this yet
        'major': backend_data.get('major', 'Computer Science'),
        'primarylanguage': 'English',  # Backend doesn't have this yet
        'university': backend_data.get('uni_name', ''),
        'dropout': backend_data.get('dropout', 0),
        'study_hours': backend_data.get('study_hours', 0.0)
    }
        
        # Convert disability boolean to string
        if isinstance(backend_data.get('disability'), bool):
            features['disability'] = 'None' if not backend_data['disability'] else 'Unknown'
        
        # Map academic_year integer to category
        year_mapping = {1: 'freshman', 2: 'sophomore', 3: 'junior', 4: 'senior', 5: 'graduate', 6: 'phd'}
        if isinstance(features['academicyear'], int):
            features['academicyear'] = year_mapping.get(features['academicyear'], 'freshman')
        
        # Create DataFrame
        df = pd.DataFrame([features])
        
        return df
    
    def encode_features(self, df):
        """
        Apply label encoding with unknown value handling
        
        Args:
            df: DataFrame with raw features
            
        Returns:
            DataFrame with encoded features
        """
        df_encoded = df.copy()
        
        for col, encoder in self.encoders.items():
            if col not in df_encoded.columns:
                continue
                
            # Handle unseen categories
            def safe_encode(value):
                try:
                    # Try to encode
                    return encoder.transform([str(value)])[0]
                except ValueError:
                    # Unknown category - use most common class (index 0)
                    print(f"⚠️ Unknown value '{value}' in {col}, using default")
                    return 0
            
            df_encoded[col] = df[col].apply(safe_encode)
        
        # Ensure correct column order (same as training)
        expected_columns = ['academicyear', 'athleticstatus', 'countryoforigin', 
                   'countryofresidence', 'disability', 'dob', 'gender', 
                   'major', 'primarylanguage', 'university', 'dropout', 
                   'study_hours']

        
        # Reorder columns
        df_encoded = df_encoded[expected_columns]
        
        return df_encoded
    
    def preprocess(self, backend_data):
        """
        Complete preprocessing pipeline
        
        Args:
            backend_data: dict from backend API
            
        Returns:
            numpy array ready for model.predict()
        """
        # Step 1: Prepare features
        df = self.prepare_features(backend_data)
        print(f"📋 Prepared features: {df.shape}")
        
        # Step 2: Encode categorical features
        df_encoded = self.encode_features(df)
        print(f"🔢 Encoded features: {df_encoded.shape}")
        
        # Step 3: Convert to numpy array
        X = df_encoded.values
        
        return X