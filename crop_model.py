import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as metrics

class CropRecommendationModel:
    def __init__(self, model_path=None):
        if model_path:
            self.load_model(model_path)
        else:
            self.model = None
            self.crop_labels = None
            self.feature_names = None
    
    def load_model(self, model_path):
        """Load pre-trained model"""
        try:
            with open(model_path, 'rb') as file:
                self.model = pickle.load(file)
            print(f"Model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
    
    def train_new_model(self, csv_path='Crop_recommendation.csv'):
        """Train a new model from CSV"""
        df = pd.read_csv(csv_path)
        
        # Prepare features and labels
        features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        X = df[features]
        y = df['label']
        
        # Get unique crops
        self.crop_labels = y.unique().tolist()
        self.feature_names = features
        
        # Split data
        train_X, test_X, train_y, test_y = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(train_X, train_y)
        
        # Evaluate
        prediction = self.model.predict(test_X)
        accuracy = metrics.accuracy_score(prediction, test_y)
        
        print(f"Model trained with accuracy: {accuracy:.2%}")
        print(f"Number of crop classes: {len(self.crop_labels)}")
        
        return accuracy
    
    def save_model(self, model_path='models/crop_model.pkl'):
        """Save trained model"""
        import os
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        with open(model_path, 'wb') as file:
            pickle.dump({
                'model': self.model,
                'crop_labels': self.crop_labels,
                'feature_names': self.feature_names
            }, file)
        print(f"Model saved to {model_path}")
    
    def recommend_crop(self, features_dict):
        """Recommend crop based on environmental features"""
        if self.model is None:
            return {"error": "Model not loaded"}
        
        # Convert dictionary to numpy array
        features_list = []
        for feature in self.feature_names:
            if feature in features_dict:
                features_list.append(features_dict[feature])
            else:
                features_list.append(0)  # Default value
        
        features_array = np.array([features_list])
        
        # Get prediction
        prediction = self.model.predict(features_array)
        probabilities = self.model.predict_proba(features_array)
        
        # Get top 3 recommendations
        top_3_idx = np.argsort(probabilities[0])[-3:][::-1]
        recommendations = []
        
        for idx in top_3_idx:
            crop = self.crop_labels[idx]
            prob = probabilities[0][idx] * 100
            recommendations.append({
                'crop': crop,
                'probability': float(prob),
                'suitability': 'Excellent' if prob > 80 else 
                              'Good' if prob > 60 else 'Moderate'
            })
        
        return {
            'recommended_crop': prediction[0],
            'recommendations': recommendations,
            'input_features': features_dict
        }
    
    def get_crop_info(self, crop_name):
        """Get information about a specific crop"""
        # This can be extended with a database
        crop_info = {
            'rice': {'water_needs': 'High', 'season': 'Kharif/Rabi'},
            'maize': {'water_needs': 'Moderate', 'season': 'Kharif'},
            'chickpea': {'water_needs': 'Low', 'season': 'Rabi'},
            'kidneybeans': {'water_needs': 'Moderate', 'season': 'Kharif'},
            'pigeonpeas': {'water_needs': 'Low', 'season': 'Kharif'},
            'mothbeans': {'water_needs': 'Low', 'season': 'Kharif'},
            'mungbean': {'water_needs': 'Low', 'season': 'Kharif'},
            'blackgram': {'water_needs': 'Moderate', 'season': 'Kharif'},
            'lentil': {'water_needs': 'Low', 'season': 'Rabi'},
            'pomegranate': {'water_needs': 'Moderate', 'season': 'Year-round'},
            'banana': {'water_needs': 'High', 'season': 'Year-round'},
            'mango': {'water_needs': 'Moderate', 'season': 'Summer'},
            'grapes': {'water_needs': 'Moderate', 'season': 'Summer'},
            'watermelon': {'water_needs': 'High', 'season': 'Summer'},
            'muskmelon': {'water_needs': 'Moderate', 'season': 'Summer'},
            'apple': {'water_needs': 'Moderate', 'season': 'Winter'},
            'orange': {'water_needs': 'Moderate', 'season': 'Winter'},
            'papaya': {'water_needs': 'Moderate', 'season': 'Year-round'},
            'coconut': {'water_needs': 'High', 'season': 'Year-round'},
            'cotton': {'water_needs': 'High', 'season': 'Kharif'},
            'jute': {'water_needs': 'High', 'season': 'Kharif'},
            'coffee': {'water_needs': 'High', 'season': 'Year-round'}
        }
        
        return crop_info.get(crop_name.lower(), {
            'water_needs': 'Moderate',
            'season': 'Varies',
            'notes': 'Consult local agriculture department'
        })