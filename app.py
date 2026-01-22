from flask import Flask, request, jsonify
from flask_cors import CORS
from vegetation import FieldHealthAnalyzer as VegetationAnalyzer
from crop_model import CropRecommendationModel
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
MODEL_PATH = 'models/crop_model.pkl'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Initialize analyzers
vegetation_analyzer = VegetationAnalyzer()
crop_model = CropRecommendationModel(MODEL_PATH)

# Train model if not exists
if crop_model.model is None:
    print("Training crop recommendation model...")
    crop_model.train_new_model()
    crop_model.save_model(MODEL_PATH)

@app.route('/analyze', methods=['POST'])
def analyze_image():
    """
    Enhanced analysis with crop recommendation
    Accepts: multipart/form-data with 'image' file
    Optional parameters: crop_name, location_features
    """
    try:
        # Check if image file is present
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Get optional parameters
        crop_name = request.form.get('crop_name')
        location_features = request.form.get('location_features', '{}')
        
        # Analyze vegetation
        results = vegetation_analyzer.analyze(file)
        
        # Add crop-specific analysis if crop name is provided
        if crop_name:
            results['crop_specific'] = {
                'crop_name': crop_name,
                'analysis': vegetation_analyzer.get_crop_info(crop_name)
            }
        
        # Add crop recommendation if location features provided
        try:
            import json
            features = json.loads(location_features)
            if features:
                crop_rec = crop_model.recommend_crop(features)
                results['crop_recommendation'] = crop_rec
        except:
            pass  # Skip if no valid features
        
        # Add metadata
        results['metadata'] = {
            'filename': file.filename,
            'crop_name': crop_name,
            'processed_at': datetime.now().isoformat()
        }
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'error': str(e), 'details': 'Image processing failed'}), 500

@app.route('/recommend-crop', methods=['POST'])
def recommend_crop():
    """Get crop recommendation based on environmental features"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Required features
        required_features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        
        # Check if all features are present
        missing = [f for f in required_features if f not in data]
        if missing:
            return jsonify({
                'error': f'Missing features: {missing}',
                'required': required_features
            }), 400
        
        # Get recommendation
        recommendation = crop_model.recommend_crop(data)
        
        # Add crop information
        crop_info = crop_model.get_crop_info(recommendation['recommended_crop'])
        recommendation['crop_information'] = crop_info
        
        return jsonify(recommendation)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/crops', methods=['GET'])
def list_crops():
    """Get list of all available crops in the model"""
    try:
        crops = crop_model.crop_labels if crop_model.crop_labels else []
        return jsonify({
            'crops': crops,
            'count': len(crops)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/crop-info/<crop_name>', methods=['GET'])
def get_crop_information(crop_name):
    """Get detailed information about a specific crop"""
    try:
        info = crop_model.get_crop_info(crop_name)
        return jsonify({
            'crop': crop_name,
            'information': info
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)