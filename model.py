# save_model.py
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as metrics

print("=" * 60)
print("SAVING YOUR TRAINED CROP RECOMMENDATION MODEL")
print("=" * 60)

# 1. Load your dataset
df = pd.read_csv("Crop_recommendation.csv")
print(f"✅ Dataset loaded: {len(df)} rows")

# 2. Prepare features and labels (YOUR EXACT CODE)
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']

# 3. Get unique crops
crop_labels = y.unique().tolist()
print(f"🌱 Found {len(crop_labels)} crops:")
for i, crop in enumerate(crop_labels, 1):
    print(f"   {i:2}. {crop}")

# 4. Split the data (YOUR EXACT CODE)
train_X, test_X, train_y, test_y = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"📊 Train set: {len(train_X)} samples")
print(f"📊 Test set: {len(test_X)} samples")

# 5. Train the model (YOUR EXACT CODE)
print("\n🤖 Training Logistic Regression model...")
logi = LogisticRegression(max_iter=1000)
logi.fit(train_X, train_y)

# 6. Test the model
prediction = logi.predict(test_X)
accuracy = metrics.accuracy_score(prediction, test_y)
print(f"✅ Model trained with accuracy: {accuracy:.2%}")

# 7. Save the model to PKL file
model_data = {
    'model': logi,
    'crop_labels': crop_labels,
    'feature_names': ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'],
    'accuracy': accuracy,
    'train_size': len(train_X),
    'test_size': len(test_X)
}

# Create models directory if it doesn't exist
import os
os.makedirs('models', exist_ok=True)

# Save the model
model_path = 'models/crop_model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(model_data, f)

print(f"\n💾 Model saved to: {model_path}")
print(f"📦 File size: {os.path.getsize(model_path)/1024:.1f} KB")

# 8. Test loading the model
print("\n🧪 Testing saved model...")
with open(model_path, 'rb') as f:
    loaded_data = pickle.load(f)
    loaded_model = loaded_data['model']
    
    # Quick test
    test_sample = [[90, 42, 43, 20, 80, 6.5, 200]]
    prediction = loaded_model.predict(test_sample)
    probability = loaded_model.predict_proba(test_sample)
    
    print(f"✅ Model loads correctly!")
    print(f"   Test prediction: {prediction[0]}")
    print(f"   Confidence: {probability[0].max():.1%}")

print("\n" + "=" * 60)
print("🎉 MODEL SAVED SUCCESSFULLY!")
print("Now you can run: python app.py")
print("=" * 60)