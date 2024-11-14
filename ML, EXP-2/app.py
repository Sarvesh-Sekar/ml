from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import joblib

app = Flask(__name__)

def load_data():
    # Simulated data loading
    X = np.random.rand(100, 4096)
    y = np.random.randint(2, size=100)
    return X, y

def train_model():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, y_train)
    joblib.dump(model, 'plant_classifier.pkl')  # Save the model
    return model

def load_model():
    return joblib.load('plant_classifier.pkl')  # Load the saved model

model = train_model()  # Initial model training; use load_model() on restart

def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64))
    features = resized.flatten() / 255.0
    return features.reshape(1, -1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    try:
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({"error": "Invalid image format"}), 400

        features = extract_features(image)
        prediction = model.predict(features)
        prediction_prob = model.predict_proba(features)
        result = {
            'prediction': int(prediction[0]),
            'confidence': float(np.max(prediction_prob))
        }
        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
