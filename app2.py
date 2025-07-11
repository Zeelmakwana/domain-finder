import numpy as np
from flask import Flask, request, render_template
import pickle

# Load the trained model once
with open('df.pkl', 'rb') as f:
    model = pickle.load(f)

# Setup Flask app
app = Flask(__name__)

# Home page route
@app.route('/')
def home():
    return render_template("index.html")

# Prediction route (renders same page with result)
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get inputs as integers
        features = [int(x) for x in request.form.values()]
        features_arr = np.array(features).reshape(1, -1)
        # Predict using model
        prediction = model.predict(features_arr)
        # Return to index with result
        return render_template('index.html', prediction=prediction[0])
    except Exception as e:
        # Return to index with error message
        return render_template('index.html', prediction=f"Error: {str(e)}")

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
