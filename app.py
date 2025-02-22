import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Load the trained model and pre-fitted scaler
model = joblib.load("extratrees_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get data from form
        data = np.array([[float(request.form[key]) for key in ["GDP per Capita", "Social Support", "Life Expectancy", "Freedom", "Corruption"]]])

        # Make prediction
        prediction = model.predict(data)[0]

        return jsonify({"prediction": float(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
