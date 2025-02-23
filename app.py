import joblib
import numpy as np
import pandas as pd
import datetime
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import MinMaxScaler
app = Flask(__name__)

# Load the trained model
model = joblib.load("./models/extratrees_model.pkl")



# CSV file for tracking model performance
LOG_FILE = "model_performance_log.csv"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input data from form
        raw_data = np.array([[float(request.form[key]) for key in ["GDP per Capita", "Social Support", "Life Expectancy", "Freedom", "Corruption"]]])

        # Make prediction
        prediction = model.predict(raw_data)[0]

        # Log the input data and prediction
        log_prediction(raw_data, prediction)

        return jsonify({"prediction": float(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)})

def log_prediction(raw_data, prediction):
    """Log raw input and model predictions for monitoring."""
    log_entry = {
        "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Raw GDP per Capita": raw_data[0][0],
        "Raw Social Support": raw_data[0][1],
        "Raw Life Expectancy": raw_data[0][2],
        "Raw Freedom": raw_data[0][3],
        "Raw Corruption": raw_data[0][4],
        "Prediction": prediction
    }

    # Convert to DataFrame and append to CSV
    df = pd.DataFrame([log_entry])
    df.to_csv(LOG_FILE, mode="a", index=False, header=not pd.io.common.file_exists(LOG_FILE))

if __name__ == "__main__":
    app.run(debug=True)
