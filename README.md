# Happiness Score Prediction

## Overview
This project analyzes and predicts global happiness scores based on socio-economic indicators. It involves **data preprocessing, exploratory data analysis (EDA), feature selection, model training, evaluation, and interpretability analysis (SHAP, LIME)** to uncover key insights. A **web application** has been deployed to allow real-time happiness score predictions.

## Web Application
The web application allows users to input key socio-economic factors and predict happiness scores. It is accessible at:

**[Happiness Score Predictor](https://predictor-happiness-score.onrender.com/)**

---

## Features
- **Data-Driven Prediction:** Multiple machine learning models, including Extra Trees, XGBoost, LightGBM, and SVR, predict happiness scores.
- **High Accuracy:** Advanced feature selection and hyperparameter tuning optimize model performance.
- **User-Friendly Deployment:** A web interface built with **Flask** allows real-time predictions.
- **Explainable AI:** SHAP analysis provides insights into how each feature impacts happiness scores.

---

## Dataset
The dataset contains World Happiness Reports from **2015 to 2019**, featuring:
- **GDP per Capita**
- **Social Support**
- **Life Expectancy**
- **Freedom**
- **Corruption**
- **Generosity**

These socio-economic indicators help predict the happiness score for different countries.

---

## Project Structure
1. **Data Preprocessing & Feature Engineering**: Cleaning, handling missing values, and normalizing data.
2. **Exploratory Data Analysis (EDA)**: Visualizing trends using histograms, scatter plots, and correlation heatmaps.
3. **Feature Selection**: Identifying the most influential factors for happiness.
4. **Model Training & Optimization**: Training multiple regression models and selecting the best-performing one.
5. **Model Interpretability (SHAP)**: Understanding feature contributions to model predictions.
6. **Deployment**: Building a web interface using **Flask** for real-time happiness score predictions.

---

## Exploratory Data Analysis (EDA)
### Histograms and Boxplots
Histograms were used to analyze the distribution of features, while boxplots identified outliers.

### Correlation Heatmap
A correlation matrix was generated to analyze relationships between happiness scores and predictors.

### Happiness Trends Over Time
Line plots were used to track happiness scores over **five years**.

---

## Model Training and Evaluation
### Train-Test Split
- **80% Training Data**
- **20% Testing Data**

### Models Trained and Performance Metrics
| Model                 | MAE   | MSE   | RMSE  | R² Score |
|-----------------------|-------|-------|-------|----------|
| Linear Regression     | 0.4521 | 0.3432 | 0.5858 | 0.7178  |
| Ridge Regression      | 0.4526 | 0.3441 | 0.5866 | 0.7170  |
| Lasso Regression      | 0.4521 | 0.3432 | 0.5858 | 0.7178  |
| Random Forest        | 0.4005 | 0.2707 | 0.5203 | 0.7774  |
| XGBoost              | 0.4214 | 0.2758 | 0.5251 | 0.7733  |
| Gradient Boosting    | 0.4133 | 0.2731 | 0.5226 | 0.7754  |
| LightGBM            | 0.4380 | 0.2935 | 0.5417 | 0.7587  |
| SVR                 | 0.3956 | 0.2510 | 0.5010 | 0.7937  |
| Extra Trees         | 0.3914 | 0.2502 | 0.5002 | 0.7943  |
| Bagging Regressor   | 0.4011 | 0.2696 | 0.5192 | 0.7783  |
| Bayesian Ridge      | 0.4524 | 0.3437 | 0.5862 | 0.7175  |

### Best Performing Models
- **Extra Trees Regressor**
- **Support Vector Regressor (SVR)**

---

## SHAP Analysis (Feature Importance)
Feature importance was analyzed using **SHAP** to interpret model predictions.

### Observations
- **GDP per Capita** has the most influence on happiness scores.
- **Life Expectancy and Freedom** significantly contribute to happiness levels.
- **Corruption** has a negative impact on happiness.

---

## How to Run This Project

### Clone the Repository
```bash
git clone https://github.com/Azrael7667/Happiness_Score_ML.git
cd Happiness_Score_ML
```

### Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Deploy Web Application
```bash
python app.py
```
Then open [http://127.0.0.1:5000/](http://127.0.0.1:5000/) in your browser.

---

## Model Deployment
The Happiness Score Predictor is deployed using Render, allowing real-time predictions.

### Deployment Stack
- **Backend:** Flask
- **Frontend:** HTML/CSS
- **Machine Learning:** Scikit-Learn, XGBoost, LightGBM, SVR
- **Model Serialization:** Joblib
- **Explainability:** SHAP, LIME

---

## View Report and Kaggle Notebook
- **Project Report:** [Link to Report](https://github.com/Azrael7667/Happiness_Score_ML/tree/master/Report)
- **Kaggle Notebook:** [View on Kaggle](https://www.kaggle.com/code/silwalsolomon/happiness-score-data-prediction/edit)

---

## Future Improvements
- Add more socio-economic features such as education levels and unemployment rates.
- Experiment with deep learning models for improved accuracy.
- Enhance interpretability with LIME and additional explainability methods.
- Expand the dataset to include more recent happiness reports.

---
