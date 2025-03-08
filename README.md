# House Price Prediction - Machine Learning Deployment

## 📌 Objective
This project aims to develop and deploy a machine learning model that predicts house prices based on various attributes. The solution involves data preprocessing, model training, hyperparameter tuning, and deployment as a REST API.

---

## 📂 Project Structure
```bash
├── house_data.csv/               # Dataset (CSV files, if applicable)
├── model.pkl              # Trained models and scalers
│   app.py           # API script
├── house.py             # Model Training and performance report
├── README.md            # Project documentation
```

---

## 📊 Part 1: Data Preprocessing

1. **Load Dataset**: The dataset is sourced from [Kaggle's House Price Prediction Dataset](https://www.kaggle.com) or the California Housing Dataset.
2. **Exploratory Data Analysis (EDA)**:
   - Checked for missing values, outliers, and distributions.
   - Visualized feature correlations using heatmaps.
3. **Feature Engineering**:
   - Handled missing values appropriately.
   - Encoded categorical variables.
   - Scaled numerical features using `StandardScaler`.
   - Selected relevant features based on correlation analysis.

---

## 🤖 Part 2: Model Training & Evaluation

1. **Data Splitting**: The dataset was split into training and testing sets using an 80-20 ratio.
2. **Model Selection**:
   - Linear Regression
   - Decision Tree
   - Random Forest
   - XGBoost (Selected based on performance)
3. **Model Optimization**:
   - Used `GridSearchCV` for hyperparameter tuning.
   - Evaluated using RMSE, MAE, and R² scores.
4. **Best Model Performance**:
   - RMSE: 315345.60953437665
   - MAE: 197067.73426215802
   - R² Score: 197067.73426215802

---

## 🌐 Part 3: Model Deployment

1. **Flask/FastAPI Application**:
   - Developed a REST API with an endpoint `/predict`.
   - Accepts JSON input with house features.
   - Returns the predicted house price.
2. **Testing the API**:
   - Used Postman and CURL for request validation.
3. **Containerization (Optional)**:
   - Dockerized the application for easy deployment.

---

## 🚀 Usage Guide

### 1️⃣ Setup Environment
```bash
pip install -r requirements.txt
```

### 2️⃣ Run API
```bash
python app.py
```

### 3️⃣ Test API using CURL
```bash
curl -X POST http://127.0.0.1:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"bedrooms": 3, "bathrooms": 2, "floors": 1, "yr_built": 2000}'
```

---

## 📜 Report & Documentation
- **Steps Taken**: Data preprocessing, model selection, optimization, and deployment strategies.
- **Authors**: Mohesh T

---

## 🛠️ Future Improvements
- Enhance model performance with additional features.
- Deploy the API on cloud services (AWS/GCP/Heroku).
- Implement a front-end UI for user interaction.

---

## 🏆 Acknowledgments
- [Scikit-learn](https://scikit-learn.org/)
- [Flask](https://flask.palletsprojects.com/)
- [FastAPI](https://fastapi.tiangolo.com/)

---

⭐ _If you find this project helpful, feel free to give it a star on GitHub!_
