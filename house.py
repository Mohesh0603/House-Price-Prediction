import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('house_data.csv')

# Exploratory Data Analysis (EDA)
print("First 5 rows of the dataset:")
print(df.head())

print("\nSummary statistics of the dataset:")
print(df.describe())

print("\nInformation about the dataset:")
print(df.info())

# Visualize distributions of features
df.hist(bins=50, figsize=(20, 15))
plt.show()

# Visualize correlations
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()

# Handle missing values
df = df.dropna()

# Convert 'yr_built' to numeric, handle non-numeric values
df['yr_built'] = pd.to_numeric(df['yr_built'], errors='coerce')

# Drop rows with non-numeric 'yr_built' values
df = df.dropna(subset=['yr_built'])

# Feature engineering (example: scaling, encoding)
X = df[['bedrooms', 'bathrooms', 'floors', 'yr_built']]
y = df['price']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a regression model
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

# Evaluate the model
y_pred = lr.predict(X_test_scaled)
rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
print(f"RMSE: {rmse}")
print(f"MAE: {mean_absolute_error(y_test, y_pred)}")
print(f"R²: {r2_score(y_test, y_pred)}")

# Optimize the model using GridSearchCV
param_grid = {'fit_intercept': [True, False], 'copy_X': [True, False], 'n_jobs': [None, -1]}
grid_search = GridSearchCV(LinearRegression(), param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)
best_model = grid_search.best_estimator_

# Save the trained model and scaler using Pickle
pickle.dump(best_model, open('model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))
