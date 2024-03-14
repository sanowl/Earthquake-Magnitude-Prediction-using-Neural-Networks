import numpy as np
import pandas as pd
import matplotlib .pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt

# Load the data into a pandas DataFrame
data = pd.read_csv('database.csv')

# Select the relevant columns
X = data[['Latitude', 'Longitude', 'Depth', 'Type', 'Magnitude Type']]
y = data['Magnitude']

# Fill missing values with mean for numerical columns
num_imputer = SimpleImputer(strategy='mean')
X['Depth'] = num_imputer.fit_transform(X[['Depth']])

# One-hot encode categorical columns
categorical_cols = ['Type', 'Magnitude Type']
categorical_transformer = OneHotEncoder(handle_unknown='ignore')
ct = ColumnTransformer(transformers=[('encoder', categorical_transformer, categorical_cols)], remainder='passthrough')
X = ct.fit_transform(X)

# Dimensionality reduction using PCA
pca = PCA(n_components=0.95)  # Preserve 95% of the variance
X = pca.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the Gradient Boosting Regression model
model = GradientBoostingRegressor(n_estimators=100, random_state=42, max_features='sqrt', verbose=1)
model.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred = model.predict(X_test)
rmse = sqrt(mean_squared_error(y_test, y_pred))
print(f"Root Mean Squared Error: {rmse:.2f}")

# Plot actual vs predicted values
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Magnitude')
plt.ylabel('Predicted Magnitude')
plt.title('Actual vs Predicted Magnitudes')
plt.show()

# Make predictions on new data
new_data = [[19.246, 145.616, 131.6, 'Earthquake', 'MW']]
new_data = ct.transform(new_data)
new_data = pca.transform(new_data)
new_data = scaler.transform(new_data)
predicted_magnitude = model.predict(new_data)
print(f"Predicted Magnitude: {predicted_magnitude[0]:.2f}")