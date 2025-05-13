"""
Date: 12th of May
Author: Beloslava Malakova
Description: general linear regression per LSOA, used for experimenting which features work best.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt


FILE = Path("data/Data Aggregated.csv")
df = pd.read_csv(FILE, parse_dates=['Date'])

# Drop unused columns
df.drop(columns=['Unnamed: 0', 'LSOA code', 'LSOA name', 'Ward code', 'Ward name'], errors='ignore', inplace=True)

df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df.drop(columns=['Date'], inplace=True)

# Select independent variables (features)
features = [
    'LSOA Area Size (HA)',
    'Overall Ranking - IMD',
    'Housing rank',
    'Health rank',
    'Living environment rank',
    'Crime rank',
    'Education rank',
    'Income rank',
    'Employment rank',
    'Year',
    'Month'
]

df_model = df.dropna(subset=features + ['Burglary Count'])
X = df_model[features]
y = df_model['Burglary Count']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit regression model
model = LinearRegression()
model.fit(X_scaled, y)

# Print intercept and coefficients
coefficients = pd.Series(model.coef_, index=X.columns)
print("Intercept:", model.intercept_)
print("Coefficients:\n", coefficients.sort_values(key=abs, ascending=False))

# Predict and evaluate
y_pred = model.predict(X_scaled)
print("\nModel performance:")
print("R²:", r2_score(y, y_pred))
print("MAE:", mean_absolute_error(y, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y, y_pred)))

# plot coefficients
coefficients.sort_values().plot(kind='barh', title='Impact on Burglary Count')
plt.xlabel("Coefficient Value")
plt.tight_layout()
plt.show()
