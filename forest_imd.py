import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# Load and prepare data
df = pd.read_csv("Data Aggregated.csv", parse_dates=['Date'])
df.drop(columns=['Unnamed: 0', 'LSOA name', 'Ward code', 'Ward name'], errors='ignore', inplace=True)
df = df.rename({'Overall Ranking - IMD' : 'IMD'}, axis=1)
df['YearMonth'] = df['Date'].dt.to_period('M')
df['YearMonth'] = df['YearMonth'].dt.to_timestamp()
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month

df = df.drop(columns = ['Date'])
df = df.dropna()

print(df.head())

# Encode LSOA codes to numerical valuesdd
le = LabelEncoder()
df['LSOA code'] = le.fit_transform(df['LSOA code'])

# Split into train set and test set

features = ['IMD','LSOA code', 'Year', 'Month']

df_train = df[df['Year'] < 2023]
df_test = df[df['Year'] >= 2023]

X_train = df_train[features]
y_train = df_train['Burglary Count']

X_test = df_test[features]
y_test = df_test['Burglary Count']

# Fit the model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Make predictions

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculate scores

print("Train R2 score: ", r2_score(y_train, y_train_pred))
print('Train MAE: ', mean_absolute_error(y_train, y_train_pred))
print("Test R2 score: ", r2_score(y_test, y_test_pred))
print('Test MAE: ', mean_absolute_error(y_test, y_test_pred))

# Visualize results

plt.figure(figsize=(16,4))
sns.lineplot(x = df['YearMonth'], y = df['Burglary Count'], label = 'True values')
sns.lineplot(x = df_train['YearMonth'], y = y_train_pred, label = 'Train set predictions', linestyle = '--')
sns.lineplot(x = df_test['YearMonth'], y = y_test_pred, label = 'Test set predictions', linestyle = '--')
plt.title('Random Forest on Monthly Burglary Count (Train < 2023 | Test >= 2023)')
plt.xlabel('Year-Month')
plt.ylabel('Burglary count')
plt.tight_layout()
plt.legend()
plt.show()
