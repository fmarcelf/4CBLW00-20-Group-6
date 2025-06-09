import pandas as pd

# Clean data and save new dataframe to use for dashboard

df = pd.read_csv("Data/Data - Transformed.csv", parse_dates=['Date'])
df.dropna()
df.drop(columns=['Unnamed: 0', 'Ward code', 'Ward name'], inplace=True)
df['YearMonth'] = df['Date'].dt.to_period('M')

df = df.groupby(['YearMonth', 'LSOA code', 'LSOA name']).size().reset_index()
df = df.rename({0: "Burglaries"}, axis=1)
df['Year'] = df['YearMonth'].dt.year
df['Month'] = df['YearMonth'].dt.month
print(df.head())

# Save to csv
df.to_csv('Data/data.csv')

