import pandas as pd

# Clean data and save new dataframe to use for dashboard

df = pd.read_csv("Data/Data - Transformed.csv", parse_dates=['Date'])
df.dropna()
df.drop(columns=['Unnamed: 0', 'LSOA name', 'Ward code', 'Ward name'], inplace=True)
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month

df = df.groupby(['Year','Month', 'LSOA code']).size().reset_index()
df = df.rename({0: "Burglaries"}, axis=1)
print(df.head())

# Save to csv
df.to_csv('Data/data.csv')

