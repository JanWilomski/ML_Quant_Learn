import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('data/wig20_d.csv')

df['Data'] = pd.to_datetime(df['Data'])

df.set_index('Data', inplace=True)

df['Future_Close'] = df['Zamkniecie'].shift(-3)
df['Future_Return'] = (df['Future_Close'] - df['Zamkniecie'])/(df['Zamkniecie'])*100


df['Signal'] = pd.cut(df['Future_Return'], bins=[-100, -1, 1, 100], labels=[0,1,2])

df.dropna(inplace=True)

print(df[['Zamkniecie', 'Future_Close', 'Future_Return', 'Signal']].head(10))

print(df['Signal'].value_counts())
print(f"\nBraki danych: {df['Signal'].isna().sum()}")
print(f"\nOstatnie wiersze:\n{df[['Zamkniecie', 'Future_Close', 'Signal']].tail()}")

df['Returns'] = df['Zamkniecie'].pct_change()*100
df['SMA_7'] = df['Zamkniecie'].rolling(window=7).mean()
df['SMA_14'] = df['Zamkniecie'].rolling(window=14).mean()
df['SMA_60'] = df['Zamkniecie'].rolling(window=60).mean()
df['SMA_128'] = df['Zamkniecie'].rolling(window=128).mean()
df['Price_to_SMA_7'] = (df['Zamkniecie'] / df['SMA_7'] - 1) * 100
df['Price_to_SMA_14'] = (df['Zamkniecie'] / df['SMA_14'] - 1) * 100
df['Price_to_SMA_60'] = (df['Zamkniecie'] / df['SMA_60'] - 1) * 100
df['Price_to_SMA_128'] = (df['Zamkniecie'] / df['SMA_128'] - 1) * 100
df['SMA_7_return'] = df['SMA_7'].pct_change()*100
df['SMA_14_return'] = df['SMA_14'].pct_change()*100
df['SMA_60_return'] = df['SMA_60'].pct_change()*100
df['SMA_128_return'] = df['SMA_128'].pct_change()*100


df.dropna(inplace=True)


print(df[['Zamkniecie', 'Returns', 'SMA_7', 'Price_to_SMA_7', 'Signal']].head(15))

print(f"Liczba wierszy po usunięciu NaN: {len(df)}")
print(f"Zakres dat: {df.index.min()} do {df.index.max()}")
print(f"\nRozkład sygnałów po czyszczeniu:")
print(df['Signal'].value_counts())


features = ['Returns', 'Wolumen', 'Price_to_SMA_7', 'Price_to_SMA_14', 'Price_to_SMA_60', 'Price_to_SMA_128', 'SMA_7_return', 'SMA_14_return', 'SMA_60_return', 'SMA_128_return']


x_train = df[df.index.year < 2025][features]
y_train = df[df.index.year < 2025]['Signal']
x_test = df[df.index.year == 2025][features]
y_test = df[df.index.year == 2025]['Signal']

print(f"X_train shape: {x_train.shape}")
print(f"X_test shape: {x_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

print(x_train.describe())

