import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




df = pd.read_csv('data/wig20_d.csv')

df['Data'] = pd.to_datetime(df['Data'])

df.set_index('Data', inplace=True)


print(df.head())
print("=======================")
print(df.describe())
print("=======================")
print(df.shape)
print("=======================")
print(df.isnull().sum())
print("=======================")
df.info()
print("=======================")
print(df.loc['2023'])


plt.figure(figsize=(15, 5))
plt.plot(df.index, df['Zamkniecie'], label='WIG20', linewidth=2)

plt.title('WIG20 - Cena zamknięcia', fontsize=14)
plt.xlabel('Data')
plt.ylabel('Cena')
plt.legend()
plt.grid(True, alpha=0.3)  # siatka dla czytelności
plt.tight_layout()

plt.show()