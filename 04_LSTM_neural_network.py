import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tensorflow.keras.layers import LSTM


df = pd.read_csv('data/wig20_d.csv')

df['Data'] = pd.to_datetime(df['Data'])

df.set_index('Data', inplace=True)

df['Future_Close'] = df['Zamkniecie'].shift(-1)
df['Future_Return'] = (df['Future_Close'] - df['Zamkniecie'])/(df['Zamkniecie'])*100

df.dropna(inplace=True)
df['Signal'] = pd.cut(df['Future_Return'], bins=[-100, -0.5, 0.5, 100], labels=[0,1,2]).astype(int)



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

df['Sentiment'] = pd.cut(df['Returns'], bins=[-100, 0, 100], labels=[0,1]).astype(int)

df.dropna(inplace=True)

features = ['Returns', 'Wolumen', 'Price_to_SMA_7', 'Price_to_SMA_14', 'Price_to_SMA_60', 'Price_to_SMA_128', 'SMA_7_return', 'SMA_14_return', 'SMA_60_return', 'SMA_128_return', 'Sentiment']


x_train = df[df.index.year < 2025][features]
y_train = df[df.index.year < 2025]['Signal']
x_test = df[df.index.year == 2025][features]
y_test = df[df.index.year == 2025]['Signal']



scaler = StandardScaler()

scaler.fit(x_train)

x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

def create_sequences(data, labels, timesteps=50):
    """
    Tworzy sekwencje dla LSTM

    data: array (n_samples, n_features)
    labels: array (n_samples,)
    timesteps: ile dni wstecz

    Returns:
    X: array (n_sequences, timesteps, n_features)
    y: array (n_sequences,)
    """
    X, y = [], []

    for i in range(timesteps, len(data)):
        X.append(data[i-timesteps:i])
        y.append(labels[i])
        pass

    return np.array(X), np.array(y)

# Stwórz sekwencje dla LSTM
timesteps = 50

X_train_lstm, y_train_lstm = create_sequences(x_train_scaled, y_train.values, timesteps)
X_test_lstm, y_test_lstm = create_sequences(x_test_scaled, y_test.values, timesteps)

#=========================================================================

model = keras.Sequential()

model.add(layers.Dense(32, activation='relu', input_dim=len(features)))


model.add(layers.Dense(3, activation='softmax'))

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


history = model.fit(
    x_train_scaled,
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

#=========================================================================

model_lstm = keras.Sequential()

# Warstwa LSTM - 128 units, input_shape dla pierwszej warstwy
model_lstm.add(LSTM(128, input_shape=(timesteps, 11)))


# Warstwa wyjściowa - taka sama jak w MLP
model_lstm.add(layers.Dense(3, activation='softmax'))

# Kompilacja
model_lstm.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Trening LSTM
history_lstm = model_lstm.fit(
    X_train_lstm,
    y_train_lstm,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Ocena na test set
test_loss_lstm, test_accuracy_lstm = model_lstm.evaluate(X_test_lstm, y_test_lstm)
test_loss, test_accuracy = model.evaluate(x_test_scaled, y_test)

print(f"\n{'='*50}")
print(f"PORÓWNANIE WYNIKÓW:")
print(f"{'='*50}")
print(f"MLP Test Accuracy:  {test_accuracy:.4f}")
print(f"LSTM Test Accuracy: {test_accuracy_lstm:.4f}")
print(f"{'='*50}")









