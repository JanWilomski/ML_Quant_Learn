import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix
import seaborn as sns


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

def create_sequences(data, labels, timesteps=20):
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
timesteps = 20

X_train_lstm, y_train_lstm = create_sequences(x_train_scaled, y_train.values, timesteps)
X_test_lstm, y_test_lstm = create_sequences(x_test_scaled, y_test.values, timesteps)

print(f"X_train_lstm shape: {X_train_lstm.shape}")
print(f"y_train_lstm shape: {y_train_lstm.shape}")
print(f"X_test_lstm shape: {X_test_lstm.shape}")
print(f"y_test_lstm shape: {y_test_lstm.shape}")


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

# Wykres accuracy w czasie
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Accuracy w czasie')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Loss w czasie')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
#plt.show()

test_loss, test_accuracy = model.evaluate(x_test_scaled, y_test)
print(f"\n{'='*50}")
print(f"WYNIKI NA TEST SET (2025):")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"{'='*50}")


y_pred_probs = model.predict(x_test_scaled)
y_pred_classes = np.argmax(y_pred_probs, axis=1)

print(f"Shape predykcji (prawdopodobieństwa): {y_pred_probs.shape}")
print(f"Shape predykcji (klasy): {y_pred_classes.shape}")
print(f"\nPierwsze 5 przykładów:")
print("Prawdopodobieństwa:")
print(y_pred_probs[:5])
print("\nPrzewidywane klasy:")
print(y_pred_classes[:5])
print("\nPrawdziwe klasy:")
print(y_test.values[:5])

print("\nRozkład predykcji na test set:")
print(pd.Series(y_pred_classes).value_counts())
print("\nRozkład prawdziwych klas na test set:")
print(pd.Series(y_test.values).value_counts())



# Stwórz confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)

# Wizualizacja
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['SELL (0)', 'HOLD (1)', 'BUY (2)'],
            yticklabels=['SELL (0)', 'HOLD (1)', 'BUY (2)'])
plt.title('Confusion Matrix - Test Set (2025)')
plt.ylabel('Przewidywana klasa')
plt.xlabel('Prawdziwa klasa')
plt.tight_layout()
plt.show()

# Szczegółowe statystyki
print("\nConfusion Matrix:")
print(cm)
print(f"\nPoprawne predykcje (przekątna): {cm.diagonal().sum()}")
print(f"Wszystkie predykcje: {cm.sum()}")
print(f"Accuracy: {cm.diagonal().sum() / cm.sum():.4f}")





