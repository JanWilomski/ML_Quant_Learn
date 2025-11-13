from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import layers



class TradingEnvironment:
    def __init__(self, data, initial_balance=10000):
        """
        data: DataFrame z kolumną 'Zamkniecie'
        initial_balance: startowy kapitał
        """
        self.data = data.reset_index(drop=True)  # Reset index dla łatwego dostępu
        self.initial_balance = initial_balance
        self.n_steps = len(data)

        # Zmienne stanu
        self.current_step = 0
        self.balance = initial_balance
        self.position = None  # None = brak pozycji, lub dict z info o pozycji
        self.total_profit = 0

    def reset(self):
        """Resetuje environment na początek"""
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = None
        self.total_profit = 0
        return self._get_state()

    def _get_state(self):
        """
        Zwraca state dla agenta.
        Zawiera: cenę aktualną, czy ma pozycję, P&L
        """
        current_price = self.data.iloc[self.current_step]['Zamkniecie']

        # Ma pozycję?
        if self.position is not None:
            has_position = 1
            entry_price = self.position['entry_price']
            position_pnl = current_price - entry_price  # P&L w PLN
        else:
            has_position = 0
            entry_price = 0
            position_pnl = 0

        # State: [cena, czy_ma_pozycję, cena_wejścia, P&L, balance]
        state = np.array([
            current_price,
            has_position,
            entry_price,
            position_pnl,
            self.balance
        ])

        return state

    def step(self, action):
        """
        Wykonuje akcję: 0=HOLD, 1=BUY, 2=SELL
        Zwraca: (state, reward, done, info)
        """
        reward = 0
        current_price = self.data.iloc[self.current_step]['Zamkniecie']

        # HOLD (0)
        if action == 0:
            pass  # Nic nie rób

        # BUY (1)
        elif action == 1:
            if self.position is None:  # Można kupić tylko gdy nie ma pozycji
                self.position = {
                    'entry_price': current_price,
                    'entry_step': self.current_step
                }
            # Jeśli ma już pozycję - ignoruj akcję (lub daj małą karę?)

        # SELL (2)
        elif action == 2:
            if self.position is not None:  # Można sprzedać tylko gdy ma pozycję
                # Oblicz zysk
                profit = current_price - self.position['entry_price']
                self.balance += profit
                self.total_profit += profit
                reward = profit  # REWARD = zrealizowany zysk!

                # Zamknij pozycję
                self.position = None
            # Jeśli nie ma pozycji - ignoruj

        # Przejdź do następnego dnia
        self.current_step += 1

        # Sprawdź czy koniec
        done = (self.current_step >= self.n_steps) or (self.balance <= 0)

        # Nowy state
        next_state = self._get_state() if not done else None

        info = {
            'balance': self.balance,
            'total_profit': self.total_profit,
            'step': self.current_step
        }

        return next_state, reward, done, info


df = pd.read_csv('data/wig20_d.csv')

df['Data'] = pd.to_datetime(df['Data'])

df.set_index('Data', inplace=True)

df['Future_Close'] = df['Zamkniecie'].shift(-3)
df['Future_Return'] = (df['Future_Close'] - df['Zamkniecie'])/(df['Zamkniecie'])*100


df['Signal'] = pd.cut(df['Future_Return'], bins=[-100, -1, 1, 100], labels=[0,1,2])

df.dropna(inplace=True)


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



# Stwórz environment z danych treningowych
train_data = df[df.index.year < 2025][['Zamkniecie']].copy()
env = TradingEnvironment(train_data, initial_balance=10000)


def build_dqn_model(state_size=5, action_size=3):
    model = keras.Sequential()

    # TODO: dodaj warstwy
    # Hidden layer 1: Dense(64, activation='relu', input_dim=state_size)
    # Hidden layer 2: Dense(32, activation='relu')
    # Output layer: Dense(action_size, activation='linear')
    model.add(layers.Dense(64, activation='relu', input_dim=state_size))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(action_size, activation='linear'))

    model.compile(optimizer='adam', loss='mse')
    return model

model = build_dqn_model()
model.summary()


class SimpleDQNAgent:
    def __init__(self, state_size=5, action_size=3, learning_rate=0.001, gamma=0.95):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma  # discount factor (jak ważna jest przyszłość)
        self.epsilon = 1.0  # exploration rate (na początku losuje)
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

        self.model = self.build_model(learning_rate)

    def build_model(self, learning_rate):
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_dim=self.state_size),
            layers.Dense(32, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                      loss='mse')
        return model

    def act(self, state):
        """Wybiera akcję: epsilon-greedy"""
        if np.random.random() <= self.epsilon:
            return np.random.randint(0, self.action_size)  # Losowa (explore)

        q_values = self.model.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(q_values[0])  # Najlepsza według Q (exploit)

    def train(self, state, action, reward, next_state, done):
        """Uczy się z pojedynczego kroku"""
        target = reward
        if not done:
            # Q-learning: target = reward + gamma * max(Q(next_state))
            target = reward + self.gamma * np.max(
                self.model.predict(next_state.reshape(1, -1), verbose=0)[0]
            )

        # Aktualna predykcja Q
        target_f = self.model.predict(state.reshape(1, -1), verbose=0)
        target_f[0][action] = target  # Popraw Q dla tej akcji

        # Trenuj
        self.model.fit(state.reshape(1, -1), target_f, epochs=1, verbose=0)

        # Zmniejsz epsilon (mniej exploration)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# Stwórz agenta i environment
agent = SimpleDQNAgent(state_size=5, action_size=3)
env = TradingEnvironment(train_data, initial_balance=10000)

episodes = 100
rewards_history = []

for episode in range(episodes):
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)

        if not done:
            agent.train(state, action, reward, next_state, done)
            state = next_state

        total_reward += reward

    rewards_history.append(total_reward)

    if episode % 10 == 0:
        print(
            f"Episode {episode}/{episodes}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.3f}, Final Balance: {info['balance']:.2f}")

# Wykres
plt.plot(rewards_history)
plt.title('Total Reward per Episode')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.show()