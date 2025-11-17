from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tqdm import tqdm
import os



class TradingEnvironment:
    def __init__(self, data, initial_balance=10000, position_size=10000):
        """
        data: DataFrame z kolumną 'Zamkniecie'
        initial_balance: startowy kapitał
        """
        self.data = data.reset_index(drop=True)  # Reset index dla łatwego dostępu
        self.initial_balance = initial_balance
        self.position_size = position_size
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
        Zawiera: wszystkie features + info o pozycji
        """
        # Weź wszystkie wartości z aktualnego wiersza (oprócz close - bo już jest w features)
        current_row = self.data.iloc[self.current_step]

        # Ma pozycję?
        if self.position is not None:
            has_position = 1
            entry_price = self.position['entry_price']
            current_price = current_row['close']
            position_pnl = current_price - entry_price
        else:
            has_position = 0
            entry_price = 0
            position_pnl = 0

        # State = [wszystkie features z DataFrame] + [info o pozycji]
        features_array = current_row.values  # Wszystkie kolumny jako array
        position_info = np.array([has_position, entry_price, position_pnl, self.balance])

        state = np.concatenate([features_array, position_info])

        return state

    def step(self, action):
        reward = 0
        current_price = self.data.iloc[self.current_step]['close']

        # HOLD (0)
        if action == 0:
            pass

        # BUY (1)
        elif action == 1:
            if self.position is None:
                # Otwórz LONG
                self.position = {
                    'entry_price': current_price,
                    'entry_step': self.current_step,
                    'type': 'long'  # ← NOWE
                }
            elif self.position['type'] == 'short':
                # Zamknij SHORT
                profit_per_unit = self.position['entry_price'] - current_price  # ← Odwrotnie!
                profit = profit_per_unit * self.position_size
                self.balance += profit
                self.total_profit += profit
                reward = profit
                self.position = None
            # Jeśli ma już LONG → ignoruj

        # SELL (2)
        elif action == 2:
            if self.position is None:
                # Otwórz SHORT
                self.position = {
                    'entry_price': current_price,
                    'entry_step': self.current_step,
                    'type': 'short'  # ← NOWE
                }
            elif self.position['type'] == 'long':
                # Zamknij LONG
                profit_per_unit = current_price - self.position['entry_price']
                profit = profit_per_unit * self.position_size
                self.balance += profit
                self.total_profit += profit
                reward = profit
                self.position = None

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


df = pd.read_csv('data/eurusd_h1_27_01_2025-17_11_2025.csv', sep=';')

df_vwap = pd.read_csv('data/eurusd_vwap_27_01_2025-17_11_2025_vwap.csv', sep=';')

df_bollinger = pd.read_csv('data/eurusd_bollinger_27_01_2025-17_11_2025_bollinger.csv', sep=';')



df['datetime'] = pd.to_datetime(df['datetime'])



df.set_index('datetime', inplace=True)

df.sort_index(inplace=True)

print(df.head())

df['returns'] = df['close'].pct_change()*100
df['SMA_7'] = df['close'].rolling(window=7).mean()
df['SMA_14'] = df['close'].rolling(window=14).mean()
df['SMA_60'] = df['close'].rolling(window=60).mean()
df['SMA_128'] = df['close'].rolling(window=128).mean()
df['Price_to_SMA_7'] = (df['close'] / df['SMA_7'] - 1) * 100
df['Price_to_SMA_14'] = (df['close'] / df['SMA_14'] - 1) * 100
df['Price_to_SMA_60'] = (df['close'] / df['SMA_60'] - 1) * 100
df['Price_to_SMA_128'] = (df['close'] / df['SMA_128'] - 1) * 100
df['SMA_7_return'] = df['SMA_7'].pct_change()*100
df['SMA_14_return'] = df['SMA_14'].pct_change()*100
df['SMA_60_return'] = df['SMA_60'].pct_change()*100
df['SMA_128_return'] = df['SMA_128'].pct_change()*100
df['vwap'] = df_vwap['vwap']
df['bollinger_upper'] = df_bollinger['upper_band']
df['bollinger_lower'] = df_bollinger['lower_band']
df['bollinger_middle'] = df_bollinger['middle_band']

df.dropna(inplace=True)

df['Sentiment'] = pd.cut(df['returns'], bins=[-100, 0, 100], labels=[0,1]).astype(int)

df.dropna(inplace=True)

features = ['close', 'SMA_7', 'SMA_14', 'SMA_128', 'SMA_7_return', 'SMA_14_return', 'SMA_128_return', 'Price_to_SMA_7', 'Price_to_SMA_14', 'Price_to_SMA_60','Sentiment', 'vwap', 'bollinger_upper', 'bollinger_lower', 'bollinger_middle']

# Oblicz indeksy podziału
total_len = len(df)
train_end = int(total_len * 0.70)  # 70% train
val_end = int(total_len * 0.85)    # następne 15% val (85% - 70%)
# Reszta (15%) to test

# Podziel dane
train_data = df.iloc[:train_end][features].copy()
val_data = df.iloc[train_end:val_end][features].copy()
test_data = df.iloc[val_end:][features].copy()

print(f"\n{'='*50}")
print(f"Podział danych:")
print(f"Train: {len(train_data)} godzin ({len(train_data)/24:.1f} dni)")
print(f"Val:   {len(val_data)} godzin ({len(val_data)/24:.1f} dni)")
print(f"Test:  {len(test_data)} godzin ({len(test_data)/24:.1f} dni)")
print(f"{'='*50}\n")

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
# Stwórz agenta
state_size = len(features) + 4
agent = SimpleDQNAgent(state_size=state_size, action_size=3)






# Sprawdź czy istnieje zapisany model
if os.path.exists('trading_agent.keras'):
    print("Znaleziono zapisany model! Ładuję...")
    agent.model = keras.models.load_model('trading_agent.keras')
    agent.epsilon = 0.01  # Ustaw niskie epsilon (mało exploration)
    print("✓ Model załadowany")

    # Pytanie: czy trenować dalej?
    train_more = input("Czy chcesz trenować dalej? (tak/nie): ").lower()
    if train_more != 'tak':
        episodes = 0  # Pomiń trening
else:
    print("Brak zapisanego modelu. Rozpoczynam trening od zera...")

# Stwórz 3 environments
train_env = TradingEnvironment(train_data, initial_balance=10000)
val_env = TradingEnvironment(val_data, initial_balance=10000)
test_env = TradingEnvironment(test_data, initial_balance=10000)

# Użyj train_env do testu state
test_state = train_env.reset()



# TEST: Sprawdź czy state ma poprawny rozmiar
def test_agent(agent, env, name="Test"):
    """Testuje agenta bez treningu"""
    state = env.reset()
    done = False
    total_reward = 0

    # Wyłącz exploration
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0

    while not done:
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)

        if not done:
            state = next_state

        total_reward += reward

    # Przywróć epsilon
    agent.epsilon = old_epsilon

    return total_reward, info['balance']

# Sprawdź czy agent przyjmuje state
action = agent.act(test_state)
print(f"Agent wybrał akcję: {action} ({'HOLD' if action==0 else 'BUY' if action==1 else 'SELL'})\n")

episodes = 20
val_frequency = 5  # Testuj na validation co 5 episodes
best_val_reward = -float('inf')
best_model_path = 'best_trading_agent.keras'

rewards_history = []
val_rewards_history = []

for episode in range(episodes):
    state = train_env.reset()  # Użyj train_env!
    total_reward = 0
    done = False

    with tqdm(total=len(train_data), desc=f"Episode {episode + 1}/{episodes}") as pbar:
        while not done:
            action = agent.act(state)
            next_state, reward, done, info = train_env.step(action)

            if not done:
                agent.train(state, action, reward, next_state, done)
                state = next_state

            total_reward += reward
            pbar.update(1)
            pbar.set_postfix({'reward': f'{total_reward:.2f}', 'balance': f'{info["balance"]:.2f}'})

    rewards_history.append(total_reward)

    # Testuj na validation co val_frequency episodes
    if (episode + 1) % val_frequency == 0:
        val_reward, val_balance = test_agent(agent, val_env, "Validation")
        val_rewards_history.append(val_reward)

        print(
            f"Episode {episode + 1} - Train Reward: {total_reward:.2f}, Val Reward: {val_reward:.2f}, Val Balance: {val_balance:.2f}")

        # Zapisz jeśli najlepszy na validation
        if val_reward > best_val_reward:
            best_val_reward = val_reward
            agent.model.save(best_model_path)
            print(f"  ✓ Nowy najlepszy model! Val Reward: {val_reward:.2f}")
    else:
        print(f"Episode {episode + 1} - Train Reward: {total_reward:.2f}")


# Wykres
plt.plot(rewards_history)
plt.title('Total Reward per Episode')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.show()

test_env = TradingEnvironment(test_data, initial_balance=10000)

# Reset i testuj (BEZ treningu!)
state = test_env.reset()
done = False
test_reward = 0

# Wyłącz exploration (agent używa tylko tego czego się nauczył)
agent.epsilon = 0.0  # ← WAŻNE: epsilon=0 = nie losuj, używaj najlepszych Q

print(f"\n{'=' * 50}")
print("TEST NA DANYCH 2025")
print(f"{'=' * 50}")

# Sprawdź Q-wartości dla pierwszego state z 2025
first_state = test_env.reset()
q_values = agent.model.predict(first_state.reshape(1, -1), verbose=0)[0]

print("\nQ-wartości dla pierwszego dnia 2025:")
print(f"HOLD: {q_values[0]:.2f}")
print(f"BUY:  {q_values[1]:.2f}")
print(f"SELL: {q_values[2]:.2f}")
print(f"Agent wybiera akcję: {np.argmax(q_values)} ({'HOLD' if np.argmax(q_values)==0 else 'BUY' if np.argmax(q_values)==1 else 'SELL'})")

with tqdm(total=len(test_data), desc="Testing on 2025") as pbar:
    while not done:
        action = agent.act(state)  # ← Tylko ACT, BEZ train()!
        next_state, reward, done, info = test_env.step(action)

        if not done:
            state = next_state

        test_reward += reward
        pbar.update(1)
        pbar.set_postfix({'reward': f'{test_reward:.2f}', 'balance': f'{info["balance"]:.2f}'})

print(f"\n{'=' * 50}")
print(f"TEST ZAKOŃCZONY")
print(f"Final Balance: {info['balance']:.2f}")
print(f"Total Profit: {info['total_profit']:.2f}")
print(f"Total Reward: {test_reward:.2f}")
print(f"{'=' * 50}")

