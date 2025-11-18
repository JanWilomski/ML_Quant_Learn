from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tqdm import tqdm
import os
import tensorflow as tf


# ===== SKOPIUJ KLASY Z POPRZEDNIEGO PLIKU =====
# (TradingEnvironment i PolicyGradientAgent - identyczne)

class TradingEnvironment:
    def __init__(self, data, initial_balance=10000, position_size=1000,
                 max_episode_steps=None, random_start=False):
        """
        data: DataFrame z kolumną 'close'
        initial_balance: startowy kapitał
        position_size: wielkość pozycji (dla WIG20 możemy użyć mniejszą - 1000 zamiast 10000)
        """
        self.data = data.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.position_size = position_size
        self.n_steps = len(data)
        self.max_episode_steps = max_episode_steps
        self.random_start = random_start

        self.current_step = 0
        self.episode_start = 0
        self.balance = initial_balance
        self.position = None
        self.total_profit = 0

    def reset(self):
        if self.random_start and self.max_episode_steps is not None and self.max_episode_steps < self.n_steps:
            self.episode_start = np.random.randint(0, self.n_steps - self.max_episode_steps)
        else:
            self.episode_start = 0

        self.current_step = self.episode_start
        self.balance = self.initial_balance
        self.position = None
        self.total_profit = 0
        return self._get_state()

    def _get_state(self):
        current_row = self.data.iloc[self.current_step]

        if self.position is not None:
            has_position = 1
            entry_price = self.position['entry_price']
            current_price = current_row['close']
            position_pnl = current_price - entry_price
        else:
            has_position = 0
            entry_price = 0
            position_pnl = 0

        features_array = current_row.values
        position_info = np.array([has_position, entry_price, position_pnl, self.balance])
        state = np.concatenate([features_array, position_info])
        return state

    def step(self, action):
        reward = 0
        current_price = self.data.iloc[self.current_step]['close']

        if action == 0:  # HOLD
            pass
        elif action == 1:  # BUY
            if self.position is None:
                self.position = {'entry_price': current_price, 'entry_step': self.current_step, 'type': 'long'}
            elif self.position['type'] == 'short':
                profit_per_unit = self.position['entry_price'] - current_price
                profit = profit_per_unit * self.position_size
                self.balance += profit
                self.total_profit += profit
                reward = profit
                self.position = None
        elif action == 2:  # SELL
            if self.position is None:
                self.position = {'entry_price': current_price, 'entry_step': self.current_step, 'type': 'short'}
            elif self.position['type'] == 'long':
                profit_per_unit = current_price - self.position['entry_price']
                profit = profit_per_unit * self.position_size
                self.balance += profit
                self.total_profit += profit
                reward = profit
                self.position = None

        self.current_step += 1

        done_data = (self.current_step >= self.n_steps)
        if self.max_episode_steps is not None:
            done_length = (self.current_step >= self.episode_start + self.max_episode_steps)
        else:
            done_length = False
        done_bankrupt = (self.balance <= 0)
        done = done_data or done_length or done_bankrupt

        if done and self.position is not None:
            current_price = self.data.iloc[self.current_step - 1]['close']
            if self.position['type'] == 'long':
                profit_per_unit = current_price - self.position['entry_price']
            else:
                profit_per_unit = self.position['entry_price'] - current_price
            profit = profit_per_unit * self.position_size
            self.balance += profit
            self.total_profit += profit
            reward = profit
            self.position = None

        next_state = self._get_state() if not done else None
        info = {'balance': self.balance, 'total_profit': self.total_profit, 'step': self.current_step}
        return next_state, reward, done, info


class PolicyGradientAgent:
    def __init__(self, state_size=5, action_size=3, learning_rate=0.0001, gamma=0.95, temperature=3.0):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.temperature = temperature
        self.model = self.build_model(learning_rate)

    def build_model(self, learning_rate):
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_dim=self.state_size),
            layers.Dense(32, activation='relu'),
            layers.Dense(self.action_size, activation='linear',
                         kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
                         bias_initializer=keras.initializers.Zeros())
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
        return model

    def act(self, state, greedy=False):
        logits = self.model.predict(state.reshape(1, -1), verbose=0)[0]
        logits_scaled = logits / self.temperature
        exp_logits = np.exp(logits_scaled - np.max(logits_scaled))
        probabilities = exp_logits / np.sum(exp_logits)

        if greedy:
            return np.argmax(probabilities)
        return np.random.choice(self.action_size, p=probabilities)

    def train(self, states, actions, rewards):
        returns = self.compute_returns(rewards, self.gamma)
        returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)

        states = np.array(states)
        actions = np.array(actions)

        with tf.GradientTape() as tape:
            logits = self.model(states, training=True)
            logits_scaled = logits / self.temperature
            action_probs = tf.nn.softmax(logits_scaled, axis=-1)

            indices = tf.range(len(actions)) * self.action_size + actions
            action_probs_for_actions = tf.gather(tf.reshape(action_probs, [-1]), indices)

            log_probs = tf.math.log(action_probs_for_actions + 1e-8)
            entropy = -tf.reduce_sum(action_probs * tf.math.log(action_probs + 1e-8), axis=1)
            loss = tf.reduce_mean(-(log_probs * returns + 0.01 * entropy))

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

    def compute_returns(self, rewards, gamma):
        returns = []
        running_return = 0
        for i in range(len(rewards) - 1, -1, -1):
            running_return = rewards[i] + gamma * running_return
            returns.append(running_return)
        returns.reverse()
        return np.array(returns)


# ========== WCZYTAJ DANE WIG20 ==========

print("Wczytuję dane WIG20...")
df = pd.read_csv('data/wig20_h1_11_04_2024-17_11_2025.csv')  # Zmień na swoją ścieżkę!

# Parsuj kolumny date i time
df['datetime'] = pd.to_datetime(df['date'].astype(str) + df['time'].astype(str).str.zfill(6),
                                format='%Y%m%d%H%M%S')

df = df.set_index('datetime').sort_index()

# Zostaw tylko potrzebne kolumny
df = df[['open', 'high', 'low', 'close', 'volume']]

print(f"Wczytano {len(df)} wierszy")
print(f"Okres: {df.index[0]} - {df.index[-1]}")
print(f"\nPrzykładowe dane:")
print(df.head())

# ========== FEATURE ENGINEERING ==========

print("\nTworzę features...")

# Podstawowe
df['returns'] = df['close'].pct_change() * 100

# Moving Averages
df['SMA_7'] = df['close'].rolling(window=7).mean()
df['SMA_14'] = df['close'].rolling(window=14).mean()
df['SMA_30'] = df['close'].rolling(window=30).mean()

# Relatywne pozycje do MA (WAŻNE!)
df['price_to_SMA_7'] = (df['close'] / df['SMA_7'] - 1) * 100
df['price_to_SMA_14'] = (df['close'] / df['SMA_14'] - 1) * 100
df['price_to_SMA_30'] = (df['close'] / df['SMA_30'] - 1) * 100

# Zwroty MA
df['SMA_7_return'] = df['SMA_7'].pct_change() * 100
df['SMA_14_return'] = df['SMA_14'].pct_change() * 100
df['SMA_30_return'] = df['SMA_30'].pct_change() * 100


# Volume features
df['volume_ma'] = df['volume'].rolling(window=20).mean()
df['volume_ratio'] = df['volume'] / df['volume_ma']


# RSI
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


df['RSI'] = calculate_rsi(df['close'])

# Sentiment (prosty - bazujący na returns)
df['Sentiment'] = pd.cut(df['returns'], bins=[-100, -0.5, 0.5, 100], labels=[0, 1, 2]).astype(float)

# Usuń NaN
df.dropna(inplace=True)

print(f"Po utworzeniu features: {len(df)} wierszy")

# ========== WYBIERZ FEATURES ==========

features = [
    'close',  # Surowa cena (potrzebna do PnL)
    'returns',  # Zwroty
    'SMA_7', 'SMA_14', 'SMA_30',  # Surowe MA
    'price_to_SMA_7', 'price_to_SMA_14', 'price_to_SMA_30',  # Relatywne do MA
    'SMA_7_return', 'SMA_14_return', 'SMA_30_return',  # Zwroty MA
    'volume',
    'volume_ratio',  # Relatywny volume
    'RSI',  # RSI
    'Sentiment'  # Prosty sentiment
]

print(f"\nUżywam {len(features)} features:")
print(features)

# ========== PODZIAŁ DANYCH ==========

total_len = len(df)
train_end = int(total_len * 0.70)
val_end = int(total_len * 0.85)

train_raw = df.iloc[:train_end][features].copy()
val_raw = df.iloc[train_end:val_end][features].copy()
test_raw = df.iloc[val_end:][features].copy()

# Normalizuj WSZYSTKO oprócz close (close używamy do PnL)
features_to_scale = [f for f in features if f != 'close']

scaler = StandardScaler()
train_scaled = train_raw.copy()
val_scaled = val_raw.copy()
test_scaled = test_raw.copy()

train_scaled[features_to_scale] = scaler.fit_transform(train_raw[features_to_scale])
val_scaled[features_to_scale] = scaler.transform(val_raw[features_to_scale])
test_scaled[features_to_scale] = scaler.transform(test_raw[features_to_scale])

train_data = train_scaled
val_data = val_scaled
test_data = test_scaled

print(f"\n{'=' * 50}")
print(f"Podział danych:")
print(f"Train: {len(train_data)} godzin ({len(train_data) / 24:.1f} dni)")
print(f"Val:   {len(val_data)} godzin ({len(val_data) / 24:.1f} dni)")
print(f"Test:  {len(test_data)} godzin ({len(test_data) / 24:.1f} dni)")
print(f"{'=' * 50}\n")

# ========== STWÓRZ AGENTA ==========

state_size = len(features) + 4
agent = PolicyGradientAgent(state_size=state_size, action_size=3, temperature=3.0)

print(f"Agent stworzony: state_size={state_size}")

# ========== ENVIRONMENTS ==========

MAX_EPISODE_STEPS = 500

train_env = TradingEnvironment(
    train_data, initial_balance=10000, position_size=1000,  # Mniejsza pozycja dla WIG20
    max_episode_steps=MAX_EPISODE_STEPS, random_start=True
)

val_env = TradingEnvironment(
    val_data, initial_balance=10000, position_size=1000,
    max_episode_steps=None, random_start=False
)

test_env = TradingEnvironment(
    test_data, initial_balance=10000, position_size=1000,
    max_episode_steps=None, random_start=False
)


def test_agent(agent, env, n_runs=3):
    """Testuje agenta N razy i zwraca średnią (Opcja B - multiple runs)"""
    rewards = []
    balances = []

    for _ in range(n_runs):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.act(state, greedy=False)  # Stochastyczny!
            next_state, reward, done, info = env.step(action)

            if not done:
                state = next_state

            total_reward += reward

        rewards.append(total_reward)
        balances.append(info['balance'])

    return np.mean(rewards), np.mean(balances)


# ========== TRENING ==========

print("Rozpoczynam trening...\n")

episodes = 100  # Więcej epizodów dla WIG20
best_val_reward = -float('inf')

rewards_history = []
val_rewards_history = []

for episode in range(episodes):
    state = train_env.reset()

    # Pokaż prawdopodobieństwa co 20 epizodów
    if episode % 20 == 0:
        first_probs_logits = agent.model.predict(state.reshape(1, -1), verbose=0)[0]
        first_probs_logits_scaled = first_probs_logits / agent.temperature
        exp_logits = np.exp(first_probs_logits_scaled - np.max(first_probs_logits_scaled))
        first_probs = exp_logits / np.sum(exp_logits)
        tqdm.write(
            f"Episode {episode + 1} - Prawdopodobieństwa: HOLD={first_probs[0]:.3f}, BUY={first_probs[1]:.3f}, SELL={first_probs[2]:.3f}")

    total_reward = 0
    done = False
    states = []
    actions = []
    rewards_ep = []
    action_counts = {0: 0, 1: 0, 2: 0}
    last_info = None

    ep_len = train_env.max_episode_steps or len(train_data)
    with tqdm(total=ep_len, desc=f"Episode {episode + 1}/{episodes}", leave=False) as pbar:
        while not done:
            action = agent.act(state)
            action_counts[action] += 1
            next_state, reward, done, info = train_env.step(action)

            last_info = info
            states.append(state)
            actions.append(action)
            rewards_ep.append(reward)
            state = next_state
            total_reward += reward

            pbar.update(1)
            pbar.set_postfix({'reward': f'{total_reward:.2f}'})

    # Trenuj
    agent.train(states, actions, rewards_ep)
    rewards_history.append(total_reward)

    train_balance = last_info['balance'] if last_info is not None else train_env.balance

    # Validation (multiple runs - Opcja B)
    val_reward, val_balance = test_agent(agent, val_env, n_runs=3)
    val_rewards_history.append(val_reward)

    # Print co 5 epizodów
    if (episode + 1) % 5 == 0:
        tqdm.write(f"  Akcje: HOLD={action_counts[0]}, BUY={action_counts[1]}, SELL={action_counts[2]}")
        tqdm.write(
            f"Episode {episode + 1} - "
            f"Train Reward: {total_reward:.2f}, Train Balance: {train_balance:.2f}, "
            f"Val Reward: {val_reward:.2f}, Val Balance: {val_balance:.2f}"
        )

        if val_reward > best_val_reward:
            best_val_reward = val_reward
            agent.model.save('best_pg_wig20.keras')
            tqdm.write(f"  ✓ Nowy najlepszy model! Val Reward: {val_reward:.2f}")

# ========== WYKRESY ==========

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(rewards_history)
plt.title('Training Reward per Episode')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.grid(True)

plt.subplot(1, 3, 2)
if len(val_rewards_history) > 0:
    plt.plot(range(1, len(val_rewards_history) + 1), val_rewards_history, 'o-')
    plt.title('Validation Reward (avg of 3 runs)')
    plt.xlabel('Episode')
    plt.ylabel('Val Reward')
    plt.grid(True)

plt.subplot(1, 3, 3)
# Moving average train rewards
window = 10
if len(rewards_history) >= window:
    ma_rewards = pd.Series(rewards_history).rolling(window=window).mean()
    plt.plot(ma_rewards)
    plt.title(f'Training Reward (MA-{window})')
    plt.xlabel('Episode')
    plt.ylabel('MA Reward')
    plt.grid(True)

plt.tight_layout()
plt.savefig('pg_wig20_results.png')
print("\n✓ Wykres zapisany jako 'pg_wig20_results.png'\n")

# ========== TEST ==========

print(f"\n{'=' * 50}")
print("TEST NA DANYCH TESTOWYCH (avg of 5 runs)")
print(f"{'=' * 50}\n")

test_reward, test_balance = test_agent(agent, test_env, n_runs=5)

print(f"Test Reward (avg): {test_reward:.2f}")
print(f"Test Balance (avg): {test_balance:.2f}")
print(f"Test Profit (avg): {test_balance - 10000:.2f}")
print(f"{'=' * 50}\n")

print("✓ Trening zakończony!")
print("\nPodsumowanie:")
print(f"- Best Val Reward: {best_val_reward:.2f}")
print(f"- Test Reward: {test_reward:.2f}")
print(f"- Model zapisany jako: best_pg_wig20.keras")