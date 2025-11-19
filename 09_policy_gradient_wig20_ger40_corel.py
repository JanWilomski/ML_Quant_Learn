import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Wyłącz TensorFlow warnings

from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tqdm import tqdm
import tensorflow as tf


# ============================================
#   TRADING ENVIRONMENT - ARBITRAŻ WIG20/DAX
# ============================================

class ArbitrageEnvironment:
    def __init__(self, data, initial_balance=10000, position_size=10,
                 max_episode_steps=None, random_start=False, features=None,
                 reward_scale=500.0, transaction_cost=0.0001):
        """
        Environment do arbitrażu WIG20 vs DAX

        data: DataFrame z kolumnami:
              - 'wig20_close_raw' (surowa cena WIG20 do PnL)
              - zeskalowane feature'y z WIG20 i DAX
        features: lista nazw kolumn features (bez 'wig20_close_raw')
        position_size: wielkość pozycji
        transaction_cost: koszt transakcji jako % (np. 0.0001 = 0.01%)
        """
        self.data = data.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.position_size = position_size
        self.n_steps = len(data)
        self.max_episode_steps = max_episode_steps
        self.random_start = random_start
        self.reward_scale = reward_scale
        self.transaction_cost = transaction_cost

        if features is None:
            self.features = [c for c in self.data.columns if c != 'wig20_close_raw']
        else:
            self.features = features

        self.current_step = 0
        self.episode_start = 0
        self.balance = initial_balance
        self.position = None
        self.total_profit = 0
        self.trade_count = 0

    def reset(self):
        if self.random_start and self.max_episode_steps and self.max_episode_steps < self.n_steps:
            max_start = self.n_steps - self.max_episode_steps
            self.episode_start = np.random.randint(0, max_start)
        else:
            self.episode_start = 0

        self.current_step = self.episode_start
        self.balance = self.initial_balance
        self.position = None
        self.total_profit = 0
        self.trade_count = 0
        return self._get_state()

    def _get_state(self):
        """State = features z WIG20 i DAX + info o pozycji"""
        current_row = self.data.iloc[self.current_step]
        current_price = current_row['wig20_close_raw']

        if self.position is not None:
            has_position = 1
            entry_price = self.position['entry_price']
            position_pnl = current_price - entry_price
        else:
            has_position = 0
            entry_price = current_price
            position_pnl = 0.0

        # Znormalizowane info o pozycji
        entry_price_rel = (entry_price / current_price) - 1.0
        pnl_rel = position_pnl / current_price
        balance_rel = (self.balance - self.initial_balance) / self.initial_balance

        position_info = np.array([
            has_position,
            entry_price_rel,
            pnl_rel,
            balance_rel
        ], dtype=np.float32)

        features_array = current_row[self.features].values.astype(np.float32)
        state = np.concatenate([features_array, position_info])

        return state

    def step(self, action):
        reward = 0.0
        current_price = self.data.iloc[self.current_step]['wig20_close_raw']

        # HOLD (0)
        if action == 0:
            if self.position is not None:
                reward = -0.001

        # BUY WIG20 (1)
        elif action == 1:
            if self.position is None:
                cost = current_price * self.position_size * self.transaction_cost
                self.balance -= cost
                self.position = {
                    'entry_price': current_price,
                    'entry_step': self.current_step,
                    'type': 'long'
                }
                self.trade_count += 1
            elif self.position['type'] == 'short':
                profit_per_unit = self.position['entry_price'] - current_price
                raw_profit = profit_per_unit * self.position_size
                cost = current_price * self.position_size * self.transaction_cost
                raw_profit -= cost

                self.balance += raw_profit
                self.total_profit += raw_profit
                reward = np.clip(raw_profit / self.reward_scale, -1.0, 1.0)
                self.position = None
                self.trade_count += 1

        # SELL WIG20 (2)
        elif action == 2:
            if self.position is None:
                cost = current_price * self.position_size * self.transaction_cost
                self.balance -= cost
                self.position = {
                    'entry_price': current_price,
                    'entry_step': self.current_step,
                    'type': 'short'
                }
                self.trade_count += 1
            elif self.position['type'] == 'long':
                profit_per_unit = current_price - self.position['entry_price']
                raw_profit = profit_per_unit * self.position_size
                cost = current_price * self.position_size * self.transaction_cost
                raw_profit -= cost

                self.balance += raw_profit
                self.total_profit += raw_profit
                reward = np.clip(raw_profit / self.reward_scale, -1.0, 1.0)
                self.position = None
                self.trade_count += 1

        self.current_step += 1

        done_data = (self.current_step >= self.n_steps)
        done_length = False
        if self.max_episode_steps:
            done_length = (self.current_step >= self.episode_start + self.max_episode_steps)
        done_bankrupt = (self.balance <= 0)

        done = done_data or done_length or done_bankrupt

        # Auto-close pozycji na końcu
        if done and self.position is not None:
            current_price = self.data.iloc[self.current_step - 1]['wig20_close_raw']
            if self.position['type'] == 'long':
                profit_per_unit = current_price - self.position['entry_price']
            else:
                profit_per_unit = self.position['entry_price'] - current_price

            raw_profit = profit_per_unit * self.position_size
            cost = current_price * self.position_size * self.transaction_cost
            raw_profit -= cost

            self.balance += raw_profit
            self.total_profit += raw_profit
            reward += np.clip(raw_profit / self.reward_scale, -1.0, 1.0)
            self.position = None

        next_state = self._get_state() if not done else None

        info = {
            'balance': self.balance,
            'total_profit': self.total_profit,
            'step': self.current_step,
            'trade_count': self.trade_count
        }

        return next_state, reward, done, info


# ============================================
#   POLICY GRADIENT AGENT (z temperature)
# ============================================

class PolicyGradientAgent:
    def __init__(self, state_size=5, action_size=3,
                 learning_rate=0.0001, gamma=0.95,
                 temperature=5.0, epsilon=0.1):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.temperature = temperature
        self.epsilon = epsilon
        self.model = self.build_model(learning_rate)

    def build_model(self, learning_rate):
        model = keras.Sequential([
            layers.Input(shape=(self.state_size,)),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(self.action_size, activation='linear',
                         kernel_initializer=keras.initializers.RandomNormal(stddev=0.01))
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                      loss='mse')
        return model

    def act(self, state, greedy=False):
        logits = self.model.predict(state.reshape(1, -1), verbose=0)[0]
        logits_scaled = logits / self.temperature
        logits_scaled = np.clip(logits_scaled, -2.0, 2.0)

        exp_logits = np.exp(logits_scaled - np.max(logits_scaled))
        probabilities = exp_logits / np.sum(exp_logits)

        if greedy:
            return np.argmax(probabilities)

        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)

        return np.random.choice(self.action_size, p=probabilities)

    def train(self, states, actions, rewards):
        returns = self.compute_returns(rewards, self.gamma)
        returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)

        states = np.array(states)
        actions = np.array(actions)

        entropy_coef = 0.01

        with tf.GradientTape() as tape:
            logits = self.model(states, training=True)
            logits_scaled = logits / self.temperature
            logits_scaled = tf.clip_by_value(logits_scaled, -2.0, 2.0)
            action_probs = tf.nn.softmax(logits_scaled, axis=-1)

            indices = tf.range(len(actions)) * self.action_size + actions
            action_probs_for_actions = tf.gather(tf.reshape(action_probs, [-1]), indices)

            log_probs = tf.math.log(action_probs_for_actions + 1e-8)
            entropy = -tf.reduce_sum(action_probs * tf.math.log(action_probs + 1e-8), axis=1)
            loss = tf.reduce_mean(-(log_probs * returns + entropy_coef * entropy))

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

    def compute_returns(self, rewards, gamma):
        returns = []
        running_return = 0.0
        for i in range(len(rewards) - 1, -1, -1):
            running_return = rewards[i] + gamma * running_return
            returns.append(running_return)
        returns.reverse()
        return np.array(returns)


# ============================================
#   ŁADOWANIE I PRZYGOTOWANIE DANYCH
# ============================================

print("Wczytuję dane WIG20 i DAX...")

# Wczytaj dane
wig20 = pd.read_csv('data/PL20.proM1.csv')
dax = pd.read_csv('data/DE30.proM1.csv')

# Parsuj datetime
wig20['datetime'] = pd.to_datetime(wig20['datetime'])
dax['datetime'] = pd.to_datetime(dax['datetime'])

# Posortuj
wig20 = wig20.sort_values('datetime').reset_index(drop=True)
dax = dax.sort_values('datetime').reset_index(drop=True)

print(f"WIG20: {len(wig20)} wierszy, {wig20['datetime'].min()} - {wig20['datetime'].max()}")
print(f"DAX: {len(dax)} wierszy, {dax['datetime'].min()} - {dax['datetime'].max()}")

# ============================================
#   MERGE DANYCH (po datetime)
# ============================================

print("\nŁączę dane WIG20 i DAX...")

df = pd.merge(
    wig20[['datetime', 'open', 'high', 'low', 'close', 'volume']],
    dax[['datetime', 'close']],
    on='datetime',
    suffixes=('_wig20', '_dax'),
    how='inner'
)

print(f"Po merge: {len(df)} wierszy")

df = df.rename(columns={
    'open_wig20': 'wig20_open',
    'high_wig20': 'wig20_high',
    'low_wig20': 'wig20_low',
    'close_wig20': 'wig20_close',
    'volume_wig20': 'wig20_volume',
    'close_dax': 'dax_close'
})

df = df.set_index('datetime').sort_index()
df['wig20_close_raw'] = df['wig20_close'].copy()

print("\nPrzykładowe dane po merge:")
print(df.head())

# ============================================
#   FEATURE ENGINEERING - ARBITRAŻ
# ============================================

print("\nTworzę features arbitrażowe...")

# 1. PODSTAWOWE FEATURES WIG20
df['wig20_returns'] = df['wig20_close'].pct_change() * 100
df['wig20_sma_5'] = df['wig20_close'].rolling(window=5).mean()
df['wig20_sma_15'] = df['wig20_close'].rolling(window=15).mean()
df['wig20_sma_60'] = df['wig20_close'].rolling(window=60).mean()

df['wig20_price_to_sma5'] = (df['wig20_close'] / df['wig20_sma_5'] - 1) * 100
df['wig20_price_to_sma15'] = (df['wig20_close'] / df['wig20_sma_15'] - 1) * 100
df['wig20_price_to_sma60'] = (df['wig20_close'] / df['wig20_sma_60'] - 1) * 100

# 2. PODSTAWOWE FEATURES DAX
df['dax_returns'] = df['dax_close'].pct_change() * 100
df['dax_sma_5'] = df['dax_close'].rolling(window=5).mean()
df['dax_sma_15'] = df['dax_close'].rolling(window=15).mean()
df['dax_sma_60'] = df['dax_close'].rolling(window=60).mean()

df['dax_price_to_sma5'] = (df['dax_close'] / df['dax_sma_5'] - 1) * 100
df['dax_price_to_sma15'] = (df['dax_close'] / df['dax_sma_15'] - 1) * 100
df['dax_price_to_sma60'] = (df['dax_close'] / df['dax_sma_60'] - 1) * 100

# 3. ARBITRAŻ FEATURES - TO JEST KLUCZOWE! 🎯
df['wig20_normalized'] = (df['wig20_close'] / df['wig20_close'].iloc[0]) * 100
df['dax_normalized'] = (df['dax_close'] / df['dax_close'].iloc[0]) * 100

df['spread'] = df['wig20_normalized'] - df['dax_normalized']
df['spread_sma'] = df['spread'].rolling(window=30).mean()
df['spread_std'] = df['spread'].rolling(window=30).std()
df['spread_zscore'] = (df['spread'] - df['spread_sma']) / (df['spread_std'] + 1e-8)


def rolling_correlation(series1, series2, window):
    """Oblicza korelację kroczącą między dwoma seriami"""
    return series1.rolling(window).corr(series2)


df['correlation_30'] = rolling_correlation(df['wig20_returns'], df['dax_returns'], 30)
df['correlation_60'] = rolling_correlation(df['wig20_returns'], df['dax_returns'], 60)

df['dax_returns_lag1'] = df['dax_returns'].shift(1)
df['dax_returns_lag2'] = df['dax_returns'].shift(2)
df['dax_returns_lag3'] = df['dax_returns'].shift(3)

df['price_ratio'] = df['wig20_close'] / df['dax_close']
df['price_ratio_sma'] = df['price_ratio'].rolling(window=30).mean()
df['price_ratio_deviation'] = (df['price_ratio'] / df['price_ratio_sma'] - 1) * 100

df['wig20_momentum'] = df['wig20_returns'].rolling(window=10).mean()
df['dax_momentum'] = df['dax_returns'].rolling(window=10).mean()
df['momentum_divergence'] = df['wig20_momentum'] - df['dax_momentum']

df['wig20_volatility'] = df['wig20_returns'].rolling(window=20).std()
df['dax_volatility'] = df['dax_returns'].rolling(window=20).std()
df['volatility_ratio'] = df['wig20_volatility'] / (df['dax_volatility'] + 1e-8)

df.dropna(inplace=True)

print(f"Po utworzeniu features: {len(df)} wierszy")

# ============================================
#   WYBIERZ FEATURES DO MODELU
# ============================================

features = [
    'wig20_close',
    'wig20_returns',
    'wig20_price_to_sma5',
    'wig20_price_to_sma15',
    'wig20_price_to_sma60',
    'dax_close',
    'dax_returns',
    'dax_price_to_sma5',
    'dax_price_to_sma15',
    'dax_price_to_sma60',
    'spread',
    'spread_zscore',
    'correlation_30',
    'correlation_60',
    'dax_returns_lag1',
    'dax_returns_lag2',
    'dax_returns_lag3',
    'price_ratio_deviation',
    'momentum_divergence',
    'volatility_ratio'
]

print(f"\nUżywam {len(features)} features:")
for f in features:
    print(f"  - {f}")

# ============================================
#   PODZIAŁ DANYCH + SKALOWANIE
# ============================================

total_len = len(df)
train_end = int(total_len * 0.70)
val_end = int(total_len * 0.85)

cols_for_env = features + ['wig20_close_raw']

train_raw = df.iloc[:train_end][cols_for_env].copy()
val_raw = df.iloc[train_end:val_end][cols_for_env].copy()
test_raw = df.iloc[val_end:][cols_for_env].copy()

features_to_scale = [f for f in features]

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
print("Podział danych:")
print(f"Train: {len(train_data)} minut ({len(train_data) / 60:.1f} godzin)")
print(f"Val:   {len(val_data)} minut ({len(val_data) / 60:.1f} godzin)")
print(f"Test:  {len(test_data)} minut ({len(test_data) / 60:.1f} godzin)")
print(f"{'=' * 50}\n")

# ============================================
#   STWÓRZ AGENTA I ENVIRONMENTS
# ============================================

state_size = len(features) + 4
agent = PolicyGradientAgent(
    state_size=state_size,
    action_size=3,
    learning_rate=0.0001,
    temperature=5.0,
    epsilon=0.1
)

print(f"Agent stworzony: state_size={state_size}")

MAX_EPISODE_STEPS = 500

train_env = ArbitrageEnvironment(
    train_data,
    initial_balance=10000,
    position_size=10,
    max_episode_steps=MAX_EPISODE_STEPS,
    random_start=True,
    features=features,
    reward_scale=500.0,
    transaction_cost=0.0001
)

val_env = ArbitrageEnvironment(
    val_data,
    initial_balance=10000,
    position_size=10,
    max_episode_steps=None,
    random_start=False,
    features=features,
    reward_scale=500.0,
    transaction_cost=0.0001
)

test_env = ArbitrageEnvironment(
    test_data,
    initial_balance=10000,
    position_size=10,
    max_episode_steps=None,
    random_start=False,
    features=features,
    reward_scale=500.0,
    transaction_cost=0.0001
)


def test_agent(agent, env, n_runs=1):  # Zmienione z 3 na 1 dla szybkości
    """Testuje agenta N razy i zwraca średnią"""
    rewards = []
    balances = []
    trade_counts = []

    for _ in range(n_runs):
        state = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action = agent.act(state, greedy=False)
            next_state, reward, done, info = env.step(action)
            if not done:
                state = next_state
            total_reward += reward

        rewards.append(total_reward)
        balances.append(info['balance'])
        trade_counts.append(info['trade_count'])

    return np.mean(rewards), np.mean(balances), np.mean(trade_counts)


# ============================================
#   TRENING Z BATCH UPDATES
# ============================================

print("Rozpoczynam trening arbitrażu WIG20-DAX...\n")

episodes = 100
BATCH_SIZE = 10  # Trenuj co 10 epizodów
MAX_BUFFER_STEPS = 5000  # Safety limit

best_val_reward = -float('inf')

# Bufory na doświadczenia
batch_states = []
batch_actions = []
batch_rewards = []

rewards_history = []
val_rewards_history = []
val_balance_history = []

for episode in range(episodes):
    state = train_env.reset()

    # Pokaż prawdopodobieństwa tylko co BATCH_SIZE epizodów
    if (episode + 1) % BATCH_SIZE == 0:
        first_logits = agent.model.predict(state.reshape(1, -1), verbose=0)[0]
        first_logits_scaled = first_logits / agent.temperature
        first_logits_scaled = np.clip(first_logits_scaled, -2.0, 2.0)
        exp_logits = np.exp(first_logits_scaled - np.max(first_logits_scaled))
        first_probs = exp_logits / np.sum(exp_logits)

        tqdm.write(
            f"\nBatch {(episode + 1) // BATCH_SIZE} - "
            f"Prawdopodobieństwa: HOLD={first_probs[0]:.3f}, BUY={first_probs[1]:.3f}, SELL={first_probs[2]:.3f}"
        )

    total_reward = 0.0
    done = False
    states = []
    actions = []
    rewards_ep = []
    action_counts = {0: 0, 1: 0, 2: 0}
    last_info = None

    ep_len = train_env.max_episode_steps or len(train_data)

    # Zbieraj doświadczenia (BEZ treningu!)
    with tqdm(total=ep_len, desc=f"Episode {episode + 1}/{episodes}",
              leave=False, position=0, file=sys.stdout, mininterval=0.5) as pbar:
        while not done:
            action = agent.act(state)
            action_counts[action] += 1
            next_state, reward, done, info = train_env.step(action)

            last_info = info
            states.append(state)
            actions.append(action)
            rewards_ep.append(reward)
            state = next_state if not done else state
            total_reward += reward

            pbar.update(1)
            pbar.set_postfix({'reward': f'{total_reward:.2f}'})

    # Dodaj do bufora batch'a
    batch_states.extend(states)
    batch_actions.extend(actions)
    batch_rewards.extend(rewards_ep)

    rewards_history.append(total_reward)
    train_balance = last_info['balance'] if last_info else train_env.balance
    train_trades = last_info['trade_count'] if last_info else 0

    # Safety: jeśli bufor za duży - trenuj wcześniej
    if len(batch_states) > MAX_BUFFER_STEPS:
        tqdm.write(f"⚠️ Bufor: {len(batch_states)} kroków - trenuję wcześniej!")
        agent.train(batch_states, batch_actions, batch_rewards)
        batch_states = []
        batch_actions = []
        batch_rewards = []

    # TRENUJ co BATCH_SIZE epizodów
    if (episode + 1) % BATCH_SIZE == 0:
        batch_num = (episode + 1) // BATCH_SIZE

        tqdm.write(f"\n{'=' * 60}")
        tqdm.write(f"🔄 BATCH {batch_num}/{episodes // BATCH_SIZE}")
        tqdm.write(f"   Trenuję na {len(batch_states)} krokach...")

        # Trenuj
        agent.train(batch_states, batch_actions, batch_rewards)

        # Wyczyść bufor
        batch_states = []
        batch_actions = []
        batch_rewards = []

        # Statystyki z ostatnich BATCH_SIZE epizodów
        recent_rewards = rewards_history[-BATCH_SIZE:]
        avg_reward = np.mean(recent_rewards)

        tqdm.write(f"   Średni Train Reward ({BATCH_SIZE} ep): {avg_reward:.2f}")
        tqdm.write(f"   Ostatni Train Balance: {train_balance:.2f}")
        tqdm.write(f"   Ostatnie Trades: {train_trades}")

        # Walidacja
        val_reward, val_balance, val_trades = test_agent(agent, val_env, n_runs=1)
        val_rewards_history.append(val_reward)
        val_balance_history.append(val_balance)

        tqdm.write(
            f"   Val Reward: {val_reward:.2f}, Balance: {val_balance:.2f}, Trades: {val_trades:.0f}")

        # Zapisz najlepszy model
        if val_reward > best_val_reward:
            best_val_reward = val_reward
            agent.model.save('best_arbitrage_wig20_dax.keras')
            tqdm.write(f"   ✅ Nowy najlepszy model! Val Reward: {val_reward:.2f}")

        tqdm.write(f"{'=' * 60}\n")

        # Decay epsilona
        agent.epsilon = max(0.01, agent.epsilon * 0.99)

print("\n✓ Trening zakończony!")

# ============================================
#   WYKRESY
# ============================================

plt.figure(figsize=(15, 10))

# 1. Training Reward
plt.subplot(3, 2, 1)
plt.plot(rewards_history)
plt.title('Training Reward per Episode')
plt.xlabel('Episode')
plt.ylabel('Reward (scaled)')
plt.grid(True, alpha=0.3)

# 2. Validation Reward
plt.subplot(3, 2, 2)
if len(val_rewards_history) > 0:
    batch_indices = [i * BATCH_SIZE for i in range(1, len(val_rewards_history) + 1)]
    plt.plot(batch_indices, val_rewards_history, 'o-')
    plt.title('Validation Reward (per batch)')
    plt.xlabel('Episode')
    plt.ylabel('Val Reward (scaled)')
    plt.grid(True, alpha=0.3)

# 3. Validation Balance
plt.subplot(3, 2, 3)
if len(val_balance_history) > 0:
    batch_indices = [i * BATCH_SIZE for i in range(1, len(val_balance_history) + 1)]
    plt.plot(batch_indices, val_balance_history, 'o-', color='green')
    plt.axhline(y=10000, color='r', linestyle='--', label='Initial Balance')
    plt.title('Validation Balance (per batch)')
    plt.xlabel('Episode')
    plt.ylabel('Balance (PLN)')
    plt.legend()
    plt.grid(True, alpha=0.3)

# 4. Moving Average Rewards
plt.subplot(3, 2, 4)
window = 10
if len(rewards_history) >= window:
    ma_rewards = pd.Series(rewards_history).rolling(window=window).mean()
    plt.plot(ma_rewards, label=f'MA-{window}')
    plt.title(f'Training Reward (MA-{window})')
    plt.xlabel('Episode')
    plt.ylabel('MA Reward')
    plt.grid(True, alpha=0.3)
    plt.legend()

# 5. Spread Z-Score
plt.subplot(3, 2, 5)
sample = df['spread_zscore'].iloc[-1000:]
plt.plot(sample.values)
plt.axhline(y=2, color='r', linestyle='--', alpha=0.5, label='Overbought (+2σ)')
plt.axhline(y=-2, color='g', linestyle='--', alpha=0.5, label='Oversold (-2σ)')
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
plt.title('Spread Z-Score (ostatnie 1000min)')
plt.xlabel('Czas')
plt.ylabel('Z-Score')
plt.legend()
plt.grid(True, alpha=0.3)

# 6. Korelacja WIG20-DAX
plt.subplot(3, 2, 6)
sample_corr = df['correlation_30'].iloc[-1000:]
plt.plot(sample_corr.values)
plt.title('Korelacja WIG20-DAX (30-minutowa)')
plt.xlabel('Czas')
plt.ylabel('Correlation')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('arbitrage_wig20_dax_results.png', dpi=150)
print("\n✓ Wykres zapisany jako 'arbitrage_wig20_dax_results.png'\n")

# ============================================
#   TEST
# ============================================

print(f"\n{'=' * 50}")
print("TEST NA DANYCH TESTOWYCH")
print(f"{'=' * 50}\n")

test_reward, test_balance, test_trades = test_agent(agent, test_env, n_runs=5)

print(f"Test Reward (avg, scaled): {test_reward:.2f}")
print(f"Test Balance (avg): {test_balance:.2f}")
print(f"Test Profit (avg): {test_balance - 10000:.2f} PLN")
print(f"Test Trades (avg): {test_trades:.0f}")
if test_trades > 0:
    print(f"Profit per Trade: {(test_balance - 10000) / test_trades:.2f} PLN")
print(f"{'=' * 50}\n")

# ============================================
#   ANALIZA STRATEGII
# ============================================

print("ANALIZA STRATEGII ARBITRAŻOWEJ:")
print("=" * 50)
print("\nKluczowe features dla arbitrażu:")
print("1. spread_zscore - najważniejszy! Pokazuje czy WIG20 jest przewartościowany/niedowartościowany względem DAX")
print("2. dax_returns_lag1/2/3 - lead-lag, czy ruch DAX wyprzedza WIG20")
print("3. correlation_30/60 - jak silnie rynki się poruszają razem")
print("4. momentum_divergence - czy momentum rynków się rozjechało")
print("\nInterpretacja spread_zscore:")
print("  > +2σ: WIG20 zbyt drogi względem DAX → SELL WIG20")
print("  < -2σ: WIG20 zbyt tani względem DAX → BUY WIG20")
print("  ~0: brak możliwości arbitrażu → HOLD")

print(f"\n{'=' * 50}")
print("✓ Trening zakończony!")
print(f"Best Val Reward: {best_val_reward:.2f}")
print(f"Test Reward: {test_reward:.2f}")
print(f"Model zapisany: best_arbitrage_wig20_dax.keras")
print(f"{'=' * 50}")