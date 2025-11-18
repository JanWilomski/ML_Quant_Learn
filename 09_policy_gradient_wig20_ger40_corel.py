from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tqdm import tqdm
import os
import tensorflow as tf


# ============================================
#   ENVIRONMENT
# ============================================

class TradingEnvironment:
    def __init__(self, data, initial_balance=10000, position_size=10,
                 max_episode_steps=None, random_start=False, features=None,
                 reward_scale=500.0):
        """
        data: DataFrame z kolumnami:
              - 'close_raw'  (surowa cena do PnL)
              - zeskalowane feature'y (np. 'close', 'returns', RSI itd.)
        features: lista nazw kolumn, które mają iść do sieci (bez 'close_raw')
        position_size: wielkość pozycji (np. 10 pkt * 10 PLN = 100 PLN / punkt)
        reward_scale: skala do normalizacji nagrody (profit / reward_scale)
        """
        self.data = data.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.position_size = position_size
        self.n_steps = len(data)
        self.max_episode_steps = max_episode_steps
        self.random_start = random_start
        self.reward_scale = reward_scale

        if features is None:
            self.features = [c for c in self.data.columns if c != 'close_raw']
        else:
            self.features = features

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
        current_price = current_row['close_raw']

        if self.position is not None:
            has_position = 1
            entry_price = self.position['entry_price']
            position_pnl = current_price - entry_price
        else:
            has_position = 0
            entry_price = current_price
            position_pnl = 0.0

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
        current_price = self.data.iloc[self.current_step]['close_raw']

        # HOLD
        if action == 0:
            pass

        # BUY
        elif action == 1:
            if self.position is None:
                self.position = {
                    'entry_price': current_price,
                    'entry_step': self.current_step,
                    'type': 'long'
                }
            elif self.position['type'] == 'short':
                profit_per_unit = self.position['entry_price'] - current_price
                raw_profit = profit_per_unit * self.position_size
                self.balance += raw_profit
                self.total_profit += raw_profit
                reward = np.clip(raw_profit / self.reward_scale, -1.0, 1.0)
                self.position = None

        # SELL
        elif action == 2:
            if self.position is None:
                self.position = {
                    'entry_price': current_price,
                    'entry_step': self.current_step,
                    'type': 'short'
                }
            elif self.position['type'] == 'long':
                profit_per_unit = current_price - self.position['entry_price']
                raw_profit = profit_per_unit * self.position_size
                self.balance += raw_profit
                self.total_profit += raw_profit
                reward = np.clip(raw_profit / self.reward_scale, -1.0, 1.0)
                self.position = None

        self.current_step += 1

        done_data = (self.current_step >= self.n_steps)
        done_length = False
        if self.max_episode_steps is not None:
            done_length = (self.current_step >= self.episode_start + self.max_episode_steps)
        done_bankrupt = (self.balance <= 0)

        done = done_data or done_length or done_bankrupt

        # auto-close pozycji na końcu
        if done and self.position is not None:
            current_price = self.data.iloc[self.current_step - 1]['close_raw']
            if self.position['type'] == 'long':
                profit_per_unit = current_price - self.position['entry_price']
            else:
                profit_per_unit = self.position['entry_price'] - current_price
            raw_profit = profit_per_unit * self.position_size
            self.balance += raw_profit
            self.total_profit += raw_profit
            reward += np.clip(raw_profit / self.reward_scale, -1.0, 1.0)
            self.position = None

        next_state = self._get_state() if not done else None
        info = {
            'balance': self.balance,
            'total_profit': self.total_profit,
            'step': self.current_step
        }
        return next_state, reward, done, info


# ============================================
#   POLICY GRADIENT AGENT
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
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(self.action_size, activation='linear',
                         kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
                         bias_initializer=keras.initializers.Zeros())
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

        # epsilon-greedy nad policy
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)

        return np.random.choice(self.action_size, p=probabilities)

    def train(self, states, actions, rewards):
        returns = self.compute_returns(rewards, self.gamma)
        returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)

        states = np.array(states)
        actions = np.array(actions)

        entropy_coef = 0.1

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
#   DANE WIG20 + FEATURE ENGINEERING
# ============================================

print("Wczytuję dane WIG20...")
df = pd.read_csv('data/wig20_h1_11_04_2024-17_11_2025.csv')

df['datetime'] = pd.to_datetime(
    df['date'].astype(str) + df['time'].astype(str).str.zfill(6),
    format='%Y%m%d%H%M%S'
)
df = df.set_index('datetime').sort_index()

df = df[['open', 'high', 'low', 'close', 'volume']]
df['close_raw'] = df['close']  # surowa cena dla PnL

print(f"Wczytano {len(df)} wierszy")
print(f"Okres: {df.index[0]} - {df.index[-1]}")
print("\nPrzykładowe dane:")
print(df.head())

print("\nTworzę features...")

df['returns'] = df['close'].pct_change() * 100

df['SMA_7'] = df['close'].rolling(window=7).mean()
df['SMA_14'] = df['close'].rolling(window=14).mean()
df['SMA_30'] = df['close'].rolling(window=30).mean()

df['price_to_SMA_7'] = (df['close'] / df['SMA_7'] - 1) * 100
df['price_to_SMA_14'] = (df['close'] / df['SMA_14'] - 1) * 100
df['price_to_SMA_30'] = (df['close'] / df['SMA_30'] - 1) * 100

df['SMA_7_return'] = df['SMA_7'].pct_change() * 100
df['SMA_14_return'] = df['SMA_14'].pct_change() * 100
df['SMA_30_return'] = df['SMA_30'].pct_change() * 100

df['volume_ma'] = df['volume'].rolling(window=20).mean()
df['volume_ratio'] = df['volume'] / df['volume_ma']


def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


df['RSI'] = calculate_rsi(df['close'])

df['Sentiment'] = pd.cut(df['returns'], bins=[-100, -0.5, 0.5, 100],
                         labels=[0, 1, 2]).astype(float)

df.dropna(inplace=True)

print(f"Po utworzeniu features: {len(df)} wierszy")

features = [
    'close',
    'returns',
    'SMA_7', 'SMA_14', 'SMA_30',
    'price_to_SMA_7', 'price_to_SMA_14', 'price_to_SMA_30',
    'SMA_7_return', 'SMA_14_return', 'SMA_30_return',
    'volume',
    'volume_ratio',
    'RSI',
    'Sentiment'
]

print(f"\nUżywam {len(features)} features:")
print(features)

# ============================================
#   PODZIAŁ DANYCH + SKALOWANIE
# ============================================

total_len = len(df)
train_end = int(total_len * 0.70)
val_end = int(total_len * 0.85)

cols_for_env = features + ['close_raw']

train_raw = df.iloc[:train_end][cols_for_env].copy()
val_raw = df.iloc[train_end:val_end][cols_for_env].copy()
test_raw = df.iloc[val_end:][cols_for_env].copy()

features_to_scale = [f for f in features if f != 'close']  # 'close' możesz też skalować, jak chcesz

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
print(f"Train: {len(train_data)} godzin ({len(train_data) / 24:.1f} dni)")
print(f"Val:   {len(val_data)} godzin ({len(val_data) / 24:.1f} dni)")
print(f"Test:  {len(test_data)} godzin ({len(test_data) / 24:.1f} dni)")
print(f"{'=' * 50}\n")

print("Kolumny train_data:", train_data.columns.tolist())

# ============================================
#   AGENT + ENV
# ============================================

state_size = len(features) + 4  # + info o pozycji
agent = PolicyGradientAgent(
    state_size=state_size,
    action_size=3,
    temperature=5.0,
    epsilon=0.1
)

print(f"Agent stworzony: state_size={state_size}")

MAX_EPISODE_STEPS = 500

train_env = TradingEnvironment(
    train_data,
    initial_balance=10000,
    position_size=10,
    max_episode_steps=MAX_EPISODE_STEPS,
    random_start=True,
    features=features,
    reward_scale=500.0
)

val_env = TradingEnvironment(
    val_data,
    initial_balance=10000,
    position_size=10,
    max_episode_steps=None,
    random_start=False,
    features=features,
    reward_scale=500.0
)

test_env = TradingEnvironment(
    test_data,
    initial_balance=10000,
    position_size=10,
    max_episode_steps=None,
    random_start=False,
    features=features,
    reward_scale=500.0
)


def test_agent(agent, env, n_runs=3):
    """Testuje agenta N razy i zwraca średnią (reward, balance, trade_count)"""
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
        trade_counts.append(info.get('trade_count', 0))  # Safe get

    return np.mean(rewards), np.mean(balances), np.mean(trade_counts)

# ============================================
#   TRENING
# ============================================

print("Rozpoczynam trening...\n")

# ============================================
#   TRENING Z BATCH_SIZE = 10
# ============================================

episodes = 100
BATCH_SIZE = 10  # Trenuj co 10 epizodów
MAX_EPISODE_STEPS = 300  # Zmniejsz z 500 żeby nie zabrakło RAM

# Bufor na doświadczenia
batch_states = []
batch_actions = []
batch_rewards = []

rewards_history = []
val_rewards_history = []
best_val_reward = -float('inf')

for episode in range(episodes):
    state = train_env.reset()

    # Pokaż prawdopodobieństwa tylko co 10 epizodów (mniej clutter)
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
    with tqdm(total=ep_len, desc=f"Episode {episode + 1}/{episodes}", leave=False) as pbar:
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
    train_trades = last_info.get('trade_count', 0) if last_info else 0

    # TRENUJ tylko co BATCH_SIZE epizodów
    if (episode + 1) % BATCH_SIZE == 0:
        batch_num = (episode + 1) // BATCH_SIZE

        tqdm.write(f"\n{'=' * 60}")
        tqdm.write(f"🔄 BATCH {batch_num}/{episodes // BATCH_SIZE}")
        tqdm.write(f"   Trenuję na {len(batch_states)} krokach...")

        # Trenuj
        agent.train(batch_states, batch_actions, batch_rewards)

        # Statystyki z ostatnich 10 epizodów
        recent_rewards = rewards_history[-BATCH_SIZE:]
        avg_reward = np.mean(recent_rewards)

        tqdm.write(f"   Średni Train Reward (10 ep): {avg_reward:.2f}")
        tqdm.write(f"   Ostatni Train Balance: {train_balance:.2f}")
        tqdm.write(f"   Ostatnie Trades: {train_trades}")

        # Wyczyść bufor
        batch_states = []
        batch_actions = []
        batch_rewards = []

        # Walidacja (tylko 1 run żeby przyspieszyć)
        val_reward, val_balance, val_trades = test_agent(agent, val_env, n_runs=1)
        val_rewards_history.append(val_reward)

        tqdm.write(f"   Val Reward: {val_reward:.2f}, Balance: {val_balance:.2f}, Trades: {val_trades:.0f}")

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

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(rewards_history)
plt.title('Training Reward per Episode')
plt.xlabel('Episode')
plt.ylabel('Total Reward (scaled)')
plt.grid(True)

plt.subplot(1, 3, 2)
if len(val_rewards_history) > 0:
    plt.plot(range(1, len(val_rewards_history) + 1), val_rewards_history, 'o-')
    plt.title('Validation Reward (avg of 3 runs)')
    plt.xlabel('Episode')
    plt.ylabel('Val Reward (scaled)')
    plt.grid(True)

plt.subplot(1, 3, 3)
window = 10
if len(rewards_history) >= window:
    ma_rewards = pd.Series(rewards_history).rolling(window=window).mean()
    plt.plot(ma_rewards)
    plt.title(f'Training Reward (MA-{window})')
    plt.xlabel('Episode')
    plt.ylabel('MA Reward (scaled)')
    plt.grid(True)

plt.tight_layout()
plt.savefig('pg_wig20_results.png')
print("\n✓ Wykres zapisany jako 'pg_wig20_results.png'\n")

# ============================================
#   TEST
# ============================================

print(f"\n{'=' * 50}")
print("TEST NA DANYCH TESTOWYCH (avg of 5 runs)")
print(f"{'=' * 50}\n")

test_reward, test_balance = test_agent(agent, test_env, n_runs=5)

print(f"Test Reward (avg, scaled): {test_reward:.2f}")
print(f"Test Balance (avg): {test_balance:.2f}")
print(f"Test Profit (avg, nominal): {test_balance - 10000:.2f}")
print(f"{'=' * 50}\n")

print("✓ Trening zakończony!")
print("\nPodsumowanie:")
print(f"- Best Val Reward (scaled): {best_val_reward:.2f}")
print(f"- Test Reward (scaled): {test_reward:.2f}")
print(f"- Model zapisany jako: best_pg_wig20.keras")
