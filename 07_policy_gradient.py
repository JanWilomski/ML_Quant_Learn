from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tqdm import tqdm
import os
import tensorflow as tf


class TradingEnvironment:
    def __init__(self, data, initial_balance=10000, position_size=10000,
                 max_episode_steps=None, random_start=False):
        """
        data: DataFrame z kolumną 'close'
        initial_balance: startowy kapitał
        max_episode_steps: maks. liczba kroków w jednym epizodzie (np. 500)
        random_start: jeśli True – epizod startuje w losowym miejscu w danych
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
        """Resetuje environment na początek epizodu (niekoniecznie od początku danych)"""
        # Ustal początek epizodu
        if self.random_start and self.max_episode_steps is not None and self.max_episode_steps < self.n_steps:
            # Możemy zacząć gdzieś tak, żeby starczyło danych na cały epizod
            self.episode_start = np.random.randint(0, self.n_steps - self.max_episode_steps)
        else:
            # Standardowo od początku
            self.episode_start = 0

        self.current_step = self.episode_start
        self.balance = self.initial_balance
        self.position = None
        self.total_profit = 0

        return self._get_state()

    def _get_state(self):
        """Zwraca state dla agenta"""
        current_row = self.data.iloc[self.current_step]
        current_price = current_row['close']

        if self.position is not None:
            has_position = 1
            entry_price = self.position['entry_price']
            position_pnl = current_price - entry_price
        else:
            has_position = 0
            entry_price = current_price  # żeby liczenie względne było stabilne
            position_pnl = 0.0

        # --- NORMALIZACJA INFORMACJI O POZYCJI ---

        # cena wejścia względnie do bieżącej (np. +0.5% → 0.005)
        entry_price_rel = (entry_price / current_price) - 1.0

        # PnL na 1 jednostkę jako % ceny
        pnl_rel = position_pnl / current_price

        # balans jako % zmiany względem startu (np. +5% → 0.05)
        balance_rel = (self.balance - self.initial_balance) / self.initial_balance

        # wszystko jest w okolicach [-1, 1] zamiast 10000
        position_info = np.array([
            has_position,
            entry_price_rel,
            pnl_rel,
            balance_rel
        ], dtype=np.float32)

        features_array = current_row.values.astype(np.float32)
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
                self.position = {
                    'entry_price': current_price,
                    'entry_step': self.current_step,
                    'type': 'long'
                }
            elif self.position['type'] == 'short':
                profit_per_unit = self.position['entry_price'] - current_price
                profit = profit_per_unit * self.position_size
                self.balance += profit
                self.total_profit += profit
                reward = profit
                self.position = None

        # SELL (2)
        elif action == 2:
            if self.position is None:
                self.position = {
                    'entry_price': current_price,
                    'entry_step': self.current_step,
                    'type': 'short'
                }
            elif self.position['type'] == 'long':
                profit_per_unit = current_price - self.position['entry_price']
                profit = profit_per_unit * self.position_size
                self.balance += profit
                self.total_profit += profit
                reward = profit
                self.position = None

        self.current_step += 1

        # 1) koniec danych
        done_data = (self.current_step >= self.n_steps)
        # 2) koniec epizodu po max_episode_steps
        if self.max_episode_steps is not None:
            done_length = (self.current_step >= self.episode_start + self.max_episode_steps)
        else:
            done_length = False
        # 3) bankructwo
        done_bankrupt = (self.balance <= 0)

        done = done_data or done_length or done_bankrupt

        # Auto-close pozycji jeśli done
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

        info = {
            'balance': self.balance,
            'total_profit': self.total_profit,
            'step': self.current_step
        }

        return next_state, reward, done, info


class PolicyGradientAgent:
    def __init__(self, state_size=5, action_size=3, learning_rate=0.0001, gamma=0.95, temperature=1.0):
        """
        Policy Gradient Agent z temperature scaling

        temperature: kontroluje jak "pewny" jest agent
            - Wysoka (5-10): bardzo losowe akcje (dobra eksploracja)
            - Niska (1-2): pewne akcje (mała eksploracja)
            - 3.0: dobry balans
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.temperature = temperature
        self.model = self.build_model(learning_rate)

    def build_model(self, learning_rate):
        """
        UWAGA: Ostatnia warstwa ma activation='linear', nie 'softmax'!
        Softmax aplikujemy ręcznie z temperature w act() i train()
        """
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_dim=self.state_size),
            layers.Dense(32, activation='relu'),
            layers.Dense(self.action_size, activation='linear',  # ← LINEAR!
                         kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
                         bias_initializer=keras.initializers.Zeros())
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                      loss='mse')
        return model

    def act(self, state, greedy=False):
        logits = self.model.predict(state.reshape(1, -1), verbose=0)[0]
        logits_scaled = logits / self.temperature

        # Przytnij logity – zabezpieczenie przed ekstremami
        logits_scaled = np.clip(logits_scaled, -5.0, 5.0)

        exp_logits = np.exp(logits_scaled - np.max(logits_scaled))
        probabilities = exp_logits / np.sum(exp_logits)

        if greedy:
            return np.argmax(probabilities)
        return np.random.choice(self.action_size, p=probabilities)

    def train(self, states, actions, rewards):
        """Policy Gradient z temperature scaling"""
        # 1. Oblicz returns
        returns = self.compute_returns(rewards, self.gamma)

        # 2. Normalizacja
        returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)

        # 3. Konwertuj
        states = np.array(states)
        actions = np.array(actions)

        # 4. Policy Gradient
        with tf.GradientTape() as tape:
            logits = self.model(states, training=True)

            logits_scaled = logits / self.temperature
            logits_scaled = tf.clip_by_value(logits_scaled, -5.0, 5.0)

            action_probs = tf.nn.softmax(logits_scaled, axis=-1)


            # Wybierz prawdopodobieństwa dla wykonanych akcji
            indices = tf.range(len(actions)) * self.action_size + actions
            action_probs_for_actions = tf.gather(tf.reshape(action_probs, [-1]), indices)

            # Policy Gradient Loss
            log_probs = tf.math.log(action_probs_for_actions + 1e-8)
            entropy = -tf.reduce_sum(action_probs * tf.math.log(action_probs + 1e-8), axis=1)
            loss = -(log_probs * returns + 0.01 * entropy)

        # 5. Gradient descent
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

    def compute_returns(self, rewards, gamma):
        """Oblicza discounted returns"""
        returns = []
        running_return = 0

        for i in range(len(rewards) - 1, -1, -1):
            running_return = rewards[i] + gamma * running_return
            returns.append(running_return)

        returns.reverse()
        return np.array(returns)


# ========== WCZYTAJ I PRZYGOTUJ DANE ==========

df = pd.read_csv('data/eurusd_h1_27_01_2025-17_11_2025.csv', sep=';')
df_vwap = pd.read_csv('data/eurusd_h1_27_01_2025-17_11_2025_vwap.csv', sep=';')
df_bollinger = pd.read_csv('data/eurusd_h1_27_01_2025-17_11_2025_bollinger_bands.csv', sep=';')

df['datetime'] = pd.to_datetime(df['datetime'])
df_vwap['datetime'] = pd.to_datetime(df_vwap['datetime'])
df_bollinger['datetime'] = pd.to_datetime(df_bollinger['datetime'])

df.set_index('datetime', inplace=True)
df.sort_index(inplace=True)

df_vwap.set_index('datetime', inplace=True)
df_vwap.sort_index(inplace=True)

df_bollinger.set_index('datetime', inplace=True)
df_bollinger.sort_index(inplace=True)

# Stwórz features
df['returns'] = df['close'].pct_change() * 100
df['SMA_7'] = df['close'].rolling(window=7).mean()
df['SMA_14'] = df['close'].rolling(window=14).mean()
df['Price_to_SMA_7'] = (df['close'] / df['SMA_7'] - 1) * 100
df['Price_to_SMA_14'] = (df['close'] / df['SMA_14'] - 1) * 100
df['SMA_7_return'] = df['SMA_7'].pct_change() * 100
df['SMA_14_return'] = df['SMA_14'].pct_change() * 100
df['vwap'] = df_vwap['vwap']
df['bollinger_upper'] = df_bollinger['upper_band']
df['bollinger_lower'] = df_bollinger['lower_band']
df['bollinger_middle'] = df_bollinger['middle_band']

df['price_to_bb_upper'] = (df['close'] - df['bollinger_upper']) / df['close'] * 100
df['price_to_bb_lower'] = (df['close'] - df['bollinger_lower']) / df['close'] * 100
df['bb_width'] = (df['bollinger_upper'] - df['bollinger_lower']) / df['close'] * 100

df.dropna(inplace=True)

df['Sentiment'] = pd.cut(df['returns'], bins=[-100, 0, 100], labels=[0, 1]).astype(int)

df.dropna(inplace=True)

features = ['close', 'SMA_7', 'SMA_14', 'SMA_7_return', 'SMA_14_return',
            'Price_to_SMA_7', 'Price_to_SMA_14', 'Sentiment', 'vwap',
            'bollinger_upper', 'bollinger_lower', 'bollinger_middle', 'price_to_bb_upper', 'price_to_bb_lower', 'bb_width']

# NIE skalujemy 'close' – jest potrzebny do liczenia PnL
features_to_scale = [f for f in features if f != 'close']

# Podział na train / val / test
total_len = len(df)
train_end = int(total_len * 0.70)
val_end = int(total_len * 0.85)

train_raw = df.iloc[:train_end][features].copy()
val_raw   = df.iloc[train_end:val_end][features].copy()
test_raw  = df.iloc[val_end:][features].copy()

# NORMALIZUJ tylko wybrane feature'y
scaler = StandardScaler()

train_scaled = train_raw.copy()
val_scaled   = val_raw.copy()
test_scaled  = test_raw.copy()

train_scaled[features_to_scale] = scaler.fit_transform(train_raw[features_to_scale])
val_scaled[features_to_scale]   = scaler.transform(val_raw[features_to_scale])
test_scaled[features_to_scale]  = scaler.transform(test_raw[features_to_scale])

train_data = train_scaled
val_data   = val_scaled
test_data  = test_scaled

print(f"\n{'=' * 50}")
print(f"Podział danych:")
print(f"Train: {len(train_data)} godzin ({len(train_data) / 24:.1f} dni)")
print(f"Val:   {len(val_data)} godzin ({len(val_data) / 24:.1f} dni)")
print(f"Test:  {len(test_data)} godzin ({len(test_data) / 24:.1f} dni)")
print(f"{'=' * 50}\n")


# ========== STWÓRZ AGENTA ==========

state_size = len(features) + 4
agent = PolicyGradientAgent(state_size=state_size, action_size=3, temperature=3.0)

# Sprawdź inicjalizację
test_state = np.random.randn(state_size)
test_probs = agent.model.predict(test_state.reshape(1, -1), verbose=0)[0]
print(f"Test inicjalizacji logitów: {test_probs}")
print(f"Powinny być małe wartości wokół 0\n")

# ========== ENVIRONMENTS ==========

MAX_EPISODE_STEPS = 500

train_env = TradingEnvironment(
    train_data,
    initial_balance=10000,
    position_size=10000,
    max_episode_steps=MAX_EPISODE_STEPS,
    random_start=True
)

# Walidacja: bez random start, bez limitu długości (całe dane)
val_env = TradingEnvironment(
    val_data,
    initial_balance=10000,
    position_size=10000,
    max_episode_steps=None,
    random_start=False
)

# Test: tak samo jak walidacja
test_env = TradingEnvironment(
    test_data,
    initial_balance=10000,
    position_size=10000,
    max_episode_steps=None,
    random_start=False
)


def test_agent(agent, env, n_runs=5):
    """Testuje agenta N razy i zwraca średnią"""
    rewards = []
    balances = []

    for _ in range(n_runs):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.act(state, greedy=False)  # stochastyczny
            next_state, reward, done, info = env.step(action)

            if not done:
                state = next_state

            total_reward += reward

        rewards.append(total_reward)
        balances.append(info['balance'])

    # Zwróć średnią
    return np.mean(rewards), np.mean(balances)


# ========== TRENING ==========

print("Rozpoczynam trening...\n")

episodes = 50  # Więcej epizodów bo policy gradient potrzebuje więcej uczenia
best_val_reward = -float('inf')

rewards_history = []
val_rewards_history = []

for episode in range(episodes):
    state = train_env.reset()


    first_probs_logits = agent.model.predict(state.reshape(1, -1), verbose=0)[0]
    first_probs_logits_scaled = first_probs_logits / agent.temperature
    exp_logits = np.exp(first_probs_logits_scaled - np.max(first_probs_logits_scaled))
    first_probs = exp_logits / np.sum(exp_logits)
    tqdm.write(
        f"Episode {episode + 1} - Prawdopodobieństwa: "
        f"HOLD={first_probs[0]:.3f}, BUY={first_probs[1]:.3f}, SELL={first_probs[2]:.3f}"
    )

    total_reward = 0
    done = False

    states = []
    actions = []
    rewards = []

    action_counts = {0: 0, 1: 0, 2: 0}
    last_info = None

    ep_len = train_env.max_episode_steps or len(train_data)
    with tqdm(total=ep_len, desc=f"Episode {episode + 1}/{episodes}", leave=True) as pbar:

        while not done:
            action = agent.act(state)
            action_counts[action] += 1
            next_state, reward, done, info = train_env.step(action)

            last_info = info

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            state = next_state

            total_reward += reward
            pbar.update(1)
            pbar.set_postfix({'reward': f'{total_reward:.2f}'})

    # Trenuj
    agent.train(states, actions, rewards)
    rewards_history.append(total_reward)

    train_balance = last_info['balance'] if last_info is not None else train_env.balance


    # Validation PO KAŻDYM EPIZODZIE
    val_reward, val_balance = test_agent(agent, val_env)
    val_rewards_history.append(val_reward)

    tqdm.write(f"  Akcje: HOLD={action_counts[0]}, BUY={action_counts[1]}, SELL={action_counts[2]}")
    tqdm.write(
        f"Episode {episode + 1} - "
        f"Train Reward: {total_reward:.2f}, Train Balance: {train_balance:.2f}, "
        f"Val Reward: {val_reward:.2f}, Val Balance: {val_balance:.2f}"
    )
    if val_reward > best_val_reward:
        best_val_reward = val_reward
        agent.model.save('best_pg_agent.keras')
        tqdm.write(f"  ✓ Nowy najlepszy model! Val Reward: {val_reward:.2f}")

# ========== WYKRES ==========

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(rewards_history)
plt.title('Training Reward per Episode')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.grid(True)

plt.subplot(1, 2, 2)
if len(val_rewards_history) > 0:
    plt.plot(range(1, len(val_rewards_history) + 1), val_rewards_history, 'o-')
    plt.title('Validation Reward')
    plt.xlabel('Episode')
    plt.ylabel('Val Reward')
    plt.grid(True)

plt.tight_layout()
plt.savefig('pg_training_results.png')
print("\n✓ Wykres zapisany jako 'pg_training_results.png'\n")

# ========== TEST ==========

print(f"\n{'=' * 50}")
print("TEST NA DANYCH TESTOWYCH")
print(f"{'=' * 50}\n")

test_reward, test_balance = test_agent(agent, test_env)

print(f"Test Reward: {test_reward:.2f}")
print(f"Test Balance: {test_balance:.2f}")
print(f"Test Profit: {test_balance - 10000:.2f}")
print(f"{'=' * 50}\n")

print("✓ Trening zakończony!")
