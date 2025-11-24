"""
A2C (Advantage Actor-Critic) dla tickowych danych WIG20 (co 15 sekund)

🎯 ZOPTYMALIZOWANE DLA TICKÓW:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TICKOWE TIMEFRAMES (15s intervals):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- 1 minuta = 4 ticki
- 5 minut = 20 ticków
- 15 minut = 60 ticków
- 1 godzina = 240 ticków
- Dzień giełdowy (9:00-16:30) = 450 min = 1800 ticków

PARAMETRY ENVIRONMENT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- MAX_EPISODE_STEPS: 480 ticków (2 godziny)
  Dlaczego? Krótsze epizody = stabilniejsze uczenie dla ticków

- position_size: 5 (mniejszy niż minutowe dane)
  Dlaczego? Tickowe ruchy są mniejsze

- reward_scale: 2.0 (mniejszy niż minutowe dane)
  Dlaczego? Tickowe zmiany są delikatniejsze

- transaction_cost: 0.000005 (realistische dla ticków)
  Dlaczego? Większy impact przy częstszym tradingu

FEATURES (TYLKO RELATIVE!):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Returns (pct_change * 100)
✅ Price to SMA (jako % deviation)
✅ SMA returns (momentum)
✅ Volatility (rolling std of returns)
✅ Intraday position (% from open)
✅ Time features (hour_sin, hour_cos)

❌ NIE używamy surowych cen w NN!
❌ Ceny RAW tylko dla PnL calculation

TRAINING PARAMETERS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- episodes: 200 (więcej danych = więcej epizodów)
- BATCH_SIZE: 5 (częste aktualizacje)
- actor_lr: 0.0005
- critic_lr: 0.001
- temperature: 5.0 (wysoka eksploracja)
- epsilon: 0.1 (decay 0.995)
"""

import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tqdm import tqdm
import tensorflow as tf


# ============================================
#   TRADING ENVIRONMENT DLA TICKÓW
# ============================================

class TickTradingEnvironment:
    def __init__(self, data, initial_balance=10000, position_size=5,
                 max_episode_steps=None, random_start=False, features=None,
                 reward_scale=2.0, transaction_cost=0.000005):
        """
        Environment dla tickowych danych (co 15 sekund)

        Args:
            data: DataFrame z kolumnami features + 'price_raw'
            position_size: wielkość pozycji (np. 5 pkt)
            max_episode_steps: długość epizodu w tickach (np. 480 = 2h)
            reward_scale: skala do normalizacji reward
            transaction_cost: koszt transakcji (% ceny)
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
            self.features = [c for c in self.data.columns if c != 'price_raw']
        else:
            self.features = features

        self.current_step = 0
        self.episode_start = 0
        self.balance = initial_balance
        self.position = None
        self.total_profit = 0
        self.trade_count = 0

    def reset(self):
        """Resetuje environment - losowy start dla eksploracji"""
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
        """State = features + position info (wszystko znormalizowane)"""
        current_row = self.data.iloc[self.current_step]
        current_price = current_row['price_raw']

        if self.position is not None:
            has_position = 1
            entry_price = self.position['entry_price']
            position_pnl = current_price - entry_price
        else:
            has_position = 0
            entry_price = current_price
            position_pnl = 0.0

        # Normalizacja position info (wszystko jako % lub relative)
        entry_price_rel = (entry_price / current_price) - 1.0
        pnl_rel = position_pnl / current_price
        balance_rel = (self.balance - self.initial_balance) / self.initial_balance

        # Time in episode (0.0 - 1.0)
        if self.max_episode_steps:
            time_in_episode = (self.current_step - self.episode_start) / self.max_episode_steps
        else:
            time_in_episode = 0.0

        position_info = np.array([
            has_position,
            entry_price_rel,
            pnl_rel,
            balance_rel,
            time_in_episode
        ], dtype=np.float32)

        features_array = current_row[self.features].values.astype(np.float32)
        state = np.concatenate([features_array, position_info])

        return state

    def step(self, action):
        """Wykonuje akcję: 0=HOLD, 1=BUY, 2=SELL"""
        reward = 0.0
        current_price = self.data.iloc[self.current_step]['price_raw']

        # HOLD (0)
        if action == 0:
            # Mała kara za trzymanie pozycji (opportunity cost)
            if self.position is not None:
                reward = -0.0001

        # BUY / Close SHORT (1)
        elif action == 1:
            if self.position is None:
                # Otwórz LONG
                cost = current_price * self.position_size * self.transaction_cost
                self.balance -= cost
                self.position = {
                    'entry_price': current_price,
                    'entry_step': self.current_step,
                    'type': 'long'
                }
                self.trade_count += 1

            elif self.position['type'] == 'short':
                # Zamknij SHORT
                profit_per_unit = self.position['entry_price'] - current_price
                raw_profit = profit_per_unit * self.position_size
                cost = current_price * self.position_size * self.transaction_cost
                raw_profit -= cost

                self.balance += raw_profit
                self.total_profit += raw_profit

                # Reward z bonusem za profit
                base_reward = np.clip(raw_profit / self.reward_scale, -1.0, 1.0)
                if raw_profit > 0:
                    reward = base_reward + 0.05
                else:
                    reward = base_reward - 0.02

                self.position = None
                self.trade_count += 1

        # SELL / Close LONG (2)
        elif action == 2:
            if self.position is None:
                # Otwórz SHORT
                cost = current_price * self.position_size * self.transaction_cost
                self.balance -= cost
                self.position = {
                    'entry_price': current_price,
                    'entry_step': self.current_step,
                    'type': 'short'
                }
                self.trade_count += 1

            elif self.position['type'] == 'long':
                # Zamknij LONG
                profit_per_unit = current_price - self.position['entry_price']
                raw_profit = profit_per_unit * self.position_size
                cost = current_price * self.position_size * self.transaction_cost
                raw_profit -= cost

                self.balance += raw_profit
                self.total_profit += raw_profit

                # Reward z bonusem za profit
                base_reward = np.clip(raw_profit / self.reward_scale, -1.0, 1.0)
                if raw_profit > 0:
                    reward = base_reward + 0.05
                else:
                    reward = base_reward - 0.02

                self.position = None
                self.trade_count += 1

        self.current_step += 1

        # Check if done
        done_data = (self.current_step >= self.n_steps)
        done_length = False
        if self.max_episode_steps:
            done_length = (self.current_step >= self.episode_start + self.max_episode_steps)
        done_bankrupt = (self.balance <= 0)

        done = done_data or done_length or done_bankrupt

        # Auto-close pozycji na końcu epizodu
        if done and self.position is not None:
            current_price = self.data.iloc[self.current_step - 1]['price_raw']

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
#   A2C AGENT
# ============================================

class A2CAgent:
    def __init__(self, state_size=5, action_size=3,
                 actor_lr=0.0005, critic_lr=0.001, gamma=0.95,
                 temperature=5.0, epsilon=0.1):
        """
        A2C Agent z osobnymi sieciami Actor i Critic
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.temperature = temperature
        self.epsilon = epsilon

        # Buduj obie sieci
        self.actor = self.build_actor(actor_lr)
        self.critic = self.build_critic(critic_lr)

        print(f"\n🎭 A2C Agent utworzony:")
        print(f"   State size: {state_size}")
        print(f"   Actor LR: {actor_lr}")
        print(f"   Critic LR: {critic_lr}")
        print(f"   Temperature: {temperature}")
        print(f"   Epsilon: {epsilon}\n")

    def build_actor(self, learning_rate):
        """ACTOR: State → Action probabilities"""
        model = keras.Sequential([
            layers.Input(shape=(self.state_size,)),
            layers.Dense(128, activation='relu',
                         kernel_initializer='he_normal'),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu',
                         kernel_initializer='he_normal'),
            layers.Dense(32, activation='relu',
                         kernel_initializer='he_normal'),
            layers.Dense(self.action_size, activation='linear',
                         kernel_initializer=keras.initializers.RandomNormal(stddev=0.01))
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                      loss='mse')
        return model

    def build_critic(self, learning_rate):
        """CRITIC: State → Value V(s)"""
        model = keras.Sequential([
            layers.Input(shape=(self.state_size,)),
            layers.Dense(128, activation='relu',
                         kernel_initializer='he_normal'),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu',
                         kernel_initializer='he_normal'),
            layers.Dense(32, activation='relu',
                         kernel_initializer='he_normal'),
            layers.Dense(1, activation='linear',
                         kernel_initializer='zeros')
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                      loss='mse')
        return model

    def act(self, state, greedy=False):
        """Wybierz akcję używając Actor network"""
        logits = self.actor.predict(state.reshape(1, -1), verbose=0)[0]
        logits_scaled = logits / self.temperature
        logits_scaled = np.clip(logits_scaled, -2.0, 2.0)

        exp_logits = np.exp(logits_scaled - np.max(logits_scaled))
        probabilities = exp_logits / np.sum(exp_logits)

        if greedy:
            return np.argmax(probabilities)

        # Epsilon-greedy dla eksploracji
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)

        return np.random.choice(self.action_size, p=probabilities)

    def train(self, states, actions, rewards, next_states, dones):
        """
        A2C Training z TD learning
        """
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)

        # Oblicz TD targets i advantages
        values = self.critic.predict(states, verbose=0).flatten()
        next_values = self.critic.predict(next_states, verbose=0).flatten()
        td_targets = rewards + self.gamma * next_values * (1 - dones)
        advantages = td_targets - values

        # Normalizuj advantages
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        # Trenuj Critic
        critic_loss = self.critic.fit(
            states,
            td_targets,
            epochs=1,
            verbose=0,
            batch_size=min(32, len(states))
        ).history['loss'][0]

        # Trenuj Actor
        entropy_coef = 0.01

        with tf.GradientTape() as tape:
            logits = self.actor(states, training=True)
            logits_scaled = logits / self.temperature
            logits_scaled = tf.clip_by_value(logits_scaled, -2.0, 2.0)
            action_probs = tf.nn.softmax(logits_scaled, axis=-1)

            indices = tf.range(len(actions)) * self.action_size + actions
            action_probs_for_actions = tf.gather(tf.reshape(action_probs, [-1]), indices)

            log_probs = tf.math.log(action_probs_for_actions + 1e-8)
            entropy = -tf.reduce_sum(action_probs * tf.math.log(action_probs + 1e-8), axis=1)

            advantages_tf = tf.constant(advantages, dtype=tf.float32)
            actor_loss = tf.reduce_mean(-(log_probs * advantages_tf + entropy_coef * entropy))

        gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(gradients, self.actor.trainable_variables))

        return {
            'actor_loss': float(actor_loss.numpy()),
            'critic_loss': float(critic_loss),
            'mean_advantage': float(np.mean(advantages)),
            'mean_value': float(np.mean(values)),
            'mean_td_target': float(np.mean(td_targets))
        }


# ============================================
#   ŁADOWANIE I PRZYGOTOWANIE TICKOWYCH DANYCH
# ============================================

print("=" * 70)
print("A2C dla tickowych danych WIG20 (co 15 sekund)")
print("=" * 70)
print("\n📊 Wczytuję tickowe dane WIG20...")

# Wczytaj dane tickowe
df = pd.read_csv('data/wig20_tick_data.csv')  # Zmień na właściwą nazwę pliku

print(f"Format pliku:")
print(df.head())
print(f"\nKolumny: {df.columns.tolist()}")

# Parsuj datetime
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values('datetime').reset_index(drop=True)

print(f"\nRAW DATA: {len(df)} ticków")
print(f"Zakres: {df['datetime'].min()} - {df['datetime'].max()}")
print(f"Okres: {(df['datetime'].max() - df['datetime'].min()).days} dni")

# Podstawowa walidacja
if 'price' not in df.columns:
    print("❌ BŁĄD: Brak kolumny 'price' w danych!")
    sys.exit(1)

# Filtruj godziny giełdowe (9:00-16:30)
print(f"\n🕐 Filtruję godziny giełdowe (9:00-16:30)...")
df['hour'] = df['datetime'].dt.hour
df['minute'] = df['datetime'].dt.minute

df_trading = df[
    ((df['hour'] >= 9) & (df['hour'] < 16)) |
    ((df['hour'] == 16) & (df['minute'] <= 30))
    ].copy()

print(f"Po filtrze: {len(df_trading)} ticków")
print(f"Usunięto: {len(df) - len(df_trading)} ticków poza godzinami")

# Ustaw index
df = df_trading.set_index('datetime').sort_index()

# Dodaj kopię surowej ceny (do PnL)
df['price_raw'] = df['price'].copy()

print(f"\n{'=' * 70}")
print("🎯 Feature Engineering (TYLKO RELATIVE VALUES)")
print(f"{'=' * 70}\n")

# TICKOWE TIMEFRAMES
TICK_1MIN = 4  # 1 minuta = 4 ticki (60s / 15s)
TICK_5MIN = 20  # 5 minut = 20 ticków
TICK_15MIN = 60  # 15 minut = 60 ticków
TICK_1H = 240  # 1 godzina = 240 ticków

print("Tickowe timeframes:")
print(f"  1 min  = {TICK_1MIN} ticków")
print(f"  5 min  = {TICK_5MIN} ticków")
print(f"  15 min = {TICK_15MIN} ticków")
print(f"  1 hour = {TICK_1H} ticków\n")

# ============================================
#   RELATIVE FEATURES (wszystkie jako %)
# ============================================

print("Tworzę relative features...")

# Returns (% change)
df['returns'] = df['price'].pct_change() * 100

# Time features (intraday seasonality)
df['hour'] = df.index.hour
df['minute'] = df.index.minute
minutes_since_open = (df['hour'] - 9) * 60 + df['minute']
df['time_of_day'] = minutes_since_open / 450  # 0.0 - 1.0

df['hour_sin'] = np.sin(2 * np.pi * df['time_of_day'])
df['hour_cos'] = np.cos(2 * np.pi * df['time_of_day'])

# Session open (pierwsza cena każdego dnia)
df['date'] = df.index.date
df['session_open'] = df.groupby('date')['price'].transform('first')

# Distance from session open (%)
df['distance_from_open'] = (df['price'] / df['session_open'] - 1) * 100

# Moving averages (tickowe timeframes)
df['sma_1min'] = df['price'].rolling(window=TICK_1MIN).mean()
df['sma_5min'] = df['price'].rolling(window=TICK_5MIN).mean()
df['sma_15min'] = df['price'].rolling(window=TICK_15MIN).mean()
df['sma_1h'] = df['price'].rolling(window=TICK_1H).mean()

# Price to SMA (% deviation)
df['price_to_sma_1min'] = (df['price'] / df['sma_1min'] - 1) * 100
df['price_to_sma_5min'] = (df['price'] / df['sma_5min'] - 1) * 100
df['price_to_sma_15min'] = (df['price'] / df['sma_15min'] - 1) * 100
df['price_to_sma_1h'] = (df['price'] / df['sma_1h'] - 1) * 100

# SMA returns (momentum)
df['sma_1min_return'] = df['sma_1min'].pct_change() * 100
df['sma_5min_return'] = df['sma_5min'].pct_change() * 100
df['sma_15min_return'] = df['sma_15min'].pct_change() * 100

# Volatility (rolling std of returns)
df['volatility_1min'] = df['returns'].rolling(window=TICK_1MIN).std()
df['volatility_5min'] = df['returns'].rolling(window=TICK_5MIN).std()
df['volatility_15min'] = df['returns'].rolling(window=TICK_15MIN).std()

# Volatility of volatility (uncertainty)
df['vol_of_vol'] = df['volatility_5min'].rolling(window=TICK_15MIN).std()

# Price momentum (krótkoterminowy trend)
df['momentum_1min'] = df['returns'].rolling(window=TICK_1MIN).mean()
df['momentum_5min'] = df['returns'].rolling(window=TICK_5MIN).mean()

# Cleanup pomocniczych kolumn
helper_cols = ['hour', 'minute', 'date', 'session_open',
               'sma_1min', 'sma_5min', 'sma_15min', 'sma_1h']

for col in helper_cols:
    if col in df.columns:
        df = df.drop(columns=[col])

# Usuń NaN
df.dropna(inplace=True)

print(f"✅ Po utworzeniu features: {len(df)} ticków")
print(f"   To jest {len(df) / (TICK_1H * 7.5):.1f} dni giełdowych\n")

# ============================================
#   FEATURES LIST (wszystkie relative!)
# ============================================

features = [
    # Returns & Momentum
    'returns',
    'momentum_1min',
    'momentum_5min',

    # Price position
    'distance_from_open',
    'price_to_sma_1min',
    'price_to_sma_5min',
    'price_to_sma_15min',
    'price_to_sma_1h',

    # Trend (SMA returns)
    'sma_1min_return',
    'sma_5min_return',
    'sma_15min_return',

    # Volatility
    'volatility_1min',
    'volatility_5min',
    'volatility_15min',
    'vol_of_vol',

    # Time
    'hour_sin',
    'hour_cos',
    'time_of_day'
]

print(f"✅ FEATURES: {len(features)} pure relative values")
print(f"❌ ZERO surowych cen w NN input!\n")

for i, f in enumerate(features, 1):
    print(f"  {i:2d}. {f}")

print(f"\n{'=' * 70}\n")

# Sprawdź czy wszystkie features istnieją
missing = [f for f in features if f not in df.columns]
if missing:
    print(f"❌ BŁĄD: Brakujące features: {missing}")
    sys.exit(1)

# ============================================
#   PODZIAŁ DANYCH
# ============================================

print(f"{'=' * 70}")
print("📊 Podział danych:")
print(f"{'=' * 70}\n")

total_len = len(df)
train_end = int(total_len * 0.70)
val_end = int(total_len * 0.85)

cols_for_env = features + ['price_raw']

train_data = df.iloc[:train_end][cols_for_env].copy()
val_data = df.iloc[train_end:val_end][cols_for_env].copy()
test_data = df.iloc[val_end:][cols_for_env].copy()

# Przelicz na dni giełdowe (7.5h = 450 min = 1800 ticków)
ticks_per_day = TICK_1H * 7.5

train_days = len(train_data) / ticks_per_day
val_days = len(val_data) / ticks_per_day
test_days = len(test_data) / ticks_per_day

print(f"Train: {len(train_data)} ticków ({train_days:.1f} dni)")
print(f"Val:   {len(val_data)} ticków ({val_days:.1f} dni)")
print(f"Test:  {len(test_data)} ticków ({test_days:.1f} dni)")

# ============================================
#   PARAMETRY ENVIRONMENT
# ============================================

print(f"\n{'=' * 70}")
print("🎯 Parametry Environment dla ticków:")
print(f"{'=' * 70}\n")

# MAX_EPISODE_STEPS: 2 godziny = 480 ticków
# Dlaczego 2h? Krótsze epizody są stabilniejsze dla tickowych danych
MAX_EPISODE_STEPS = 480  # 2 godziny

print(f"MAX_EPISODE_STEPS = {MAX_EPISODE_STEPS} ticków (2 godziny)")
print(f"Position size = 5 (mniejszy dla ticków)")
print(f"Reward scale = 2.0 (tickowe zmiany są mniejsze)")
print(f"Transaction cost = 0.000005 (realistyczny dla ticków)")
print(f"\n{'=' * 70}\n")

# ============================================
#   AGENT I ENVIRONMENTS
# ============================================

state_size = len(features) + 5  # features + position info
agent = A2CAgent(
    state_size=state_size,
    action_size=3,
    actor_lr=0.0005,
    critic_lr=0.001,
    temperature=5.0,
    epsilon=0.1
)

train_env = TickTradingEnvironment(
    train_data,
    initial_balance=10000,
    position_size=5,  # Mniejszy dla ticków
    max_episode_steps=MAX_EPISODE_STEPS,
    random_start=True,
    features=features,
    reward_scale=2.0,  # Mniejszy dla tickowych zmian
    transaction_cost=0.000005  # Realistyczny dla ticków
)

val_env = TickTradingEnvironment(
    val_data,
    initial_balance=10000,
    position_size=5,
    max_episode_steps=None,
    random_start=False,
    features=features,
    reward_scale=2.0,
    transaction_cost=0.000005
)

test_env = TickTradingEnvironment(
    test_data,
    initial_balance=10000,
    position_size=5,
    max_episode_steps=None,
    random_start=False,
    features=features,
    reward_scale=2.0,
    transaction_cost=0.000005
)


def test_agent(agent, env, n_runs=1):
    """Testuje agenta bez treningu"""
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
#   TRENING A2C Z BATCH
# ============================================

print(f"{'=' * 70}")
print("🚀 Rozpoczynam trening A2C...")
print(f"{'=' * 70}\n")

episodes = 200
BATCH_SIZE = 5

print(f"PARAMETRY TRENOWANIA:")
print(f"  Episodes: {episodes}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Liczba batchy: {episodes // BATCH_SIZE}")
print(f"  Coverage: {episodes * MAX_EPISODE_STEPS / len(train_data):.1%} train data")
print(f"\n{'=' * 70}\n")

best_val_reward = -float('inf')

batch_states = []
batch_actions = []
batch_rewards = []
batch_next_states = []
batch_dones = []

rewards_history = []
val_rewards_history = []
val_balance_history = []
actor_loss_history = []
critic_loss_history = []

for episode in range(episodes):
    state = train_env.reset()

    # Log początkowych prawdopodobieństw co batch
    if (episode + 1) % BATCH_SIZE == 0:
        first_logits = agent.actor.predict(state.reshape(1, -1), verbose=0)[0]
        first_logits_scaled = first_logits / agent.temperature
        first_logits_scaled = np.clip(first_logits_scaled, -2.0, 2.0)
        exp_logits = np.exp(first_logits_scaled - np.max(first_logits_scaled))
        first_probs = exp_logits / np.sum(exp_logits)

        first_value = agent.critic.predict(state.reshape(1, -1), verbose=0)[0][0]

        tqdm.write(
            f"\nBatch {(episode + 1) // BATCH_SIZE}/{episodes // BATCH_SIZE} - "
            f"Probs: HOLD={first_probs[0]:.3f}, BUY={first_probs[1]:.3f}, SELL={first_probs[2]:.3f}, "
            f"Value={first_value:.2f}"
        )

    total_reward = 0.0
    done = False
    states = []
    actions = []
    rewards_ep = []
    next_states = []
    dones = []
    action_counts = {0: 0, 1: 0, 2: 0}
    last_info = None

    ep_len = train_env.max_episode_steps or len(train_data)

    with tqdm(total=ep_len, desc=f"Ep {episode + 1}/{episodes}",
              leave=False, position=0, file=sys.stdout, mininterval=0.5) as pbar:
        while not done:
            action = agent.act(state)
            action_counts[action] += 1
            next_state, reward, done, info = train_env.step(action)

            last_info = info

            states.append(state)
            actions.append(action)
            rewards_ep.append(reward)
            next_states.append(next_state if not done else np.zeros_like(state))
            dones.append(1.0 if done else 0.0)

            state = next_state if not done else state
            total_reward += reward

            pbar.update(1)
            pbar.set_postfix({'reward': f'{total_reward:.2f}'})

    # Dodaj do batch
    batch_states.extend(states)
    batch_actions.extend(actions)
    batch_rewards.extend(rewards_ep)
    batch_next_states.extend(next_states)
    batch_dones.extend(dones)

    rewards_history.append(total_reward)
    train_balance = last_info['balance'] if last_info else train_env.balance
    train_trades = last_info['trade_count'] if last_info else 0

    # Train co BATCH_SIZE epizodów
    if (episode + 1) % BATCH_SIZE == 0:
        batch_num = (episode + 1) // BATCH_SIZE

        tqdm.write(f"\n{'=' * 70}")
        tqdm.write(f"🔄 BATCH {batch_num}/{episodes // BATCH_SIZE}")
        tqdm.write(f"   Trenuję na {len(batch_states)} krokach...")

        train_stats = agent.train(
            batch_states,
            batch_actions,
            batch_rewards,
            batch_next_states,
            batch_dones
        )

        actor_loss_history.append(train_stats['actor_loss'])
        critic_loss_history.append(train_stats['critic_loss'])

        # Wyczyść batch
        batch_states = []
        batch_actions = []
        batch_rewards = []
        batch_next_states = []
        batch_dones = []

        recent_rewards = rewards_history[-BATCH_SIZE:]
        avg_reward = np.mean(recent_rewards)

        tqdm.write(f"   Actor Loss: {train_stats['actor_loss']:.4f}")
        tqdm.write(f"   Critic Loss: {train_stats['critic_loss']:.4f}")
        tqdm.write(f"   Mean Advantage: {train_stats['mean_advantage']:.4f}")
        tqdm.write(f"   Średni Train Reward: {avg_reward:.2f}")

        # Validation
        val_reward, val_balance, val_trades = test_agent(agent, val_env, n_runs=1)
        val_rewards_history.append(val_reward)
        val_balance_history.append(val_balance)

        tqdm.write(f"   Val: Reward={val_reward:.2f}, Balance={val_balance:.2f}, Trades={val_trades:.0f}")

        if val_reward > best_val_reward:
            best_val_reward = val_reward
            agent.actor.save('best_a2c_tick_actor.keras')
            agent.critic.save('best_a2c_tick_critic.keras')
            tqdm.write(f"   ✅ Nowy najlepszy! Val Reward: {val_reward:.2f}")

        tqdm.write(f"{'=' * 70}\n")

        # Epsilon decay
        agent.epsilon = max(0.01, agent.epsilon * 0.995)

print("\n✓ Trening zakończony!")

# ============================================
#   WYKRESY
# ============================================

plt.figure(figsize=(18, 12))

# Training Rewards
plt.subplot(3, 3, 1)
plt.plot(rewards_history, alpha=0.6)
plt.title('Training Reward per Episode')
plt.xlabel('Episode')
plt.ylabel('Reward (scaled)')
plt.grid(True, alpha=0.3)

# Validation Rewards
plt.subplot(3, 3, 2)
if len(val_rewards_history) > 0:
    batch_indices = [i * BATCH_SIZE for i in range(1, len(val_rewards_history) + 1)]
    plt.plot(batch_indices, val_rewards_history, 'o-', color='green')
    plt.title('Validation Reward')
    plt.xlabel('Episode')
    plt.ylabel('Val Reward (scaled)')
    plt.grid(True, alpha=0.3)

# Validation Balance
plt.subplot(3, 3, 3)
if len(val_balance_history) > 0:
    batch_indices = [i * BATCH_SIZE for i in range(1, len(val_balance_history) + 1)]
    plt.plot(batch_indices, val_balance_history, 'o-', color='blue')
    plt.axhline(y=10000, color='r', linestyle='--', label='Initial')
    plt.title('Validation Balance')
    plt.xlabel('Episode')
    plt.ylabel('Balance (PLN)')
    plt.legend()
    plt.grid(True, alpha=0.3)

# MA Rewards
plt.subplot(3, 3, 4)
window = 20
if len(rewards_history) >= window:
    ma_rewards = pd.Series(rewards_history).rolling(window=window).mean()
    plt.plot(ma_rewards, label=f'MA-{window}')
    plt.title(f'Training Reward (MA-{window})')
    plt.xlabel('Episode')
    plt.ylabel('MA Reward')
    plt.grid(True, alpha=0.3)
    plt.legend()

# Actor Loss
plt.subplot(3, 3, 5)
if len(actor_loss_history) > 0:
    batch_indices = [i * BATCH_SIZE for i in range(1, len(actor_loss_history) + 1)]
    plt.plot(batch_indices, actor_loss_history, 'o-', color='orange')
    plt.title('Actor Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)

# Critic Loss
plt.subplot(3, 3, 6)
if len(critic_loss_history) > 0:
    batch_indices = [i * BATCH_SIZE for i in range(1, len(critic_loss_history) + 1)]
    plt.plot(batch_indices, critic_loss_history, 'o-', color='purple')
    plt.title('Critic Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)

# Returns distribution
plt.subplot(3, 3, 7)
sample_returns = df['returns'].iloc[-5000:]
plt.hist(sample_returns, bins=50, alpha=0.7, edgecolor='black')
plt.title('Returns Distribution (last 5000 ticks)')
plt.xlabel('Returns (%)')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)

# Volatility over time
plt.subplot(3, 3, 8)
sample_vol = df['volatility_5min'].iloc[-5000:]
plt.plot(sample_vol.values)
plt.title('5-min Volatility (last 5000 ticks)')
plt.xlabel('Time')
plt.ylabel('Volatility')
plt.grid(True, alpha=0.3)

# Train vs Val
plt.subplot(3, 3, 9)
if len(val_rewards_history) > 0:
    train_per_batch = [np.mean(rewards_history[i * BATCH_SIZE:(i + 1) * BATCH_SIZE])
                       for i in range(len(val_rewards_history))]
    batch_indices = [i * BATCH_SIZE for i in range(1, len(val_rewards_history) + 1)]
    plt.plot(batch_indices, train_per_batch, 'o-', label='Train (avg)', alpha=0.7)
    plt.plot(batch_indices, val_rewards_history, 'o-', label='Val', alpha=0.7)
    plt.title('Train vs Val Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('a2c_tick_results.png', dpi=150)
print("\n✓ Wykres zapisany: a2c_tick_results.png")

# ============================================
#   TEST
# ============================================

print(f"\n{'=' * 70}")
print("🧪 TEST NA DANYCH TESTOWYCH")
print(f"{'=' * 70}\n")

test_reward, test_balance, test_trades = test_agent(agent, test_env, n_runs=5)

print(f"Test Reward (avg, scaled): {test_reward:.2f}")
print(f"Test Balance (avg): {test_balance:.2f}")
print(f"Test Profit (avg): {test_balance - 10000:.2f} PLN")
print(f"Test Trades (avg): {test_trades:.0f}")
if test_trades > 0:
    print(f"Profit per Trade: {(test_balance - 10000) / test_trades:.2f} PLN")

print(f"\n{'=' * 70}")
print("✅ TRENING A2C ZAKOŃCZONY!")
print(f"{'=' * 70}")
print(f"\nBest Val Reward: {best_val_reward:.2f}")
print(f"Test Reward: {test_reward:.2f}")
print(f"\nModele zapisane:")
print(f"  - best_a2c_tick_actor.keras")
print(f"  - best_a2c_tick_critic.keras")

print(f"\n{'=' * 70}")
print("📊 PODSUMOWANIE TICKOWYCH DANYCH")
print(f"{'=' * 70}")
print(f"\n✅ TICKOWE TIMEFRAMES:")
print(f"   1 min  = {TICK_1MIN} ticków")
print(f"   5 min  = {TICK_5MIN} ticków")
print(f"   15 min = {TICK_15MIN} ticków")
print(f"   1 hour = {TICK_1H} ticków")
print(f"\n✅ FEATURES: {len(features)} pure relative values")
print(f"   ❌ ZERO surowych cen w NN!")
print(f"\n✅ ENVIRONMENT:")
print(f"   Episode length: {MAX_EPISODE_STEPS} ticków (2h)")
print(f"   Position size: 5 (vs 10 dla minutowych)")
print(f"   Reward scale: 2.0 (vs 5.0 dla minutowych)")
print(f"   Transaction cost: 0.000005 (realistyczny)")
print(f"\n{'=' * 70}\n")