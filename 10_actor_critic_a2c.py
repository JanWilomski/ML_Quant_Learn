"""
A2C (Advantage Actor-Critic) dla arbitrażu WIG20-DAX

🔧 FIXED VERSION dla ~400 DNI DANYCH + REDUCED FEATURES (18 zamiast 34)

ZMIANY vs wersja dla 100 dni:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PARAMETRY TRENOWANIA (główne zmiany):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- episodes: 100 → 250 (żeby agent zobaczył dane ~1 raz)
  Dlaczego? 400 dni * 0.7 (train) = 280 dni, 250 epizodów = 0.89x coverage

- BATCH_SIZE: 10 → 5 (częstsze aktualizacje)
  Dlaczego? 250/5 = 50 aktualizacji vs 100/10 = 10 aktualizacji
  Więcej small batches = lepsze tracking progressu

- actor_lr: 0.0005 → 0.0007 (nieco szybsze uczenie)
  Dlaczego? Więcej danych = możemy uczyć się szybciej bez overfittingu

- critic_lr: 0.001 → 0.0015 (nieco szybsze uczenie)
  Dlaczego? Critic potrzebuje więcej czasu na naukę z większego datasetu

PARAMETRY ENVIRONMENT (bez zmian - już zoptymalizowane):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- position_size: 10 ✓
- reward_scale: 5.0 ✓
- transaction_cost: 0.000001 ✓
- temperature: 5.0 ✓
- epsilon: 0.1 (decay 0.99) ✓

EXPECTED RESULTS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Batch 10 (50 epizodów):  Prawdopodobieństwa ~0.30/0.35/0.35
- Batch 25 (125 epizodów): Prawdopodobieństwa ~0.25/0.38/0.37
- Batch 50 (250 epizodów): Silna strategia, Critic Loss < 0.1
- Val Balance: 10800-11500 PLN (zysk +800-1500)
- Test Balance: 10600-11200 PLN (zysk +600-1200)

Training time: ~45-60 min (vs 20-30 min dla 100 epizodów)
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
#   TRADING ENVIRONMENT (identyczny jak w FIXED)
# ============================================

class ArbitrageEnvironment:
    def __init__(self, data, initial_balance=10000, position_size=1,
                 max_episode_steps=None, random_start=False, features=None,
                 reward_scale=50.0, transaction_cost=0.00001):
        """
        Environment do arbitrażu WIG20 vs DAX

        WAŻNE: data powinno zawierać TYLKO godziny giełdowe (9:00-16:30)
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
        """Losuj początek epizodu tylko z początków dni giełdowych"""
        if self.random_start and self.max_episode_steps and self.max_episode_steps < self.n_steps:
            possible_starts = list(range(0, self.n_steps - self.max_episode_steps, self.max_episode_steps))
            if len(possible_starts) > 0:
                self.episode_start = np.random.choice(possible_starts)
            else:
                self.episode_start = 0
        else:
            self.episode_start = 0

        self.current_step = self.episode_start
        self.balance = self.initial_balance
        self.position = None
        self.total_profit = 0
        self.trade_count = 0
        return self._get_state()

    def _get_state(self):
        """State = features + info o pozycji + time of day"""
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

        entry_price_rel = (entry_price / current_price) - 1.0
        pnl_rel = position_pnl / current_price
        balance_rel = (self.balance - self.initial_balance) / self.initial_balance

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
        reward = 0.0
        current_price = self.data.iloc[self.current_step]['wig20_close_raw']

        # HOLD (0)
        if action == 0:
            if self.position is not None:
                reward = -0.0001

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

                base_reward = np.clip(raw_profit / self.reward_scale, -1.0, 1.0)
                if raw_profit > 0:
                    reward = base_reward + 0.05
                else:
                    reward = base_reward - 0.02

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

                base_reward = np.clip(raw_profit / self.reward_scale, -1.0, 1.0)
                if raw_profit > 0:
                    reward = base_reward + 0.05
                else:
                    reward = base_reward - 0.02

                self.position = None
                self.trade_count += 1

        self.current_step += 1

        done_data = (self.current_step >= self.n_steps)
        done_length = False
        if self.max_episode_steps:
            done_length = (self.current_step >= self.episode_start + self.max_episode_steps)
        done_bankrupt = (self.balance <= 0)

        done = done_data or done_length or done_bankrupt

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
#   A2C AGENT
# ============================================

class A2CAgent:
    def __init__(self, state_size=5, action_size=3,
                 actor_lr=0.0001, critic_lr=0.0005, gamma=0.95,
                 temperature=5.0, epsilon=0.1):
        """
        A2C Agent z osobnymi sieciami Actor i Critic

        NOWOŚĆ vs Policy Gradient:
        - Actor: π(a|s) - policy network (wybiera akcje)
        - Critic: V(s) - value network (ocenia states)
        - Training z TD error (advantage) zamiast Monte Carlo returns
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
        print(f"   Actor LR: {actor_lr}")
        print(f"   Critic LR: {critic_lr}")
        print(f"   Temperature: {temperature}")
        print(f"   Gamma: {gamma}\n")

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

        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)

        return np.random.choice(self.action_size, p=probabilities)

    def train(self, states, actions, rewards, next_states, dones):
        """
        A2C Training

        Args:
            states: array of states
            actions: array of actions taken
            rewards: array of immediate rewards
            next_states: array of next states
            dones: array of done flags
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
#   ŁADOWANIE I PRZYGOTOWANIE DANYCH
#   (UŻYTKOWNIK ZMIENIŁ TEN FRAGMENT - NIE EDYTUJ!)
# ============================================

print("=" * 70)
print("A2C (Advantage Actor-Critic) - Arbitraż WIG20-DAX")
print("🔥 WERSJA DLA ~400 DNI DANYCH")
print("=" * 70)
print("\nWczytuję dane WIG20 i DAX...")


# WIG20
print("📊 Wczytuję WIG20...")
wig20 = pd.read_csv('data/gpw_wig20_m1.csv')
wig20['datetime'] = pd.to_datetime(wig20['datetime'])
wig20 = wig20.sort_values('datetime').reset_index(drop=True)

print(f"WIG20 RAW: {len(wig20)} wierszy")
print(f"Zakres: {wig20['datetime'].min()} - {wig20['datetime'].max()}")
print(f"Kolumny: {wig20.columns.tolist()}")

# Sprawdź czy WIG20 ma volume
has_wig20_volume = 'volume' in wig20.columns
if not has_wig20_volume:
    print("⚠️ WIG20 nie ma kolumny 'volume' - dodaję kolumnę z wartościami 0")
    wig20['volume'] = 0

# DAX - z poprawkami formatu i czasu
print("\n📊 Wczytuję DAX...")

try:
    # UWAGA: Nowy format ma separator przecinek (,) a nie średnik (;)
    dax_raw = pd.read_csv('data/ger40_m1.csv', sep=';')

    print(f"DAX RAW: {len(dax_raw)} wierszy")
    print(f"Kolumny: {dax_raw.columns.tolist()}")
    print("Przykładowe surowe dane:")
    print(dax_raw.head(3))

    # Parsuj datetime z formatu '20220103 020000'
    dax_raw['datetime'] = pd.to_datetime(dax_raw['datetime'], format='%Y%m%d %H%M%S')

    print("\n🕐 PRZED przesunięciem czasowym:")
    print(f"Min godzina: {dax_raw['datetime'].dt.hour.min()}:00")
    print(f"Max godzina: {dax_raw['datetime'].dt.hour.max()}:00")
    print(f"Przykład: {dax_raw['datetime'].iloc[0]}")

    # Sprawdź automatycznie jakie przesunięcie jest potrzebne
    first_hour = dax_raw['datetime'].dt.hour.min()

    if first_hour <= 3:
        hours_offset = 7
        print(f"\n✅ Wykryto dane w UTC (pierwsza godzina: {first_hour}:00)")
        print(f"Dodaję {hours_offset}h aby dostać czas lokalny")
    elif first_hour >= 8:
        hours_offset = 0
        print(f"\n✅ Wykryto dane już w czasie lokalnym (pierwsza godzina: {first_hour}:00)")
    else:
        hours_offset = 6
        print(f"\n⚠️ Niejednoznaczna strefa czasowa, używam domyślnego przesunięcia: {hours_offset}h")

    dax_raw['datetime'] = dax_raw['datetime'] + pd.Timedelta(hours=hours_offset)

    print("\n🕐 PO przesunięciu czasowym:")
    print(f"Min godzina: {dax_raw['datetime'].dt.hour.min()}:00")
    print(f"Max godzina: {dax_raw['datetime'].dt.hour.max()}:00")
    print(f"Przykład: {dax_raw['datetime'].iloc[0]}")

    # Weryfikacja
    trading_hours_count = ((dax_raw['datetime'].dt.hour >= 9) &
                          (dax_raw['datetime'].dt.hour <= 17)).sum()
    trading_hours_pct = 100 * trading_hours_count / len(dax_raw)

    print(f"\nWierszy w godzinach 9-17: {trading_hours_count} / {len(dax_raw)} ({trading_hours_pct:.1f}%)")

    if trading_hours_pct < 30:
        print("\n⚠️ UWAGA: Mniej niż 30% danych w godzinach 9-17!")
    else:
        print(f"✅ Przesunięcie czasowe wygląda poprawnie!")

    dax = dax_raw[['datetime', 'open', 'high', 'low', 'close', 'volume']].copy()
    dax = dax.sort_values('datetime').reset_index(drop=True)

    print(f"\nDAX po poprawkach: {len(dax)} wierszy")
    print(f"Zakres: {dax['datetime'].min()} - {dax['datetime'].max()}")

except Exception as e:
    print(f"❌ BŁĄD przy wczytywaniu DAX: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Filtruj TYLKO godziny giełdowe: 9:00-16:30
print(f"\n{'=' * 70}")
print("🕐 Filtruję dane do godzin giełdowych (9:00-16:30)...")

wig20['hour'] = wig20['datetime'].dt.hour
wig20['minute'] = wig20['datetime'].dt.minute

wig20_trading = wig20[
    ((wig20['hour'] >= 9) & (wig20['hour'] < 16)) |
    ((wig20['hour'] == 16) & (wig20['minute'] <= 30))
].copy()

dax['hour'] = dax['datetime'].dt.hour
dax['minute'] = dax['datetime'].dt.minute

dax_trading = dax[
    ((dax['hour'] >= 9) & (dax['hour'] < 16)) |
    ((dax['hour'] == 16) & (dax['minute'] <= 30))
].copy()

print(f"WIG20 po filtrze: {len(wig20_trading)} wierszy")
print(f"DAX po filtrze: {len(dax_trading)} wierszy")

# Merge danych
print(f"\n{'=' * 70}")
print("Łączę dane WIG20 i DAX...")

df = pd.merge(
    wig20_trading[['datetime', 'open', 'high', 'low', 'close', 'volume']],
    dax_trading[['datetime', 'close']],
    on='datetime',
    suffixes=('_wig20', '_dax'),
    how='inner'
)

print(f"Po merge: {len(df)} wierszy")

if len(df) == 0:
    print("\n❌ BŁĄD: Brak wspólnych timestampów!")
    sys.exit(1)

df = df.rename(columns={
    'close_wig20': 'wig20_close',
    'close_dax': 'dax_close'
})

df = df.set_index('datetime').sort_index()
df['wig20_close_raw'] = df['wig20_close'].copy()

# Time features
df['hour'] = df.index.hour
df['minute'] = df.index.minute
minutes_since_open = (df['hour'] - 9) * 60 + df['minute']
df['time_of_day'] = minutes_since_open / 450
df['hour_sin'] = np.sin(2 * np.pi * df['time_of_day'])
df['hour_cos'] = np.cos(2 * np.pi * df['time_of_day'])

# Session open
df['date'] = df.index.date
df['session_open'] = df.groupby('date')['wig20_close'].transform('first')

# Feature Engineering - TYLKO RELATIVE
print(f"\n{'=' * 70}")
print("🎯 Tworzę features arbitrażowe (TYLKO RELATIVE VALUES)...")

# WIG20 features
df['wig20_returns'] = df['wig20_close'].pct_change() * 100
df['wig20_distance_from_open'] = (df['wig20_close'] / df['session_open'] - 1) * 100
df['wig20_high_low_range'] = ((df['high'] - df['low']) / df['wig20_close']) * 100
df['wig20_position_in_range'] = ((df['wig20_close'] - df['low']) / (df['high'] - df['low'] + 1e-8))

df['wig20_sma_5'] = df['wig20_close'].rolling(window=5).mean()
df['wig20_sma_15'] = df['wig20_close'].rolling(window=15).mean()
df['wig20_sma_60'] = df['wig20_close'].rolling(window=60).mean()

df['wig20_price_to_sma5'] = (df['wig20_close'] / df['wig20_sma_5'] - 1) * 100
df['wig20_price_to_sma15'] = (df['wig20_close'] / df['wig20_sma_15'] - 1) * 100
df['wig20_price_to_sma60'] = (df['wig20_close'] / df['wig20_sma_60'] - 1) * 100

df['wig20_sma5_return'] = df['wig20_sma_5'].pct_change() * 100
df['wig20_sma15_return'] = df['wig20_sma_15'].pct_change() * 100
df['wig20_volatility'] = df['wig20_returns'].rolling(window=20).std()

# DAX features
df['dax_returns'] = df['dax_close'].pct_change() * 100

df['dax_sma_5'] = df['dax_close'].rolling(window=5).mean()
df['dax_sma_15'] = df['dax_close'].rolling(window=15).mean()
df['dax_sma_60'] = df['dax_close'].rolling(window=60).mean()

df['dax_price_to_sma5'] = (df['dax_close'] / df['dax_sma_5'] - 1) * 100
df['dax_price_to_sma15'] = (df['dax_close'] / df['dax_sma_15'] - 1) * 100
df['dax_price_to_sma60'] = (df['dax_close'] / df['dax_sma_60'] - 1) * 100

df['dax_sma5_return'] = df['dax_sma_5'].pct_change() * 100
df['dax_sma15_return'] = df['dax_sma_15'].pct_change() * 100
df['dax_volatility'] = df['dax_returns'].rolling(window=20).std()

# Arbitrage features
df['wig20_normalized'] = (df['wig20_close'] / df['wig20_close'].iloc[0]) * 100
df['dax_normalized'] = (df['dax_close'] / df['dax_close'].iloc[0]) * 100

df['spread'] = df['wig20_normalized'] - df['dax_normalized']
df['spread_sma'] = df['spread'].rolling(window=30).mean()
df['spread_std'] = df['spread'].rolling(window=30).std()
df['spread_zscore'] = (df['spread'] - df['spread_sma']) / (df['spread_std'] + 1e-8)

df['spread_change'] = df['spread'].diff()
df['spread_pct_change'] = df['spread'].pct_change() * 100
df['spread_acceleration'] = df['spread_change'].diff()

# Correlation
def rolling_correlation(series1, series2, window):
    return series1.rolling(window).corr(series2)

df['correlation_30'] = rolling_correlation(df['wig20_returns'], df['dax_returns'], 30)
df['correlation_60'] = rolling_correlation(df['wig20_returns'], df['dax_returns'], 60)

# Lead-lag effect
df['dax_returns_lag1'] = df['dax_returns'].shift(1)
df['dax_returns_lag2'] = df['dax_returns'].shift(2)
df['dax_returns_lag3'] = df['dax_returns'].shift(3)

# Price ratio
df['price_ratio'] = df['wig20_close'] / df['dax_close']
df['price_ratio_sma'] = df['price_ratio'].rolling(window=30).mean()
df['price_ratio_deviation'] = (df['price_ratio'] / df['price_ratio_sma'] - 1) * 100

# Momentum divergence
df['wig20_momentum'] = df['wig20_returns'].rolling(window=10).mean()
df['dax_momentum'] = df['dax_returns'].rolling(window=10).mean()
df['momentum_divergence'] = df['wig20_momentum'] - df['dax_momentum']

# Volatility comparison
df['volatility_ratio'] = df['wig20_volatility'] / (df['dax_volatility'] + 1e-8)
df['volatility_spread'] = df['wig20_volatility'] - df['dax_volatility']

# Volume
df['volume_sma'] = df['volume'].rolling(window=20).mean()
df['wig20_volume_ratio'] = df['volume'] / (df['volume_sma'] + 1e-8)

# Cleanup
if 'date' in df.columns:
    df = df.drop(columns=['date'])
if 'session_open' in df.columns:
    df = df.drop(columns=['session_open'])
if 'hour' in df.columns:
    df = df.drop(columns=['hour'])
if 'minute' in df.columns:
    df = df.drop(columns=['minute'])

helper_cols = ['volume_sma', 'wig20_sma_5', 'wig20_sma_15', 'wig20_sma_60',
               'dax_sma_5', 'dax_sma_15', 'dax_sma_60',
               'spread_sma', 'spread_std',
               'price_ratio', 'price_ratio_sma', 'wig20_momentum', 'dax_momentum',
               'open', 'high', 'low', 'volume']

for col in helper_cols:
    if col in df.columns:
        df = df.drop(columns=[col])

df.dropna(inplace=True)

print(f"✅ Po utworzeniu features: {len(df)} wierszy")

# Weryfikacja kompletności dni
df_with_date = df.copy()
df_with_date['date'] = df_with_date.index.date
minutes_per_day = df_with_date.groupby('date').size()

print(f"\n{'=' * 70}")
print("WERYFIKACJA KOMPLETNOŚCI DNI:")
print(f"Liczba unikalnych dni: {len(minutes_per_day)}")
print(f"Średnia minut/dzień: {minutes_per_day.mean():.1f}")

expected_minutes = 451
complete_days = (minutes_per_day >= expected_minutes - 10).sum()
incomplete_days = len(minutes_per_day) - complete_days

print(f"Kompletne dni: {complete_days}")
print(f"Niekompletne dni: {incomplete_days}")

if incomplete_days > 0:
    print("\n🔧 Usuwam niekompletne dni...")
    complete_dates = minutes_per_day[minutes_per_day >= expected_minutes - 10].index
    df = df[np.isin(df.index.date, complete_dates)]
    print(f"Po usunięciu: {len(df)} wierszy")

# Przelicz ile to faktycznie dni
df_with_date = df.copy()
df_with_date['date'] = df_with_date.index.date
minutes_per_day = df_with_date.groupby('date').size()
actual_days = len(minutes_per_day)

TRADING_DAY_MINUTES = int(minutes_per_day.median())
print(f"\nMAX_EPISODE_STEPS = {TRADING_DAY_MINUTES} minut")
print(f"\n📊 FINALNE DANE: {actual_days} kompletnych dni tradingowych")

# Features list
# ============================================
#   REDUCED FEATURES - 18 NAJWAŻNIEJSZYCH
# ============================================
# USUNIĘTE (16 features):
# - wig20_high_low_range, wig20_position_in_range (zaszumione, mało przydatne)
# - wig20_sma5_return, wig20_sma15_return (redundant z wig20_returns)
# - dax_sma5_return, dax_sma15_return (redundant z dax_returns)
# - spread_change, spread_pct_change, spread_acceleration (redundant - zostaw tylko zscore!)
# - correlation_60 (correlation_30 wystarczy)
# - dax_returns_lag2, dax_returns_lag3 (lag1 wystarczy dla lead-lag)
# - volatility_spread (volatility_ratio ważniejszy)
# - wig20_volume_ratio (fake volume=0)
# - time_of_day (redundant z hour_sin/cos)

features = [
    # WIG20 (6 features) - najważniejsze dla podstawowych trendów
    'wig20_returns',              # Momentum WIG20
    'wig20_distance_from_open',   # Intraday position
    'wig20_price_to_sma5',        # Short-term trend
    'wig20_price_to_sma15',       # Medium-term trend
    'wig20_price_to_sma60',       # Long-term trend
    'wig20_volatility',           # Risk measure

    # DAX (5 features) - najważniejsze dla porównania
    'dax_returns',                # Momentum DAX
    'dax_price_to_sma5',          # Short-term trend
    'dax_price_to_sma15',         # Medium-term trend
    'dax_price_to_sma60',         # Long-term trend
    'dax_volatility',             # Risk measure

    # ARBITRAGE (5 features) - KLUCZOWE dla strategii!
    'spread_zscore',              # ⭐ NAJWAŻNIEJSZY! Main arbitrage signal
    'correlation_30',             # Markets relationship
    'dax_returns_lag1',           # Lead-lag effect (DAX prowadzi WIG20)
    'momentum_divergence',        # Momentum difference
    'price_ratio_deviation',      # Relative pricing
    'volatility_ratio',           # Volatility comparison

    # TIME (2 features) - seasonality w ciągu dnia
    'hour_sin',                   # Time of day (cyclical)
    'hour_cos'                    # Time of day (cyclical)
]

print(f"\n✅ ZREDUKOWANE DO: {len(features)} najważniejszych features (było 34)")
print(f"   Usunięto: 16 redundant/noisy features")
print(f"   Zostały: TYLKO najważniejsze sygnały dla arbitrażu!\n")

# Podział danych
print(f"\n{'=' * 70}")
print("📊 Podział danych:")

total_len = len(df)
train_end = int(total_len * 0.70)
val_end = int(total_len * 0.85)

cols_for_env = features + ['wig20_close_raw']

train_data = df.iloc[:train_end][cols_for_env].copy()
val_data = df.iloc[train_end:val_end][cols_for_env].copy()
test_data = df.iloc[val_end:][cols_for_env].copy()

train_days = len(train_data) / TRADING_DAY_MINUTES
val_days = len(val_data) / TRADING_DAY_MINUTES
test_days = len(test_data) / TRADING_DAY_MINUTES

print(f"Train: {len(train_data)} minut ({train_days:.1f} dni)")
print(f"Val:   {len(val_data)} minut ({val_days:.1f} dni)")
print(f"Test:  {len(test_data)} minut ({test_days:.1f} dni)")
print(f"\n💡 Z {actual_days} dni danych, train ma {train_days:.0f} dni")


# ============================================
#   AGENT I ENVIRONMENTS
#   🔥 ZOPTYMALIZOWANE PARAMETRY DLA 400 DNI
# ============================================

print(f"\n{'=' * 70}")
print("🎭 Tworzę A2C Agenta z ZOPTYMALIZOWANYMI parametrami dla ~400 dni...")
print(f"{'=' * 70}\n")

state_size = len(features) + 5
agent = A2CAgent(
    state_size=state_size,
    action_size=3,
    actor_lr=0.0007,    # ← ZWIĘKSZONE z 0.0005 (szybsze uczenie z więcej danych)
    critic_lr=0.0015,   # ← ZWIĘKSZONE z 0.001 (Critic potrzebuje więcej czasu na naukę)
    temperature=5.0,    # ← Bez zmian (wysoka eksploracja OK)
    epsilon=0.1         # ← Bez zmian (epsilon decay zadziała lepiej z więcej epizodów)
)

print("🏗️  Tworzę Environments z FIXED parametrami...")
print("   (position_size=10, reward_scale=5.0, transaction_cost=0.000001)\n")

train_env = ArbitrageEnvironment(
    train_data,
    initial_balance=10000,
    position_size=10,           # ← FIXED: było 1
    max_episode_steps=TRADING_DAY_MINUTES,
    random_start=True,
    features=features,
    reward_scale=5.0,           # ← FIXED: było 50.0
    transaction_cost=0.000001   # ← FIXED: było 0.00001
)

val_env = ArbitrageEnvironment(
    val_data,
    initial_balance=10000,
    position_size=10,
    max_episode_steps=None,
    random_start=False,
    features=features,
    reward_scale=5.0,
    transaction_cost=0.000001
)

test_env = ArbitrageEnvironment(
    test_data,
    initial_balance=10000,
    position_size=10,
    max_episode_steps=None,
    random_start=False,
    features=features,
    reward_scale=5.0,
    transaction_cost=0.000001
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
#   🔥 ZOPTYMALIZOWANE PARAMETRY DLA 400 DNI
# ============================================

print(f"\n{'=' * 70}")
print("🚀 Rozpoczynam trening A2C z ZOPTYMALIZOWANYMI parametrami...")
print(f"{'=' * 70}\n")

# GŁÓWNE ZMIANY:
episodes = 250         # ← ZWIĘKSZONE z 100 (żeby agent zobaczył dane ~1 raz)
BATCH_SIZE = 5         # ← ZMNIEJSZONE z 10 (częstsze aktualizacje = lepsze tracking)

print(f"📊 PARAMETRY TRENOWANIA:")
print(f"   Episodes: {episodes} (vs 100 poprzednio)")
print(f"   Batch size: {BATCH_SIZE} (vs 10 poprzednio)")
print(f"   Liczba batchy: {episodes // BATCH_SIZE}")
print(f"   Epizod = {TRADING_DAY_MINUTES} minut (pełny dzień giełdowy)")
print(f"   Coverage: {episodes / train_days:.2f}x train data")
print(f"\n💡 Z {train_days:.0f} dni train, {episodes} epizodów = {episodes/train_days:.1%} coverage")
print(f"   Więcej epizodów = agent widzi więcej różnych wzorców!")
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
        tqdm.write(f"   Mean Value: {train_stats['mean_value']:.2f}")
        tqdm.write(f"   Średni Train Reward: {avg_reward:.2f}")

        # Validation
        val_reward, val_balance, val_trades = test_agent(agent, val_env, n_runs=1)
        val_rewards_history.append(val_reward)
        val_balance_history.append(val_balance)

        tqdm.write(f"   Val: Reward={val_reward:.2f}, Balance={val_balance:.2f}, Trades={val_trades:.0f}")

        if val_reward > best_val_reward:
            best_val_reward = val_reward
            agent.actor.save('best_a2c_actor_400days.keras')
            agent.critic.save('best_a2c_critic_400days.keras')
            tqdm.write(f"   ✅ Nowy najlepszy! Val Reward: {val_reward:.2f}")

        tqdm.write(f"{'=' * 70}\n")

        # Wolniejszy epsilon decay (250 epizodów vs 100)
        agent.epsilon = max(0.01, agent.epsilon * 0.995)  # ← WOLNIEJSZY z 0.99

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
    plt.title('Validation Reward (per batch)')
    plt.xlabel('Episode')
    plt.ylabel('Val Reward (scaled)')
    plt.grid(True, alpha=0.3)

# Validation Balance
plt.subplot(3, 3, 3)
if len(val_balance_history) > 0:
    batch_indices = [i * BATCH_SIZE for i in range(1, len(val_balance_history) + 1)]
    plt.plot(batch_indices, val_balance_history, 'o-', color='blue')
    plt.axhline(y=10000, color='r', linestyle='--', label='Initial')
    plt.title('Validation Balance (per batch)')
    plt.xlabel('Episode')
    plt.ylabel('Balance (PLN)')
    plt.legend()
    plt.grid(True, alpha=0.3)

# MA Rewards
plt.subplot(3, 3, 4)
window = 20  # ← ZWIĘKSZONE z 10 (więcej epizodów = większe okno MA)
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
    plt.title('Actor Loss (per batch)')
    plt.xlabel('Episode')
    plt.ylabel('Actor Loss')
    plt.grid(True, alpha=0.3)

# Critic Loss
plt.subplot(3, 3, 6)
if len(critic_loss_history) > 0:
    batch_indices = [i * BATCH_SIZE for i in range(1, len(critic_loss_history) + 1)]
    plt.plot(batch_indices, critic_loss_history, 'o-', color='purple')
    plt.title('Critic Loss (per batch)')
    plt.xlabel('Episode')
    plt.ylabel('Critic Loss')
    plt.grid(True, alpha=0.3)

# Spread Z-Score
plt.subplot(3, 3, 7)
sample = df['spread_zscore'].iloc[-1000:]
plt.plot(sample.values)
plt.axhline(y=2, color='r', linestyle='--', alpha=0.5, label='Overbought')
plt.axhline(y=-2, color='g', linestyle='--', alpha=0.5, label='Oversold')
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
plt.title('Spread Z-Score (last 1000min)')
plt.xlabel('Time')
plt.ylabel('Z-Score')
plt.legend()
plt.grid(True, alpha=0.3)

# Correlation
plt.subplot(3, 3, 8)
sample_corr = df['correlation_30'].iloc[-1000:]
plt.plot(sample_corr.values)
plt.title('WIG20-DAX Correlation (30-min)')
plt.xlabel('Time')
plt.ylabel('Correlation')
plt.grid(True, alpha=0.3)

# Training vs Validation
plt.subplot(3, 3, 9)
if len(val_rewards_history) > 0:
    train_per_batch = [np.mean(rewards_history[i*BATCH_SIZE:(i+1)*BATCH_SIZE])
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
plt.savefig('a2c_arbitrage_400days_results.png', dpi=150)
print("\n✓ Wykres zapisany: a2c_arbitrage_400days_results.png")


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
print(f"  - best_a2c_actor_400days.keras")
print(f"  - best_a2c_critic_400days.keras")
print(f"\n{'=' * 70}")

# Finalne podsumowanie zmian
print(f"\n{'=' * 70}")
print("📊 PODSUMOWANIE ZMIAN DLA ~400 DNI DANYCH")
print(f"{'=' * 70}")
print(f"\n✅ PARAMETRY TRENOWANIA:")
print(f"   episodes: 100 → {episodes} (+150)")
print(f"   BATCH_SIZE: 10 → {BATCH_SIZE} (-5)")
print(f"   Liczba batchy: 10 → {episodes // BATCH_SIZE} (+{episodes // BATCH_SIZE - 10})")
print(f"   MA window: 10 → 20 (+10)")
print(f"   Epsilon decay: 0.99 → 0.995 (wolniejszy)")
print(f"\n✅ LEARNING RATES:")
print(f"   actor_lr: 0.0005 → 0.0007 (+40%)")
print(f"   critic_lr: 0.001 → 0.0015 (+50%)")
print(f"\n✅ ENVIRONMENT (bez zmian - już zoptymalizowane):")
print(f"   position_size: 10 ✓")
print(f"   reward_scale: 5.0 ✓")
print(f"   transaction_cost: 0.000001 ✓")
print(f"\n💡 DLACZEGO TE ZMIANY?")
print(f"   - Więcej danych = potrzeba więcej epizodów (coverage ~1x)")
print(f"   - Mniejszy batch = częstsze aktualizacje = lepsze tracking")
print(f"   - Wyższe LR = szybsze uczenie (więcej danych = mniej ryzyka overfittingu)")
print(f"   - Wolniejszy epsilon decay = lepsza eksploracja przez dłuższy czas")
print(f"\n{'=' * 70}\n")