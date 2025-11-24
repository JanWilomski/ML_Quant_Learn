"""
A2C (Advantage Actor-Critic) dla arbitrażu WIG20-DAX

🔧 FIXED VERSION - Poprawione parametry dla lepszego uczenia!

ZMIANY vs oryginał:
- position_size: 1 → 10 (większe zyski)
- reward_scale: 50.0 → 5.0 (silniejsze sygnały uczenia)
- actor_lr: 0.0001 → 0.0005 (szybsze uczenie Actor)
- critic_lr: 0.0005 → 0.001 (szybsze uczenie Critic)
- transaction_cost: 0.00001 → 0.000001 (mniejsze koszty)

Powody zmian - patrz: A2C_Troubleshooting.md

KLUCZOWA RÓŻNICA vs Policy Gradient (plik 09):
- Dwie sieci: Actor (policy) + Critic (value)
- Trenowanie z TD error (advantage) zamiast Monte Carlo returns
- Niższa wariancja → szybsze, stabilniejsze uczenie!

Bazuje na: 09_policy_gradient_wig20_ger40_relative.py
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
#   TRADING ENVIRONMENT (identyczny jak w 09)
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
#   A2C AGENT - GŁÓWNA NOWOŚĆ!
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

        Hyperparameters:
        - actor_lr: learning rate dla Actor (może być niższy niż critic)
        - critic_lr: learning rate dla Critic (może być wyższy)
        - temperature: kontrola exploration (jak w Policy Gradient)
        - epsilon: dodatkowe exploration
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
        """
        ACTOR: State → Action probabilities
        Identyczna architektura jak Policy Gradient!
        """
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
        """
        CRITIC: State → Value V(s)

        NOWA SIEĆ! Estymuje wartość state'u
        Output: single scalar (np. 123.5)
        """
        model = keras.Sequential([
            layers.Input(shape=(self.state_size,)),
            layers.Dense(128, activation='relu',
                         kernel_initializer='he_normal'),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu',
                         kernel_initializer='he_normal'),
            layers.Dense(32, activation='relu',
                         kernel_initializer='he_normal'),
            layers.Dense(1, activation='linear',  # ← Single value output!
                         kernel_initializer='zeros')  # Start from zero
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                      loss='mse')
        return model

    def act(self, state, greedy=False):
        """
        Wybierz akcję używając Actor network
        Identyczne jak w Policy Gradient
        """
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
        A2C Training - GŁÓWNA RÓŻNICA vs Policy Gradient!

        Kroki:
        1. Oblicz TD targets: r + γ*V(s')
        2. Oblicz advantages: TD_target - V(s)
        3. Trenuj Critic: minimalizuj (V(s) - TD_target)²
        4. Trenuj Actor: maksymalizuj log(π) * advantage

        Args:
            states: array of states
            actions: array of actions taken
            rewards: array of immediate rewards
            next_states: array of next states (NOWE vs PG!)
            dones: array of done flags (NOWE vs PG!)
        """
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)

        # ============================================
        #   KROK 1: Oblicz TD TARGETS i ADVANTAGES
        # ============================================

        # Critic predictions dla current i next states
        values = self.critic.predict(states, verbose=0).flatten()
        next_values = self.critic.predict(next_states, verbose=0).flatten()

        # TD targets: r + γ * V(s') * (1 - done)
        # (1 - done) bo jeśli done=True, nie ma next_state
        td_targets = rewards + self.gamma * next_values * (1 - dones)

        # Advantages (TD error):
        # advantage > 0 → akcja LEPSZA od średniej
        # advantage < 0 → akcja GORSZA od średniej
        advantages = td_targets - values

        # Normalizacja advantage (stabilność!)
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        # ============================================
        #   KROK 2: Trenuj CRITIC (minimalizuj TD error)
        # ============================================

        # Critic uczy się przewidywać TD targets
        critic_loss = self.critic.fit(
            states,
            td_targets,
            epochs=1,
            verbose=0,
            batch_size=min(32, len(states))
        ).history['loss'][0]

        # ============================================
        #   KROK 3: Trenuj ACTOR (policy gradient z advantage)
        # ============================================

        entropy_coef = 0.01

        with tf.GradientTape() as tape:
            # Actor predictions
            logits = self.actor(states, training=True)
            logits_scaled = logits / self.temperature
            logits_scaled = tf.clip_by_value(logits_scaled, -2.0, 2.0)
            action_probs = tf.nn.softmax(logits_scaled, axis=-1)

            # Wybierz prawdopodobieństwa dla wykonanych akcji
            indices = tf.range(len(actions)) * self.action_size + actions
            action_probs_for_actions = tf.gather(tf.reshape(action_probs, [-1]), indices)

            # Policy Gradient Loss (z advantage zamiast returns!)
            log_probs = tf.math.log(action_probs_for_actions + 1e-8)

            # Entropy bonus (zachęta do exploration)
            entropy = -tf.reduce_sum(action_probs * tf.math.log(action_probs + 1e-8), axis=1)

            # Actor loss: -log(π) * advantage + entropy_bonus
            advantages_tf = tf.constant(advantages, dtype=tf.float32)
            actor_loss = tf.reduce_mean(-(log_probs * advantages_tf + entropy_coef * entropy))

        # Update Actor
        gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(gradients, self.actor.trainable_variables))

        # Zwróć statystyki
        return {
            'actor_loss': float(actor_loss.numpy()),
            'critic_loss': float(critic_loss),
            'mean_advantage': float(np.mean(advantages)),
            'mean_value': float(np.mean(values)),
            'mean_td_target': float(np.mean(td_targets))
        }


# ============================================
#   ŁADOWANIE I PRZYGOTOWANIE DANYCH
# ============================================

print("=" * 70)
print("A2C (Advantage Actor-Critic) - Arbitraż WIG20-DAX")
print("=" * 70)
print("\nWczytuję dane WIG20 i DAX...")

# WIG20
wig20 = pd.read_csv('data/PL20.proM1.csv')
wig20['datetime'] = pd.to_datetime(wig20['datetime'])
wig20 = wig20.sort_values('datetime').reset_index(drop=True)

print(f"WIG20 RAW: {len(wig20)} wierszy, {wig20['datetime'].min()} - {wig20['datetime'].max()}")

# DAX - z poprawkami formatu i czasu
print("\n📊 Wczytuję DAX (z poprawką formatu i czasu)...")

try:
    dax_raw = pd.read_csv('data/ger40_m1.csv', sep=';', header=0)
    dax_raw.columns = ['datetime_raw', 'open', 'high', 'low', 'close', 'volume']

    dax_raw['datetime'] = pd.to_datetime(dax_raw['datetime_raw'], format='%Y%m%d %H%M%S')
    dax_raw['datetime'] = dax_raw['datetime'] + pd.Timedelta(hours=6)

    dax = dax_raw[['datetime', 'open', 'high', 'low', 'close', 'volume']].copy()
    dax = dax.sort_values('datetime').reset_index(drop=True)

    print(f"DAX po poprawkach: {len(dax)} wierszy")
    print(f"Zakres: {dax['datetime'].min()} - {dax['datetime'].max()}")

except Exception as e:
    print(f"❌ BŁĄD przy wczytywaniu DAX: {e}")
    print("Sprawdź czy masz plik data/ger40_m1.csv")
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

TRADING_DAY_MINUTES = int(minutes_per_day.median())
print(f"\nMAX_EPISODE_STEPS = {TRADING_DAY_MINUTES} minut")

# Features list
features = [
    'wig20_returns', 'wig20_distance_from_open', 'wig20_high_low_range',
    'wig20_position_in_range', 'wig20_price_to_sma5', 'wig20_price_to_sma15',
    'wig20_price_to_sma60', 'wig20_sma5_return', 'wig20_sma15_return',
    'wig20_volatility', 'dax_returns', 'dax_price_to_sma5',
    'dax_price_to_sma15', 'dax_price_to_sma60', 'dax_sma5_return',
    'dax_sma15_return', 'dax_volatility', 'spread_zscore',
    'spread_change', 'spread_pct_change', 'spread_acceleration',
    'correlation_30', 'correlation_60', 'dax_returns_lag1',
    'dax_returns_lag2', 'dax_returns_lag3', 'price_ratio_deviation',
    'momentum_divergence', 'volatility_ratio', 'volatility_spread',
    'wig20_volume_ratio', 'hour_sin', 'hour_cos', 'time_of_day'
]

print(f"\n✅ Total: {len(features)} PURE RELATIVE features")

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

print(f"Train: {len(train_data)} minut ({len(train_data) / TRADING_DAY_MINUTES:.1f} dni)")
print(f"Val:   {len(val_data)} minut ({len(val_data) / TRADING_DAY_MINUTES:.1f} dni)")
print(f"Test:  {len(test_data)} minut ({len(test_data) / TRADING_DAY_MINUTES:.1f} dni)")


# ============================================
#   AGENT I ENVIRONMENTS
# ============================================

state_size = len(features) + 5
agent = A2CAgent(
    state_size=state_size,
    action_size=3,
    actor_lr=0.0005,   # ← FIXED: było 0.0001
    critic_lr=0.001,   # ← FIXED: było 0.0005
    temperature=5.0,
    epsilon=0.1
)

train_env = ArbitrageEnvironment(
    train_data,
    initial_balance=10000,
    position_size=10,  # ← FIXED: było 1
    max_episode_steps=TRADING_DAY_MINUTES,
    random_start=True,
    features=features,
    reward_scale=5.0,  # ← FIXED: było 50.0
    transaction_cost=0.000001  # ← FIXED: było 0.00001
)

val_env = ArbitrageEnvironment(
    val_data,
    initial_balance=10000,
    position_size=10,  # ← FIXED: było 1
    max_episode_steps=None,
    random_start=False,
    features=features,
    reward_scale=5.0,  # ← FIXED: było 50.0
    transaction_cost=0.000001  # ← FIXED: było 0.00001
)

test_env = ArbitrageEnvironment(
    test_data,
    initial_balance=10000,
    position_size=10,  # ← FIXED: było 1
    max_episode_steps=None,
    random_start=False,
    features=features,
    reward_scale=5.0,  # ← FIXED: było 50.0
    transaction_cost=0.000001  # ← FIXED: było 0.00001
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

print(f"\n{'=' * 70}")
print("🚀 Rozpoczynam trening A2C...")
print(f"   Epizod = {TRADING_DAY_MINUTES} minut (pełny dzień giełdowy)")
print(f"   Batch training co 10 epizodów")
print(f"{'=' * 70}\n")

episodes = 100
BATCH_SIZE = 10

best_val_reward = -float('inf')

batch_states = []
batch_actions = []
batch_rewards = []
batch_next_states = []  # NOWE dla A2C!
batch_dones = []        # NOWE dla A2C!

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

        # Dodaj wartość z Critic
        first_value = agent.critic.predict(state.reshape(1, -1), verbose=0)[0][0]

        tqdm.write(
            f"\nBatch {(episode + 1) // BATCH_SIZE} - "
            f"Probs: HOLD={first_probs[0]:.3f}, BUY={first_probs[1]:.3f}, SELL={first_probs[2]:.3f}, "
            f"Value={first_value:.2f}"
        )

    total_reward = 0.0
    done = False
    states = []
    actions = []
    rewards_ep = []
    next_states = []  # NOWE!
    dones = []        # NOWE!
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

            # Zbieraj dane dla A2C
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

        # Trenuj A2C (z next_states i dones!)
        train_stats = agent.train(
            batch_states,
            batch_actions,
            batch_rewards,
            batch_next_states,  # NOWE!
            batch_dones         # NOWE!
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
            agent.actor.save('best_a2c_actor.keras')
            agent.critic.save('best_a2c_critic.keras')
            tqdm.write(f"   ✅ Nowy najlepszy! Val Reward: {val_reward:.2f}")

        tqdm.write(f"{'=' * 70}\n")

        agent.epsilon = max(0.01, agent.epsilon * 0.99)

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
window = 10
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
    # Average train per batch
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
plt.savefig('a2c_arbitrage_results.png', dpi=150)
print("\n✓ Wykres zapisany: a2c_arbitrage_results.png")


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
print(f"Modele zapisane:")
print(f"  - best_a2c_actor.keras")
print(f"  - best_a2c_critic.keras")
print(f"\n{'=' * 70}")