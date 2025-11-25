"""
A2C (Advantage Actor-Critic) dla tickowych danych WIG20 - PYTORCH VERSION

🔥 DLACZEGO PYTORCH:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Lepszy GPU support (zawsze działa!)
✅ Prostsze API (bardziej Pythoniczne)
✅ Automatic Mixed Precision (AMP) - jeszcze szybsze
✅ Łatwiejszy debugging
✅ Szybszy training loop

EXPECTED PERFORMANCE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CPU: ~3-4 godziny
GPU (bez AMP): ~20-30 minut
GPU (z AMP): ~10-15 minut ⚡

TICKOWE TIMEFRAMES (15s):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1 min  = 4 ticki
5 min  = 20 ticków
15 min = 60 ticków
1 hour = 240 ticków
Dzień  = 1800 ticków (9:00-16:30)
"""

import sys
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

# ============================================
#   GPU SETUP
# ============================================

print("=" * 70)
print("🔥 A2C PYTORCH dla tickowych danych WIG20")
print("=" * 70)

# Sprawdź GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n🖥️  Device: {device}")

if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    USE_AMP = True  # Automatic Mixed Precision
    print(f"⚡ Mixed Precision (AMP): Enabled")
else:
    print("❌ No GPU - using CPU")
    USE_AMP = False

print(f"\n{'=' * 70}\n")


# ============================================
#   TRADING ENVIRONMENT (bez zmian)
# ============================================

class TickTradingEnvironment:
    def __init__(self, data, initial_balance=10000, position_size=5,
                 max_episode_steps=None, random_start=False, features=None,
                 reward_scale=2.0, transaction_cost=0.000005):
        """
        Environment dla tickowych danych (co 15 sekund)
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
        current_price = self.data.iloc[self.current_step]['price_raw']

        if action == 0:
            if self.position is not None:
                reward = -0.0001

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
#   PYTORCH NEURAL NETWORKS
# ============================================

class Actor(nn.Module):
    """Actor Network - Policy π(a|s)"""

    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, state):
        return self.network(state)


class Critic(nn.Module):
    """Critic Network - Value V(s)"""

    def __init__(self, state_size):
        super(Critic, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, state):
        return self.network(state)


# ============================================
#   PYTORCH A2C AGENT
# ============================================

class PyTorchA2CAgent:
    def __init__(self, state_size, action_size,
                 actor_lr=0.0005, critic_lr=0.001,
                 gamma=0.95, temperature=5.0, epsilon=0.1):
        """
        PyTorch A2C Agent z GPU support i AMP
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.temperature = temperature
        self.epsilon = epsilon
        self.device = device

        # Sieci
        self.actor = Actor(state_size, action_size).to(device)
        self.critic = Critic(state_size).to(device)

        # Optymizatory
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # AMP scaler (dla mixed precision)
        self.scaler = GradScaler() if USE_AMP else None

        print(f"\n🎭 PyTorch A2C Agent:")
        print(f"   State size: {state_size}")
        print(f"   Device: {device}")
        print(f"   AMP: {USE_AMP}")
        print(f"   Actor params: {sum(p.numel() for p in self.actor.parameters()):,}")
        print(f"   Critic params: {sum(p.numel() for p in self.critic.parameters()):,}\n")

    def act(self, state, greedy=False):
        """Wybierz akcję"""
        # Epsilon-greedy
        if not greedy and np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)

        # Policy
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            logits = self.actor(state_tensor)
            logits_scaled = logits / self.temperature
            logits_scaled = torch.clamp(logits_scaled, -2.0, 2.0)
            probs = torch.softmax(logits_scaled, dim=-1)

            if greedy:
                action = torch.argmax(probs).item()
            else:
                action = torch.multinomial(probs, 1).item()

        return action

    def train(self, states, actions, rewards, next_states, dones):
        """
        Train Actor and Critic
        """
        # Convert to tensors
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        dones_tensor = torch.FloatTensor(dones).to(self.device)

        # ============================================
        #   CRITIC UPDATE
        # ============================================

        with torch.no_grad():
            values = self.critic(states_tensor).squeeze()
            next_values = self.critic(next_states_tensor).squeeze()
            td_targets = rewards_tensor + self.gamma * next_values * (1 - dones_tensor)
            advantages = td_targets - values

            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Critic loss
        if USE_AMP:
            with autocast():
                values_pred = self.critic(states_tensor).squeeze()
                critic_loss = nn.MSELoss()(values_pred, td_targets)

            self.critic_optimizer.zero_grad()
            self.scaler.scale(critic_loss).backward()
            self.scaler.step(self.critic_optimizer)
            self.scaler.update()
        else:
            values_pred = self.critic(states_tensor).squeeze()
            critic_loss = nn.MSELoss()(values_pred, td_targets)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

        # ============================================
        #   ACTOR UPDATE
        # ============================================

        entropy_coef = 0.01

        if USE_AMP:
            with autocast():
                logits = self.actor(states_tensor)
                logits_scaled = logits / self.temperature
                logits_scaled = torch.clamp(logits_scaled, -2.0, 2.0)
                probs = torch.softmax(logits_scaled, dim=-1)

                # Log probs dla wybranych akcji
                log_probs = torch.log(probs.gather(1, actions_tensor.unsqueeze(1)).squeeze() + 1e-8)

                # Entropy
                entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()

                # Actor loss
                actor_loss = -(log_probs * advantages.detach() + entropy_coef * entropy).mean()

            self.actor_optimizer.zero_grad()
            self.scaler.scale(actor_loss).backward()
            self.scaler.step(self.actor_optimizer)
            self.scaler.update()
        else:
            logits = self.actor(states_tensor)
            logits_scaled = logits / self.temperature
            logits_scaled = torch.clamp(logits_scaled, -2.0, 2.0)
            probs = torch.softmax(logits_scaled, dim=-1)

            log_probs = torch.log(probs.gather(1, actions_tensor.unsqueeze(1)).squeeze() + 1e-8)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()

            actor_loss = -(log_probs * advantages.detach() + entropy_coef * entropy).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'mean_advantage': advantages.mean().item(),
            'mean_value': values.mean().item()
        }

    def save(self, path_actor, path_critic):
        """Zapisz modele"""
        torch.save(self.actor.state_dict(), path_actor)
        torch.save(self.critic.state_dict(), path_critic)

    def load(self, path_actor, path_critic):
        """Wczytaj modele"""
        self.actor.load_state_dict(torch.load(path_actor))
        self.critic.load_state_dict(torch.load(path_critic))


# ============================================
#   ŁADOWANIE DANYCH
# ============================================

print("📊 Wczytuję tickowe dane WIG20...")

df = pd.read_csv('data/wig20_tick_data.csv')

print(f"Format pliku:")
print(df.head())

df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values('datetime').reset_index(drop=True)

print(f"\nRAW DATA: {len(df)} ticków")
print(f"Zakres: {df['datetime'].min()} - {df['datetime'].max()}")

if 'price' not in df.columns:
    print("❌ BŁĄD: Brak kolumny 'price'!")
    sys.exit(1)

# Filtruj godziny giełdowe
print(f"\n🕐 Filtruję godziny giełdowe (9:00-16:30)...")
df['hour'] = df['datetime'].dt.hour
df['minute'] = df['datetime'].dt.minute

df_trading = df[
    ((df['hour'] >= 9) & (df['hour'] < 16)) |
    ((df['hour'] == 16) & (df['minute'] <= 30))
    ].copy()

print(f"Po filtrze: {len(df_trading)} ticków")

df = df_trading.set_index('datetime').sort_index()
df['price_raw'] = df['price'].copy()

# Feature Engineering
print(f"\n🎯 Feature Engineering...")

TICK_1MIN = 4
TICK_5MIN = 20
TICK_15MIN = 60
TICK_1H = 240

df['returns'] = df['price'].pct_change() * 100

df['hour'] = df.index.hour
df['minute'] = df.index.minute
minutes_since_open = (df['hour'] - 9) * 60 + df['minute']
df['time_of_day'] = minutes_since_open / 450

df['hour_sin'] = np.sin(2 * np.pi * df['time_of_day'])
df['hour_cos'] = np.cos(2 * np.pi * df['time_of_day'])

df['date'] = df.index.date
df['session_open'] = df.groupby('date')['price'].transform('first')
df['distance_from_open'] = (df['price'] / df['session_open'] - 1) * 100

df['sma_1min'] = df['price'].rolling(window=TICK_1MIN).mean()
df['sma_5min'] = df['price'].rolling(window=TICK_5MIN).mean()
df['sma_15min'] = df['price'].rolling(window=TICK_15MIN).mean()
df['sma_1h'] = df['price'].rolling(window=TICK_1H).mean()

df['price_to_sma_1min'] = (df['price'] / df['sma_1min'] - 1) * 100
df['price_to_sma_5min'] = (df['price'] / df['sma_5min'] - 1) * 100
df['price_to_sma_15min'] = (df['price'] / df['sma_15min'] - 1) * 100
df['price_to_sma_1h'] = (df['price'] / df['sma_1h'] - 1) * 100

df['sma_1min_return'] = df['sma_1min'].pct_change() * 100
df['sma_5min_return'] = df['sma_5min'].pct_change() * 100
df['sma_15min_return'] = df['sma_15min'].pct_change() * 100

df['volatility_1min'] = df['returns'].rolling(window=TICK_1MIN).std()
df['volatility_5min'] = df['returns'].rolling(window=TICK_5MIN).std()
df['volatility_15min'] = df['returns'].rolling(window=TICK_15MIN).std()

df['vol_of_vol'] = df['volatility_5min'].rolling(window=TICK_15MIN).std()

df['momentum_1min'] = df['returns'].rolling(window=TICK_1MIN).mean()
df['momentum_5min'] = df['returns'].rolling(window=TICK_5MIN).mean()

helper_cols = ['hour', 'minute', 'date', 'session_open',
               'sma_1min', 'sma_5min', 'sma_15min', 'sma_1h']

for col in helper_cols:
    if col in df.columns:
        df = df.drop(columns=[col])

df.dropna(inplace=True)

print(f"✅ Features utworzone: {len(df)} ticków")

features = [
    'returns', 'momentum_1min', 'momentum_5min',
    'distance_from_open',
    'price_to_sma_1min', 'price_to_sma_5min', 'price_to_sma_15min', 'price_to_sma_1h',
    'sma_1min_return', 'sma_5min_return', 'sma_15min_return',
    'volatility_1min', 'volatility_5min', 'volatility_15min', 'vol_of_vol',
    'hour_sin', 'hour_cos', 'time_of_day'
]

print(f"\n✅ {len(features)} features ready")

# Podział danych
total_len = len(df)
train_end = int(total_len * 0.70)
val_end = int(total_len * 0.85)

cols_for_env = features + ['price_raw']

train_data = df.iloc[:train_end][cols_for_env].copy()
val_data = df.iloc[train_end:val_end][cols_for_env].copy()
test_data = df.iloc[val_end:][cols_for_env].copy()

ticks_per_day = TICK_1H * 7.5

train_days = len(train_data) / ticks_per_day
val_days = len(val_data) / ticks_per_day
test_days = len(test_data) / ticks_per_day

print(f"\n📊 Podział:")
print(f"Train: {len(train_data)} ticków ({train_days:.1f} dni)")
print(f"Val:   {len(val_data)} ticków ({val_days:.1f} dni)")
print(f"Test:  {len(test_data)} ticków ({test_days:.1f} dni)")

# ============================================
#   AGENT I ENVIRONMENTS
# ============================================

MAX_EPISODE_STEPS = 480  # 2 godziny

state_size = len(features) + 5
agent = PyTorchA2CAgent(
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
    position_size=5,
    max_episode_steps=MAX_EPISODE_STEPS,
    random_start=True,
    features=features,
    reward_scale=2.0,
    transaction_cost=0.000005
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
#   TRENING
# ============================================

print(f"{'=' * 70}")
print("🚀 Rozpoczynam PYTORCH TRAINING...")
print(f"{'=' * 70}\n")

episodes = 200
BATCH_SIZE = 5

print(f"PARAMETRY:")
print(f"  Episodes: {episodes}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Device: {device}")
print(f"  AMP: {USE_AMP}")
print(f"\n{'=' * 70}\n")

training_start_time = time.time()

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
batch_times = []

for episode in range(episodes):
    batch_start_time = time.time()

    state = train_env.reset()

    if (episode + 1) % BATCH_SIZE == 0:
        # Log początkowych prawdopodobieństw
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            logits = agent.actor(state_tensor)
            logits_scaled = logits / agent.temperature
            probs = torch.softmax(logits_scaled, dim=-1).cpu().numpy()[0]
            value = agent.critic(state_tensor).item()

        tqdm.write(
            f"\nBatch {(episode + 1) // BATCH_SIZE}/{episodes // BATCH_SIZE} - "
            f"Probs: HOLD={probs[0]:.3f}, BUY={probs[1]:.3f}, SELL={probs[2]:.3f}, "
            f"V={value:.2f}"
        )

    total_reward = 0.0
    done = False
    states = []
    actions = []
    rewards_ep = []
    next_states = []
    dones = []
    last_info = None

    ep_len = train_env.max_episode_steps or len(train_data)

    with tqdm(total=ep_len, desc=f"Ep {episode + 1}/{episodes}",
              leave=False, position=0, file=sys.stdout, mininterval=0.5) as pbar:
        while not done:
            action = agent.act(state)
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

    # Train co BATCH_SIZE epizodów
    if (episode + 1) % BATCH_SIZE == 0:
        batch_num = (episode + 1) // BATCH_SIZE

        tqdm.write(f"\n{'=' * 70}")
        tqdm.write(f"🔄 BATCH {batch_num}/{episodes // BATCH_SIZE}")
        tqdm.write(f"   Trenuję na {len(batch_states)} krokach...")

        train_start = time.time()
        train_stats = agent.train(
            batch_states,
            batch_actions,
            batch_rewards,
            batch_next_states,
            batch_dones
        )
        train_time = time.time() - train_start

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

        batch_time = time.time() - batch_start_time
        batch_times.append(batch_time)
        avg_batch_time = np.mean(batch_times)

        tqdm.write(f"   Actor Loss: {train_stats['actor_loss']:.4f}")
        tqdm.write(f"   Critic Loss: {train_stats['critic_loss']:.4f}")
        tqdm.write(f"   Train time: {train_time:.2f}s")
        tqdm.write(f"   Batch time: {batch_time:.1f}s (avg: {avg_batch_time:.1f}s)")

        # Validation
        val_reward, val_balance, val_trades = test_agent(agent, val_env, n_runs=1)
        val_rewards_history.append(val_reward)
        val_balance_history.append(val_balance)

        tqdm.write(f"   Val: Reward={val_reward:.2f}, Balance={val_balance:.2f}")

        if val_reward > best_val_reward:
            best_val_reward = val_reward
            agent.save('best_pytorch_actor.pth', 'best_pytorch_critic.pth')
            tqdm.write(f"   ✅ Best model! Val Reward: {val_reward:.2f}")

        # ETA
        batches_done = batch_num
        batches_total = episodes // BATCH_SIZE
        batches_left = batches_total - batches_done
        eta_seconds = batches_left * avg_batch_time
        eta_minutes = eta_seconds / 60

        tqdm.write(f"   ⏱️  ETA: {eta_minutes:.1f} minut")
        tqdm.write(f"{'=' * 70}\n")

        # Epsilon decay
        agent.epsilon = max(0.01, agent.epsilon * 0.995)

training_time = time.time() - training_start_time

print(f"\n✅ Trening zakończony!")
print(f"⏱️  Całkowity czas: {training_time / 60:.1f} minut")
print(f"⚡ GPU: {'YES' if torch.cuda.is_available() else 'NO'}")

# ============================================
#   WYKRESY
# ============================================

plt.figure(figsize=(18, 12))

plt.subplot(3, 3, 1)
plt.plot(rewards_history, alpha=0.6)
plt.title('Training Reward')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.grid(True, alpha=0.3)

plt.subplot(3, 3, 2)
if val_rewards_history:
    batch_indices = [i * BATCH_SIZE for i in range(1, len(val_rewards_history) + 1)]
    plt.plot(batch_indices, val_rewards_history, 'o-', color='green')
    plt.title('Validation Reward')
    plt.xlabel('Episode')
    plt.ylabel('Val Reward')
    plt.grid(True, alpha=0.3)

plt.subplot(3, 3, 3)
if val_balance_history:
    batch_indices = [i * BATCH_SIZE for i in range(1, len(val_balance_history) + 1)]
    plt.plot(batch_indices, val_balance_history, 'o-', color='blue')
    plt.axhline(y=10000, color='r', linestyle='--', label='Initial')
    plt.title('Validation Balance')
    plt.xlabel('Episode')
    plt.ylabel('Balance (PLN)')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.subplot(3, 3, 4)
if len(batch_times) > 0:
    batch_indices = [i * BATCH_SIZE for i in range(1, len(batch_times) + 1)]
    plt.plot(batch_indices, batch_times, 'o-', color='purple')
    plt.title('Batch Training Time')
    plt.xlabel('Episode')
    plt.ylabel('Time (seconds)')
    plt.grid(True, alpha=0.3)

plt.subplot(3, 3, 5)
if actor_loss_history:
    batch_indices = [i * BATCH_SIZE for i in range(1, len(actor_loss_history) + 1)]
    plt.plot(batch_indices, actor_loss_history, 'o-', color='orange')
    plt.title('Actor Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)

plt.subplot(3, 3, 6)
if critic_loss_history:
    batch_indices = [i * BATCH_SIZE for i in range(1, len(critic_loss_history) + 1)]
    plt.plot(batch_indices, critic_loss_history, 'o-', color='red')
    plt.title('Critic Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)

plt.subplot(3, 3, 7)
window = 20
if len(rewards_history) >= window:
    ma = pd.Series(rewards_history).rolling(window=window).mean()
    plt.plot(ma)
    plt.title(f'Reward MA-{window}')
    plt.xlabel('Episode')
    plt.ylabel('MA Reward')
    plt.grid(True, alpha=0.3)

plt.subplot(3, 3, 8)
if val_rewards_history:
    train_per_batch = [np.mean(rewards_history[i * BATCH_SIZE:(i + 1) * BATCH_SIZE])
                       for i in range(len(val_rewards_history))]
    batch_indices = [i * BATCH_SIZE for i in range(1, len(val_rewards_history) + 1)]
    plt.plot(batch_indices, train_per_batch, 'o-', label='Train', alpha=0.7)
    plt.plot(batch_indices, val_rewards_history, 'o-', label='Val', alpha=0.7)
    plt.title('Train vs Val')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.subplot(3, 3, 9)
perf_text = f"PyTorch A2C\n\n"
perf_text += f"Device: {device}\n"
perf_text += f"Training time: {training_time / 60:.1f} min\n"
perf_text += f"AMP: {USE_AMP}\n"
perf_text += f"Avg batch: {np.mean(batch_times) if batch_times else 0:.1f}s"
plt.text(0.5, 0.5, perf_text, ha='center', va='center', fontsize=12)
plt.axis('off')
plt.title('Performance')

plt.tight_layout()
plt.savefig('pytorch_a2c_results.png', dpi=150)
print(f"\n✓ Wykres: pytorch_a2c_results.png")

# ============================================
#   TEST
# ============================================

print(f"\n{'=' * 70}")
print("🧪 TEST")
print(f"{'=' * 70}\n")

test_reward, test_balance, test_trades = test_agent(agent, test_env, n_runs=5)

print(f"Test Reward: {test_reward:.2f}")
print(f"Test Balance: {test_balance:.2f}")
print(f"Test Profit: {test_balance - 10000:.2f} PLN")
print(f"Test Trades: {test_trades:.0f}")
if test_trades > 0:
    print(f"Profit/Trade: {(test_balance - 10000) / test_trades:.2f} PLN")

print(f"\n{'=' * 70}")
print("✅ PYTORCH A2C ZAKOŃCZONY!")
print(f"{'=' * 70}")
print(f"\nBest Val Reward: {best_val_reward:.2f}")
print(f"Training time: {training_time / 60:.1f} minut")
print(f"Device: {device}")
print(f"\nModele:")
print(f"  - best_pytorch_actor.pth")
print(f"  - best_pytorch_critic.pth")
print(f"\n{'=' * 70}\n")