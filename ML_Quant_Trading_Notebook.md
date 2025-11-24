# Machine Learning & Quant Trading
## Reference Guide - Najważniejsze Lekcje

**Jan Wilczyński | ERSTE Securities**

---

# 📚 SPIS TREŚCI

1. SUPERVISED LEARNING - Pierwsze kroki
2. REINFORCEMENT LEARNING - DQN
3. POLICY GRADIENT - Breakthrough
4. ARBITRAŻ WIG20-DAX - Zaawansowany projekt
5. FEATURE ENGINEERING - Best Practices
6. KLUCZOWE KONCEPTY MATEMATYCZNE
7. SOFTWARE ENGINEERING - Dobre praktyki
8. CODE SNIPPETS - Gotowe rozwiązania
9. NAJWAŻNIEJSZE INSIGHTS
10. CO DALEJ - Roadmap

---

# 1. SUPERVISED LEARNING - Pierwsze kroki

## 1.1 Podstawy

Supervised learning to podejście gdzie model uczy się na danych historycznych z znanymi etykietami (labels). W kontekście tradingu: przewidujemy przyszły ruch ceny na podstawie obecnych wskaźników.

## 1.2 Twoja pierwsza implementacja - MLP

```python
model = keras.Sequential([
    layers.Dense(32, activation='relu', input_dim=11),
    layers.Dense(3, activation='softmax')
])

# Signal: 0=SELL, 1=HOLD, 2=BUY
# 3-class classification problem
```

## 1.3 Kluczowa lekcja - OVERFITTING

**Problem:** Model osiągnął 83% accuracy na train, ale tylko 34% na validation!

**Dlaczego?**
- Rynki finansowe są inherently noisy
- Historyczne wzorce NIE gwarantują przyszłych wyników
- Model 'zapamiętał' training data zamiast nauczyć się ogólnych zasad
- Validation data (2025) to 'out-of-sample' - model nigdy tego nie widział

## 1.4 LSTM - czy sekwencje pomagają?

Testowałeś LSTM (Long Short-Term Memory) dla sekwencji czasowych. Rezultat: **nie pomógł znacząco**. Dlaczego?

- Rynki mają memory, ale krótkoterminowa
- Noise dominuje nad sygnałem w daily/hourly data
- LSTM potrzebuje dużo danych i długich zależności
- W finansach: regime changes > historical patterns

## ✅ CO SIĘ NAUCZYŁEŚ:

1. **Supervised learning ma limity w finansach** - przyszłość ≠ przeszłość
2. **Validation set jest kluczowy** - bez tego nie wiesz czy overfit
3. **Feature engineering > model architecture** - dobre features ważniejsze niż LSTM
4. **Relative measures lepsze niż raw values** - (price/SMA - 1) * 100

---

# 2. REINFORCEMENT LEARNING - DQN

## 2.1 Zmiana paradygmatu

Zamiast przewidywać przyszłość, **uczysz agenta podejmować decyzje**. Agent dostaje state (stan rynku), wybiera action (HOLD/BUY/SELL), otrzymuje reward (profit/loss).

## 2.2 Q-Learning - podstawy

```python
# Q-function: Q(state, action) = expected future reward
# Update rule (Bellman equation):
Q(s, a) = reward + gamma * max(Q(s', a'))

# gdzie:
# s = current state
# a = action taken
# s' = next state
# gamma = discount factor (0.95)
```

## 2.3 Epsilon-Greedy Strategy

**Exploration vs Exploitation dilemma:**

```python
if random() < epsilon:
    action = random_action()  # Explore
else:
    action = argmax(Q)  # Exploit

# epsilon = 1.0 na początku (100% exploration)
# epsilon decay = 0.995 (stopniowo maleje)
# epsilon_min = 0.01 (zawsze 1% exploration)
```

## 2.4 Problem który odkryłeś - Q-values explosion

**Symptom:** Q-wartości były ekstremalne:

```
Q(HOLD) = 15234.23
Q(BUY) = -8234.12
Q(SELL) = 25123.45
```

**Dlaczego to problem?**
- Gradient descent nie lubi dużych wartości
- Niestabilne uczenie (duże oscylacje)
- Trudno wybrać akcję (wszystkie Q podobnie duże/małe)
- Sparse rewards (reward = 0 większość czasu, tylko przy zamknięciu pozycji)

## ✅ CO SIĘ NAUCZYŁEŚ:

1. **Value-based methods (DQN) trudne dla continuous rewards**
2. **Q-values potrzebują normalizacji** - clipping, scaling
3. **Sparse rewards = problem** - agent nie wie co robi dobrze
4. **Trzeba policy-based approach** → Policy Gradient!

---

# 3. POLICY GRADIENT - Breakthrough 🚀

## 3.1 Fundamentalna różnica

| | DQN (Value-based) | Policy Gradient (Policy-based) |
|---|---|---|
| **Co uczy się?** | Q(state, action) "Jak dobra jest akcja" | π(action\|state) "Prawdopodobieństwa akcji" |
| **Output sieci** | Q-wartości [15234, -8234, 25123] | Prawdopodobieństwa [0.45, 0.27, 0.28] |
| **Wybór akcji** | argmax(Q) | Sample z rozkładu |
| **Problem** | Ekstremalne Q-values | Bardzo stabilne |

## 3.2 Temperature Scaling - TWOJE AHA MOMENT!

To była **KLUCZOWA** innowacja która sprawiła że Policy Gradient zadziałał:

```python
# PROBLEM: Surowe logity z sieci:
logits = [2.1, -0.5, 1.3]

# BEZ temperature → ekstremalne prawdopodobieństwa:
softmax([2.1, -0.5, 1.3]) = [0.73, 0.05, 0.22]
# Agent zbyt pewny! Mało exploration!

# Z TEMPERATURE = 5.0:
logits_scaled = [2.1/5, -0.5/5, 1.3/5] = [0.42, -0.1, 0.26]
softmax([0.42, -0.1, 0.26]) = [0.45, 0.27, 0.28]
# Bardziej równomierne! Więcej exploration!
```

**Intuicja:**
- Wysoka temperatura (5-10) → "płynne" prawdopodobieństwa → exploration
- Niska temperatura (1-2) → "ostre" prawdopodobieństwa → exploitation
- **Kontrolujesz exploration BEZ epsilon!**

## 3.3 Normalizacja Informacji o Pozycji

```python
# ❌ PRZED (surowe wartości - różne skale!):
state = [
    has_position,      # 0 lub 1
    entry_price,       # 2350.5
    position_pnl,      # 45.2
    balance            # 10045
]
# Skale: [0-1, 2000+, -100 to +100, 9000-11000] → NN się gubi!

# ✅ PO (wszystko relative, normalized):
entry_price_rel = (entry_price / current_price) - 1.0  # ~0.001
pnl_rel = position_pnl / current_price                 # ~0.02
balance_rel = (balance - 10000) / 10000                # ~0.0045
# Wszystko w zakresie [-1, 1] → NN lubi!
```

## 3.4 Reward Scaling & Clipping

```python
# ❌ PRZED:
reward = profit  # np. 235.5 PLN

# ✅ PO:
reward = np.clip(profit / reward_scale, -1.0, 1.0)
# reward_scale = 500 → reward zawsze w [-1, 1]

# Dlaczego?
# • Gradient descent lubi małe wartości
# • Stabilniejsze uczenie
# • Zapobiega reward explosion
```

## 3.5 Discounted Returns + Normalizacja

```python
def compute_returns(rewards, gamma=0.95):
    """Oblicz future value każdej akcji"""
    returns = []
    running_return = 0
    for r in reversed(rewards):
        running_return = r + gamma * running_return
        returns.append(running_return)
    returns.reverse()
    
    # NORMALIZACJA (kluczowe!):
    returns = (returns - mean) / (std + 1e-8)
    return returns
```

**Co to daje:**
- Akcje wczesne dostają 'kredyt' za późniejsze zyski
- Normalizacja → stabilniejsze gradienty
- Gamma (0.95) → 'jak daleko patrzysz w przyszłość'

## 3.6 Entropy Bonus

```python
loss = -(log_probs * returns + 0.01 * entropy)
#                              ↑ zachęta do exploration

# Entropy = miara 'losowości' policy
# Wysoka entropy → równomierne prawdopodobieństwa → exploration
# Niska entropy → pewne akcje → exploitation
# Bonus 0.01 * entropy → agent nie staje się zbyt pewny za szybko
```

## ✅ KLUCZOWE INNOWACJE:

1. **Temperature scaling** - kontrola exploration przez skalowanie logitów
2. **Relative position info** - wszystko normalized do [-1, 1]
3. **Reward scaling** - clip do [-1, 1] dla stabilności
4. **Discounted returns** - akcje wczesne dostają kredyt za przyszłość
5. **Entropy bonus** - zapobiega zbyt szybkiemu convergence

---

# 4. ARBITRAŻ WIG20-DAX - Zaawansowany Projekt

## 4.1 Koncepcja Arbitrażu

WIG20 i DAX są ze sobą powiązane (cointegrated) - długoterminowo poruszają się razem. ALE krótkoterminowo mogą się rozjechać. **Mean reversion** = gdy spread odejdzie za daleko od średniej, prawdopodobnie wróci.

## 4.2 Spread Z-Score - NAJWAŻNIEJSZY FEATURE!

```python
# Krok 1: Normalizuj obie ceny do 100 (baseline)
wig20_normalized = (wig20_close / wig20_close[0]) * 100
dax_normalized = (dax_close / dax_close[0]) * 100

# Krok 2: Oblicz spread (różnica)
spread = wig20_normalized - dax_normalized

# Krok 3: Z-score (ile sigma od średniej?)
spread_sma = spread.rolling(30).mean()
spread_std = spread.rolling(30).std()
spread_zscore = (spread - spread_sma) / spread_std
```

## 4.3 Interpretacja Z-Score

- **z = 0** → spread na średniej (normalne)
- **z > +2** → WIG20 za drogi względem DAX → **SELL WIG20**
- **z < -2** → WIG20 za tani względem DAX → **BUY WIG20**
- **z ≈ 0** → brak możliwości arbitrażu → **HOLD**

**Matematyka:**
- Z-score = (x - μ) / σ
- Rozkład normalny: ~95% wartości w [-2σ, +2σ]
- Więc z > +2 = top 2.5% ekstremów → okazja arbitrażu!

## 4.4 Kluczowe wyzwania techniczne

### A. Timezone & Data Quality

```python
# Problem: DAX miał 6h offset w timestampach!
# 20250101 180200 → to było 00:02, nie 18:02!

# Fix:
dax['datetime'] = pd.to_datetime(dax['datetime_raw']) + pd.Timedelta(hours=6)

# Weryfikacja:
# 'Czy 80%+ danych jest w 9-17?' → TAK = OK
```

### B. Trading Hours Only

Arbitraż działa TYLKO gdy oba rynki otwarte! 9:00-16:30 = 450 minut.

```python
trading_hours = df[
    ((df['hour'] >= 9) & (df['hour'] < 16)) |
    ((df['hour'] == 16) & (df['minute'] <= 30))
]
```

### C. Episode = Complete Trading Day

```python
# ❌ PRZED: episode = losowe 500 kroków
# Problem: epizod kończy się w środku dnia, mieszanie dni

# ✅ PO: episode = 451 minut (pełen dzień 9:00-16:30)
MAX_EPISODE_STEPS = 451

# Usuwanie niekompletnych dni:
minutes_per_day = df.groupby('date').size()
complete_days = minutes_per_day[minutes_per_day >= 451 - 10]
df = df[df.index.date.isin(complete_days.index)]
```

**Dlaczego ważne:**
- Intraday patterns mają sens tylko w pełnym dniu
- Mean reversion działa w ramach sesji
- Nie mieszamy danych z różnych dni

## 4.5 Batch Training

```python
# Zamiast trenować po każdym epizodzie (niestabilne):
# Zbieraj 10 epizodów, potem trenuj na wszystkich naraz

BATCH_SIZE = 10
batch_states = []
batch_actions = []
batch_rewards = []

for episode in range(100):
    # ... graj epizod ...
    batch_states.extend(states)
    batch_actions.extend(actions)
    batch_rewards.extend(rewards)
    
    if (episode + 1) % BATCH_SIZE == 0:
        agent.train(batch_states, batch_actions, batch_rewards)
        batch_states = []  # Clear
```

**Dlaczego lepsze:**
- Większy batch → stabilniejsze gradienty
- Mniej noise w uczeniu
- Lepsze estymacje returns

## 4.6 Najważniejsze Features dla Arbitrażu

| Feature | Opis | Dlaczego ważny |
|---------|------|----------------|
| spread_zscore | Ile sigma spread od średniej | **GŁÓWNY** sygnał arbitrażu |
| dax_returns_lag1/2/3 | DAX returns z poprzednich minut | Lead-lag effect (DAX prowadzi) |
| correlation_30/60 | Rolling correlation WIG20-DAX | Siła powiązania rynków |
| momentum_divergence | Różnica momentum WIG20-DAX | Rozbieżność trendów |
| volatility_ratio | wig20_vol / dax_vol | Risk assessment |
| time_of_day | Czas w sesji [0-1] | Volatility patterns (otwarcie/zamknięcie) |

## ✅ KLUCZOWE LEKCJE Z ARBITRAŻU:

1. **Data quality first** - timezone, market hours, completeness
2. **Spread z-score to core** - matematyczna podstawa arbitrażu
3. **Episode = trading day** - intraday patterns mają sens tylko w pełnym dniu
4. **Batch training stabilniejsze** - większe batche, mniej noise
5. **Lead-lag effect istnieje** - DAX prowadzi WIG20 o kilka minut

---

# 5. FEATURE ENGINEERING - Best Practices

## 5.1 Zasada #1: Relative > Absolute

Neural networks preferują wartości w małych zakresach. Surowe ceny są złe:

```python
# ❌ ZŁE (raw values, różne skale):
features = [wig20_close, dax_close]
# [2350, 19500] → ogromna różnica skal!

# ✅ DOBRE (relative, normalized):
wig20_returns = wig20_close.pct_change() * 100
price_to_sma = (close / sma - 1) * 100
# Wszystko w zakresie [-10, +10] → NN lubi!
```

## 5.2 Przykłady dobrych features

**Returns & Momentum:**
```python
returns = close.pct_change() * 100
sma_return = sma.pct_change() * 100
momentum = returns.rolling(10).mean()
```

**Distance from moving averages:**
```python
price_to_sma = (close / sma - 1) * 100
# Mówi: 'cena jest 2.5% powyżej SMA'
```

**Volatility:**
```python
volatility = returns.rolling(20).std()
volatility_ratio = wig20_vol / dax_vol
```

**Intraday patterns:**
```python
distance_from_open = (close / session_open - 1) * 100
position_in_range = (close - low) / (high - low)
hour_sin = sin(2π * time_of_day)  # cykliczność!
```

## 5.3 Normalizacja - StandardScaler

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_data)
val_scaled = scaler.transform(val_data)
test_scaled = scaler.transform(test_data)

# WAŻNE: fit tylko na train, transform na val/test!
# Inaczej data leakage!
```

## 5.4 Co NIE dawać do sieci

❌ **Surowe ceny** - zbyt duże wartości  
❌ **Volume bez normalizacji** - skaluje się o rzędy wielkości  
❌ **Timestamps/dates** - bez sensu dla NN  
❌ **Pomocnicze kolumny** - SMA używamy do obliczeń, ale nie dajemy do sieci  
✅ **Tylko derived, relative features**

---

# 6. KLUCZOWE KONCEPTY MATEMATYCZNE

## 6.1 Z-Score (Standardized Score)

Z-score mówi: 'o ile odchyleń standardowych wartość odbiega od średniej'

```
z = (x - μ) / σ

gdzie:
• x = aktualna wartość
• μ (mu) = średnia
• σ (sigma) = odchylenie standardowe

Interpretacja (rozkład normalny):
• ~68% wartości w [-1σ, +1σ]
• ~95% wartości w [-2σ, +2σ]
• ~99.7% wartości w [-3σ, +3σ]

Więc z > +2 = top 2.5% ekstremów!
```

## 6.2 Mean Reversion

Zasada: 'Co odchodzi od średniej, wraca do średniej'

**Warunki konieczne:**
1. **Stationary process** - średnia i wariancja stałe w czasie
2. **Cointegration** - dla par (WIG20-DAX są cointegrated)
3. **Half-life** - czas powrotu do średniej (im krótszy, tym lepiej)

**Ornstein-Uhlenbeck process** (matematyczny model mean reversion):
```
dx = θ(μ - x)dt + σdW

gdzie:
• θ = speed of mean reversion
• μ = long-term mean
• σ = volatility
• dW = Wiener process (random walk)
```

## 6.3 Cointegration

Dwie series czasowe są **cointegrated** jeśli:
- Każda z osobna jest non-stationary (random walk)
- ALE ich różnica (spread) jest stationary

**Test:** Engle-Granger cointegration test
- H0: brak cointegration
- H1: są cointegrated
- p-value < 0.05 → odrzucamy H0 → są cointegrated!

## 6.4 Discount Factor (Gamma)

W reinforcement learning: 'Jak bardzo cenisz przyszłe rewards?'

```
Return = r₀ + γr₁ + γ²r₂ + γ³r₃ + ...

gdzie:
• γ (gamma) ∈ [0, 1]
• γ = 0: tylko immediate reward
• γ = 0.95: patrzysz ~20 kroków w przyszłość
• γ = 0.99: patrzysz ~100 kroków w przyszłość
• γ = 1: nieskończony horyzont

Dla tradingu: γ = 0.95 jest dobrym kompromisem
```

## 6.5 Softmax & Temperature

Softmax converts logits → probabilities:

```
P(action_i) = exp(logit_i) / Σ exp(logit_j)

Z temperature scaling:
P(action_i) = exp(logit_i / T) / Σ exp(logit_j / T)

gdzie T = temperature:
• T → 0: argmax (deterministyczne)
• T = 1: standard softmax
• T > 1: bardziej równomierne (exploration)
• T >> 1: prawie uniform distribution
```

---

# 7. SOFTWARE ENGINEERING - Dobre Praktyki

## 7.1 Data Quality - ZAWSZE NAJPIERW!

✅ **Checklist przed treningiem:**

- [ ] Sprawdź zakres dat: min/max timestamp
- [ ] Weryfikuj timezone (szczególnie merged data!)
- [ ] Policz NaN/missing values
- [ ] Sprawdź czy dane mają sens (ceny > 0, volume >= 0)
- [ ] Dla intraday: weryfikuj market hours
- [ ] Dla arbitrażu: weryfikuj że timestampy się zgadzają
- [ ] Sprawdź kompletność dni (ile minut/dzień)
- [ ] Visualize sample data - czy wygląda OK?

## 7.2 Train/Val/Test Split

```python
# POPRAWNY split (chronological!):
total_len = len(df)
train_end = int(total_len * 0.70)  # 70% train
val_end = int(total_len * 0.85)    # 15% val
# 15% test

train_data = df.iloc[:train_end]
val_data = df.iloc[train_end:val_end]
test_data = df.iloc[val_end:]

# ❌ NIGDY random split dla time series!
# Data leakage: model widzi przyszłość!
```

## 7.3 Validation During Training

```python
# Test na validation co N episodes:
val_frequency = 5

for episode in range(100):
    # Train...
    
    if (episode + 1) % val_frequency == 0:
        val_reward = test_agent(agent, val_env)
        
        if val_reward > best_val_reward:
            best_val_reward = val_reward
            agent.model.save('best_model.keras')

# WAŻNE: Zapisuj model na podstawie VALIDATION, nie train!
```

## 7.4 Comprehensive Logging

```python
# Co logować podczas treningu:
print(f'Episode {ep}:')
print(f'  Train Reward: {train_reward:.2f}')
print(f'  Train Balance: {train_balance:.2f}')
print(f'  Action counts: HOLD={hold}, BUY={buy}, SELL={sell}')
print(f'  Val Reward: {val_reward:.2f}')
print(f'  Val Balance: {val_balance:.2f}')
print(f'  Epsilon: {agent.epsilon:.3f}')
print(f'  Probs: HOLD={p[0]:.3f}, BUY={p[1]:.3f}, SELL={p[2]:.3f}')

# Bez tego nie wiesz co się dzieje!
```

## 7.5 Multiple Test Runs

```python
# Jeden test to za mało (noise!):
def test_agent(agent, env, n_runs=5):
    rewards = []
    for _ in range(n_runs):
        # ... run episode ...
        rewards.append(total_reward)
    return np.mean(rewards), np.std(rewards)

# Raportuj: mean ± std
```

---

# 8. CODE SNIPPETS - Gotowe Rozwiązania

## 8.1 Policy Gradient Agent (minimal)

```python
class PolicyGradientAgent:
    def __init__(self, state_size, action_size, temperature=5.0):
        self.temperature = temperature
        self.model = self.build_model()
    
    def build_model(self):
        model = keras.Sequential([
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(action_size, activation='linear')
        ])
        return model
    
    def act(self, state):
        logits = self.model.predict(state.reshape(1, -1))[0]
        logits_scaled = logits / self.temperature
        probs = softmax(logits_scaled)
        return np.random.choice(action_size, p=probs)
```

## 8.2 Spread Z-Score Calculation

```python
# Normalizuj do 100
wig20_norm = (wig20 / wig20.iloc[0]) * 100
dax_norm = (dax / dax.iloc[0]) * 100

# Spread
spread = wig20_norm - dax_norm

# Z-score
spread_sma = spread.rolling(30).mean()
spread_std = spread.rolling(30).std()
spread_zscore = (spread - spread_sma) / (spread_std + 1e-8)
```

## 8.3 Relative Features Template

```python
# Returns
df['returns'] = df['close'].pct_change() * 100

# Distance from SMA
df['sma_20'] = df['close'].rolling(20).mean()
df['price_to_sma'] = (df['close'] / df['sma_20'] - 1) * 100

# Volatility
df['volatility'] = df['returns'].rolling(20).std()

# Intraday
df['dist_from_open'] = (df['close'] / session_open - 1) * 100
df['pos_in_range'] = (df['close'] - df['low']) / (df['high'] - df['low'])
```

## 8.4 Discounted Returns

```python
def compute_returns(rewards, gamma=0.95):
    returns = []
    running_return = 0
    for r in reversed(rewards):
        running_return = r + gamma * running_return
        returns.append(running_return)
    returns.reverse()
    
    # Normalize
    returns = np.array(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns
```

---

# 9. NAJWAŻNIEJSZE INSIGHTS Z WSZYSTKICH LEKCJI

## 9.1 Supervised Learning

✓ Ma limity w finansach - przyszłość ≠ przeszłość  
✓ Validation set kluczowy - bez tego nie wiesz czy overfit  
✓ Feature engineering > architecture  
✓ Overfitting jest normą, nie wyjątkiem  

## 9.2 Reinforcement Learning

✓ Policy Gradient > DQN dla continuous action spaces  
✓ Temperature scaling = najlepsza kontrola exploration  
✓ Normalize EVERYTHING - position info, rewards, features  
✓ Batch training stabilniejsze niż single-episode  
✓ Discounted returns + normalizacja = stabilne gradienty  

## 9.3 Arbitraż & Market Microstructure

✓ Data quality first - timezone, market hours, completeness  
✓ Spread z-score to matematyczna podstawa arbitrażu  
✓ Mean reversion działa (ale wymaga cointegration)  
✓ Lead-lag effects są realne (DAX prowadzi WIG20)  
✓ Episode = trading day dla intraday strategies  
✓ Transaction costs matter - bez nich model overfit do overtradingu  

## 9.4 Feature Engineering

✓ Relative measures > absolute values  
✓ Wszystko w małych zakresach (±10) dla NN  
✓ % changes, ratios, z-scores - nie surowe ceny  
✓ StandardScaler na features (fit tylko na train!)  
✓ Meaningful features > więcej features  

## 9.5 Software Engineering

✓ Comprehensive logging - bez tego debug niemożliwy  
✓ Multiple test runs - jeden to za mało (noise)  
✓ Save best model based on VALIDATION  
✓ Chronological split dla time series (NIGDY random!)  
✓ Incremental development - małe kroki, testuj często  

---

# 10. CO DALEJ - Roadmap Rozwoju

## 10.1 Najbliższe kroki (2-3 miesiące)

### Krok 1: Actor-Critic Methods (2-3 tygodnie)

- A2C/A3C - łączysz DQN (value) + Policy Gradient (policy)
- Actor uczy się π(a|s), Critic uczy się V(s)
- Stabilniejsze niż czysty Policy Gradient
- Implementuj na twoim WIG20-DAX arbitrage

### Krok 2: PPO - Proximal Policy Optimization (2-3 tygodnie)

- Standard w profesjonalnym quant RL
- Rozwiązuje 'trust region' problem
- Agent nie zmienia się za szybko (stabilność!)
- Benchmark: A2C vs PPO vs Policy Gradient

### Krok 3: Cointegration Testing (1-2 tygodnie)

- Matematycznie: Engle-Granger test na WIG20-DAX
- Sprawdź czy faktycznie są cointegrated
- Half-life mean reversion
- Użyj jako lepszego feature'a niż spread

## 10.2 Średni termin (3-6 miesięcy)

### Portfolio Optimization z RL

- Multi-asset trading (WIG20, DAX, S&P500)
- Agent uczy się alokacji kapitału
- Constraints: max drawdown, position limits
- Risk-adjusted rewards (Sharpe ratio jako reward)

### Multi-timeframe Strategies

- M1 dla execution, H1 dla strategy, D1 dla regime
- Hierarchical RL: macro-agent + micro-agent
- Regime detection (trending vs mean-reverting)

## 10.3 Zaawansowane (6+ miesięcy)

- **Bayesian Methods** - uncertainty quantification
- **Order flow & Market microstructure**
- **Alternative data** - sentiment, news
- **Options pricing** (jeśli chcesz derivatives)

## 10.4 Czego NIE robić teraz

❌ Transformers/Attention - to przeskok, najpierw Actor-Critic  
❌ Stochastic calculus - dopóki nie robisz opcji  
❌ Deep Q-Networks warianty (Rainbow) - masz już DQN, idź w policy-based  

---

# PODSUMOWANIE

Przeszedłeś systematyczną ścieżkę od supervised learning, przez DQN, do zaawansowanego Policy Gradient arbitrage trading. Po drodze nauczyłeś się:

1. **Machine Learning fundamentals** - supervised vs reinforcement learning
2. **Neural networks dla finansów** - co działa, co nie
3. **Feature engineering** - jak przygotować dane dla NN
4. **Reinforcement Learning** - od DQN do Policy Gradient
5. **Arbitrage trading** - WIG20-DAX statistical arbitrage
6. **Market microstructure** - timezone, trading hours, data quality
7. **Best practices** - validation, logging, testing

## Najważniejsze lekcje:

- Feature engineering > model architecture
- Data quality first
- Validation set jest kluczowy
- Policy Gradient > DQN dla tradingu
- Temperature scaling = game changer
- Batch training stabilniejsze
- Multiple test runs obowiązkowe

**Ten dokument to twój reference guide. Wracaj do niego gdy potrzebujesz przypomnienia konceptów, wzorów, lub best practices.**

---

**Powodzenia w dalszej nauce! 🚀**
