# XAUUSD Autonomous Scalper (Stabilized DQN)

## Overview
This repository implements a production-grade, prescriptive Deep Q-Network (DQN) algorithmic execution agent optimized for interaction with a simulated XAUUSD (Gold) market sandbox environment. Shifting away from fragile directional price prediction, the agent maps structural multi-dimensional state vectors directly to optimized execution commands via a decoupled reinforcement learning architecture designed to insulate gradient trajectories from non-stationary drift.

## Core Machine Learning Architecture

### 1. Markov Decision Process (MDP) Boundary Definition
The trading environment is modeled as a discrete-time Markov Decision Process operating under strict sequential index isolation:
* **State Space ($\mathcal{S}$):** A continuous tensor $s_t = [P_t, \text{Pos}_t]^T \in \mathbb{R}^2$, where $P_t$ represents the current normalized asset execution price and $\text{Pos}_t \in \{-1, 0, 1\}$ specifies the active portfolio inventory commitment (Short, Flat, Long).
* **Action Space ($\mathcal{A}$):** Discrete execution primitives mapping directly to order routing blocks:
  * `0`: Flat/Close (Liquidate inventory or hold current cash state)
  * `1`: Go Long (Establish or reverse into a long asset commitment)
  * `2`: Go Short (Establish or reverse into a short asset commitment)
* **Reward Function ($\mathcal{R}$):** Derived strictly from trade liquidation PnL combined with a structural step penalty to penalize systemic exposure inertia:
  $$\mathcal{R}_t = \Delta \text{PnL}_{\text{realized}} - 0.1 \cdot \mathbb{I}(\text{Pos}_t \neq 0)$$

### 2. Environment Dynamics (Synthetic Asset Generation)
The underlying simulation environment models a persistent trending gold framework overlaid with high-frequency noise profiles to establish a complex tracking sandbox:
$$P_t = 10 \cdot \sin\left(\frac{50t}{N}\right) + \left(\frac{50t}{N}\right) + 1900 + \mathcal{N}(0, 2)$$
Where $N$ represents total series scale (`data_length`) and $\mathcal{N}(0,2)$ defines localized Gaussian structural market noise.

### 3. Stability Optimization Suite
To eliminate the severe policy oscillation and moving-target divergence common in basic online reinforcement configurations, this system implements two critical decoupling pillars:
* **De-correlated Experience Replay:** Transitions $(s, a, r, s', d)$ are written to a fixed-capacity circular memory buffer $\mathcal{D}$. Training iterations execute optimization via uniform random mini-batch sampling, satisfying foundational $\text{i.i.d.}$ learning assumptions.
* **Decoupled Target Network Partition:** Two isomorphic neural networks are maintained: the *Online Network* ($\theta$) which handles immediate action derivation and loss optimization, and the *Target Network* ($\theta^-$) which functions strictly in evaluation mode to calculate stable Bellman targets:
  $$Y^{\text{DQN}}_t = r_t + \gamma \max_{a'} Q(s'_{t}, a'; \theta^-)(1 - d_t)$$
  The target weights are synchronized deterministically every $N$ episodes ($\theta^- \leftarrow \theta$).

---

## Repository Taxonomy & Dependency Governance
Project tracking and packaging adhere to modern PEP 517 build infrastructure standards utilizing `pyproject.toml` as the single source of truth for runtime configurations. Automated code hygiene is enforced deterministically via integrated configurations for:
* **Ruff & Black:** Formatted for a strict 100-character line constraint.
* **Mathematical Tensor Exceptions:** Strict PEP 8 uppercase token exceptions (`X`, `y`, `Q_target`) are whitelisted natively under `tool.ruff.lint.pep8-naming` to maximize quantitative readability without failing lint verification pipelines.

---

## Local Verification & Quickstart Execution

### Native Local Deployment
To initialize the deterministic environment, isolate testing extensions, and launch the training execution pipeline locally:

```bash
# 1. Establish python virtual environment layer
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# 2. Execute explicit dependency configuration install
pip install --upgrade pip
pip install -e .[dev]

# 3. Perform automated quality checks and execute the unit test suite
ruff check .
black --check .
pytest tests/ -v

# 4. Trigger the validation execution suite
python test_dqn.py
```

### Isolated Container Deployment
The application can be initialized inside an immutable, multi-stage runtime container environment configured to enforce absolute system clock temporal alignment with Universal Coordinated Time (UTC):

```bash
# Build the highly compressed production image layer
docker build -t xau-dqn-scalper:latest .

# Execute the containerized performance tracking pipeline
docker run --rm xau-dqn-scalper:latest
```