import numpy as np
import pytest
import torch

from env_trading import XAUUSDEnvironment
from model_dqn import TradingBrain
from train_dqn import ReplayBuffer


@pytest.fixture
def truncated_env():
    """Generates an environment instance with miniature dimensions for rapid CI validation."""
    return XAUUSDEnvironment(data_length=40)


@pytest.fixture
def micro_batch():
    """Generates a micro-batch sample to verify tensor dimensions without CPU compute strain."""
    batch_size = 4
    state_dim = 2

    states = np.random.rand(batch_size, state_dim).astype(np.float32)
    actions = np.random.randint(0, 3, size=batch_size)
    rewards = np.random.rand(batch_size).astype(np.float32)
    next_states = np.random.rand(batch_size, state_dim).astype(np.float32)
    dones = np.array([0, 0, 1, 0], dtype=np.float32)

    return states, actions, rewards, next_states, dones


def test_environment_state_dimensions(truncated_env):
    """Verifies that the MDP environment outputs structurally correct state arrays."""
    state = truncated_env.reset()
    assert isinstance(state, np.ndarray)
    assert state.shape == (2,)
    assert state[1] == 0  # Position must start flat (0)

    next_state, reward, done = truncated_env.step(1)  # Execute Buy
    assert next_state.shape == (2,)
    assert isinstance(reward, float)
    assert isinstance(done, bool)


def test_model_forward_pass():
    """Verifies the network's forward execution paths using micro-dimension tensors."""
    model = TradingBrain()
    model.eval()

    # Batch size of 2, State dimension of 2
    micro_input = torch.FloatTensor([[1920.50, 0.0], [1915.25, 1.0]])
    with torch.no_grad():
        output = model(micro_input)

    assert output.shape == (2, 3)  # Batch size x Action space size


def test_replay_buffer_sampling(micro_batch):
    """Validates the data retrieval integrity of the historical replay buffer."""
    buffer = ReplayBuffer(capacity=10)
    states, actions, rewards, next_states, dones = micro_batch

    for i in range(len(states)):
        buffer.push(states[i], actions[i], rewards[i], next_states[i], dones[i])

    assert len(buffer) == 4

    b_states, b_actions, b_rewards, b_next_states, b_dones = buffer.sample(batch_size=2)
    assert b_states.shape == (2, 2)
    assert b_actions.shape == (2,)
    assert b_rewards.shape == (2,)
    assert b_next_states.shape == (2, 2)
    assert b_dones.shape == (2,)
