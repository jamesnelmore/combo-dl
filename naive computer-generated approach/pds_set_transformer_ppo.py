import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
import math
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import os
from functools import partial

# Enable MPS fallback for operations not supported on MPS
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Device detection
device = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
print(f"Using device: {device}")
print("Note: Some operations will fall back to CPU when needed")


class CyclicGroup:
    """Cyclic group Z_n with addition modulo n"""
    def __init__(self, n: int):
        self.order = n
        self.elements = list(range(n))
        self.name = f"Z_{n}"
    
    def operation(self, a: int, b: int, op: str = 'add') -> int:
        """Group operation: addition or subtraction mod n"""
        if op == 'add':
            return (a + b) % self.order
        elif op == 'subtract':
            return (a - b) % self.order
        else:
            raise ValueError("Operation must be 'add' or 'subtract'")
    
    def difference(self, a: int, b: int) -> int:
        """Compute a - b in the group"""
        return self.operation(a, b, 'subtract')


class PDSEnvironment(gym.Env):
    """Gymnasium environment for PDS discovery"""
    
    def __init__(self, group: CyclicGroup, target_params: Tuple[int, int, int, int]):
        super().__init__()
        
        self.group = group
        self.v, self.k, self.lambda_param, self.mu = target_params
        
        # State: binary vector indicating which elements are in current PDS
        self.observation_space = spaces.MultiBinary(self.group.order)
        
        # Action: which element to flip (add/remove from PDS)
        self.action_space = spaces.Discrete(self.group.order)
        
        # Current state
        self.current_pds = np.zeros(self.group.order, dtype=np.int32)
        self.step_count = 0
        self.max_steps = 200
        
        # Precompute difference matrix for efficiency
        self.difference_matrix = self._build_difference_matrix()
        
        # Precompute target difference counts for faster comparison
        self.target_diff_counts = np.zeros(self.group.order, dtype=np.int32)
        self.target_diff_counts[1:] = self.lambda_param
    
    def _build_difference_matrix(self) -> np.ndarray:
        """Precompute (i-j) mod n for all pairs"""
        diff_matrix = np.zeros((self.group.order, self.group.order), dtype=np.int32)
        for i in range(self.group.order):
            for j in range(self.group.order):
                diff_matrix[i, j] = self.group.difference(i, j)
        return diff_matrix
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_pds = np.zeros(self.group.order, dtype=np.int32)
        self.step_count = 0
        return self.current_pds.copy(), {}
    
    def step(self, action: int):
        # Flip the bit at position 'action'
        self.current_pds[action] = 1 - self.current_pds[action]
        self.step_count += 1
        
        # Compute reward
        reward = self._compute_reward()
        
        # Check termination conditions
        terminated = self._is_valid_pds() or (self.step_count >= self.max_steps)
        truncated = False
        
        return self.current_pds.copy(), reward, terminated, truncated, {}
    
    def _compute_reward(self) -> float:
        """Compute reward for current PDS candidate"""
        current_size = np.sum(self.current_pds)
        
        # If size is very wrong, return early
        if current_size == 0:
            return -1000
        
        # Base penalty for wrong size
        size_penalty = abs(current_size - self.k) * 10
        
        # Get PDS elements and compute differences
        pds_elements = np.where(self.current_pds)[0]
        diff_counts = self._compute_difference_counts(pds_elements)
        
        # Vectorized difference penalty computation
        diff_penalty = np.sum(np.abs(diff_counts[1:] - self.target_diff_counts[1:]))
        
        # Bonus for valid PDS
        if self._is_valid_pds():
            return 10000  # Large bonus for finding valid PDS
        
        # Progressive rewards for getting closer
        base_reward = -(size_penalty + diff_penalty * 0.1)
        
        # Bonus for correct size
        if current_size == self.k:
            base_reward += 100
        
        return base_reward
    
    def _compute_difference_counts(self, pds_elements: np.ndarray) -> np.ndarray:
        """Count how many times each difference appears"""
        diff_counts = np.zeros(self.group.order, dtype=np.int32)
        
        # Vectorized difference computation
        if len(pds_elements) > 0:
            # Create all pairs of elements
            pairs = np.array(np.meshgrid(pds_elements, pds_elements)).T.reshape(-1, 2)
            # Remove self-pairs
            pairs = pairs[pairs[:, 0] != pairs[:, 1]]
            
            if len(pairs) > 0:
                # Compute all differences at once
                diffs = self.difference_matrix[pairs[:, 0], pairs[:, 1]]
                # Count occurrences
                unique_diffs, counts = np.unique(diffs, return_counts=True)
                diff_counts[unique_diffs] = counts
        
        return diff_counts
    
    def _is_valid_pds(self) -> bool:
        """Check if current state is a valid PDS"""
        current_size = np.sum(self.current_pds)
        if current_size != self.k:
            return False
        
        pds_elements = np.where(self.current_pds)[0]
        diff_counts = self._compute_difference_counts(pds_elements)
        
        # Vectorized check of difference counts
        return np.all(diff_counts[1:] == self.target_diff_counts[1:])
    
    def get_pds_elements(self) -> List[int]:
        """Get current PDS as list of group elements"""
        return np.where(self.current_pds)[0].tolist()


class SetTransformerLayer(nn.Module):
    """Single layer of Set Transformer"""
    
    def __init__(self, dim: int, num_heads: int = 4, ff_dim: int = None):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim or 4 * dim
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(dim, self.ff_dim),
            nn.ReLU(),
            nn.Linear(self.ff_dim, dim)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual connection
        attn_out, _ = self.attention(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + attn_out)
        
        # Feed-forward with residual connection
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        
        return x


class PDSSetTransformer(nn.Module):
    """Set Transformer for PDS prediction"""
    
    def __init__(self, group_size: int = 99, embed_dim: int = 128, num_layers: int = 3):
        super().__init__()
        self.group_size = group_size
        self.embed_dim = embed_dim
        
        # Element embedding: map element index to embedding
        self.element_embedding = nn.Embedding(group_size, embed_dim)
        
        # Position encoding (optional, helps with structure)
        self.pos_encoding = nn.Parameter(torch.randn(group_size, embed_dim) * 0.1)
        
        # Set transformer layers
        self.transformer_layers = nn.ModuleList([
            SetTransformerLayer(embed_dim, num_heads=8)
            for _ in range(num_layers)
        ])
        
        # Output head: predict probability for each element
        self.output_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Global context for current PDS state
        self.pds_context = nn.Sequential(
            nn.Linear(group_size, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Move model to device
        self.to(device)
    
    def forward(self, pds_state: torch.Tensor) -> torch.Tensor:
        batch_size = pds_state.shape[0]
        
        # Create element embeddings
        element_indices = torch.arange(self.group_size, device=pds_state.device)
        element_embeds = self.element_embedding(element_indices)  # [group_size, embed_dim]
        
        # Add positional encoding
        element_embeds = element_embeds + self.pos_encoding
        
        # Expand for batch
        element_embeds = element_embeds.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Add current PDS context
        pds_context = self.pds_context(pds_state.float())  # [batch_size, embed_dim]
        pds_context = pds_context.unsqueeze(1).expand(-1, self.group_size, -1)
        
        # Combine element embeddings with PDS context
        x = element_embeds + pds_context
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x)
        
        # Output probabilities for each element
        output = self.output_head(x).squeeze(-1)  # [batch_size, group_size]
        
        return output


class PDSFeaturesExtractor(BaseFeaturesExtractor):
    """Custom features extractor for PPO using Set Transformer"""
    
    def __init__(self, observation_space: gym.Space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        group_size = observation_space.shape[0]
        self.set_transformer = PDSSetTransformer(group_size, embed_dim=128)
        
        # Additional processing layers
        self.features_net = nn.Sequential(
            nn.Linear(group_size, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Get Set Transformer output
        set_output = self.set_transformer(observations)
        
        # Process through additional layers
        features = self.features_net(set_output)
        
        return features


class PDSCallback(BaseCallback):
    """Callback to monitor PDS discovery progress"""
    
    def __init__(self, eval_env, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.best_reward = -float('inf')
        self.valid_pds_found = False
        
    def _on_step(self) -> bool:
        # Every 1000 steps, evaluate current policy
        if self.n_calls % 1000 == 0:
            obs, _ = self.eval_env.reset()
            total_reward = 0
            done = False
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = self.eval_env.step(action)
                total_reward += reward
                done = terminated or truncated
            
            # Check if valid PDS was found
            if self.eval_env._is_valid_pds():
                print(f"\n🎉 VALID PDS FOUND! Elements: {self.eval_env.get_pds_elements()}")
                self.valid_pds_found = True
                return False  # Stop training
            
            if total_reward > self.best_reward:
                self.best_reward = total_reward
                print(f"Step {self.n_calls}: New best reward: {total_reward:.2f}")
                print(f"Current PDS size: {np.sum(self.eval_env.current_pds)}")
        
        return True


class MPSCompatiblePolicy(ActorCriticPolicy):
    """Custom policy with MPS-compatible initialization"""
    
    def init_weights(self, module: nn.Module, gain: float = 1.0) -> None:
        """
        Initialize weights using a method compatible with MPS
        """
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Use Kaiming initialization instead of orthogonal
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)


def make_env(group_order: int, target_params: Tuple[int, int, int, int], rank: int = 0):
    """Create a new environment instance"""
    def _init():
        group = CyclicGroup(group_order)
        env = PDSEnvironment(group, target_params)
        return env
    return _init


def train_pds_model(group_order: int = 99, target_params: Tuple[int, int, int, int] = (99, 14, 1, 2)):
    """Train PPO model to find PDS"""
    
    print(f"Training PDS model for Conway's problem: {target_params}")
    print(f"Group: Z_{group_order}")
    print(f"Using {device} device")
    
    # Create vectorized environment with multiple processes
    n_envs = 8  # Number of parallel environments
    vec_env = SubprocVecEnv([make_env(group_order, target_params, i) for i in range(n_envs)])
    
    # Create evaluation environment
    eval_env = PDSEnvironment(CyclicGroup(group_order), target_params)
    
    # Create PPO model with custom features extractor and MPS-compatible policy
    model = PPO(
        MPSCompatiblePolicy,  # Use our custom policy
        vec_env,
        verbose=1,
        learning_rate=1e-4,  # Reduced learning rate for more stable updates
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        policy_kwargs=dict(
            features_extractor_class=PDSFeaturesExtractor,
            features_extractor_kwargs=dict(features_dim=256),
            net_arch=[dict(pi=[256, 256], vf=[256, 256])]
        ),
        device=device,
        ent_coef=0.005,  # Reduced entropy coefficient
        clip_range=0.1,  # Reduced clip range for more conservative updates
        max_grad_norm=0.3,  # Reduced gradient norm clipping
        gae_lambda=0.95,
        vf_coef=0.5,
        target_kl=0.03  # Increased target KL to allow for more policy change
    )
    
    # Create callback
    callback = PDSCallback(eval_env)
    
    # Train the model
    try:
        model.learn(
            total_timesteps=1000000,  # 1M steps
            callback=callback,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("Training interrupted by user")
    
    return model, callback.valid_pds_found


def evaluate_model(model, group_order: int = 99, target_params: Tuple[int, int, int, int] = (99, 14, 1, 2), num_trials: int = 10):
    """Evaluate trained model"""
    
    group = CyclicGroup(group_order)
    env = PDSEnvironment(group, target_params)
    
    valid_pds_found = []
    best_rewards = []
    
    for trial in range(num_trials):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        
        is_valid = env._is_valid_pds()
        valid_pds_found.append(is_valid)
        best_rewards.append(total_reward)
        
        print(f"Trial {trial + 1}: Valid PDS: {is_valid}, Reward: {total_reward:.2f}")
        if is_valid:
            print(f"  PDS elements: {env.get_pds_elements()}")
    
    success_rate = sum(valid_pds_found) / num_trials
    avg_reward = np.mean(best_rewards)
    
    print(f"\nEvaluation Results:")
    print(f"Success rate: {success_rate:.2%}")
    print(f"Average reward: {avg_reward:.2f}")
    
    return success_rate, avg_reward


if __name__ == "__main__":
    # Conway's 99-graph problem parameters
    CONWAY_PARAMS = (99, 14, 1, 2)  # (v, k, lambda, mu)
    
    print("🚀 Starting PDS Set Transformer Training for Conway's Problem")
    print("=" * 60)
    
    # Train the model
    model, pds_found = train_pds_model(
        group_order=99,
        target_params=CONWAY_PARAMS
    )
    
    if pds_found:
        print("\n🎉 SUCCESS! Valid PDS found during training!")
    else:
        print("\n📊 Training completed. Evaluating model...")
        
        # Evaluate the trained model
        success_rate, avg_reward = evaluate_model(
            model,
            group_order=99,
            target_params=CONWAY_PARAMS,
            num_trials=20
        )
        
        if success_rate > 0:
            print(f"\n🎉 SUCCESS! Model found valid PDS in {success_rate:.1%} of trials!")
        else:
            print(f"\n🔄 No valid PDS found. Best average reward: {avg_reward:.2f}")
            print("Consider:")
            print("- Training longer (increase total_timesteps)")
            print("- Adjusting hyperparameters")
            print("- Trying different group (Z_9 × Z_11)")
    
    # Save the model
    model.save("pds_set_transformer_model")
    print("\n💾 Model saved as 'pds_set_transformer_model'")