"""
Example showing how to modify the hadamard_env.py training to use custom model architectures.
This demonstrates integrating custom architectures into your existing training setup.
"""

import torch
import torch.nn as nn
from hadamard_env import HadamardEnv
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import gymnasium as gym
import os


class GraphNeuralNetworkExtractor(BaseFeaturesExtractor):
    """
    Graph-based feature extractor that treats the matrix as a graph.
    Each matrix element is a node, and edges connect neighboring elements.
    """
    
    def __init__(self, observation_space: gym.Space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        self.n = observation_space.shape[0]
        self.num_nodes = self.n * self.n
        
        # Node feature embedding (matrix value + positional encoding)
        self.node_embedding = nn.Sequential(
            nn.Linear(3, 64),  # value + row + col position
            nn.ReLU(),
            nn.Linear(64, 64),
        )
        
        # Graph convolution layers
        self.gnn_layers = nn.ModuleList([
            nn.Linear(64, 64) for _ in range(3)
        ])
        
        # Global pooling and final projection
        self.global_pool = nn.Sequential(
            nn.Linear(64, features_dim),
            nn.ReLU(),
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        
        # Create node features (value + position)
        node_features = []
        for i in range(self.n):
            for j in range(self.n):
                # Node features: [value, row_pos, col_pos]
                value = observations[:, i, j].unsqueeze(1)  # (batch, 1)
                row_pos = torch.full((batch_size, 1), i / self.n, device=observations.device)
                col_pos = torch.full((batch_size, 1), j / self.n, device=observations.device)
                node_feat = torch.cat([value, row_pos, col_pos], dim=1)
                node_features.append(node_feat)
        
        # Stack node features: (batch, num_nodes, 3)
        node_features = torch.stack(node_features, dim=1)
        
        # Embed nodes
        embedded_nodes = self.node_embedding(node_features)  # (batch, num_nodes, 64)
        
        # Simple graph convolution (average neighboring nodes)
        for gnn_layer in self.gnn_layers:
            # For simplicity, we'll use global average pooling as "graph convolution"
            # In practice, you'd implement proper graph convolution with adjacency matrices
            pooled = torch.mean(embedded_nodes, dim=1, keepdim=True)  # (batch, 1, 64)
            pooled = pooled.expand(-1, self.num_nodes, -1)  # (batch, num_nodes, 64)
            embedded_nodes = gnn_layer(embedded_nodes + pooled)
            embedded_nodes = torch.relu(embedded_nodes)
        
        # Global pooling to get final features
        graph_features = torch.mean(embedded_nodes, dim=1)  # (batch, 64)
        final_features = self.global_pool(graph_features)
        
        return final_features


class ResidualMatrixExtractor(BaseFeaturesExtractor):
    """
    Residual network-based feature extractor with skip connections.
    """
    
    def __init__(self, observation_space: gym.Space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        self.n = observation_space.shape[0]
        input_dim = self.n * self.n
        
        # Initial projection
        self.input_proj = nn.Linear(input_dim, 128)
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            self._make_res_block(128, 128) for _ in range(4)
        ])
        
        # Final projection
        self.output_proj = nn.Sequential(
            nn.Linear(128, features_dim),
            nn.ReLU(),
        )
    
    def _make_res_block(self, in_dim: int, out_dim: int):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Flatten matrix
        x = observations.view(observations.shape[0], -1)
        
        # Initial projection
        x = torch.relu(self.input_proj(x))
        
        # Residual blocks with skip connections
        for res_block in self.res_blocks:
            residual = x
            x = res_block(x)
            x = torch.relu(x + residual)  # Skip connection
        
        # Final projection
        features = self.output_proj(x)
        
        return features


def train_with_custom_architecture():
    """Train the Hadamard environment with custom architectures"""
    
    # Create logs directory
    log_dir = "logs/"
    os.makedirs(log_dir, exist_ok=True)
    
    n = 8  # Matrix size
    
    # Create training environment
    def make_env():
        env = HadamardEnv(n=n)
        env = Monitor(env, log_dir)
        return env
    
    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)
    
    # Create eval environment
    def make_eval_env():
        eval_env = HadamardEnv(n=n, patience=1000)
        eval_env = Monitor(eval_env)
        return eval_env
    
    eval_env = DummyVecEnv([make_eval_env])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
    
    # Create callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./best_model",
        log_path=log_dir,
        eval_freq=5000,
        n_eval_episodes=3,
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=log_dir,
        name_prefix="hadamard_custom_model"
    )
    
    # Different custom architectures to try
    architectures = {
        "graph": GraphNeuralNetworkExtractor,
        "residual": ResidualMatrixExtractor,
    }
    
    for arch_name, arch_class in architectures.items():
        print(f"\n{'='*50}")
        print(f"Training with {arch_name} architecture")
        print(f"{'='*50}")
        
        # Policy configuration with custom feature extractor
        policy_kwargs = dict(
            features_extractor_class=arch_class,
            features_extractor_kwargs=dict(features_dim=256),
            net_arch=dict(
                pi=[256, 256, 128],  # Actor network (3 layers)
                vf=[256, 256, 128]   # Critic network (3 layers)
            ),
            activation_fn=torch.nn.ReLU,
        )
        
        # Create PPO model with custom architecture
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=1.0,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
            tensorboard_log=f"{log_dir}/{arch_name}",
        )
        
        # Train the model
        model.learn(
            total_timesteps=100000,  # Adjust as needed
            callback=[eval_callback, checkpoint_callback],
            progress_bar=True
        )
        
        # Save the final model
        model.save(f"hadamard_{arch_name}_final")
        print(f"Model saved as hadamard_{arch_name}_final")


def load_and_test_custom_model():
    """Example of loading and testing a custom model"""
    
    # Create test environment
    n = 8
    env = HadamardEnv(n=n)
    
    try:
        # Load a trained custom model
        model = PPO.load("hadamard_graph_final")
        
        # Test the model
        obs, _ = env.reset()
        for i in range(100):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            print(f"Step {i}: Action={action}, Reward={reward:.4f}")
            
            if terminated:
                print("Hadamard matrix found!")
                break
            if truncated:
                print("Episode truncated")
                break
                
    except FileNotFoundError:
        print("No trained model found. Run train_with_custom_architecture() first.")


if __name__ == "__main__":
    # Train models with custom architectures
    train_with_custom_architecture()
    
    # Test a trained model
    # load_and_test_custom_model() 