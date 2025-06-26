import torch
import torch.nn as nn
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from hadamard_env import HadamardEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
import gymnasium as gym
from typing import Dict, Any
import os
from datetime import datetime


class MatrixConvolutionalExtractor(BaseFeaturesExtractor):
    """
    Custom CNN-based feature extractor for matrix observations.
    Treats the matrix as a 2D image and applies convolutional layers.
    """
    
    def __init__(self, observation_space: gym.Space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        # Get matrix dimensions
        n_rows, n_cols = observation_space.shape
        
        # Convolutional layers to extract spatial patterns
        self.conv_net = nn.Sequential(
            # Add channel dimension for conv2d
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # Reduce to fixed size
            nn.Flatten(),
        )
        
        # Calculate flattened dimension
        with torch.no_grad():
            sample_input = torch.zeros(1, 1, n_rows, n_cols)
            conv_output = self.conv_net(sample_input)
            conv_output_dim = conv_output.shape[1]
        
        # Final projection to desired feature dimension
        self.projection = nn.Sequential(
            nn.Linear(conv_output_dim, features_dim),
            nn.ReLU(),
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Add channel dimension: (batch, height, width) -> (batch, 1, height, width)
        if observations.dim() == 3:
            observations = observations.unsqueeze(1)
        
        # Apply convolutional layers
        conv_features = self.conv_net(observations)
        
        # Project to final feature dimension
        features = self.projection(conv_features)
        
        return features


class OrthogonalityAwareExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor that explicitly computes orthogonality-related features
    for Hadamard matrix discovery.
    """
    
    def __init__(self, observation_space: gym.Space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        self.n = observation_space.shape[0]  # Matrix size
        
        # Neural network to process raw matrix
        self.matrix_encoder = nn.Sequential(
            nn.Linear(self.n * self.n, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        
        # Neural network to process orthogonality features
        gram_matrix_size = self.n * self.n
        self.ortho_encoder = nn.Sequential(
            nn.Linear(gram_matrix_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        
        # Combine features
        self.combiner = nn.Sequential(
            nn.Linear(64 + 32, features_dim),
            nn.ReLU(),
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        
        # Flatten matrix for basic processing
        flat_matrix = observations.view(batch_size, -1)
        matrix_features = self.matrix_encoder(flat_matrix)
        
        # Compute Gram matrix (orthogonality information)
        gram_matrices = torch.bmm(observations, observations.transpose(-2, -1))
        flat_gram = gram_matrices.view(batch_size, -1)
        ortho_features = self.ortho_encoder(flat_gram)
        
        # Combine all features
        combined_features = torch.cat([matrix_features, ortho_features], dim=1)
        final_features = self.combiner(combined_features)
        
        return final_features


class TransformerMatrixExtractor(BaseFeaturesExtractor):
    """
    Transformer-based feature extractor that treats matrix rows/columns as sequences.
    """
    
    def __init__(self, observation_space: gym.Space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        self.n = observation_space.shape[0]
        embed_dim = 64
        
        # Row and column embeddings
        self.row_embedding = nn.Linear(self.n, embed_dim)
        self.col_embedding = nn.Linear(self.n, embed_dim)
        
        # Transformer layers
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=8,
                dim_feedforward=256,
                batch_first=True
            ),
            num_layers=2
        )
        
        # Final projection
        self.projection = nn.Sequential(
            nn.Linear(embed_dim * 2, features_dim),  # *2 for row + col features
            nn.ReLU(),
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        
        # Process rows
        row_embeddings = self.row_embedding(observations.float())  # (batch, n, embed_dim)
        row_features = self.transformer(row_embeddings)
        row_pooled = torch.mean(row_features, dim=1)  # (batch, embed_dim)
        
        # Process columns (transpose the matrix)
        col_input = observations.transpose(-2, -1)
        col_embeddings = self.col_embedding(col_input.float())
        col_features = self.transformer(col_embeddings)
        col_pooled = torch.mean(col_features, dim=1)  # (batch, embed_dim)
        
        # Combine row and column features
        combined = torch.cat([row_pooled, col_pooled], dim=1)
        final_features = self.projection(combined)
        
        return final_features


def create_custom_model(env, architecture_type: str = "convolutional", tensorboard_log: str = None):
    """
    Create a PPO model with custom architecture and tensorboard logging.
    
    Args:
        env: The vectorized environment
        architecture_type: Type of custom architecture to use
        tensorboard_log: Directory for tensorboard logs
    """
    
    # Define feature extractor based on architecture type
    if architecture_type == "convolutional":
        feature_extractor_class = MatrixConvolutionalExtractor
    elif architecture_type == "orthogonality":
        feature_extractor_class = OrthogonalityAwareExtractor
    elif architecture_type == "transformer":
        feature_extractor_class = TransformerMatrixExtractor
    else:
        raise ValueError(f"Unknown architecture type: {architecture_type}")
    
    # Policy configuration
    policy_kwargs = dict(
        features_extractor_class=feature_extractor_class,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(
            pi=[256, 128],  # Actor network
            vf=[256, 128]   # Critic network
        ),
        activation_fn=torch.nn.ReLU,
    )
    
    # Create PPO model with custom policy and tensorboard logging
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
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        tensorboard_log=tensorboard_log,
    )
    
    return model


def train_custom_model(total_timesteps: int = 50000, matrix_size: int = 8):
    """Train and compare different custom model architectures with tensorboard logging."""
    
    # Create logs directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logs_dir = f"logs/hadamard_experiments_{timestamp}"
    os.makedirs(logs_dir, exist_ok=True)
    
    def make_env():
        env = HadamardEnv(n=matrix_size)
        env = Monitor(env)
        return env
    
    # Try different architectures
    architectures = ["convolutional", "orthogonality", "transformer"]
    
    for arch_type in architectures:
        print(f"\nTraining {arch_type} architecture...")
        
        # Create architecture-specific log directory
        arch_log_dir = os.path.join(logs_dir, arch_type)
        os.makedirs(arch_log_dir, exist_ok=True)
        
        # Create training environment
        train_env = DummyVecEnv([make_env])
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)
        
        # Create evaluation environment
        eval_env = DummyVecEnv([make_env])
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
        
        # Create model with tensorboard logging
        model = create_custom_model(
            train_env, 
            architecture_type=arch_type,
            tensorboard_log=arch_log_dir
        )
        
        # Setup evaluation callback
        eval_callback = EvalCallback(
            eval_env,
            eval_freq=2000,
            n_eval_episodes=5,
            deterministic=True,
            render=False,
            verbose=1
        )
        
        # Train the model
        model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
            progress_bar=True,
            tb_log_name=arch_type
        )
        
        # Save the model
        model.save(f"{logs_dir}/{arch_type}_model")
        train_env.save(f"{logs_dir}/{arch_type}_vecnormalize.pkl")
        
        print(f"Completed {arch_type} - model saved to {logs_dir}")
    
    print(f"\nAll experiments completed!")
    print(f"View results with: tensorboard --logdir {logs_dir}")
    print(f"Models saved in: {logs_dir}")


if __name__ == "__main__":
    # You can adjust these parameters
    TOTAL_TIMESTEPS = 1_000_000  # Increase for longer training
    MATRIX_SIZE = 8  # Hadamard matrix size
    
    train_custom_model(total_timesteps=TOTAL_TIMESTEPS, matrix_size=MATRIX_SIZE) 