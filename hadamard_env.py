import gymnasium as gym
import numpy as np
import logging
import os
import torch

class HadamardEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 1}
    
    def __init__(self, n: int, render_mode: str | None = None, patience: int = 10000):
        super().__init__()
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        if n < 2:
            logging.error("n must be at least 2")
            raise ValueError("n must be at least 2")
        if not (n == 2 or n % 4 == 0):
            logging.error("n must be 2 or a multiple of 4")
            raise ValueError("n must be 2 or a multiple of 4")
        self.n: int = n
        self._max_det: float = n ** (n / 2)

        # Action space: Coordinates of the entry to flip
        self.action_space = gym.spaces.MultiDiscrete([n, n])
        
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(n, n), dtype=np.int8)

        self.patience = patience

        self.reset()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.state = np.ones((self.n, self.n), dtype=np.int8)
        # Reset tracking variables
        self.best_reward = 0
        self.steps_without_improvement = 0
        self.episode_steps = 0

         # For reward normalization
        self.reward_history = []
        self.reward_mean = 0.0
        self.reward_std = 1.0
        return self.get_observation(), {}

    def get_observation(self) -> np.ndarray:
        return self.state.copy()

    def is_hadamard(self) -> bool:
        if not np.all(np.abs(self.state) == 1):
            return False
        
        # Check if rows are orthogonal
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if np.dot(self.state[i], self.state[j]) != 0:
                    return False
        
        print("Hadamard matrix found!")
        return True
    
    def calculate_reward(self) -> tuple[float, bool]:
        """
        Returns:
            reward: Reward for current state
            is_hadamard: True if the matrix is Hadamard, False otherwise
        """

        # Check if we found a Hadamard matrix
        if self.is_hadamard():
            return 100.0, True

        gram_matrix = self.state @ self.state.T
        
        target_matrix = self.n * np.eye(self.n)
        diff_matrix = gram_matrix - target_matrix
        
        # Use negative squared Frobenius norm as the main reward component
        # This gives a clear gradient toward the target
        frobenius_error = np.linalg.norm(diff_matrix, 'fro') ** 2
        max_possible_error = 2 * self.n * self.n * (self.n - 1)  # Rough upper bound
        
        # Convert to a reward (higher is better)
        orthogonality_reward = max(0.0, 1.0 - frobenius_error / max_possible_error)
        
        total_reward = orthogonality_reward
        
        return total_reward, False

    def step(self, action: tuple[int, int]) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Take a step in the environment.
        
        Args:
            action: Tuple (row, col) representing position to flip
            
        Returns:
            observation: The new state
            reward: Reward for the action
            terminated: True if the matrix is Hadamard
            truncated: True if the episode is truncated
            info: Empty for API compliance
        """
        if not self.action_space.contains(action):
            raise ValueError(f"Action {action} is not in action space {self.action_space}")

        row, col = action
        
        self.state[row, col] *= -1
        
        reward, is_hadamard = self.calculate_reward()
        
        # Reward shaping: give additional reward for improvement
        reward_improvement = 0.0
        if reward > self.best_reward:
            reward_improvement = (reward - self.best_reward) * 2.0  
        
        shaped_reward = reward + reward_improvement
        
        if reward > self.best_reward:
            self.best_reward = reward
            self.steps_without_improvement = 0
        else:
            self.steps_without_improvement += 1
        
        truncated = self.steps_without_improvement >= self.patience or self.episode_steps >= 50000
        self.episode_steps += 1
        
        # Update reward history for normalization
        self.reward_history.append(shaped_reward)
        self.reward_mean = np.mean(self.reward_history)
        self.reward_std = np.std(self.reward_history)
        
        return self.get_observation(), shaped_reward, is_hadamard, truncated, {}
    
if __name__ == "__main__":
    import stable_baselines3 as sb3
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import CheckpointCallback

    # Create logs directory
    log_dir = "logs/"
    os.makedirs(log_dir, exist_ok=True)

    n = 4*2
    
    # Create training environment
    def make_env():
        env = HadamardEnv(n=n)
        env = Monitor(env, log_dir)
        return env
    
    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)

    # Create eval environment
    def make_eval_env():
        eval_env = HadamardEnv(n=n, patience=1000)  # Much smaller patience for evaluation
        eval_env = Monitor(eval_env)
        return eval_env
    
    eval_env = DummyVecEnv([make_eval_env])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)  # Don't normalize rewards for eval

    # Create evaluation callback with less frequent evaluation
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./best_model",
        log_path=log_dir,
        eval_freq=10000,  # More frequent evaluation to track progress
        n_eval_episodes=5,  # Fewer eval episodes
        deterministic=True,
        render=False
    )

    # Create checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=log_dir,
        name_prefix="hadamard_model"
    )

    model = sb3.PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=3e-4,     # More standard learning rate
        n_steps=2048,           # Increased for better value estimation
        batch_size=64,          # Smaller batch size for more frequent updates
        n_epochs=10,            
        gamma=0.99,             
        gae_lambda=0.95,        
        clip_range=0.2,         
        ent_coef=0.01,          # Reduced entropy coefficient
        vf_coef=1.0,            # Increased value function coefficient
        max_grad_norm=0.5,      
        use_sde=False,          
        sde_sample_freq=-1,     
        target_kl=None,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),  # Larger networks
            activation_fn=torch.nn.ReLU
        ),
        tensorboard_log=log_dir
    )
    
    # Train with callbacks
    model.learn(
        total_timesteps=1000000,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )