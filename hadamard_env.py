import gymnasium as gym
import numpy as np
import logging
import os

class HadamardEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 1}
    
    def __init__(self, n: int, render_mode: str | None = None, patience: int = 10000000):
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

        # Action space: choose a row and column to flip
        self.action_space = gym.spaces.Discrete(n * n)
        
        # Observation space: flattened matrix of size n×n
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(n, n))

        self.initialize_state()
        self.patience = patience
        self.best_reward = 0
        self.steps_without_improvement = 0
        self.last_action = None

    def initialize_state(self) -> None:
        self.state = np.ones((self.n, self.n), dtype=np.int8)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.initialize_state()
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
        Calculates reward for current state as the normalized determinant of the matrix.
        The reward is normalized to be between 0 and 1, with 1 being the maximum possible determinant.

        Returns:
            reward: Normalized reward for current state (between 0 and 1)
            is_hadamard: True if the matrix is Hadamard, False otherwise
        """
        # Cache determinant calculation
        abs_det = np.abs(np.linalg.det(self.state))
        normalized_det = abs_det / self._max_det
        
        
        dot_products = self.state @ self.state.T
        total_dot_products = np.sum(np.abs(dot_products))
        
        # Normalize orthogonality reward to be between 0 and 1
        max_possible_dot = np.float64(self.n * self.n)
        orthogonality_reward = 1.0 - (total_dot_products - self.n) / (max_possible_dot - self.n)
        
        # Add intermediate rewards for progress
        progress_reward = 0.0
        if normalized_det > 0.5:
            progress_reward += 0.5
        if orthogonality_reward > 0.5:
            progress_reward += 0.5
            
        if np.isclose(normalized_det, 1.0) and self.is_hadamard():
            return 10.0, True
        else:
            return 0.3 * normalized_det + 0.3 * orthogonality_reward + 0.4 * progress_reward, False

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Take a step in the environment.
        
        Args:
            action: Integer representing position to flip (0 to n*n-1)
            
        Returns:
            observation: The new state
            reward: Reward for the action
            terminated: True if the matrix is Hadamard
            truncated: False
            info: Additional information
        """
        if not self.action_space.contains(action):
            raise ValueError(f"Action {action} is not in action space {self.action_space}")

        # Convert action to row and column
        row = action // self.n
        col = action % self.n
        
        # Flip the value at the chosen position
        self.state[row, col] *= -1
        
        reward, is_hadamard = self.calculate_reward()
        
        # Add small penalty for repeated actions to encourage exploration
        if self.last_action == action:
            reward -= 0.1
        
        self.last_action = action
        
        if reward > self.best_reward:
            self.best_reward = reward
            self.steps_without_improvement = 0
        else:
            self.steps_without_improvement += 1
        
        truncated = self.steps_without_improvement >= self.patience
        return self.get_observation(), reward, is_hadamard, truncated, {}
    
if __name__ == "__main__":
    import stable_baselines3 as sb3
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import CheckpointCallback

    # Create logs directory
    log_dir = "logs/"
    os.makedirs(log_dir, exist_ok=True)

    n = 4*2
    env = HadamardEnv(n=n)
    env = Monitor(env, log_dir)
    env = DummyVecEnv([lambda: env])

    # Create eval environment
    eval_env = HadamardEnv(n=n)
    eval_env = Monitor(eval_env)
    eval_env = DummyVecEnv([lambda: eval_env])

    # Create evaluation callback with less frequent evaluation
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./best_model",
        log_path=log_dir,
        eval_freq=50000,  # Reduced evaluation frequency
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
        learning_rate=1e-3,
        n_steps=1024,        # Reduced steps per update
        batch_size=128,      # Reduced batch size
        n_epochs=10,         
        gamma=0.99,          
        gae_lambda=0.95,     
        clip_range=0.2,      
        ent_coef=0.1,        
        vf_coef=0.5,         
        max_grad_norm=0.5,   
        use_sde=False,       
        sde_sample_freq=-1,  
        target_kl=None,
        policy_kwargs=dict(
            net_arch=dict(pi=[128, 128], vf=[128, 128])
        ),
        tensorboard_log=log_dir
    )
    
    # Train with callbacks
    model.learn(
        total_timesteps=1000000,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )