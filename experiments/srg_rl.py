# TODO set up a basic PPO policy that can use the edge swap environment. No custom model for now
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import torch

from combo_dl.algorithms.edge_swap_env import RegularEdgeSwapEnv
from combo_dl.experiment_logger import ExperimentLogger
from combo_dl.problems import StronglyRegularGraphs


class LoggingCallback(BaseCallback):
    """Minimal callback to integrate SB3 with ExperimentLogger and early stopping."""

    def __init__(self, experiment_logger: ExperimentLogger, env, verbose: int = 0):
        super().__init__(verbose)
        self.experiment_logger = experiment_logger
        self.env = env
        self.best_reward = float("-inf")
        self.episodes_without_improvement = 0
        self.patience = 100  # Stop if no improvement for 100 episodes
        self.early_stop_threshold = -1.0  # Stop if reward > -1.0 (very close to perfect)

    def _on_rollout_end(self):
        """Called at the end of each rollout - capture rollout metrics."""
        if hasattr(self.model, "logger") and self.model.logger is not None:
            # Extract all available metrics
            metrics = {}
            for key, value in self.model.logger.name_to_value.items():
                if key not in ["time", "fps"]:  # Skip time metrics to avoid clutter
                    metrics[key] = value

            # Log to your experiment logger
            if metrics:
                self.experiment_logger.log_metrics(metrics, self.num_timesteps)

            # Check for early stopping based on reward
            if "rollout/ep_rew_mean" in metrics:
                current_reward = metrics["rollout/ep_rew_mean"]

                if current_reward > self.best_reward:
                    self.best_reward = current_reward
                    self.episodes_without_improvement = 0

                    # Check if we've achieved near-perfect construction
                    if current_reward > self.early_stop_threshold:
                        print(f"\n🎉 PERFECT CONSTRUCTION ACHIEVED! Reward: {current_reward:.6f}")
                        print("Early stopping due to perfect SRG construction!")
                        return False
                else:
                    self.episodes_without_improvement += 1

                # Early stopping if no improvement for too long
                if self.episodes_without_improvement >= self.patience:
                    print(f"\n⏹️  Early stopping: No improvement for {self.patience} episodes")
                    print(f"Best reward achieved: {self.best_reward:.6f}")
                    return False
        return None

    def _on_step(self) -> bool:
        """Called at each step - just return True to continue."""
        return True


def main():
    # Initialize logger
    logger = ExperimentLogger(
        experiment_name="srg_ppo_training",
        project="undergraduate-thesis",
        wandb_mode="online",
        use_progress_bar=True,
    )

    problem = StronglyRegularGraphs.peterson_graph_problem()
    env = RegularEdgeSwapEnv.from_srg_problem(problem)

    print(f"Environment: n={env.n}, k={env.k}, max_steps={env.max_steps}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Log experiment start
    config = {
        "algorithm": "PPO",
        "environment": "RegularEdgeSwapEnv",
        "problem": "Peterson Graph (10,3,0,1)",
        "n": env.n,
        "k": env.k,
        "max_steps": env.max_steps,
        "total_timesteps": 30_000_000,
    }
    logger.log_experiment_start(config)

    # Configure PPO with parameters for long training run
    # Determine best device
    if torch.cuda.is_available():
        device = "cuda"
        print("Using CUDA GPU")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("Using MPS (Apple GPU)")
    else:
        device = "cpu"
        print("Using CPU")

    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,  # Slightly higher for value function learning
        n_steps=4096,  # More steps per update for better sample efficiency
        batch_size=128,  # Larger batch size
        n_epochs=20,  # More epochs per update
        gamma=0.995,  # Higher discount factor for longer episodes
        gae_lambda=0.98,  # Higher GAE lambda
        clip_range=0.15,  # Tighter clipping for stability
        ent_coef=0.005,  # Lower entropy for more exploitation
        vf_coef=0.5,  # Higher value function coefficient for better learning
        max_grad_norm=0.5,  # Slightly looser gradient clipping
        device="cpu",  # Use the determined device
        tensorboard_log="./tensorboard_logs/",  # Enable tensorboard logging
        policy_kwargs={
            "net_arch": [{"pi": [256, 256], "vf": [256, 256]}],  # Larger networks
            "activation_fn": torch.nn.ReLU,
        },
    )

    # Create callback for logging and early stopping
    callback = LoggingCallback(logger, env)

    print("Starting long training run...")
    print("Will run for up to 30M timesteps or until perfect construction is found!")
    model.learn(total_timesteps=30_000_000, progress_bar=True, callback=callback)
    print("Training completed!")

    # Evaluate the trained model
    print("\nEvaluating trained model...")
    obs, _ = env.reset()
    total_reward = 0
    episode_count = 0

    for episode in range(5):  # Run 5 evaluation episodes
        obs, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        total_reward += episode_reward
        episode_count += 1
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")

    print(f"Average evaluation reward: {total_reward / episode_count:.2f}")

    # Compare with random policy
    print("\nTesting random policy for comparison...")
    random_total_reward = 0
    for episode in range(5):
        obs, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            # Random action
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        random_total_reward += episode_reward
        print(f"Random Episode {episode + 1}: Reward = {episode_reward:.2f}")

    print(f"Random policy average reward: {random_total_reward / 5:.2f}")
    print(
        f"Trained model improvement: {total_reward / episode_count - random_total_reward / 5:.2f}"
    )

    # Save the model
    model.save("srg_ppo_model")
    print("Model saved as 'srg_ppo_model'")

    # Log experiment end
    results = {
        "final_average_reward": total_reward / episode_count,
        "trained_model_improvement": total_reward / episode_count - random_total_reward / 5,
        "random_policy_average": random_total_reward / 5,
        "best_reward_during_training": callback.best_reward,
        "episodes_without_improvement": callback.episodes_without_improvement,
        "early_stopped": callback.best_reward > callback.early_stop_threshold,
    }
    logger.log_experiment_end(results, success=True)


if __name__ == "__main__":
    main()
