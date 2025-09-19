"""Algorithm to train a model in a gym env."""

from datetime import datetime
from typing import Literal, override

import gymnasium as gym
import stable_baselines3 as sb3
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from combo_dl.algorithms import BaseAlgorithm
from combo_dl.experiment_logger import ExperimentLogger
from combo_dl.problems import BaseProblem


class GymEnvTrainer(BaseAlgorithm):
    model: sb3.PPO

    def __init__(
        self,
        features_extractor: BaseFeaturesExtractor,
        problem: BaseProblem,
        env: gym.Env,
        logger: ExperimentLogger | None = None,  # pyright: ignore[reportRedeclaration]
        total_timesteps: int = 10_000,
        device: Literal["cpu", "cuda", "mps"] = "cpu",
    ):
        if logger is None:
            date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            experiment_name = f"Deep Cross Entropy {date} {type(problem).__name__}"
            logger: ExperimentLogger = ExperimentLogger(experiment_name, use_wandb=False)
        self.device = device
        self.features_extractor = features_extractor
        self.total_timesteps = total_timesteps
        policy_kwargs = {"features_extractor": self.features_extractor}
        model = sb3.PPO(
            "MultiInputPolicy", env, policy_kwargs=policy_kwargs, device=self.device, verbose=1
        )
        super().__init__(model, problem, logger)

    @override
    def optimize(self, **kwargs) -> dict[str, Any]:
        """Train on given environment. Will not vectorize an environment automatically.

        Args:
            **kwargs: Not used, only for API compatibility

        Returns:
            Dictionary of optimization results
        """
        self.model.learn(total_timesteps=self.total_timesteps)
        # TODO figure out how to pass back metrics
