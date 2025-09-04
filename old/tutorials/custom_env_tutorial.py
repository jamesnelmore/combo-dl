import marimo

__generated_with = "0.11.23"
app = marimo.App(width="full")


@app.cell
def _():
    from typing import Optional

    import gymnasium as gym
    from gymnasium.utils.env_checker import check_env
    import marimo as mo
    import numpy as np
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env

    return Optional, PPO, check_env, gym, make_vec_env, mo, np


@app.cell
def _(Optional, check_env, gym, np):
    class GridWorldEnv(gym.Env):
        def __init__(self, size: int = 5):
            super().__init__()
            self.size = size
            self._agent_location = np.array([-1, -1], dtype=np.int32)
            self.target_location = np.array([-1, -1], dtype=np.int32)

            self.observation_space = gym.spaces.Dict(
                {
                    "agent": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
                    "target": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
                }
            )

            self.action_space = gym.spaces.Discrete(4)
            self._action_to_direction = {
                0: np.array([1, 0]),  # Move right (positive x)
                1: np.array([0, 1]),  # Move up (positive y)
                2: np.array([-1, 0]),  # Move left (negative x)
                3: np.array([0, -1]),  # Move down (negative y)
            }

        def _get_obs(self) -> dict:
            return {"agent": self._agent_location, "target": self._target_location}

        def _get_info(self) -> dict:
            return {
                "distance": np.linalg.norm(self._agent_location - self._target_location, ord=1)
            }

        def reset(
            self, seed: Optional[int] = None, options: Optional[dict] = None
        ) -> (dict, dict):
            super().reset(seed=seed)

            self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)
            self._target_location = self._agent_location

            while np.array_equal(self._target_location, self._agent_location):
                self._target_location = self.np_random.integers(0, self.size, size=2, dtype=int)

            obs = self._get_obs()
            info = self._get_info()

            return obs, info

        def step(self, action: int):
            direction = self._action_to_direction[action]
            self._agent_location = np.clip(
                a=self._agent_location + direction, a_min=0, a_max=self.size - 1
            )

            terminated = np.array_equal(self._agent_location, self._target_location)
            truncated = False  # TODO add limit
            reward = 1 if terminated else 0  # TODO add negative rewards to encourage efficiency

            observation = self._get_obs()
            info = self._get_info()

            return observation, reward, terminated, truncated, info

    gym.register(
        id="gymnasium_env/GridWorld-v0",
        entry_point=GridWorldEnv,
        max_episode_steps=300,
    )
    test_env = gym.make("gymnasium_env/GridWorld-v0")
    check_env(test_env.unwrapped)
    return GridWorldEnv, test_env


@app.cell
def _(PPO, make_vec_env):
    vec_env = make_vec_env(
        "gymnasium_env/GridWorld-v0",
        n_envs=20,
        render_mode=None,
        env_kwargs={"size": 32},
    )
    model = PPO("MultiInputPolicy", vec_env, device="mps", verbose=1)
    model.learn(total_timesteps=1_000_000)

    obs = vec_env.reset()
    while True:
        action, _states = model.predict(obs)
        (
            obs,
            rewards,
            dones,
            info,
        ) = vec_env.step(action)
    return action, dones, info, model, obs, rewards, vec_env


if __name__ == "__main__":
    app.run()
