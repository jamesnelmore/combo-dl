import marimo

__generated_with = "0.11.23"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import gymnasium as gym
    import numpy as np

    import matplotlib.pyplot as plt
    return gym, mo, np, plt


@app.cell
def _(mo):
    mo.md(r"""## BlackJack Q-Learning""")
    return


@app.cell
def _(gym, np):
    from collections import defaultdict
    from typing import TypeVar, Generic

    ObsType = TypeVar("ObsType")


    class BlackjackAgent(Generic[ObsType]):
        def __init__(
            self,
            env: gym.Env,
            learning_rate: float,
            initial_epsilon: float,
            epsilon_decay: float,
            final_epsilon: float,
            discount_factor: float = 0.95,
        ):
            self.env = env
            self.lr = learning_rate
            self.epsilon = initial_epsilon
            self.epsilon_decay = epsilon_decay
            self.final_epsilon = final_epsilon
            self.discount_factor = discount_factor

            # set of (state, action pairs)
            self.q_table = defaultdict(lambda: np.zeros(env.action_space.n))
            self.training_error = []

        def get_action(self, obs: tuple[int, int, bool]) -> int:
            """
            obs: (player_sum, dealer_card, usable_ace)
            Returns:
                action: 0 = Stand, 1 = Hit
            """

            # Implement q learning with epsilon greedy
            if np.random.random() < self.epsilon:
                return int(self.env.action_space.sample())
                pass
            else:
                return int(
                    np.argmax(self.q_table[obs])
                )  # Take the action with the highest expected reward at state obs

        def update(
            self,
            obs: ObsType,
            action: int,
            reward: float,
            terminated: bool,
            next_obs: ObsType,
        ):
            expected_future_value = (not terminated) * np.max(
                self.q_table[next_obs]
            )
            # First term is what the q_value should be (Bellman equation)
            td_error = (
                reward + self.discount_factor * expected_future_value
            ) - self.q_table[obs][action]
            # assert type(action) == int
            self.q_table[obs][action] += self.lr * td_error
            self.training_error.append(td_error)

        def decay_epsilon(self):
            self.epsilon = max(
                self.epsilon - self.epsilon_decay, self.final_epsilon
            )
    return BlackjackAgent, Generic, ObsType, TypeVar, defaultdict


@app.cell
def _(BlackjackAgent, gym, np):
    from tqdm import tqdm
    from collections import deque

    n_episodes = 1_000_000
    start_epsilon = 1.0

    env = gym.make("Blackjack-v1", sab=False)
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)
    agent = BlackjackAgent(
        env=env,
        learning_rate=0.3,
        initial_epsilon=1.0,
        epsilon_decay=start_epsilon / (n_episodes / 2),
        final_epsilon=0.1,
    )

    recent_rewards = deque(maxlen=1000)
    pbar = tqdm(range(n_episodes))
    for episode in pbar:
        obs, info = env.reset()
        episode_over = False
        total_reward = 0
        while not episode_over:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            agent.update(obs, action, reward, terminated, next_obs)
            episode_over = terminated or truncated
            obs = next_obs

        agent.decay_epsilon()
        recent_rewards.append(total_reward)
        pbar.set_postfix_str(f"rolling reward: {np.mean(recent_rewards): .2f}")
    return (
        action,
        agent,
        deque,
        env,
        episode,
        episode_over,
        info,
        n_episodes,
        next_obs,
        obs,
        pbar,
        recent_rewards,
        reward,
        start_epsilon,
        terminated,
        total_reward,
        tqdm,
        truncated,
    )


@app.cell
def _(agent, env, np, plt):
    def get_moving_avgs(arr, window, convolution_mode):
        """Compute moving average to smooth noisy data."""
        return (
            np.convolve(
                np.array(arr).flatten(), np.ones(window), mode=convolution_mode
            )
            / window
        )


    # Smooth over a 500-episode window
    rolling_length = 50000
    fig, axs = plt.subplots(ncols=3, figsize=(12, 5))

    # Episode rewards (win/loss performance)
    axs[0].set_title("Episode rewards")
    reward_moving_average = get_moving_avgs(
        env.return_queue, rolling_length, "valid"
    )
    axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
    axs[0].set_ylabel("Average Reward")
    axs[0].set_xlabel("Episode")

    # Episode lengths (how many actions per hand)
    axs[1].set_title("Episode lengths")
    length_moving_average = get_moving_avgs(
        env.length_queue, rolling_length, "valid"
    )
    axs[1].plot(range(len(length_moving_average)), length_moving_average)
    axs[1].set_ylabel("Average Episode Length")
    axs[1].set_xlabel("Episode")

    # Training error (how much we're still learning)
    axs[2].set_title("Absolute Training Error")
    training_error_moving_average = get_moving_avgs(
        np.abs(agent.training_error), rolling_length, "same"
    )
    axs[2].plot(
        range(len(training_error_moving_average)), training_error_moving_average
    )
    axs[2].set_ylabel("Temporal Difference Error")
    axs[2].set_xlabel("Step")

    plt.tight_layout()
    plt.show()
    return (
        axs,
        fig,
        get_moving_avgs,
        length_moving_average,
        reward_moving_average,
        rolling_length,
        training_error_moving_average,
    )


@app.cell
def _(agent, env, np):
    # Test the trained agent
    def test_agent(agent, env, num_episodes=10000):
        """Test agent performance without learning or exploration."""
        total_rewards = []

        # Temporarily disable exploration for testing
        old_epsilon = agent.epsilon
        agent.epsilon = 0.0  # Pure exploitation

        for _ in range(num_episodes):
            obs, info = env.reset()
            episode_reward = 0
            done = False

            while not done:
                action = agent.get_action(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                done = terminated or truncated

            total_rewards.append(episode_reward)

        # Restore original epsilon
        agent.epsilon = old_epsilon

        win_rate = np.mean(np.array(total_rewards) > 0)
        average_reward = np.mean(total_rewards)

        print(f"Test Results over {num_episodes} episodes:")
        print(f"Win Rate: {win_rate:.1%}")
        print(f"Average Reward: {average_reward:.3f}")
        print(f"Standard Deviation: {np.std(total_rewards):.3f}")


    # Test your agent
    test_agent(agent, env)
    return (test_agent,)


if __name__ == "__main__":
    app.run()
