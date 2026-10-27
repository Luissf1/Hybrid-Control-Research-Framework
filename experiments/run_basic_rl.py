"""
Basic RL Sanity Check
======================

Trains tabular Q-Learning on FrozenLake-v1 to verify that the environment
and the agent are correctly installed and configured. Useful as a fast
smoke test before running the full hybrid pipeline.
"""

import yaml
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

from src.rl.q_learning import QLearningAgent


def main() -> None:
    # Load the RL-only configuration
    with open("configs/rl_config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    rl_cfg = cfg["rl"]
    env_cfg = cfg["environment"]

    # Instantiate the environment
    env = gym.make(env_cfg["name"], is_slippery=env_cfg["is_slippery"])

    # Instantiate the agent
    agent = QLearningAgent(
        alpha=rl_cfg["learning_rate"],
        gamma=rl_cfg["gamma"],
        epsilon=rl_cfg["epsilon"],
        epsilon_decay=rl_cfg["epsilon_decay"],
        epsilon_min=rl_cfg["epsilon_min"],
        seed=42,
    )

    action_space_size = env.action_space.n
    episode_rewards = []

    # Training loop
    for ep in range(rl_cfg["episodes"]):
        obs, _ = env.reset()
        state = (int(obs),)  # FrozenLake returns a scalar; wrap as tuple
        total_reward = 0.0
        done = False
        truncated = False
        steps = 0

        while not (done or truncated) and steps < rl_cfg["max_steps"]:
            action = agent.choose_action(state, action_space_size)
            next_obs, reward, done, truncated, _ = env.step(action)
            next_state = (int(next_obs),)

            agent.update_q_table(
                state, action, reward, next_state, done, action_space_size
            )
            state = next_state
            total_reward += reward
            steps += 1

        agent.decay_epsilon()
        episode_rewards.append(total_reward)

        if (ep + 1) % 1000 == 0:
            avg = np.mean(episode_rewards[-1000:])
            print(f"Episode {ep + 1}/{rl_cfg['episodes']} | last-1000 avg: {avg:.3f}")

    # Plot the learning curve
    window = 100
    moving_avg = np.convolve(
        episode_rewards, np.ones(window) / window, mode="valid"
    )
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards, alpha=0.3)
    plt.plot(range(window - 1, len(episode_rewards)), moving_avg, color="red")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("FrozenLake — Tabular Q-Learning")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/basic_rl_curve.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()