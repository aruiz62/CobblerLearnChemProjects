import matplotlib.pyplot as plt
import gymnasium as gym

# Create the CartPole environment
env = gym.make("CartPole-v1")

# Run 10 episodes
for episode in range(10):
    state, info = env.reset()
    total_reward = 0
    steps = 0
    done = False

    while not done:
        # Randomly move left or right
        action = env.action_space.sample()

        # Apply action and get results
        state, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        steps += 1

        done = terminated or truncated

    print(f"Episode {episode + 1}: Steps = {steps}, Total Reward = {total_reward}")

env.close()

# Rule-based CartPole approach
env = gym.make("CartPole-v1")

for episode in range(10):
    state, info = env.reset()
    total_reward = 0
    steps = 0
    done = False

    while not done:
        # Move in the direction the pole is leaning
        pole_angle = state[2]

        if pole_angle < 0:
            action = 0
        else:
            action = 1

        state, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        steps += 1
        done = terminated or truncated

    print(f"Rule-Based Episode {episode + 1}: Steps = {steps}, Total Reward = {total_reward}")

env.close()

# Results from our first two experiments
random_rewards = [36, 43, 14, 25, 31, 20, 18, 12, 11, 42]

rule_rewards = [31, 44, 35, 36, 41, 55, 45, 55, 41, 34]

episodes = range(1, 11)

# Create comparison graph
plt.figure(figsize=(10, 6))

plt.plot(episodes, random_rewards, marker="o", label="Random")
plt.plot(episodes, rule_rewards, marker="s", label="Rule-Based")

plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("CartPole: Random vs Rule-Based")
plt.legend()
plt.grid(True)

plt.savefig("cartpole_comparison.png", dpi=300, bbox_inches="tight")
plt.show()