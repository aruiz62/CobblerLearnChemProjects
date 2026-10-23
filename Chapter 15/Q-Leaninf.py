
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Step 1 & 2: Create the FrozenLake environment
env = gym.make("FrozenLake-v1", is_slippery=True)

states = env.observation_space.n
actions = env.action_space.n

print("Number of states:", states)
print("Number of actions:", actions)

# Step 3: Create Q-table
q_table = np.zeros((states, actions))

print("\nStarting Q-table:")
print(q_table)

# Step 4: Train the AI
episodes = 20000
learning_rate = 0.8
discount_factor = 0.95
epsilon = 1.0
epsilon_decay = 0.9995
epsilon_min = 0.01

rewards = []

for episode in range(episodes):
    state, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        new_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        future_reward = 0 if done else np.max(q_table[new_state])

        q_table[state, action] += learning_rate * (
            reward + discount_factor * future_reward
            - q_table[state, action]
        )

        state = new_state
        total_reward += reward

    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    rewards.append(total_reward)

print("\nTraining completed!")

# Step 5: Print final Q-table
print("\nFinal Q-table:")
print(np.round(q_table, 3))

# Show best directions
arrows = ["←", "↓", "→", "↑"]
lake = env.unwrapped.desc

print("\nLearned policy:")

for row in range(4):
    line = []
    for col in range(4):
        state = row * 4 + col
        tile = lake[row, col].decode("utf-8")

        if tile == "H":
            line.append("H")
        elif tile == "G":
            line.append("G")
        else:
            line.append(arrows[np.argmax(q_table[state])])

    print(" ".join(line))

# Test AI 100 times
successes = 0

for test in range(100):
    state, info = env.reset()
    done = False

    while not done:
        action = np.argmax(q_table[state])
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if done and reward == 1:
            successes += 1

print("\nTEST RESULTS")
print("Successful attempts:", successes)
print("Total attempts: 100")
print("Success rate:", successes, "%")

# Step 6: Graph learning progress
output_folder = Path(__file__).resolve().parent

window = 500
moving_average = np.convolve(
    rewards,
    np.ones(window) / window,
    mode="valid"
)

plt.figure(figsize=(9, 5))
plt.plot(
    np.arange(window, episodes + 1),
    moving_average,
    color="blue"
)
plt.title("Q-Learning Training Progress")
plt.xlabel("Training Episodes")
plt.ylabel("Average Success Rate")
plt.grid(True)

plt.savefig(
    output_folder / "q_learning_progress.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

env.close()

print("\nGraph saved successfully!")
