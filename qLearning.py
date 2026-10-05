import random
import numpy as np

# 1. Setup the Environment Parameters
num_states = 3  # 5 positions: 0, 1, 2, 3, 4
num_actions = 3  # 0: Left, 1: Right
goal_state = 2

# 2. Initialize Q-Table with zeros
# Rows = States, Columns = Actions
q_table = np.zeros((num_states, num_actions))

# 3. Hyperparameters
alpha = 0.1  # Learning rate (how fast it accepts new information)
gamma = 0.9  # Discount factor (importance of future rewards)
epsilon = 0.9  # Exploration rate (probability of choosing random actions)
epsilon_decay = 0.995  # Decay rate for exploration
episodes = 500  # Number of games to play


# Helper function to step through the environment
def take_action(state, action):
    # Determine the next state based on action
    if action == 1:  # Move Right
        next_state = min(state + 1, num_states - 1)
    else:  # Move Left
        next_state = max(state - 1, 0)

    # Distribute rewards
    if next_state == goal_state:
        reward = 10
        done = True
    else:
        reward = -1  # Negative reward penalty to encourage speed
        done = False

    return next_state, reward, done


# 4. Main Training Loop
print("Training the agent...")
for episode in range(episodes):
    state = 0  # Always start at position 0
    done = False

    while not done:
        # Epsilon-Greedy Action Selection
        if random.uniform(0, 1) < epsilon:
            action = random.choice([0, 1])  # Explore: choose a random action
        else:
            action = np.argmax(q_table[state])  # Exploit: choose the best known action

        # Interact with the environment
        next_state, reward, done = take_action(state, action)

        # Bellman Equation update for Q-value
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])

        # Formula: Q(s,a) = (1-alpha)*old_Q + alpha*(reward + gamma*max_next_Q)
        q_table[state, action] = (1 - alpha) * old_value + alpha * (
            reward + gamma * next_max
        )

        # Move to the next state
        state = next_state

    # Decay exploration rate over time
    epsilon *= epsilon_decay

print("Training finished!\n")

# 5. Output Results
print("Final Trained Q-Table:")
print("State | Action 0 (Left) | Action 1 (Right)")
print("-" * 42)
for i in range(num_states):
    print(f"  {i}   |    {q_table[i, 0]:7.2f}     |    {q_table[i, 1]:7.2f}")

# 6. Test the Trained Agent
print("\nTesting the trained agent's policy:")
state = 0
steps = [state]
while state != goal_state:
    action = np.argmax(q_table[state])  # Follow the maximum Q-value
    state, _, _ = take_action(state, action)
    steps.append(state)

print(f"Path taken by agent: {steps}")
