import numpy as np

n_states = 25
n_actions = 4
goal_state = 24
n_episodes = 12

Q_Table = np.zeros((n_states, n_actions))
first_flag = False
total_reward = 0

f = open("Q_Table.txt", "w")

for episode in range(n_episodes):
    state = 0
    steps = 0
    total_reward = 0
    while(state != goal_state):
        f.write("Cell: " + str(state) + " ")
        steps += 1
        if episode == 0 and first_flag == False:
            action = np.random.randint(0, n_actions)
            first_flag = True
        else:
            max_Q = np.max(Q_Table[state])
            max_indices = np.where(Q_Table[state] == max_Q)[0]
            f.write("Choices:" + str(max_indices) + " ")
            if len(max_indices) > 1:
                action = np.random.choice(max_indices)
            else:
                action = np.argmax(Q_Table[state])

        if action == 0:
            next_state = state - 5
        elif action == 1:
            next_state = state + 5 
        elif action == 2:
            next_state = state + 1
        else:
            next_state = state - 1
        
        if next_state == goal_state:
            reward = 100
            total_reward += reward
        else:
            reward = 0

        if next_state < 0 or next_state > 24 or (state % 5 == 4 and action == 2) or (state % 5 == 0 and action == 3):
            next_state = state
            reward = -1
            total_reward += reward
        f.write("Choice: " + str(action) + " ")
        f.write("Reward: " + str(reward) + "\n")
        
        Q_Table[state, action] += .5 * (reward + .85 * np.max(Q_Table[next_state]) - Q_Table[state, action])
        
        state = next_state
    f.write("Total Reward: " + str(total_reward) + "\n")
    f.write("Steps: " + str(steps) + "\n")
print(Q_Table)
