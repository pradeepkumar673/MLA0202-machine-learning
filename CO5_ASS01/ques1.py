import numpy as np
import random

grid_size = 5
start_state = (0, 0)
goal_state = (4, 4)
obstacles = [(1, 1), (2, 2), (3, 1), (1, 3)]

actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
action_effects = {
    'UP': (-1, 0),
    'DOWN': (1, 0),
    'LEFT': (0, -1),
    'RIGHT': (0, 1)
}

perpendicular_actions = {
    'UP': ['LEFT', 'RIGHT'],
    'DOWN': ['LEFT', 'RIGHT'],
    'LEFT': ['UP', 'DOWN'],
    'RIGHT': ['UP', 'DOWN']
}

prob_intended = 0.8
prob_perp = 0.1
gamma = 0.95
threshold = 1e-4

battery_limit = 30
max_moves = 25

def get_next_state(state, action):
    if state == goal_state:
        return state
    dr, dc = action_effects[action]
    nr, nc = state[0] + dr, state[1] + dc
    if 0 <= nr < grid_size and 0 <= nc < grid_size and (nr, nc) not in obstacles:
        return (nr, nc)
    return state

def get_reward(state, next_state):
    if next_state == goal_state:
        return 100
    if next_state == state and state != goal_state:
        return -10
    return -1

def value_iteration():
    V = np.zeros((grid_size, grid_size))
    iteration = 0
    while True:
        delta = 0
        new_V = np.copy(V)
        for r in range(grid_size):
            for c in range(grid_size):
                state = (r, c)
                if state == goal_state or state in obstacles:
                    continue
                action_values = []
                for a in actions:
                    val = 0
                    actual_actions = [(a, prob_intended)]
                    for pa in perpendicular_actions[a]:
                        actual_actions.append((pa, prob_perp))
                    for act, prob in actual_actions:
                        ns = get_next_state(state, act)
                        rew = get_reward(state, ns)
                        val += prob * (rew + gamma * V[ns[0], ns[1]])
                    action_values.append(val)
                new_V[r, c] = max(action_values)
                delta = max(delta, abs(new_V[r, c] - V[r, c]))
        V = new_V
        iteration += 1
        if delta < threshold:
            break
    
    policy = {}
    for r in range(grid_size):
        for c in range(grid_size):
            state = (r, c)
            if state == goal_state or state in obstacles:
                policy[state] = 'GOAL' if state == goal_state else 'WALL'
                continue
            best_val = -float('inf')
            best_action = None
            for a in actions:
                val = 0
                actual_actions = [(a, prob_intended)]
                for pa in perpendicular_actions[a]:
                    actual_actions.append((pa, prob_perp))
                for act, prob in actual_actions:
                    ns = get_next_state(state, act)
                    rew = get_reward(state, ns)
                    val += prob * (rew + gamma * V[ns[0], ns[1]])
                if val > best_val:
                    best_val = val
                    best_action = a
            policy[state] = best_action
            
    return V, policy, iteration

def simulate_policy(policy, epsilon=0.1):
    current_state = start_state
    battery = battery_limit
    moves = 0
    total_reward = 0
    path = [current_state]
    
    print("Simulating Warehouse Robot Navigation:")
    print("Initial Battery:", battery, "| Max Allowed Moves:", max_moves)
    
    while current_state != goal_state and moves < max_moves and battery > 0:
        if random.random() < epsilon:
            chosen_action = random.choice(actions)
        else:
            chosen_action = policy[current_state]
            
        r = random.random()
        if r < prob_intended:
            actual_action = chosen_action
        elif r < prob_intended + prob_perp:
            actual_action = perpendicular_actions[chosen_action][0]
        else:
            actual_action = perpendicular_actions[chosen_action][1]
            
        next_s = get_next_state(current_state, actual_action)
        rew = get_reward(current_state, next_s)
        
        total_reward += rew
        battery -= 1
        moves += 1
        current_state = next_s
        path.append(current_state)
        
        print("Move", moves, "| Intended:", chosen_action, "| Executed:", actual_action, "| State:", current_state, "| Battery Left:", battery)
        
    reached = (current_state == goal_state)
    return reached, path, moves, battery, total_reward

print("==================================================")
print("Question 1: Value Iteration for Robot Navigation")
print("==================================================")

V, policy, iterations = value_iteration()

print("Value Iteration converged in", iterations, "iterations.")
print("\nOptimal Value Function:")
print(np.round(V, 2))

print("\nOptimal Policy Grid:")
policy_grid = np.empty((grid_size, grid_size), dtype=object)
for r in range(grid_size):
    for c in range(grid_size):
        policy_grid[r, c] = policy[(r, c)]
print(policy_grid)

print("\n--------------------------------------------------")
success, path, moves, remaining_battery, score = simulate_policy(policy)
print("--------------------------------------------------")
print("Path Taken:", path)
print("Destination Reached:", success)
print("Total Moves:", moves, "(Limit: 25)")
print("Remaining Battery:", remaining_battery, "/ 30")
print("Total Reward:", score)

print("\n==================================================")
print("JUSTIFICATION & CONSTRAINT ANALYSIS:")
print("1. Battery & Move Constraints: Step cost of -1 incentivizes shorter paths.")
print("   Battery limit of 30 and max move limit of 25 are respected as optimal")
print("   path takes minimal steps under stochastic transitions.")
print("2. Value Iteration Choice: Preferred over policy iteration here as the grid")
print("   state space is small and state values converge rapidly in", iterations, "steps.")
print("3. Stochastic Handling: Expected utility accounts for 80% success and 20% drift,")
print("   preventing the robot from taking unsafe paths near obstacles.")
print("4. Exploration Strategy: Epsilon-greedy (eps=0.1) allows testing policy robustness")
print("   against unpredictable slip events.")
print("==================================================")
