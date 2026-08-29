import numpy as np
import random

GRID_SIZE = 4
DELIVERY_STATE = (3, 3)
HAZARD_STATE = (1, 2)
ACTIONS = ['NORTH', 'SOUTH', 'EAST', 'WEST']
ACTION_MAP = {
    'NORTH': (-1, 0),
    'SOUTH': (1, 0),
    'EAST': (0, 1),
    'WEST': (0, -1)
}
GAMMA = 0.9
THRESHOLD = 1e-4

def get_next_state(r, c, action):
    dr, dc = ACTION_MAP[action]
    nr, nc = r + dr, c + dc
    if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
        return nr, nc
    return r, c

def get_reward(r, c, next_r, next_c):
    if (next_r, next_c) == DELIVERY_STATE:
        return 50.0
    if (next_r, next_c) == HAZARD_STATE:
        return -10.0
    return -1.0

def policy_evaluation(policy, V):
    while True:
        delta = 0.0
        new_V = np.copy(V)
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if (r, c) == DELIVERY_STATE:
                    continue
                action = policy[r, c]
                nr, nc = get_next_state(r, c, action)
                reward = get_reward(r, c, nr, nc)
                new_V[r, c] = reward + GAMMA * V[nr, nc]
                delta = max(delta, abs(new_V[r, c] - V[r, c]))
        V = new_V
        if delta < THRESHOLD:
            break
    return V

def policy_improvement(policy, V):
    policy_stable = True
    new_policy = np.copy(policy)
    
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if (r, c) == DELIVERY_STATE:
                continue
            old_action = policy[r, c]
            best_action = None
            best_val = -float('inf')
            
            for a in ACTIONS:
                nr, nc = get_next_state(r, c, a)
                reward = get_reward(r, c, nr, nc)
                val = reward + GAMMA * V[nr, nc]
                if val > best_val:
                    best_val = val
                    best_action = a
                    
            new_policy[r, c] = best_action
            if best_action != old_action:
                policy_stable = False
                
    return new_policy, policy_stable

def policy_iteration():
    policy = np.random.choice(ACTIONS, size=(GRID_SIZE, GRID_SIZE))
    policy[DELIVERY_STATE] = 'TARGET'
    V = np.zeros((GRID_SIZE, GRID_SIZE))
    
    iteration = 0
    while True:
        iteration += 1
        V = policy_evaluation(policy, V)
        policy, policy_stable = policy_improvement(policy, V)
        if policy_stable:
            break
            
    return V, policy, iteration

def main():
    random.seed(42)
    np.random.seed(42)
    
    V, optimal_policy, iterations = policy_iteration()
    
    print("--- Policy Iteration for Autonomous Drone Navigation ---")
    print(f"Policy converged in {iterations} iterations\n")
    
    print("Optimal State-Value Function V(s):")
    for r in range(GRID_SIZE):
        row_str = ""
        for c in range(GRID_SIZE):
            row_str += f"{V[r, c]:8.2f}"
        print(row_str)
        
    print("\nOptimal Drone Navigation Policy:")
    for r in range(GRID_SIZE):
        row_str = ""
        for c in range(GRID_SIZE):
            row_str += f"{optimal_policy[r, c]:>10}"
        print(row_str)

if __name__ == "__main__":
    main()
