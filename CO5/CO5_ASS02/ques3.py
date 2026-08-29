import numpy as np

GRID_SIZE = 5
GOAL_STATE = (4, 4)
OBSTACLES = [(1, 1), (2, 2), (3, 1)]
ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']
ACTION_OFFSETS = {
    'UP': (-1, 0),
    'DOWN': (1, 0),
    'LEFT': (0, -1),
    'RIGHT': (0, 1)
}
PERPENDICULAR_ACTIONS = {
    'UP': ['LEFT', 'RIGHT'],
    'DOWN': ['LEFT', 'RIGHT'],
    'LEFT': ['UP', 'DOWN'],
    'RIGHT': ['UP', 'DOWN']
}
GAMMA = 0.95
THRESHOLD = 1e-4

def get_next_state(r, c, action):
    dr, dc = ACTION_OFFSETS[action]
    nr, nc = r + dr, c + dc
    if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and (nr, nc) not in OBSTACLES:
        return nr, nc
    return r, c

def get_reward(r, c, next_r, next_c):
    if (next_r, next_c) == GOAL_STATE:
        return 100.0
    if (r, c) == (next_r, next_c):
        return -5.0
    return -1.0

def value_iteration():
    V = np.zeros((GRID_SIZE, GRID_SIZE))
    iteration = 0
    
    while True:
        delta = 0.0
        new_V = np.copy(V)
        
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if (r, c) == GOAL_STATE or (r, c) in OBSTACLES:
                    continue
                
                best_val = -float('inf')
                for a in ACTIONS:
                    val = 0.0
                    transitions = [(a, 0.8)] + [(pa, 0.1) for pa in PERPENDICULAR_ACTIONS[a]]
                    
                    for act, prob in transitions:
                        nr, nc = get_next_state(r, c, act)
                        reward = get_reward(r, c, nr, nc)
                        val += prob * (reward + GAMMA * V[nr, nc])
                        
                    if val > best_val:
                        best_val = val
                        
                new_V[r, c] = best_val
                delta = max(delta, abs(new_V[r, c] - V[r, c]))
                
        V = new_V
        iteration += 1
        if delta < THRESHOLD:
            break
            
    policy = np.empty((GRID_SIZE, GRID_SIZE), dtype=object)
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if (r, c) == GOAL_STATE:
                policy[r, c] = 'GOAL'
                continue
            if (r, c) in OBSTACLES:
                policy[r, c] = 'OBS'
                continue
                
            best_val = -float('inf')
            best_action = None
            for a in ACTIONS:
                val = 0.0
                transitions = [(a, 0.8)] + [(pa, 0.1) for pa in PERPENDICULAR_ACTIONS[a]]
                for act, prob in transitions:
                    nr, nc = get_next_state(r, c, act)
                    reward = get_reward(r, c, nr, nc)
                    val += prob * (reward + GAMMA * V[nr, nc])
                if val > best_val:
                    best_val = val
                    best_action = a
            policy[r, c] = best_action
            
    return V, policy, iteration

def main():
    V, policy, iterations = value_iteration()
    
    print("--- Value Iteration for Robot Path Planning ---")
    print(f"Converged in {iterations} iterations\n")
    
    print("Optimal Value Function V(s):")
    for r in range(GRID_SIZE):
        row_str = ""
        for c in range(GRID_SIZE):
            if (r, c) in OBSTACLES:
                row_str += f"{'OBS':>8}"
            else:
                row_str += f"{V[r, c]:8.2f}"
        print(row_str)
        
    print("\nOptimal Policy:")
    for r in range(GRID_SIZE):
        row_str = ""
        for c in range(GRID_SIZE):
            row_str += f"{policy[r, c]:>8}"
        print(row_str)

if __name__ == "__main__":
    main()
