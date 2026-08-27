import random
import numpy as np

NUM_STATES = 7
START_STATE = 0
TERMINAL_STATE = 6
ACTIONS = [-1, 1]

def step(state, action):
    next_state = state + action
    if next_state < 0:
        next_state = 0
    elif next_state >= NUM_STATES:
        next_state = NUM_STATES - 1
        
    if next_state == TERMINAL_STATE:
        reward = 10.0
    else:
        reward = -0.1
        
    return next_state, reward

def td_learning(num_episodes, alpha=0.1, gamma=0.9, epsilon=0.2):
    V = np.zeros(NUM_STATES)
    
    for episode in range(num_episodes):
        state = START_STATE
        
        while state != TERMINAL_STATE:
            if random.random() < epsilon:
                action = random.choice(ACTIONS)
            else:
                best_action = ACTIONS[0]
                best_val = -float('inf')
                for a in ACTIONS:
                    ns, _ = step(state, a)
                    if V[ns] > best_val:
                        best_val = V[ns]
                        best_action = a
                action = best_action
                
            next_state, reward = step(state, action)
            
            if next_state == TERMINAL_STATE:
                td_target = reward
            else:
                td_target = reward + gamma * V[next_state]
                
            td_error = td_target - V[state]
            V[state] += alpha * td_error
            
            state = next_state
            
    return V

def main():
    random.seed(42)
    np.random.seed(42)
    
    num_episodes = 500
    alpha = 0.1
    gamma = 0.9
    
    V = td_learning(num_episodes, alpha=alpha, gamma=gamma)
    
    print("--- Temporal Difference (TD) Learning for Game Score Prediction ---")
    print(f"Total Episodes: {num_episodes}")
    print(f"Learning Rate (alpha): {alpha}")
    print(f"Discount Factor (gamma): {gamma}\n")
    
    print("Learned State-Value Estimates V(s):")
    for s in range(NUM_STATES):
        if s == TERMINAL_STATE:
            print(f"State {s} (Terminal Goal): {V[s]:.3f}")
        else:
            print(f"State {s}: {V[s]:.3f}")

if __name__ == "__main__":
    main()
