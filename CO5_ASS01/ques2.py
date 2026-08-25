import numpy as np
import random
import math

true_ctr = [0.05, 0.12, 0.08, 0.25, 0.18]
k_arms = len(true_ctr)
total_interactions = 10000

best_possible_ctr = max(true_ctr)
optimal_arm = np.argmax(true_ctr)

def run_epsilon_greedy(epsilon=0.1):
    counts = np.zeros(k_arms)
    rewards = np.zeros(k_arms)
    total_reward = 0
    regret_history = []
    cum_regret = 0
    
    for t in range(1, total_interactions + 1):
        if random.random() < epsilon:
            arm = random.randint(0, k_arms - 1)
        else:
            if np.sum(counts) == 0:
                arm = random.randint(0, k_arms - 1)
            else:
                arm = np.argmax(rewards / np.maximum(counts, 1))
                
        reward = 1 if random.random() < true_ctr[arm] else 0
        counts[arm] += 1
        rewards[arm] += reward
        total_reward += reward
        
        regret = best_possible_ctr - true_ctr[arm]
        cum_regret += regret
        regret_history.append(cum_regret)
        
    return counts, rewards, total_reward, cum_regret, regret_history

def run_ucb1(c_param=1.5):
    counts = np.zeros(k_arms)
    rewards = np.zeros(k_arms)
    total_reward = 0
    regret_history = []
    cum_regret = 0
    
    for t in range(1, total_interactions + 1):
        if t <= k_arms:
            arm = t - 1
        else:
            ucb_values = (rewards / counts) + c_param * np.sqrt(np.log(t) / counts)
            arm = np.argmax(ucb_values)
            
        reward = 1 if random.random() < true_ctr[arm] else 0
        counts[arm] += 1
        rewards[arm] += reward
        total_reward += reward
        
        regret = best_possible_ctr - true_ctr[arm]
        cum_regret += regret
        regret_history.append(cum_regret)
        
    return counts, rewards, total_reward, cum_regret, regret_history

def run_thompson_sampling():
    successes = np.ones(k_arms)
    failures = np.ones(k_arms)
    counts = np.zeros(k_arms)
    rewards = np.zeros(k_arms)
    total_reward = 0
    regret_history = []
    cum_regret = 0
    
    for t in range(1, total_interactions + 1):
        sampled_theta = [np.random.beta(successes[i], failures[i]) for i in range(k_arms)]
        arm = np.argmax(sampled_theta)
        
        reward = 1 if random.random() < true_ctr[arm] else 0
        counts[arm] += 1
        rewards[arm] += reward
        total_reward += reward
        
        if reward == 1:
            successes[arm] += 1
        else:
            failures[arm] += 1
            
        regret = best_possible_ctr - true_ctr[arm]
        cum_regret += regret
        regret_history.append(cum_regret)
        
    return counts, rewards, total_reward, cum_regret, regret_history

print("==================================================")
print("Question 2: Multi-Armed Bandit Advertisement Selection")
print("==================================================")
print("True CTRs of Ads (0 to 4):", true_ctr)
print("Optimal Ad:", optimal_arm, "with CTR:", best_possible_ctr)
print("Budget / Interactions:", total_interactions)

eps_counts, eps_rewards, eps_tot_rew, eps_regret, _ = run_epsilon_greedy(0.1)
ucb_counts, ucb_rewards, ucb_tot_rew, ucb_regret, _ = run_ucb1(1.5)
ts_counts, ts_rewards, ts_tot_rew, ts_regret, _ = run_thompson_sampling()

print("\n--------------------------------------------------")
print("1. Epsilon-Greedy Strategy (Epsilon = 0.1):")
print("Ad Pull Counts:", eps_counts.astype(int))
print("Estimated CTRs:", np.round(eps_rewards / np.maximum(eps_counts, 1), 4))
print("Total Clicks (Rewards):", int(eps_tot_rew))
print("Cumulative Regret:", np.round(eps_regret, 2))

print("\n--------------------------------------------------")
print("2. Upper Confidence Bound (UCB1) Strategy:")
print("Ad Pull Counts:", ucb_counts.astype(int))
print("Estimated CTRs:", np.round(ucb_rewards / np.maximum(ucb_counts, 1), 4))
print("Total Clicks (Rewards):", int(ucb_tot_rew))
print("Cumulative Regret:", np.round(ucb_regret, 2))

print("\n--------------------------------------------------")
print("3. Thompson Sampling Strategy (Bayesian Beta-Binomial):")
print("Ad Pull Counts:", ts_counts.astype(int))
print("Estimated CTRs:", np.round(ts_rewards / np.maximum(ts_counts, 1), 4))
print("Total Clicks (Rewards):", int(ts_tot_rew))
print("Cumulative Regret:", np.round(ts_regret, 2))

print("\n==================================================")
print("JUSTIFICATION & CONSTRAINT ANALYSIS:")
print("1. K-Armed Bandit Formulation: K=5 ads correspond to 5 arms.")
print("   Showing 1 ad per visitor simulates a Bernoulli bandit environment")
print("   where reward = 1 (click) or 0 (no click).")
print("2. Constraint Satisfaction (10,000 interactions): Thompson Sampling")
print("   and UCB achieve logarithmic cumulative regret O(log N), minimizing")
print("   wasteful exploration within the fixed 10,000 budget limit.")
print("3. Comparison of Exploration Strategies:")
print("   - Epsilon-Greedy continues non-zero random sampling even after finding best ad.")
print("   - UCB balances mean reward with uncertainty bounds.")
print("   - Thompson Sampling (Bayesian) dynamically adapts posterior distribution,")
print("     rapidly focusing selections on Ad", optimal_arm, "with minimal regret.")
print("4. Reward Mechanism Influence: Positive feedback (clicks) updates Beta posterior")
print("   or UCB mean, driving the policy toward optimal ad selection.")
print("==================================================")
