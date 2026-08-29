import random
import numpy as np

class EpsilonGreedyBandit:
    def __init__(self, movies, true_rewards, epsilon=0.15):
        self.movies = movies
        self.num_movies = len(movies)
        self.true_rewards = true_rewards
        self.epsilon = epsilon
        self.q_estimates = np.zeros(self.num_movies)
        self.action_counts = np.zeros(self.num_movies)

    def select_movie(self):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_movies - 1)
        else:
            max_val = np.max(self.q_estimates)
            best_indices = np.where(self.q_estimates == max_val)[0]
            return random.choice(best_indices)

    def simulate_user_feedback(self, movie_idx):
        reward = random.gauss(self.true_rewards[movie_idx], 1.0)
        return reward

    def update(self, movie_idx, reward):
        self.action_counts[movie_idx] += 1
        n = self.action_counts[movie_idx]
        self.q_estimates[movie_idx] += (reward - self.q_estimates[movie_idx]) / n

    def get_best_movie(self):
        best_idx = int(np.argmax(self.q_estimates))
        return self.movies[best_idx], self.q_estimates[best_idx]

def main():
    random.seed(42)
    np.random.seed(42)

    movies = ["Movie A (Sci-Fi)", "Movie B (Comedy)", "Movie C (Action)", "Movie D (Drama)", "Movie E (Documentary)"]
    true_rewards = [3.2, 4.5, 2.1, 4.0, 1.8]
    
    bandit = EpsilonGreedyBandit(movies, true_rewards, epsilon=0.15)
    num_interactions = 2000

    for step in range(num_interactions):
        chosen_movie_idx = bandit.select_movie()
        reward = bandit.simulate_user_feedback(chosen_movie_idx)
        bandit.update(chosen_movie_idx, reward)

    print("--- Epsilon-Greedy K-Armed Bandit Simulation ---")
    print(f"Total Interactions: {num_interactions}")
    print(f"Epsilon value: {bandit.epsilon}\n")

    for i in range(len(movies)):
        print(f"{movies[i]}: True Reward = {true_rewards[i]}, Estimated Reward = {bandit.q_estimates[i]:.3f}, Selected = {int(bandit.action_counts[i])} times")

    best_movie, best_reward = bandit.get_best_movie()
    print(f"\nMovie with Highest Expected Reward: {best_movie} (Estimated Reward = {best_reward:.3f})")

if __name__ == "__main__":
    main()
