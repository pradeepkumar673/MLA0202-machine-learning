import random
import numpy as np

def simulate_single_delivery():
    base_time = 45.0
    distance = random.uniform(15.0, 30.0)
    speed = random.uniform(25.0, 40.0)
    travel_time = (distance / speed) * 60.0
    
    traffic_condition = random.choice(['light', 'moderate', 'heavy'])
    if traffic_condition == 'light':
        traffic_delay = random.uniform(0.0, 5.0)
    elif traffic_condition == 'moderate':
        traffic_delay = random.uniform(5.0, 15.0)
    else:
        traffic_delay = random.uniform(15.0, 35.0)
        
    weather_delay = random.choice([0.0, 5.0, 12.0])
    
    total_time = base_time + travel_time + traffic_delay + weather_delay
    return total_time

def monte_carlo_on_time_estimation(num_simulations, max_allowed_time):
    on_time_count = 0
    delivery_times = []
    
    for i in range(num_simulations):
        time_taken = simulate_single_delivery()
        delivery_times.append(time_taken)
        if time_taken <= max_allowed_time:
            on_time_count += 1
            
    estimated_probability = on_time_count / num_simulations
    avg_delivery_time = float(np.mean(delivery_times))
    
    return estimated_probability, on_time_count, avg_delivery_time

def main():
    random.seed(42)
    np.random.seed(42)
    
    num_simulations = 10000
    max_allowed_time = 90.0
    
    prob, count, avg_time = monte_carlo_on_time_estimation(num_simulations, max_allowed_time)
    
    print("--- Monte Carlo Delivery Simulation Results ---")
    print(f"Total Simulations: {num_simulations}")
    print(f"Max Allowed Delivery Time: {max_allowed_time} mins")
    print(f"On-Time Deliveries: {count}")
    print(f"Average Delivery Time: {avg_time:.2f} mins")
    print(f"Estimated Probability of On-Time Delivery: {prob * 100:.2f}%")

if __name__ == "__main__":
    main()
