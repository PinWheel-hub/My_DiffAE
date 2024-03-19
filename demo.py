import numpy as np

def simulate_task_completion(T, a, num_simulations=10000):
    total_time = 0

    for _ in range(num_simulations):
        time = 0
        while True:
            time_to_failure = np.random.exponential(1 / a)
            if time_to_failure < T:
                time += time_to_failure
            else:
                time += T
                break
        total_time += time

    return total_time / num_simulations

T = 26 
a = 0.1 

expected_completion_time = simulate_task_completion(T, a)

print(f'T: {T}, a: {a}')
print('simulation:', expected_completion_time)
print('我的:', (-1 + np.exp(a * T)) / a)
print('gpt4:', T * np.exp(a * T))
print('文心:', np.exp(a * T) / a + T)
