import numpy as np

class Task:
    def __init__(self, name, success_prob, cost, duration):
        self.name = name
        self.success_prob = success_prob
        self.cost = cost
        self.duration = duration
        self.outcome_probabilities = [success_prob, 1 - success_prob]  # Success or failure

tasks = [
    Task("task1", success_prob=0.8, cost=100, duration=5),
    Task("task2", success_prob=0.6, cost=150, duration=10),
    Task("task3", success_prob=0.4, cost=200, duration=8)
]

class DPScheduler:
    def __init__(self, tasks, max_time):
        self.tasks = tasks
        self.max_time = max_time
        self.dp = np.full((len(tasks) + 1, max_time + 1), -float('inf'))  # Use -inf for uninitialized states
        self.decision_path = np.zeros_like(self.dp)
        self.dp[-1, :] = 0  # Final state reward is zero

    def simulate_heuristic_policy(self):
        state_space = []
        for t in range(1, self.max_time + 1):
            state = []
            for task in self.tasks:
                prob_outcome = np.random.choice([1, 0], p=task.outcome_probabilities)
                cost = task.cost if prob_outcome == 1 else 0
                duration = task.duration if prob_outcome == 1 else 0
                state.append((prob_outcome, cost, duration))
            state_space.append(state)
        return state_space

    def calculate_cost_to_go(self, state_space):
        # Compute the Bellman equation in confined state-space
        for t in reversed(range(self.max_time)):
            for i, task in enumerate(self.tasks):
                if t + task.duration <= self.max_time:
                    # Calculate reward assuming task succeeds
                    reward_if_success = task.success_prob * (self.dp[i + 1, t + task.duration] - task.cost)
                    # Update DP table only if reward improves
                    if reward_if_success > self.dp[i, t]:
                        self.dp[i, t] = reward_if_success
                        self.decision_path[i, t] = 1  # Mark decision path for this task

    def get_optimal_policy(self):
        policy = []
        current_time = 0
        for i, task in enumerate(self.tasks):
            for t in range(current_time, self.max_time):
                if self.decision_path[i, t] == 1:
                    policy.append(f"{task.name} at time {t}")
                    current_time = t + task.duration  # Move forward in time by task duration
                    break
        return policy

    def run(self):
        state_space = self.simulate_heuristic_policy()  # Step 1: Heuristic State Space Confinement
        self.calculate_cost_to_go(state_space)  # Step 2: Bellman Equation over Heuristic State Space
        return self.get_optimal_policy()  # Step 3: Optimal Policy Extraction

max_time = 20
scheduler = DPScheduler(tasks, max_time)
optimal_policy = scheduler.run()

print("Optimal Task Execution Policy:", optimal_policy)

import matplotlib.pyplot as plt

def visualize_execution_schedule(optimal_policy, tasks):
    # Prepare data for the bar chart
    task_starts = []
    task_durations = []
    task_names = []
    
    task_dict = {task.name: task for task in tasks}  # Quick lookup for task details
    
    for entry in optimal_policy:
        task_name, start_time_str = entry.split(" at time ")
        start_time = int(start_time_str)
        
        task = task_dict[task_name]
        
        task_names.append(task_name)
        task_starts.append(start_time)
        task_durations.append(task.duration)

    # Create a bar chart
    plt.figure(figsize=(12, 6))
    plt.barh(task_names, task_durations, left=task_starts, color='skyblue', edgecolor='black')
    
    # Adding labels
    for i, (start, duration) in enumerate(zip(task_starts, task_durations)):
        plt.text(start + duration / 2, i, f"{start}-{start + duration}", 
                 ha='center', va='center', color="black")
    
    plt.xlabel("Time")
    plt.ylabel("Tasks")
    plt.title("Task Execution Schedule")
    plt.grid(axis='x')
    plt.xlim(0, max(task_starts) + max(task_durations) + 1)  # Extend x-limits for better visibility
    plt.show()

# Assuming 'optimal_policy' is the output of scheduler.run() and 'tasks' is the task list
visualize_execution_schedule(optimal_policy, tasks)
