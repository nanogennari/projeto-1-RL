import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from classes import RecyclingRobotEnvironment, TemporalDifferenceAgent, State

# Set up plotting
plt.style.use('default')
sns.set_style("whitegrid")

print("Starting simple parameter exploration...")

# ====== PART 1: MDP Parameters (Alpha and Beta) ======
print("\n=== Exploring MDP Parameters (α, β) ===")

alpha_values = [0.5, 0.7, 0.9]
beta_values = [0.3, 0.6, 0.8]
mdp_results = []

for alpha in alpha_values:
    for beta in beta_values:
        print(f"Testing MDP: α={alpha}, β={beta}")

        # Create environment and agent with default learning params
        env = RecyclingRobotEnvironment(alpha=alpha, beta=beta, r_search=3.0, r_wait=1.0)
        agent = TemporalDifferenceAgent(env, learning_rate=0.1, discount_factor=0.9, epsilon=0.1)

        # Training
        total_rewards = []
        policies = []

        for epoch in range(40):
            epoch_reward = 0
            state = env.reset()

            for step in range(500):
                action = agent.select_action(state)
                next_state, reward = env.step(action)
                agent.update_q_value(state, action, reward, next_state)
                epoch_reward += reward
                state = next_state

            total_rewards.append(epoch_reward)

            # Record policy every 5 epochs
            if epoch % 5 == 0:
                current_policy = agent.get_policy()
                policy_string = f"{current_policy[State.HIGH].value}_{current_policy[State.LOW].value}"
                policies.append(policy_string)

            agent.epsilon = max(0.01, agent.epsilon * 0.995)

        # Calculate metrics
        avg_reward = np.mean(total_rewards[-5:])
        final_policy = policies[-1]
        policy_stability = policies.count(final_policy) / len(policies)

        mdp_results.append({
            'alpha': alpha,
            'beta': beta,
            'avg_reward': avg_reward,
            'policy_stability': policy_stability
        })

print(f"MDP exploration completed: {len(mdp_results)} experiments")

# ====== PART 2: Learning Parameters ======
print("\n=== Exploring Learning Parameters (lr, γ) ===")

learning_rates = [0.05, 0.1, 0.2]
discount_factors = [0.8, 0.9, 0.95]
learning_results = []

for lr in learning_rates:
    for gamma in discount_factors:
        print(f"Testing Learning: lr={lr}, γ={gamma}")

        # Create environment and agent with default MDP params
        env = RecyclingRobotEnvironment(alpha=0.7, beta=0.6, r_search=3.0, r_wait=1.0)
        agent = TemporalDifferenceAgent(env, learning_rate=lr, discount_factor=gamma, epsilon=0.1)

        # Training
        total_rewards = []
        policies = []

        for epoch in range(40):
            epoch_reward = 0
            state = env.reset()

            for step in range(500):
                action = agent.select_action(state)
                next_state, reward = env.step(action)
                agent.update_q_value(state, action, reward, next_state)
                epoch_reward += reward
                state = next_state

            total_rewards.append(epoch_reward)

            # Record policy every 5 epochs
            if epoch % 5 == 0:
                current_policy = agent.get_policy()
                policy_string = f"{current_policy[State.HIGH].value}_{current_policy[State.LOW].value}"
                policies.append(policy_string)

            agent.epsilon = max(0.01, agent.epsilon * 0.995)

        # Calculate metrics
        avg_reward = np.mean(total_rewards[-5:])
        final_policy = policies[-1]
        policy_stability = policies.count(final_policy) / len(policies)

        learning_results.append({
            'learning_rate': lr,
            'discount_factor': gamma,
            'avg_reward': avg_reward,
            'policy_stability': policy_stability
        })

print(f"Learning exploration completed: {len(learning_results)} experiments")

# ====== PLOTTING ======
print("\n=== Creating plots ===")

# MDP Parameters plots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Parameter Exploration Results', fontsize=16, fontweight='bold')

# MDP Average Reward
mdp_reward_matrix = np.zeros((len(alpha_values), len(beta_values)))
for i, a in enumerate(alpha_values):
    for j, b in enumerate(beta_values):
        reward = [r['avg_reward'] for r in mdp_results if r['alpha'] == a and r['beta'] == b][0]
        mdp_reward_matrix[i, j] = reward

sns.heatmap(mdp_reward_matrix, annot=True, fmt='.0f',
           xticklabels=beta_values, yticklabels=alpha_values,
           cmap='viridis', ax=axes[0,0])
axes[0,0].set_title('MDP: Average Reward')
axes[0,0].set_xlabel('Beta (β)')
axes[0,0].set_ylabel('Alpha (α)')

# MDP Policy Stability
mdp_stability_matrix = np.zeros((len(alpha_values), len(beta_values)))
for i, a in enumerate(alpha_values):
    for j, b in enumerate(beta_values):
        stability = [r['policy_stability'] for r in mdp_results if r['alpha'] == a and r['beta'] == b][0]
        mdp_stability_matrix[i, j] = stability

sns.heatmap(mdp_stability_matrix, annot=True, fmt='.2f',
           xticklabels=beta_values, yticklabels=alpha_values,
           cmap='Blues', ax=axes[0,1])
axes[0,1].set_title('MDP: Policy Stability')
axes[0,1].set_xlabel('Beta (β)')
axes[0,1].set_ylabel('Alpha (α)')

# Learning Average Reward
learning_reward_matrix = np.zeros((len(learning_rates), len(discount_factors)))
for i, lr in enumerate(learning_rates):
    for j, gamma in enumerate(discount_factors):
        reward = [r['avg_reward'] for r in learning_results if r['learning_rate'] == lr and r['discount_factor'] == gamma][0]
        learning_reward_matrix[i, j] = reward

sns.heatmap(learning_reward_matrix, annot=True, fmt='.0f',
           xticklabels=discount_factors, yticklabels=learning_rates,
           cmap='viridis', ax=axes[1,0])
axes[1,0].set_title('Learning: Average Reward')
axes[1,0].set_xlabel('Discount Factor (γ)')
axes[1,0].set_ylabel('Learning Rate')

# Learning Policy Stability
learning_stability_matrix = np.zeros((len(learning_rates), len(discount_factors)))
for i, lr in enumerate(learning_rates):
    for j, gamma in enumerate(discount_factors):
        stability = [r['policy_stability'] for r in learning_results if r['learning_rate'] == lr and r['discount_factor'] == gamma][0]
        learning_stability_matrix[i, j] = stability

sns.heatmap(learning_stability_matrix, annot=True, fmt='.2f',
           xticklabels=discount_factors, yticklabels=learning_rates,
           cmap='Blues', ax=axes[1,1])
axes[1,1].set_title('Learning: Policy Stability')
axes[1,1].set_xlabel('Discount Factor (γ)')
axes[1,1].set_ylabel('Learning Rate')

plt.tight_layout()
plt.savefig('parameter_exploration.png)', dpi=300, bbox_inches='tight')
print("Plot saved as parameter_exploration.png)")
plt.show()

# ====== RESULTS SUMMARY ======
print("\n=== MDP Parameters Results ===")
print(f"{'Alpha':<8} {'Beta':<8} {'Avg Reward':<12} {'Stability':<10}")
print("-" * 40)
for r in sorted(mdp_results, key=lambda x: x['avg_reward'], reverse=True):
    print(f"{r['alpha']:<8} {r['beta']:<8} {r['avg_reward']:<12.0f} {r['policy_stability']:<10.2f}")

print("\n=== Learning Parameters Results ===")
print(f"{'LR':<8} {'Gamma':<8} {'Avg Reward':<12} {'Stability':<10}")
print("-" * 40)
for r in sorted(learning_results, key=lambda x: x['avg_reward'], reverse=True):
    print(f"{r['learning_rate']:<8} {r['discount_factor']:<8} {r['avg_reward']:<12.0f} {r['policy_stability']:<10.2f}")

# Find best overall
best_mdp = max(mdp_results, key=lambda x: x['avg_reward'])
best_learning = max(learning_results, key=lambda x: x['avg_reward'])

print(f"\n=== Best Configurations ===")
print(f"Best MDP: α={best_mdp['alpha']}, β={best_mdp['beta']}, reward={best_mdp['avg_reward']:.0f}")
print(f"Best Learning: lr={best_learning['learning_rate']}, γ={best_learning['discount_factor']}, reward={best_learning['avg_reward']:.0f}")