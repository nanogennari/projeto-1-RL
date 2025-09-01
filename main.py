"""
Main script for the Recycling Robot RL experiment.

This script runs the complete training and visualization pipeline.
"""

from classes import RecyclingRobotExperiment


def main() -> None:
    """Main experiment function."""
    # Optimal parameters found through systematic exploration:
    # - Best MDP: α=0.9, β=0.6 (reward=1327, stability=75%)
    # - Best Learning: lr=0.05, γ=0.9 (reward=1143, stability=88%)
    experiment = RecyclingRobotExperiment(
        # Learning parameters (optimized)
        steps_per_epoch=1000,
        learning_rate=0.05,     # Optimal: slower but more stable learning
        discount_factor=0.9,    # Optimal: balanced long-term planning
        epsilon=0.1,            # Low exploration for faster convergence
        epsilon_decay=0.995,
        min_epsilon=0.01,
        # MDP environment parameters (optimized)
        alpha=0.9,              # Optimal: maximum efficiency in high state
        beta=0.6,               # Optimal: balanced risk in low state
        r_search=3.0,           # Standard: reward for active searching
        r_wait=1.0              # Standard: reward for waiting
    )

    experiment.run_complete_experiment(num_runs=5, num_epochs=100)


if __name__ == "__main__":
    main()