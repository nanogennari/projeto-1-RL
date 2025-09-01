"""
Main script for the Recycling Robot RL experiment.

This script runs the complete training and visualization pipeline.
"""

from classes import RecyclingRobotExperiment


def main() -> None:
    """Main experiment function."""
    experiment = RecyclingRobotExperiment(
        steps_per_epoch=1000,
        learning_rate=0.1,
        discount_factor=0.9,
        epsilon=0.3,
        epsilon_decay=0.995,
        min_epsilon=0.01
    )

    experiment.run_complete_experiment(num_runs=5, num_epochs=100)


if __name__ == "__main__":
    main()