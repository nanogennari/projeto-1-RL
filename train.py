"""
Training script for the Recycling Robot RL agent.

This script trains a TD learning agent on the recycling robot environment
and tracks performance metrics across multiple epochs.
"""

from pathlib import Path
from typing import NamedTuple
import numpy as np
from recycling_robot import RecyclingRobotEnvironment, TemporalDifferenceAgent, State


class TrainingResults(NamedTuple):
    """Results from a training run."""
    epoch_rewards: list[float]
    final_policy: dict[State, str]
    final_q_values: dict[tuple[State, str], float]


class RecyclingRobotTrainer:
    """
    Trainer for the Recycling Robot TD learning agent.

    Manages training loops, data collection, and results persistence.
    """

    def __init__(
        self,
        steps_per_epoch: int = 1000,
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        epsilon: float = 0.1,
        epsilon_decay: float = 0.995,
        min_epsilon: float = 0.01
    ) -> None:
        """
        Initialize the trainer.

        Args:
            steps_per_epoch: Number of steps per training epoch
            learning_rate: TD learning rate (alpha)
            discount_factor: Discount factor (gamma)
            epsilon: Initial exploration rate
            epsilon_decay: Decay rate for epsilon per epoch
            min_epsilon: Minimum epsilon value
        """
        self.steps_per_epoch = steps_per_epoch
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        # Initialize environment and agent
        self.env = RecyclingRobotEnvironment()
        self.agent = TemporalDifferenceAgent(
            environment=self.env,
            learning_rate=self.learning_rate,
            discount_factor=self.discount_factor,
            epsilon=self.epsilon
        )

    def run_epoch(self) -> float:
        """
        Run one training epoch.

        Returns:
            Total reward accumulated during the epoch
        """
        total_reward = 0.0
        state = self.env.reset()

        for step in range(self.steps_per_epoch):
            # Select and execute action
            action = self.agent.select_action(state)
            next_state, reward = self.env.step(action)

            # Update Q-values using TD learning
            self.agent.update_q_value(state, action, reward, next_state)

            # Accumulate reward and move to next state
            total_reward += reward
            state = next_state

        return total_reward

    def train(self, num_epochs: int) -> TrainingResults:
        """
        Train the agent for multiple epochs.

        Args:
            num_epochs: Number of epochs to train

        Returns:
            Training results including epoch rewards and final policy
        """
        epoch_rewards: list[float] = []

        print(f"Starting training for {num_epochs} epochs...")
        print(f"Steps per epoch: {self.steps_per_epoch}")
        print(f"Environment parameters: α={self.env.alpha}, β={self.env.beta}")
        print(f"Rewards: r_search={self.env.r_search}, r_wait={self.env.r_wait}")
        print(f"Learning parameters: α={self.learning_rate}, γ={self.discount_factor}")
        print("-" * 60)

        for epoch in range(num_epochs):
            # Run training epoch
            epoch_reward = self.run_epoch()
            epoch_rewards.append(epoch_reward)

            # Decay exploration rate
            self.agent.epsilon = max(
                self.min_epsilon,
                self.agent.epsilon * self.epsilon_decay
            )

            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch + 1:4d}: Reward = {epoch_reward:8.2f}, "
                      f"ε = {self.agent.epsilon:.4f}")

        # Extract final policy and Q-values for analysis
        final_policy = {
            state: action.value for state, action in self.agent.get_policy().items()
        }

        final_q_values = {
            (state.value, action.value): q_val
            for (state, action), q_val in self.agent.q_values.items()
        }

        print("-" * 60)
        print("Training completed!")
        print(f"Final policy: {final_policy}")

        return TrainingResults(epoch_rewards, final_policy, final_q_values)

    def save_rewards(self, epoch_rewards: list[float], filename: str = "rewards.txt") -> None:
        """
        Save epoch rewards to file.

        Args:
            epoch_rewards: List of rewards per epoch
            filename: Output filename
        """
        filepath = Path(filename)

        with open(filepath, 'w') as f:
            f.write("epoch,reward\n")
            for epoch, reward in enumerate(epoch_rewards, 1):
                f.write(f"{epoch},{reward:.6f}\n")

        print(f"Rewards saved to {filepath}")

    def run_multiple_training_runs(
        self,
        num_runs: int,
        num_epochs: int
    ) -> tuple[list[list[float]], dict[State, str], dict[tuple[State, str], float]]:
        """
        Run multiple independent training runs for averaging.

        Args:
            num_runs: Number of independent training runs
            num_epochs: Number of epochs per run

        Returns:
            Tuple of (all_epoch_rewards, final_policy, final_q_values)
        """
        all_rewards: list[list[float]] = []

        print(f"Running {num_runs} independent training runs...")

        for run in range(num_runs):
            print(f"\n=== Training Run {run + 1}/{num_runs} ===")

            # Reset agent for new run
            self.agent._init_value_function()
            self.agent.epsilon = self.epsilon

            # Train and collect results
            results = self.train(num_epochs)
            all_rewards.append(results.epoch_rewards)

            # Use the final run's policy and Q-values
            if run == num_runs - 1:
                final_policy = results.final_policy
                final_q_values = results.final_q_values

        return all_rewards, final_policy, final_q_values


def main() -> None:
    """Main training function."""
    # Training parameters
    NUM_EPOCHS = 100
    NUM_RUNS = 5
    STEPS_PER_EPOCH = 1000

    # Create trainer
    trainer = RecyclingRobotTrainer(
        steps_per_epoch=STEPS_PER_EPOCH,
        learning_rate=0.1,
        discount_factor=0.9,
        epsilon=0.3,
        epsilon_decay=0.995,
        min_epsilon=0.01
    )

    # Run multiple training runs
    all_rewards, final_policy, final_q_values = trainer.run_multiple_training_runs(
        num_runs=NUM_RUNS,
        num_epochs=NUM_EPOCHS
    )

    # Calculate average rewards across runs
    avg_rewards = np.mean(all_rewards, axis=0)
    std_rewards = np.std(all_rewards, axis=0)

    # Save average rewards
    trainer.save_rewards(avg_rewards.tolist(), "rewards.txt")

    # Save detailed results
    with open("training_results.txt", 'w') as f:
        f.write("=== Recycling Robot Training Results ===\n\n")
        f.write(f"Training Parameters:\n")
        f.write(f"- Number of runs: {NUM_RUNS}\n")
        f.write(f"- Epochs per run: {NUM_EPOCHS}\n")
        f.write(f"- Steps per epoch: {STEPS_PER_EPOCH}\n")
        f.write(f"- Learning rate: {trainer.learning_rate}\n")
        f.write(f"- Discount factor: {trainer.discount_factor}\n")
        f.write(f"- Initial epsilon: {trainer.epsilon}\n\n")

        f.write(f"Environment Parameters:\n")
        f.write(f"- Alpha (α): {trainer.env.alpha}\n")
        f.write(f"- Beta (β): {trainer.env.beta}\n")
        f.write(f"- r_search: {trainer.env.r_search}\n")
        f.write(f"- r_wait: {trainer.env.r_wait}\n\n")

        f.write(f"Final Policy:\n")
        for state, action in final_policy.items():
            f.write(f"- {state}: {action}\n")
        f.write(f"\nFinal Q-Values:\n")
        for (state, action), q_val in final_q_values.items():
            f.write(f"- Q({state}, {action}): {q_val:.4f}\n")

        f.write(f"\nFinal Average Reward: {avg_rewards[-1]:.2f} ± {std_rewards[-1]:.2f}\n")

    print(f"\nTraining completed! Results saved to training_results.txt")
    print(f"Average final reward: {avg_rewards[-1]:.2f} ± {std_rewards[-1]:.2f}")


if __name__ == "__main__":
    main()