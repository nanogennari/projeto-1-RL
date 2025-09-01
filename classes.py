"""
Core classes for the Recycling Robot Reinforcement Learning implementation.

This module contains the environment and agent classes for the recycling robot
problem from Sutton & Barto's "Reinforcement Learning: An Introduction".
"""

from enum import Enum
from typing import NamedTuple
from pathlib import Path
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class State(Enum):
    """Battery charge states for the recycling robot."""
    HIGH = "high"
    LOW = "low"


class Action(Enum):
    """Available actions for the recycling robot."""
    SEARCH = "search"
    WAIT = "wait"
    RECHARGE = "recharge"


class Transition(NamedTuple):
    """Represents a state transition with next state and reward."""
    next_state: State
    reward: float


class RecyclingRobotEnvironment:
    """
    Recycling Robot MDP Environment.

    Implements the finite MDP described in Sutton & Barto Example 3.3.
    The robot has two battery states (high, low) and can search, wait, or recharge.
    """

    def __init__(
        self,
        alpha: float = 0.7,
        beta: float = 0.6,
        r_search: float = 3.0,
        r_wait: float = 1.0
    ) -> None:
        """
        Initialize the recycling robot environment.

        Args:
            alpha: Probability that searching in high state keeps battery high
            beta: Probability that searching in low state keeps battery low
            r_search: Expected reward for searching (must be > r_wait)
            r_wait: Expected reward for waiting

        Raises:
            ValueError: If r_search <= r_wait
        """
        if r_search <= r_wait:
            raise ValueError("r_search must be greater than r_wait")

        self.alpha = alpha
        self.beta = beta
        self.r_search = r_search
        self.r_wait = r_wait
        self.rescue_penalty = -3.0

        self.current_state = State.HIGH

        # Define valid actions for each state
        self._valid_actions = {
            State.HIGH: [Action.SEARCH, Action.WAIT],
            State.LOW: [Action.SEARCH, Action.WAIT, Action.RECHARGE]
        }

    def get_valid_actions(self, state: State) -> list[Action]:
        """
        Get valid actions for a given state.

        Args:
            state: The current state

        Returns:
            List of valid actions for the state
        """
        return self._valid_actions[state].copy()

    def reset(self, initial_state: State = State.HIGH) -> State:
        """
        Reset the environment to initial state.

        Args:
            initial_state: State to reset to (default: HIGH)

        Returns:
            The initial state
        """
        self.current_state = initial_state
        return self.current_state

    def step(self, action: Action) -> tuple[State, float]:
        """
        Execute an action and return the next state and reward.

        Args:
            action: Action to execute

        Returns:
            Tuple of (next_state, reward)

        Raises:
            ValueError: If action is not valid for current state
        """
        if action not in self._valid_actions[self.current_state]:
            raise ValueError(f"Action {action} not valid for state {self.current_state}")

        transition = self._get_transition(self.current_state, action)
        self.current_state = transition.next_state

        return transition.next_state, transition.reward

    def _get_transition(self, state: State, action: Action) -> Transition:
        """
        Get the stochastic transition for a state-action pair.

        Args:
            state: Current state
            action: Action to take

        Returns:
            Transition with next state and reward
        """
        if state == State.HIGH:
            if action == Action.SEARCH:
                if random.random() < self.alpha:
                    return Transition(State.HIGH, self.r_search)
                else:
                    return Transition(State.LOW, self.r_search)
            elif action == Action.WAIT:
                return Transition(State.HIGH, self.r_wait)

        elif state == State.LOW:
            if action == Action.SEARCH:
                if random.random() < self.beta:
                    return Transition(State.LOW, self.r_search)
                else:
                    # Battery depleted, robot rescued and recharged
                    return Transition(State.HIGH, self.rescue_penalty)
            elif action == Action.WAIT:
                return Transition(State.LOW, self.r_wait)
            elif action == Action.RECHARGE:
                return Transition(State.HIGH, 0.0)

        raise ValueError(f"Invalid state-action pair: {state}, {action}")

    def get_all_states(self) -> list[State]:
        """Get all possible states."""
        return list(State)

    def get_all_actions(self) -> list[Action]:
        """Get all possible actions."""
        return list(Action)


class TemporalDifferenceAgent:
    """
    Temporal Difference learning agent for the Recycling Robot.

    Implements TD(0) learning with epsilon-greedy policy for action selection.
    """

    def __init__(
        self,
        environment: RecyclingRobotEnvironment,
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        epsilon: float = 0.1
    ) -> None:
        """
        Initialize the TD learning agent.

        Args:
            environment: The recycling robot environment
            learning_rate: Step size parameter (alpha)
            discount_factor: Discount factor (gamma)
            epsilon: Exploration parameter for epsilon-greedy policy
        """
        self.env = environment
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon

        # Initialize value function for all state-action pairs
        self._init_value_function()

    def _init_value_function(self) -> None:
        """Initialize the action-value function Q(s,a) to zeros."""
        self.q_values: dict[tuple[State, Action], float] = {}

        for state in self.env.get_all_states():
            for action in self.env.get_valid_actions(state):
                self.q_values[(state, action)] = 0.0

    def get_q_value(self, state: State, action: Action) -> float:
        """
        Get the Q-value for a state-action pair.

        Args:
            state: The state
            action: The action

        Returns:
            Q-value for the state-action pair
        """
        return self.q_values.get((state, action), 0.0)

    def get_best_action(self, state: State) -> Action:
        """
        Get the best action for a state according to current Q-values.

        Args:
            state: The current state

        Returns:
            Action with highest Q-value
        """
        valid_actions = self.env.get_valid_actions(state)
        return max(valid_actions, key=lambda a: self.get_q_value(state, a))

    def select_action(self, state: State) -> Action:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current state

        Returns:
            Selected action
        """
        if random.random() < self.epsilon:
            # Explore: choose random action
            return random.choice(self.env.get_valid_actions(state))
        else:
            # Exploit: choose best action
            return self.get_best_action(state)

    def update_q_value(
        self,
        state: State,
        action: Action,
        reward: float,
        next_state: State
    ) -> None:
        """
        Update Q-value using TD(0) update rule.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state reached
        """
        current_q = self.get_q_value(state, action)

        # Get maximum Q-value for next state
        next_actions = self.env.get_valid_actions(next_state)
        max_next_q = max(self.get_q_value(next_state, a) for a in next_actions)

        # TD(0) update: Q(s,a) <- Q(s,a) + α[r + γ*max_a Q(s',a) - Q(s,a)]
        td_target = reward + self.gamma * max_next_q
        td_error = td_target - current_q
        new_q = current_q + self.lr * td_error

        self.q_values[(state, action)] = new_q

    def get_policy(self) -> dict[State, Action]:
        """
        Extract the greedy policy from current Q-values.

        Returns:
            Dictionary mapping states to best actions
        """
        policy = {}
        for state in self.env.get_all_states():
            policy[state] = self.get_best_action(state)
        return policy

    def get_action_probabilities(self, state: State) -> dict[Action, float]:
        """
        Get action probabilities for visualization.

        Args:
            state: The state to get probabilities for

        Returns:
            Dictionary mapping actions to their selection probabilities
        """
        valid_actions = self.env.get_valid_actions(state)
        best_action = self.get_best_action(state)

        probs = {}
        for action in valid_actions:
            if action == best_action:
                probs[action] = 1.0 - self.epsilon + (self.epsilon / len(valid_actions))
            else:
                probs[action] = self.epsilon / len(valid_actions)

        return probs


class TrainingResults(NamedTuple):
    """Results from a training run."""
    epoch_rewards: list[float]
    final_agent: 'TemporalDifferenceAgent'
    final_policy: dict[State, str]
    final_q_values: dict[tuple[State, str], float]


class RecyclingRobotExperiment:
    """
    Complete experiment pipeline for the Recycling Robot RL problem.

    Handles training, visualization, and results analysis.
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
        Initialize the experiment.

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

        # Set up plotting style
        plt.style.use('default')
        sns.set_style("whitegrid")
        sns.set_palette("husl")

    def run_epoch(self, agent: TemporalDifferenceAgent) -> float:
        """
        Run one training epoch.

        Args:
            agent: The TD agent to train

        Returns:
            Total reward accumulated during the epoch
        """
        total_reward = 0.0
        state = self.env.reset()

        for _ in range(self.steps_per_epoch):
            # Select and execute action
            action = agent.select_action(state)
            next_state, reward = self.env.step(action)

            # Update Q-values using TD learning
            agent.update_q_value(state, action, reward, next_state)

            # Accumulate reward and move to next state
            total_reward += reward
            state = next_state

        return total_reward

    def train_agent(self, num_epochs: int) -> TrainingResults:
        """
        Train the agent for multiple epochs.

        Args:
            num_epochs: Number of epochs to train

        Returns:
            Training results including agent and metrics
        """
        agent = TemporalDifferenceAgent(
            environment=self.env,
            learning_rate=self.learning_rate,
            discount_factor=self.discount_factor,
            epsilon=self.epsilon
        )

        epoch_rewards: list[float] = []

        print(f"Training Recycling Robot for {num_epochs} epochs...")
        print(f"Steps per epoch: {self.steps_per_epoch}")
        print(f"Environment parameters: α={self.env.alpha}, β={self.env.beta}")
        print(f"Rewards: r_search={self.env.r_search}, r_wait={self.env.r_wait}")
        print(f"Learning parameters: α={self.learning_rate}, γ={self.discount_factor}")
        print("-" * 60)

        for epoch in range(num_epochs):
            # Run training epoch
            epoch_reward = self.run_epoch(agent)
            epoch_rewards.append(epoch_reward)

            # Decay exploration rate
            agent.epsilon = max(self.min_epsilon, agent.epsilon * self.epsilon_decay)

            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch + 1:4d}: Reward = {epoch_reward:8.2f}, "
                      f"ε = {agent.epsilon:.4f}")

        # Extract final policy and Q-values for analysis
        final_policy = {
            state: action.value for state, action in agent.get_policy().items()
        }

        final_q_values = {
            (state.value, action.value): q_val
            for (state, action), q_val in agent.q_values.items()
        }

        print("-" * 60)
        print("Training completed!")
        print(f"Final policy: {final_policy}")

        return TrainingResults(epoch_rewards, agent, final_policy, final_q_values)

    def run_multiple_training_runs(
        self,
        num_runs: int,
        num_epochs: int
    ) -> tuple[list[list[float]], TemporalDifferenceAgent, dict[State, str], dict[tuple[State, str], float]]:
        """
        Run multiple independent training runs for averaging.

        Args:
            num_runs: Number of independent training runs
            num_epochs: Number of epochs per run

        Returns:
            Tuple of (all_epoch_rewards, final_agent, final_policy, final_q_values)
        """
        all_rewards: list[list[float]] = []

        print(f"Running {num_runs} independent training runs...")

        for run in range(num_runs):
            print(f"\n=== Training Run {run + 1}/{num_runs} ===")

            # Train and collect results
            results = self.train_agent(num_epochs)
            all_rewards.append(results.epoch_rewards)

            # Use the final run's agent and results for visualization
            if run == num_runs - 1:
                final_agent = results.final_agent
                final_policy = results.final_policy
                final_q_values = results.final_q_values

        return all_rewards, final_agent, final_policy, final_q_values

    def save_results(
        self,
        all_rewards: list[list[float]],
        final_policy: dict[State, str],
        final_q_values: dict[tuple[State, str], float],
        num_runs: int,
        num_epochs: int
    ) -> None:
        """Save training results to files."""
        # Calculate average rewards
        avg_rewards = np.mean(all_rewards, axis=0)
        std_rewards = np.std(all_rewards, axis=0)

        # Save average rewards
        with open("rewards.txt", 'w') as f:
            f.write("epoch,reward\n")
            for epoch, reward in enumerate(avg_rewards, 1):
                f.write(f"{epoch},{reward:.6f}\n")

        # Save detailed results
        with open("training_results.txt", 'w') as f:
            f.write("=== Recycling Robot Training Results ===\n\n")
            f.write(f"Training Parameters:\n")
            f.write(f"- Number of runs: {num_runs}\n")
            f.write(f"- Epochs per run: {num_epochs}\n")
            f.write(f"- Steps per epoch: {self.steps_per_epoch}\n")
            f.write(f"- Learning rate: {self.learning_rate}\n")
            f.write(f"- Discount factor: {self.discount_factor}\n")
            f.write(f"- Initial epsilon: {self.epsilon}\n\n")

            f.write(f"Environment Parameters:\n")
            f.write(f"- Alpha (α): {self.env.alpha}\n")
            f.write(f"- Beta (β): {self.env.beta}\n")
            f.write(f"- r_search: {self.env.r_search}\n")
            f.write(f"- r_wait: {self.env.r_wait}\n\n")

            f.write(f"Final Policy:\n")
            for state, action in final_policy.items():
                f.write(f"- {state}: {action}\n")
            f.write(f"\nFinal Q-Values:\n")
            for (state, action), q_val in final_q_values.items():
                f.write(f"- Q({state}, {action}): {q_val:.4f}\n")

            f.write(f"\nFinal Average Reward: {avg_rewards[-1]:.2f} ± {std_rewards[-1]:.2f}\n")

        print(f"\nResults saved!")
        print(f"- rewards.txt")
        print(f"- training_results.txt")
        print(f"Average final reward: {avg_rewards[-1]:.2f} ± {std_rewards[-1]:.2f}")

    def plot_training_curve(
        self,
        all_rewards: list[list[float]],
        window_size: int = 10,
        save_path: str = "training_curve.png"
    ) -> None:
        """Plot training curve with multiple runs."""
        avg_rewards = np.mean(all_rewards, axis=0)
        std_rewards = np.std(all_rewards, axis=0)
        epochs = np.arange(1, len(avg_rewards) + 1)

        plt.figure(figsize=(12, 8))

        # Plot individual runs with low alpha
        for rewards in all_rewards:
            plt.plot(epochs, rewards, alpha=0.2, color='lightblue', linewidth=0.8)

        # Plot average with confidence band
        plt.plot(epochs, avg_rewards, linewidth=2, color='darkblue', label='Average')
        plt.fill_between(epochs, avg_rewards - std_rewards, avg_rewards + std_rewards,
                       alpha=0.3, color='blue', label='Standard Deviation')

        # Plot moving average
        if len(avg_rewards) >= window_size:
            moving_avg = pd.Series(avg_rewards).rolling(window=window_size).mean()
            plt.plot(epochs, moving_avg, linewidth=2, color='red',
                   label=f'Moving Average (window={window_size})')

        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Cumulative Reward', fontsize=14)
        plt.title('Recycling Robot Learning Progress', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)

        # Add statistics
        stats_text = (
            f'Runs: {len(all_rewards)}\n'
            f'Final Reward: {avg_rewards[-1]:.2f} ± {std_rewards[-1]:.2f}\n'
            f'Max Reward: {avg_rewards.max():.2f}\n'
            f'Mean Reward: {avg_rewards.mean():.2f}'
        )
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=11,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curve saved to {save_path}")
        plt.show()

    def create_policy_heatmap(
        self,
        agent: TemporalDifferenceAgent,
        save_path: str = "policy_heatmap.png"
    ) -> None:
        """Create policy heatmap using the trained agent."""
        states = [State.HIGH, State.LOW]
        actions = [Action.SEARCH, Action.WAIT, Action.RECHARGE]

        # Create data matrix
        policy_data = []
        state_labels = []

        for state in states:
            state_probs = agent.get_action_probabilities(state)
            valid_actions = agent.env.get_valid_actions(state)

            row = []
            for action in actions:
                if action in valid_actions:
                    row.append(state_probs[action])
                else:
                    row.append(np.nan)

            policy_data.append(row)
            state_labels.append(state.value.upper())

        df = pd.DataFrame(
            policy_data,
            index=state_labels,
            columns=[action.value.upper() for action in actions]
        )

        plt.figure(figsize=(10, 6))
        sns.heatmap(
            df, annot=True, fmt='.3f', cmap='YlOrRd', cbar_kws={'label': 'Action Probability'},
            mask=df.isna(), linewidths=0.5, square=True
        )

        plt.title('Recycling Robot Optimal Policy\n(Action Selection Probabilities)',
                fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Actions', fontsize=14)
        plt.ylabel('Battery States', fontsize=14)

        # Add N/A for invalid actions
        for i, state in enumerate(states):
            valid_actions = agent.env.get_valid_actions(state)
            for j, action in enumerate(actions):
                if action not in valid_actions:
                    plt.text(j + 0.5, i + 0.5, 'N/A', ha='center', va='center',
                           fontsize=12, color='gray', fontweight='bold')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Policy heatmap saved to {save_path}")
        plt.show()

    def create_q_values_heatmap(
        self,
        agent: TemporalDifferenceAgent,
        save_path: str = "q_values_heatmap.png"
    ) -> None:
        """Create Q-values heatmap using the trained agent."""
        states = [State.HIGH, State.LOW]
        actions = [Action.SEARCH, Action.WAIT, Action.RECHARGE]

        q_matrix = []
        state_labels = []

        for state in states:
            row = []
            for action in actions:
                if action in agent.env.get_valid_actions(state):
                    row.append(agent.get_q_value(state, action))
                else:
                    row.append(np.nan)
            q_matrix.append(row)
            state_labels.append(state.value.upper())

        df = pd.DataFrame(
            q_matrix,
            index=state_labels,
            columns=[action.value.upper() for action in actions]
        )

        plt.figure(figsize=(10, 6))
        sns.heatmap(
            df, annot=True, fmt='.2f', cmap='RdYlBu_r', center=0,
            cbar_kws={'label': 'Q-Value'}, mask=df.isna(),
            linewidths=0.5, square=True
        )

        plt.title('Recycling Robot Q-Values\n(State-Action Value Function)',
                fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Actions', fontsize=14)
        plt.ylabel('Battery States', fontsize=14)

        # Add N/A for invalid actions
        for i, state in enumerate(states):
            valid_actions = agent.env.get_valid_actions(state)
            for j, action in enumerate(actions):
                if action not in valid_actions:
                    plt.text(j + 0.5, i + 0.5, 'N/A', ha='center', va='center',
                           fontsize=12, color='gray', fontweight='bold')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Q-values heatmap saved to {save_path}")
        plt.show()

    def run_complete_experiment(
        self,
        num_runs: int = 5,
        num_epochs: int = 100
    ) -> None:
        """Run the complete training and visualization pipeline."""
        print("Starting Recycling Robot RL Experiment")
        print("=" * 60)

        # Training phase
        all_rewards, final_agent, final_policy, final_q_values = self.run_multiple_training_runs(
            num_runs=num_runs, num_epochs=num_epochs
        )

        # Save results
        self.save_results(all_rewards, final_policy, final_q_values, num_runs, num_epochs)

        # Visualization phase
        print("\nGenerating visualizations...")
        self.plot_training_curve(all_rewards)
        self.create_policy_heatmap(final_agent)
        self.create_q_values_heatmap(final_agent)

        print("\nExperiment completed successfully!")
        print("\nGenerated files:")
        files = ["rewards.txt", "training_results.txt", "training_curve.png",
                "policy_heatmap.png", "q_values_heatmap.png"]
        for file in files:
            if Path(file).exists():
                print(f"  * {file}")