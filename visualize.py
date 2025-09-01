"""
Visualization module for Recycling Robot RL results.

This module creates plots for training progress and policy visualization.
"""

from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from recycling_robot import RecyclingRobotEnvironment, TemporalDifferenceAgent, State, Action


class RecyclingRobotVisualizer:
    """
    Visualizer for Recycling Robot training results and policies.

    Creates publication-quality plots for training curves and policy heatmaps.
    """

    def __init__(self, style: str = "whitegrid") -> None:
        """
        Initialize the visualizer.

        Args:
            style: Seaborn style for plots
        """
        plt.style.use('default')
        sns.set_style(style)
        sns.set_palette("husl")

    def load_rewards_data(self, filename: str = "rewards.txt") -> pd.DataFrame:
        """
        Load rewards data from file.

        Args:
            filename: Path to rewards file

        Returns:
            DataFrame with epoch and reward columns
        """
        filepath = Path(filename)
        if not filepath.exists():
            raise FileNotFoundError(f"Rewards file not found: {filepath}")

        return pd.read_csv(filepath)

    def plot_training_curve(
        self,
        rewards_data: pd.DataFrame,
        window_size: int = 10,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot training curve with moving average.

        Args:
            rewards_data: DataFrame with epoch and reward columns
            window_size: Window size for moving average smoothing
            save_path: Optional path to save the plot
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        epochs = rewards_data['epoch']
        rewards = rewards_data['reward']

        # Plot raw rewards
        ax.plot(epochs, rewards, alpha=0.3, color='lightblue', label='Raw Rewards')

        # Plot moving average
        if len(rewards) >= window_size:
            moving_avg = rewards.rolling(window=window_size).mean()
            ax.plot(epochs, moving_avg, linewidth=2, color='darkblue',
                   label=f'Moving Average (window={window_size})')

        # Add trend line
        z = np.polyfit(epochs, rewards, 1)
        p = np.poly1d(z)
        ax.plot(epochs, p(epochs), linestyle='--', color='red', alpha=0.8,
               label=f'Trend (slope={z[0]:.3f})')

        ax.set_xlabel('Epoch', fontsize=14)
        ax.set_ylabel('Cumulative Reward', fontsize=14)
        ax.set_title('Recycling Robot Learning Progress', fontsize=16, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)

        # Add statistics text box
        stats_text = (
            f'Final Reward: {rewards.iloc[-1]:.2f}\n'
            f'Max Reward: {rewards.max():.2f}\n'
            f'Mean Reward: {rewards.mean():.2f}\n'
            f'Std Reward: {rewards.std():.2f}'
        )
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training curve saved to {save_path}")

        plt.show()

    def create_policy_heatmap(
        self,
        agent: TemporalDifferenceAgent,
        save_path: Optional[str] = None
    ) -> None:
        """
        Create a heatmap visualization of the optimal policy.

        Args:
            agent: Trained TD agent
            save_path: Optional path to save the plot
        """
        # Get policy and action probabilities for each state
        states = [State.HIGH, State.LOW]
        actions = [Action.SEARCH, Action.WAIT, Action.RECHARGE]

        # Create data matrix for heatmap
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
                    row.append(np.nan)  # Invalid action for this state

            policy_data.append(row)
            state_labels.append(state.value.upper())

        # Convert to DataFrame for seaborn
        df = pd.DataFrame(
            policy_data,
            index=state_labels,
            columns=[action.value.upper() for action in actions]
        )

        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 6))

        # Custom colormap that handles NaN values
        cmap = sns.color_palette("YlOrRd", as_cmap=True)

        heatmap = sns.heatmap(
            df,
            annot=True,
            fmt='.3f',
            cmap=cmap,
            cbar_kws={'label': 'Action Probability'},
            ax=ax,
            mask=df.isna(),  # Mask invalid actions
            linewidths=0.5,
            square=True
        )

        # Customize the plot
        ax.set_title('Recycling Robot Optimal Policy\n(Action Selection Probabilities)',
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Actions', fontsize=14)
        ax.set_ylabel('Battery States', fontsize=14)

        # Add text annotations for invalid actions
        for i, state in enumerate(states):
            valid_actions = agent.env.get_valid_actions(state)
            for j, action in enumerate(actions):
                if action not in valid_actions:
                    ax.text(j + 0.5, i + 0.5, 'N/A', ha='center', va='center',
                           fontsize=12, color='gray', fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Policy heatmap saved to {save_path}")

        plt.show()

    def create_q_values_heatmap(
        self,
        agent: TemporalDifferenceAgent,
        save_path: Optional[str] = None
    ) -> None:
        """
        Create a heatmap of Q-values for all state-action pairs.

        Args:
            agent: Trained TD agent
            save_path: Optional path to save the plot
        """
        states = [State.HIGH, State.LOW]
        actions = [Action.SEARCH, Action.WAIT, Action.RECHARGE]

        # Create Q-values matrix
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

        # Convert to DataFrame
        df = pd.DataFrame(
            q_matrix,
            index=state_labels,
            columns=[action.value.upper() for action in actions]
        )

        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 6))

        sns.heatmap(
            df,
            annot=True,
            fmt='.2f',
            cmap='RdYlBu_r',
            center=0,
            cbar_kws={'label': 'Q-Value'},
            ax=ax,
            mask=df.isna(),
            linewidths=0.5,
            square=True
        )

        ax.set_title('Recycling Robot Q-Values\n(State-Action Value Function)',
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Actions', fontsize=14)
        ax.set_ylabel('Battery States', fontsize=14)

        # Add text for invalid actions
        for i, state in enumerate(states):
            valid_actions = agent.env.get_valid_actions(state)
            for j, action in enumerate(actions):
                if action not in valid_actions:
                    ax.text(j + 0.5, i + 0.5, 'N/A', ha='center', va='center',
                           fontsize=12, color='gray', fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Q-values heatmap saved to {save_path}")

        plt.show()

    def create_comprehensive_report(
        self,
        rewards_file: str = "rewards.txt",
        window_size: int = 10
    ) -> None:
        """
        Create a comprehensive visualization report.

        Args:
            rewards_file: Path to rewards data file
            window_size: Moving average window size
        """
        print("Creating comprehensive visualization report...")

        # Load rewards data
        try:
            rewards_data = self.load_rewards_data(rewards_file)
        except FileNotFoundError:
            print(f"Error: Could not find rewards file '{rewards_file}'")
            print("Please run the training script first.")
            return

        # Create training curve plot
        print("1. Generating training curve...")
        self.plot_training_curve(
            rewards_data,
            window_size=window_size,
            save_path="training_curve.png"
        )

        # Create a trained agent for policy visualization
        print("2. Training agent for policy visualization...")
        env = RecyclingRobotEnvironment()
        agent = TemporalDifferenceAgent(env, epsilon=0.0)  # Greedy policy for visualization

        # Quick training for policy extraction
        for _ in range(1000):
            state = env.reset()
            for _ in range(100):
                action = agent.select_action(state)
                next_state, reward = env.step(action)
                agent.update_q_value(state, action, reward, next_state)
                state = next_state

        # Create policy and Q-value heatmaps
        print("3. Generating policy heatmap...")
        self.create_policy_heatmap(agent, save_path="policy_heatmap.png")

        print("4. Generating Q-values heatmap...")
        self.create_q_values_heatmap(agent, save_path="q_values_heatmap.png")

        print("\nVisualization report completed!")
        print("Generated files:")
        print("- training_curve.png")
        print("- policy_heatmap.png")
        print("- q_values_heatmap.png")


def main() -> None:
    """Main visualization function."""
    visualizer = RecyclingRobotVisualizer()

    # Check if rewards file exists
    if not Path("rewards.txt").exists():
        print("Rewards file not found. Running training first...")
        import subprocess
        try:
            subprocess.run(["python", "train.py"], check=True)
        except subprocess.CalledProcessError:
            print("Error: Could not run training script")
            return
        except FileNotFoundError:
            print("Error: Python not found. Please run 'python train.py' manually first.")
            return

    # Create comprehensive report
    visualizer.create_comprehensive_report()


if __name__ == "__main__":
    main()