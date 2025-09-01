"""
Recycling Robot Reinforcement Learning Implementation

This module implements the Recycling Robot problem from Sutton & Barto's
"Reinforcement Learning: An Introduction" (Example 3.3) using Temporal
Difference (TD) learning.
"""

from enum import Enum
from typing import NamedTuple
import random


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
        new_q = current_q + self.ls * td_error

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