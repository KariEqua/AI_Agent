import random
import numpy as np
import pickle


class AgentAI:

    def __init__(self, actions: list, alpha=0.5, epsilon=0.1):
        self.q = dict()
        self.alpha = alpha
        self.epsilon = epsilon
        self.actions = actions
        self.load_q_values()

    def load_q_values(self, file_path='q_values.pickle'):
        """
        Load Q-values from a pickle file.
        """
        # Reading the dictionary from a pickle file
        try:
            with open(file_path, 'rb') as pickle_file:
                self.q = pickle.load(pickle_file)
        except FileNotFoundError:
            print("Q-values file not found. Starting with empty Q-values.")

    def save_q_values(self, file_path='q_values.pickle'):
        """
        Save Q-values to a file using pickle serialization.
        """
        # Writing the dictionary to a file using pickle
        with open(file_path, 'wb') as pickle_file:
            pickle.dump(self.q, pickle_file)

    def get_q_value(self, board: np.ndarray, action):
        """
        Return the Q-value for the state `board` and the action `action`.
        If no Q-value exists yet in `self.q`, return 0.
        """
        if (board.tobytes(), action) in self.q:
            return self.q[(board.tobytes(), action)]
        return 0

    def update(self, old_board, action, new_board, reward):
        """
        Update Q-learning model, given an old board, an action taken
        in that state, a new resulting board, and the reward received
        from taking that action.
        """
        old = self.get_q_value(old_board, action)
        best_future = self.best_future_reward(new_board)
        self.update_q_value(old_board, action, old, reward, best_future)

    def update_q_value(self, board, action, old_q, reward, future_rewards):
        """
        Update the Q-value for the state `board` and the action `action`
        given the previous Q-value `old_q`, a current reward `reward`,
        and an estiamte of future reward `future_rewards`.

        Using the formula:
        Q(s, a) <- old value estimate
                   + alpha * (new value estimate - old value estimate)

        where `old value estimate` is the previous Q-value,
        `alpha` is the learning rate, and `new value estimate`
        is the sum of the current reward and estimated future reward.
        """
        new_q = old_q + self.alpha * (reward + future_rewards - old_q)
        self.q[(board.tobytes(), action)] = new_q

    def best_future_reward(self, board):
        """
        Given a state `board`, consider all possible `(board, action)`
        pairs available in that state and return the maximum of all
        of their Q-values.

        Use 0 as the Q-value if a `(board, action)` pair has no
        Q-value in `self.q`. If there are no available actions in
        `board`, return 0.
        """
        max_q = 0
        for action in self.actions:
            q_value = self.get_q_value(board, action)
            if q_value > max_q:
                max_q = q_value
        return max_q

    def choose_action(self, board, epsilon=False):
        """
        Given a state `board`, return an action `move` to take.

        If `epsilon` is `False`, then return the best action
        available in the state (the one with the highest Q-value,
        using 0 for pairs that have no Q-values).

        If `epsilon` is `True`, then with probability
        `self.epsilon` choose a random available action,
        otherwise choose the best action available.

        """
        best_action = self.best_action(board)
        if epsilon:
            random_action = random.choice(self.actions)
            probabilities = [1 - self.epsilon, self.epsilon]
            move = random.choices([best_action, random_action], probabilities)[0]
        else:
            move = best_action

        return move

    def best_action(self, board):
        """
        For a given 'board' and all possible actions,
        return action with highest Q-value.
        If multiple actions have the same Q-value, return random action.
        """

        best_action = random.choice(self.actions)
        max_q = self.get_q_value(board, best_action)

        for action in self.actions:
            q_value = self.get_q_value(board, action)
            if q_value > max_q:
                max_q = q_value
                best_action = action

        return best_action

