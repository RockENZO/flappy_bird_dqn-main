import numpy as np
import pygame
from pytorch_mlp import MLPRegression
import argparse
from console import FlappyBirdEnv
from collections import deque

STUDENT_ID = 'a1880714'
DEGREE = 'UG'  # or 'PG'


class MyAgent:
    def __init__(self, show_screen=False, load_model_path=None, mode=None):
        # do not modify these
        self.show_screen = show_screen
        if mode is None:
            self.mode = 'train'  # mode is either 'train' or 'eval', we will set the mode of your agent to eval mode
        else:
            self.mode = mode

        # modify these
        self.storage = deque(maxlen=10000)  # a data structure of your choice (D in the Algorithm 2)
        # A neural network MLP model which can be used as Q
        self.network = MLPRegression(input_dim=5, output_dim=2, learning_rate=0.001)
        # network2 has identical structure to network1, network2 is the Q_f
        self.network2 = MLPRegression(input_dim=5, output_dim=2, learning_rate=0.001)
        # initialise Q_f's parameter by Q's, here is an example
        MyAgent.update_network_model(net_to_update=self.network2, net_as_source=self.network)

        self.epsilon = 1.0  # probability ε in Algorithm 2
        self.min_epsilon = 0.01
        self.epsilon_decay = 0.995
        self.n = 64  # the number of samples you'd want to draw from the storage each time
        self.discount_factor = 0.99  # γ in Algorithm 2

        # do not modify this
        if load_model_path:
            self.load_model(load_model_path)

    def build_state(self, state: dict) -> np.ndarray:
        """
        Convert the raw game state into a normalized feature vector.
        Args:
            state: The raw state dictionary from the game environment.
        Returns:
            A numpy array representing the feature vector.
        """
        # Extract and normalize features
        bird_y = state['bird_y'] / state['screen_height']
        bird_velocity = state['bird_velocity'] / 15  # Adjust based on observed max velocity
        pipes = state['pipes']
        if pipes:
            pipe_x = pipes[0]['x'] / state['screen_width']
            pipe_gap_center = ((pipes[0]['top'] + pipes[0]['bottom']) / 2) / state['screen_height']
            distance_to_gap = abs(state['bird_y'] - pipe_gap_center) / state['screen_height']
        else:
            pipe_x, pipe_gap_center, distance_to_gap = 1.0, 0.5, 0.5
        return np.array([bird_y, bird_velocity, pipe_x, pipe_gap_center, distance_to_gap])

    def compute_reward(self, state: dict) -> float:
        """
        Compute the reward based on the current game state.
        Args:
            state: The raw state dictionary from the game environment.
        Returns:
            A float representing the reward.
        """
        if state.get('done', False):
            return -1.0 if state['done_type'] == 'hit_pipe' else -0.5
        else:
            reward = 0.1  # Base reward for staying alive
            if state['pipes']:
                pipe = state['pipes'][0]
                pipe_gap_center = (pipe['top'] + pipe['bottom']) / 2
                distance_to_gap = abs(state['bird_y'] - pipe_gap_center) / state['screen_height']
                reward -= distance_to_gap ** 2  # Quadratic penalty for distance
                if pipe['x'] + pipe['width'] < state['bird_x']:
                    reward += 2.0  # Reward for passing a pipe
            return reward

    def choose_action(self, state: dict, action_table: dict) -> int:
        """
        This function should be called when the agent action is requested.
        Args:
            state: input state representation (the state dictionary from the game environment)
            action_table: the action code dictionary
        Returns:
            action: the action code as specified by the action_table
        """
        # Map valid actions to indices (0: jump, 1: do_nothing)
        valid_actions = {action_table['jump']: 0, action_table['do_nothing']: 1}

        phi_t = self.build_state(state)
        if self.mode == 'train':
            # ε-greedy action selection
            if np.random.rand() < self.epsilon:
                # Select a random valid action
                a_t = np.random.choice(list(valid_actions.keys()))
            else:
                # Select the action that maximizes Q(ϕ_t, a)
                q_values = self.network.predict(np.array([phi_t]))
                a_t = max(valid_actions, key=lambda action: q_values[0][valid_actions[action]])

            # Store the partial transition (ϕ_t, a_t, r_t=None, q_t+1=None) in memory
            self.storage.append({'phi_t': phi_t, 'action': valid_actions[a_t], 'reward': None, 'q_t1': None})
        elif self.mode == 'eval':
            # Always select the action that maximizes Q(ϕ_t, a) in evaluation mode
            q_values = self.network.predict(np.array([phi_t]))
            a_t = max(valid_actions, key=lambda action: q_values[0][valid_actions[action]])

        return a_t

    def receive_after_action_observation(self, state: dict, action_table: dict) -> None:
        """
        This function should be called to notify the agent of the post-action observation.
        Args:
            state: post-action state representation (the state dictionary from the game environment)
            action_table: the action code dictionary
        Returns:
            None
        """
        if self.mode == 'train':
            # Build the state representation (ϕ_t+1)
            phi_t1 = self.build_state(state)

            # Define the reward r_t based on the current state
            reward = self.compute_reward(state)

            # Compute the Q-value for t+1
            if state.get('done', False):  # Check if the game is over
                q_t1 = 0  # Terminal state
            else:
                q_t1 = np.max(self.network2.predict(np.array([phi_t1])))

            # Update the last transition in memory with (r_t, q_t+1)
            if self.storage:
                self.storage[-1]['reward'] = reward
                self.storage[-1]['q_t1'] = q_t1

            # Sample a random minibatch of transitions from memory
            if len(self.storage) >= self.n:
                minibatch = np.random.choice(self.storage, self.n, replace=False)

                # Prepare training data
                X, Y, W = [], [], []
                for transition in minibatch:
                    phi_t = transition['phi_t']
                    action = transition['action']
                    r_t = transition['reward']
                    q_t1 = transition['q_t1']

                    # Compute the target value
                    target = r_t + self.discount_factor * q_t1
                    y = np.zeros(2)  # Match the output size of the network (2 actions: jump, do_nothing)
                    y[action] = target
                    w = np.zeros(2)  # Match the output size of the network
                    w[action] = 1

                    X.append(phi_t)
                    Y.append(y)
                    W.append(w)

                # Train the Q network
                self.network.fit_step(np.array(X), np.array(Y), np.array(W))

            # Optionally decay epsilon
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)


    def save_model(self, path: str = 'my_model.ckpt'):
        """
        Save the MLP model. Unless you decide to implement the MLP model yourself, do not modify this function.

        Args:
            path: the full path to save the model weights, ending with the file name and extension

        Returns:

        """
        self.network.save_model(path=path)

    def load_model(self, path: str = 'my_model.ckpt'):
        """
        Load the MLP model weights.  Unless you decide to implement the MLP model yourself, do not modify this function.
        Args:
            path: the full path to load the model weights, ending with the file name and extension

        Returns:

        """
        self.network.load_model(path=path)

    @staticmethod
    def update_network_model(net_to_update: MLPRegression, net_as_source: MLPRegression):
        """
        Update one MLP model's model parameter by the parameter of another MLP model.
        Args:
            net_to_update: the MLP to be updated
            net_as_source: the MLP to supply the model parameters

        Returns:
            None
        """
        net_to_update.load_state_dict(net_as_source.state_dict())


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--level', type=int, default=3)

    args = parser.parse_args()

    # bare-bone code to train your agent (you may extend this part as well, we won't run your agent training code)
    env = FlappyBirdEnv(config_file_path='config.yml', show_screen=True, level=args.level, game_length=10)
    agent = MyAgent(show_screen=True)
    episodes = 10000
    clear_memory_frequency = 20  # Clear memory every 50 episodes
    update_frequency = 5  # Update Q_f every 10 episodes
    for episode in range(episodes):
        env.play(player=agent)

        # env.score has the score value from the last play
        # env.mileage has the mileage value from the last play
        print(env.score)
        print(env.mileage)
        print(f"Episode {episode}: Score = {env.score}, Epsilon = {agent.epsilon}")

        # store the best model based on your judgement
        agent.save_model(path='my_model.ckpt')

        # you'd want to clear the memory after one or a few episodes
        if (episode + 1) % clear_memory_frequency == 0:
            agent.storage = deque(maxlen=10000)

        # you'd want to update the fixed Q-target network (Q_f) with Q's model parameter after one or a few episodes
        if (episode + 1) % update_frequency == 0:
            MyAgent.update_network_model(net_to_update=agent.network2, net_as_source=agent.network)

    # the below resembles how we evaluate your agent
    env2 = FlappyBirdEnv(config_file_path='config.yml', show_screen=False, level=args.level)
    agent2 = MyAgent(show_screen=False, load_model_path='my_model.ckpt', mode='eval')

    episodes = 10
    scores = list()
    for episode in range(episodes):
        env2.play(player=agent2)
        scores.append(env2.score)

    print(np.max(scores))
    print(np.mean(scores))
