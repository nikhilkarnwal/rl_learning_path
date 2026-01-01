from .base import BaseAgent

class DQNAgent(BaseAgent):
    def __init__(self, env, config):
        super().__init__(env, config)
        # TODO: Define Q-Network and Target Network
        # self.q_net = ...
        
    def select_action(self, observation):
        return self.env.action_space.sample()

    def update(self, batch):
        # TODO: Implement Q-learning update
        pass

    def save(self, path):
        pass
    
    def load(self, path):
        pass
