from .base import BaseAgent
import torch
import torch.nn as nn

class PolicyGradientAgent(BaseAgent):
    def __init__(self, env, config):
        super().__init__(env, config)
        # TODO: Define Policy Network
        # self.policy = ...
        
    def select_action(self, observation):
        # TODO: Implement action selection
        return self.env.action_space.sample()

    def update(self, rollouts):
        # TODO: Implement REINFORCE update
        pass
    
    def save(self, path):
        pass
    
    def load(self, path):
        pass
