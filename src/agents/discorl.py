from .base import BaseAgent

class DiscoRLAgent(BaseAgent):
    """
    Placeholder for Distribution-Conditioned RL or Meta-RL.
    """
    def __init__(self, env, config):
        super().__init__(env, config)
        
    def select_action(self, observation):
        return self.env.action_space.sample()

    def update(self, *args):
        pass

    def save(self, path):
        pass
    
    def load(self, path):
        pass
