from .base import BaseAgent

class PPOAgent(BaseAgent):
    def __init__(self, env, config):
        super().__init__(env, config)
        # TODO: Define Actor and Critic Networks
        
    def select_action(self, observation):
        return self.env.action_space.sample()

    def update(self, rollouts):
        # TODO: Implement PPO clipping update
        pass

    def save(self, path):
        pass
    
    def load(self, path):
        pass
