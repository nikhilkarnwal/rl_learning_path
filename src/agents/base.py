from abc import ABC, abstractmethod
import gymnasium as gym

class BaseAgent(ABC):
    def __init__(self, env: gym.Env, config):
        self.env = env
        self.config = config

    @abstractmethod
    def select_action(self, observation):
        pass

    @abstractmethod
    def update(self, *args, **kwargs):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def load(self, path):
        pass
