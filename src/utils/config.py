import dataclasses
from typing import Optional

@dataclasses.dataclass
class Config:
    env_id: str = "CartPole-v1"
    seed: int = 42
    total_timesteps: int = 100000
    learning_rate: float = 3e-4
    gamma: float = 0.99
    device: str = "cpu"
    log_dir: str = "runs"
    exp_name: str = "experiment"
    track: bool = False # Track with wandb or similar if needed

    # Trainer specific params
    num_episodes: int = 1000  # Number of episodes to train
    max_episode_steps: int = 1000  # Max steps per episode
    log_interval: int = 10  # Log metrics every N episodes
    
    # Agent specific params can be added here or in subclasses
