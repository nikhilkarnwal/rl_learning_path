import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import gymnasium as gym
import torch
import numpy as np
import argparse
from src.utils.config import Config
from src.utils.logger import Logger
from src.agents.pg import PolicyGradientAgent
from src.trainers.pg_trainer import PolicyGradientTrainer

def train(config: Config):
    run_name = f"{config.env_id}__{config.exp_name}__{config.seed}"
    logger = Logger(config.log_dir, run_name)

    # Seeding
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # Create environment
    env = gym.make(config.env_id)

    # Initialize Agent
    agent = PolicyGradientAgent(env, config)
    print(f"Initialized PolicyGradientAgent for {config.env_id}")

    # Initialize Trainer
    trainer = PolicyGradientTrainer(env, agent, config, logger)
    
    # Train the agent
    trainer.train()
    
    env.close()
    logger.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str)
    parser.add_argument("--exp-name", type=str, default="test")
    parser.add_argument("--test-env", action="store_true", help="Just test env setup")
    args = parser.parse_args()

    if args.env_id is None:
        env_id = None
    else:
        env_id = args.env_id

    if args.test_env:
        print(f"Testing environment {env_id}...")
        env = gym.make(env_id, render_mode='human')
        env.reset()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print("Environment Step Successful!")
        print(f"Obs: {obs.shape}, Reward: {reward}")
        env.close()
    else:
        if env_id is None:
            config = Config(
                exp_name=args.exp_name
            )
        else:
            config = Config(
                env_id=env_id,
                exp_name=args.exp_name
            )
        train(config)
