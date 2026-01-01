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
# Import agents here as implemented
# from src.agents.pg import PolicyGradientAgent

def train(config: Config):
    run_name = f"{config.env_id}__{config.exp_name}__{config.seed}"
    logger = Logger(config.log_dir, run_name)

    # Seeding
    # random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    env = gym.make(config.env_id)
    # env = gym.wrappers.RecordVideo(env, f"videos/{run_name}") # Optional

    # Initialize Agent (TODO: Factory or selection logic)
    # agent = PolicyGradientAgent(env, config)
    print("Agent not initialized yet.")
    agent = None 

    if agent is None:
        print("No agent selected. Exiting.")
        return

    # Training Loop (Basic Skeleton - specific agents might override or use a runner)
    # global_step = 0
    # while global_step < config.total_timesteps:
    #     obs, _ = env.reset()
    #     done = False
    #     while not done:
    #         action = agent.select_action(obs)
    #         next_obs, reward, terminated, truncated, info = env.step(action)
    #         done = terminated or truncated
    #         # agent.store(...)
    #         # agent.update(...)
    #         obs = next_obs
    #         global_step += 1
    
    env.close()
    logger.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default="CartPole-v1")
    parser.add_argument("--exp-name", type=str, default="test")
    parser.add_argument("--test-env", action="store_true", help="Just test env setup")
    args = parser.parse_args()

    if args.test_env:
        print(f"Testing environment {args.env_id}...")
        env = gym.make(args.env_id)
        env.reset()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print("Environment Step Successful!")
        print(f"Obs: {obs.shape}, Reward: {reward}")
        env.close()
    else:
        config = Config(
            env_id=args.env_id,
            exp_name=args.exp_name
        )
        train(config)
