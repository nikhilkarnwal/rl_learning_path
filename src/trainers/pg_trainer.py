"""Policy Gradient Trainer for REINFORCE algorithm."""

import gymnasium as gym
import numpy as np
import torch
from typing import Optional
from src.agents.pg import PolicyGradientAgent
from src.utils.experience import ExperienceBuffer
from src.utils.logger import Logger
from src.utils.config import Config


class PolicyGradientTrainer:
    """
    Trainer for Policy Gradient (REINFORCE) algorithm.
    
    Manages the training loop, episode rollouts, experience collection,
    and policy updates.
    """
    
    def __init__(
        self,
        env: gym.Env,
        agent: PolicyGradientAgent,
        config: Config,
        logger: Optional[Logger] = None
    ):
        """
        Initialize the trainer.
        
        Args:
            env: Gymnasium environment
            agent: Policy Gradient agent
            config: Configuration object
            logger: Optional logger for tracking metrics
        """
        self.env = env
        self.agent = agent
        self.config = config
        self.logger = logger
        
        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []
        self.trajs = []
        
    def run_episode(self) -> tuple[float, int, ExperienceBuffer]:
        """
        Run a single episode and collect experiences.
        
        Returns:
            tuple: (total_reward, episode_length, experience_buffer)
        """
        buffer = ExperienceBuffer()
        obs, _ = self.env.reset()
        done = False
        total_reward = 0.0
        episode_length = 0
        
        max_steps = getattr(self.config, 'max_episode_steps', 1000)
        
        while not done and episode_length < max_steps:
            # Select action using the policy
            action, log_prob = self.agent.select_action(obs)
            
            # Take action in environment
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Store experience
            buffer.add(
                state=obs,
                action=action,
                log_prob=log_prob,
                reward=reward,
                next_state=next_obs,
                done=done
            )
            
            # Update for next iteration
            obs = next_obs
            total_reward += reward
            episode_length += 1
        
        return total_reward, episode_length, buffer
    
    def train(self, num_episodes: Optional[int] = None):
        """
        Main training loop.
        
        Args:
            num_episodes: Number of episodes to train for.
                         If None, uses config.num_episodes
        """
        if num_episodes is None:
            num_episodes = getattr(self.config, 'num_episodes', 1000)
        
        log_interval = getattr(self.config, 'log_interval', 10)
        
        print(f"Starting training for {num_episodes} episodes...")
        
        # fetch 1000 samples then update policy, apply while loop
        for episode in range(num_episodes):
            rollouts = []
            total_episode_length = 0
            total_episode_reward = 0
            while total_episode_length < getattr(self.config, 'batch_size', 1000):
                episode_reward, episode_length, buffer = self.run_episode()
                total_episode_length += episode_length
                total_episode_reward += episode_reward
                rollouts.append(buffer)
            
            # Update policy using collected experiences
            # loss = self.agent.update(rollouts)

            # run update_v2
            loss = self.agent.update_v2(rollouts, gamma=0.99, use_baseline=True)
            
            # Track metrics
            self.episode_rewards.append(total_episode_reward / len(rollouts))
            self.episode_lengths.append(total_episode_length / len(rollouts))
            self.losses.append(loss)

            # Log metrics
            if episode % log_interval == 0:
                self.log_metrics(episode, num_episodes)
        
        print("\nTraining completed!")
        self.print_final_summary()
    
    def log_metrics(self, episode: int, total_episodes: int):
        """
        Log training metrics.
        
        Args:
            episode: Current episode number
            total_episodes: Total number of episodes
        """
        log_interval = getattr(self.config, 'log_interval', 10)
        recent_rewards = self.episode_rewards[-log_interval:]
        recent_lengths = self.episode_lengths[-log_interval:]
        recent_losses = self.losses[-log_interval:]
        
        avg_reward = np.mean(recent_rewards)
        avg_length = np.mean(recent_lengths)
        avg_loss = np.mean(recent_losses)
        
        print(f"Episode {episode}/{total_episodes} | "
              f"Avg Reward: {avg_reward:.2f} | "
              f"Avg Length: {avg_length:.1f} | "
              f"Avg Loss: {avg_loss:.4f}")
        
        # Log to logger if available
        if self.logger:
            self.logger.log_scalar('episode', episode, episode)
            self.logger.log_scalar('avg_reward', avg_reward, episode)
            self.logger.log_scalar('avg_length', avg_length, episode)
            self.logger.log_scalar('avg_loss', avg_loss, episode)
    
    def print_final_summary(self):
        """Print final training summary."""
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        print(f"Total Episodes: {len(self.episode_rewards)}")
        print(f"Average Reward (last 100): {np.mean(self.episode_rewards[-100:]):.2f}")
        print(f"Average Length (last 100): {np.mean(self.episode_lengths[-100:]):.1f}")
        print(f"Best Episode Reward: {max(self.episode_rewards):.2f}")
        print(f"Final Loss: {self.losses[-1]:.4f}")
        print("="*60 + "\n")
