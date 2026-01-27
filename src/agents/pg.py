from typing import List
from .base import BaseAgent
import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
from src.networks.policy_network import PolicyNetwork
from src.utils.experience import ExperienceBuffer

class PolicyGradientAgent(BaseAgent):
    def __init__(self, env, config):
        super().__init__(env, config)
        
        # Get environment dimensions
        if isinstance(env.observation_space, gym.spaces.Box):
            input_dim = np.prod(env.observation_space.shape)
        else:
            input_dim = env.observation_space.n
            
        if isinstance(env.action_space, gym.spaces.Discrete):
            output_dim = env.action_space.n
        else:
            raise ValueError("PolicyGradient only supports discrete action spaces")
        
        # Initialize Policy Network
        hidden_sizes = getattr(config, 'hidden_sizes', [128, 64])
        self.policy = PolicyNetwork(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_sizes=hidden_sizes,
            activation='relu'
        )
        
        # Optimizer
        learning_rate = getattr(config, 'learning_rate', 3e-4)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        
    def select_action(self, observation):
        """Select action using the policy network."""
        # Convert observation to tensor
        if not isinstance(observation, torch.Tensor):
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
        else:
            obs_tensor = observation.unsqueeze(0) if observation.dim() == 1 else observation
        
        action_probs = self.policy.get_log_probs(obs_tensor)
        action_dist = torch.distributions.Categorical(logits=action_probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        
        return action.item(), log_prob

    def update(self, rollouts: List[ExperienceBuffer]):
        # Code for training the policy network
        # Extract log_probs and rewards as lists, then convert to tensors
        loss_vals = []
        for rollout in rollouts:
            log_probs_list = [exp.log_prob for exp in rollout.experiences]
            rewards_list = [exp.reward for exp in rollout.experiences]
        
            # Convert directly to tensors (more efficient than torch.stack)
            log_probs = torch.stack(log_probs_list)
            rewards = torch.FloatTensor(rewards_list).sum()

            loss_val = -torch.sum(log_probs) * rewards
            loss_vals.append(loss_val)
        loss_val_tensor = torch.stack(loss_vals).mean()
        self.optimizer.zero_grad()
        loss_val_tensor.backward()
        self.optimizer.step()
        
        return loss_val_tensor.item()

    def update_v2(self, rollouts: List[ExperienceBuffer], gamma: float = 1.0, use_baseline: bool = True):
        """
        Update policy using correct REINFORCE algorithm with returns (G_t).
        
        Collects all log_probs and returns from all rollouts, normalizes globally
        for better variance reduction, then updates policy.
        
        Args:
            rollouts: List of experience buffers from episodes
            gamma: Discount factor (default=1.0 for undiscounted returns)
            use_baseline: Whether to normalize returns to reduce variance
        
        Returns:
            Loss value
        """
        loss_val = 0
        for epoch in range(self.config.num_epochs):
            # Collect all log_probs and returns from all rollouts
            all_log_probs = []
            all_returns = []
            baseline = 0.0
            
            for rollout in rollouts:
                log_probs_list = [exp.log_prob for exp in rollout.experiences]
                rewards_list = [exp.reward for exp in rollout.experiences]
            
                # Store log_probs
                all_log_probs.extend(log_probs_list)
                
                # Compute returns G_t = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + ...
                returns = []
                G = 0
                for r in reversed(rewards_list):
                    G = r + gamma * G
                    returns.insert(0, G)
                all_returns.extend(returns)
                baseline += sum(rewards_list)
            
            # Convert to tensors (flatten across all rollouts)
            log_probs = torch.stack(all_log_probs)
            returns = torch.FloatTensor(all_returns)
            
            # Normalize returns globally (baseline subtraction to reduce variance)
            if use_baseline and len(returns) > 1:
                baseline = returns.mean()
                returns = (returns - baseline)
            
            # REINFORCE loss: -sum(log_prob_t * G_t)
            # Each action's log_prob is weighted by its return
            # Negative because we want to maximize return (minimize negative return)
            loss = -torch.mean(log_probs * returns.detach())
            
            # Update policy
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            loss_val = loss.item()
            print(f"Epoch {epoch}: Loss = {loss_val}")
        return loss_val

    
    def save(self, path):
        pass
    
    def load(self, path):
        pass
