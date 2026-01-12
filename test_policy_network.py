"""
Test script to demonstrate the PolicyNetwork usage
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import torch
from src.networks import PolicyNetwork

def test_policy_network():
    print("=" * 50)
    print("Testing PolicyNetwork")
    print("=" * 50)
    
    # Example: CartPole-v1 has 4 observations, 2 actions
    input_dim = 4
    output_dim = 2
    
    # Create network with default hidden sizes [128, 64]
    policy = PolicyNetwork(input_dim, output_dim)
    
    print(f"\nNetwork Architecture:")
    print(policy)
    
    # Test with a single observation
    print(f"\n{'='*50}")
    print("Test 1: Single observation")
    print("="*50)
    obs = torch.randn(input_dim)
    print(f"Input observation shape: {obs.shape}")
    
    action_probs = policy(obs)
    print(f"Output action probabilities: {action_probs}")
    print(f"Sum of probabilities: {action_probs.sum():.6f}")
    
    # Test with batch of observations
    print(f"\n{'='*50}")
    print("Test 2: Batch of observations")
    print("="*50)
    batch_size = 5
    obs_batch = torch.randn(batch_size, input_dim)
    print(f"Input batch shape: {obs_batch.shape}")
    
    action_probs_batch = policy(obs_batch)
    print(f"Output batch shape: {action_probs_batch.shape}")
    print(f"Action probabilities for batch:")
    print(action_probs_batch)
    print(f"Sum of probabilities per sample: {action_probs_batch.sum(dim=1)}")
    
    # Test log probabilities
    print(f"\n{'='*50}")
    print("Test 3: Log probabilities")
    print("="*50)
    log_probs = policy.get_log_probs(obs)
    print(f"Log probabilities: {log_probs}")
    
    # Test custom architecture
    print(f"\n{'='*50}")
    print("Test 4: Custom architecture")
    print("="*50)
    custom_policy = PolicyNetwork(
        input_dim=8,
        output_dim=4,
        hidden_sizes=[256, 128, 64],
        activation='tanh'
    )
    print(custom_policy)
    
    print(f"\n{'='*50}")
    print("All tests passed! ✓")
    print("="*50)

if __name__ == "__main__":
    test_policy_network()
