import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNetwork(nn.Module):
    """
    Modular Policy Network for Policy Gradient methods.
    
    Takes observations as input and outputs action probabilities via softmax.
    Can be configured with custom hidden layer sizes.
    
    Args:
        input_dim (int): Dimension of the observation space
        output_dim (int): Dimension of the action space (number of discrete actions)
        hidden_sizes (list[int], optional): List of hidden layer sizes. 
                                           Defaults to [128, 64]
        activation (str, optional): Activation function ('relu', 'tanh'). 
                                   Defaults to 'relu'
    """
    
    def __init__(self, input_dim: int, output_dim: int, 
                 hidden_sizes: list = None, activation: str = 'relu'):
        super(PolicyNetwork, self).__init__()
        
        if hidden_sizes is None:
            hidden_sizes = [128, 64]
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Select activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Build network layers
        layers = []
        prev_size = input_dim
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(self.activation)
            prev_size = hidden_size
        
        # Output layer (no activation, softmax applied in forward)
        layers.append(nn.Linear(prev_size, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input observation tensor of shape (batch_size, input_dim)
                             or (input_dim,) for single observation
        
        Returns:
            torch.Tensor: Action probabilities after softmax of shape 
                         (batch_size, output_dim) or (output_dim,)
        """
        logits = self.network(x)
        action_probs = F.softmax(logits, dim=-1)
        return action_probs
    
    def get_action_logits(self, x):
        """
        Get raw logits before softmax (useful for some algorithms).
        
        Args:
            x (torch.Tensor): Input observation tensor
        
        Returns:
            torch.Tensor: Raw logits from the network
        """
        return self.network(x)
    
    def get_log_probs(self, x):
        """
        Get log probabilities (more numerically stable than log(softmax())).
        
        Args:
            x (torch.Tensor): Input observation tensor
        
        Returns:
            torch.Tensor: Log probabilities of actions
        """
        logits = self.network(x)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs
