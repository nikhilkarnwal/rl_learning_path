"""Experience storage for reinforcement learning."""

import torch
from typing import Any, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class Experience:
    """
    Stores a single experience tuple (state, action, reward).
    
    This is commonly used in reinforcement learning to store transitions
    observed during environment interaction.
    
    Attributes:
        state: The state observation (can be numpy array, list, or any state representation)
        action: The action taken in this state
        reward: The reward received after taking the action
        next_state: Optional next state (useful for temporal difference learning)
        done: Optional flag indicating if episode terminated after this step
    """
    state: Any
    action: Any
    log_prob: torch.Tensor
    reward: float
    next_state: Optional[Any] = None
    done: Optional[bool] = None
    
    def __repr__(self) -> str:
        """String representation of the experience."""
        base = f"Experience(s={self._format_value(self.state)}, a={self.action}, r={self.reward}"
        if self.next_state is not None:
            base += f", s'={self._format_value(self.next_state)}"
        if self.done is not None:
            base += f", done={self.done}"
        return base + ")"
    
    def _format_value(self, value: Any) -> str:
        """Format value for display, truncating numpy arrays."""
        if isinstance(value, np.ndarray):
            if value.size > 4:
                return f"array(shape={value.shape})"
            return f"array({value.tolist()})"
        return str(value)


class ExperienceBuffer:
    """
    A simple buffer to store multiple experiences.
    
    Useful for collecting experiences during an episode or for batch processing.
    """
    
    def __init__(self):
        """Initialize an empty experience buffer."""
        self.experiences: list[Experience] = []
    
    def add(self, state: Any, action: Any, log_prob: float, reward: float, 
            next_state: Optional[Any] = None, done: Optional[bool] = None) -> None:
        """
        Add a new experience to the buffer.
        
        Args:
            state: The state observation
            action: The action taken
            log_prob: The log probability of the action
            reward: The reward received
            next_state: Optional next state
            done: Optional termination flag
        """
        exp = Experience(state, action, log_prob, reward, next_state, done)
        self.experiences.append(exp)
    
    def clear(self) -> None:
        """Clear all experiences from the buffer."""
        self.experiences.clear()
    
    def __len__(self) -> int:
        """Return the number of experiences in the buffer."""
        return len(self.experiences)
    
    def __getitem__(self, idx: int) -> Experience:
        """Get experience at index."""
        return self.experiences[idx]
    
    def __iter__(self):
        """Iterate over experiences."""
        return iter(self.experiences)
    
    def get_all(self) -> list[Experience]:
        """Get all experiences as a list."""
        return self.experiences
