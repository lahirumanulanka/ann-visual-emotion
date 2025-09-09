# Seed utilities for reproducible experiments
import os
import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    Set random seed for reproducible results.
    
    Args:
        seed (int): Random seed value
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Additional settings for reproducibility on CUDA
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"Random seed set to: {seed}")


def get_random_state():
    """
    Get current random state for all random number generators.
    
    Returns:
        dict: Dictionary containing random states
    """
    state = {
        'python_random': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state()
    }
    
    if torch.cuda.is_available():
        state['torch_cuda'] = torch.cuda.get_rng_state()
    
    return state


def set_random_state(state):
    """
    Set random state for all random number generators.
    
    Args:
        state (dict): Dictionary containing random states
    """
    random.setstate(state['python_random'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    
    if torch.cuda.is_available() and 'torch_cuda' in state:
        torch.cuda.set_rng_state(state['torch_cuda'])
