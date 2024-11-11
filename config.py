import os
from pathlib import Path
import torch 

class Config:
    # Paths
    ROOT_DIR = Path(__file__).parent.parent
    DATA_DIR = ROOT_DIR / "data"
    TRAINING_DIR = DATA_DIR / "training"
    EVALUATION_DIR = DATA_DIR / "evaluation"
    TEST_DIR = DATA_DIR / "test"
    
    # Model parameters
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    
    # Feature extraction parameters
    MAX_GRID_SIZE = 30
    MIN_GRID_SIZE = 1
    NUM_COLORS = 10  # 0-9
    
    # Meta-learning parameters
    META_LEARNING_RATE = 0.0001
    META_BATCH_SIZE = 16
    
    # Visualization parameters
    COLOR_MAP = {
        0: '#000000',  # Black
        1: '#0074D9',  # Blue
        2: '#FF4136',  # Red
        3: '#2ECC40',  # Green
        4: '#FFDC00',  # Yellow
        5: '#B10DC9',  # Purple
        6: '#FF851B',  # Orange
        7: '#7FDBFF',  # Light Blue
        8: '#01FF70',  # Light Green
        9: '#F012BE'   # Pink
    }