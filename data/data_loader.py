# import json
# import numpy as np
# from pathlib import Path

# class ARCDataLoader:
#     def __init__(self, data_dir):
#         self.data_dir = Path(data_dir)
        
#     def load_task(self, task_file):
#         """Load a single task from JSON file."""
#         with open(self.data_dir / task_file, 'r') as f:
#             data = json.load(f)
#         return data
    
#     def preprocess_grid(self, grid):
#         """Convert grid to numpy array and normalize."""
#         return np.array(grid) / 9.0  # Normalize to [0, 1]
    
#     def load_dataset(self, split='training'):
#         """Load all tasks from a specific split."""
#         tasks = []
#         split_dir = self.data_dir / split
#         for task_file in split_dir.glob('*.json'):
#             task_data = self.load_task(task_file)
#             tasks.append({
#                 'task_id': task_file.stem,
#                 'data': task_data
#             })
#         return tasks


# import numpy as np

# class DataProcessor:
#     def preprocess_grid(self, grid):
#         # Normalize grid values, handle padding if needed, etc.
#         return np.array(grid)

#     def extract_features(self, grid):
#         # Extract color patterns, object boundaries, symmetries, etc.
#         features = {}
#         # Implement feature extraction logic here
#         return features

# import json
# import os

# class DataLoader:
#     def __init__(self, base_dir):
#         self.base_dir = base_dir

#     def load_data(self, subset='training'):
#         """Load JSON files from the specified subset (training/evaluation/test)."""
#         data_path = os.path.join(self.base_dir, subset)
#         tasks = {}
#         for filename in os.listdir(data_path):
#             if filename.endswith('.json'):
#                 with open(os.path.join(data_path, filename), 'r') as f:
#                     tasks[filename] = json.load(f)
#         return tasks

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from config import Config

class ARCDataLoader:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir

    def load_task(self, task_file: Path) -> Dict[str, Any]:
        with open(task_file, 'r') as f:
            return json.load(f)
    
    def load_all_tasks(self) -> List[Dict[str, Any]]:
        tasks = []
        for task_file in self.data_dir.glob('*.json'):
            tasks.append(self.load_task(task_file))
        return tasks
    
    def get_train_test_pairs(self, task: Dict[str, Any]) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[Tuple[np.ndarray, np.ndarray]]]:
        train_pairs = [(np.array(pair['input']), np.array(pair['output'])) 
                    for pair in task['train']]
        
        test_pairs = [(np.array(pair['input']), np.array(pair['output']) 
                    if 'output' in pair else None) 
                    for pair in task['test']]
        
        return train_pairs, test_pairs
