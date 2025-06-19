import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


# Define a color map for values 0-9 (ARC uses values between 0 and 9)
ARC_COLORS = [
    "#000000",  # 0 - black
    "#0074D9",  # 1 - blue
    "#2ECC40",  # 2 - green
    "#FF4136",  # 3 - red
    "#FFDC00",  # 4 - yellow
    "#AAAAAA",  # 5 - gray
    "#F012BE",  # 6 - magenta
    "#FF851B",  # 7 - orange
    "#870C25",  # 8 - dark red
    "#FFFFFF",  # 9 - white
]


class ARCTask:
    def __init__(self, task_id: str, train_pairs: List[Dict], test_pairs: List[Dict], solutions: Optional[List[List[List[int]]]] = None):
        self.task_id = task_id
        self.train_pairs = train_pairs
        self.test_pairs = test_pairs
        self.solutions = solutions or []

    def get_train_io(self) -> List[Tuple[List[List[int]], List[List[int]]]]:
        return [(pair['input'], pair['output']) for pair in self.train_pairs]

    def get_test_inputs(self) -> List[List[List[int]]]:
        return [pair['input'] for pair in self.test_pairs]

    def get_test_outputs(self) -> List[List[List[int]]]:
        return self.solutions

    def visualize_grid(self, grid: List[List[int]], title: str = ""):
        """Show a single grid."""
        arr = np.array(grid)
        cmap = plt.matplotlib.colors.ListedColormap(ARC_COLORS)
        plt.imshow(arr, cmap=cmap, vmin=0, vmax=9)
        plt.axis("off")
        if title:
            plt.title(title)
        plt.show()

    def visualize_train_pairs(self):
        for i, (inp, out) in enumerate(self.get_train_io()):
            fig, axs = plt.subplots(1, 2, figsize=(6, 3))
            for ax, grid, label in zip(axs, [inp, out], ["Input", "Output"]):
                ax.imshow(np.array(grid), cmap=plt.matplotlib.colors.ListedColormap(ARC_COLORS), vmin=0, vmax=9)
                ax.set_title(f"Train {i+1} - {label}")
                ax.axis("off")
            plt.tight_layout()
            plt.show()

    def visualize_test_cases(self):
        for i, test_input in enumerate(self.get_test_inputs()):
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.imshow(np.array(test_input), cmap=plt.matplotlib.colors.ListedColormap(ARC_COLORS), vmin=0, vmax=9)
            ax.set_title(f"Test {i+1} - Input")
            ax.axis("off")
            plt.show()

            if i < len(self.solutions):
                fig, ax = plt.subplots(figsize=(3, 3))
                ax.imshow(np.array(self.solutions[i]), cmap=plt.matplotlib.colors.ListedColormap(ARC_COLORS), vmin=0, vmax=9)
                ax.set_title(f"Test {i+1} - Expected Output")
                ax.axis("off")
                plt.show()


class ARCDataLoader:
    def __init__(self, challenges_path: str, solutions_path: Optional[str] = None):
        self.challenges_path = Path(challenges_path)
        self.solutions_path = Path(solutions_path) if solutions_path else None
        self.tasks: Dict[str, ARCTask] = {}

    def load(self):
        with open(self.challenges_path, 'r') as f:
            raw_challenges = json.load(f)

        raw_solutions = {}
        if self.solutions_path and self.solutions_path.exists():
            with open(self.solutions_path, 'r') as f:
                raw_solutions = json.load(f)

        for task_id, task_data in raw_challenges.items():
            train = task_data.get("train", [])
            test = task_data.get("test", [])
            solutions = raw_solutions.get(task_id, [])
            self.tasks[task_id] = ARCTask(task_id, train, test, solutions)

    def get_task(self, task_id: str) -> Optional[ARCTask]:
        return self.tasks.get(task_id)

    def get_all_tasks(self) -> List[ARCTask]:
        return list(self.tasks.values())
