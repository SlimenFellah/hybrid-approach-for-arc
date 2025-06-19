import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

#-------------------start-------------------

from typing import List, Tuple
import numpy as np
from dataloader.arc_data_loader import ARCTask

def compare_grids(pred: List[List[int]], target: List[List[int]]) -> bool:
    return np.array_equal(np.array(pred), np.array(target))


def evaluate_task(task: ARCTask, predictions: List[List[List[int]]]) -> Tuple[int, int]:
    expected_outputs = task.get_test_outputs()
    assert len(predictions) == len(expected_outputs), "Mismatch in number of predictions and targets."

    total = len(predictions)
    correct = 0

    for pred, target in zip(predictions, expected_outputs):
        if compare_grids(pred, target):
            correct += 1

    return correct, total


def print_evaluation_summary(task_id: str, correct: int, total: int):
    print(f"Task {task_id} — Accuracy: {correct}/{total} ({100.0 * correct / total:.1f}%)")
