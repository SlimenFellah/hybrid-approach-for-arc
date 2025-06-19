import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple


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
