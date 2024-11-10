from typing import List, Dict, Any
import numpy as np

class HeuristicEngine:
    def __init__(self):
        self.heuristics = [
            self._check_grid_growth,
            self._check_pattern_repeat,
            self._check_object_movement,
            self._check_value_progression
        ]
    
    def analyze(self, input_grid: np.ndarray, output_grid: np.ndarray) -> List[Dict[str, Any]]:
        results = []
        for heuristic in self.heuristics:
            if result := heuristic(input_grid, output_grid):
                results.appen