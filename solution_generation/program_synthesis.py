from typing import List, Dict, Any
import numpy as np

class Transformation:
    def __init__(self, name: str, func: callable):
        self.name = name
        self.func = func
        
    def apply(self, grid: np.ndarray) -> np.ndarray:
        return self.func(grid)

class ProgramSynthesizer:
    def __init__(self):
        self.transformations = self._init_transformations()
        self.max_program_length = 4
        
    def _init_transformations(self) -> Dict[str, Transformation]:
        transformations = {}
        
        # Basic transformations
        transformations['rotate_90'] = Transformation(
            'rotate_90',
            lambda x: np.rot90(x)
        )
        
        transformations['flip_horizontal'] = Transformation(
            'flip_horizontal',
            lambda x: np.fliplr(x)
        )
        
        transformations['flip_vertical'] = Transformation(
            'flip_vertical',
            lambda x: np.flipud(x)
        )
        
        transformations['shift_right'] = Transformation(
            'shift_right',
            lambda x: np.roll(x, 1, axis=1)
        )
        
        # Add more transformations here
        
        return transformations
    
    def synthesize_program(self, input_grids: List[np.ndarray], 
                          output_grids: List[np.ndarray]) -> List[Transformation]:
        best_program = []
        best_score = float('inf')
        
        # Simple beam search
        beam_width = 5
        beam = [[]]
        
        for _ in range(self.max_program_length):
            new_beam = []
            for program in beam:
                for trans in self.transformations.values():
                    new_program = program + [trans]
                    score = self._evaluate_program(new_program, input_grids, output_grids)
                    new_beam.append((new_program, score))
            
            new_beam.sort(key=lambda x: x[1])
            beam = [p for p, _ in new_beam[:beam_width]]
            
            if new_beam[0][1] < best_score:
                best_score = new_beam[0][1]
                best_program = new_beam[0][0]
                
            if best_score == 0:  # Perfect match
                break
                
        return best_program
    
    def _evaluate_program(self, program: List[Transformation], 
                        input_grids: List[np.ndarray],
                        output_grids: List[np.ndarray]) -> float:
        score = 0
        for input_grid, target_grid in zip(input_grids, output_grids):
            result = input_grid.copy()
            for trans in program:
                result = trans.apply(result)
            score += np.sum(result != target_grid)
        return score
