from typing import List, Dict, Any
import numpy as np

class RuleBasedPatternRecognizer:
    def __init__(self):
        self.rules = [
            self._check_copy_rule,
            self._check_rotation_rule,
            self._check_mirror_rule,
            self._check_color_change_rule,
            self._check_shape_change_rule
        ]
    
    def analyze_pattern(self, input_grid: np.ndarray, output_grid: np.ndarray) -> List[Dict[str, Any]]:
        matched_rules = []
        
        for rule_checker in self.rules:
            if result := rule_checker(input_grid, output_grid):
                matched_rules.append(result)
        
        return matched_rules
    
    def _check_copy_rule(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Dict[str, Any]:
        if np.array_equal(input_grid, output_grid):
            return {'rule': 'copy', 'confidence': 1.0}
        return None
    
    def _check_rotation_rule(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Dict[str, Any]:
        for k in range(1, 4):
            if np.array_equal(np.rot90(input_grid, k), output_grid):
                return {'rule': 'rotation', 'params': {'angle': 90 * k}, 'confidence': 1.0}
        return None
    
    def _check_mirror_rule(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Dict[str, Any]:
        if np.array_equal(np.flip(input_grid, 0), output_grid):
            return {'rule': 'mirror', 'params': {'axis': 'horizontal'}, 'confidence': 1.0}
        if np.array_equal(np.flip(input_grid, 1), output_grid):
            return {'rule': 'mirror', 'params': {'axis': 'vertical'}, 'confidence': 1.0}
        return None
    
    def _check_color_change_rule(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Dict[str, Any]:
        if input_grid.shape != output_grid.shape:
            return None
            
        unique_input = np.unique(input_grid)
        unique_output = np.unique(output_grid)
        
        if len(unique_input) == len(unique_output):
            mapping = {}
            for i, j in zip(unique_input, unique_output):
                mapping[i] = j
                
            mapped_input = np.vectorize(lambda x: mapping.get(x, x))(input_grid)
            if np.array_equal(mapped_input, output_grid):
                return {
                    'rule': 'color_change',
                    'params': {'mapping': mapping},
                    'confidence': 1.0
                }
        return None
    
    def _check_shape_change_rule(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Dict[str, Any]:
        input_objects = self._get_objects(input_grid)
        output_objects = self._get_objects(output_grid)
        
        if len(input_objects) == len(output_objects):
            shape_changes = []
            for i_obj, o_obj in zip(input_objects, output_objects):
                if i_obj['value'] == o_obj['value'] and i_obj['size'] != o_obj['size']:
                    shape_changes.append({
                        'original_size': i_obj['size'],
                        'new_size': o_obj['size'],
                        'value': i_obj['value']
                    })
                    
            if shape_changes:
                return {
                    'rule': 'shape_change',
                    'params': {'changes': shape_changes},
                    'confidence': 0.8
                }
        return None
    
    def _get_objects(self, grid: np.ndarray) -> List[Dict[str, Any]]:
        from scipy import ndimage
        labeled_array, num_features = ndimage.label(grid > 0)
        objects = []
        
        for i in range(1, num_features + 1):
            mask = labeled_array == i
            objects.append({
                'value': grid[mask][0],
                'size': np.sum(mask),
                'mask': mask
            })
            
        return objects