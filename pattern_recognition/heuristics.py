# from typing import List, Dict, Any
# import numpy as np

# class HeuristicEngine:
#     def __init__(self):
#         self.heuristics = [
#             self._check_grid_growth,
#             self._check_pattern_repeat,
#             self._check_object_movement,
#             self._check_value_progression
#         ]
    
#     def analyze(self, input_grid: np.ndarray, output_grid: np.ndarray) -> List[Dict[str, Any]]:
#         results = []
#         for heuristic in self.heuristics:
#             if result := heuristic(input_grid, output_grid):
#                 results.appen



from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from dataclasses import dataclass
from enum import Enum
import matplotlib.pyplot as plt
from scipy import ndimage

class PatternType(Enum):
    GRID_GROWTH = "grid_growth"
    PATTERN_REPEAT = "pattern_repeat"
    OBJECT_MOVEMENT = "object_movement"
    VALUE_PROGRESSION = "value_progression"
    COLOR_TRANSFORM = "color_transform"
    SYMMETRY = "symmetry"
    ROTATION = "rotation"
    SHAPE_CHANGE = "shape_change"

@dataclass
class PatternInfo:
    type: PatternType
    confidence: float
    params: Dict[str, Any]
    visualization: Optional[np.ndarray] = None

class HeuristicEngine:
    def __init__(self, visualization_enabled: bool = True):
        self.visualization_enabled = visualization_enabled
        self.heuristics = [
            self._check_grid_growth,
            self._check_pattern_repeat,
            self._check_object_movement,
            self._check_value_progression,
            self._check_color_transform,
            self._check_symmetry,
            self._check_rotation,
            self._check_shape_change
        ]
        
    def analyze(self, input_grid: np.ndarray, output_grid: np.ndarray) -> List[PatternInfo]:
        """Analyze input and output grids to detect patterns."""
        results = []
        for heuristic in self.heuristics:
            if result := heuristic(input_grid, output_grid):
                results.append(result)
                if self.visualization_enabled:
                    self._visualize_pattern(result, input_grid, output_grid)
        return results

    def _check_grid_growth(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Optional[PatternInfo]:
        """Check if output grid is a growth/shrinkage of input grid."""
        in_shape = input_grid.shape
        out_shape = output_grid.shape
        
        if in_shape != out_shape:
            growth_factor = (out_shape[0] / in_shape[0], out_shape[1] / in_shape[1])
            if abs(growth_factor[0] - growth_factor[1]) < 0.1:  # Uniform growth
                return PatternInfo(
                    type=PatternType.GRID_GROWTH,
                    confidence=0.9,
                    params={"growth_factor": growth_factor}
                )
        return None

    def _check_pattern_repeat(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Optional[PatternInfo]:
        """Detect repeating patterns in the transformation."""
        # Find unique values and their positions
        unique_vals = np.unique(input_grid[input_grid != 0])
        
        for val in unique_vals:
            in_positions = np.where(input_grid == val)
            out_positions = np.where(output_grid == val)
            
            if len(in_positions[0]) > 0 and len(out_positions[0]) > 0:
                # Check for regular spacing
                in_spacing = np.diff(in_positions[0])
                out_spacing = np.diff(out_positions[0])
                
                if np.all(out_spacing == out_spacing[0]) and not np.all(in_spacing == in_spacing[0]):
                    return PatternInfo(
                        type=PatternType.PATTERN_REPEAT,
                        confidence=0.8,
                        params={"value": val, "spacing": out_spacing[0]}
                    )
        return None

    def _check_object_movement(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Optional[PatternInfo]:
        """Detect object movement patterns."""
        movements = {}
        
        for val in np.unique(input_grid):
            if val == 0:  # Skip background
                continue
                
            in_center = ndimage.center_of_mass(input_grid == val)
            out_center = ndimage.center_of_mass(output_grid == val)
            
            if not (np.isnan(in_center).any() or np.isnan(out_center).any()):
                movement = (out_center[0] - in_center[0], out_center[1] - in_center[1])
                movements[val] = movement

        if movements:
            return PatternInfo(
                type=PatternType.OBJECT_MOVEMENT,
                confidence=0.7,
                params={"movements": movements}
            )
        return None

    def _check_value_progression(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Optional[PatternInfo]:
        """Check for numerical progression patterns in values."""
        in_vals = np.unique(input_grid[input_grid != 0])
        out_vals = np.unique(output_grid[output_grid != 0])
        
        if len(in_vals) > 1 and len(out_vals) > 1:
            # Check arithmetic progression
            in_diff = np.diff(in_vals)
            out_diff = np.diff(out_vals)
            
            if np.all(out_diff == out_diff[0]):
                return PatternInfo(
                    type=PatternType.VALUE_PROGRESSION,
                    confidence=0.6,
                    params={"progression_type": "arithmetic", "diff": out_diff[0]}
                )
        return None

    def _check_color_transform(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Optional[PatternInfo]:
        """Detect color transformation patterns."""
        color_map = {}
        
        for i in range(input_grid.shape[0]):
            for j in range(input_grid.shape[1]):
                if input_grid[i,j] != 0:
                    if input_grid[i,j] not in color_map:
                        color_map[input_grid[i,j]] = output_grid[i,j]
                    elif color_map[input_grid[i,j]] != output_grid[i,j]:
                        return None
        
        if color_map:
            return PatternInfo(
                type=PatternType.COLOR_TRANSFORM,
                confidence=0.9,
                params={"color_map": color_map}
            )
        return None

    def _check_symmetry(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Optional[PatternInfo]:
        """Check for symmetry patterns in the transformation."""
        # Check horizontal symmetry
        h_sym_in = np.all(input_grid == np.fliplr(input_grid))
        h_sym_out = np.all(output_grid == np.fliplr(output_grid))
        
        # Check vertical symmetry
        v_sym_in = np.all(input_grid == np.flipud(input_grid))
        v_sym_out = np.all(output_grid == np.flipud(output_grid))
        
        if (not h_sym_in and h_sym_out) or (not v_sym_in and v_sym_out):
            return PatternInfo(
                type=PatternType.SYMMETRY,
                confidence=0.8,
                params={
                    "horizontal": h_sym_out,
                    "vertical": v_sym_out
                }
            )
        return None

    def _check_rotation(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Optional[PatternInfo]:
        """Detect rotation patterns."""
        for k in range(1, 4):  # Check 90, 180, 270 degrees
            rotated = np.rot90(input_grid, k)
            if np.array_equal(rotated, output_grid):
                return PatternInfo(
                    type=PatternType.ROTATION,
                    confidence=1.0,
                    params={"angle": 90 * k}
                )
        return None

    def _check_shape_change(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Optional[PatternInfo]:
        """Detect shape transformation patterns."""
        in_shapes = []
        out_shapes = []
        
        # Extract connected components
        for val in np.unique(input_grid)[1:]:  # Skip background
            in_mask = input_grid == val
            out_mask = output_grid == val
            
            if np.any(in_mask) and np.any(out_mask):
                in_shapes.append(in_mask)
                out_shapes.append(out_mask)
        
        if in_shapes and out_shapes:
            # Compare basic shape properties
            area_ratios = []
            for in_shape, out_shape in zip(in_shapes, out_shapes):
                in_area = np.sum(in_shape)
                out_area = np.sum(out_shape)
                if in_area > 0:
                    area_ratios.append(out_area / in_area)
            
            if len(set(area_ratios)) == 1:  # Consistent transformation
                return PatternInfo(
                    type=PatternType.SHAPE_CHANGE,
                    confidence=0.7,
                    params={"area_ratio": area_ratios[0]}
                )
        return None

    def _visualize_pattern(self, pattern: PatternInfo, input_grid: np.ndarray, output_grid: np.ndarray) -> None:
        """Visualize detected pattern for debugging and analysis."""
        plt.figure(figsize=(12, 4))
        
        # Input grid
        plt.subplot(131)
        plt.imshow(input_grid, cmap='tab10')
        plt.title('Input Grid')
        plt.colorbar()
        
        # Output grid
        plt.subplot(132)
        plt.imshow(output_grid, cmap='tab10')
        plt.title('Output Grid')
        plt.colorbar()
        
        # Pattern visualization
        plt.subplot(133)
        if pattern.type == PatternType.OBJECT_MOVEMENT:
            plt.imshow(input_grid, cmap='tab10', alpha=0.5)
            for val, (dy, dx) in pattern.params['movements'].items():
                y, x = ndimage.center_of_mass(input_grid == val)
                plt.arrow(x, y, dx, dy, color='red', head_width=0.3)
        elif pattern.type == PatternType.PATTERN_REPEAT:
            plt.imshow(output_grid, cmap='tab10')
            positions = np.where(output_grid == pattern.params['value'])
            plt.plot(positions[1], positions[0], 'rx')
        else:
            plt.imshow(output_grid, cmap='tab10')
            
        plt.title(f'{pattern.type.value}\nConfidence: {pattern.confidence:.2f}')
        plt.tight_layout()
        plt.show()

    def get_pattern_summary(self, patterns: List[PatternInfo]) -> str:
        """Generate a human-readable summary of detected patterns."""
        if not patterns:
            return "No patterns detected."
            
        summary = "Detected Patterns:\n"
        for pattern in patterns:
            summary += f"\n- {pattern.type.value} (confidence: {pattern.confidence:.2f})\n"
            for param, value in pattern.params.items():
                summary += f"  {param}: {value}\n"
        return summary