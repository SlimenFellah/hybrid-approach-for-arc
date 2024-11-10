# # helpers.py
# import numpy as np
# from typing import List, Tuple, Dict, Any
# from src.config import Config

# class GridHelpers:
#     @staticmethod
#     def get_grid_properties(grid: np.ndarray) -> Dict[str, Any]:
#         """Extract basic properties of a grid"""
#         return {
#             'height': grid.shape[0],
#             'width': grid.shape[1],
#             'unique_colors': np.unique(grid).tolist(),
#             'color_counts': {int(k): int(v) for k, v in zip(*np.unique(grid, return_counts=True))},
#             'symmetrical': GridHelpers.is_symmetrical(grid),
#             'has_patterns': GridHelpers.detect_basic_patterns(grid)
#         }
    
#     @staticmethod
#     def is_symmetrical(grid: np.ndarray) -> Dict[str, bool]:
#         """Check different types of symmetry"""
#         h, w = grid.shape
        
#         # Horizontal symmetry
#         h_sym = np.all(grid == np.flipud(grid))
        
#         # Vertical symmetry
#         v_sym = np.all(grid == np.fliplr(grid))
        
#         # Diagonal symmetry (if square)
#         d_sym = False
#         if h == w:
#             d_sym = np.all(grid == grid.T)
        
#         return {
#             'horizontal': h_sym,
#             'vertical': v_sym,
#             'diagonal': d_sym
#         }
    
#     @staticmethod
#     def detect_basic_patterns(grid: np.ndarray) -> Dict[str, bool]:
#         """Detect basic patterns in the grid"""
#         patterns = {
#             'has_repetition': False,
#             'has_progression': False,
#             'has_alternation': False
#         }
        
#         # Check for repetition (same pattern repeating)
#         for i in range(1, grid.shape[0]//2 + 1):
#             if np.all(grid[i:] == grid[:-i]):
#                 patterns['has_repetition'] = True
#                 break
        
#         # Check for progression (increasing/decreasing values)
#         diffs = np.diff(grid, axis=0)
#         if np.all(diffs == diffs[0]):
#             patterns['has_progression'] = True
        
#         # Check for alternation (alternating values)
#         if np.all(grid[::2] == grid[0]) and np.all(grid[1::2] == grid[1]):
#             patterns['has_alternation'] = True
        
#         return patterns
    
#     @staticmethod
#     def find_objects(grid: np.ndarray) -> List[Dict[str, Any]]:
#         """Find distinct objects in the grid"""
#         objects = []
#         visited = np.zeros_like(grid, dtype=bool)
        
#         def flood_fill(x: int, y: int, color: int) -> List[Tuple[int, int]]:
#             if (x < 0 or x >= grid.shape[0] or 
#                 y < 0 or y >= grid.shape[1] or
#                 visited[x,y] or grid[x,y] != color):
#                 return []
            
#             visited[x,y] = True
#             pixels = [(x,y)]
            
#             for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]:
#                 pixels.extend(flood_fill(x+dx, y+dy, color))
            
#             return pixels
        
#         for i in range(grid.shape[0]):
#             for j in range(grid.shape[1]):
#                 if not visited[i,j] and grid[i,j] != 0:
#                     pixels = flood_fill(i, j, grid[i,j])
#                     if pixels:
#                         xs, ys = zip(*pixels)
#                         objects.append({
#                             'color': int(grid[i,j]),
#                             'size': len(pixels),
#                             'bbox': [min(xs), min(ys), max(xs), max(ys)],
#                             'pixels': pixels
#                         })
        
#         return objects
    
#     @staticmethod
#     def extract_subgrid(grid: np.ndarray, bbox: List[int]) -> np.ndarray:
#         """Extract a subgrid given a bounding box"""
#         x1, y1, x2, y2 = bbox
#         return grid[x1:x2+1, y1:y2+1]
    
#     @staticmethod
#     def apply_transformation(grid: np.ndarray, 
#                            transformation: str, 
#                            params: Dict = None) -> np.ndarray:
#         """Apply a basic transformation to the grid"""
#         if params is None:
#             params = {}
            
#         if transformation == 'rotate':
#             return np.rot90(grid, k=params.get('k', 1))
#         elif transformation == 'flip':
#             if params.get('axis') == 0:
#                 return np.flipud(grid)
#             else:
#                 return np.fliplr(grid)
#         elif transformation == 'shift':
#             return np.roll(grid, shift=params.get('shift', 1), 
#                          axis=params.get('axis', 0))
#         elif transformation == 'mask':
#             mask = params.get('mask', grid != 0)
#             result = grid.copy()
#             result[~mask] = 0
#             return result
#         else:
#             return grid

# class ValidationHelpers:
#     @staticmethod
#     def validate_grid_values(grid: np.ndarray) -> bool:
#         """Validate that grid contains only allowed values"""
#         return np.all((grid >= 0) & (grid <= 9))








# helpers.py
# import numpy as np
# from typing import Dict, List, Tuple, Any
# from scipy import ndimage
# from sklearn.metrics import pairwise_distances

# def analyze_grid_patterns(grid: np.ndarray) -> Dict[str, Any]:
#     """
#     Analyze various patterns in the grid.
#     """
#     patterns = {}
    
#     # Detect repeating patterns
#     patterns['repeating'] = detect_repeating_patterns(grid)
    
#     # Detect geometric shapes
#     patterns['shapes'] = detect_shapes(grid)
    
#     # Detect color patterns
#     patterns['color_patterns'] = detect_color_patterns(grid)
    
#     # Detect spatial relationships
#     patterns['spatial'] = detect_spatial_patterns(grid)
    
#     return patterns

# def detect_repeating_patterns(grid: np.ndarray) -> Dict[str, Any]:
#     """
#     Detect repeating patterns in the grid.
#     """
#     patterns = {}
    
#     # Find horizontal repetitions
#     for i in range(1, grid.shape[1]):
#         if np.array_equal(grid[:, :i], grid[:, i:2*i]):
#             patterns['horizontal'] = {'period': i}
    
#     # Find vertical repetitions
#     for i in range(1, grid.shape[0]):
#         if np.array_equal(grid[:i, :], grid[i:2*i, :]):
#             patterns['vertical'] = {'period': i}
    
#     # Find diagonal patterns
#     diag = np.diagonal(grid)
#     anti_diag = np.diagonal(np.fliplr(grid))
#     patterns['diagonal'] = {
#         'main_diagonal': is_pattern(diag),
#         'anti_diagonal': is_pattern(anti_diag)
#     }
    
#     return patterns

# def detect_shapes(grid: np.ndarray) -> Dict[str, Any]:
#     """
#     Detect basic geometric shapes in the grid.
#     """
#     shapes = {}
    
#     # Label connected components
#     labeled, num_features = ndimage.label(grid > 0)
    
#     for i in range(1, num_features + 1):
#         shape = grid[labeled == i]
#         bbox = ndimage.find_objects(labeled == i)[0]
        
#         # Calculate shape properties
#         area = np.sum(labeled == i)
#         perimeter = calculate_perimeter(labeled == i)
        
#         # Detect shape type
#         shape_type = classify_shape(shape, area, perimeter, bbox)
#         if shape_type:
#             shapes[f'shape_{i}'] = shape_type
    
#     return shapes

# def detect_color_patterns(grid: np.ndarray) -> Dict[str, Any]:
#     """
#     Detect patterns in color usage.
#     """
#     patterns = {}
    
#     # Analyze color frequencies
#     unique, counts = np.unique(grid, return_counts=True)
#     patterns['frequencies'] = dict(zip(unique, counts))
    
#     # Analyze color transitions
#     transitions = analyze_color_transitions(grid)
#     patterns['transitions'] = transitions
    
#     # Detect color gradients
#     patterns['gradients'] = detect_gradients(grid)
    
#     return patterns

# def detect_spatial_patterns(grid: np.ndarray) -> Dict[str, Any]:
#     """
#     Detect spatial relationships between elements.
#     """
#     patterns = {}
    
#     # Find alignments
#     patterns['alignments'] = detect_alignments(grid)
    
#     # Find clusters
#     patterns['clusters'] = detect_clusters(grid)
    
#     # Analyze density distribution
#     patterns['density'] = analyze_density(grid)
    
#     return patterns

# def detect_symmetry(grid: np.ndarray) -> Dict[str, bool]:
#     """
#     Detect various types of symmetry in the grid.
#     """
#     symmetry = {}
    
#     # Horizontal symmetry
#     symmetry['horizontal'] = np.array_equal(grid, np.flipud(grid))
    
#     # Vertical symmetry
#     symmetry['vertical'] = np.array_equal(grid, np.fliplr(grid))
    
#     # Diagonal symmetry
#     symmetry['diagonal'] = np.array_equal(grid, grid.T)
    
#     # Rotational symmetry
#     rot_90 = np.rot90(grid)
#     rot_180 = np.rot90(rot_90)
#     symmetry['rotational_90'] = np.array_equal(grid, rot_90)
#     symmetry['rotational_180'] = np.array_equal(grid, rot_180)
    
#     return symmetry

# def calculate_grid_statistics(input_grid: np.ndarray, output_grid: np.ndarray) -> Dict[str, Any]:
#     """
#     Calculate various statistical measures between input and output grids.
#     """
#     stats = {}
    
#     # Basic statistics
#     stats['input_mean'] = np.mean(input_grid)
#     stats['output_mean'] = np.mean(output_grid)
#     stats['input_std'] = np.std(input_grid)
#     stats['output_std'] = np.std(output_grid)
    
#     # Shape changes
#     stats['size_ratio'] = (output_grid.size / input_grid.size)
    
#     # Color statistics
#     stats['color_change_ratio'] = np.mean(input_grid != output_grid)
    
#     # Structural similarity
#     stats['structural_similarity'] = calculate_structural_similarity(input_grid, output_grid)
    
#     return stats

# def find_transformations(input_grid: np.ndarray, output_grid: np.ndarray) -> Dict[str, Any]:
#     """
#     Identify possible transformations between input and output grids.
#     """
#     transformations = {}
    
#     # Check for basic transformations
#     if np.array_equal(output_grid, np.rot90(input_grid)):
#         transformations['rotation'] = 90
#     elif np.array_equal(output_grid, np.rot90(input_grid, 2)):
#         transformations['rotation'] = 180
#     elif np.array_equal(output_grid, np.rot90(input_grid, 3)):
#         transformations['rotation'] = 270
    
#     # Check for flips
#     if np.array_equal(output_grid, np.flipud(input_grid)):
#         transformations['flip'] = 'vertical'
#     elif np.array_equal(output_grid, np.fliplr(input_grid)):
#         transformations['flip'] = 'horizontal'
    
#     # Check for color transformations
#     color_mapping = analyze_color_mapping(input_grid, output_grid)
#     if color_mapping:
#         transformations['color_mapping'] = color_mapping
    
#     # Check for shape transformations
#     shape_changes = analyze_shape_changes(input_grid, output_grid)
#     if shape_changes:
#         transformations['shape_changes'] = shape_changes
    
#     return transformations

# # Helper functions for pattern detection
# def is_pattern(arr: np.ndarray) -> bool:
#     """Check if array contains a repeating pattern."""
#     if len(arr) < 2:
#         return False
    
#     # Try different pattern lengths
#     for length in range(1, len(arr) // 2 + 1):
#         pattern = arr[:length]
#         is_repeating = True
        
#         for i in range(length, len(arr), length):
            
            
            
            
            

# helpers.py
import numpy as np
from typing import List, Tuple, Dict, Any
from itertools import product
import torch
from config import Config

class GridHelpers:
    @staticmethod
    def get_grid_size(grid: np.ndarray) -> Tuple[int, int]:
        """Returns the height and width of a grid."""
        return grid.shape[0], grid.shape[1]
    
    @staticmethod
    def is_valid_grid(grid: np.ndarray) -> bool:
        """Checks if a grid is valid (contains only integers 0-9)."""
        return (isinstance(grid, np.ndarray) and 
                grid.dtype in [np.int32, np.int64] and 
                np.all(grid >= 0) and np.all(grid <= 9))

    @staticmethod
    def get_unique_colors(grid: np.ndarray) -> List[int]:
        """Returns list of unique colors (excluding 0) in the grid."""
        return sorted(list(set(grid.flatten()) - {0}))
    
    @staticmethod
    def find_objects(grid: np.ndarray) -> List[np.ndarray]:
        """
        Finds connected components (objects) in the grid.
        Returns list of binary masks for each object.
        """
        objects = []
        visited = np.zeros_like(grid, dtype=bool)
        
        def flood_fill(i: int, j: int, color: int) -> np.ndarray:
            if (i < 0 or i >= grid.shape[0] or 
                j < 0 or j >= grid.shape[1] or 
                visited[i,j] or grid[i,j] != color):
                return np.zeros_like(grid)
            
            mask = np.zeros_like(grid)
            stack = [(i,j)]
            
            while stack:
                ci, cj = stack.pop()
                if visited[ci,cj]:
                    continue
                    
                visited[ci,cj] = True
                mask[ci,cj] = 1
                
                for di, dj in [(0,1), (1,0), (0,-1), (-1,0)]:
                    ni, nj = ci + di, cj + dj
                    if (0 <= ni < grid.shape[0] and 
                        0 <= nj < grid.shape[1] and 
                        not visited[ni,nj] and 
                        grid[ni,nj] == color):
                        stack.append((ni,nj))
            
            return mask
        
        for i, j in product(range(grid.shape[0]), range(grid.shape[1])):
            if not visited[i,j] and grid[i,j] != 0:
                obj_mask = flood_fill(i, j, grid[i,j])
                if np.any(obj_mask):
                    objects.append(obj_mask * grid[i,j])
        
        return objects

class TransformationHelpers:
    @staticmethod
    def rotate_grid(grid: np.ndarray, k: int = 1) -> np.ndarray:
        """Rotates grid k*90 degrees counterclockwise."""
        return np.rot90(grid, k)
    
    @staticmethod
    def flip_grid(grid: np.ndarray, axis: int = 0) -> np.ndarray:
        """Flips grid along specified axis (0=horizontal, 1=vertical)."""
        return np.flip(grid, axis)
    
    @staticmethod
    def crop_grid(grid: np.ndarray) -> np.ndarray:
        """Crops grid to minimum bounding box containing non-zero elements."""
        non_zero = np.nonzero(grid)
        if len(non_zero[0]) == 0:
            return np.zeros((1, 1), dtype=grid.dtype)
            
        return grid[
            non_zero[0].min():non_zero[0].max()+1,
            non_zero[1].min():non_zero[1].max()+1
        ]
    
    @staticmethod
    def extend_grid(grid: np.ndarray, new_shape: Tuple[int, int]) -> np.ndarray:
        """Extends grid to new shape by padding with zeros."""
        if new_shape[0] < grid.shape[0] or new_shape[1] < grid.shape[1]:
            raise ValueError("New shape must be larger than current shape")
            
        result = np.zeros(new_shape, dtype=grid.dtype)
        result[:grid.shape[0], :grid.shape[1]] = grid
        return result

class PatternHelpers:
    @staticmethod
    def find_symmetry(grid: np.ndarray) -> Dict[str, bool]:
        """Checks for various types of symmetry in the grid."""
        h, w = grid.shape
        
        # Vertical symmetry
        v_sym = all(np.array_equal(grid[:, i], grid[:, w-1-i]) 
                for i in range(w//2))
        
        # Horizontal symmetry
        h_sym = all(np.array_equal(grid[i, :], grid[h-1-i, :]) 
                for i in range(h//2))
        
        # Diagonal symmetry (main diagonal)
        if h == w:  # Only check diagonal symmetry for square grids
            d_sym = all(np.array_equal(grid[i, j], grid[j, i]) 
                    for i, j in product(range(h), range(w)) if i < j)
        else:
            d_sym = False
            
        return {
            'vertical': v_sym,
            'horizontal': h_sym,
            'diagonal': d_sym
        }
    
    @staticmethod
    def find_patterns(grid: np.ndarray) -> Dict[str, Any]:
        """Analyzes grid for common patterns."""
        patterns = {
            'repeating_rows': [],
            'repeating_cols': [],
            'color_counts': {},
            'symmetry': PatternHelpers.find_symmetry(grid)
        }
        
        # Find repeating rows
        for i in range(grid.shape[0]):
            for j in range(i+1, grid.shape[0]):
                if np.array_equal(grid[i], grid[j]):
                    patterns['repeating_rows'].append((i,j))
                    
        # Find repeating columns
        for i in range(grid.shape[1]):
            for j in range(i+1, grid.shape[1]):
                if np.array_equal(grid[:,i], grid[:,j]):
                    patterns['repeating_cols'].append((i,j))
                    
        # Count colors
        unique, counts = np.unique(grid, return_counts=True)
        patterns['color_counts'] = dict(zip(unique, counts))
        
        return patterns

class ModelHelpers:
    @staticmethod
    def to_tensor(grid: np.ndarray) -> torch.Tensor:
        """Converts numpy grid to PyTorch tensor."""
        return torch.from_numpy(grid).float().to(Config.DEVICE)
    
    @staticmethod
    def to_numpy(tensor: torch.Tensor) -> np.ndarray:
        """Converts PyTorch tensor to numpy array."""
        return tensor.cpu().detach().numpy()
    
    @staticmethod
    def create_grid_mask(grid: np.ndarray) -> np.ndarray:
        """Creates attention mask for grid (0 for padding, 1 for content)."""
        return (grid != 0).astype(np.float32)

class MetricHelpers:
    @staticmethod
    def calculate_accuracy(prediction: np.ndarray, target: np.ndarray) -> float:
        """Calculates exact match accuracy between prediction and target."""
        if prediction.shape != target.shape:
            return 0.0
        return float(np.array_equal(prediction, target))
    
    @staticmethod
    def calculate_similarity(grid1: np.ndarray, grid2: np.ndarray) -> float:
        """Calculates similarity score between two grids."""
        if grid1.shape != grid2.shape:
            return 0.0
        return float(np.sum(grid1 == grid2)) / grid1.size
    
    @staticmethod
    def calculate_metrics(prediction: np.ndarray, 
                        target: np.ndarray) -> Dict[str, float]:
        """Calculates various metrics between prediction and target."""
        return {
            'accuracy': MetricHelpers.calculate_accuracy(prediction, target),
            'similarity': MetricHelpers.calculate_similarity(prediction, target),
            'shape_match': float(prediction.shape == target.shape),
            'color_match': float(set(np.unique(prediction)) == 
                            set(np.unique(target)))
        }

class DebugHelpers:
    @staticmethod
    def print_grid(grid: np.ndarray, title: str = None):
        """Prints a formatted representation of the grid."""
        if title:
            print(f"\n{title}")
        print("\nShape:", grid.shape)
        print("\nGrid:")
        for row in grid:
            print(" ".join(f"{x:2d}" for x in row))
        print("\nUnique values:", np.unique(grid))
    
    @staticmethod
    def log_transformation(func_name: str, input_grid: np.ndarray, 
                        output_grid: np.ndarray):
        """Logs details about a grid transformation."""
        print(f"\nTransformation: {func_name}")
        print(f"Input shape: {input_grid.shape}")
        print(f"Output shape: {output_grid.shape}")
        print(f"Input unique values: {np.unique(input_grid)}")
        print(f"Output unique values: {np.unique(output_grid)}")