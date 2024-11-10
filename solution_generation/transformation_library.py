import numpy as np
from typing import Dict, Callable, List

class TransformationLibrary:
    def __init__(self):
        self.transformations: Dict[str, Callable] = self._init_transformations()
        
    def _init_transformations(self) -> Dict[str, Callable]:
        return {
            # Basic geometric transformations
            'rotate_90': lambda x: np.rot90(x),
            'rotate_180': lambda x: np.rot90(x, 2),
            'rotate_270': lambda x: np.rot90(x, 3),
            'flip_horizontal': lambda x: np.fliplr(x),
            'flip_vertical': lambda x: np.flipud(x),
            
            # Color transformations
            'increment_color': lambda x: (x + 1) % 10,
            'decrement_color': lambda x: (x - 1) % 10,
            'invert_color': lambda x: 9 - x if x != 0 else 0,
            
            # Pattern transformations
            'expand': lambda x: np.repeat(np.repeat(x, 2, axis=0), 2, axis=1),
            'compress': self._compress,
            'fill_holes': self._fill_holes,
            'extract_border': self._extract_border,
            
            # Complex transformations
            'symmetrize': self._symmetrize,
            'pattern_complete': self._pattern_complete,
        }
    
    @staticmethod
    def _compress(grid: np.ndarray) -> np.ndarray:
        """Remove empty rows and columns"""
        mask_rows = ~np.all(grid == 0, axis=1)
        mask_cols = ~np.all(grid == 0, axis=0)
        return grid[mask_rows][:, mask_cols]
    
    @staticmethod
    def _fill_holes(grid: np.ndarray) -> np.ndarray:
        """Fill single-cell holes with surrounding color"""
        result = grid.copy()
        h, w = grid.shape
        for i in range(1, h-1):
            for j in range(1, w-1):
                if grid[i,j] == 0:
                    neighbors = [
                        grid[i-1,j], grid[i+1,j],
                        grid[i,j-1], grid[i,j+1]
                    ]
                    non_zero = [n for n in neighbors if n != 0]
                    if len(non_zero) >= 3:
                        result[i,j] = max(set(non_zero), key=non_zero.count)
        return result
    
    @staticmethod
    def _extract_border(grid: np.ndarray) -> np.ndarray:
        """Extract the border of shapes"""
        result = np.zeros_like(grid)
        h, w = grid.shape
        for i in range(h):
            for j in range(w):
                if grid[i,j] != 0:
                    # Check if it's a border pixel
                    is_border = False
                    for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                        ni, nj = i + di, j + dj
                        if (ni < 0 or ni >= h or nj < 0 or nj >= w or 
                            grid[ni,nj] == 0):
                            is_border = True
                            break
                    if is_border:
                        result[i,j] = grid[i,j]
        return result
    
    @staticmethod
    def _symmetrize(grid: np.ndarray) -> np.ndarray:
        """Make the grid symmetric"""
        h, w = grid.shape
        result = grid.copy()
        # Horizontal symmetry
        for i in range(h):
            for j in range(w//2):
                if grid[i,j] != grid[i,w-1-j]:
                    result[i,j] = result[i,w-1-j] = max(
                        grid[i,j], grid[i,w-1-j])
        return result
    
    @staticmethod
    def _pattern_complete(grid: np.ndarray) -> np.ndarray:
        """Complete repeating patterns"""
        # Simple pattern completion - repeats the first found pattern
        h, w = grid.shape
        result = grid.copy()
        
        # Try to find horizontal pattern
        for pattern_width in range(1, w//2 + 1):
            pattern = grid[:, :pattern_width]
            is_pattern = True
            for i in range(pattern_width, w, pattern_width):
                if i + pattern_width > w:
                    break
                if not np.array_equal(pattern, grid[:, i:i+pattern_width]):
                    is_pattern = False
                    break
            if is_pattern:
                # Complete the pattern
                for i in range(0, w, pattern_width):
                    result[:, i:i+pattern_width] = pattern
                
        return result
    
    def apply_transformation(self, grid: np.ndarray, 
                        transformation_name: str) -> np.ndarray:
        if transformation_name not in self.transformations:
            raise ValueError(f"Unknown transformation: {transformation_name}")
        return self.transformations[transformation_name](grid)
    
    def get_all_transformations(self) -> List[str]:
        return list(self.transformations.keys())