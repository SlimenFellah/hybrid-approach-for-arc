from typing import List, Dict, Any
import numpy as np
from scipy import ndimage

class SymbolicFeatureExtractor:
    @staticmethod
    def extract_patterns(grid: np.ndarray) -> Dict[str, Any]:
        patterns = {
            'symmetry': SymbolicFeatureExtractor._detect_symmetry(grid),
            'repetition': SymbolicFeatureExtractor._detect_repetition(grid),
            'objects': SymbolicFeatureExtractor._detect_objects(grid),
            'relationships': SymbolicFeatureExtractor._detect_relationships(grid)
        }
        return patterns
    
    @staticmethod
    def _detect_symmetry(grid: np.ndarray) -> Dict[str, bool]:
        h, w = grid.shape
        return {
            'horizontal': np.array_equal(grid, np.flip(grid, 0)),
            'vertical': np.array_equal(grid, np.flip(grid, 1)),
            'diagonal': np.array_equal(grid, grid.T)
        }
    
    @staticmethod
    def _detect_repetition(grid: np.ndarray) -> Dict[str, Any]:
        patterns = {}
        
        # Detect horizontal patterns
        for i in range(1, grid.shape[0]):
            if np.array_equal(grid[i], grid[0]):
                patterns['horizontal_period'] = i
                break
        
        # Detect vertical patterns
        for i in range(1, grid.shape[1]):
            if np.array_equal(grid[:, i], grid[:, 0]):
                patterns['vertical_period'] = i
                break
                
        return patterns
    
    @staticmethod
    def _detect_objects(grid: np.ndarray) -> List[Dict[str, Any]]:
        objects = []
        labeled_array, num_features = ndimage.label(grid > 0)
        
        for i in range(1, num_features + 1):
            obj_mask = labeled_array == i
            obj = grid * obj_mask
            
            # Get object properties
            coords = np.where(obj_mask)
            min_y, max_y = np.min(coords[0]), np.max(coords[0])
            min_x, max_x = np.min(coords[1]), np.max(coords[1])
            
            objects.append({
                'value': grid[coords][0],
                'size': np.sum(obj_mask),
                'bbox': (min_y, min_x, max_y, max_x),
                'centroid': (np.mean(coords[0]), np.mean(coords[1])),
                'mask': obj_mask
            })
            
        return objects
    
    @staticmethod
    def _detect_relationships(grid: np.ndarray) -> Dict[str, Any]:
        objects = SymbolicFeatureExtractor._detect_objects(grid)
        relationships = {
            'spatial': [],
            'value': []
        }
        
        # Analyze spatial relationships between objects
        for i, obj1 in enumerate(objects):
            for j, obj2 in enumerate(objects[i+1:], i+1):
                y1, x1 = obj1['centroid']
                y2, x2 = obj2['centroid']
                
                dx = x2 - x1
                dy = y2 - y1
                dist = np.sqrt(dx*dx + dy*dy)
                
                relationships['spatial'].append({
                    'obj1': i,
                    'obj2': j,
                    'distance': dist,
                    'direction': np.arctan2(dy, dx)
                })
                
                # Analyze value relationships
                relationships['value'].append({
                    'obj1': i,
                    'obj2': j,
                    'value_diff': obj2['value'] - obj1['value']
                })
                
        return relationships