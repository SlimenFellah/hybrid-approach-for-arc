class ARCPreprocessor:
    @staticmethod
    def pad_grid(grid: np.ndarray, target_size: int = Config.MAX_GRID_SIZE) -> np.ndarray:
        """Pad grid to target size with zeros"""
        h, w = grid.shape
        pad_h = max(0, target_size - h)
        pad_w = max(0, target_size - w)
        
        return np.pad(grid, ((0, pad_h), (0, pad_w)), mode='constant')
    
    @staticmethod
    def normalize_grid(grid: np.ndarray) -> np.ndarray:
        """Normalize grid values to [0, 1]"""
        return grid / Config.NUM_COLORS
    
    @staticmethod
    def extract_grid_features(grid: np.ndarray) -> Dict[str, Any]:
        """Extract basic grid features"""
        features = {
            'shape': grid.shape,
            'unique_values': np.unique(grid),
            'num_nonzero': np.count_nonzero(grid),
            'mean': np.mean(grid),
            'std': np.std(grid),
            'min': np.min(grid),
            'max': np.max(grid)
        }
        return features