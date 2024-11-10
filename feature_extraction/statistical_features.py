import numpy as np
from scipy import stats
from typing import Dict, Any

class StatisticalFeatureExtractor:
    @staticmethod
    def extract_features(grid: np.ndarray) -> Dict[str, Any]:
        features = {
            'basic_stats': StatisticalFeatureExtractor._basic_statistics(grid),
            'distribution': StatisticalFeatureExtractor._distribution_features(grid),
            'spatial': StatisticalFeatureExtractor._spatial_statistics(grid)
        }
        return features
    
    @staticmethod
    def _basic_statistics(grid: np.ndarray) -> Dict[str, float]:
        return {
            'mean': float(np.mean(grid)),
            'std': float(np.std(grid)),
            'min': float(np.min(grid)),
            'max': float(np.max(grid)),
            'median': float(np.median(grid)),
            'unique_count': len(np.unique(grid))
        }
    
    @staticmethod
    def _distribution_features(grid: np.ndarray) -> Dict[str, float]:
        flat_grid = grid.flatten()
        return {
            'skewness': float(stats.skew(flat_grid)),
            'kurtosis': float(stats.kurtosis(flat_grid)),
            'entropy': float(stats.entropy(np.bincount(flat_grid.astype(int))))
        }
    
    @staticmethod
    def _spatial_statistics(grid: np.ndarray) -> Dict[str, Any]:
        rows = np.mean(grid, axis=1)
        cols = np.mean(grid, axis=0)
        
        return {
            'row_variance': float(np.var(rows)),
            'col_variance': float(np.var(cols)),
            'spatial_autocorr': float(StatisticalFeatureExtractor._moran_i(grid))
        }
    
    @staticmethod
    def _moran_i(grid: np.ndarray) -> float:
        """Calculate Moran's I spatial autocorrelation"""
        n = grid.size
        mean = np.mean(grid)
        z = grid - mean
        
        # Create weight matrix (using queen's case adjacency)
        weights = np.zeros((n, n))
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i + di, j + dj
                        if 0 <= ni < grid.shape[0] and 0 <= nj < grid.shape[1]:
                            idx1 = i * grid.shape[1] + j
                            idx2 = ni * grid.shape[1] + nj
                            weights[idx1, idx2] = 1
        
        # Calculate Moran's I
        w_sum = np.sum(weights)
        numerator = np.sum(weights * np.outer(z.flatten(), z.flatten()))
        denominator = np.sum(z * z)
        
        return (n / w_sum) * (numerator / denominator)
