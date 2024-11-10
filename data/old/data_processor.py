import numpy as np

class DataProcessor:
    def preprocess_grid(self, grid):
        """Converts grid data into a NumPy array for easier manipulation."""
        return np.array(grid)

    def extract_features(self, grid):
        """Extracts features such as color patterns, object boundaries, and symmetries."""
        features = {
            'color_count': np.unique(grid, return_counts=True),
            'symmetry': self.check_symmetry(grid),
            # Additional features can be added as needed
        }
        return features

    def check_symmetry(self, grid):
        """Checks if the grid is symmetrical and returns results."""
        return np.array_equal(grid, np.flip(grid, axis=1))  # Horizontal symmetry
