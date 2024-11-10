import numpy as np

class Transformation:
    @staticmethod
    def rotate(grid, degrees=90):
        # Rotate the grid by the specified degrees
        return np.rot90(grid, k=degrees // 90)

    @staticmethod
    def change_color(grid, from_color, to_color):
        # Replace one color with another
        grid[grid == from_color] = to_color
        return grid
