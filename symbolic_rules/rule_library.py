import numpy as np
from typing import List


def to_np(grid: List[List[int]]) -> np.ndarray:
    return np.array(grid, dtype=int)


def to_list(grid: np.ndarray) -> List[List[int]]:
    return grid.tolist()


def horizontal_tile(grid: List[List[int]], count: int) -> List[List[int]]:
    arr = to_np(grid)
    return to_list(np.tile(arr, (1, count)))


def vertical_tile(grid: List[List[int]], count: int) -> List[List[int]]:
    arr = to_np(grid)
    return to_list(np.tile(arr, (count, 1)))


def tile_grid(grid: List[List[int]], rows: int, cols: int) -> List[List[int]]:
    arr = to_np(grid)
    return to_list(np.tile(arr, (rows, cols)))


def flip_horizontal(grid: List[List[int]]) -> List[List[int]]:
    arr = to_np(grid)
    return to_list(np.fliplr(arr))


def flip_vertical(grid: List[List[int]]) -> List[List[int]]:
    arr = to_np(grid)
    return to_list(np.flipud(arr))


def rotate_90(grid: List[List[int]]) -> List[List[int]]:
    arr = to_np(grid)
    return to_list(np.rot90(arr, k=-1))  # rotate clockwise


def repeat_tile_flip_pattern(grid: List[List[int]]) -> List[List[int]]:
    """
    Custom pattern seen in 00576224: repeat and flip in alternating tiles
    """
    arr = to_np(grid)
    tile_h, tile_w = arr.shape
    result = np.zeros((tile_h * 3, tile_w * 3), dtype=int)

    for i in range(3):
        for j in range(3):
            tile = arr.copy()
            if (i + j) % 2 == 1:
                tile = np.flip(tile)
            result[i * tile_h: (i + 1) * tile_h, j * tile_w: (j + 1) * tile_w] = tile

    return to_list(result)
