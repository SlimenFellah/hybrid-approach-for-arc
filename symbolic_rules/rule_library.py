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


def repeat_tile_flip_row_blocks(grid: List[List[int]]) -> List[List[int]]:
    """
    Repeats the input 3x3 times, flipping every alternate 2-row block vertically and horizontally.
    Matches behavior of ARC task 00576224.
    """
    arr = to_np(grid)
    tile_h, tile_w = arr.shape
    result = np.zeros((tile_h * 3, tile_w * 3), dtype=int)

    for i in range(3):
        for j in range(3):
            tile = arr.copy()
            if i % 2 == 1:
                tile = np.flip(tile)  # Flip both axes
            # Write tile at correct block
            row_start = i * tile_h
            col_start = j * tile_w
            result[row_start:row_start + tile_h, col_start:col_start + tile_w] = tile

    return to_list(result)


def repeat_tile(grid: List[List[int]], rows: int, cols: int) -> List[List[int]]:
    arr = to_np(grid)
    return to_list(np.tile(arr, (rows, cols)))


def flip_alternate_rows(grid: List[List[int]]) -> List[List[int]]:
    arr = to_np(grid)
    result = arr.copy()
    for i in range(1, arr.shape[0], 2):
        result[i] = result[i][::-1]
    return to_list(result)


def replace_color(grid: List[List[int]], from_color: int, to_color: int) -> List[List[int]]:
    arr = to_np(grid)
    arr[arr == from_color] = to_color
    return to_list(arr)


def rotate_until_match(input_grid, target_grid):
    input_arr = to_np(input_grid)
    target_arr = to_np(target_grid)

    for k in range(4):  # Try 0, 90, 180, 270 degrees
        rotated = np.rot90(input_arr, -k)
        if np.array_equal(rotated, target_arr):
            return to_list(rotated), k * 90

    return None, None
