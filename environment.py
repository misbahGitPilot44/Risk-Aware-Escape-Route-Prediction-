
# environment.py

## This module creates the building layout and stores the start, exits, and walls.

## The environment uses four-directional movement. Fire is initially absent, so we'll add it in a later step.

import numpy as np

from config import (
    ROWS,
    COLS,
    EMPTY,
    WALL,
    FIRE,
    EXIT,
)


class Environment:
    def __init__(self):
        self.rows = ROWS
        self.cols = COLS

        # 0 = floor, 1 = wall, 2 = fire, 3 = exit
        self.grid = np.zeros(
            (self.rows, self.cols),
            dtype=int
        )

        # Risk values are separate from cell types.
        self.risk_map = np.zeros(
            (self.rows, self.cols),
            dtype=float
        )

        # Coordinates use (row, column).
        self.start = (1, 1)

        self.exits = [
            (1, 18),
            (18, 1),
            (18, 18),
        ]

        self.fire_cells = set()

        self._create_building()

    def _create_building(self):
        # Outer boundary walls
        self.grid[0, :] = WALL
        self.grid[-1, :] = WALL
        self.grid[:, 0] = WALL
        self.grid[:, -1] = WALL

        # Interior walls with gaps for corridors
        for row in range(3, 17):
            if row != 9:
                self.grid[row, 6] = WALL

        for col in range(3, 17):
            if col != 10:
                self.grid[7, col] = WALL

        for row in range(9, 18):
            if row != 13:
                self.grid[row, 13] = WALL

        # Mark emergency exits
        for row, col in self.exits:
            self.grid[row, col] = EXIT

        # Ensure start is traversable
        self.grid[self.start] = EMPTY

    def in_bounds(self, position):
        row, col = position

        return (
            0 <= row < self.rows
            and 0 <= col < self.cols
        )

    def is_blocked(self, position):
        row, col = position

        return (
            not self.in_bounds(position)
            or self.grid[row, col] == WALL
            or self.grid[row, col] == FIRE
        )

    def neighbors(self, position):
        row, col = position

        candidates = [
            (row - 1, col),
            (row + 1, col),
            (row, col - 1),
            (row, col + 1),
        ]

        return [
            cell
            for cell in candidates
            if self.in_bounds(cell)
            and not self.is_blocked(cell)
        ]
