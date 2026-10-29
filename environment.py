"""Deterministic building grid and mutable environmental state."""

from math import isfinite
from numbers import Integral
from typing import Iterable, Optional, Tuple

import numpy as np

from config import (
    COLS,
    DYNAMIC_BLOCKED,
    EMPTY,
    EXIT,
    FIRE,
    MAX_RISK,
    MIN_RISK,
    ROWS,
    WALL,
)

Position = Tuple[int, int]


class BuildingEnvironment:
    """A deterministic four-neighbor building map with mutable hazards.

    Coordinates use ``(row, column)`` and both ``grid`` and ``risk_map`` use
    NumPy arrays indexed as ``[row, column]``. Exits remain traversable unless
    fire or a dynamic obstacle blocks them.
    """

    def __init__(
        self,
        rows: int = ROWS,
        cols: int = COLS,
        start: Optional[Position] = None,
        exits: Optional[Iterable[Position]] = None,
    ):
        if (
            isinstance(rows, bool) or not isinstance(rows, Integral)
            or isinstance(cols, bool) or not isinstance(cols, Integral)
            or rows < 4 or cols < 4
        ):
            raise ValueError("Building dimensions must be integers of at least 4")

        self.rows = int(rows)
        self.cols = int(cols)
        default_start, default_exits = self._default_positions(self.rows, self.cols)
        self._initial_start = self._normalize_position(
            default_start if start is None else start
        )
        self._initial_exits = [
            self._normalize_position(position)
            for position in (default_exits if exits is None else exits)
        ]
        if len(set(self._initial_exits)) != len(self._initial_exits):
            raise ValueError("Exit positions must be unique")

        self.start = self._initial_start
        self.exits = list(self._initial_exits)
        self.grid = np.zeros((self.rows, self.cols), dtype=int)
        self.risk_map = np.zeros((self.rows, self.cols), dtype=float)
        self.boundary_walls = set()
        self.permanent_walls = set()
        self.removable_walls = set()
        self.fire_cells = set()
        self.dynamic_blocked_cells = set()
        self.version = 0
        self._initialize_layout()

    @staticmethod
    def _default_positions(rows: int, cols: int):
        if rows == ROWS and cols == COLS:
            return (1, 1), [(1, 18), (18, 1), (18, 18)]
        return (1, 1), [(1, cols - 2), (rows - 2, 1), (rows - 2, cols - 2)]

    @staticmethod
    def _normalize_position(position) -> Position:
        if not isinstance(position, (tuple, list)) or len(position) != 2:
            raise ValueError("Positions must be (row, column) pairs")
        if any(isinstance(value, bool) or not isinstance(value, Integral)
               for value in position):
            raise ValueError("Position coordinates must be integers")
        return int(position[0]), int(position[1])

    def _in_bounds(self, position: Position) -> bool:
        row, col = position
        return 0 <= row < self.rows and 0 <= col < self.cols

    def _make_permanent_walls(self) -> set:
        walls = set()

        for row in range(self.rows):
            walls.add((row, 0))
            walls.add((row, self.cols - 1))
        for col in range(self.cols):
            walls.add((0, col))
            walls.add((self.rows - 1, col))

        # Three wall segments create rooms and corridors. Each has a fixed
        # connecting gap, scaled to the configured grid dimensions.
        vertical_col = (self.cols * 6) // 20
        vertical_gap = (self.rows * 9) // 20
        for row in range(3, self.rows - 3):
            if row != vertical_gap:
                walls.add((row, vertical_col))

        horizontal_row = (self.rows * 7) // 20
        horizontal_gap = self.cols // 2
        for col in range(3, self.cols - 3):
            if col != horizontal_gap:
                walls.add((horizontal_row, col))

        second_vertical_col = (self.cols * 13) // 20
        second_vertical_gap = (self.rows * 13) // 20
        for row in range(max(3, (self.rows * 9) // 20), self.rows - 2):
            if row != second_vertical_gap:
                walls.add((row, second_vertical_col))

        return walls

    def _initialize_layout(self) -> None:
        self.grid.fill(EMPTY)
        self.risk_map.fill(0.0)
        self.fire_cells.clear()
        self.dynamic_blocked_cells.clear()
        self.removable_walls.clear()
        self.boundary_walls = {
            (row, col)
            for row in range(self.rows)
            for col in range(self.cols)
            if row in (0, self.rows - 1) or col in (0, self.cols - 1)
        }
        all_walls = self._make_permanent_walls()
        protected = {self.start, *self.exits}

        if not self._in_bounds(self.start):
            raise ValueError(f"Start position {self.start} is outside the grid")
        if not self.exits:
            raise ValueError("At least one emergency exit is required")
        for exit_position in self.exits:
            if not self._in_bounds(exit_position):
                raise ValueError(f"Exit {exit_position} is outside the grid")

        interior_walls = all_walls - self.boundary_walls
        conflict = protected.intersection(interior_walls | self.boundary_walls)
        if conflict:
            raise ValueError(f"Start and exits must be on walkable cells; conflicts: {conflict}")

        self.permanent_walls = all_walls
        for row, col in all_walls:
            self.grid[row, col] = WALL
        for row, col in self.exits:
            self.grid[row, col] = EXIT
        self.grid[self.start] = EMPTY

    def reset(self) -> None:
        """Restore the constructor's start, exits, walls, and empty risk state."""
        self.start = self._initial_start
        self.exits = list(self._initial_exits)
        self._initialize_layout()
        self.version += 1

    def in_bounds(self, position) -> bool:
        try:
            normalized = self._normalize_position(position)
        except ValueError:
            return False
        return self._in_bounds(normalized)

    def is_blocked(self, position) -> bool:
        if not self.in_bounds(position):
            return True
        row, col = self._normalize_position(position)
        return self.grid[row, col] in (WALL, FIRE, DYNAMIC_BLOCKED)

    def neighbors(self, position):
        if not self.in_bounds(position):
            return []
        row, col = self._normalize_position(position)
        candidates = (
            (row - 1, col),
            (row + 1, col),
            (row, col - 1),
            (row, col + 1),
        )
        return [cell for cell in candidates if not self.is_blocked(cell)]

    def _require_in_bounds(self, position) -> Position:
        normalized = self._normalize_position(position)
        if not self._in_bounds(normalized):
            raise ValueError(f"Position {normalized} is outside the grid")
        return normalized

    def add_wall(self, position) -> None:
        """Add a removable wall to a currently walkable, unprotected cell."""
        cell = self._require_in_bounds(position)
        if cell in self.boundary_walls:
            raise ValueError("Boundary walls are permanent")
        if cell == self.start or cell in self.exits:
            raise ValueError("The start and emergency exits are protected")
        if cell in self.permanent_walls:
            return
        if self.grid[cell] in (FIRE, DYNAMIC_BLOCKED):
            raise ValueError("Clear fire or dynamic blockage before adding a wall")
        self.grid[cell] = WALL
        self.removable_walls.add(cell)
        self.version += 1

    def remove_wall(self, position) -> bool:
        """Remove a wall added with ``add_wall``; protected walls stay intact."""
        cell = self._require_in_bounds(position)
        if cell in self.boundary_walls or cell in self.permanent_walls:
            return False
        if cell not in self.removable_walls:
            return False
        self.removable_walls.remove(cell)
        self.grid[cell] = EXIT if cell in self.exits else EMPTY
        self.version += 1
        return True

    def set_fire(self, position, active: bool = True) -> None:
        """Set or clear fire state on a non-wall cell; no propagation occurs."""
        cell = self._require_in_bounds(position)
        if active:
            if cell in self.permanent_walls or cell in self.removable_walls:
                raise ValueError("Fire cannot be placed inside a wall")
            if cell in self.dynamic_blocked_cells:
                raise ValueError("Clear the dynamic blockage before setting fire")
            if cell not in self.fire_cells:
                self.fire_cells.add(cell)
                self.grid[cell] = FIRE
                self.version += 1
        elif cell in self.fire_cells:
            self.fire_cells.remove(cell)
            self.grid[cell] = EXIT if cell in self.exits else EMPTY
            self.version += 1

    def set_blocked(self, position, blocked: bool = True) -> None:
        """Set or clear a dynamic non-fire blockage."""
        cell = self._require_in_bounds(position)
        if blocked:
            if cell in self.permanent_walls or cell in self.removable_walls:
                raise ValueError("A wall is already blocking this cell")
            if cell in self.fire_cells:
                raise ValueError("Clear fire before changing dynamic blockage")
            if cell not in self.dynamic_blocked_cells:
                self.dynamic_blocked_cells.add(cell)
                self.grid[cell] = DYNAMIC_BLOCKED
                self.version += 1
        elif cell in self.dynamic_blocked_cells:
            self.dynamic_blocked_cells.remove(cell)
            self.grid[cell] = EXIT if cell in self.exits else EMPTY
            self.version += 1

    def add_blocked(self, position) -> None:
        """Convenience alias for adding a dynamic blockage."""
        self.set_blocked(position, True)

    def remove_blocked(self, position) -> None:
        """Convenience alias for clearing a dynamic blockage."""
        self.set_blocked(position, False)

    def set_risk(self, position, value: float) -> None:
        """Set a cell's finite risk value on the shared 0–100 scale."""
        cell = self._require_in_bounds(position)
        try:
            risk = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError("Risk must be a finite number between 0 and 100") from error
        if not isfinite(risk) or not MIN_RISK <= risk <= MAX_RISK:
            raise ValueError(f"Risk must be finite and between {MIN_RISK} and {MAX_RISK}")
        if self.risk_map[cell] != risk:
            self.risk_map[cell] = risk
            self.version += 1


# Existing code and callers can continue importing Environment.
Environment = BuildingEnvironment
