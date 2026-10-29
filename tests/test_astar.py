"""Deterministic tests for the risk-aware grid search."""

from math import inf, isclose

import numpy as np
import pytest

from astar import astar, astar_with_cost, calculate_path_cost
from config import FIRE, WALL


class GridEnvironment:
    """Small controlled environment matching the project's grid interface."""

    def __init__(self, rows=5, cols=7):
        self.rows = rows
        self.cols = cols
        self.grid = np.zeros((rows, cols), dtype=int)
        self.risk_map = np.zeros((rows, cols), dtype=float)

    def in_bounds(self, position):
        row, col = position
        return 0 <= row < self.rows and 0 <= col < self.cols

    def is_blocked(self, position):
        if not self.in_bounds(position):
            return True
        return self.grid[position] in (WALL, FIRE)

    def neighbors(self, position):
        row, col = position
        candidates = ((row - 1, col), (row + 1, col),
                      (row, col - 1), (row, col + 1))
        return [cell for cell in candidates
                if self.in_bounds(cell) and not self.is_blocked(cell)]


def test_zero_risk_returns_shortest_path():
    environment = GridEnvironment()

    result = astar_with_cost(environment, (2, 0), (2, 6))

    assert result is not None
    path, cost = result
    assert len(path) - 1 == 6
    assert cost == 6.0
    assert astar(environment, (2, 0), (2, 6)) == path


def test_path_avoids_walls():
    environment = GridEnvironment(rows=3, cols=5)
    environment.grid[1, 1:4] = WALL

    result = astar_with_cost(environment, (1, 0), (1, 4))

    assert result is not None
    path, cost = result
    assert len(path) - 1 == 6
    assert cost == 6.0
    assert all(not environment.is_blocked(cell) for cell in path)


def test_lower_cost_longer_risk_aware_detour_is_selected():
    environment = GridEnvironment()
    environment.risk_map[2, 1:6] = 100

    result = astar_with_cost(environment, (2, 0), (2, 6))

    assert result is not None
    path, cost = result
    assert len(path) - 1 > 6
    assert all(row != 2 for row, col in path[1:-1])
    assert cost == calculate_path_cost(environment, path)
    assert cost < 6 + 5 * (5 * 1.0)


def test_fire_cells_are_not_traversed():
    environment = GridEnvironment(rows=3, cols=5)
    environment.grid[1, 2] = FIRE

    result = astar_with_cost(environment, (1, 0), (1, 4))

    assert result is not None
    assert (1, 2) not in result[0]


def test_unreachable_goal_returns_none():
    environment = GridEnvironment(rows=3, cols=3)
    environment.grid[0, 1] = WALL
    environment.grid[1, 0] = WALL
    environment.grid[1, 2] = WALL
    environment.grid[2, 1] = WALL

    assert astar_with_cost(environment, (1, 1), (0, 0)) is None
    assert astar(environment, (1, 1), (0, 0)) is None


def test_start_equal_to_goal_has_zero_cost():
    environment = GridEnvironment()

    assert astar_with_cost(environment, (2, 2), (2, 2)) == ([(2, 2)], 0.0)
    assert calculate_path_cost(environment, [(2, 2)]) == 0.0


@pytest.mark.parametrize("risk", [-0.01, 100.01, float("nan"), inf, -inf])
def test_invalid_risk_values_raise_value_error(risk):
    environment = GridEnvironment(rows=1, cols=2)
    environment.risk_map[0, 1] = risk

    with pytest.raises(ValueError, match="Risk must be finite"):
        astar_with_cost(environment, (0, 0), (0, 1))


@pytest.mark.parametrize("start,goal", [
    ((-1, 0), (0, 1)),
    ((0, 0), (1, 2)),
    ((0.5, 0), (0, 1)),
    ((0, 0), (True, 1)),
])
def test_invalid_positions_are_rejected(start, goal):
    environment = GridEnvironment(rows=1, cols=2)

    assert astar_with_cost(environment, start, goal) is None


def test_start_or_goal_blocked_is_rejected():
    environment = GridEnvironment(rows=1, cols=3)
    environment.grid[0, 0] = WALL

    assert astar_with_cost(environment, (0, 0), (0, 2)) is None
    environment.grid[0, 0] = 0
    environment.grid[0, 2] = FIRE
    assert astar_with_cost(environment, (0, 0), (0, 2)) is None


def test_reported_cost_matches_independent_edge_sum():
    environment = GridEnvironment(rows=3, cols=5)
    environment.risk_map[0, 1:4] = [25, 50, 75]
    environment.risk_map[1, 2] = 10

    result = astar_with_cost(environment, (1, 0), (1, 4))

    assert result is not None
    path, reported_cost = result
    independent_sum = sum(
        1.0 + 5.0 * (environment.risk_map[row, col] / 100.0)
        for row, col in path[1:]
    )
    assert isclose(reported_cost, independent_sum)
    assert isclose(reported_cost, calculate_path_cost(environment, path))


def test_empty_path_cost_is_infinite():
    environment = GridEnvironment()

    assert calculate_path_cost(environment, None) == inf
