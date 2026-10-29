"""Deterministic tests for comparing routes to multiple exits."""

import numpy as np

from astar import calculate_path_cost
from config import FIRE, WALL
from route_planner import evaluate_exits, find_best_route


class GridEnvironment:
    """Small controlled implementation of the project's environment API."""

    def __init__(self, rows=5, cols=7, start=(2, 0), exits=None):
        self.rows = rows
        self.cols = cols
        self.start = start
        self.exits = list(exits or [])
        self.grid = np.zeros((rows, cols), dtype=int)
        self.risk_map = np.zeros((rows, cols), dtype=float)

    def in_bounds(self, position):
        row, col = position
        return 0 <= row < self.rows and 0 <= col < self.cols

    def is_blocked(self, position):
        return (
            not self.in_bounds(position)
            or self.grid[position] in (WALL, FIRE)
        )

    def neighbors(self, position):
        row, col = position
        candidates = ((row - 1, col), (row + 1, col),
                      (row, col - 1), (row, col + 1))
        return [cell for cell in candidates
                if self.in_bounds(cell) and not self.is_blocked(cell)]


def test_nearest_exit_wins_when_risk_is_zero():
    environment = GridEnvironment(
        start=(1, 1), exits=[(1, 2), (1, 6), (4, 1)]
    )

    result = find_best_route(environment)

    assert result is not None
    assert result["exit"] == (1, 2)
    assert result["steps"] == 1
    assert result["cost"] == 1.0


def test_farther_low_risk_exit_beats_shorter_high_risk_exit():
    environment = GridEnvironment(
        start=(2, 0), exits=[(2, 4), (0, 6)]
    )
    # Isolate the short corridor. Reaching (0, 6) from the start takes 8 steps.
    environment.grid[1, 1:6] = WALL
    environment.grid[3, 1:6] = WALL
    environment.grid[2, 5:7] = WALL
    environment.risk_map[2, 1:5] = 100

    result = find_best_route(environment)

    assert result is not None
    assert result["exit"] == (0, 6)
    assert result["steps"] > 4
    assert result["cost"] == 8.0
    by_exit = {item["exit"]: item for item in result["evaluations"]}
    assert by_exit[(2, 4)]["cost"] == 24.0


def test_unreachable_exit_is_skipped():
    environment = GridEnvironment(
        rows=3, cols=5, start=(1, 0), exits=[(1, 2), (1, 4)]
    )
    environment.grid[1, 2] = FIRE

    result = find_best_route(environment)

    assert result is not None
    assert result["exit"] == (1, 4)
    by_exit = {item["exit"]: item for item in result["evaluations"]}
    assert by_exit[(1, 2)]["reachable"] is False
    assert by_exit[(1, 2)]["path"] is None


def test_all_exits_unreachable_returns_none():
    environment = GridEnvironment(
        rows=3, cols=3, start=(1, 1), exits=[(0, 0), (2, 2)]
    )
    environment.grid[0, 1] = WALL
    environment.grid[1, 0] = WALL
    environment.grid[1, 2] = WALL
    environment.grid[2, 1] = WALL

    assert find_best_route(environment) is None
    evaluations = evaluate_exits(environment)
    assert len(evaluations) == 2
    assert all(not item["reachable"] for item in evaluations)


def test_planning_uses_supplied_current_position():
    environment = GridEnvironment(
        rows=4, cols=5, start=(0, 0), exits=[(3, 4)]
    )
    current_position = (3, 0)

    result = find_best_route(environment, start=current_position)

    assert result is not None
    assert result["path"][0] == current_position
    assert result["steps"] == 4


def test_selected_cost_matches_shared_path_cost():
    environment = GridEnvironment(
        start=(2, 0), exits=[(2, 6), (0, 6)]
    )
    environment.risk_map[2, 1:4] = [25, 50, 75]

    result = find_best_route(environment)

    assert result is not None
    assert result["cost"] == calculate_path_cost(environment, result["path"])


def test_every_exit_has_a_comparison_record():
    environment = GridEnvironment(
        rows=3, cols=5, start=(1, 0), exits=[(1, 2), (1, 4)]
    )
    environment.grid[1, 2] = FIRE

    evaluations = evaluate_exits(environment)

    assert [item["exit"] for item in evaluations] == environment.exits
    assert len(evaluations) == len(environment.exits)
    assert evaluations[0]["reachable"] is False
    assert evaluations[0]["steps"] is None
    assert evaluations[0]["risk"] is None
    assert evaluations[0]["cost"] is None
    assert evaluations[1]["reachable"] is True


def test_equal_cost_tie_uses_fewer_steps_first():
    environment = GridEnvironment(
        rows=3, cols=5, start=(1, 0), exits=[(1, 1), (1, 4)]
    )
    # Near exit costs 1 + 5*(60/100) = 4; farther exit costs 4 steps.
    environment.risk_map[1, 1] = 60

    result = find_best_route(environment)

    assert result is not None
    assert result["exit"] == (1, 1)
    assert result["cost"] == 4.0
    assert result["steps"] == 1


def test_equal_cost_and_steps_tie_uses_exit_coordinates():
    environment = GridEnvironment(
        start=(2, 3), exits=[(3, 3), (1, 3)]
    )

    result = find_best_route(environment)

    assert result is not None
    assert result["exit"] == (1, 3)
