"""Deterministic scenarios for in-progress route validation and replanning."""

import numpy as np

from astar import calculate_path_cost
from config import FIRE, WALL
from replanning import ReplanningController
from route_planner import find_best_route


class GridEnvironment:
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


def _initial_route(environment):
    result = find_best_route(environment)
    assert result is not None
    return result


def test_valid_unchanged_route_does_not_replan():
    environment = GridEnvironment(exits=[(2, 6)])
    route = _initial_route(environment)
    controller = ReplanningController()

    result = controller.update(environment, environment.start, route)

    assert result["replanned"] is False
    assert result["safe_exit_exists"] is True
    assert result["path"] == route["path"]
    assert result["cost"] == route["cost"]


def test_fire_on_remaining_route_triggers_replan():
    environment = GridEnvironment(exits=[(2, 6)])
    route = _initial_route(environment)
    controller = ReplanningController()
    controller.update(environment, environment.start, route)
    current_position = (2, 1)
    environment.grid[2, 2] = FIRE

    result = controller.update(environment, current_position, route)

    assert result["replanned"] is True
    assert result["reason"] == "remaining_route_blocked"
    assert result["path"][0] == current_position
    assert (2, 2) not in result["path"]


def test_risk_increase_on_route_triggers_safer_alternative():
    environment = GridEnvironment(exits=[(2, 6)])
    route = _initial_route(environment)
    controller = ReplanningController()
    controller.update(environment, environment.start, route)
    environment.risk_map[2, 1:6] = 100

    result = controller.update(environment, environment.start, route)

    assert result["replanned"] is True
    assert result["reason"] == "lower_cost_route_available"
    assert result["cost"] < calculate_path_cost(environment, route["path"])
    assert all(row != 2 for row, col in result["path"][1:-1])


def test_environment_change_can_make_a_different_exit_preferable():
    environment = GridEnvironment(exits=[(2, 4), (0, 6)])
    environment.grid[1, 1:6] = WALL
    environment.grid[3, 1:6] = WALL
    environment.grid[2, 5:7] = WALL
    route = _initial_route(environment)
    assert route["exit"] == (2, 4)
    controller = ReplanningController()
    controller.update(environment, environment.start, route)

    environment.risk_map[2, 1:5] = 100
    result = controller.update(environment, environment.start, route)

    assert result["replanned"] is True
    assert result["reason"] == "better_exit_available"
    assert result["exit"] == (0, 6)


def test_replan_after_movement_starts_at_current_position():
    environment = GridEnvironment(exits=[(2, 6)])
    route = _initial_route(environment)
    controller = ReplanningController()
    controller.update(environment, environment.start, route)
    current_position = (2, 3)
    environment.grid[2, 4] = FIRE

    result = controller.update(environment, current_position, route)

    assert result["replanned"] is True
    assert result["path"][0] == current_position
    assert environment.start not in result["path"]


def test_blocked_current_position_returns_trapped_status():
    environment = GridEnvironment(exits=[(2, 6)])
    route = _initial_route(environment)
    current_position = (2, 1)
    environment.grid[current_position] = FIRE

    result = ReplanningController().update(environment, current_position, route)

    assert result["replanned"] is False
    assert result["status"] == "trapped"
    assert result["safe_exit_exists"] is False


def test_all_exits_unreachable_returns_explicit_no_route():
    environment = GridEnvironment(exits=[(2, 6)])
    route = _initial_route(environment)
    controller = ReplanningController()
    controller.update(environment, environment.start, route)
    environment.grid[1, 0] = FIRE
    environment.grid[3, 0] = FIRE
    environment.grid[2, 1] = FIRE

    result = controller.update(environment, environment.start, route)

    assert result["replanned"] is True
    assert result["reason"] == "remaining_route_blocked"
    assert result["status"] == "no_route"
    assert result["safe_exit_exists"] is False
    assert result["path"] is None


def test_still_optimal_route_is_not_replanned_after_irrelevant_change():
    environment = GridEnvironment(exits=[(2, 6)])
    route = _initial_route(environment)
    controller = ReplanningController()
    controller.update(environment, environment.start, route)
    environment.risk_map[4, 0] = 100

    result = controller.update(environment, environment.start, route)

    assert result["replanned"] is False
    assert result["reason"] == "no_significant_improvement"
    assert result["path"] == route["path"]


def test_replanning_uses_current_risk_instead_of_stored_route_cost():
    environment = GridEnvironment(exits=[(2, 6)])
    route = _initial_route(environment)
    assert route["cost"] == 6.0
    controller = ReplanningController()
    controller.update(environment, environment.start, route)
    environment.risk_map[2, 1:6] = 100

    result = controller.update(environment, environment.start, route)

    assert result["replanned"] is True
    assert result["cost"] == calculate_path_cost(environment, result["path"])
    assert result["cost"] != route["cost"]


def test_replacement_route_is_continuous_and_unblocked():
    environment = GridEnvironment(exits=[(2, 6)])
    route = _initial_route(environment)
    controller = ReplanningController()
    controller.update(environment, environment.start, route)
    environment.grid[2, 2] = FIRE

    result = controller.update(environment, (2, 1), route)
    path = result["path"]

    assert result["replanned"] is True
    assert all(not environment.is_blocked(cell) for cell in path)
    assert all(
        abs(first[0] - second[0]) + abs(first[1] - second[1]) == 1
        for first, second in zip(path, path[1:])
    )
