"""Risk-aware A* route finding for four-directional grid environments."""

from heapq import heappop, heappush
from math import isfinite, inf
from numbers import Integral
from typing import Optional, Sequence, Tuple

from config import DISTANCE_WEIGHT, MAX_RISK, MIN_RISK, RISK_WEIGHT

Position = Tuple[int, int]
Path = Sequence[Position]


def _valid_position(environment, position) -> bool:
    """Return whether position is a two-integer coordinate inside the grid."""
    if not isinstance(position, (tuple, list)) or len(position) != 2:
        return False
    if any(isinstance(value, bool) or not isinstance(value, Integral)
           for value in position):
        return False
    return environment.in_bounds(tuple(position))


def _heuristic(position: Position, goal: Position) -> float:
    """Lower bound on remaining cost for four-directional movement."""
    distance = abs(position[0] - goal[0]) + abs(position[1] - goal[1])
    return DISTANCE_WEIGHT * distance


def movement_cost(environment, source: Position, destination: Position) -> float:
    """Return the cost of entering destination under the configured risk scale.

    Risk is stored on a 0..MAX_RISK scale and normalized exactly once here.
    Invalid, non-finite, or out-of-range risk values raise ``ValueError``.
    """
    if not _valid_position(environment, source) or not _valid_position(environment, destination):
        raise ValueError("Movement positions must be valid in-bounds grid coordinates")
    if (
        not isfinite(DISTANCE_WEIGHT)
        or DISTANCE_WEIGHT <= 0
        or not isfinite(RISK_WEIGHT)
        or RISK_WEIGHT < 0
    ):
        raise ValueError("Distance weight must be positive and risk weight non-negative")
    if not isfinite(MAX_RISK) or MAX_RISK <= 0:
        raise ValueError("Maximum risk must be a finite positive value")

    try:
        risk = float(environment.risk_map[destination[0], destination[1]])
    except (TypeError, ValueError, IndexError, KeyError) as error:
        raise ValueError("Destination cell must have a numeric risk value") from error

    if not isfinite(risk) or not MIN_RISK <= risk <= MAX_RISK:
        raise ValueError(
            f"Risk must be finite and between {MIN_RISK} and {MAX_RISK}; got {risk!r}"
        )

    cost = DISTANCE_WEIGHT + RISK_WEIGHT * (risk / MAX_RISK)
    if not isfinite(cost) or cost < 0:
        raise ValueError("Configured movement weights must produce a finite non-negative cost")
    return cost


def calculate_path_cost(environment, path: Optional[Path]) -> float:
    """Sum the shared movement cost for each edge in path.

    An absent or empty path has infinite cost. Invalid or blocked transitions
    raise ``ValueError`` rather than reporting a misleading total.
    """
    if not path:
        return inf

    for position in path:
        if not _valid_position(environment, position):
            raise ValueError("Path contains an invalid grid coordinate")
        if environment.is_blocked(tuple(position)):
            raise ValueError("Path contains a blocked cell")

    total = 0.0
    for source, destination in zip(path, path[1:]):
        source = tuple(source)
        destination = tuple(destination)
        if abs(source[0] - destination[0]) + abs(source[1] - destination[1]) != 1:
            raise ValueError("Path contains a non-adjacent movement")
        total += movement_cost(environment, source, destination)
    return total


def _reconstruct_path(came_from, current: Position) -> list[Position]:
    path = [current]

    while current in came_from:
        current = came_from[current]
        path.append(current)

    path.reverse()
    return path


def astar_with_cost(environment, start, goal) -> Optional[Tuple[list[Position], float]]:
    """Return ``(path, total_cost)`` for a lowest-cost route, or ``None``.

    Four-directional moves cost ``DISTANCE_WEIGHT + RISK_WEIGHT * risk/100``
    where risk is the value of the destination cell. Manhattan distance times
    ``DISTANCE_WEIGHT`` is admissible and consistent because risk penalties are
    non-negative, so every edge costs at least the distance weight.
    """
    if not _valid_position(environment, start) or not _valid_position(environment, goal):
        return None
    start = tuple(start)
    goal = tuple(goal)
    if environment.is_blocked(start) or environment.is_blocked(goal):
        return None

    open_set = []
    heappush(open_set, (_heuristic(start, goal), 0.0, start))
    came_from = {}
    g_score = {start: 0.0}

    while open_set:
        _, queued_cost, current = heappop(open_set)

        # A better route may have been queued after this entry was inserted.
        if queued_cost != g_score.get(current):
            continue

        if current == goal:
            path = _reconstruct_path(came_from, current)
            return path, queued_cost

        for neighbor in environment.neighbors(current):
            if not _valid_position(environment, neighbor):
                continue
            neighbor = tuple(neighbor)
            if environment.is_blocked(neighbor):
                continue
            tentative_cost = queued_cost + movement_cost(environment, current, neighbor)

            if tentative_cost < g_score.get(neighbor, inf):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_cost
                priority = tentative_cost + _heuristic(neighbor, goal)
                heappush(open_set, (priority, tentative_cost, neighbor))

    return None


def astar(environment, start, goal) -> Optional[list[Position]]:
    """Return the lowest-cost path from start to goal, or ``None``.

    This path-only wrapper preserves the original public API. Use
    :func:`astar_with_cost` when the accumulated cost is also required.
    """
    result = astar_with_cost(environment, start, goal)
    return None if result is None else result[0]
