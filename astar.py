"""A* route finding for the building environment."""

from heapq import heappop, heappush
from math import inf

from config import RISK_WEIGHT


def _heuristic(position, goal):
    """Return the Manhattan distance between two grid positions."""
    return abs(position[0] - goal[0]) + abs(position[1] - goal[1])


def _reconstruct_path(came_from, current):
    path = [current]

    while current in came_from:
        current = came_from[current]
        path.append(current)

    path.reverse()
    return path


def astar(environment, start, goal):
    """Find a lowest-cost route from ``start`` to ``goal``.

    Movement is four-directional and each cell's risk contributes to the
    movement cost according to ``RISK_WEIGHT``.
    """
    if environment.is_blocked(start) or environment.is_blocked(goal):
        return None

    open_set = []
    heappush(open_set, (_heuristic(start, goal), start))
    came_from = {}
    g_score = {start: 0.0}

    while open_set:
        _, current = heappop(open_set)

        if current == goal:
            return _reconstruct_path(came_from, current)

        current_cost = g_score[current]
        for neighbor in environment.neighbors(current):
            row, col = neighbor
            risk = float(environment.risk_map[row, col])
            tentative_cost = current_cost + 1.0 + (RISK_WEIGHT * risk)

            if tentative_cost < g_score.get(neighbor, inf):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_cost
                priority = tentative_cost + _heuristic(neighbor, goal)
                heappush(open_set, (priority, neighbor))

    return None