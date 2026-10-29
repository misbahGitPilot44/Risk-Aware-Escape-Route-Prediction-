"""Evaluate and select risk-aware routes across all configured exits."""

from math import isclose, inf
from typing import Any, Dict, List, Optional, Tuple

from astar import astar_with_cost

Position = Tuple[int, int]
ExitEvaluation = Dict[str, Any]


def path_cost(path) -> float:
    """Return movement steps for compatibility with the original helper.

    This is route length in moves, not the risk-weighted cost. Use the
    ``cost`` field returned by :func:`find_best_route` for weighted cost.
    """
    if not path:
        return inf
    return len(path) - 1


def _path_risk(environment, path) -> float:
    """Sum risk for entered cells, excluding the initial position."""
    return sum(
        float(environment.risk_map[row, col])
        for row, col in path[1:]
    )


def evaluate_exits(environment, start: Optional[Position] = None) -> List[ExitEvaluation]:
    """Return route metrics for every configured exit from the same start.

    Unreachable exits are included with ``reachable=False`` and ``None`` for
    path and route metrics. This keeps the full comparison available to a UI.
    """
    origin = environment.start if start is None else start
    evaluations: List[ExitEvaluation] = []

    for exit_position in environment.exits:
        result = astar_with_cost(environment, origin, exit_position)
        if result is None:
            evaluations.append({
                "exit": tuple(exit_position),
                "reachable": False,
                "path": None,
                "steps": None,
                "risk": None,
                "cost": None,
            })
            continue

        path, total_cost = result
        evaluations.append({
            "exit": tuple(exit_position),
            "reachable": True,
            "path": path,
            "steps": len(path) - 1,
            "risk": _path_risk(environment, path),
            "cost": total_cost,
        })

    return evaluations


def _is_better(candidate: ExitEvaluation, current: ExitEvaluation) -> bool:
    """Apply cost, step-count, then coordinate ordering deterministically."""
    candidate_cost = candidate["cost"]
    current_cost = current["cost"]
    if not isclose(candidate_cost, current_cost, rel_tol=1e-9, abs_tol=1e-9):
        return candidate_cost < current_cost

    candidate_key = (candidate["steps"], candidate["exit"])
    current_key = (current["steps"], current["exit"])
    return candidate_key < current_key


def find_best_route(environment, start: Optional[Position] = None) -> Optional[Dict[str, Any]]:
    """Choose the reachable exit with the lowest risk-weighted route cost.

    Near-equal floating-point costs (within 1e-9 relative or absolute
    tolerance) are treated as ties. Ties prefer fewer steps, then the
    lexicographically smaller ``(row, col)`` exit coordinate.

    Returns the selected route fields plus an ``evaluations`` list containing
    every exit's result, or ``None`` when no exit is reachable.
    """
    evaluations = evaluate_exits(environment, start)
    reachable = [item for item in evaluations if item["reachable"]]
    if not reachable:
        return None

    best = reachable[0]
    for candidate in reachable[1:]:
        if _is_better(candidate, best):
            best = candidate

    return {**best, "evaluations": evaluations}
