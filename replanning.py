"""Stateful checks for invalid or newly suboptimal escape routes."""

from math import isclose
from numbers import Integral
from typing import Any, Dict, Optional, Sequence, Tuple

from astar import calculate_path_cost
from route_planner import find_best_route

Position = Tuple[int, int]
RouteInput = Any


class ReplanningController:
    """Check an in-progress route and replan when current conditions warrant it.

    The controller remembers the last observed grid, risk map, and exit list.
    It only runs a fresh multiple-exit search when that state changes, the
    remaining route becomes invalid, or no usable route was supplied.
    """

    def __init__(self, improvement_threshold: float = 0.05):
        if not 0.0 <= improvement_threshold < 1.0:
            raise ValueError("improvement_threshold must be in [0, 1)")
        self.improvement_threshold = improvement_threshold
        self._last_environment_state = None

    @staticmethod
    def _matrix_snapshot(matrix) -> tuple:
        """Return an immutable row snapshot for NumPy arrays or row iterables."""
        return tuple(tuple(row) for row in matrix)

    def _environment_snapshot(self, environment) -> tuple:
        return (
            self._matrix_snapshot(environment.grid),
            self._matrix_snapshot(environment.risk_map),
            tuple(tuple(exit_position) for exit_position in environment.exits),
        )

    @staticmethod
    def _extract_path(route: RouteInput) -> Optional[Sequence[Position]]:
        if route is None:
            return None
        if isinstance(route, dict):
            route = route.get("path")
        if not isinstance(route, (tuple, list)) or not route:
            return None
        return route

    @staticmethod
    def _valid_position(environment, position) -> bool:
        if not isinstance(position, (tuple, list)) or len(position) != 2:
            return False
        if any(isinstance(value, bool) or not isinstance(value, Integral)
               for value in position):
            return False
        return environment.in_bounds(tuple(position))

    @staticmethod
    def _route_exit(route: RouteInput, path, previous_exit) -> Optional[Position]:
        if previous_exit is not None:
            return tuple(previous_exit)
        if isinstance(route, dict) and route.get("exit") is not None:
            return tuple(route["exit"])
        return tuple(path[-1]) if path else None

    @staticmethod
    def _path_risk(environment, path) -> float:
        return sum(float(environment.risk_map[row, col]) for row, col in path[1:])

    def _result(
        self,
        *,
        replanned: bool,
        reason: str,
        path=None,
        exit_position=None,
        cost=None,
        risk=None,
        safe_exit_exists: bool,
        status: str,
        evaluations=None,
    ) -> Dict[str, Any]:
        return {
            "replanned": replanned,
            "reason": reason,
            "path": path,
            "exit": exit_position,
            "cost": cost,
            "steps": None if path is None else len(path) - 1,
            "risk": risk,
            "safe_exit_exists": safe_exit_exists,
            "status": status,
            "evaluations": evaluations or [],
        }

    def _attempt_replan(self, environment, current_position, reason: str):
        new_route = find_best_route(environment, start=current_position)
        if new_route is None:
            return self._result(
                replanned=True,
                reason=reason,
                safe_exit_exists=False,
                status="no_route",
            )

        return self._result(
            replanned=True,
            reason=reason,
            path=new_route["path"],
            exit_position=new_route["exit"],
            cost=new_route["cost"],
            risk=new_route["risk"],
            safe_exit_exists=True,
            status="route_found",
            evaluations=new_route["evaluations"],
        )

    def update(
        self,
        environment,
        current_position: Position,
        existing_route: RouteInput,
        previous_exit: Optional[Position] = None,
    ) -> Dict[str, Any]:
        """Return current route status and replan only when needed.

        ``existing_route`` may be a Phase 2 route-result dictionary or a path.
        The returned path always starts at ``current_position``. The person is
        considered trapped when the environment marks their current cell
        blocked; that case does not call A* from an unsafe cell.
        """
        if not self._valid_position(environment, current_position):
            return self._result(
                replanned=False,
                reason="invalid_current_position",
                safe_exit_exists=False,
                status="trapped",
            )

        current_position = tuple(current_position)
        if environment.is_blocked(current_position):
            return self._result(
                replanned=False,
                reason="current_position_blocked",
                safe_exit_exists=False,
                status="trapped",
            )

        state = self._environment_snapshot(environment)
        environment_changed = state != self._last_environment_state
        self._last_environment_state = state

        path = self._extract_path(existing_route)
        if path is None:
            return self._attempt_replan(environment, current_position, "no_existing_route")

        normalized_path = []
        for position in path:
            if not self._valid_position(environment, position):
                return self._attempt_replan(environment, current_position, "invalid_route")
            normalized_path.append(tuple(position))

        # Check route continuity, including the already-traversed prefix. A
        # blocked cell in that prefix is harmless; only remaining cells matter.
        for first, second in zip(normalized_path, normalized_path[1:]):
            if abs(first[0] - second[0]) + abs(first[1] - second[1]) != 1:
                return self._attempt_replan(environment, current_position, "discontinuous_route")

        try:
            progress_index = normalized_path.index(current_position)
        except ValueError:
            return self._attempt_replan(
                environment, current_position, "current_position_not_on_route"
            )

        remaining_path = normalized_path[progress_index:]
        if any(environment.is_blocked(position) for position in remaining_path):
            return self._attempt_replan(environment, current_position, "remaining_route_blocked")

        try:
            current_cost = calculate_path_cost(environment, remaining_path)
        except ValueError:
            return self._attempt_replan(environment, current_position, "remaining_route_invalid")

        selected_exit = self._route_exit(existing_route, normalized_path, previous_exit)
        current_risk = self._path_risk(environment, remaining_path)

        # With no observed environmental change, retain a valid route and avoid
        # running A* for every exit on every simulation update.
        if not environment_changed:
            return self._result(
                replanned=False,
                reason="environment_unchanged_route_valid",
                path=remaining_path,
                exit_position=selected_exit,
                cost=current_cost,
                risk=current_risk,
                safe_exit_exists=True,
                status="route_valid",
            )

        best_route = find_best_route(environment, start=current_position)
        if best_route is None:
            return self._result(
                replanned=True,
                reason="no_reachable_exit",
                path=None,
                exit_position=None,
                cost=None,
                risk=None,
                safe_exit_exists=False,
                status="no_route",
            )

        best_cost = best_route["cost"]
        required_improvement = max(1e-9, abs(current_cost) * self.improvement_threshold)
        meaningfully_cheaper = best_cost < current_cost - required_improvement
        if meaningfully_cheaper:
            reason = (
                "better_exit_available"
                if best_route["exit"] != selected_exit
                else "lower_cost_route_available"
            )
            return self._result(
                replanned=True,
                reason=reason,
                path=best_route["path"],
                exit_position=best_route["exit"],
                cost=best_route["cost"],
                risk=best_route["risk"],
                safe_exit_exists=True,
                status="route_found",
                evaluations=best_route["evaluations"],
            )

        return self._result(
            replanned=False,
            reason="no_significant_improvement",
            path=remaining_path,
            exit_position=selected_exit,
            cost=current_cost,
            risk=current_risk,
            safe_exit_exists=True,
            status="route_valid",
            evaluations=best_route["evaluations"],
        )
