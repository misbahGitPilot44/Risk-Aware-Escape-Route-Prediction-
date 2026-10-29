"""Testable simulation controller connecting movement, fire, and replanning."""

from math import isfinite

from astar import calculate_path_cost
from config import FIRE_INTERVAL_MS, PLAYER_MOVE_INTERVAL_MS
from environment import BuildingEnvironment
from fire_simulation import FireSimulation
from metrics import SimulationMetrics
from replanning import ReplanningController
from route_planner import find_best_route


class EvacuationSimulation:
    """Advance one evacuation on one shared environment instance.

    Each call to :meth:`update` handles at most one fire step and one player
    move, even if its delta is large. Remaining elapsed time is retained for
    later updates, avoiding accidental multi-cell jumps in a rendered frame.
    """

    READY = "READY"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    EVACUATING = "EVACUATING"
    REPLANNING = "REPLANNING"
    ESCAPED = "ESCAPED"
    TRAPPED = "TRAPPED"
    NO_ROUTE = "NO_ROUTE"

    def __init__(
        self,
        environment=None,
        fire_simulation=None,
        replanning_controller=None,
        player_move_interval: float = PLAYER_MOVE_INTERVAL_MS / 1000.0,
        fire_step_interval: float = FIRE_INTERVAL_MS / 1000.0,
        seed=None,
        fire_spread_probability: float = 0.35,
    ):
        self.environment = environment or BuildingEnvironment()
        self.player_move_interval = self._validate_interval(
            player_move_interval, "player_move_interval"
        )
        self.fire_step_interval = self._validate_interval(
            fire_step_interval, "fire_step_interval"
        )
        self.fire_simulation = fire_simulation or FireSimulation(
            self.environment,
            seed=seed,
            base_spread_probability=fire_spread_probability,
        )
        if self.fire_simulation.environment is not self.environment:
            raise ValueError("FireSimulation must use the same environment instance")
        self._initial_fire_direction = self.fire_simulation.direction
        self._initial_spread_probability = self.fire_simulation.base_spread_probability
        self.risk_model = self.fire_simulation.risk_model
        self.replanning_controller = replanning_controller or ReplanningController()
        self._replanning_threshold = self.replanning_controller.improvement_threshold
        self.metrics = SimulationMetrics()
        self._running = False
        self._fire_elapsed = 0.0
        self._movement_elapsed = 0.0
        self._replanning_notice = 0.0
        self.last_message = ""
        self.last_replan_reason = ""
        self._initialize_run()

    @staticmethod
    def _validate_interval(value, name: str) -> float:
        try:
            interval = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must be a finite positive number") from error
        if not isfinite(interval) or interval <= 0:
            raise ValueError(f"{name} must be a finite positive number")
        return interval

    def _initialize_run(self) -> None:
        self.player_position = tuple(self.environment.start)
        self.current_route = find_best_route(
            self.environment, start=self.player_position
        )
        self.selected_exit = (
            None if self.current_route is None else self.current_route["exit"]
        )
        self.status = self.READY if self.current_route is not None else self.NO_ROUTE
        if self.player_position in self.environment.exits:
            self.status = self.ESCAPED
        if self.environment.is_blocked(self.player_position):
            self.status = self.TRAPPED
        self.initial_route_length = (
            0 if self.current_route is None else self.current_route["steps"]
        )
        self._running = False
        self._fire_elapsed = 0.0
        self._movement_elapsed = 0.0
        self._replanning_notice = 0.0
        self.last_message = ""
        self.last_replan_reason = ""
        self.metrics.reset(self.current_route)
        self.metrics.final_status = self.status

    def start(self) -> None:
        """Start a ready run, or resume it if it is paused."""
        if self.status == self.PAUSED:
            self.resume()
            return
        if self.status in (self.ESCAPED, self.TRAPPED, self.NO_ROUTE):
            return
        self._running = True
        self.status = self.RUNNING

    def pause(self) -> None:
        """Pause simulation time while leaving the UI/event loop responsive."""
        if self._running:
            self._running = False
            self.status = self.PAUSED

    def resume(self) -> None:
        """Resume a paused run without resetting accumulated timers."""
        if self.status == self.PAUSED and self.current_route is not None:
            self._running = True
            self.status = self.EVACUATING if self.metrics.movement_steps else self.RUNNING

    def _finish(self, status: str) -> None:
        self._running = False
        self.status = status
        self.metrics.final_status = status

    def _sync_metrics(self) -> None:
        self.metrics.final_status = self.status
        self.metrics.fire_steps = self.fire_simulation.step_count
        self.metrics.selected_exit = self.selected_exit

    def _store_route_check(self, check_result) -> None:
        if check_result["status"] == "trapped":
            self._finish(self.TRAPPED)
            self.last_message = "Current position is blocked; evacuation stopped."
            return
        if check_result["status"] == "no_route" or not check_result["safe_exit_exists"]:
            self._finish(self.NO_ROUTE)
            self.last_message = "No safe reachable exit remains."
            return

        self.current_route = {
            "path": check_result["path"],
            "exit": check_result["exit"],
            "cost": check_result["cost"],
            "steps": check_result["steps"],
            "risk": check_result["risk"],
        }
        self.selected_exit = check_result["exit"]
        if check_result["replanned"] or check_result["evaluations"]:
            self.metrics.final_route_length = check_result["steps"]
            self.metrics.total_weighted_route_cost = check_result["cost"]
        if check_result["replanned"]:
            self.metrics.replanning_events += 1
            self.last_replan_reason = check_result["reason"]
            self.last_message = f"Route recalculated: {check_result['reason']}"
            self._replanning_notice = 0.75
            self.status = self.REPLANNING

    def _move_one_cell(self) -> None:
        if self.current_route is None:
            self._finish(self.NO_ROUTE)
            return
        path = self.current_route["path"]
        if not path or tuple(path[0]) != self.player_position:
            self._finish(self.NO_ROUTE)
            self.last_message = "The active route is invalid; movement stopped."
            return
        if len(path) == 1:
            if self.player_position in self.environment.exits:
                self._finish(self.ESCAPED)
                self.last_message = "Evacuation successful."
            else:
                self._finish(self.NO_ROUTE)
            return

        next_position = tuple(path[1])
        if (
            not self.environment.in_bounds(next_position)
            or self.environment.is_blocked(next_position)
        ):
            # Do not move into a newly blocked cell. The next update will
            # replan from this current position before movement resumes.
            self._fire_elapsed = min(self._fire_elapsed, self.fire_step_interval)
            self._movement_elapsed = min(self._movement_elapsed, self.player_move_interval)
            self.last_message = "Next cell is blocked; waiting for a safe route."
            return

        self.player_position = next_position
        self.metrics.movement_steps += 1
        remaining = path[1:]
        self.current_route = {
            **self.current_route,
            "path": remaining,
            "steps": len(remaining) - 1,
            "cost": calculate_path_cost(self.environment, remaining),
            "risk": sum(
                float(self.environment.risk_map[row, col])
                for row, col in remaining[1:]
            ),
        }

        if self.player_position in self.environment.exits:
            self._finish(self.ESCAPED)
            self.last_message = "Evacuation successful."
        elif self.status != self.REPLANNING:
            self.status = self.EVACUATING

    def update(self, delta_time: float) -> None:
        """Advance clocks and process at most one fire step and one move."""
        try:
            delta = float(delta_time)
        except (TypeError, ValueError) as error:
            raise ValueError("delta_time must be a finite non-negative number") from error
        if not isfinite(delta) or delta < 0:
            raise ValueError("delta_time must be a finite non-negative number")
        if not self._running or self.status in (self.ESCAPED, self.TRAPPED, self.NO_ROUTE):
            return

        self.metrics.elapsed_time += delta
        self._fire_elapsed += delta
        self._movement_elapsed += delta
        self._replanning_notice = max(0.0, self._replanning_notice - delta)

        if self._fire_elapsed + 1e-9 >= self.fire_step_interval:
            self.fire_simulation.step()
            self._fire_elapsed = max(0.0, self._fire_elapsed - self.fire_step_interval)
            self.metrics.fire_steps = self.fire_simulation.step_count

        if self.environment.is_blocked(self.player_position):
            self._finish(self.TRAPPED)
            self.last_message = "Fire or an obstacle has blocked the current position."
            self._sync_metrics()
            return

        check_result = self.replanning_controller.update(
            self.environment,
            self.player_position,
            self.current_route,
            self.selected_exit,
        )
        self._store_route_check(check_result)
        if self.status in (self.TRAPPED, self.NO_ROUTE):
            self._sync_metrics()
            return

        if self._movement_elapsed + 1e-9 >= self.player_move_interval:
            self._movement_elapsed = max(
                0.0, self._movement_elapsed - self.player_move_interval
            )
            self._move_one_cell()

        if self.status == self.REPLANNING and self._replanning_notice <= 0:
            self.status = self.EVACUATING if self.metrics.movement_steps else self.RUNNING
        elif self.status not in (self.REPLANNING, self.ESCAPED, self.TRAPPED, self.NO_ROUTE):
            self.status = self.EVACUATING if self.metrics.movement_steps else self.RUNNING
        self._sync_metrics()

    def step_once(self) -> None:
        """While paused, advance one logical tick (at most one fire step/move)."""
        if self.status != self.PAUSED:
            return
        self._running = True
        self.status = self.RUNNING
        self.update(max(self.player_move_interval, self.fire_step_interval))
        if self.status not in (self.ESCAPED, self.TRAPPED, self.NO_ROUTE):
            self._running = False
            self.status = self.PAUSED
            self.metrics.final_status = self.PAUSED

    def ignite(self, position, intensity: float = 100.0):
        """Manually ignite a cell; route changes are checked on the next update."""
        return self.fire_simulation.ignite(position, intensity)

    def set_fire_direction(self, direction: str) -> None:
        """Change the direction bias used for future spread steps."""
        self.fire_simulation.set_direction(direction)

    def reset(self) -> None:
        """Restore the environment and every controller/timer to READY state."""
        self.environment.reset()
        self.fire_simulation.reset()
        self.fire_simulation.set_direction(self._initial_fire_direction)
        self.fire_simulation.set_spread_probability(self._initial_spread_probability)
        self.replanning_controller = ReplanningController(self._replanning_threshold)
        self._initialize_run()

    def summary(self) -> dict:
        """Return the current run metrics as a serializable dictionary."""
        self._sync_metrics()
        return self.metrics.as_dict()
