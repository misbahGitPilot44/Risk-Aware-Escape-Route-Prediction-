"""Small set of run statistics for the evacuation simulation."""

from dataclasses import asdict, dataclass
from typing import Optional, Tuple


@dataclass
class SimulationMetrics:
    """Counters and final route details collected during one run."""

    elapsed_time: float = 0.0
    movement_steps: int = 0
    replanning_events: int = 0
    initial_route_length: int = 0
    final_route_length: int = 0
    fire_steps: int = 0
    selected_exit: Optional[Tuple[int, int]] = None
    final_status: str = "READY"
    total_weighted_route_cost: Optional[float] = None

    def reset(self, route=None) -> None:
        """Clear counters and seed route fields for a fresh simulation run."""
        self.elapsed_time = 0.0
        self.movement_steps = 0
        self.replanning_events = 0
        self.initial_route_length = 0 if route is None else len(route["path"]) - 1
        self.final_route_length = self.initial_route_length
        self.fire_steps = 0
        self.selected_exit = None if route is None else tuple(route["exit"])
        self.final_status = "READY" if route is not None else "NO_ROUTE"
        self.total_weighted_route_cost = None if route is None else route["cost"]

    def as_dict(self) -> dict:
        """Return a plain dictionary suitable for terminal output or a UI."""
        return asdict(self)
