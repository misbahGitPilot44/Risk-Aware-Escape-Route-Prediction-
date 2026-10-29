"""Fire-related cell risk calculations for the shared environment risk map."""

from math import isfinite
from numbers import Integral
from typing import Mapping, Optional, Tuple

from config import FIRE, MAX_RISK, MIN_RISK, WALL

Position = Tuple[int, int]


class RiskModel:
    """Recalculate fire risk directly into ``environment.risk_map``.

    A burning cell has risk 100. For other non-wall cells, each fire source
    contributes ``intensity / (Manhattan distance + 1)``; the cell takes the
    maximum contribution. Risk is therefore strongest beside a fire and
    decreases predictably with distance. This simple model is for simulation,
    not a physical prediction of fire or smoke.
    """

    def __init__(self, environment):
        self.environment = environment

    def _position(self, position) -> Position:
        if not isinstance(position, (tuple, list)) or len(position) != 2:
            raise ValueError("Positions must be (row, column) pairs")
        if any(isinstance(value, bool) or not isinstance(value, Integral)
               for value in position):
            raise ValueError("Position coordinates must be integers")
        normalized = int(position[0]), int(position[1])
        if not self.environment.in_bounds(normalized):
            raise ValueError(f"Position {normalized} is outside the grid")
        return normalized

    def _normalize_sources(self, fire_intensities: Optional[Mapping] = None) -> dict:
        if fire_intensities is None:
            fire_intensities = {
                tuple(position): float(MAX_RISK)
                for position in self.environment.fire_cells
            }

        sources = {}
        for position, raw_intensity in fire_intensities.items():
            cell = self._position(position)
            try:
                intensity = float(raw_intensity)
            except (TypeError, ValueError) as error:
                raise ValueError("Fire intensity must be a finite value from 0 to 100") from error
            if not isfinite(intensity) or not MIN_RISK <= intensity <= MAX_RISK:
                raise ValueError("Fire intensity must be a finite value from 0 to 100")
            if intensity > 0:
                sources[cell] = intensity
        return sources

    def calculate_risk(
        self,
        position: Position,
        fire_intensities: Optional[Mapping] = None,
    ) -> float:
        """Calculate one cell's risk from the supplied or active fire sources."""
        cell = self._position(position)
        sources = self._normalize_sources(fire_intensities)
        return self._calculate_from_sources(cell, sources)

    def _calculate_from_sources(self, cell: Position, sources: dict) -> float:
        row, col = cell

        if self.environment.grid[row, col] == WALL:
            return 0.0
        if self.environment.grid[row, col] == FIRE or cell in sources:
            return float(MAX_RISK)

        contributions = []
        for (fire_row, fire_col), intensity in sources.items():
            distance = abs(row - fire_row) + abs(col - fire_col)
            contributions.append(intensity / (distance + 1))
        return min(float(MAX_RISK), max(contributions, default=0.0))

    def update(self, fire_intensities: Optional[Mapping] = None) -> None:
        """Recalculate fire risk and write values via ``environment.set_risk``."""
        sources = self._normalize_sources(fire_intensities)
        for row in range(self.environment.rows):
            for col in range(self.environment.cols):
                position = (row, col)
                risk = self._calculate_from_sources(position, sources)
                self.environment.set_risk(position, risk)
