"""Seedable, controllable four-directional fire spread on a building grid."""

from math import isfinite
from numbers import Integral
from typing import Dict, Optional, Tuple

import numpy as np

from config import DYNAMIC_BLOCKED, FIRE, MAX_RISK, MIN_RISK, WALL
from risk_model import RiskModel

Position = Tuple[int, int]
_DIRECTIONS = {
    "north": (-1, 0),
    "south": (1, 0),
    "west": (0, -1),
    "east": (0, 1),
}


class FireSimulation:
    """Manage fire intensity and spread while the environment owns the grid.

    Spread probability for a source cell is
    ``base_probability * intensity / 100``. A neighbor in the selected
    direction gets that value multiplied by ``directional_bias`` and capped
    at 1. Independent chances from multiple burning neighbors combine as
    ``1 - product(1 - p_i)``. Each target is sampled at most once per step.
    """

    def __init__(
        self,
        environment,
        seed: Optional[int] = None,
        base_spread_probability: float = 0.35,
        direction: str = "all",
        directional_bias: float = 2.0,
        propagated_intensity: float = 60.0,
        intensity_growth: float = 5.0,
    ):
        self.environment = environment
        self.seed = seed
        self.risk_model = RiskModel(environment)
        self._rng = np.random.default_rng(seed)
        self._intensities: Dict[Position, float] = {}
        self.step_count = 0
        self.directional_bias = self._validate_nonnegative(
            directional_bias, "directional_bias", minimum=1.0
        )
        self.propagated_intensity = self._validate_intensity(propagated_intensity)
        self.intensity_growth = self._validate_nonnegative(
            intensity_growth, "intensity_growth", maximum=MAX_RISK
        )
        self._direction = "all"
        self.set_spread_probability(base_spread_probability)
        self.set_direction(direction)
        self._sync_active_fire()
        self._update_risk()

    @staticmethod
    def _normalize_position(environment, position) -> Position:
        if not isinstance(position, (tuple, list)) or len(position) != 2:
            raise ValueError("Positions must be (row, column) pairs")
        if any(isinstance(value, bool) or not isinstance(value, Integral)
               for value in position):
            raise ValueError("Position coordinates must be integers")
        cell = int(position[0]), int(position[1])
        if not environment.in_bounds(cell):
            raise ValueError(f"Position {cell} is outside the grid")
        return cell

    @staticmethod
    def _validate_intensity(value) -> float:
        try:
            intensity = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError("Intensity must be a finite value in (0, 100]") from error
        if not isfinite(intensity) or not MIN_RISK < intensity <= MAX_RISK:
            raise ValueError("Intensity must be a finite value in (0, 100]")
        return intensity

    @staticmethod
    def _validate_nonnegative(value, name: str, minimum: float = 0.0,
                              maximum: float = float("inf")) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must be a finite value in range") from error
        if not isfinite(number) or not minimum <= number <= maximum:
            raise ValueError(f"{name} must be a finite value in range")
        return number

    @property
    def direction(self) -> str:
        return self._direction

    def set_direction(self, direction: str) -> None:
        """Set spread bias to north, south, east, west, or omnidirectional."""
        if not isinstance(direction, str):
            raise ValueError("Direction must be north, south, east, west, or all")
        normalized = direction.lower()
        if normalized not in (*_DIRECTIONS, "all"):
            raise ValueError("Direction must be north, south, east, west, or all")
        self._direction = normalized

    def set_spread_probability(self, probability: float) -> None:
        """Set the base probability per eligible neighbor, from 0 to 1."""
        try:
            value = float(probability)
        except (TypeError, ValueError) as error:
            raise ValueError("Spread probability must be finite and between 0 and 1") from error
        if not isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("Spread probability must be finite and between 0 and 1")
        self.base_spread_probability = value

    def spread_probability(self, intensity: float, move_direction: str) -> float:
        """Return the configured chance for one source-to-neighbor direction."""
        source_intensity = self._validate_intensity(intensity)
        if not isinstance(move_direction, str) or move_direction not in _DIRECTIONS:
            raise ValueError("Move direction must be north, south, east, or west")
        probability = self.base_spread_probability * (source_intensity / MAX_RISK)
        if self._direction != "all" and move_direction == self._direction:
            probability *= self.directional_bias
        return min(1.0, probability)

    def _sync_active_fire(self) -> None:
        """Mirror environment fire changes while preserving known intensities."""
        active = {tuple(position) for position in self.environment.fire_cells}
        for position in active - self._intensities.keys():
            self._intensities[position] = float(MAX_RISK)
        for position in self._intensities.keys() - active:
            del self._intensities[position]

    def _update_risk(self) -> None:
        self.risk_model.update(self._intensities)

    def _validate_ignition_cell(self, position) -> Position:
        cell = self._normalize_position(self.environment, position)
        if cell in self.environment.fire_cells:
            return cell
        if self.environment.grid[cell] in (WALL, DYNAMIC_BLOCKED):
            raise ValueError(f"Cannot ignite blocked cell {cell}")
        if self.environment.is_blocked(cell):
            raise ValueError(f"Cannot ignite blocked cell {cell}")
        return cell

    def ignite(self, position, intensity: float = 100.0) -> Position:
        """Ignite one valid cell and immediately update surrounding risk."""
        cell = self._validate_ignition_cell(position)
        value = self._validate_intensity(intensity)
        if cell not in self._intensities:
            self.environment.set_fire(cell, active=True)
            self._intensities[cell] = value
        else:
            self._intensities[cell] = max(self._intensities[cell], value)
        self._update_risk()
        return cell

    @staticmethod
    def _move_direction(source: Position, destination: Position) -> str:
        delta = (destination[0] - source[0], destination[1] - source[1])
        for direction, direction_delta in _DIRECTIONS.items():
            if delta == direction_delta:
                return direction
        raise ValueError("Fire spread only supports four-directional neighbors")

    def step(self):
        """Advance one synchronous spread step and return newly burning cells."""
        self._sync_active_fire()
        burning_at_start = sorted(self._intensities)

        # Existing fires grow first. Newly ignited cells are held separately
        # until all spread decisions are complete, so they cannot spread twice.
        for cell in burning_at_start:
            self._intensities[cell] = min(
                float(MAX_RISK), self._intensities[cell] + self.intensity_growth
            )

        source_chances: Dict[Position, list] = {}
        for source in burning_at_start:
            intensity = self._intensities[source]
            for target in self.environment.neighbors(source):
                target = tuple(target)
                if target in self._intensities:
                    continue
                direction = self._move_direction(source, target)
                chance = self.spread_probability(intensity, direction)
                source_chances.setdefault(target, []).append(chance)

        newly_burning = []
        for target in sorted(source_chances):
            combined_probability = 1.0
            for chance in source_chances[target]:
                combined_probability *= 1.0 - chance
            combined_probability = 1.0 - combined_probability
            if combined_probability >= 1.0 or self._rng.random() < combined_probability:
                newly_burning.append(target)

        for cell in newly_burning:
            self.environment.set_fire(cell, active=True)
            self._intensities[cell] = self.propagated_intensity

        self.step_count += 1
        self._update_risk()
        return newly_burning

    def extinguish(self, position) -> bool:
        """Extinguish one active fire and recalculate risk so it can decline."""
        cell = self._normalize_position(self.environment, position)
        self._sync_active_fire()
        if cell not in self._intensities:
            return False
        del self._intensities[cell]
        self.environment.set_fire(cell, active=False)
        self._update_risk()
        return True

    def reset(self) -> None:
        """Clear fire managed by the simulator, risks, steps, and RNG sequence."""
        self._sync_active_fire()
        for cell in sorted(self._intensities):
            self.environment.set_fire(cell, active=False)
        self._intensities.clear()
        self.step_count = 0
        self._rng = np.random.default_rng(self.seed)
        self._update_risk()

    def get_active_fire_cells(self):
        """Return a stable tuple of currently active environment fire cells."""
        self._sync_active_fire()
        return tuple(sorted(self._intensities))

    def get_intensity(self, position) -> float:
        """Return active intensity at position, or zero when it is not burning."""
        cell = self._normalize_position(self.environment, position)
        self._sync_active_fire()
        return self._intensities.get(cell, 0.0)
