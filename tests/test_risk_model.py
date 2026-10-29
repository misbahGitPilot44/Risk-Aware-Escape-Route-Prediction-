"""Tests for the fire-distance risk model."""

import unittest

from config import MAX_RISK, WALL
from environment import BuildingEnvironment
from risk_model import RiskModel


class RiskModelTests(unittest.TestCase):
    def setUp(self):
        self.environment = BuildingEnvironment()
        self.model = RiskModel(self.environment)
        self.fire = (5, 5)
        self.environment.set_fire(self.fire)

    def test_burning_cell_has_maximum_risk(self):
        self.model.update()
        self.assertEqual(self.environment.risk_map[self.fire], MAX_RISK)

    def test_adjacent_cells_have_higher_risk_than_distant_cells(self):
        self.model.update({self.fire: 80})
        adjacent = self.environment.risk_map[4, 5]
        distant = self.environment.risk_map[2, 5]
        self.assertGreater(adjacent, distant)
        self.assertEqual(adjacent, 40.0)
        self.assertAlmostEqual(distant, 80.0 / 4.0)

    def test_higher_intensity_produces_higher_or_equal_risk(self):
        low = self.model.calculate_risk((4, 5), {self.fire: 20})
        high = self.model.calculate_risk((4, 5), {self.fire: 90})
        self.assertGreater(high, low)

    def test_risk_is_clamped_to_zero_to_one_hundred(self):
        self.model.update({self.fire: 100})
        self.assertGreaterEqual(float(self.environment.risk_map.min()), 0.0)
        self.assertLessEqual(float(self.environment.risk_map.max()), 100.0)

    def test_walls_remain_blocked_after_risk_recalculation(self):
        wall = (3, 6)
        self.assertEqual(self.environment.grid[wall], WALL)
        self.model.update()
        self.assertTrue(self.environment.is_blocked(wall))
        self.assertEqual(self.environment.risk_map[wall], 0.0)

    def test_risk_decreases_after_fire_source_is_removed(self):
        self.model.update({self.fire: 100})
        before = self.environment.risk_map[4, 5]
        self.environment.set_fire(self.fire, active=False)
        self.model.update({})
        self.assertGreater(before, self.environment.risk_map[4, 5])
        self.assertEqual(self.environment.risk_map[4, 5], 0.0)

    def test_updates_shared_environment_risk_map(self):
        self.model.update({self.fire: 60})
        self.assertEqual(self.environment.risk_map[self.fire], 100.0)
        self.assertEqual(self.environment.risk_map[4, 5], 30.0)


if __name__ == "__main__":
    unittest.main()
