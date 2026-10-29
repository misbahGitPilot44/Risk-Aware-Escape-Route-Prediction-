"""Unit tests for the configurable deterministic building environment."""

import unittest

import numpy as np

from astar import astar
from config import DYNAMIC_BLOCKED, EMPTY, EXIT, FIRE, WALL
from environment import BuildingEnvironment, Environment
from replanning import ReplanningController
from route_planner import find_best_route


class BuildingEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.environment = BuildingEnvironment()

    def test_default_grid_dimensions(self):
        self.assertEqual(self.environment.grid.shape, (20, 20))
        self.assertEqual(self.environment.risk_map.shape, (20, 20))

    def test_default_start_is_valid_and_walkable(self):
        self.assertEqual(self.environment.start, (1, 1))
        self.assertTrue(self.environment.in_bounds(self.environment.start))
        self.assertFalse(self.environment.is_blocked(self.environment.start))

    def test_all_default_exits_are_valid_and_walkable(self):
        self.assertEqual(
            self.environment.exits,
            [(1, 18), (18, 1), (18, 18)],
        )
        for exit_position in self.environment.exits:
            self.assertTrue(self.environment.in_bounds(exit_position))
            self.assertFalse(self.environment.is_blocked(exit_position))
            self.assertEqual(self.environment.grid[exit_position], EXIT)

    def test_outer_boundaries_are_blocked_and_protected(self):
        for position in ((0, 1), (19, 1), (1, 0), (1, 19)):
            self.assertTrue(self.environment.is_blocked(position))
            self.assertFalse(self.environment.remove_wall(position))

    def test_permanent_interior_wall_is_blocked_and_protected(self):
        position = (3, 6)
        self.assertEqual(self.environment.grid[position], WALL)
        self.assertTrue(self.environment.is_blocked(position))
        self.assertFalse(self.environment.remove_wall(position))

    def test_walkable_floor_is_accessible(self):
        self.assertEqual(self.environment.grid[2, 2], EMPTY)
        self.assertFalse(self.environment.is_blocked((2, 2)))

    def test_neighbors_are_in_bounds_and_unblocked(self):
        for position in self.environment.neighbors((1, 1)):
            self.assertTrue(self.environment.in_bounds(position))
            self.assertFalse(self.environment.is_blocked(position))
        self.assertEqual(set(self.environment.neighbors((1, 1))), {(2, 1), (1, 2)})

    def test_out_of_bounds_positions_are_rejected(self):
        self.assertFalse(self.environment.in_bounds((-1, 0)))
        self.assertFalse(self.environment.in_bounds((20, 0)))
        self.assertTrue(self.environment.is_blocked((20, 0)))
        self.assertEqual(self.environment.neighbors((-1, 0)), [])

    def test_wall_can_be_added(self):
        self.environment.add_wall((2, 2))
        self.assertEqual(self.environment.grid[2, 2], WALL)
        self.assertTrue(self.environment.is_blocked((2, 2)))

    def test_removable_wall_can_be_removed(self):
        self.environment.add_wall((2, 2))
        self.assertTrue(self.environment.remove_wall((2, 2)))
        self.assertFalse(self.environment.is_blocked((2, 2)))
        self.assertEqual(self.environment.grid[2, 2], EMPTY)

    def test_boundary_and_protected_cells_cannot_be_walled(self):
        with self.assertRaises(ValueError):
            self.environment.add_wall((0, 2))
        with self.assertRaises(ValueError):
            self.environment.add_wall(self.environment.start)
        with self.assertRaises(ValueError):
            self.environment.add_wall(self.environment.exits[0])

    def test_fire_is_blocked_and_can_be_cleared(self):
        position = (2, 2)
        self.environment.set_fire(position)
        self.assertIn(position, self.environment.fire_cells)
        self.assertEqual(self.environment.grid[position], FIRE)
        self.assertTrue(self.environment.is_blocked(position))
        self.environment.set_fire(position, active=False)
        self.assertNotIn(position, self.environment.fire_cells)
        self.assertEqual(self.environment.grid[position], EMPTY)
        self.assertFalse(self.environment.is_blocked(position))

    def test_fire_clear_restores_exit_cell_type(self):
        position = self.environment.exits[0]
        self.environment.set_fire(position)
        self.assertTrue(self.environment.is_blocked(position))
        self.environment.set_fire(position, active=False)
        self.assertEqual(self.environment.grid[position], EXIT)
        self.assertFalse(self.environment.is_blocked(position))

    def test_dynamic_blockage_is_blocked_and_clearable(self):
        position = (2, 2)
        self.environment.add_blocked(position)
        self.assertEqual(self.environment.grid[position], DYNAMIC_BLOCKED)
        self.assertTrue(self.environment.is_blocked(position))
        self.assertNotIn(position, self.environment.neighbors((2, 1)))
        self.environment.remove_blocked(position)
        self.assertFalse(self.environment.is_blocked(position))

    def test_risk_values_update_and_share_grid_shape(self):
        self.environment.set_risk((2, 2), 42.5)
        self.assertEqual(self.environment.risk_map[2, 2], 42.5)
        self.assertEqual(self.environment.grid.shape, self.environment.risk_map.shape)

    def test_invalid_risk_values_are_rejected(self):
        for value in (-1, 100.01, float("nan"), float("inf"), "invalid"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    self.environment.set_risk((2, 2), value)

    def test_reset_restores_start_exits_walls_hazards_and_risk(self):
        original_grid = self.environment.grid.copy()
        self.environment.add_wall((2, 2))
        self.environment.set_fire((2, 3))
        self.environment.add_blocked((2, 4))
        self.environment.set_risk((2, 5), 75)
        self.environment.start = (2, 2)
        self.environment.exits.append((3, 3))

        self.environment.reset()

        self.assertEqual(self.environment.start, (1, 1))
        self.assertEqual(self.environment.exits, [(1, 18), (18, 1), (18, 18)])
        np.testing.assert_array_equal(self.environment.grid, original_grid)
        self.assertTrue(np.all(self.environment.risk_map == 0))
        self.assertFalse(self.environment.fire_cells)
        self.assertFalse(self.environment.dynamic_blocked_cells)
        self.assertFalse(self.environment.removable_walls)

    def test_custom_dimensions_and_positions_are_supported(self):
        environment = BuildingEnvironment(
            rows=8,
            cols=10,
            start=(1, 1),
            exits=[(1, 8), (6, 1), (6, 8)],
        )
        self.assertEqual(environment.grid.shape, (8, 10))
        self.assertEqual(environment.risk_map.shape, (8, 10))
        self.assertGreater(len(environment.permanent_walls), 2 * (8 + 10) - 4)

    def test_constructor_rejects_invalid_start_and_exits(self):
        with self.assertRaises(ValueError):
            BuildingEnvironment(start=(0, 0))
        with self.assertRaises(ValueError):
            BuildingEnvironment(exits=[(0, 1)])

    def test_existing_environment_name_remains_compatible(self):
        self.assertIs(Environment, BuildingEnvironment)
        self.assertEqual(Environment().grid.shape, (20, 20))

    def test_default_start_reaches_an_exit_with_existing_astar(self):
        reachable = [
            astar(self.environment, self.environment.start, exit_position)
            for exit_position in self.environment.exits
        ]
        self.assertTrue(any(path is not None for path in reachable))

    def test_multiple_exit_planner_works_with_building_environment(self):
        result = find_best_route(self.environment)
        self.assertIsNotNone(result)
        self.assertIn(result["exit"], self.environment.exits)
        self.assertEqual(len(result["evaluations"]), len(self.environment.exits))

    def test_replanning_controller_detects_environment_fire_update(self):
        route = find_best_route(self.environment)
        self.assertIsNotNone(route)
        controller = ReplanningController()
        controller.update(self.environment, self.environment.start, route)
        blocked_next = route["path"][1]
        self.environment.set_fire(blocked_next)

        result = controller.update(self.environment, self.environment.start, route)

        self.assertTrue(result["replanned"])
        self.assertNotIn(blocked_next, result["path"])
        self.assertEqual(result["path"][0], self.environment.start)


if __name__ == "__main__":
    unittest.main()
