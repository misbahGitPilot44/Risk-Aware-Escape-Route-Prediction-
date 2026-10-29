"""Tests for controlled fire ignition, spread, and planner integration."""

import unittest

from astar import astar
from config import FIRE, WALL
from environment import BuildingEnvironment
from fire_simulation import FireSimulation
from replanning import ReplanningController
from route_planner import find_best_route


class FireSimulationTests(unittest.TestCase):
    def setUp(self):
        self.environment = BuildingEnvironment()

    def test_fire_can_be_ignited_on_walkable_cell(self):
        simulation = FireSimulation(self.environment, seed=1)
        position = (5, 5)
        simulation.ignite(position, intensity=80)
        self.assertIn(position, self.environment.fire_cells)
        self.assertEqual(self.environment.grid[position], FIRE)
        self.assertEqual(simulation.get_intensity(position), 80.0)

    def test_fire_cannot_be_ignited_inside_permanent_wall(self):
        simulation = FireSimulation(self.environment)
        with self.assertRaises(ValueError):
            simulation.ignite((3, 6))

    def test_invalid_coordinates_and_intensity_are_rejected(self):
        simulation = FireSimulation(self.environment)
        for position in ((-1, 0), (20, 0), (1.5, 2), (1,)):
            with self.subTest(position=position):
                with self.assertRaises(ValueError):
                    simulation.ignite(position)
        for intensity in (0, -1, 100.1, float("nan"), float("inf")):
            with self.subTest(intensity=intensity):
                with self.assertRaises(ValueError):
                    simulation.ignite((5, 5), intensity)

    def test_fire_is_blocked_and_excluded_from_neighbors(self):
        simulation = FireSimulation(self.environment)
        position = (5, 5)
        simulation.ignite(position)
        self.assertTrue(self.environment.is_blocked(position))
        self.assertNotIn(position, self.environment.neighbors((5, 4)))

    def test_spread_does_not_pass_through_permanent_walls(self):
        environment = BuildingEnvironment()
        # Seal the target from every side except across the wall separating it
        # from the source. The western source region has no route around it.
        for wall in (
            (5, 6), (4, 5), (6, 5), (4, 6), (6, 6),
            (4, 7), (6, 7), (5, 8),
        ):
            environment.add_wall(wall)
        simulation = FireSimulation(environment, base_spread_probability=1.0)
        simulation.ignite((5, 5))

        for _ in range(4):
            simulation.step()

        self.assertTrue(environment.is_blocked((5, 6)))
        self.assertEqual(environment.grid[5, 6], WALL)
        self.assertNotIn((5, 7), simulation.get_active_fire_cells())

    def test_fire_does_not_spread_outside_building_boundary(self):
        simulation = FireSimulation(self.environment, base_spread_probability=1.0)
        simulation.ignite((1, 1))
        simulation.step()
        self.assertNotIn((0, 1), simulation.get_active_fire_cells())
        self.assertNotIn((1, 0), simulation.get_active_fire_cells())

    def test_newly_ignited_cells_do_not_spread_again_in_same_step(self):
        simulation = FireSimulation(self.environment, base_spread_probability=1.0)
        simulation.ignite((5, 5))
        simulation.step()
        self.assertIn((4, 5), simulation.get_active_fire_cells())
        self.assertNotIn((3, 5), simulation.get_active_fire_cells())

    def test_full_spread_probability_ignites_all_eligible_neighbors(self):
        simulation = FireSimulation(self.environment, base_spread_probability=1.0)
        source = (5, 5)
        eligible = set(self.environment.neighbors(source))
        simulation.ignite(source, intensity=100)

        newly_ignited = simulation.step()

        self.assertEqual(set(newly_ignited), eligible)
        self.assertTrue(eligible.issubset(set(simulation.get_active_fire_cells())))

    def test_zero_spread_probability_prevents_new_ignition(self):
        simulation = FireSimulation(self.environment, base_spread_probability=0.0)
        simulation.ignite((5, 5))
        self.assertEqual(simulation.step(), [])
        self.assertEqual(simulation.get_active_fire_cells(), ((5, 5),))

    def test_directional_bias_changes_probability_and_can_change_live(self):
        simulation = FireSimulation(
            self.environment,
            base_spread_probability=0.4,
            direction="east",
            directional_bias=2.0,
        )
        self.assertGreater(
            simulation.spread_probability(100, "east"),
            simulation.spread_probability(100, "north"),
        )
        simulation.ignite((5, 5))
        simulation.step()
        simulation.set_direction("north")
        self.assertGreater(
            simulation.spread_probability(100, "north"),
            simulation.spread_probability(100, "east"),
        )
        simulation.set_direction("all")
        self.assertEqual(
            simulation.spread_probability(100, "north"),
            simulation.spread_probability(100, "east"),
        )

    def test_invalid_direction_and_probability_are_rejected(self):
        simulation = FireSimulation(self.environment)
        with self.assertRaises(ValueError):
            simulation.set_direction("north-east")
        for probability in (-0.1, 1.1, float("nan"), float("inf")):
            with self.subTest(probability=probability):
                with self.assertRaises(ValueError):
                    simulation.set_spread_probability(probability)

    def test_multiple_ignition_points_are_supported(self):
        simulation = FireSimulation(self.environment)
        simulation.ignite((5, 5), 40)
        simulation.ignite((10, 10), 90)
        self.assertEqual(simulation.get_active_fire_cells(), ((5, 5), (10, 10)))

    def test_extinguishing_clears_fire_and_reduces_risk(self):
        simulation = FireSimulation(self.environment)
        source = (5, 5)
        simulation.ignite(source)
        before = self.environment.risk_map[4, 5]
        self.assertTrue(simulation.extinguish(source))
        self.assertFalse(self.environment.is_blocked(source))
        self.assertLess(self.environment.risk_map[4, 5], before)
        self.assertFalse(simulation.extinguish(source))

    def test_reset_clears_fire_risk_and_step_count(self):
        simulation = FireSimulation(self.environment, seed=4, base_spread_probability=1.0)
        simulation.ignite((5, 5))
        simulation.step()
        simulation.reset()
        self.assertEqual(simulation.step_count, 0)
        self.assertEqual(simulation.get_active_fire_cells(), ())
        self.assertTrue((self.environment.risk_map == 0).all())
        self.assertTrue((self.environment.grid == FIRE).sum() == 0)

    def test_fixed_seed_repeats_the_same_spread(self):
        first_environment = BuildingEnvironment()
        second_environment = BuildingEnvironment()
        first = FireSimulation(first_environment, seed=123, base_spread_probability=0.45)
        second = FireSimulation(second_environment, seed=123, base_spread_probability=0.45)
        first.ignite((5, 5), 85)
        second.ignite((5, 5), 85)

        for _ in range(5):
            self.assertEqual(first.step(), second.step())

        self.assertEqual(first.get_active_fire_cells(), second.get_active_fire_cells())

    def test_intensity_grows_and_new_fire_gets_configured_intensity(self):
        simulation = FireSimulation(
            self.environment,
            seed=1,
            base_spread_probability=1.0,
            propagated_intensity=60,
            intensity_growth=5,
        )
        simulation.ignite((5, 5), intensity=70)
        simulation.step()
        self.assertEqual(simulation.get_intensity((5, 5)), 75.0)
        self.assertEqual(simulation.get_intensity((4, 5)), 60.0)

    def test_astar_and_exit_planner_avoid_new_fire(self):
        route = find_best_route(self.environment)
        self.assertIsNotNone(route)
        fire_cell = route["path"][2]
        simulation = FireSimulation(self.environment)
        simulation.ignite(fire_cell)

        path = astar(self.environment, self.environment.start, route["exit"])
        updated = find_best_route(self.environment)

        self.assertNotIn(fire_cell, path)
        self.assertIsNotNone(updated)
        self.assertEqual(len(updated["evaluations"]), len(self.environment.exits))

    def test_replanning_controller_detects_fire_spread_on_old_route(self):
        route = find_best_route(self.environment)
        self.assertIsNotNone(route)
        controller = ReplanningController()
        current_position = route["path"][1]
        controller.update(self.environment, current_position, route)
        simulation = FireSimulation(self.environment, base_spread_probability=1.0)
        simulation.ignite(route["path"][3])
        simulation.step()

        result = controller.update(self.environment, current_position, route)

        self.assertTrue(result["replanned"])
        self.assertEqual(result["reason"], "remaining_route_blocked")
        self.assertEqual(result["path"][0], current_position)
        self.assertNotIn(route["path"][2], result["path"])


if __name__ == "__main__":
    unittest.main()
