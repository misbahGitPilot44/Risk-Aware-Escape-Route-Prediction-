"""Headless tests for simulation timing, movement, replanning, and reset."""

import unittest

from config import FIRE
from environment import BuildingEnvironment
from simulation import EvacuationSimulation


class EvacuationSimulationTests(unittest.TestCase):
    def make_simulation(self, environment=None, **kwargs):
        return EvacuationSimulation(
            environment=environment or BuildingEnvironment(),
            seed=7,
            fire_spread_probability=0.0,
            **kwargs,
        )

    def test_simulation_initializes_player_and_route(self):
        simulation = self.make_simulation()
        self.assertEqual(simulation.status, simulation.READY)
        self.assertEqual(simulation.player_position, simulation.environment.start)
        self.assertIsNotNone(simulation.current_route)
        self.assertIn(simulation.selected_exit, simulation.environment.exits)

    def test_player_moves_one_cell_along_route(self):
        simulation = self.make_simulation()
        first_step = simulation.current_route["path"][1]
        simulation.start()
        simulation.update(simulation.player_move_interval)
        self.assertEqual(simulation.player_position, first_step)
        self.assertEqual(simulation.metrics.movement_steps, 1)

    def test_player_never_moves_into_a_wall(self):
        simulation = self.make_simulation()
        blocked = simulation.current_route["path"][1]
        simulation.environment.add_wall(blocked)
        simulation.start()
        simulation.update(simulation.player_move_interval)
        self.assertNotEqual(simulation.player_position, blocked)
        self.assertFalse(simulation.environment.is_blocked(simulation.player_position))

    def test_player_never_moves_into_fire(self):
        simulation = self.make_simulation()
        blocked = simulation.current_route["path"][1]
        simulation.ignite(blocked)
        simulation.start()
        simulation.update(simulation.player_move_interval)
        self.assertNotEqual(simulation.player_position, blocked)
        self.assertEqual(simulation.environment.grid[blocked], FIRE)

    def test_fire_and_movement_use_independent_intervals(self):
        simulation = self.make_simulation()
        simulation.ignite((5, 5))
        simulation.start()
        simulation.update(0.3)
        self.assertEqual(simulation.metrics.movement_steps, 1)
        self.assertEqual(simulation.fire_simulation.step_count, 0)
        simulation.update(0.6)
        self.assertEqual(simulation.metrics.movement_steps, 2)
        self.assertEqual(simulation.fire_simulation.step_count, 0)
        simulation.update(0.1)
        self.assertEqual(simulation.metrics.movement_steps, 3)
        self.assertEqual(simulation.fire_simulation.step_count, 1)

    def test_fire_advances_only_when_its_interval_elapses(self):
        simulation = self.make_simulation()
        simulation.ignite((5, 5))
        simulation.start()
        simulation.update(0.99)
        self.assertEqual(simulation.fire_simulation.step_count, 0)
        simulation.update(0.01)
        self.assertEqual(simulation.fire_simulation.step_count, 1)

    def test_large_delta_never_causes_multiple_moves_in_one_update(self):
        simulation = self.make_simulation()
        simulation.start()
        simulation.update(2.0)
        self.assertEqual(simulation.metrics.movement_steps, 1)
        simulation.update(0.0)
        self.assertEqual(simulation.metrics.movement_steps, 2)

    def test_blocked_route_triggers_replanning_from_current_position(self):
        simulation = self.make_simulation()
        current = simulation.player_position
        blocked = simulation.current_route["path"][1]
        simulation.environment.set_fire(blocked)
        simulation.start()
        simulation.update(0.0)
        self.assertEqual(simulation.metrics.replanning_events, 1)
        self.assertEqual(simulation.current_route["path"][0], current)
        self.assertNotIn(blocked, simulation.current_route["path"])

    def test_moved_player_replans_from_its_current_cell(self):
        simulation = self.make_simulation()
        simulation.start()
        simulation.update(simulation.player_move_interval)
        current = simulation.player_position
        blocked = simulation.current_route["path"][1]
        simulation.environment.set_fire(blocked)
        simulation.update(0.0)
        self.assertEqual(simulation.current_route["path"][0], current)
        self.assertNotIn(simulation.environment.start, simulation.current_route["path"])

    def test_replanner_can_switch_to_a_safer_exit(self):
        simulation = self.make_simulation()
        original_exit = simulation.selected_exit
        self.assertEqual(original_exit, (1, 18))
        for col in range(2, 19):
            simulation.environment.set_risk((1, col), 100)
        simulation.start()
        simulation.update(0.0)
        self.assertEqual(simulation.selected_exit, (18, 1))
        self.assertEqual(simulation.metrics.replanning_events, 1)

    def test_player_reaches_exit_and_simulation_stops(self):
        environment = BuildingEnvironment(
            rows=8, cols=10, start=(1, 1), exits=[(1, 2)]
        )
        simulation = self.make_simulation(environment)
        simulation.start()
        simulation.update(simulation.player_move_interval)
        self.assertEqual(simulation.player_position, (1, 2))
        self.assertEqual(simulation.status, simulation.ESCAPED)
        self.assertEqual(simulation.metrics.final_status, simulation.ESCAPED)

    def test_initial_no_route_state_stops_simulation(self):
        environment = BuildingEnvironment(
            rows=8, cols=10, start=(2, 2), exits=[(1, 8)]
        )
        for cell in ((1, 2), (3, 2), (2, 1), (2, 3)):
            environment.add_wall(cell)
        simulation = self.make_simulation(environment)
        self.assertEqual(simulation.status, simulation.NO_ROUTE)
        simulation.start()
        simulation.update(5.0)
        self.assertEqual(simulation.metrics.movement_steps, 0)

    def test_blocked_player_becomes_trapped_and_stops(self):
        simulation = self.make_simulation()
        simulation.start()
        simulation.ignite(simulation.player_position)
        simulation.update(0.0)
        self.assertEqual(simulation.status, simulation.TRAPPED)
        self.assertEqual(simulation.metrics.movement_steps, 0)

    def test_pause_stops_time_fire_and_movement(self):
        simulation = self.make_simulation()
        simulation.ignite((5, 5))
        simulation.start()
        simulation.pause()
        position = simulation.player_position
        simulation.update(5.0)
        self.assertEqual(simulation.status, simulation.PAUSED)
        self.assertEqual(simulation.player_position, position)
        self.assertEqual(simulation.metrics.elapsed_time, 0.0)
        self.assertEqual(simulation.fire_simulation.step_count, 0)

    def test_single_step_while_paused_advances_one_move_and_fire_step(self):
        simulation = self.make_simulation()
        simulation.ignite((5, 5))
        simulation.start()
        simulation.pause()
        simulation.step_once()
        self.assertEqual(simulation.status, simulation.PAUSED)
        self.assertEqual(simulation.metrics.movement_steps, 1)
        self.assertEqual(simulation.fire_simulation.step_count, 1)

    def test_reset_restores_environment_fire_route_and_metrics(self):
        simulation = self.make_simulation()
        simulation.start()
        simulation.update(simulation.player_move_interval)
        simulation.ignite((5, 5))
        simulation.reset()
        self.assertEqual(simulation.status, simulation.READY)
        self.assertEqual(simulation.player_position, simulation.environment.start)
        self.assertFalse(simulation.environment.fire_cells)
        self.assertTrue((simulation.environment.risk_map == 0).all())
        self.assertEqual(simulation.fire_simulation.step_count, 0)
        self.assertEqual(simulation.metrics.movement_steps, 0)
        self.assertEqual(simulation.metrics.replanning_events, 0)
        self.assertIsNotNone(simulation.current_route)

    def test_direction_change_is_forwarded_to_fire_simulator(self):
        simulation = self.make_simulation()
        simulation.fire_simulation.set_spread_probability(0.4)
        simulation.set_fire_direction("east")
        self.assertEqual(simulation.fire_simulation.direction, "east")
        self.assertGreater(
            simulation.fire_simulation.spread_probability(100, "east"),
            simulation.fire_simulation.spread_probability(100, "north"),
        )
        simulation.reset()
        self.assertEqual(simulation.fire_simulation.direction, "all")

    def test_metrics_track_elapsed_steps_replans_and_route(self):
        simulation = self.make_simulation()
        initial_length = simulation.current_route["steps"]
        simulation.start()
        simulation.update(0.3)
        summary = simulation.summary()
        self.assertEqual(summary["elapsed_time"], 0.3)
        self.assertEqual(summary["movement_steps"], 1)
        self.assertEqual(summary["initial_route_length"], initial_length)
        self.assertEqual(summary["final_route_length"], initial_length)
        self.assertIsNotNone(summary["total_weighted_route_cost"])
        self.assertEqual(summary["selected_exit"], simulation.selected_exit)
        self.assertEqual(summary["final_status"], simulation.status)

    def test_invalid_delta_is_rejected(self):
        simulation = self.make_simulation()
        simulation.start()
        for delta in (-1, float("nan"), float("inf")):
            with self.subTest(delta=delta):
                with self.assertRaises(ValueError):
                    simulation.update(delta)


if __name__ == "__main__":
    unittest.main()
