"""Pygame view and controls for :class:`EvacuationSimulation`."""

from config import (
    BACKGROUND,
    CELL_SIZE,
    COLS,
    EXIT_COLOR,
    FIRE_COLOR,
    FLOOR_COLOR,
    FPS,
    GRID_COLOR,
    PANEL_WIDTH,
    PATH_COLOR,
    PLAYER_COLOR,
    RISK_COLOR,
    WALL_COLOR,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
)
from config import DYNAMIC_BLOCKED, FIRE, WALL
from environment import BuildingEnvironment
from simulation import EvacuationSimulation


class PygameVisualizer:
    """Render a simulation and route keyboard/mouse events to its controller."""

    def __init__(self, simulation: EvacuationSimulation):
        try:
            import pygame
        except ImportError as error:
            raise RuntimeError(
                "Pygame is required for the graphical application but is not installed."
            ) from error

        self.pygame = pygame
        self.simulation = simulation
        pygame.font.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Risk-Aware Emergency Escape Simulation")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("segoeui", 19)
        self.small_font = pygame.font.SysFont("segoeui", 15)
        self.running = True
        self.ignition_mode = False
        self.notice = "Press I to enable click-to-ignite."
        self.notice_until = 0
        self.grid_width = simulation.environment.cols * CELL_SIZE

    def _draw_text(self, text, position, color=(35, 39, 44), font=None):
        surface = (font or self.font).render(str(text), True, color)
        self.screen.blit(surface, position)

    @staticmethod
    def _risk_floor_color(risk):
        fraction = max(0.0, min(1.0, float(risk) / 100.0))
        floor = FLOOR_COLOR
        tint = RISK_COLOR
        return tuple(
            round(floor[index] * (1.0 - fraction) + tint[index] * fraction)
            for index in range(3)
        )

    def _draw_grid(self):
        pygame = self.pygame
        environment = self.simulation.environment
        cell = CELL_SIZE
        for row in range(environment.rows):
            for col in range(environment.cols):
                position = (row, col)
                kind = int(environment.grid[position])
                rect = pygame.Rect(col * cell, row * cell, cell, cell)

                if kind == WALL:
                    color = WALL_COLOR
                elif kind == FIRE:
                    color = FIRE_COLOR
                elif kind == DYNAMIC_BLOCKED:
                    color = (105, 105, 112)
                else:
                    color = self._risk_floor_color(environment.risk_map[position])
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, GRID_COLOR, rect, 1)

                if kind == FIRE:
                    pygame.draw.circle(
                        self.screen,
                        (255, 190, 70),
                        rect.center,
                        max(3, cell // 5),
                    )
                elif kind == DYNAMIC_BLOCKED:
                    pygame.draw.line(self.screen, (235, 235, 235), rect.topleft,
                                     rect.bottomright, 2)
                    pygame.draw.line(self.screen, (235, 235, 235), rect.topright,
                                     rect.bottomleft, 2)

                if position in environment.exits:
                    marker = rect.inflate(-cell // 2, -cell // 2)
                    pygame.draw.rect(self.screen, EXIT_COLOR, marker, 2, border_radius=4)
                    self._draw_text("E", (rect.x + 9, rect.y + 5),
                                    (255, 255, 255), self.small_font)

        path = [] if self.simulation.current_route is None else self.simulation.current_route["path"]
        if path:
            centers = [
                (col * cell + cell // 2, row * cell + cell // 2)
                for row, col in path
            ]
            if len(centers) > 1:
                pygame.draw.lines(self.screen, PATH_COLOR, False, centers, max(3, cell // 7))
            for row, col in path[1:-1]:
                rect = pygame.Rect(col * cell, row * cell, cell, cell)
                pygame.draw.rect(self.screen, PATH_COLOR, rect, 2)

        start = tuple(environment.start)
        player = self.simulation.player_position
        if start != player and environment.grid[start] != WALL:
            start_center = (start[1] * cell + cell // 2, start[0] * cell + cell // 2)
            pygame.draw.circle(self.screen, (255, 205, 70), start_center, cell // 5)

        player_center = (player[1] * cell + cell // 2, player[0] * cell + cell // 2)
        pygame.draw.circle(self.screen, PLAYER_COLOR, player_center, cell // 3)
        pygame.draw.circle(self.screen, (255, 255, 255), player_center, cell // 3, 2)

    def _draw_panel(self):
        pygame = self.pygame
        panel_x = self.grid_width
        pygame.draw.rect(
            self.screen,
            BACKGROUND,
            pygame.Rect(panel_x, 0, PANEL_WIDTH, WINDOW_HEIGHT),
        )
        pygame.draw.line(self.screen, (205, 210, 215), (panel_x, 0),
                         (panel_x, WINDOW_HEIGHT), 2)
        x = panel_x + 16
        y = 16
        self._draw_text("EVACUATION STATUS", (x, y))
        y += 32

        simulation = self.simulation
        route = simulation.current_route
        values = (
            ("Status", simulation.status),
            ("Player", simulation.player_position),
            ("Selected exit", simulation.selected_exit or "None"),
            ("Sim time", f"{simulation.metrics.elapsed_time:.1f} s"),
            ("Fire steps", simulation.fire_simulation.step_count),
            ("Replans", simulation.metrics.replanning_events),
            ("Route cost", "—" if route is None else f"{route['cost']:.2f}"),
            ("Route steps", "—" if route is None else len(route["path"]) - 1),
            ("Direction", simulation.fire_simulation.direction.title()),
            ("Active fire", len(simulation.fire_simulation.get_active_fire_cells())),
        )
        for label, value in values:
            self._draw_text(f"{label}: {value}", (x, y), font=self.small_font)
            y += 24

        y += 8
        self._draw_text(
            f"Click ignition: {'ON' if self.ignition_mode else 'OFF'}",
            (x, y),
            (150, 60, 30) if self.ignition_mode else (70, 75, 80),
            self.small_font,
        )
        y += 30
        self._draw_text("SPACE  Start / pause", (x, y), font=self.small_font)
        y += 21
        self._draw_text("N  Single step while paused", (x, y), font=self.small_font)
        y += 21
        self._draw_text("R  Reset    I  Ignition mode", (x, y), font=self.small_font)
        y += 21
        self._draw_text("1–5  Fire direction", (x, y), font=self.small_font)
        y += 21
        self._draw_text("ESC  Quit", (x, y), font=self.small_font)

        if simulation.last_message:
            y += 30
            self._draw_text("Last event:", (x, y), (50, 75, 100), self.small_font)
            y += 19
            message = simulation.last_message
            for start in range(0, len(message), 34):
                self._draw_text(message[start:start + 34], (x, y), (50, 75, 100), self.small_font)
                y += 18

        now = pygame.time.get_ticks()
        if now < self.notice_until:
            self._draw_text(self.notice, (x, WINDOW_HEIGHT - 32), (170, 45, 35), self.small_font)

    def draw(self):
        self.screen.fill(FLOOR_COLOR)
        self._draw_grid()
        self._draw_panel()
        self.pygame.display.flip()

    def _notice(self, message, duration_ms=2200):
        self.notice = message
        self.notice_until = self.pygame.time.get_ticks() + duration_ms

    def _handle_key(self, key):
        pygame = self.pygame
        simulation = self.simulation
        if key == pygame.K_ESCAPE:
            self.running = False
        elif key == pygame.K_SPACE:
            if simulation.status in (simulation.RUNNING, simulation.EVACUATING,
                                     simulation.REPLANNING):
                simulation.pause()
            elif simulation.status == simulation.PAUSED:
                simulation.resume()
            else:
                simulation.start()
        elif key == pygame.K_r:
            simulation.reset()
            self._notice("Simulation reset.")
        elif key == pygame.K_n:
            simulation.step_once()
        elif key == pygame.K_i:
            self.ignition_mode = not self.ignition_mode
            self._notice(f"Click-to-ignite {'enabled' if self.ignition_mode else 'disabled'}.")
        elif key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5):
            directions = {
                pygame.K_1: "north",
                pygame.K_2: "south",
                pygame.K_3: "east",
                pygame.K_4: "west",
                pygame.K_5: "all",
            }
            simulation.set_fire_direction(directions[key])
            self._notice(f"Fire direction: {directions[key]}.")

    def _handle_click(self, position):
        if not self.ignition_mode:
            return
        x, y = position
        if x < 0 or x >= self.grid_width or y < 0 or y >= self.simulation.environment.rows * CELL_SIZE:
            self._notice("Click a cell inside the building grid.")
            return
        cell = (y // CELL_SIZE, x // CELL_SIZE)
        environment = self.simulation.environment
        if cell == self.simulation.player_position or cell in environment.exits:
            self._notice("Cannot ignite the player or an exit.")
            return
        try:
            self.simulation.ignite(cell)
        except ValueError as error:
            self._notice(str(error))
        else:
            self._notice(f"Fire ignited at {cell}.")

    def run(self):
        pygame = self.pygame
        while self.running:
            delta_time = self.clock.tick(FPS) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    self._handle_key(event.key)
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self._handle_click(event.pos)

            self.simulation.update(delta_time)
            self.draw()

        pygame.quit()
        summary = self.simulation.summary()
        print("Simulation summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")


def run_app(seed=None):
    """Create the default environment and launch its Pygame application."""
    environment = BuildingEnvironment()
    simulation = EvacuationSimulation(environment=environment, seed=seed)
    PygameVisualizer(simulation).run()
