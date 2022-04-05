import pygame

from tetris_ai.game import Tetris

FPS = 60


class TetrisView:
    def __init__(self, game: Tetris, parent: pygame.Surface, box_size=20):
        self.clock = pygame.time.Clock()
        self.width = box_size * game.grid_size_x
        self.height = box_size * game.grid_size_y
        self.box_size = box_size
        self.display = parent
        self.game = game

    def render(self):
        self.display.fill((255, 255, 255))

        self._render_grid(self.game.grid_size_x, self.game.grid_size_y)
        self._render_current_shape(self.game.current_shape)
        self._render_field(self.game.field)

        pygame.display.update()
        self.clock.tick(60)

    def _render_field(self, field):
        for y, raw in enumerate(field):
            for x, col in enumerate(raw):
                if col:
                    figure = pygame.Rect(
                        x * self.box_size,
                        y * self.box_size,
                        self.box_size - 2,
                        self.box_size - 2,
                    )
                    pygame.draw.rect(self.display, 1, figure)

    def _render_grid(self, grid_size_x, grid_size_y):
        for x in range(grid_size_x):
            for y in range(grid_size_y):
                rect = pygame.Rect(
                    x * self.box_size, y * self.box_size, self.box_size, self.box_size
                )
                pygame.draw.rect(self.display, (40, 40, 40), rect, 1)

    def _render_current_shape(self, current_shape):
        for box in current_shape.boxes:
            figure = pygame.Rect(
                box.x * self.box_size,
                box.y * self.box_size,
                self.box_size - 2,
                self.box_size - 2,
            )
            pygame.draw.rect(self.display, (0, 0, 0), figure)
