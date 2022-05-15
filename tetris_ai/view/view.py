from typing import Optional

import pygame

from tetris_ai.game import Tetris

FPS = 60
WIDTH = 400
HEIGHT = 600
BOX_SIZE = 25
FONT_SIZE = 50


class TetrisView:
    def __init__(
        self,
        game: Tetris,
        parent: Optional[pygame.Surface] = None,
        tetromino_color=(225, 225, 225),
        grid_line_color=(40, 40, 40),
        background_color=(0, 0, 0),
        font_path="tetris_ai/view/font/Uni-Sans-Heavy.otf",
    ):
        self.clock = pygame.time.Clock()
        self.box_size = BOX_SIZE
        self.game = game

        if parent is None:
            pygame.init()
            pygame.display.init()
            parent = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("Tetris")

        self.display = parent
        self.grid = pygame.Surface(
            (BOX_SIZE * game.grid_size_x, BOX_SIZE * game.grid_size_y)
        )

        self.tetromino_color = tetromino_color
        self.grid_line_color = grid_line_color
        self.background_color = background_color
        self.font = pygame.font.Font(font_path, FONT_SIZE)

    def render(self):
        self.display.fill(self.background_color)

        self._render_grid(self.game.grid_size_x, self.game.grid_size_y)
        self._render_current_shape(self.game.current_shape)
        self._render_field(self.game.field)
        self._render_score()

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
                    pygame.draw.rect(self.grid, self.tetromino_color, figure)

    def _render_grid(self, grid_size_x, grid_size_y):
        self.display.blit(
            self.grid, self.grid.get_rect(center=(WIDTH * 0.5, HEIGHT * 0.52))
        )
        self.grid.fill(self.background_color)
        for x in range(grid_size_x):
            for y in range(grid_size_y):
                rect = pygame.Rect(
                    x * self.box_size, y * self.box_size, self.box_size, self.box_size
                )
                pygame.draw.rect(self.grid, self.grid_line_color, rect, 1)

    def _render_current_shape(self, current_shape):
        for box in current_shape.boxes:
            figure = pygame.Rect(
                box.x * self.box_size,
                box.y * self.box_size,
                self.box_size - 2,
                self.box_size - 2,
            )
            pygame.draw.rect(self.grid, self.tetromino_color, figure)

    def _render_score(self):
        score = self.font.render(str(self.game.total_score), True, self.tetromino_color)

        self.display.blit(score, score.get_rect(center=(WIDTH * 0.5, HEIGHT * 0.06)))
