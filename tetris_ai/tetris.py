from collections import namedtuple
from random import choice

import pygame

BOX_SIZE = 20
GRID_SIZE_X = 10
GRID_SIZE_Y = 20
FPS = 60
WIDTH = BOX_SIZE * GRID_SIZE_X
HEIGHT = BOX_SIZE * GRID_SIZE_Y

ShapeTuple = namedtuple("Shape", ["name", "coordinates", "center", "color"])

SHAPES = (
    ShapeTuple(
        name="O",
        coordinates=((0, 0), (1, 0), (0, 1), (1, 1)),
        center=None,
        color="black",
    ),
    ShapeTuple(
        name="I",
        coordinates=((0, 0), (1, 0), (2, 0), (3, 0)),
        center=(1, 0),
        color="lightblue",
    ),
    ShapeTuple(
        name="J",
        coordinates=((2, 0), (0, 1), (1, 1), (2, 1)),
        center=None,
        color="orange",
    ),
    ShapeTuple(
        name="L",
        coordinates=((0, 0), (0, 1), (1, 1), (2, 1)),
        center=None,
        color="blue",
    ),
    ShapeTuple(
        name="S",
        coordinates=((0, 1), (1, 1), (1, 0), (2, 0)),
        center=None,
        color="green",
    ),
    ShapeTuple(
        name="Z",
        coordinates=((0, 0), (1, 0), (1, 1), (2, 1)),
        center=None,
        color="red",
    ),
    ShapeTuple(
        name="T",
        coordinates=((1, 0), (0, 1), (1, 1), (2, 1)),
        center=None,
        color="purple",
    ),
)


class Tetris:
    def __init__(self):
        self.reset()

    def start(self):
        self.init_window()
        while True:
            self.handle_events()
            self._game_logic()
            self.render()
            pygame.display.update()
            self.clock.tick(60)

    def reset(self):
        self.field = [[0 for i in range(GRID_SIZE_X)] for j in range(GRID_SIZE_Y)]
        self.next_shape = None
        self.current_shape = self.spawn_new_shape()
        self.anim = 0
        self.anim_limit = 2000

    def spawn_new_shape(self, shape=None):
        if shape is None:
            shape = self.next_shape or choice(SHAPES)
            self.next_shape = choice(SHAPES)

        return Shape(self.field, shape=shape)

    def save_current_shape(self):
        ...

    def level_up(self):
        ...

    def _game_logic(self):
        self.anim += 60
        if self.anim > self.anim_limit:
            if not self.current_shape.fall():
                for i in range(4):
                    self.field[self.current_shape.boxes[i].y][
                        self.current_shape.boxes[i].x
                    ] = [50, 50, 50]
                self.current_shape = self.spawn_new_shape()

                if self.is_game_over():
                    self.game_over()
            self.anim = 0

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    self.current_shape.rotate()
                if event.key == pygame.K_DOWN:
                    self.current_shape.move(x=0, y=1)
                if event.key == pygame.K_LEFT:
                    self.current_shape.move(x=-1, y=0)
                if event.key == pygame.K_RIGHT:
                    self.current_shape.move(x=1, y=0)
                if event.key == pygame.K_SPACE:
                    self.current_shape.hard_drop()
                if event.key == pygame.K_c:
                    self.save_current_shape()
                if event.key == pygame.K_ESCAPE:
                    self.reset()

    def init_window(self):
        pygame.init()
        self.display = pygame.display.set_mode((500, 500))
        self.grid = [
            pygame.Rect(x * BOX_SIZE, y * BOX_SIZE, BOX_SIZE, BOX_SIZE)
            for x in range(GRID_SIZE_X)
            for y in range(GRID_SIZE_Y)
        ]
        self.canvas = pygame.Surface((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()

    def render(self):
        self.display.fill((255, 255, 255))
        self.canvas.fill((255, 255, 255))

        [pygame.draw.rect(self.display, (40, 40, 40), rect, 1) for rect in self.grid]

        self.current_shape.render(parent=self.display)

        for y, raw in enumerate(self.field):
            for x, col in enumerate(raw):
                if col:
                    figure = pygame.Rect(
                        x * BOX_SIZE, y * BOX_SIZE, BOX_SIZE - 2, BOX_SIZE - 2
                    )
                    pygame.draw.rect(self.display, col, figure)

    def remove_complete_lines(self):
        ...

    def is_game_over(self):
        """
        Check if created shape is able to fall.
        """
        for box in self.current_shape.boxes:
            if not self.current_shape.can_move_box(box, 0, 1):
                return True
        return False

    def game_over(self):
        print("game_over")


class Shape:
    def __init__(self, filed, shape=None):
        self.field = filed
        self.name = shape.name
        self.center = shape.center
        self.boxes = [
            pygame.Rect(x + GRID_SIZE_X // 2 - 1, y + 1, 1, 1)
            for x, y in shape.coordinates
        ]

    def move(self, x, y):
        if not self.can_move_shape(x, y):
            return False
        else:
            for box in self.boxes:
                box.x += x
                box.y += y
            return True

    def fall(self):
        return self.move(0, 1)

    def hard_drop(self):
        ...

    def rotate(self):
        # TODO totaly broken and need debug
        center = self.boxes[2]

        def rotation_coords(box):
            x = center.x - box.y + center.y
            y = center.y + box.x - center.x
            return x, y

        # Check if shape can be rotated
        for box in self.boxes:
            x, y = rotation_coords(box)
            if not self.can_move_box(box, box.x - x, box.y - y):
                return False

        # Rotate shape
        for box in self.boxes:
            box.x, box.y = rotation_coords(box)

        return True

    def delete(self):
        self.boxes.clear()

    def can_move_box(self, box, x, y):
        if box.x + x < 0 or box.x + x > GRID_SIZE_X - 1:
            return False
        if box.y + y > GRID_SIZE_Y - 1 or self.field[box.y + y][box.x + x]:
            return False
        return True

    def can_move_shape(self, x, y):
        for box in self.boxes:
            if not self.can_move_box(box, x, y):
                return False
        return True

    def render(self, parent: pygame.Surface):
        for box in self.boxes:
            figure = pygame.Rect(
                box.x * BOX_SIZE, box.y * BOX_SIZE, BOX_SIZE - 2, BOX_SIZE - 2
            )
            pygame.draw.rect(parent, (0, 0, 0), figure)

    def __str__(self):
        return f"{self.name} Shape"

    def __repr__(self):
        return f"<{self.__str__()}>"


if __name__ == "__main__":
    game = Tetris().start()
