from collections import namedtuple
from random import choice

import pygame

BOX_SIZE = 20
GRID_SIZE_X = 10
GRID_SIZE_Y = 20
FPS = 60
WIDTH = BOX_SIZE * GRID_SIZE_X
HEIGHT = BOX_SIZE * GRID_SIZE_Y
STARTING_POSITION = GRID_SIZE_X // 2 - 1

GREY = [50, 50, 50]

ShapeTuple = namedtuple("Shape", ["name", "coordinates", "can_be_rotated", "color"])

SHAPES = (
    ShapeTuple(
        name="O",
        coordinates=((0, 0), (1, 0), (0, 1), (1, 1)),
        can_be_rotated=False,
        color="black",
    ),
    ShapeTuple(
        name="I",
        coordinates=((0, 0), (1, 0), (2, 0), (3, 0)),
        can_be_rotated=True,
        color="lightblue",
    ),
    ShapeTuple(
        name="J",
        coordinates=((2, 0), (0, 1), (1, 1), (2, 1)),
        can_be_rotated=True,
        color="orange",
    ),
    ShapeTuple(
        name="L",
        coordinates=((0, 0), (0, 1), (1, 1), (2, 1)),
        can_be_rotated=True,
        color="blue",
    ),
    ShapeTuple(
        name="S",
        coordinates=((0, 1), (1, 1), (1, 0), (2, 0)),
        can_be_rotated=True,
        color="green",
    ),
    ShapeTuple(
        name="Z",
        coordinates=((0, 0), (1, 0), (1, 1), (2, 1)),
        can_be_rotated=True,
        color="red",
    ),
    ShapeTuple(
        name="T",
        coordinates=((1, 0), (0, 1), (1, 1), (2, 1)),
        can_be_rotated=True,
        color="purple",
    ),
)


class Tetris:
    def __init__(self):
        self.reset()

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
                self.froze_shape()
                self.current_shape = self.spawn_new_shape()

                if self.is_game_over():
                    self.game_over()
            self.anim = 0

    def froze_shape(self):
        for i in range(4):

            self.field[self.current_shape.boxes[i].y][
                self.current_shape.boxes[i].x
            ] = GREY

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
        self.boxes = [
            pygame.Rect(x + STARTING_POSITION, y + 1, 1, 1)
            for x, y in shape.coordinates
        ]
        self.center = self.boxes[2] if shape.can_be_rotated else None

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
        center = self.center

        # If center is None, shape can't be rotated
        if center is None:
            return False

        def rotation_coords(box):
            x = center.x - box.y + center.y
            y = center.y + box.x - center.x
            return x, y

        # TODO check if the shape can be rotated with shift left and right when it reaches the bounds
        # Check if shape can be rotated
        for box in self.boxes:
            x, y = rotation_coords(box)
            if not self.can_move_box_to(x, y):
                return False

        # Rotate shape
        for box in self.boxes:
            box.x, box.y = rotation_coords(box)

        return True

    def delete(self):
        self.boxes.clear()

    def can_move_box(self, box, x, y):
        return self.can_move_box_to(box.x + x, box.y + y)

    def can_move_box_to(self, x, y):
        if x < 0 or x > GRID_SIZE_X - 1:
            return False
        if y > GRID_SIZE_Y - 1 or self.field[y][x]:
            return False
        return True

    def can_move_shape(self, x, y):
        for box in self.boxes:
            if not self.can_move_box(box, x, y):
                return False
        return True

    def __str__(self):
        return f"{self.name} Shape"

    def __repr__(self):
        return f"<{self.__str__()}>"
