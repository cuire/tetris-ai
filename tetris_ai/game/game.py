from __future__ import annotations

from dataclasses import dataclass
from random import choice
from typing import Union

import numpy as np

from tetris_ai.game.shape import *


@dataclass
class Point:
    x: int
    y: int


SHAPES = (O_SHAPE, I_SHAPE, J_SHAPE, L_SHAPE, S_SHAPE, Z_SHAPE, T_SHAPE)


class Tetris:
    """

    ### Action Space
    The action is a 'ndarray' which can take values from 1 to 7.
    | Num | Action                    |
    |-----|---------------------------|
    | 1   | Rotate current_shape      |
    | 2   | Move current_shape down   |
    | 3   | Move current_shape left   |
    | 4   | Move current_shape right  |
    | 5   | 'Hard drop' current_shape |
    | 6   | Hold current_shape        |
    | 7   | Reset game variables      |

    """

    field: np.ndarray
    current_shape: Tetromino
    can_hold_shape: bool
    holded_shape: Union[Shape, None] = None
    next_shape: Union[Shape, None] = None
    shape_spawm_delay: int
    total_score: float = 0.0

    def __init__(self, grid_size_x: int = 10, grid_size_y: int = 20):
        self.grid_size_x = grid_size_x
        self.grid_size_y = grid_size_y

        self.gravity = None

    def reset(self, field: np.ndarray = None, starting_shape=None, gravity=True):
        if field is None:
            self.field = np.zeros((self.grid_size_y, self.grid_size_x), dtype=int)
        else:
            self.field = field

        self.next_shape = None
        self.can_hold_shape = True
        self.current_shape = self.spawn_new_shape(shape=starting_shape)
        self.shape_spawm_delay = self.get_shape_spawn_delay()

        self.gravity = gravity

    def spawn_new_shape(self, shape: Shape = None):
        if shape is None:
            shape = self.next_shape or choice(SHAPES)
            self.next_shape = choice(SHAPES)

        return Tetromino(board=self, shape=shape)

    def hold_current_shape(self):
        if not self.can_hold_shape:
            return False

        current_shape = self.current_shape.shape
        self.current_shape = self.spawn_new_shape(shape=self.holded_shape)
        self.holded_shape = current_shape

        self.can_hold_shape = False

        return True

    def step(self, action: int):
        assert self.field is not None

        reward = 0.0
        done = False

        self.handle_action(action)

        self.shape_spawm_delay -= 1
        if self.shape_spawm_delay <= 0 and self.gravity:
            is_shape_falling = self.current_shape.fall()
            self.shape_spawm_delay = self.get_shape_spawn_delay()

            if not is_shape_falling:
                self.froze_current_shape()
                self.can_hold_shape = True

                removed_lines_count = self.remove_complete_lines()
                self.current_shape = self.spawn_new_shape()

                reward = float(self.get_reward(removed_lines_count))
                done = self.is_game_over()

        self.total_score += reward

        return self.get_current_state(), reward, done

    def handle_action(self, action: int):
        if action == 1:
            self.current_shape.rotate()
        if action == 2:
            self.current_shape.move(x=0, y=1)
        if action == 3:
            self.current_shape.move(x=-1, y=0)
        if action == 4:
            self.current_shape.move(x=1, y=0)
        if action == 5:
            self.current_shape.hard_drop()
        if action == 6:
            self.hold_current_shape()
        if action == 7:
            self.reset()

    def get_current_state(self):
        field = np.copy(self.field)
        for box in self.current_shape.boxes:
            field[box.y][box.x] = 1
        return np.resize(field, (field.shape[0] * field.shape[1],))

    def froze_current_shape(self):
        for box in self.current_shape.boxes:
            self.field[box.y][box.x] = 1

    def remove_complete_lines(self):
        complete_lines = [
            index for index in range(len(self.field)) if all(self.field[index])
        ]

        self.field = np.delete(self.field, complete_lines, axis=0)

        self.field = np.insert(
            self.field,
            0,
            np.zeros((len(complete_lines), self.grid_size_x), dtype=int),
            axis=0,
        )

        return len(complete_lines)

    def get_shape_spawn_delay(self):
        return 10

    def is_game_over(self):
        """
        Check if created shape is able to fall.
        """
        for box in self.current_shape.boxes:
            if not self.current_shape.can_move_box(box, 0, 1):
                return True
        return False

    def get_reward(self, removed_lines_count: int):
        return {0: 0, 1: 100, 2: 300, 3: 700, 4: 1500}.get(removed_lines_count)


class Tetromino:
    def __init__(self, board: Tetris, shape: Shape):
        self.name = shape.name
        self.board = board
        self.shape = shape

        starting_position = board.grid_size_x // 2 - 1
        self.boxes = np.array(
            [Point(x=x + starting_position, y=y + 1) for x, y in shape.coordinates]
        )
        self.center = self.boxes[2] if shape.can_be_rotated else None

    def move(self, x: int, y: int):
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
        can_fall = True
        while can_fall:
            can_fall = self.fall()

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

    def can_move_box(self, box: Point, x: int, y: int):
        return self.can_move_box_to(box.x + x, box.y + y)

    def can_move_box_to(self, x: int, y: int):
        if x < 0 or x > self.board.grid_size_x - 1:
            return False
        if y > self.board.grid_size_y - 1 or y < 0 or self.board.field[y][x]:
            return False
        return True

    def can_move_shape(self, x: int, y: int):
        for box in self.boxes:
            if not self.can_move_box(box, x, y):
                return False
        return True

    def __str__(self):
        return f"{self.name} Tetromino"

    def __repr__(self):
        return f"<{self.__str__()}>"
