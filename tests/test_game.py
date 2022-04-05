import numpy as np
import pytest
from tetris_ai.game import (
    J_SHAPE,
    L_SHAPE,
    O_SHAPE,
    SHAPES,
    T_SHAPE,
    Z_SHAPE,
    Shape,
    ShapeTuple,
    Tetris,
)


@pytest.mark.parametrize("x, y", [(1, 5), (2, 5), (3, 5), (4, 5), (10, 20)])
def test_empty_field_creation(x, y):
    game = Tetris(grid_size_x=x, grid_size_y=y)
    game.reset()
    field_with_zeros = np.zeros((y, x), dtype=int)
    assert np.array_equal(game.field, field_with_zeros)


def test_existing_field_creation():
    game = Tetris()
    test_field = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 1, 1, 0, 0, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
        ]
    )
    game.reset(field=test_field)
    assert np.array_equal(game.field, test_field)


def test_new_shape_creation():
    tetromino = Tetris().spawn_new_shape()
    assert tetromino.shape


@pytest.mark.parametrize("shape", list(SHAPES))
def test_existing_shape_creation(shape):
    tetromino = Tetris().spawn_new_shape(shape=shape)
    assert tetromino.shape == shape


def test_holding_current_shape(game_with_empty_field):
    assert game_with_empty_field.hold_current_shape()
    assert game_with_empty_field.holded_shape is not None
    assert game_with_empty_field.can_hold_shape is False


def test_cannt_hold_shape_case(game_with_empty_field):
    game_with_empty_field.hold_current_shape()
    assert game_with_empty_field.hold_current_shape() is False


@pytest.mark.parametrize("shape", list(SHAPES))
def test_froze_shape(shape):
    game = Tetris()
    game.reset(starting_shape=shape)
    game.froze_current_shape()

    frozen_shape_clone = Shape(board=game, shape=shape)
    for box in frozen_shape_clone.boxes:
        assert game.field[box.y][box.x] == 1

    assert np.count_nonzero(game.field == 1) == len(frozen_shape_clone.boxes)


def test_remove_complite_lines(game_with_filled_field):
    assert game_with_filled_field.remove_complete_lines() == 20


def test_is_game_over(game_with_filled_field):
    assert game_with_filled_field.is_game_over()


def test_game_is_not_over(game_with_empty_field):
    assert not game_with_empty_field.is_game_over()


@pytest.mark.parametrize("action", range(8))
def test_game_step(game_with_empty_field, action):
    # TODO check current state of game
    for _ in range(
        game_with_empty_field.get_shape_spawn_delay()
        * game_with_empty_field.grid_size_y
    ):
        game_with_empty_field.step(action)


def test_game_level_up(game_with_empty_field):
    game_with_empty_field.level_up()


@pytest.mark.parametrize(
    "x, y, expected",
    [
        (0, 1, True),
        (0, 2, True),
        (-1, 0, True),
        (-15, 0, False),
        (0, -15, False),
    ],
)
def test_can_move_shape_on_empty_field(game_with_empty_field, x, y, expected):
    shape = Shape(board=game_with_empty_field, shape=O_SHAPE)
    assert shape.can_move_shape(x, y) is expected


def test_can_move_shape_on_filled_field(game_with_filled_field):
    shape = Shape(board=game_with_filled_field, shape=O_SHAPE)
    assert shape.can_move_shape(1, 1) is False


@pytest.mark.parametrize(
    "x, y, expected",
    [
        (0, 1, True),
        (0, 2, True),
        (-1, 0, True),
        (-15, 0, False),
        (0, -15, False),
    ],
)
def test_move_shape(game_with_empty_field, x, y, expected):
    shape = Shape(board=game_with_empty_field, shape=O_SHAPE)
    assert shape.move(x, y) is expected

    if expected is False:
        return

    new_shape = Shape(board=game_with_empty_field, shape=O_SHAPE)
    for box in new_shape.boxes:
        box.x += x
        box.y += y

    for index in range(len(new_shape.boxes)):
        assert shape.boxes[index].x == new_shape.boxes[index].x
        assert shape.boxes[index].y == new_shape.boxes[index].y


@pytest.mark.parametrize("shape", list(SHAPES))
def test_rotate_shape(shape):
    game = Tetris(grid_size_x=10, grid_size_y=20)
    game.reset()
    tetromino = Shape(board=game, shape=shape)
    tetromino.move(0, 3)
    assert tetromino.rotate() is shape.can_be_rotated

    # TODO check tetromino coordinates


@pytest.mark.parametrize("shape", [J_SHAPE, L_SHAPE, Z_SHAPE, T_SHAPE])
def test_imposible_shape_rotation(shape):
    """test all shapes except O, because it can't be rotated"""
    game = Tetris(grid_size_x=10, grid_size_y=20)
    test_field = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]
    )
    game.reset(field=test_field)
    tetromino = Shape(board=game, shape=shape)

    # move to lower bound of field
    assert tetromino.move(0, 2)

    assert tetromino.rotate() is False


def test_shape_rotation_with_shift():
    ...


def test_shape_fall(game_with_empty_field):
    tetromino = Shape(board=game_with_empty_field, shape=O_SHAPE)
    assert tetromino.fall()


def test_shape_hard_drop(game_with_empty_field):
    # TODO add checks
    tetromino = Shape(board=game_with_empty_field, shape=O_SHAPE)
    tetromino.hard_drop()


def test_shape_repr(game_with_empty_field):
    tetromino = Shape(board=game_with_empty_field, shape=O_SHAPE)
    assert tetromino.__repr__() == "<O Shape>"
