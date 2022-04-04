import numpy as np
import pytest
from tetris_ai.game import SHAPES, Shape, ShapeTuple, Tetris


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


def test_holding_current_shape():
    game = Tetris()
    game.reset()
    done = game.hold_current_shape()
    assert game.holded_shape is not None
    assert game.can_hold_shape is False
    assert done


def test_cannt_hold_shape_case():
    game = Tetris()
    game.hold_current_shape()
    done = game.hold_current_shape()
    assert not done


@pytest.mark.parametrize("shape", list(SHAPES))
def test_froze_shape(shape):
    game = Tetris()
    game.reset(starting_shape=shape)
    game.froze_current_shape()

    frozen_shape_clone = Shape(board=game, shape=shape)
    for box in frozen_shape_clone.boxes:
        assert game.field[box.y][box.x] == 1

    assert np.count_nonzero(game.field == 1) == len(frozen_shape_clone.boxes)


def test_remove_complite_lines():
    game = Tetris()
    test_field = np.array(
        [
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
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]
    )
    game.reset(field=test_field)
    assert game.remove_complete_lines() == 20


def test_is_game_over():
    game = Tetris()
    test_field = np.array(
        [
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
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]
    )
    game.reset(field=test_field)
    assert game.is_game_over()


def test_game_is_not_over():
    game = Tetris()
    game.reset()
    assert not game.is_game_over()


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
def test_can_move_shape_on_empty_field(x, y, expected):
    game = Tetris(grid_size_x=10, grid_size_y=20)
    game.reset()
    shape = Shape(board=game, shape=SHAPES[0])
    assert shape.can_move_shape(x, y) is expected


def test_can_move_shape_on_filled_field():
    game = Tetris()
    test_field = np.array(
        [
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
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]
    )
    game.reset(field=test_field)
    shape = Shape(board=game, shape=SHAPES[0])
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
def test_move_shape(x, y, expected):
    game = Tetris(grid_size_x=10, grid_size_y=20)
    game.reset()
    shape = Shape(board=game, shape=SHAPES[0])
    assert shape.move(x, y) is expected
