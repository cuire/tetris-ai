from collections import namedtuple

Shape = namedtuple("Shape", ["name", "coordinates", "can_be_rotated", "color"])

O_SHAPE = Shape(
    name="O",
    coordinates=((0, 0), (1, 0), (0, 1), (1, 1)),
    can_be_rotated=False,
    color="black",
)

I_SHAPE = Shape(
    name="I",
    coordinates=((0, 0), (1, 0), (2, 0), (3, 0)),
    can_be_rotated=True,
    color="lightblue",
)

J_SHAPE = Shape(
    name="J",
    coordinates=((2, 0), (0, 1), (1, 1), (2, 1)),
    can_be_rotated=True,
    color="orange",
)

L_SHAPE = Shape(
    name="L",
    coordinates=((0, 0), (0, 1), (1, 1), (2, 1)),
    can_be_rotated=True,
    color="blue",
)

S_SHAPE = Shape(
    name="S",
    coordinates=((0, 1), (1, 1), (1, 0), (2, 0)),
    can_be_rotated=True,
    color="green",
)

Z_SHAPE = Shape(
    name="Z",
    coordinates=((0, 0), (1, 0), (1, 1), (2, 1)),
    can_be_rotated=True,
    color="red",
)

T_SHAPE = Shape(
    name="T",
    coordinates=((1, 0), (0, 1), (1, 1), (2, 1)),
    can_be_rotated=True,
    color="purple",
)


__all__ = [
    "Shape",
    "O_SHAPE",
    "I_SHAPE",
    "J_SHAPE",
    "L_SHAPE",
    "S_SHAPE",
    "Z_SHAPE",
    "T_SHAPE",
]
