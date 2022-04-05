from collections import namedtuple

ShapeTuple = namedtuple("Shape", ["name", "coordinates", "can_be_rotated", "color"])

O_SHAPE = ShapeTuple(
    name="O",
    coordinates=((0, 0), (1, 0), (0, 1), (1, 1)),
    can_be_rotated=False,
    color="black",
)

I_SHAPE = ShapeTuple(
    name="I",
    coordinates=((0, 0), (1, 0), (2, 0), (3, 0)),
    can_be_rotated=True,
    color="lightblue",
)

J_SHAPE = ShapeTuple(
    name="J",
    coordinates=((2, 0), (0, 1), (1, 1), (2, 1)),
    can_be_rotated=True,
    color="orange",
)

L_SHAPE = ShapeTuple(
    name="L",
    coordinates=((0, 0), (0, 1), (1, 1), (2, 1)),
    can_be_rotated=True,
    color="blue",
)

S_SHAPE = ShapeTuple(
    name="S",
    coordinates=((0, 1), (1, 1), (1, 0), (2, 0)),
    can_be_rotated=True,
    color="green",
)

Z_SHAPE = ShapeTuple(
    name="Z",
    coordinates=((0, 0), (1, 0), (1, 1), (2, 1)),
    can_be_rotated=True,
    color="red",
)

T_SHAPE = ShapeTuple(
    name="T",
    coordinates=((1, 0), (0, 1), (1, 1), (2, 1)),
    can_be_rotated=True,
    color="purple",
)


__all__ = [
    "ShapeTuple",
    "O_SHAPE",
    "I_SHAPE",
    "J_SHAPE",
    "L_SHAPE",
    "S_SHAPE",
    "Z_SHAPE",
    "T_SHAPE",
]
