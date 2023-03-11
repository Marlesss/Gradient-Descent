from typing import List, Union
from math import sqrt


class Vector:
    coords: List[float]

    # def __init__(self, coords: List[float]):
    #     self.coords = coords

    def __init__(self, *args: Union[int, float]):
        assert all(map(lambda arg: isinstance(arg, float) or isinstance(arg, int), args))
        self.coords = list(args)

    def __repr__(self):
        return str(self.coords)

    def __len__(self):
        return len(self.coords)

    def vector_length(self) -> float:
        return sqrt(sum(map(lambda x: x ** 2, self.coords)))

    def __getitem__(self, item):
        if isinstance(item, int) and 0 <= item < len(self):
            return self.coords[item]

    def coords_op(self, func):
        return [func(i) for i in range(len(self.coords))]

    def __neg__(self):
        return Vector(self.coords_op(lambda i: -self.coords[i]))

    def __abs__(self):
        return Vector(self.coords_op(lambda i: abs(self[i])))

    def __add__(self, other):
        assert isinstance(other, Vector)
        assert len(self.coords) == len(other.coords)
        return Vector(self.coords_op(lambda i: self.coords[i] + other.coords[i]))

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            return Vector(self.coords_op(lambda i: self.coords[i] * other))

    def __rmul__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            return Vector(self.coords_op(lambda i: self.coords[i] * other))

    # Compare

    def __eq__(self, other):
        return isinstance(other, Vector) and len(self) == len(other) and all(
            self.coords_op(lambda i: self[i] == other[i]))

    def __lt__(self, other):
        if isinstance(other, Vector) and len(self) == len(other):
            return all(self.coords_op(lambda i: self[i] < other[i]))
        return False
