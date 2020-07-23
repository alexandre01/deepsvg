from __future__ import annotations
import numpy as np
from enum import Enum
import torch
from typing import List, Union
Num = Union[int, float]
float_type = (int, float, np.float32)


def det(a: Point, b: Point):
    return a.pos[0] * b.pos[1] - a.pos[1] * b.pos[0]


def get_rotation_matrix(angle: Union[Angle, float]):
    if isinstance(angle, Angle):
        theta = angle.rad
    else:
        theta = angle
    c, s = np.cos(theta), np.sin(theta)
    rot_m = np.array([[c, -s],
                      [s, c]], dtype=np.float32)
    return rot_m


def union_bbox(bbox_list: List[Bbox]):
    res = None
    for bbox in bbox_list:
        res = bbox.union(res)
    return res


class Geom:
    def copy(self):
        raise NotImplementedError

    def to_str(self):
        raise NotImplementedError

    def to_tensor(self):
        raise NotImplementedError

    @staticmethod
    def from_tensor(vector: torch.Tensor):
        raise NotImplementedError

    def scale(self, factor):
        pass

    def translate(self, vec):
        pass

    def rotate(self, angle: Union[Angle, float]):
        pass

    def numericalize(self, n=256):
        raise NotImplementedError


######### Point
class Point(Geom):
    num_args = 2

    def __init__(self, x=None, y=None):
        if isinstance(x, np.ndarray):
            self.pos = x.astype(np.float32)
        elif x is None and y is None:
            self.pos = np.array([0., 0.], dtype=np.float32)
        elif (isinstance(x, float_type) or x is None) and (isinstance(y, float_type) or y is None):
            if x is None:
                x = y
            if y is None:
                y = x
            self.pos = np.array([x, y], dtype=np.float32)
        else:
            raise ValueError()

    def copy(self):
        return Point(self.pos.copy())

    @property
    def x(self):
        return self.pos[0]

    @property
    def y(self):
        return self.pos[1]

    def xproj(self):
        return Point(self.x, 0.)

    def yproj(self):
        return Point(0., self.y)

    def __add__(self, other):
        return Point(self.pos + other.pos)

    def __sub__(self, other):
        return self + other.__neg__()

    def __mul__(self, lmbda):
        if isinstance(lmbda, Point):
            return Point(self.pos * lmbda.pos)

        assert isinstance(lmbda, float_type)
        return Point(lmbda * self.pos)

    def __rmul__(self, lmbda):
        return self * lmbda

    def __truediv__(self, lmbda):
        if isinstance(lmbda, Point):
            return Point(self.pos / lmbda.pos)

        assert isinstance(lmbda, float_type)
        return self * (1 / lmbda)

    def __neg__(self):
        return self * -1

    def __repr__(self):
        return f"P({self.x}, {self.y})"

    def to_str(self):
        return f"{self.x} {self.y}"

    def tolist(self):
        return self.pos.tolist()

    def to_tensor(self):
        return torch.tensor(self.pos)

    @staticmethod
    def from_tensor(vector: torch.Tensor):
        return Point(*vector.tolist())

    def translate(self, vec: Point):
        self.pos += vec.pos

    def matmul(self, m):
        return Point(m @ self.pos)

    def rotate(self, angle: Union[Angle, float]):
        rot_m = get_rotation_matrix(angle)
        return self.matmul(rot_m)

    def rotate_(self, angle: Union[Angle, float]):
        rot_m = get_rotation_matrix(angle)
        self.pos = rot_m @ self.pos

    def scale(self, factor):
        self.pos *= factor

    def dot(self, other: Point):
        return self.pos.dot(other.pos)

    def norm(self):
        return float(np.linalg.norm(self.pos))

    def cross(self, other: Point):
        return np.cross(self.pos, other.pos)

    def dist(self, other: Point):
        return (self - other).norm()

    def angle(self, other: Point, signed=False):
        rad = np.arccos(np.clip(self.normalize().dot(other.normalize()), -1., 1.))

        if signed:
            sign = 1 if det(self, other) >= 0 else -1
            rad *= sign
        return Angle.Rad(rad)

    def distToLine(self, p1: Point, p2: Point):
        if p1.isclose(p2):
            return self.dist(p1)

        return abs((p2 - p1).cross(p1 - self)) / (p2 - p1).norm()

    def normalize(self):
        return self / self.norm()

    def numericalize(self, n=256):
        self.pos = self.pos.round().clip(min=0, max=n-1)

    def isclose(self, other: Point):
        return np.allclose(self.pos, other.pos)

    def iszero(self):
        return np.all(self.pos == 0)

    def pointwise_min(self, other: Point):
        return Point(min(self.x, other.x), min(self.y, other.y))

    def pointwise_max(self, other: Point):
        return Point(max(self.x, other.x), max(self.y, other.y))


class Radius(Point):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def copy(self):
        return Radius(self.pos.copy())

    def __repr__(self):
        return f"Rad({self.pos[0]}, {self.pos[1]})"

    def translate(self, vec: Point):
        pass


class Size(Point):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def copy(self):
        return Size(self.pos.copy())

    def __repr__(self):
        return f"Size({self.pos[0]}, {self.pos[1]})"

    def max(self):
        return self.pos.max()

    def min(self):
        return self.pos.min()

    def translate(self, vec: Point):
        pass


######### Coord
class Coord(Geom):
    num_args = 1

    class XY(Enum):
        X = "x"
        Y = "y"

    def __init__(self, coord, xy: XY = XY.X):
        self.coord = coord
        self.xy = xy

    def __repr__(self):
        return f"{self.xy.value}({self.coord})"

    def to_str(self):
        return str(self.coord)

    def to_tensor(self):
        return torch.tensor([self.coord])

    def __add__(self, other):
        if isinstance(other, float_type):
            return Coord(self.coord + other, self.xy)
        elif isinstance(other, Coord):
            if self.xy != other.xy:
                raise ValueError()
            return Coord(self.coord + other.coord, self.xy)
        elif isinstance(other, Point):
            return Coord(self.coord + getattr(other, self.xy.value), self.xy)
        else:
            raise ValueError()

    def __sub__(self, other):
        return self + other.__neg__()

    def __mul__(self, lmbda):
        assert isinstance(lmbda, float_type)
        return Coord(lmbda * self.coord)

    def __neg__(self):
        return self * -1

    def scale(self, factor):
        self.coord *= factor

    def translate(self, vec: Point):
        self.coord += getattr(vec, self.xy.value)

    def to_point(self, pos: Point, is_absolute=True):
        point = pos.copy() if is_absolute else Point(0.)
        point.pos[int(self.xy == Coord.XY.Y)] = self.coord
        return point


class XCoord(Coord):
    def __init__(self, coord):
        super().__init__(coord, xy=Coord.XY.X)

    def copy(self):
        return XCoord(self.coord)


class YCoord(Coord):
    def __init__(self, coord):
        super().__init__(coord, xy=Coord.XY.Y)

    def copy(self):
        return YCoord(self.coord)


######### Bbox
class Bbox(Geom):
    num_args = 4

    def __init__(self, x=None, y=None, w=None, h=None):
        if isinstance(x, Point) and isinstance(y, Point):
            self.xy = x
            wh = y - x
            self.wh = Size(wh.x, wh.y)
        elif (isinstance(x, float_type) or x is None) and (isinstance(y, float_type) or y is None):
            if x is None:
                x = 0.
            if y is None:
                y = float(x)

            if w is None and h is None:
                w, h = float(x), float(y)
                x, y = 0., 0.
            self.xy = Point(x, y)
            self.wh = Size(w, h)
        else:
            raise ValueError()

    @property
    def xy2(self):
        return self.xy + self.wh

    def copy(self):
        bbox = Bbox()
        bbox.xy = self.xy.copy()
        bbox.wh = self.wh.copy()
        return bbox

    @property
    def size(self):
        return self.wh

    @property
    def center(self):
        return self.xy + self.wh / 2

    def __repr__(self):
        return f"Bbox({self.xy.to_str()} {self.wh.to_str()})"

    def to_str(self):
        return f"{self.xy.to_str()} {self.wh.to_str()}"

    def to_tensor(self):
        return torch.tensor([*self.xy.to_tensor(), *self.wh.to_tensor()])

    def make_square(self, min_size=None):
        center = self.center
        size = self.wh.max()

        if min_size is not None:
            size = max(size, min_size)

        self.wh = Size(size, size)
        self.xy = center - self.wh / 2

        return self

    def translate(self, vec):
        self.xy.translate(vec)

    def scale(self, factor):
        self.xy.scale(factor)
        self.wh.scale(factor)

    def union(self, other: Bbox):
        if other is None:
            return self
        return Bbox(self.xy.pointwise_min(other.xy), self.xy2.pointwise_max(other.xy2))

    def intersect(self, other: Bbox):
        if other is None:
            return self

        bbox = Bbox(self.xy.pointwise_max(other.xy), self.xy2.pointwise_min(other.xy2))
        if bbox.wh.x < 0 or bbox.wh.y < 0:
            return None

        return bbox

    @staticmethod
    def from_points(points: List[Point]):
        if not points:
            return None
        xy = xy2 = points[0]
        for p in points[1:]:
            xy = xy.pointwise_min(p)
            xy2 = xy2.pointwise_max(p)
        return Bbox(xy, xy2)

    def to_rectangle(self, *args, **kwargs):
        from .svg_primitive import SVGRectangle
        return SVGRectangle(self.xy, self.wh, *args, **kwargs)

    def area(self):
        return self.wh.pos.prod()

    def overlap(self, other):
        inter = self.intersect(other)
        if inter is None:
            return 0.
        return inter.area() / self.area()


######### Angle
class Angle(Geom):
    num_args = 1

    def __init__(self, deg):
        self.deg = deg

    @property
    def rad(self):
        return np.deg2rad(self.deg)

    def copy(self):
        return Angle(self.deg)

    def __repr__(self):
        return f"α({self.deg})"

    def to_str(self):
        return str(self.deg)

    def to_tensor(self):
        return torch.tensor([self.deg])

    @staticmethod
    def from_tensor(vector: torch.Tensor):
        return Angle(vector.item())

    @staticmethod
    def Rad(rad):
        return Angle(np.rad2deg(rad))

    def __add__(self, other: Angle):
        return Angle(self.deg + other.deg)

    def __sub__(self, other: Angle):
        return self + other.__neg__()

    def __mul__(self, lmbda):
        assert isinstance(lmbda, float_type)
        return Angle(lmbda * self.deg)

    def __rmul__(self, lmbda):
        assert isinstance(lmbda, float_type)
        return self * lmbda

    def __truediv__(self, lmbda):
        assert isinstance(lmbda, float_type)
        return self * (1 / lmbda)

    def __neg__(self):
        return self * -1


######### Flag
class Flag(Geom):
    num_args = 1

    def __init__(self, flag):
        self.flag = int(flag)

    def copy(self):
        return Flag(self.flag)

    def __repr__(self):
        return f"flag({self.flag})"

    def to_str(self):
        return str(self.flag)

    def to_tensor(self):
        return torch.tensor([self.flag])

    def __invert__(self):
        return Flag(1 - self.flag)

    @staticmethod
    def from_tensor(vector: torch.Tensor):
        return Flag(vector.item())
