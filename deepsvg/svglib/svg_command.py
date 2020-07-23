from __future__ import annotations
from .geom import *
from deepsvg.difflib.tensor import SVGTensor
from .util_fns import get_roots
from enum import Enum
import torch
import math
from typing import List, Union
Num = Union[int, float]


class SVGCmdEnum(Enum):
    MOVE_TO = "m"
    LINE_TO = "l"
    CUBIC_BEZIER = "c"
    CLOSE_PATH = "z"
    ELLIPTIC_ARC = "a"
    QUAD_BEZIER = "q"
    LINE_TO_HORIZONTAL = "h"
    LINE_TO_VERTICAL = "v"
    CUBIC_BEZIER_REFL = "s"
    QUAD_BEZIER_REFL = "t"


svgCmdArgTypes = {
    SVGCmdEnum.MOVE_TO.value: [Point],
    SVGCmdEnum.LINE_TO.value: [Point],
    SVGCmdEnum.CUBIC_BEZIER.value: [Point, Point, Point],
    SVGCmdEnum.CLOSE_PATH.value: [],
    SVGCmdEnum.ELLIPTIC_ARC.value: [Radius, Angle, Flag, Flag, Point],
    SVGCmdEnum.QUAD_BEZIER.value: [Point, Point],
    SVGCmdEnum.LINE_TO_HORIZONTAL.value: [XCoord],
    SVGCmdEnum.LINE_TO_VERTICAL.value: [YCoord],
    SVGCmdEnum.CUBIC_BEZIER_REFL.value: [Point, Point],
    SVGCmdEnum.QUAD_BEZIER_REFL.value: [Point],
}


class SVGCommand:
    def __init__(self, command: SVGCmdEnum, args: List[Geom], start_pos: Point, end_pos: Point):
        self.command = command
        self.args = args

        self.start_pos = start_pos
        self.end_pos = end_pos

    def copy(self):
        raise NotImplementedError

    @staticmethod
    def from_str(cmd_str: str, args_str: List[Num], pos=None, initial_pos=None, prev_command: SVGCommand = None):
        if pos is None:
            pos = Point(0.)
        if initial_pos is None:
            initial_pos = Point(0.)

        cmd = SVGCmdEnum(cmd_str.lower())

        # Implicit MoveTo commands are treated as LineTo
        if cmd is SVGCmdEnum.MOVE_TO and len(args_str) > 2:
            l_cmd_str = SVGCmdEnum.LINE_TO.value
            if cmd_str.isupper():
                l_cmd_str = l_cmd_str.upper()

            l1, pos, initial_pos = SVGCommand.from_str(cmd_str, args_str[:2], pos, initial_pos)
            l2, pos, initial_pos = SVGCommand.from_str(l_cmd_str, args_str[2:], pos, initial_pos)
            return [*l1, *l2], pos, initial_pos

        nb_args = len(args_str)

        if cmd is SVGCmdEnum.CLOSE_PATH:
            assert nb_args == 0, f"Expected no argument for command {cmd_str}: {nb_args} given"
            return [SVGCommandClose(pos, initial_pos)], initial_pos, initial_pos

        expected_nb_args = sum([ArgType.num_args for ArgType in svgCmdArgTypes[cmd.value]])
        assert nb_args % expected_nb_args == 0, f"Expected {expected_nb_args} arguments for command {cmd_str}: {nb_args} given"

        l = []
        i = 0
        for _ in range(nb_args // expected_nb_args):
            args = []
            for ArgType in svgCmdArgTypes[cmd.value]:
                num_args = ArgType.num_args
                arg = ArgType(*args_str[i:i+num_args])

                if cmd_str.islower():
                    arg.translate(pos)
                if isinstance(arg, Coord):
                    arg = arg.to_point(pos)

                args.append(arg)
                i += num_args

            if cmd is SVGCmdEnum.LINE_TO or cmd is SVGCmdEnum.LINE_TO_VERTICAL or cmd is SVGCmdEnum.LINE_TO_HORIZONTAL:
                cmd_parsed = SVGCommandLine(pos, *args)
            elif cmd is SVGCmdEnum.MOVE_TO:
                cmd_parsed = SVGCommandMove(pos, *args)
            elif cmd is SVGCmdEnum.ELLIPTIC_ARC:
                cmd_parsed = SVGCommandArc(pos, *args)
            elif cmd is SVGCmdEnum.CUBIC_BEZIER:
                cmd_parsed = SVGCommandBezier(pos, *args)
            elif cmd is SVGCmdEnum.QUAD_BEZIER:
                cmd_parsed = SVGCommandBezier(pos, args[0], args[0], args[1])
            elif cmd is SVGCmdEnum.QUAD_BEZIER_REFL or cmd is SVGCmdEnum.CUBIC_BEZIER_REFL:
                if isinstance(prev_command, SVGCommandBezier):
                    control1 = pos * 2 - prev_command.control2
                else:
                    control1 = pos
                control2 = args[0] if cmd is SVGCmdEnum.CUBIC_BEZIER_REFL else control1
                cmd_parsed = SVGCommandBezier(pos, control1, control2, args[-1])

            prev_command = cmd_parsed
            pos = cmd_parsed.end_pos

            if cmd is SVGCmdEnum.MOVE_TO:
                initial_pos = pos

            l.append(cmd_parsed)

        return l, pos, initial_pos

    def __repr__(self):
        cmd = self.command.value.upper()
        return f"{cmd}{self.get_geoms()}"

    def to_str(self):
        cmd = self.command.value.upper()
        return f"{cmd}{' '.join([arg.to_str() for arg in self.args])}"

    def to_tensor(self, PAD_VAL=-1):
        raise NotImplementedError

    @staticmethod
    def from_tensor(vector: torch.Tensor):
        cmd_index, args = int(vector[0]), vector[1:]

        cmd = SVGCmdEnum(SVGTensor.COMMANDS_SIMPLIFIED[cmd_index])
        radius = Radius(*args[:2].tolist())
        x_axis_rotation = Angle(*args[2:3].tolist())
        large_arc_flag = Flag(args[3].item())
        sweep_flag = Flag(args[4].item())
        start_pos = Point(*args[5:7].tolist())
        control1 = Point(*args[7:9].tolist())
        control2 = Point(*args[9:11].tolist())
        end_pos = Point(*args[11:].tolist())

        return SVGCommand.from_args(cmd, radius, x_axis_rotation, large_arc_flag, sweep_flag, start_pos, control1, control2, end_pos)

    @staticmethod
    def from_args(command: SVGCmdEnum, radius: Radius, x_axis_rotation: Angle, large_arc_flag: Flag,
                  sweep_flag: Flag, start_pos: Point, control1: Point, control2: Point, end_pos: Point):
        if command is SVGCmdEnum.MOVE_TO:
            return SVGCommandMove(start_pos, end_pos)
        elif command is SVGCmdEnum.LINE_TO:
            return SVGCommandLine(start_pos, end_pos)
        elif command is SVGCmdEnum.CUBIC_BEZIER:
            return SVGCommandBezier(start_pos, control1, control2, end_pos)
        elif command is SVGCmdEnum.CLOSE_PATH:
            return SVGCommandClose(start_pos, end_pos)
        elif command is SVGCmdEnum.ELLIPTIC_ARC:
            return SVGCommandArc(start_pos, radius, x_axis_rotation, large_arc_flag, sweep_flag, end_pos)

    def draw(self, *args, **kwargs):
        from .svg_path import SVGPath
        return SVGPath([self]).draw(*args, **kwargs)

    def reverse(self):
        raise NotImplementedError

    def is_left_to(self, other: SVGCommand):
        p1, p2 = self.start_pos, other.start_pos

        if p1.y == p2.y:
            return p1.x < p2.x

        return p1.y < p2.y or (np.isclose(p1.norm(), p2.norm()) and p1.x < p2.x)

    def numericalize(self, n=256):
        raise NotImplementedError

    def get_geoms(self):
        return [self.start_pos, self.end_pos]

    def get_points_viz(self, first=False, last=False):
        from .svg_primitive import SVGCircle
        color = "red" if first else "purple" if last else "deepskyblue"  # "#C4C4C4"
        opacity = 0.75 if first or last else 1.0
        return [SVGCircle(self.end_pos, radius=Radius(0.4), color=color, fill=True, stroke_width=".1", opacity=opacity)]

    def get_handles_viz(self):
        return []

    def sample_points(self, n=10, return_array=False):
        return []

    def split(self, n=2):
        raise NotImplementedError

    def length(self):
        raise NotImplementedError

    def bbox(self):
        raise NotImplementedError


class SVGCommandLinear(SVGCommand):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def to_tensor(self, PAD_VAL=-1):
        cmd_index = SVGTensor.COMMANDS_SIMPLIFIED.index(self.command.value)
        return torch.tensor([cmd_index,
                             *([PAD_VAL] * 5),
                             *self.start_pos.to_tensor(),
                             *([PAD_VAL] * 4),
                             *self.end_pos.to_tensor()])

    def numericalize(self, n=256):
        self.start_pos.numericalize(n)
        self.end_pos.numericalize(n)

    def copy(self):
        return self.__class__(self.start_pos.copy(), self.end_pos.copy())

    def reverse(self):
        return self.__class__(self.end_pos, self.start_pos)

    def split(self, n=2):
        return [self]

    def bbox(self):
        return Bbox(self.start_pos, self.end_pos)


class SVGCommandMove(SVGCommandLinear):
    def __init__(self, start_pos: Point, end_pos: Point=None):
        if end_pos is None:
            start_pos, end_pos = Point(0.), start_pos
        super().__init__(SVGCmdEnum.MOVE_TO, [end_pos], start_pos, end_pos)

    def get_points_viz(self, first=False, last=False):
        from .svg_primitive import SVGLine
        points_viz = super().get_points_viz(first, last)
        points_viz.append(SVGLine(self.start_pos, self.end_pos, color="red", dasharray=0.5))
        return points_viz

    def bbox(self):
        return Bbox(self.end_pos, self.end_pos)


class SVGCommandLine(SVGCommandLinear):
    def __init__(self, start_pos: Point, end_pos: Point):
        super().__init__(SVGCmdEnum.LINE_TO, [end_pos], start_pos, end_pos)

    def sample_points(self, n=10, return_array=False):
        z = np.linspace(0., 1., n)

        if return_array:
            points = (1-z)[:, None] * self.start_pos.pos[None] + z[:, None] * self.end_pos.pos[None]
            return points

        points = [(1 - alpha) * self.start_pos + alpha * self.end_pos for alpha in z]
        return points

    def split(self, n=2):
        points = self.sample_points(n+1)
        return [SVGCommandLine(p1, p2) for p1, p2 in zip(points[:-1], points[1:])]

    def length(self):
        return self.start_pos.dist(self.end_pos)


class SVGCommandClose(SVGCommandLinear):
    def __init__(self, start_pos: Point, end_pos: Point):
        super().__init__(SVGCmdEnum.CLOSE_PATH, [], start_pos, end_pos)

    def get_points_viz(self, first=False, last=False):
        return []


class SVGCommandBezier(SVGCommand):
    def __init__(self, start_pos: Point, control1: Point, control2: Point, end_pos: Point):
        if control2 is None:
            control2 = control1.copy()
        super().__init__(SVGCmdEnum.CUBIC_BEZIER, [control1, control2, end_pos], start_pos, end_pos)

        self.control1 = control1
        self.control2 = control2

    @property
    def p1(self):
        return self.start_pos

    @property
    def p2(self):
        return self.end_pos

    @property
    def q1(self):
        return self.control1

    @property
    def q2(self):
        return self.control2

    def copy(self):
        return SVGCommandBezier(self.start_pos.copy(), self.control1.copy(), self.control2.copy(), self.end_pos.copy())

    def to_tensor(self, PAD_VAL=-1):
        cmd_index = SVGTensor.COMMANDS_SIMPLIFIED.index(SVGCmdEnum.CUBIC_BEZIER.value)
        return torch.tensor([cmd_index,
                             *([PAD_VAL] * 5),
                             *self.start_pos.to_tensor(),
                             *self.control1.to_tensor(),
                             *self.control2.to_tensor(),
                             *self.end_pos.to_tensor()])

    def to_vector(self):
        return np.array([
            self.start_pos.tolist(),
            self.control1.tolist(),
            self.control2.tolist(),
            self.end_pos.tolist()
        ])

    @staticmethod
    def from_vector(vector):
        return SVGCommandBezier(Point(vector[0]), Point(vector[1]), Point(vector[2]), Point(vector[3]))

    def reverse(self):
        return SVGCommandBezier(self.end_pos, self.control2, self.control1, self.start_pos)

    def numericalize(self, n=256):
        self.start_pos.numericalize(n)
        self.control1.numericalize(n)
        self.control2.numericalize(n)
        self.end_pos.numericalize(n)

    def get_geoms(self):
        return [self.start_pos, self.control1, self.control2, self.end_pos]

    def get_handles_viz(self):
        from .svg_primitive import SVGLine, SVGCircle
        anchor_1 = SVGCircle(self.control1, radius=Radius(0.4), color="lime", fill=True, stroke_width=".1")
        anchor_2 = SVGCircle(self.control2, radius=Radius(0.4), color="lime", fill=True, stroke_width=".1")

        handle_1 = SVGLine(self.start_pos, self.control1, color="grey", dasharray=0.5, stroke_width=".1")
        handle_2 = SVGLine(self.end_pos, self.control2, color="grey", dasharray=0.5, stroke_width=".1")
        return [handle_1, handle_2, anchor_1, anchor_2]

    def eval(self, t):
        return (1 - t)**3 * self.start_pos + 3 * (1 - t)**2 * t * self.control1 + 3 * (1 - t) * t**2 * self.control2 + t**3 * self.end_pos

    def derivative(self, t, n=1):
        if n == 1:
            return 3 * (1 - t)**2 * (self.control1 - self.start_pos) + 6 * (1 - t) * t * (self.control2 - self.control1) + 3 * t**2 * (self.end_pos - self.control2)
        elif n == 2:
            return 6 * (1 - t) * (self.control2 - 2 * self.control1 + self.start_pos) + 6 * t * (self.end_pos - 2 * self.control2 + self.control1)

        raise NotImplementedError

    def angle(self, other: SVGCommandBezier):
        t1, t2 = self.derivative(1.), -other.derivative(0.)
        if np.isclose(t1.norm(), 0.) or np.isclose(t2.norm(), 0.):
            return 0.
        angle = np.arccos(np.clip(t1.normalize().dot(t2.normalize()), -1., 1.))
        return np.rad2deg(angle)

    def sample_points(self, n=10, return_array=False):
        b = self.to_vector()

        z = np.linspace(0., 1., n)
        Z = np.stack([np.ones_like(z), z, z**2, z**3], axis=1)
        Q = np.array([[1., 0., 0., 0.],
                      [-3, 3., 0., 0.],
                      [3., -6, 3., 0.],
                      [-1, 3., -3, 1]])

        points = Z @ Q @ b

        if return_array:
            return points

        return [Point(p) for p in points]

    def _split_two(self, z=.5):
        b = self.to_vector()

        Q1 = np.array([[1, 0, 0, 0],
                       [-(z - 1), z, 0, 0],
                       [(z - 1) ** 2, -2 * (z - 1) * z, z ** 2, 0],
                       [-(z - 1) ** 3, 3 * (z - 1) ** 2 * z, -3 * (z - 1) * z ** 2, z ** 3]])
        Q2 = np.array([[-(z - 1) ** 3, 3 * (z - 1) ** 2 * z, -3 * (z - 1) * z ** 2, z ** 3],
                       [0, (z - 1) ** 2, -2 * (z - 1) * z, z ** 2],
                       [0, 0, -(z - 1), z],
                       [0, 0, 0, 1]])

        return SVGCommandBezier.from_vector(Q1 @ b), SVGCommandBezier.from_vector(Q2 @ b)

    def split(self, n=2):
        b_list = []
        b = self

        for i in range(n - 1):
            z = 1. / (n - i)
            b1, b = b._split_two(z)
            b_list.append(b1)
        b_list.append(b)
        return b_list

    def length(self):
        p = self.sample_points(n=100, return_array=True)
        return np.linalg.norm(p[1:] - p[:-1], axis=-1).sum()

    def bbox(self):
        return Bbox.from_points(self.find_extrema())

    def find_roots(self):
        a = 3 * (-self.p1 + 3 * self.q1 - 3 * self.q2 + self.p2)
        b = 6 * (self.p1 - 2 * self.q1 + self.q2)
        c = 3 * (self.q1 - self.p1)

        x_roots, y_roots = get_roots(a.x, b.x, c.x), get_roots(a.y, b.y, c.y)
        roots_cat = [*x_roots, *y_roots]
        roots = [root for root in roots_cat if 0 <= root <= 1]
        return roots

    def find_extrema(self):
        points = [self.start_pos, self.end_pos]
        points.extend([self.eval(root) for root in self.find_roots()])
        return points


class SVGCommandArc(SVGCommand):
    def __init__(self, start_pos: Point, radius: Radius, x_axis_rotation: Angle, large_arc_flag: Flag, sweep_flag: Flag, end_pos: Point):
        super().__init__(SVGCmdEnum.ELLIPTIC_ARC, [radius, x_axis_rotation, large_arc_flag, sweep_flag, end_pos], start_pos, end_pos)

        self.radius = radius
        self.x_axis_rotation = x_axis_rotation
        self.large_arc_flag = large_arc_flag
        self.sweep_flag = sweep_flag

    def copy(self):
        return SVGCommandArc(self.start_pos.copy(), self.radius.copy(), self.x_axis_rotation.copy(), self.large_arc_flag.copy(),
                             self.sweep_flag.copy(), self.end_pos.copy())

    def to_tensor(self, PAD_VAL=-1):
        cmd_index = SVGTensor.COMMANDS_SIMPLIFIED.index(SVGCmdEnum.ELLIPTIC_ARC.value)
        return torch.tensor([cmd_index,
                             *self.radius.to_tensor(),
                             *self.x_axis_rotation.to_tensor(),
                             *self.large_arc_flag.to_tensor(),
                             *self.sweep_flag.to_tensor(),
                             *self.start_pos.to_tensor(),
                             *([PAD_VAL] * 4),
                             *self.end_pos.to_tensor()])

    def _get_center_parametrization(self):
        r = self.radius
        p1, p2 = self.start_pos, self.end_pos

        h, m = 0.5 * (p1 - p2), 0.5 * (p1 + p2)
        p1_trans = h.rotate(-self.x_axis_rotation)

        sign = -1 if self.large_arc_flag.flag == self.sweep_flag.flag else 1
        x2, y2, rx2, ry2 = p1_trans.x**2, p1_trans.y**2, r.x**2, r.y**2
        sqrt = math.sqrt(max((rx2*ry2 - rx2*y2 - ry2*x2) / (rx2*y2 + ry2*x2), 0.))
        c_trans = sign * sqrt * Point(r.x * p1_trans.y / r.y, -r.y * p1_trans.x / r.x)

        c = c_trans.rotate(self.x_axis_rotation) + m

        d, ns = (p1_trans - c_trans) / r, -(p1_trans + c_trans) / r

        theta_1 = Point(1, 0).angle(d, signed=True)

        delta_theta = d.angle(ns, signed=True)
        delta_theta.deg %= 360
        if self.sweep_flag.flag == 0 and delta_theta.deg > 0:
            delta_theta = delta_theta - Angle(360)
        if self.sweep_flag == 1 and delta_theta.deg < 0:
            delta_theta = delta_theta + Angle(360)

        return c, theta_1, delta_theta

    def _get_point(self, c: Point, t: float_type):
        r = self.radius
        return c + Point(r.x * np.cos(t), r.y * np.sin(t)).rotate(self.x_axis_rotation)

    def _get_derivative(self, t: float_type):
        r = self.radius
        return Point(-r.x * np.sin(t), r.y * np.cos(t)).rotate(self.x_axis_rotation)

    def to_beziers(self):
        """ References:
        https://www.w3.org/TR/2018/CR-SVG2-20180807/implnote.html
        https://mortoray.com/2017/02/16/rendering-an-svg-elliptical-arc-as-bezier-curves/
        http://www.spaceroots.org/documents/ellipse/elliptical-arc.pdf """
        beziers = []

        c, theta_1, delta_theta = self._get_center_parametrization()
        nb_curves = max(int(abs(delta_theta.deg) // 45), 1)
        etas = [theta_1 + i * delta_theta / nb_curves for i in range(nb_curves+1)]
        for eta_1, eta_2 in zip(etas[:-1], etas[1:]):
            e1, e2 = eta_1.rad, eta_2.rad
            alpha = np.sin(e2 - e1) * (math.sqrt(4 + 3 * np.tan(0.5 * (e2 - e1))**2) - 1) / 3
            p1, p2 = self._get_point(c, e1), self._get_point(c, e2)
            q1 = p1 + alpha * self._get_derivative(e1)
            q2 = p2 - alpha * self._get_derivative(e2)
            beziers.append(SVGCommandBezier(p1, q1, q2, p2))

        return beziers

    def reverse(self):
        return SVGCommandArc(self.end_pos, self.radius, self.x_axis_rotation, self.large_arc_flag, ~self.sweep_flag, self.start_pos)

    def numericalize(self, n=256):
        raise NotImplementedError

    def get_geoms(self):
        return [self.start_pos, self.radius, self.x_axis_rotation, self.large_arc_flag, self.sweep_flag, self.end_pos]

    def split(self, n=2):
        raise NotImplementedError

    def sample_points(self, n=10, return_array=False):
        raise NotImplementedError
