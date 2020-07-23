from __future__ import annotations
from .geom import *
import deepsvg.svglib.geom as geom
import re
import torch
from typing import List, Union
from xml.dom import minidom
import math
import shapely.geometry
import numpy as np

from .geom import union_bbox
from .svg_command import SVGCommand, SVGCommandMove, SVGCommandClose, SVGCommandBezier, SVGCommandLine, SVGCommandArc


COMMANDS = "MmZzLlHhVvCcSsQqTtAa"
COMMAND_RE = re.compile(r"([MmZzLlHhVvCcSsQqTtAa])")
FLOAT_RE = re.compile(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?")


empty_command = SVGCommandMove(Point(0.))


class Orientation:
    COUNTER_CLOCKWISE = 0
    CLOCKWISE = 1


class Filling:
    OUTLINE = 0
    FILL = 1
    ERASE = 2


class SVGPath:
    def __init__(self, path_commands: List[SVGCommand] = None, origin: Point = None, closed=False, filling=Filling.OUTLINE):
        self.origin = origin or Point(0.)
        self.path_commands = path_commands
        self.closed = closed

        self.filling = filling

    @property
    def start_command(self):
        return SVGCommandMove(self.origin, self.start_pos)

    @property
    def start_pos(self):
        return self.path_commands[0].start_pos

    @property
    def end_pos(self):
        return self.path_commands[-1].end_pos

    def to_group(self, *args, **kwargs):
        from .svg_primitive import SVGPathGroup
        return SVGPathGroup([self], *args, **kwargs)

    def set_filling(self, filling=True):
        self.filling = Filling.FILL if filling else Filling.ERASE
        return self

    def __len__(self):
        return 1 + len(self.path_commands)

    def __getitem__(self, idx):
        if idx == 0:
            return self.start_command
        return self.path_commands[idx-1]

    def all_commands(self, with_close=True):
        close_cmd = [SVGCommandClose(self.path_commands[-1].end_pos.copy(), self.start_pos.copy())] if self.closed and self.path_commands and with_close \
                    else ()
        return [self.start_command, *self.path_commands, *close_cmd]

    def copy(self):
        return SVGPath([path_command.copy() for path_command in self.path_commands], self.origin.copy(), self.closed, filling=self.filling)

    @staticmethod
    def _tokenize_path(path_str):
        cmd = None
        for x in COMMAND_RE.split(path_str):
            if x and x in COMMANDS:
                cmd = x
            elif cmd is not None:
                yield cmd, list(map(float, FLOAT_RE.findall(x)))

    @staticmethod
    def from_xml(x: minidom.Element):
        stroke = x.getAttribute('stroke')
        dasharray = x.getAttribute('dasharray')
        stroke_width = x.getAttribute('stroke-width')

        fill = not x.hasAttribute("fill") or not x.getAttribute("fill") == "none"

        filling = Filling.OUTLINE if not x.hasAttribute("filling") else int(x.getAttribute("filling"))

        s = x.getAttribute('d')
        return SVGPath.from_str(s, fill=fill, filling=filling)

    @staticmethod
    def from_str(s: str, fill=False, filling=Filling.OUTLINE, add_closing=False):
        path_commands = []
        pos = initial_pos = Point(0.)
        prev_command = None
        for cmd, args in SVGPath._tokenize_path(s):
            cmd_parsed, pos, initial_pos = SVGCommand.from_str(cmd, args, pos, initial_pos, prev_command)
            prev_command = cmd_parsed[-1]
            path_commands.extend(cmd_parsed)

        return SVGPath.from_commands(path_commands, fill=fill, filling=filling, add_closing=add_closing)

    @staticmethod
    def from_tensor(tensor: torch.Tensor, allow_empty=False):
        return SVGPath.from_commands([SVGCommand.from_tensor(row) for row in tensor], allow_empty=allow_empty)

    @staticmethod
    def from_commands(path_commands: List[SVGCommand], fill=False, filling=Filling.OUTLINE, add_closing=False, allow_empty=False):
        from .svg_primitive import SVGPathGroup

        if not path_commands:
            return SVGPathGroup([])

        svg_paths = []
        svg_path = None

        for command in path_commands:
            if isinstance(command, SVGCommandMove):
                if svg_path is not None and (allow_empty or svg_path.path_commands):  # SVGPath contains at least one command
                    if add_closing:
                        svg_path.closed = True
                    if not svg_path.path_commands:
                        svg_path.path_commands.append(empty_command)
                    svg_paths.append(svg_path)

                svg_path = SVGPath([], command.start_pos.copy(), filling=filling)
            else:
                if svg_path is None:
                    # Ignore commands until the first moveTo commands
                    continue

                if isinstance(command, SVGCommandClose):
                    if allow_empty or svg_path.path_commands:  # SVGPath contains at least one command
                        svg_path.closed = True
                        if not svg_path.path_commands:
                            svg_path.path_commands.append(empty_command)
                        svg_paths.append(svg_path)
                    svg_path = None
                else:
                    svg_path.path_commands.append(command)
        if svg_path is not None and (allow_empty or svg_path.path_commands):  # SVGPath contains at least one command
            if add_closing:
                svg_path.closed = True
            if not svg_path.path_commands:
                svg_path.path_commands.append(empty_command)
            svg_paths.append(svg_path)
        return SVGPathGroup(svg_paths, fill=fill)

    def __repr__(self):
        return "SVGPath({})".format(" ".join(command.__repr__() for command in self.all_commands()))

    def to_str(self, fill=False):
        return " ".join(command.to_str() for command in self.all_commands())

    def to_tensor(self, PAD_VAL=-1):
        return torch.stack([command.to_tensor(PAD_VAL=PAD_VAL) for command in self.all_commands()])

    def _get_viz_elements(self, with_points=False, with_handles=False, with_bboxes=False, color_firstlast=False, with_moves=True):
        points = self._get_points_viz(color_firstlast, with_moves) if with_points else ()
        handles = self._get_handles_viz() if with_handles else ()
        return [*points, *handles]

    def draw(self, viewbox=Bbox(24), *args, **kwargs):
        from .svg import SVG
        return SVG([self.to_group()], viewbox=viewbox).draw(*args, **kwargs)

    def _get_points_viz(self, color_firstlast=True, with_moves=True):
        points = []
        commands = self.all_commands(with_close=False)
        n = len(commands)
        for i, command in enumerate(commands):
            if not isinstance(command, SVGCommandMove) or with_moves:
                points_viz = command.get_points_viz(first=(color_firstlast and i <= 1), last=(color_firstlast and i >= n-2))
                points.extend(points_viz)
        return points

    def _get_handles_viz(self):
        handles = []
        for command in self.path_commands:
            handles.extend(command.get_handles_viz())
        return handles

    def _get_unique_geoms(self):
        geoms = []
        for command in self.all_commands():
            geoms.extend(command.get_geoms())
        return list(set(geoms))

    def translate(self, vec):
        for geom in self._get_unique_geoms():
            geom.translate(vec)
        return self

    def rotate(self, angle):
        for geom in self._get_unique_geoms():
            geom.rotate_(angle)
        return self

    def scale(self, factor):
        for geom in self._get_unique_geoms():
            geom.scale(factor)
        return self

    def filter_consecutives(self):
        path_commands = []
        for command in self.path_commands:
            if not command.start_pos.isclose(command.end_pos):
                path_commands.append(command)
        self.path_commands = path_commands
        return self

    def filter_duplicates(self, min_dist=0.2):
        path_commands = []
        current_command = None
        for command in self.path_commands:
            if current_command is None:
                path_commands.append(command)
                current_command = command

            if command.end_pos.dist(current_command.end_pos) >= min_dist:
                command.start_pos = current_command.end_pos
                path_commands.append(command)
                current_command = command

        self.path_commands = path_commands
        return self

    def duplicate_extremities(self):
        self.path_commands = [SVGCommandLine(self.start_pos, self.start_pos),
                              *self.path_commands,
                              SVGCommandLine(self.end_pos, self.end_pos)]
        return self

    def is_clockwise(self):
        if len(self.path_commands) == 1:
            cmd = self.path_commands[0]
            return cmd.start_pos.tolist() <= cmd.end_pos.tolist()

        det_total = 0.
        for cmd in self.path_commands:
            det_total += geom.det(cmd.start_pos, cmd.end_pos)
        return det_total >= 0.

    def set_orientation(self, orientation):
        """
        orientation: 1 (clockwise), 0 (counter-clockwise)
        """
        if orientation == self.is_clockwise():
            return self
        return self.reverse()

    def set_closed(self, closed=True):
        self.closed = closed
        return self

    def reverse(self):
        path_commands = []

        for command in reversed(self.path_commands):
            path_commands.append(command.reverse())

        self.path_commands = path_commands
        return self

    def reverse_non_closed(self):
        if not self.start_pos.isclose(self.end_pos):
            return self.reverse()
        return self

    def simplify_arcs(self):
        path_commands = []
        for command in self.path_commands:
            if isinstance(command, SVGCommandArc):
                if command.radius.iszero():
                    continue
                if command.start_pos.isclose(command.end_pos):
                    continue
                path_commands.extend(command.to_beziers())
            else:
                path_commands.append(command)

        self.path_commands = path_commands
        return self

    def _get_topleftmost_command(self):
        topleftmost_cmd = None
        topleftmost_idx = 0

        for i, cmd in enumerate(self.path_commands):
            if topleftmost_cmd is None or cmd.is_left_to(topleftmost_cmd):
                topleftmost_cmd = cmd
                topleftmost_idx = i

        return topleftmost_cmd, topleftmost_idx

    def reorder(self):
        if self.closed:
            topleftmost_cmd, topleftmost_idx = self._get_topleftmost_command()

            self.path_commands = [
                *self.path_commands[topleftmost_idx:],
                *self.path_commands[:topleftmost_idx]
            ]

        return self

    def to_video(self, wrapper, clips=None, svg_commands=None, color="grey"):
        from .svg import SVG
        from .svg_primitive import SVGLine, SVGCircle

        if clips is None:
            clips = []
        if svg_commands is None:
            svg_commands = []
        svg_dots, svg_moves = [], []

        for command in self.all_commands():
            start_pos, end_pos = command.start_pos, command.end_pos

            if isinstance(command, SVGCommandMove):
                move = SVGLine(start_pos, end_pos, color="teal", dasharray=0.5)
                svg_moves.append(move)

            dot = SVGCircle(end_pos, radius=Radius(0.1), color="red")
            svg_dots.append(dot)

            svg_path = SVGPath(svg_commands).to_group(color=color)
            svg_new_path = SVGPath([SVGCommandMove(start_pos), command]).to_group(color="red")

            svg_paths = [svg_path, svg_new_path]  if svg_commands else [svg_new_path]
            im = SVG([*svg_paths, *svg_moves, *svg_dots]).draw(do_display=False, return_png=True, with_points=False)
            clips.append(wrapper(np.array(im)))

            svg_dots[-1].color = "grey"
            svg_commands.append(command)
            svg_moves = []

        return clips, svg_commands

    def numericalize(self, n=256):
        for command in self.all_commands():
            command.numericalize(n)

    def smooth(self):
        # https://github.com/paperjs/paper.js/blob/c7d85b663edb728ec78fffa9f828435eaf78d9c9/src/path/Path.js#L1288
        n = len(self.path_commands)
        knots = [self.start_pos, *(path_commmand.end_pos for path_commmand in self.path_commands)]
        r = [knots[0] + 2 * knots[1]]
        f = [2]
        p = [Point(0.)] * (n + 1)

        # Solve with the Thomas algorithm
        for i in range(1, n):
            internal = i < n - 1
            a = 1
            b = 4 if internal else 2
            u = 4 if internal else 3
            v = 2 if internal else 0
            m = a / f[i-1]

            f.append(b-m)
            r.append(u * knots[i] + v * knots[i + 1] - m * r[i-1])

        p[n-1] = r[n-1] / f[n-1]
        for i in range(n-2, -1, -1):
            p[i] = (r[i] - p[i+1]) / f[i]
        p[n] = (3 * knots[n] - p[n-1]) / 2

        for i in range(n):
            p1, p2 = knots[i], knots[i+1]
            c1, c2 = p[i], 2 * p2 - p[i+1]
            self.path_commands[i] = SVGCommandBezier(p1, c1, c2, p2)

        return self

    def simplify_heuristic(self):
        return self.copy().split(max_dist=2, include_lines=False) \
            .simplify(tolerance=0.1, epsilon=0.2, angle_threshold=150) \
            .split(max_dist=7.5)

    def simplify(self, tolerance=0.1, epsilon=0.1, angle_threshold=179., force_smooth=False):
        # https://github.com/paperjs/paper.js/blob/c044b698c6b224c10a7747664b2a4cd00a416a25/src/path/PathFitter.js#L44
        points = [self.start_pos, *(path_command.end_pos for path_command in self.path_commands)]

        def subdivide_indices():
            segments_list = []
            current_segment = []
            prev_command = None

            for i, command in enumerate(self.path_commands):
                if isinstance(command, SVGCommandLine):
                    if current_segment:
                        segments_list.append(current_segment)
                        current_segment = []
                    prev_command = None

                    continue

                if prev_command is not None and prev_command.angle(command) < angle_threshold:
                    if current_segment:
                        segments_list.append(current_segment)
                        current_segment = []

                current_segment.append(i)
                prev_command = command

            if current_segment:
                segments_list.append(current_segment)

            return segments_list

        path_commands = []

        def computeMaxError(first, last, curve: SVGCommandBezier, u):
            maxDist = 0.
            index = (last - first + 1) // 2
            for i in range(1, last - first):
                dist = curve.eval(u[i]).dist(points[first + i]) ** 2
                if dist >= maxDist:
                    maxDist = dist
                    index = first + i
            return maxDist, index

        def chordLengthParametrize(first, last):
            u = [0.]
            for i in range(1, last - first + 1):
                u.append(u[i-1] + points[first + i].dist(points[first + i-1]))

            for i, _ in enumerate(u[1:], 1):
                u[i] /= u[-1]

            return u

        def isMachineZero(val):
            MACHINE_EPSILON = 1.12e-16
            return val >= -MACHINE_EPSILON and val <= MACHINE_EPSILON

        def findRoot(curve: SVGCommandBezier, point, u):
            """
               Newton's root finding algorithm calculates f(x)=0 by reiterating
               x_n+1 = x_n - f(x_n)/f'(x_n)
               We are trying to find curve parameter u for some point p that minimizes
               the distance from that point to the curve. Distance point to curve is d=q(u)-p.
               At minimum distance the point is perpendicular to the curve.
               We are solving
               f = q(u)-p * q'(u) = 0
               with
               f' = q'(u) * q'(u) + q(u)-p * q''(u)
               gives
               u_n+1 = u_n - |q(u_n)-p * q'(u_n)| / |q'(u_n)**2 + q(u_n)-p * q''(u_n)|
            """
            diff = curve.eval(u) - point
            d1, d2 = curve.derivative(u, n=1), curve.derivative(u, n=2)
            numerator = diff.dot(d1)
            denominator = d1.dot(d1) + diff.dot(d2)

            return u if isMachineZero(denominator) else u - numerator / denominator

        def reparametrize(first, last, u, curve: SVGCommandBezier):
            for i in range(0, last - first + 1):
                u[i] = findRoot(curve, points[first + i], u[i])

            for i in range(1, len(u)):
                if u[i] <= u[i-1]:
                    return False

            return True

        def generateBezier(first, last, uPrime, tan1, tan2):
            epsilon = 1e-12
            p1, p2 = points[first], points[last]
            C = np.zeros((2, 2))
            X = np.zeros(2)

            for i in range(last - first + 1):
                u = uPrime[i]
                t = 1 - u
                b = 3 * u * t
                b0 = t**3
                b1 = b * t
                b2 = b * u
                b3 = u**3
                a1 = tan1 * b1
                a2 = tan2 * b2
                tmp = points[first + i] - p1 * (b0 + b1) - p2 * (b2 + b3)

                C[0, 0] += a1.dot(a1)
                C[0, 1] += a1.dot(a2)
                C[1, 0] = C[0, 1]
                C[1, 1] += a2.dot(a2)
                X[0] += a1.dot(tmp)
                X[1] += a2.dot(tmp)

            detC0C1 = C[0, 0] * C[1, 1] - C[1, 0] * C[0, 1]
            if abs(detC0C1) > epsilon:
                detC0X = C[0, 0] * X[1] - C[1, 0] * X[0]
                detXC1 = X[0] * C[1, 1] - X[1] * C[0, 1]
                alpha1 = detXC1 / detC0C1
                alpha2 = detC0X / detC0C1
            else:
                c0 = C[0, 0] + C[0, 1]
                c1 = C[1, 0] + C[1, 1]
                alpha1 = alpha2 = X[0] / c0 if abs(c0) > epsilon else (X[1] / c1 if abs(c1) > epsilon else 0)

            segLength = p2.dist(p1)
            eps = epsilon * segLength
            handle1 = handle2 = None

            if alpha1 < eps or alpha2 < eps:
                alpha1 = alpha2 = segLength / 3
            else:
                line = p2 - p1
                handle1 = tan1 * alpha1
                handle2 = tan2 * alpha2

                if handle1.dot(line) - handle2.dot(line) > segLength**2:
                    alpha1 = alpha2 = segLength / 3
                    handle1 = handle2 = None

            if handle1 is None or handle2 is None:
                handle1 = tan1 * alpha1
                handle2 = tan2 * alpha2

            return SVGCommandBezier(p1, p1 + handle1, p2 + handle2, p2)

        def computeLinearMaxError(first, last):
            maxDist = 0.
            index = (last - first + 1) // 2

            p1, p2 = points[first], points[last]
            for i in range(first + 1, last):
                dist = points[i].distToLine(p1, p2)
                if dist >= maxDist:
                    maxDist = dist
                    index = i
            return maxDist, index

        def ramerDouglasPeucker(first, last, epsilon):
            max_error, split_index = computeLinearMaxError(first, last)

            if max_error > epsilon:
                ramerDouglasPeucker(first, split_index, epsilon)
                ramerDouglasPeucker(split_index, last, epsilon)
            else:
                p1, p2 = points[first], points[last]
                path_commands.append(SVGCommandLine(p1, p2))

        def fitCubic(error, first, last, tan1=None, tan2=None):
            # For convenience, compute extremity tangents if not provided
            if tan1 is None and tan2 is None:
                tan1 = (points[first + 1] - points[first]).normalize()
                tan2 = (points[last - 1] - points[last]).normalize()

            if last - first == 1:
                p1, p2 = points[first], points[last]
                dist = p1.dist(p2) / 3
                path_commands.append(SVGCommandBezier(p1, p1 + dist * tan1, p2 + dist * tan2, p2))
                return

            uPrime = chordLengthParametrize(first, last)
            maxError = max(error, error**2)
            parametersInOrder = True

            for i in range(5):
                curve = generateBezier(first, last, uPrime, tan1, tan2)

                max_error, split_index = computeMaxError(first, last, curve, uPrime)

                if max_error < error and parametersInOrder:
                    path_commands.append(curve)
                    return

                if max_error >= maxError:
                    break

                parametersInOrder = reparametrize(first, last, uPrime, curve)
                maxError = max_error

            tanCenter = (points[split_index-1] - points[split_index+1]).normalize()
            fitCubic(error, first, split_index, tan1, tanCenter)
            fitCubic(error, split_index, last, -tanCenter, tan2)

        segments_list = subdivide_indices()
        if force_smooth:
            fitCubic(tolerance, 0, len(points) - 1)
        else:
            if segments_list:
                seg = segments_list[0]
                ramerDouglasPeucker(0, seg[0], epsilon)

                for seg, seg_next in zip(segments_list[:-1], segments_list[1:]):
                    fitCubic(tolerance, seg[0], seg[-1] + 1)
                    ramerDouglasPeucker(seg[-1] + 1, seg_next[0], epsilon)

                seg = segments_list[-1]
                fitCubic(tolerance, seg[0], seg[-1] + 1)
                ramerDouglasPeucker(seg[-1] + 1, len(points) - 1, epsilon)
            else:
                ramerDouglasPeucker(0, len(points) - 1, epsilon)

        self.path_commands = path_commands

        return self

    def split(self, n=None, max_dist=None, include_lines=True):
        path_commands = []

        for command in self.path_commands:
            if isinstance(command, SVGCommandLine) and not include_lines:
                path_commands.append(command)
            else:
                l = command.length()
                if max_dist is not None:
                    n = max(math.ceil(l / max_dist), 1)

                path_commands.extend(command.split(n=n))

        self.path_commands = path_commands

        return self

    def bbox(self):
        return union_bbox([cmd.bbox() for cmd in self.path_commands])

    def sample_points(self, max_dist=0.4):
        points = []

        for command in self.path_commands:
            l = command.length()
            n = max(math.ceil(l / max_dist), 1)
            points.extend(command.sample_points(n=n, return_array=True)[None])
        points = np.concatenate(points, axis=0)
        return points

    def to_shapely(self):
        polygon = shapely.geometry.Polygon(self.sample_points())

        if not polygon.is_valid:
            polygon = polygon.buffer(0)

        return polygon

    def to_points(self):
        return np.array([self.start_pos.pos, *(cmd.end_pos.pos for cmd in self.path_commands)])
