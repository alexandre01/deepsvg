from __future__ import annotations
from .geom import *
import torch
import re
from typing import List, Union
from xml.dom import minidom
from .svg_path import SVGPath
from .svg_command import SVGCommandLine, SVGCommandArc, SVGCommandBezier, SVGCommandClose
import shapely
import shapely.ops
import shapely.geometry
import networkx as nx


FLOAT_RE = re.compile(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?")


def extract_args(args):
    return list(map(float, FLOAT_RE.findall(args)))


class SVGPrimitive:
    """
    Reference: https://developer.mozilla.org/en-US/docs/Web/SVG/Tutorial/Basic_Shapes
    """
    def __init__(self, color="black", fill=False, dasharray=None, stroke_width=".3", opacity=1.0):
        self.color = color
        self.dasharray = dasharray
        self.stroke_width = stroke_width
        self.opacity = opacity

        self.fill = fill

    def _get_fill_attr(self):
        fill_attr = f'fill="{self.color}" fill-opacity="{self.opacity}"' if self.fill else f'fill="none" stroke="{self.color}" stroke-width="{self.stroke_width}" stroke-opacity="{self.opacity}"'
        if self.dasharray is not None and not self.fill:
            fill_attr += f' stroke-dasharray="{self.dasharray}"'
        return fill_attr

    @classmethod
    def from_xml(cls, x: minidom.Element):
        raise NotImplementedError

    def draw(self, viewbox=Bbox(24), *args, **kwargs):
        from .svg import SVG
        return SVG([self], viewbox=viewbox).draw(*args, **kwargs)

    def _get_viz_elements(self, with_points=False, with_handles=False, with_bboxes=False, color_firstlast=True, with_moves=True):
        return []

    def to_path(self):
        raise NotImplementedError

    def copy(self):
        raise NotImplementedError

    def bbox(self):
        raise NotImplementedError

    def fill_(self, fill=True):
        self.fill = fill
        return self


class SVGEllipse(SVGPrimitive):
    def __init__(self, center: Point, radius: Radius, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.center = center
        self.radius = radius

    def __repr__(self):
        return f'SVGEllipse(c={self.center} r={self.radius})'

    def to_str(self, *args, **kwargs):
        fill_attr = self._get_fill_attr()
        return f'<ellipse {fill_attr} cx="{self.center.x}" cy="{self.center.y}" rx="{self.radius.x}" ry="{self.radius.y}"/>'

    @classmethod
    def from_xml(_, x: minidom.Element):
        fill = not x.hasAttribute("fill") or not x.getAttribute("fill") == "none"

        center = Point(float(x.getAttribute("cx")), float(x.getAttribute("cy")))
        radius = Radius(float(x.getAttribute("rx")), float(x.getAttribute("ry")))
        return SVGEllipse(center, radius, fill=fill)

    def to_path(self):
        p0, p1 = self.center + self.radius.xproj(), self.center + self.radius.yproj()
        p2, p3 = self.center - self.radius.xproj(), self.center - self.radius.yproj()
        commands = [
            SVGCommandArc(p0, self.radius, Angle(0.), Flag(0.), Flag(1.), p1),
            SVGCommandArc(p1, self.radius, Angle(0.), Flag(0.), Flag(1.), p2),
            SVGCommandArc(p2, self.radius, Angle(0.), Flag(0.), Flag(1.), p3),
            SVGCommandArc(p3, self.radius, Angle(0.), Flag(0.), Flag(1.), p0),
        ]
        return SVGPath(commands, closed=True).to_group(fill=self.fill)


class SVGCircle(SVGEllipse):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __repr__(self):
        return f'SVGCircle(c={self.center} r={self.radius})'

    def to_str(self, *args, **kwargs):
        fill_attr = self._get_fill_attr()
        return f'<circle {fill_attr} cx="{self.center.x}" cy="{self.center.y}" r="{self.radius.x}"/>'

    @classmethod
    def from_xml(_, x: minidom.Element):
        fill = not x.hasAttribute("fill") or not x.getAttribute("fill") == "none"

        center = Point(float(x.getAttribute("cx")), float(x.getAttribute("cy")))
        radius = Radius(float(x.getAttribute("r")))
        return SVGCircle(center, radius, fill=fill)


class SVGRectangle(SVGPrimitive):
    def __init__(self, xy: Point, wh: Size, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.xy = xy
        self.wh = wh

    def __repr__(self):
        return f'SVGRectangle(xy={self.xy} wh={self.wh})'

    def to_str(self, *args, **kwargs):
        fill_attr = self._get_fill_attr()
        return f'<rect {fill_attr} x="{self.xy.x}" y="{self.xy.y}" width="{self.wh.x}" height="{self.wh.y}"/>'

    @classmethod
    def from_xml(_, x: minidom.Element):
        fill = not x.hasAttribute("fill") or not x.getAttribute("fill") == "none"

        xy = Point(0.)
        if x.hasAttribute("x"):
            xy.pos[0] = float(x.getAttribute("x"))
        if x.hasAttribute("y"):
            xy.pos[1] = float(x.getAttribute("y"))
        wh = Size(float(x.getAttribute("width")), float(x.getAttribute("height")))
        return SVGRectangle(xy, wh, fill=fill)

    def to_path(self):
        p0, p1, p2, p3 = self.xy, self.xy + self.wh.xproj(), self.xy + self.wh, self.xy + self.wh.yproj()
        commands = [
            SVGCommandLine(p0, p1),
            SVGCommandLine(p1, p2),
            SVGCommandLine(p2, p3),
            SVGCommandLine(p3, p0)
        ]
        return SVGPath(commands, closed=True).to_group(fill=self.fill)


class SVGLine(SVGPrimitive):
    def __init__(self, start_pos: Point, end_pos: Point, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.start_pos = start_pos
        self.end_pos = end_pos

    def __repr__(self):
        return f'SVGLine(xy1={self.start_pos} xy2={self.end_pos})'

    def to_str(self, *args, **kwargs):
        fill_attr = self._get_fill_attr()
        return f'<line {fill_attr} x1="{self.start_pos.x}" y1="{self.start_pos.y}" x2="{self.end_pos.x}" y2="{self.end_pos.y}"/>'

    @classmethod
    def from_xml(_, x: minidom.Element):
        fill = not x.hasAttribute("fill") or not x.getAttribute("fill") == "none"

        start_pos = Point(float(x.getAttribute("x1") or 0.), float(x.getAttribute("y1") or 0.))
        end_pos = Point(float(x.getAttribute("x2") or 0.), float(x.getAttribute("y2") or 0.))
        return SVGLine(start_pos, end_pos, fill=fill)

    def to_path(self):
        return SVGPath([SVGCommandLine(self.start_pos, self.end_pos)]).to_group(fill=self.fill)


class SVGPolyline(SVGPrimitive):
    def __init__(self, points: List[Point], *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.points = points

    def __repr__(self):
        return f'SVGPolyline(points={self.points})'

    def to_str(self, *args, **kwargs):
        fill_attr = self._get_fill_attr()
        return '<polyline {} points="{}"/>'.format(fill_attr, ' '.join([p.to_str() for p in self.points]))

    @classmethod
    def from_xml(cls, x: minidom.Element):
        fill = not x.hasAttribute("fill") or not x.getAttribute("fill") == "none"

        args = extract_args(x.getAttribute("points"))
        assert len(args) % 2 == 0, f"Expected even number of arguments for SVGPolyline: {len(args)} given"
        points = [Point(x, args[2*i+1]) for i, x in enumerate(args[::2])]
        return cls(points, fill=fill)

    def to_path(self):
        commands = [SVGCommandLine(p1, p2) for p1, p2 in zip(self.points[:-1], self.points[1:])]
        is_closed = self.__class__.__name__ == "SVGPolygon"
        return SVGPath(commands, closed=is_closed).to_group(fill=self.fill)


class SVGPolygon(SVGPolyline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __repr__(self):
        return f'SVGPolygon(points={self.points})'

    def to_str(self, *args, **kwargs):
        fill_attr = self._get_fill_attr()
        return '<polygon {} points="{}"/>'.format(fill_attr, ' '.join([p.to_str() for p in self.points]))


class SVGPathGroup(SVGPrimitive):
    def __init__(self, svg_paths: List[SVGPath] = None, origin=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.svg_paths = svg_paths

        if origin is None:
            origin = Point(0.)
        self.origin = origin

    # Alias
    @property
    def paths(self):
        return self.svg_paths

    @property
    def path(self):
        return self.svg_paths[0]

    def __getitem__(self, idx):
        return self.svg_paths[idx]

    def __len__(self):
        return len(self.paths)

    def total_len(self):
        return sum([len(path) for path in self.svg_paths])

    @property
    def start_pos(self):
        return self.svg_paths[0].start_pos

    @property
    def end_pos(self):
        last_path = self.svg_paths[-1]
        if last_path.closed:
            return last_path.start_pos
        return last_path.end_pos

    def set_origin(self, origin: Point):
        self.origin = origin
        if self.svg_paths:
            self.svg_paths[0].origin = origin
        self.recompute_origins()

    def append(self, path: SVGPath):
        self.svg_paths.append(path)

    def copy(self):
        return SVGPathGroup([svg_path.copy() for svg_path in self.svg_paths], self.origin.copy(),
                            self.color, self.fill, self.dasharray, self.stroke_width, self.opacity)

    def __repr__(self):
        return "SVGPathGroup({})".format(", ".join(svg_path.__repr__() for svg_path in self.svg_paths))

    def _get_viz_elements(self, with_points=False, with_handles=False, with_bboxes=False, color_firstlast=True, with_moves=True):
        viz_elements = []
        for svg_path in self.svg_paths:
            viz_elements.extend(svg_path._get_viz_elements(with_points, with_handles, with_bboxes, color_firstlast, with_moves))

        if with_bboxes:
            viz_elements.append(self._get_bbox_viz())

        return viz_elements

    def _get_bbox_viz(self):
        color = "red" if self.color == "black" else self.color
        bbox = self.bbox().to_rectangle(color=color)
        return bbox

    def to_path(self):
        return self

    def to_str(self, with_markers=False, *args, **kwargs):
        fill_attr = self._get_fill_attr()
        marker_attr = 'marker-start="url(#arrow)"' if with_markers else ''
        return '<path {} {} filling="{}" d="{}"></path>'.format(fill_attr, marker_attr, self.path.filling,
                                                   " ".join(svg_path.to_str() for svg_path in self.svg_paths))

    def to_tensor(self, PAD_VAL=-1):
        return torch.cat([p.to_tensor(PAD_VAL=PAD_VAL) for p in self.svg_paths], dim=0)

    def _apply_to_paths(self, method, *args, **kwargs):
        for path in self.svg_paths:
            getattr(path, method)(*args, **kwargs)
        return self

    def translate(self, vec):
        return self._apply_to_paths("translate", vec)

    def rotate(self, angle: Angle):
        return self._apply_to_paths("rotate", angle)

    def scale(self, factor):
        return self._apply_to_paths("scale", factor)

    def numericalize(self, n=256):
        return self._apply_to_paths("numericalize", n)

    def drop_z(self):
        return self._apply_to_paths("set_closed", False)

    def recompute_origins(self):
        origin = self.origin
        for path in self.svg_paths:
            path.origin = origin.copy()
            origin = path.end_pos
        return self

    def reorder(self):
        self._apply_to_paths("reorder")
        self.recompute_origins()
        return self

    def filter_empty(self):
        self.svg_paths = [path for path in self.svg_paths if path.path_commands]
        return self

    def canonicalize(self):
        self.svg_paths = sorted(self.svg_paths, key=lambda x: x.start_pos.tolist()[::-1])
        if not self.svg_paths[0].is_clockwise():
            self._apply_to_paths("reverse")

        self.recompute_origins()
        return self

    def reverse(self):
        self._apply_to_paths("reverse")

        self.recompute_origins()
        return self

    def duplicate_extremities(self):
        self._apply_to_paths("duplicate_extremities")
        return self

    def reverse_non_closed(self):
        self._apply_to_paths("reverse_non_closed")

        self.recompute_origins()
        return self

    def simplify(self, tolerance=0.1, epsilon=0.1, angle_threshold=179., force_smooth=False):
        self._apply_to_paths("simplify", tolerance=tolerance, epsilon=epsilon, angle_threshold=angle_threshold,
                             force_smooth=force_smooth)
        self.recompute_origins()
        return self

    def split_paths(self):
        return [SVGPathGroup([svg_path], self.origin,
                             self.color, self.fill, self.dasharray, self.stroke_width, self.opacity)
                for svg_path in self.svg_paths]

    def split(self, n=None, max_dist=None, include_lines=True):
        return self._apply_to_paths("split", n=n, max_dist=max_dist, include_lines=include_lines)

    def simplify_arcs(self):
        return self._apply_to_paths("simplify_arcs")

    def filter_consecutives(self):
        return self._apply_to_paths("filter_consecutives")

    def filter_duplicates(self):
        return self._apply_to_paths("filter_duplicates")

    def bbox(self):
        return union_bbox([path.bbox() for path in self.svg_paths])

    def to_shapely(self):
        return shapely.ops.unary_union([path.to_shapely() for path in self.svg_paths])

    def compute_filling(self):
        if self.fill:
            G = self.overlap_graph()

            root_nodes = [i for i, d in G.in_degree() if d == 0]

            for root in root_nodes:
                if not self.svg_paths[root].closed:
                    continue

                current = [(1, root)]

                while current:
                    visited = set()
                    neighbors = set()
                    for d, n in current:
                        self.svg_paths[n].set_filling(d != 0)

                        for n2 in G.neighbors(n):
                            if not n2 in visited:
                                d2 = d + (self.svg_paths[n2].is_clockwise() == self.svg_paths[n].is_clockwise()) * 2 - 1
                                visited.add(n2)
                                neighbors.add((d2, n2))

                    G.remove_nodes_from([n for d, n in current])

                    current = [(d, n) for d, n in neighbors if G.in_degree(n) == 0]

        return self

    def overlap_graph(self, threshold=0.9, draw=False):
        G = nx.DiGraph()
        shapes = [path.to_shapely() for path in self.svg_paths]

        for i, path1 in enumerate(shapes):
            G.add_node(i)

            if self.svg_paths[i].closed:
                for j, path2 in enumerate(shapes):
                    if i != j and self.svg_paths[j].closed:
                        overlap = path1.intersection(path2).area / path1.area
                        if overlap > threshold:
                            G.add_edge(j, i, weight=overlap)

        if draw:
            pos = nx.spring_layout(G)
            nx.draw_networkx(G, pos, with_labels=True)
            labels = nx.get_edge_attributes(G, 'weight')
            nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
        return G

    def bbox_overlap(self, other: SVGPathGroup):
        return self.bbox().overlap(other.bbox())

    def to_points(self):
        return np.concatenate([path.to_points() for path in self.svg_paths])
