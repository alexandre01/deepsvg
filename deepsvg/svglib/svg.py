from __future__ import annotations
from .geom import *
from xml.dom import expatbuilder
import torch
from typing import List, Union
import IPython.display as ipd
import cairosvg
from PIL import Image
import io
import os
from moviepy.editor import ImageClip, concatenate_videoclips, ipython_display
import math
import random
import networkx as nx

Num = Union[int, float]

from .svg_command import SVGCommandBezier
from .svg_path import SVGPath, Filling, Orientation
from .svg_primitive import SVGPathGroup, SVGRectangle, SVGCircle, SVGEllipse, SVGLine, SVGPolyline, SVGPolygon
from .geom import union_bbox


class SVG:
    def __init__(self, svg_path_groups: List[SVGPathGroup], viewbox: Bbox = None):
        if viewbox is None:
            viewbox = Bbox(24)

        self.svg_path_groups = svg_path_groups
        self.viewbox = viewbox

    def __add__(self, other: SVG):
        svg = self.copy()
        svg.svg_path_groups.extend(other.svg_path_groups)
        return svg

    @property
    def paths(self):
        for path_group in self.svg_path_groups:
            for path in path_group.svg_paths:
                yield path

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            assert len(idx) == 2, "Dimension out of range"
            i, j = idx
            return self.svg_path_groups[i][j]

        return self.svg_path_groups[idx]

    def __len__(self):
        return len(self.svg_path_groups)

    def total_length(self):
        return sum([path_group.total_len() for path_group in self.svg_path_groups])

    @property
    def start_pos(self):
        return Point(0.)

    @property
    def end_pos(self):
        if not self.svg_path_groups:
            return Point(0.)

        return self.svg_path_groups[-1].end_pos

    def copy(self):
        return SVG([svg_path_group.copy() for svg_path_group in self.svg_path_groups], self.viewbox.copy())

    @staticmethod
    def load_svg(file_path):
        with open(file_path, "r") as f:
            return SVG.from_str(f.read())

    @staticmethod
    def load_splineset(spline_str: str, width, height, add_closing=True):
        if "SplineSet" not in spline_str:
            raise ValueError("Not a SplineSet")

        spline = spline_str[spline_str.index('SplineSet') + 10:spline_str.index('EndSplineSet')]
        svg_str = SVG._spline_to_svg_str(spline, height)

        if not svg_str:
            raise ValueError("Empty SplineSet")

        svg_path_group = SVGPath.from_str(svg_str, add_closing=add_closing)
        return SVG([svg_path_group], viewbox=Bbox(width, height))

    @staticmethod
    def _spline_to_svg_str(spline_str: str, height, replace_with_prev=False):
        path = []
        prev_xy = []
        for line in spline_str.splitlines():
            if not line:
                continue
            tokens = line.split(' ')
            cmd = tokens[-2]
            if cmd not in 'cml':
                raise ValueError(f"Command not recognized: {cmd}")
            args = tokens[:-2]
            args = [float(x) for x in args if x]

            if replace_with_prev and cmd in 'c':
                args[:2] = prev_xy
            prev_xy = args[-2:]

            new_y_args = []
            for i, a in enumerate(args):
                if i % 2 == 1:
                    new_y_args.append(str(height - a))
                else:
                    new_y_args.append(str(a))

            path.extend([cmd.upper()] + new_y_args)
        return " ".join(path)

    @staticmethod
    def from_str(svg_str: str):
        svg_path_groups = []
        svg_dom = expatbuilder.parseString(svg_str, False)
        svg_root = svg_dom.getElementsByTagName('svg')[0]

        viewbox_list = list(map(float, svg_root.getAttribute("viewBox").split(" ")))
        view_box = Bbox(*viewbox_list)

        primitives = {
            "path": SVGPath,
            "rect": SVGRectangle,
            "circle": SVGCircle, "ellipse": SVGEllipse,
            "line": SVGLine,
            "polyline": SVGPolyline, "polygon": SVGPolygon
        }

        for tag, Primitive in primitives.items():
            for x in svg_dom.getElementsByTagName(tag):
                svg_path_groups.append(Primitive.from_xml(x))

        return SVG(svg_path_groups, view_box)

    def to_tensor(self, concat_groups=True, PAD_VAL=-1):
        group_tensors = [p.to_tensor(PAD_VAL=PAD_VAL) for p in self.svg_path_groups]

        if concat_groups:
            return torch.cat(group_tensors, dim=0)

        return group_tensors

    def to_fillings(self):
        return [p.path.filling for p in self.svg_path_groups]

    @staticmethod
    def from_tensor(tensor: torch.Tensor, viewbox: Bbox = None, allow_empty=False):
        if viewbox is None:
            viewbox = Bbox(24)

        svg = SVG([SVGPath.from_tensor(tensor, allow_empty=allow_empty)], viewbox=viewbox)
        return svg

    @staticmethod
    def from_tensors(tensors: List[torch.Tensor], viewbox: Bbox = None, allow_empty=False):
        if viewbox is None:
            viewbox = Bbox(24)

        svg = SVG([SVGPath.from_tensor(t, allow_empty=allow_empty) for t in tensors], viewbox=viewbox)
        return svg

    def save_svg(self, file_path):
        with open(file_path, "w") as f:
            f.write(self.to_str())

    def save_png(self, file_path):
        cairosvg.svg2png(bytestring=self.to_str(), write_to=file_path)

    def draw(self, fill=False, file_path=None, do_display=True, return_png=False,
             with_points=False, with_handles=False, with_bboxes=False, with_markers=False, color_firstlast=False,
             with_moves=True):
        if file_path is not None:
            _, file_extension = os.path.splitext(file_path)
            if file_extension == ".svg":
                self.save_svg(file_path)
            elif file_extension == ".png":
                self.save_png(file_path)
            else:
                raise ValueError(f"Unsupported file_path extension {file_extension}")

        svg_str = self.to_str(fill=fill, with_points=with_points, with_handles=with_handles, with_bboxes=with_bboxes,
                              with_markers=with_markers, color_firstlast=color_firstlast, with_moves=with_moves)

        if do_display:
            ipd.display(ipd.SVG(svg_str))

        if return_png:
            if file_path is None:
                img_data = cairosvg.svg2png(bytestring=svg_str)
                return Image.open(io.BytesIO(img_data))
            else:
                _, file_extension = os.path.splitext(file_path)

                if file_extension == ".svg":
                    img_data = cairosvg.svg2png(url=file_path)
                    return Image.open(io.BytesIO(img_data))
                else:
                    return Image.open(file_path)

    def draw_colored(self, *args, **kwargs):
        self.copy().normalize().split_paths().set_color("random").draw(*args, **kwargs)

    def __repr__(self):
        return "SVG[{}](\n{}\n)".format(self.viewbox,
                                        ",\n".join([f"\t{svg_path_group}" for svg_path_group in self.svg_path_groups]))

    def _get_viz_elements(self, with_points=False, with_handles=False, with_bboxes=False, color_firstlast=False,
                          with_moves=True):
        viz_elements = []
        for svg_path_group in self.svg_path_groups:
            viz_elements.extend(
                svg_path_group._get_viz_elements(with_points, with_handles, with_bboxes, color_firstlast, with_moves))
        return viz_elements

    def _markers(self):
        return ('<defs>'
                '<marker id="arrow" viewBox="0 0 10 10" markerWidth="4" markerHeight="4" refX="0" refY="3" orient="auto" markerUnits="strokeWidth">'
                '<path d="M0,0 L0,6 L9,3 z" fill="#f00" />'
                '</marker>'
                '</defs>')

    def to_str(self, fill=False, with_points=False, with_handles=False, with_bboxes=False, with_markers=False,
               color_firstlast=False, with_moves=True) -> str:
        viz_elements = self._get_viz_elements(with_points, with_handles, with_bboxes, color_firstlast, with_moves)
        newline = "\n"
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{self.viewbox.to_str()}" height="200px" width="200px">'
            f'{self._markers() if with_markers else ""}'
            f'{newline.join(svg_path_group.to_str(fill=fill, with_markers=with_markers) for svg_path_group in [*self.svg_path_groups, *viz_elements])}'
            '</svg>')

    def _apply_to_paths(self, method, *args, **kwargs):
        for path_group in self.svg_path_groups:
            getattr(path_group, method)(*args, **kwargs)
        return self

    def split_paths(self):
        path_groups = []
        for path_group in self.svg_path_groups:
            path_groups.extend(path_group.split_paths())
        self.svg_path_groups = path_groups
        return self

    def merge_groups(self):
        path_group = self.svg_path_groups[0]
        for path_group in self.svg_path_groups[1:]:
            path_group.svg_paths.extend(path_group.svg_paths)
        self.svg_path_groups = [path_group]
        return self

    def empty(self):
        return len(self.svg_path_groups) == 0

    def drop_z(self):
        return self._apply_to_paths("drop_z")

    def filter_empty(self):
        self._apply_to_paths("filter_empty")
        self.svg_path_groups = [path_group for path_group in self.svg_path_groups if path_group.svg_paths]
        return self

    def translate(self, vec: Point):
        return self._apply_to_paths("translate", vec)

    def rotate(self, angle: Angle, center: Point = None):
        if center is None:
            center = self.viewbox.center

        self.translate(-self.viewbox.center)
        self._apply_to_paths("rotate", angle)
        self.translate(center)

        return self

    def zoom(self, factor, center: Point = None):
        if center is None:
            center = self.viewbox.center

        self.translate(-self.viewbox.center)
        self._apply_to_paths("scale", factor)
        self.translate(center)

        return self

    def normalize(self, viewbox: Bbox = None):
        if viewbox is None:
            viewbox = Bbox(24)

        size = self.viewbox.size
        scale_factor = viewbox.size.min() / size.max()
        self.zoom(scale_factor, viewbox.center)
        self.viewbox = viewbox

        return self

    def compute_filling(self):
        return self._apply_to_paths("compute_filling")

    def recompute_origins(self):
        origin = self.start_pos

        for path_group in self.svg_path_groups:
            path_group.set_origin(origin.copy())
            origin = path_group.end_pos

    def canonicalize_new(self, normalize=False):
        self.to_path().simplify_arcs()

        self.compute_filling()

        if normalize:
            self.normalize()

        self.split_paths()

        self.filter_consecutives()
        self.filter_empty()
        self._apply_to_paths("reorder")
        self.svg_path_groups = sorted(self.svg_path_groups, key=lambda x: x.start_pos.tolist()[::-1])
        self._apply_to_paths("canonicalize")
        self.recompute_origins()

        self.drop_z()

        return self

    def canonicalize(self, normalize=False):
        self.to_path().simplify_arcs()

        if normalize:
            self.normalize()

        self.split_paths()
        self.filter_consecutives()
        self.filter_empty()
        self._apply_to_paths("reorder")
        self.svg_path_groups = sorted(self.svg_path_groups, key=lambda x: x.start_pos.tolist()[::-1])
        self._apply_to_paths("canonicalize")
        self.recompute_origins()

        self.drop_z()

        return self

    def reorder(self):
        return self._apply_to_paths("reorder")

    def canonicalize_old(self):
        self.filter_empty()
        self._apply_to_paths("reorder")
        self.svg_path_groups = sorted(self.svg_path_groups, key=lambda x: x.start_pos.tolist()[::-1])
        self._apply_to_paths("canonicalize")
        self.split_paths()
        self.recompute_origins()

        self.drop_z()

        return self

    def to_video(self, wrapper, color="grey"):
        clips, svg_commands = [], []

        im = SVG([]).draw(do_display=False, return_png=True)
        clips.append(wrapper(np.array(im)))

        for svg_path in self.paths:
            clips, svg_commands = svg_path.to_video(wrapper, clips, svg_commands, color=color)

        im = self.draw(do_display=False, return_png=True)
        clips.append(wrapper(np.array(im)))

        return clips

    def animate(self, file_path=None, frame_duration=0.1, do_display=True):
        clips = self.to_video(lambda img: ImageClip(img).set_duration(frame_duration))

        clip = concatenate_videoclips(clips, method="compose", bg_color=(255, 255, 255))

        if file_path is not None:
            clip.write_gif(file_path, fps=24, verbose=False, logger=None)

        if do_display:
            src = clip if file_path is None else file_path
            ipd.display(ipython_display(src, fps=24, rd_kwargs=dict(logger=None), autoplay=1, loop=1))

    def numericalize(self, n=256):
        self.normalize(viewbox=Bbox(n))
        return self._apply_to_paths("numericalize", n)

    def simplify(self, tolerance=0.1, epsilon=0.1, angle_threshold=179., force_smooth=False):
        self._apply_to_paths("simplify", tolerance=tolerance, epsilon=epsilon, angle_threshold=angle_threshold,
                             force_smooth=force_smooth)
        self.recompute_origins()
        return self

    def reverse(self):
        self._apply_to_paths("reverse")
        return self

    def reverse_non_closed(self):
        self._apply_to_paths("reverse_non_closed")
        return self

    def duplicate_extremities(self):
        self._apply_to_paths("duplicate_extremities")
        return self

    def simplify_heuristic(self, tolerance=0.1, force_smooth=False):
        return self.copy().split(max_dist=2, include_lines=False) \
            .simplify(tolerance=tolerance, epsilon=0.2, angle_threshold=150, force_smooth=force_smooth) \
            .split(max_dist=7.5)

    def simplify_heuristic2(self):
        return self.copy().split(max_dist=2, include_lines=False) \
            .simplify(tolerance=0.2, epsilon=0.2, angle_threshold=150) \
            .split(max_dist=7.5)

    def split(self, n=None, max_dist=None, include_lines=True):
        return self._apply_to_paths("split", n=n, max_dist=max_dist, include_lines=include_lines)

    @staticmethod
    def unit_circle():
        d = 2 * (math.sqrt(2) - 1) / 3

        circle = SVGPath([
            SVGCommandBezier(Point(.5, 0.), Point(.5 + d, 0.), Point(1., .5 - d), Point(1., .5)),
            SVGCommandBezier(Point(1., .5), Point(1., .5 + d), Point(.5 + d, 1.), Point(.5, 1.)),
            SVGCommandBezier(Point(.5, 1.), Point(.5 - d, 1.), Point(0., .5 + d), Point(0., .5)),
            SVGCommandBezier(Point(0., .5), Point(0., .5 - d), Point(.5 - d, 0.), Point(.5, 0.))
        ]).to_group()

        return SVG([circle], viewbox=Bbox(1))

    @staticmethod
    def unit_square():
        square = SVGPath.from_str("m 0,0 h1 v1 h-1 v-1")
        return SVG([square], viewbox=Bbox(1))

    def add_path_group(self, path_group: SVGPathGroup):
        path_group.set_origin(self.end_pos.copy())
        self.svg_path_groups.append(path_group)

        return self

    def add_path_groups(self, path_groups: List[SVGPathGroup]):
        for path_group in path_groups:
            self.add_path_group(path_group)

        return self

    def simplify_arcs(self):
        return self._apply_to_paths("simplify_arcs")

    def to_path(self):
        for i, path_group in enumerate(self.svg_path_groups):
            self.svg_path_groups[i] = path_group.to_path()
        return self

    def filter_consecutives(self):
        return self._apply_to_paths("filter_consecutives")

    def filter_duplicates(self):
        return self._apply_to_paths("filter_duplicates")

    def set_color(self, color):
        colors = ["deepskyblue", "lime", "deeppink", "gold", "coral", "darkviolet", "royalblue", "darkmagenta", "teal",
                  "gold",
                  "green", "maroon", "aqua", "grey", "steelblue", "lime", "orange"]

        if color == "random_random":
            random.shuffle(colors)

        if isinstance(color, list):
            colors = color

        for i, path_group in enumerate(self.svg_path_groups):
            if color == "random" or color == "random_random" or isinstance(color, list):
                c = colors[i % len(colors)]
            else:
                c = color
            path_group.color = c
        return self

    def bbox(self):
        return union_bbox([path_group.bbox() for path_group in self.svg_path_groups])

    def overlap_graph(self, threshold=0.95, draw=False):
        G = nx.DiGraph()
        shapes = [group.to_shapely() for group in self.svg_path_groups]

        for i, group1 in enumerate(shapes):
            G.add_node(i)

            if self.svg_path_groups[i].path.filling != Filling.OUTLINE:

                for j, group2 in enumerate(shapes):
                    if i != j and self.svg_path_groups[j].path.filling == Filling.FILL:
                        overlap = group1.intersection(group2).area / group1.area
                        if overlap > threshold:
                            G.add_edge(j, i, weight=overlap)

        if draw:
            pos = nx.spring_layout(G)
            nx.draw_networkx(G, pos, with_labels=True)
            labels = nx.get_edge_attributes(G, 'weight')
            nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
        return G

    def group_overlapping_paths(self):
        G = self.overlap_graph()

        path_groups = []
        root_nodes = [i for i, d in G.in_degree() if d == 0]

        for root in root_nodes:
            if self[root].path.filling == Filling.FILL:
                current = [root]

                while current:
                    n = current.pop(0)

                    fill_neighbors, erase_neighbors = [], []
                    for m in G.neighbors(n):
                        if G.in_degree(m) == 1:
                            if self[m].path.filling == Filling.ERASE:
                                erase_neighbors.append(m)
                            else:
                                fill_neighbors.append(m)
                    G.remove_node(n)

                    path_group = SVGPathGroup([self[n].path.copy().set_orientation(Orientation.CLOCKWISE)], fill=True)
                    if erase_neighbors:
                        for n in erase_neighbors:
                            neighbor = self[n].path.copy().set_orientation(Orientation.COUNTER_CLOCKWISE)
                            path_group.append(neighbor)
                        G.remove_nodes_from(erase_neighbors)

                    path_groups.append(path_group)

                    current.extend(fill_neighbors)

        # Add outlines in the end
        for path_group in self.svg_path_groups:
            if path_group.path.filling == Filling.OUTLINE:
                path_groups.append(path_group)

        return SVG(path_groups)

    def to_points(self, sort=True):
        points = np.concatenate([path_group.to_points() for path_group in self.svg_path_groups])

        if sort:
            ind = np.lexsort((points[:, 0], points[:, 1]))
            points = points[ind]

            # Remove duplicates
            row_mask = np.append([True], np.any(np.diff(points, axis=0), 1))
            points = points[row_mask]

        return points

    def permute(self, indices=None):
        if indices is not None:
            self.svg_path_groups = [self.svg_path_groups[i] for i in indices]
        return self

    def fill_(self, fill=True):
        return self._apply_to_paths("fill_", fill)
