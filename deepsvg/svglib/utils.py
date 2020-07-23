import deepsvg.svglib.svg as svg_lib
from .geom import Bbox, Point
import math
import numpy as np
import IPython.display as ipd
from moviepy.editor import ImageClip, concatenate_videoclips, ipython_display


def make_grid(svgs, num_cols=3, grid_width=24):
    """
        svgs: List[svg_lib.SVG]
    """
    nb_rows = math.ceil(len(svgs) / num_cols)
    grid = svg_lib.SVG([], viewbox=Bbox(grid_width * num_cols, grid_width * nb_rows))

    for i, svg in enumerate(svgs):
        row, col = i // num_cols, i % num_cols
        svg = svg.copy().translate(Point(grid_width * col, grid_width * row))

        grid.add_path_groups(svg.svg_path_groups)

    return grid


def make_grid_grid(svg_grid, grid_width=24):
    """
        svg_grid: List[List[svg_lib.SVG]]
    """
    nb_rows = len(svg_grid)
    num_cols = len(svg_grid[0])
    grid = svg_lib.SVG([], viewbox=Bbox(grid_width * num_cols, grid_width * nb_rows))

    for i, row in enumerate(svg_grid):
        for j, svg in enumerate(row):
            svg = svg.copy().translate(Point(grid_width * j, grid_width * i))

            grid.add_path_groups(svg.svg_path_groups)

    return grid


def make_grid_lines(svg_grid, grid_width=24):
    """
        svg_grid: List[List[svg_lib.SVG]]
    """
    nb_rows = len(svg_grid)
    num_cols = max(len(r) for r in svg_grid)
    grid = svg_lib.SVG([], viewbox=Bbox(grid_width * num_cols, grid_width * nb_rows))

    for i, row in enumerate(svg_grid):
        for j, svg in enumerate(row):
            j_shift = (num_cols - len(row)) // 2
            svg = svg.copy().translate(Point(grid_width * (j + j_shift), grid_width * i))

            grid.add_path_groups(svg.svg_path_groups)

    return grid


COLORS = ["aliceblue", "antiquewhite", "aqua", "aquamarine", "azure", "beige", "bisque", "black", "blanchedalmond",
          "blue", "blueviolet", "brown", "burlywood", "cadetblue", "chartreuse", "chocolate", "coral", "cornflowerblue",
          "cornsilk", "crimson", "cyan", "darkblue", "darkcyan", "darkgoldenrod", "darkgray", "darkgreen", "darkgrey",
          "darkkhaki", "darkmagenta", "darkolivegreen", "darkorange", "darkorchid", "darkred", "darksalmon",
          "darkseagreen", "darkslateblue", "darkslategray", "darkslategrey", "darkturquoise", "darkviolet", "deeppink",
          "deepskyblue", "dimgray", "dimgrey", "dodgerblue", "firebrick", "floralwhite", "forestgreen", "fuchsia",
          "gainsboro", "ghostwhite", "gold", "goldenrod", "gray", "green", "greenyellow", "grey", "honeydew", "hotpink",
          "indianred", "indigo", "ivory", "khaki", "lavender", "lavenderblush", "lawngreen", "lemonchiffon",
          "lightblue", "lightcoral", "lightcyan", "lightgoldenrodyellow", "lightgray", "lightgreen", "lightgrey",
          "lightpink", "lightsalmon", "lightseagreen", "lightskyblue", "lightslategray", "lightslategrey",
          "lightsteelblue", "lightyellow", "lime", "limegreen", "linen", "magenta", "maroon", "mediumaquamarine",
          "mediumblue", "mediumorchid", "mediumpurple", "mediumseagreen", "mediumslateblue", "mediumspringgreen",
          "mediumturquoise", "mediumvioletred", "midnightblue", "mintcream", "mistyrose", "moccasin", "navajowhite",
          "navy", "oldlace", "olive", "olivedrab", "orange", "orangered", "orchid", "palegoldenrod", "palegreen",
          "paleturquoise", "palevioletred", "papayawhip", "peachpuff", "peru", "pink", "plum", "powderblue", "purple",
          "red", "rosybrown", "royalblue", "saddlebrown", "salmon", "sandybrown", "seagreen", "seashell", "sienna",
          "silver", "skyblue", "slateblue", "slategray", "slategrey", "snow", "springgreen", "steelblue", "tan", "teal",
          "thistle", "tomato", "turquoise", "violet", "wheat", "white", "whitesmoke", "yellow", "yellowgreen"]


def to_gif(img_list, file_path=None, frame_duration=0.1, do_display=True):
    clips = [ImageClip(np.array(img)).set_duration(frame_duration) for img in img_list]

    clip = concatenate_videoclips(clips, method="compose", bg_color=(255, 255, 255))

    if file_path is not None:
        clip.write_gif(file_path, fps=24, verbose=False, logger=None)

    if do_display:
        src = clip if file_path is None else file_path
        ipd.display(ipython_display(src, fps=24, rd_kwargs=dict(logger=None), autoplay=1, loop=1))
