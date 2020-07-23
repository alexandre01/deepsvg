from deepsvg.svglib.svg import SVG
from deepsvg.svglib.svg_path import SVGPath
from deepsvg.svglib.geom import Bbox


color_dict = {
    "deepskyblue": [0., 0.69, 0.97],
    "lime": [0.02, 1., 0.01],
    "deeppink": [1., 0.07, 0.53],
    "gold": [1., 0.81, 0.01],
    "coral": [1., 0.45, 0.27],
    "darkviolet": [0.53, 0.01, 0.8],
    "royalblue": [0.21, 0.36, 0.86],
    "darkmagenta":  [0.5, 0., 0.5],
    "teal": [0., 0.45, 0.45],
    "green": [0., 0.45, 0.],
    "maroon": [0.45, 0., 0.],
    "aqua": [0., 1., 1.],
    "grey": [0.45, 0.45, 0.45],
    "steelblue": [0.24, 0.46, 0.67],
    "orange": [1., 0.6, 0.01]
}

colors = ["deepskyblue", "lime", "deeppink", "gold", "coral", "darkviolet", "royalblue", "darkmagenta", "teal",
          "gold", "green", "maroon", "aqua", "grey", "steelblue", "lime", "orange"]


class Keys:
    LEFT = 276
    UP = 273
    RIGHT = 275
    DOWN = 274

    SPACEBAR = 32


def dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** .5


def preprocess_svg_path(svg_path: SVGPath, force_smooth=False):
    svg = SVG([svg_path.to_group()], viewbox=Bbox(256)).normalize()
    svg.canonicalize()
    svg.filter_duplicates()
    svg = svg.simplify_heuristic(force_smooth=force_smooth)
    svg.normalize()
    svg.numericalize(256)

    return svg[0].path


def normalized_path(svg_path):
    svg = SVG([svg_path.copy().to_group()], viewbox=Bbox(256)).normalize()
    return svg[0].path


def flip_vertical(p):
    return [p[0], 255 - p[1]]


def easein_easeout(t):
    return t * t / (2. * (t * t - t) + 1.)


def d_easein_easeout(t):
    return 3 * (1 - t) * t / (2 * t * t - 2 * t + 1) ** 2
