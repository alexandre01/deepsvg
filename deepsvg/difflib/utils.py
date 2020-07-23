import torch
import matplotlib.pyplot as plt
import PIL.Image
import io


def set_viewbox(viewbox):
    plt.xlim(0, viewbox[0])
    plt.ylim(viewbox[1], 0)


def plot_points(p, viewbox=None, show_color=False, show_colorbar=False, image_file=None, return_img=False):
    cm = plt.cm.get_cmap('RdYlBu')
    plt.gca().set_aspect('equal')
    plt.gca().invert_yaxis()
    plt.gca().axis('off')

    if viewbox is not None:
        set_viewbox(viewbox)

    kwargs = {"c": range(len(p)), "cmap": cm} if show_color else {}
    plt.scatter(p[:, 0], p[:, 1], **kwargs)

    if show_color and show_colorbar:
        plt.colorbar()

    if image_file is not None:
        plt.savefig(image_file, bbox_inches='tight')

    if return_img:
        buf = io.BytesIO()
        plt.gcf().savefig(buf)
        buf.seek(0)
        return PIL.Image.open(buf)


def plot_matching(p1, p2, matching, viewbox=None):
    plt.gca().set_aspect('equal')
    plt.gca().invert_yaxis()
    plt.axis("off")

    if viewbox is not None:
        set_viewbox(viewbox)

    plt.scatter(p1[:, 0], p1[:, 1], color="C0")
    plt.scatter(p2[:, 0], p2[:, 1], color="C1")

    for start, end in zip(p1[::10], p2[matching][::10]):
        plt.plot([start[0], end[0]], [start[1], end[1]], color="C2")


def is_clockwise(p):
    start, end = p[:-1], p[1:]
    return torch.stack([start, end], dim=-1).det().sum() > 0


def make_clockwise(p):
    if not is_clockwise(p):
        return p.flip(dims=[0])
    return p


def reorder(p, i):
    return torch.cat([p[i:], p[:i]])


def get_length(p):
    start, end = p[:-1], p[1:]
    return torch.norm(end - start, dim=-1).sum()


def get_length_distribution(p, normalize=True):
    start, end = p[:-1], p[1:]
    length_distr = torch.norm(end - start, dim=-1).cumsum(dim=0)
    length_distr = torch.cat([length_distr.new_zeros(1),
                              length_distr])

    if normalize:
        length_distr = length_distr / length_distr[-1]

    return length_distr
