import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from configs.deepsvg.hierarchical_ordered import Config

from deepsvg import utils
from deepsvg.svglib.svg import SVG
from deepsvg.difflib.tensor import SVGTensor
from deepsvg.svglib.geom import Bbox
from deepsvg.svgtensor_dataset import load_dataset, SVGFinetuneDataset
from deepsvg.utils.utils import batchify

from .state.project import DeepSVGProject, Frame
from .utils import easein_easeout


device = torch.device("cuda:0"if torch.cuda.is_available() else "cpu")
pretrained_path = "./pretrained/hierarchical_ordered.pth.tar"

cfg = Config()
cfg.model_cfg.dropout = 0.  # for faster convergence
model = cfg.make_model().to(device)
model.eval()


dataset = load_dataset(cfg)


def decode(z):
    commands_y, args_y = model.greedy_sample(z=z)
    tensor_pred = SVGTensor.from_cmd_args(commands_y[0].cpu(), args_y[0].cpu())
    svg_path_sample = SVG.from_tensor(tensor_pred.data, viewbox=Bbox(256))

    return svg_path_sample


def encode_svg(svg):
    data = dataset.get(model_args=[*cfg.model_args, "tensor_grouped"], svg=svg)
    model_args = batchify((data[key] for key in cfg.model_args), device)
    z = model(*model_args, encode_mode=True)
    return z


def interpolate_svg(svg1, svg2, n=10, ease=True):
    z1, z2 = encode_svg(svg1), encode_svg(svg2)

    alphas = torch.linspace(0., 1., n+2)[1:-1]
    if ease:
        alphas = easein_easeout(alphas)

    z_list = [(1 - a) * z1 + a * z2 for a in alphas]
    svgs = [decode(z) for z in z_list]

    return svgs


def finetune_model(project: DeepSVGProject, nb_augmentations=3500):
    keyframe_ids = [i for i, frame in enumerate(project.frames) if frame.keyframe]

    if len(keyframe_ids) < 2:
        return

    svgs = [project.frames[i].svg for i in keyframe_ids]

    utils.load_model(pretrained_path, model)
    print("Finetuning...")
    finetune_dataset = SVGFinetuneDataset(dataset, svgs, frac=1.0, nb_augmentations=nb_augmentations)
    dataloader = DataLoader(finetune_dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=False,
                            num_workers=cfg.loader_num_workers, collate_fn=cfg.collate_fn)

    # Optimizer, lr & warmup schedulers
    optimizers = cfg.make_optimizers(model)
    scheduler_lrs = cfg.make_schedulers(optimizers, epoch_size=len(dataloader))
    scheduler_warmups = cfg.make_warmup_schedulers(optimizers, scheduler_lrs)

    loss_fns = [l.to(device) for l in cfg.make_losses()]

    epoch = 0
    for step, data in enumerate(dataloader):
        model.train()
        model_args = [data[arg].to(device) for arg in cfg.model_args]
        labels = data["label"].to(device) if "label" in data else None
        params_dict, weights_dict = cfg.get_params(step, epoch), cfg.get_weights(step, epoch)

        for i, (loss_fn, optimizer, scheduler_lr, scheduler_warmup, optimizer_start) in enumerate(
                zip(loss_fns, optimizers, scheduler_lrs, scheduler_warmups, cfg.optimizer_starts), 1):
            optimizer.zero_grad()

            output = model(*model_args, params=params_dict)
            loss_dict = loss_fn(output, labels, weights=weights_dict)

            loss_dict["loss"].backward()
            if cfg.grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            optimizer.step()
            if scheduler_lr is not None:
                scheduler_lr.step()
            if scheduler_warmup is not None:
                scheduler_warmup.step()

            if step % 20 == 0:
                print(f"Step {step}: loss: {loss_dict['loss']}")

    print("Finetuning done.")


def compute_interpolation(project: DeepSVGProject):
    finetune_model(project)

    keyframe_ids = [i for i, frame in enumerate(project.frames) if frame.keyframe]

    if len(keyframe_ids) < 2:
        return

    model.eval()

    for i1, i2 in zip(keyframe_ids[:-1], keyframe_ids[1:]):
        frames_inbetween = i2 - i1 - 1
        if frames_inbetween == 0:
            continue

        svgs = interpolate_svg(project.frames[i1].svg, project.frames[i2].svg, n=frames_inbetween, ease=False)
        for di, svg in enumerate(svgs, 1):
            project.frames[i1 + di] = Frame(i1 + di, keyframe=False, svg=svg)
