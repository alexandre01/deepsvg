from deepsvg.config import _Config
from deepsvg import utils
from deepsvg.utils import Stats, TrainVars, Timer
import os
from tensorboardX import SummaryWriter
import torch.nn as nn
import torch
from datetime import datetime
from torch.utils.data import DataLoader
import argparse
import importlib


# Reproducibility
utils.set_seed(42)


def train(cfg: _Config, model_name, experiment_name="", log_dir="./logs", debug=False, resume=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Parameters")
    cfg.print_params()

    print("Loading dataset")
    dataset_load_function = importlib.import_module(cfg.dataloader_module).load_dataset
    dataset = dataset_load_function(cfg)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
                            num_workers=cfg.loader_num_workers, collate_fn=cfg.collate_fn)
    model = cfg.make_model().to(device)

    if cfg.pretrained_path is not None:
        print(f"Loading pretrained model {cfg.pretrained_path}")
        utils.load_model(cfg.pretrained_path, model)

    stats = Stats(num_steps=cfg.num_steps, num_epochs=cfg.num_epochs, steps_per_epoch=len(dataloader),
                  stats_to_print=cfg.stats_to_print)
    train_vars = TrainVars()
    timer = Timer()

    stats.num_parameters = utils.count_parameters(model)
    print(f"#Parameters: {stats.num_parameters:,}")

    # Summary Writer
    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    experiment_identifier = f"{model_name}_{experiment_name}_{current_time}"

    summary_writer = SummaryWriter(os.path.join(log_dir, "tensorboard", "debug" if debug else "full", experiment_identifier))
    checkpoint_dir = os.path.join(log_dir, "models", model_name, experiment_name)
    visualization_dir = os.path.join(log_dir, "visualization", model_name, experiment_name)

    cfg.set_train_vars(train_vars, dataloader)

    # Optimizer, lr & warmup schedulers
    optimizers = cfg.make_optimizers(model)
    scheduler_lrs = cfg.make_schedulers(optimizers, epoch_size=len(dataloader))
    scheduler_warmups = cfg.make_warmup_schedulers(optimizers, scheduler_lrs)

    loss_fns = [l.to(device) for l in cfg.make_losses()]

    if resume:
        ckpt_exists = utils.load_ckpt_list(checkpoint_dir, model, None, optimizers, scheduler_lrs, scheduler_warmups, stats, train_vars)

    if resume and ckpt_exists:
        print(f"Resuming model at epoch {stats.epoch+1}")
        stats.num_steps = cfg.num_epochs * len(dataloader)
    else:
        # Run a single forward pass on the single-device model for initialization of some modules
        single_foward_dataloader = DataLoader(dataset, batch_size=cfg.batch_size // cfg.num_gpus, shuffle=True, drop_last=True,
                                      num_workers=cfg.loader_num_workers, collate_fn=cfg.collate_fn)
        data = next(iter(single_foward_dataloader))
        model_args, params_dict = [data[arg].to(device) for arg in cfg.model_args], cfg.get_params(0, 0)
        model(*model_args, params=params_dict)

    model = nn.DataParallel(model)

    epoch_range = utils.infinite_range(stats.epoch) if cfg.num_epochs is None else range(stats.epoch, cfg.num_epochs)
    for epoch in epoch_range:
        print(f"Epoch {epoch+1}")

        for n_iter, data in enumerate(dataloader):
            step = n_iter + epoch * len(dataloader)

            if cfg.num_steps is not None and step > cfg.num_steps:
                return

            model.train()
            model_args = [data[arg].to(device) for arg in cfg.model_args]
            labels = data["label"].to(device) if "label" in data else None
            params_dict, weights_dict = cfg.get_params(step, epoch), cfg.get_weights(step, epoch)

            for i, (loss_fn, optimizer, scheduler_lr, scheduler_warmup, optimizer_start) in enumerate(zip(loss_fns, optimizers, scheduler_lrs, scheduler_warmups, cfg.optimizer_starts), 1):
                optimizer.zero_grad()

                output = model(*model_args, params=params_dict)
                loss_dict = loss_fn(output, labels, weights=weights_dict)

                if step >= optimizer_start:
                    loss_dict["loss"].backward()
                    if cfg.grad_clip is not None:
                        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

                    optimizer.step()
                    if scheduler_lr is not None:
                        scheduler_lr.step()
                    if scheduler_warmup is not None:
                        scheduler_warmup.step()

                stats.update_stats_to_print("train", loss_dict.keys())
                stats.update("train", step, epoch, {
                    ("lr" if i == 1 else f"lr_{i}"): optimizer.param_groups[0]['lr'],
                    **loss_dict
                })

            stats.update("train", step, epoch, {
                **weights_dict,
                "time": timer.get_elapsed_time()
            })

            if step % cfg.log_every == 0 and step > 0:
                print(stats.get_summary("train"))
                stats.write_tensorboard(summary_writer, "train")
                summary_writer.flush()

            if step % cfg.val_every == 0 and step > 0:
                model.eval()

                with torch.no_grad():
                    # Visualization
                    output = None
                    cfg.visualize(model, output, train_vars, step, epoch, summary_writer, visualization_dir)

                timer.reset()

            if not debug and step % cfg.ckpt_every == 0 and step > 0:
                utils.save_ckpt_list(checkpoint_dir, model, cfg, optimizers, scheduler_lrs, scheduler_warmups, stats, train_vars)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DeepSVG Trainer')
    parser.add_argument("--config-module", type=str, required=True)
    parser.add_argument("--log-dir", type=str, default="./logs")
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--resume", action="store_true", default=False)

    args = parser.parse_args()

    cfg = importlib.import_module(args.config_module).Config()
    model_name, experiment_name = args.config_module.split(".")[-2:]

    train(cfg, model_name, experiment_name, log_dir=args.log_dir, debug=args.debug, resume=args.resume)
