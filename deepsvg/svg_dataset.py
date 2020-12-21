from deepsvg.config import _Config
from deepsvg.difflib.tensor import SVGTensor
from deepsvg.svglib.svg import SVG
from deepsvg.svglib.geom import Point

import math
import torch
import torch.utils.data
import random
from typing import List, Union
import pandas as pd
import os
import pickle
Num = Union[int, float]


class SVGDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, meta_filepath, model_args, max_num_groups, max_seq_len, max_total_len=None,
                 filter_uni=None, filter_platform=None, filter_category=None, train_ratio=1.0, df=None, PAD_VAL=-1,
                 nb_augmentations=1, already_preprocessed=True):
        self.data_dir = data_dir

        self.already_preprocessed = already_preprocessed

        self.MAX_NUM_GROUPS = max_num_groups
        self.MAX_SEQ_LEN = max_seq_len
        self.MAX_TOTAL_LEN = max_total_len

        if max_total_len is None:
            self.MAX_TOTAL_LEN = max_num_groups * max_seq_len

        if df is None:
            df = pd.read_csv(meta_filepath)

        if len(df) > 0:
            if filter_uni is not None:
                df = df[df.uni.isin(filter_uni)]

            if filter_platform is not None:
                df = df[df.platform.isin(filter_platform)]

            if filter_category is not None:
                df = df[df.category.isin(filter_category)]

            df = df[(df.nb_groups <= max_num_groups) & (df.max_len_group <= max_seq_len)]
            if max_total_len is not None:
                df = df[df.total_len <= max_total_len]

        self.df = df.sample(frac=train_ratio) if train_ratio < 1.0 else df

        self.model_args = model_args

        self.PAD_VAL = PAD_VAL

        self.nb_augmentations = nb_augmentations

    def search_name(self, name):
        return self.df[self.df.commonName.str.contains(name)]

    def _filter_categories(self, filter_category):
        self.df = self.df[self.df.category.isin(filter_category)]

    @staticmethod
    def _uni_to_label(uni):
        if 48 <= uni <= 57:
            return uni - 48
        elif 65 <= uni <= 90:
            return uni - 65 + 10
        return uni - 97 + 36

    @staticmethod
    def _label_to_uni(label_id):
        if 0 <= label_id <= 9:
            return label_id + 48
        elif 10 <= label_id <= 35:
            return label_id + 65 - 10
        return label_id + 97 - 36

    @staticmethod
    def _category_to_label(category):
        categories = ['characters', 'free-icons', 'logos', 'alphabet', 'animals', 'arrows', 'astrology', 'baby', 'beauty',
                      'business', 'cinema', 'city', 'clothing', 'computer-hardware', 'crime', 'cultures', 'data', 'diy',
                      'drinks', 'ecommerce', 'editing', 'files', 'finance', 'folders', 'food', 'gaming', 'hands', 'healthcare',
                      'holidays', 'household', 'industry', 'maps', 'media-controls', 'messaging', 'military', 'mobile',
                      'music', 'nature', 'network', 'photo-video', 'plants', 'printing',  'profile', 'programming', 'science',
                      'security', 'shopping', 'social-networks', 'sports', 'time-and-date', 'transport', 'travel', 'user-interface',
                      'users', 'weather', 'flags', 'emoji', 'men', 'women']
        return categories.index(category)

    def get_label(self, idx=0, entry=None):
        if entry is None:
            entry = self.df.iloc[idx]

        if "uni" in self.df.columns:  # Font dataset
            label = self._uni_to_label(entry.uni)
            return torch.tensor(label)
        elif "category" in self.df.columns:  # Icons dataset
            label = self._category_to_label(entry.category)
            return torch.tensor(label)

        return None

    def idx_to_id(self, idx):
        return self.df.iloc[idx].id

    def entry_from_id(self, id):
        return self.df[self.df.id == str(id)].iloc[0]

    def _load_svg(self, icon_id):
        svg = SVG.load_svg(os.path.join(self.data_dir, f"{icon_id}.svg"))

        if not self.already_preprocessed:
            svg.fill_(False)
            svg.normalize().zoom(0.9)
            svg.canonicalize()
            svg = svg.simplify_heuristic()

        return svg

    def __len__(self):
        return len(self.df) * self.nb_augmentations

    def random_icon(self):
        return self[random.randrange(0, len(self))]

    def random_id(self):
        idx = random.randrange(0, len(self)) % len(self.df)
        return self.idx_to_id(idx)

    def random_id_by_uni(self, uni):
        df = self.df[self.df.uni == uni]
        return df.id.sample().iloc[0]

    def __getitem__(self, idx):
        return self.get(idx, self.model_args)

    @staticmethod
    def _augment(svg, mean=False):
        dx, dy = (0, 0) if mean else (5 * random.random() - 2.5, 5 * random.random() - 2.5)
        factor = 0.7 if mean else 0.2 * random.random() + 0.6

        return svg.zoom(factor).translate(Point(dx, dy))

    @staticmethod
    def simplify(svg, normalize=True):
        svg.canonicalize(normalize=normalize)
        svg = svg.simplify_heuristic()
        return svg.normalize()

    @staticmethod
    def preprocess(svg, augment=True, numericalize=True, mean=False):
        if augment:
            svg = SVGDataset._augment(svg, mean=mean)
        if numericalize:
            return svg.numericalize(256)
        return svg

    def get(self, idx=0, model_args=None, random_aug=True, id=None, svg: SVG=None):
        if id is None:
            idx = idx % len(self.df)
            id = self.idx_to_id(idx)

        if svg is None:
            svg = self._load_svg(id)

            svg = SVGDataset.preprocess(svg, augment=random_aug)

        t_sep, fillings = svg.to_tensor(concat_groups=False, PAD_VAL=self.PAD_VAL), svg.to_fillings()

        label = self.get_label(idx)

        return self.get_data(t_sep, fillings, model_args=model_args, label=label)

    def get_data(self, t_sep, fillings, model_args=None, label=None):
        res = {}

        if model_args is None:
            model_args = self.model_args

        pad_len = max(self.MAX_NUM_GROUPS - len(t_sep), 0)

        t_sep.extend([torch.empty(0, 14)] * pad_len)
        fillings.extend([0] * pad_len)

        t_grouped = [SVGTensor.from_data(torch.cat(t_sep, dim=0), PAD_VAL=self.PAD_VAL).add_eos().add_sos().pad(
            seq_len=self.MAX_TOTAL_LEN + 2)]
        t_sep = [SVGTensor.from_data(t, PAD_VAL=self.PAD_VAL, filling=f).add_eos().add_sos().pad(seq_len=self.MAX_SEQ_LEN + 2) for
                 t, f in zip(t_sep, fillings)]

        for arg in set(model_args):
            if "_grouped" in arg:
                arg_ = arg.split("_grouped")[0]
                t_list = t_grouped
            else:
                arg_ = arg
                t_list = t_sep

            if arg_ == "tensor":
                res[arg] = t_list

            if arg_ == "commands":
                res[arg] = torch.stack([t.cmds() for t in t_list])

            if arg_ == "args_rel":
                res[arg] = torch.stack([t.get_relative_args() for t in t_list])
            if arg_ == "args":
                res[arg] = torch.stack([t.args() for t in t_list])

        if "filling" in model_args:
            res["filling"] = torch.stack([torch.tensor(t.filling) for t in t_sep]).unsqueeze(-1)

        if "label" in model_args:
            res["label"] = label

        return res


def load_dataset(cfg: _Config, already_preprocessed=True):
    dataset = SVGDataset(cfg.data_dir, cfg.meta_filepath, cfg.model_args, cfg.max_num_groups, cfg.max_seq_len, cfg.max_total_len,
                         cfg.filter_uni, cfg.filter_platform, cfg.filter_category, cfg.train_ratio,
                         nb_augmentations=cfg.nb_augmentations, already_preprocessed=already_preprocessed)
    return dataset
