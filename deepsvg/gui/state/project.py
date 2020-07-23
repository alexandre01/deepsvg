import os
import uuid
import json
import numpy as np
from moviepy.editor import ImageClip, concatenate_videoclips
import shutil

from deepsvg.svglib.svg import SVG
from deepsvg.svglib.geom import Bbox

from ..config import ROOT_DIR


class Frame:
    def __init__(self, index, keyframe=False, svg=None):
        self.index = index
        self.keyframe = keyframe

        if svg is None:
            svg = SVG([], viewbox=Bbox(256))
        self.svg = svg

        self.kivy_bezierpaths = None

    def to_dict(self):
        return {
            "index": self.index,
            "keyframe": self.keyframe
        }

    @staticmethod
    def load_dict(frame):
        f = Frame(frame["index"], frame["keyframe"])
        return f


class DeepSVGProject:
    def __init__(self, name="Title"):
        self.name = name
        self.uid = str(uuid.uuid4())

        self.frames = [Frame(index=0)]

    @property
    def filename(self):
        return os.path.join(ROOT_DIR, f"{self.uid}.json")

    @property
    def base_dir(self):
        base_dir = os.path.join(ROOT_DIR, self.uid)

        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        return base_dir

    @property
    def cache_dir(self):
        cache_dir = os.path.join(self.base_dir, "cache")

        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        return cache_dir

    def load_project(self, file_path):
        with open(file_path, "r") as f:
            data = json.load(f)

            self.name = data["name"]
            self.uid = data["uid"]

            self.load_frames(data["frames"])

        shutil.rmtree(self.cache_dir)

    def load_frames(self, frames):
        self.frames = [Frame.load_dict(frame) for frame in frames]

        for frame in self.frames:
            frame.svg = SVG.load_svg(os.path.join(self.base_dir, f"{frame.index}.svg"))

    def save_project(self):
        with open(self.filename, "w") as f:
            data = {
                "name": self.name,
                "uid": self.uid,

                "frames": [frame.to_dict() for frame in self.frames]
            }

            json.dump(data, f)

        self.save_frames()

    def save_frames(self):
        for frame in self.frames:
            frame.svg.save_svg(os.path.join(self.base_dir, f"{frame.index}.svg"))

    def export_to_gif(self, frame_duration=0.1, loop_mode=0):
        from .state import LoopMode

        imgs = [frame.svg.copy().normalize().draw(do_display=False, return_png=True) for frame in self.frames]

        if loop_mode == LoopMode.REVERSE:
            imgs = imgs[::-1]
        elif loop_mode == LoopMode.PINGPONG:
            imgs = imgs + imgs[::-1]

        clips = [ImageClip(np.array(img)).set_duration(frame_duration) for img in imgs]

        clip = concatenate_videoclips(clips, method="compose", bg_color=(255, 255, 255))

        file_path = os.path.join(ROOT_DIR, f"{self.uid}.gif")
        clip.write_gif(file_path, fps=24, verbose=False, logger=None)
