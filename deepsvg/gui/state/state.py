from .project import DeepSVGProject
from ..config import STATE_PATH
import pickle
import os


class ToolMode:
    MOVE = 0
    PEN = 1
    PENCIL = 2
    PLAY = 3


class DrawMode:
    STILL = 0
    DRAW = 1
    HOLDING_DOWN = 2


class LoopMode:
    NORMAL = 0
    REVERSE = 1
    PINGPONG = 2


class PlaybackMode:
    NORMAL = 0
    EASE = 1


class LoopOrientation:
    FORWARD = 1
    BACKWARD = -1


class State:
    def __init__(self):
        self.project_file = None
        self.project = DeepSVGProject()

        self.loop_mode = LoopMode.PINGPONG
        self.loop_orientation = LoopOrientation.FORWARD
        self.playback_mode = PlaybackMode.EASE

        self.delay = 1 / 10.

        self.modified = False

        # Keep track of previously selected current_frame, separately from timeline's selected_frame attribute
        self.current_frame = -1

        self.current_path = None
        self.draw_mode = DrawMode.STILL

        self.clipboard = None

        # UI references
        self.main_widget = None
        self.header = None
        self.sidebar = None
        self.draw_viewbox = None
        self.timeline = None

    def save_state(self):
        with open(STATE_PATH, "wb") as f:
            state_dict = {k: v for k, v in self.__dict__.items() if k in ["project_file"]}
            pickle.dump(state_dict, f)

    def load_state(self):
        if os.path.exists(STATE_PATH):
            with open(STATE_PATH, "rb") as f:
                self.__dict__.update(pickle.load(f))

    def load_project(self):
        if self.project_file is not None:
            self.project.load_project(self.project_file)
        else:
            self.project_file = self.project.filename
