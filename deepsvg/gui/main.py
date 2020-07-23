from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.scatter import Scatter
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView
from kivy.properties import BooleanProperty, StringProperty, NumericProperty, ListProperty, ObjectProperty
from kivy.uix.behaviors import ButtonBehavior
from kivy.vector import Vector
from kivy.metrics import dp
from kivy.clock import Clock
from kivy.uix.popup import Popup

from kivy.config import Config
Config.set('graphics', 'width', '1400')
Config.set('graphics', 'height', '800')
from kivy.core.window import Window

import os
from typing import List

from deepsvg.svglib.geom import Point
from deepsvg.svglib.svg_command import SVGCommandMove, SVGCommandLine, SVGCommandBezier
from deepsvg.svgtensor_dataset import SVGTensorDataset

from .layout.aligned_textinput import AlignedTextInput
from .state.state import State, ToolMode, DrawMode, LoopMode, PlaybackMode
from .state.project import Frame
from .config import ROOT_DIR
from .interpolate import compute_interpolation
from .utils import *


if not os.path.exists(ROOT_DIR):
    os.makedirs(ROOT_DIR)

state = State()
state.load_state()
state.load_project()


class HeaderIcon(Button):
    index = NumericProperty(0)
    source = StringProperty("")

    def on_press(self):
        state.header.selected_tool = self.index


class Header(BoxLayout):
    selected_tool = NumericProperty(0)
    title = StringProperty(state.project.name)
    is_playing = BooleanProperty(False)
    delay = NumericProperty(state.delay)

    def on_selected_tool(self, *args):
        if self.selected_tool in [ToolMode.MOVE, ToolMode.PEN, ToolMode.PENCIL] and state.header.is_playing:
            state.header.pause_animation()

    def on_done(self, *args):
        if self.selected_tool == ToolMode.PEN and state.draw_mode == DrawMode.DRAW:
            path = state.current_path

            last_segment = path.children[-1]
            path.remove_widget(last_segment)

            state.draw_viewbox.on_path_done(state.current_path)

            state.draw_mode = DrawMode.STILL
            state.current_path = None
            self.selected_tool = ToolMode.MOVE

    def on_erase(self):
        state.modified = True
        state.draw_viewbox.clear()

        state.timeline.make_keyframe(False)

    def add_frame(self, keyframe=False):
        frame_idx = state.timeline._add_frame(keyframe=keyframe)

        state.project.frames.append(Frame(frame_idx, keyframe))

        self.load_next_frame(frame_idx=frame_idx)

    def play_animation(self):
        self.is_playing = True
        state.sidebar.selected_path_idx = -1
        self.clock = Clock.schedule_once(self.load_next_frame)

    def load_next_frame(self, dt=0, frame_idx=None, *args):
        if state.timeline.nb_frames > 0:
            if frame_idx is None:
                frame_idx_tmp = state.timeline.selected_frame + state.loop_orientation

                if frame_idx_tmp < 0 or frame_idx_tmp >= state.timeline.nb_frames:
                    if state.loop_mode in [LoopMode.NORMAL, LoopMode.REVERSE]:
                        frame_idx = frame_idx_tmp % state.timeline.nb_frames
                    else:  # LoopMode.PINGPONG
                        state.loop_orientation *= -1
                        frame_idx = (state.timeline.selected_frame + state.loop_orientation) % state.timeline.nb_frames
                else:
                    frame_idx = frame_idx_tmp

            state.timeline.selected_frame = frame_idx

            if self.is_playing:
                if state.playback_mode == PlaybackMode.EASE:
                    t = frame_idx / state.timeline.nb_frames
                    delay = 2 * state.delay / (1 + d_easein_easeout(t))
                else:
                    delay = state.delay
                self.clock = Clock.schedule_once(self.load_next_frame, delay)

    def pause_animation(self):
        self.clock.cancel()
        state.sidebar.selected_path_idx = -1
        self.is_playing = False

        state.timeline.on_selected_frame()  # re-render frame to display sidebar layers

    def on_title(self, title):
        state.project.name = title

    def interpolate(self):
        state.draw_viewbox.save_frame()

        compute_interpolation(state.project)


class PathLayerView(ButtonBehavior, BoxLayout):
    index = NumericProperty(0)
    source = StringProperty("")

    def __init__(self, index, **kwargs):
        super().__init__(**kwargs)

        self.index = index
        self.source = os.path.join(state.project.cache_dir, f"{state.timeline.selected_frame}_{index}.png")

    def on_press(self):
        state.sidebar.selected_path_idx = self.index

    def move_up(self):
        if self.index > 0:
            state.sidebar.swap_paths(self.index, self.index - 1)

    def move_down(self):
        if self.index < state.sidebar.nb_paths - 1:
            state.sidebar.swap_paths(self.index, self.index + 1)

    def reverse(self):
        state.sidebar.reverse_path(self.index)


class Sidebar(ScrollView):
    selected_path_idx = NumericProperty(-1)

    @property
    def sidebar(self):
        return self.ids.sidebar

    @property
    def nb_paths(self):
        return len(self.sidebar.children)

    def on_selected_path_idx(self, *args):
        state.draw_viewbox.unselect_all()

        if self.selected_path_idx >= 0:
            state.draw_viewbox.get_path(self.selected_path_idx).selected = True

    def _add_path(self, idx=None):
        if idx is None:
            idx = self.nb_paths
        new_pathlayer = PathLayerView(idx)
        self.sidebar.add_widget(new_pathlayer)
        return idx

    def get_path(self, path_idx):
        index = self.nb_paths - 1 - path_idx
        return self.sidebar.children[index]

    def erase(self):
        self.sidebar.clear_widgets()
        self.selected_path_idx = -1

    def swap_paths(self, idx1, idx2):
        path_layer1, path_layer2 = self.get_path(idx1), self.get_path(idx2)
        path1, path2 = state.draw_viewbox.get_path(idx1), state.draw_viewbox.get_path(idx2)

        path_layer1.index, path_layer2.index = idx2, idx1
        path1.color, path2.color = path2.color, path1.color
        path1.index, path2.index = path2.index, path1.index

        id1, id2 = self.nb_paths - 1 - idx1, self.nb_paths - 1 - idx2
        self.sidebar.children[id1], self.sidebar.children[id2] = path_layer2, path_layer1
        state.draw_viewbox.children[id1], state.draw_viewbox.children[id2] = path2, path1

        self.selected_path_idx = idx2
        state.modified = True

    def reverse_path(self, idx):
        path = state.draw_viewbox.get_path(idx)
        svg_path = path.to_svg_path().reverse()
        new_path = BezierPath.from_svg_path(svg_path, color=path.color, index=path.index, selected=path.selected)

        id = self.nb_paths - 1 - idx
        state.draw_viewbox.remove_widget(path)
        state.draw_viewbox.add_widget(new_path, index=id)

        self.selected_path_idx = idx
        state.modified = True

    def select(self, path_idx):
        if self.selected_path_idx >= 0:
            state.draw_viewbox.get_path(state.sidebar.selected_path_idx).selected = False
        self.selected_path_idx = path_idx


class BezierSegment(Widget):
    is_curved = BooleanProperty(True)

    is_finished = BooleanProperty(True)
    select_dist = NumericProperty(3)

    p1 = ListProperty([0, 0])
    q1 = ListProperty([0, 0])
    q2 = ListProperty([0, 0])
    p2 = ListProperty([0, 0])

    def clone(self):
        segment = BezierSegment()
        segment.is_curved = self.is_curved
        segment.p1 = self.p1  # shallow copy
        segment.q1 = self.q1
        segment.q2 = self.q2
        segment.p2 = self.p2
        return segment

    @staticmethod
    def line(p1, p2):
        segment = BezierSegment()
        segment.is_curved = False
        segment.p1 = segment.q1 = p1
        segment.p2 = segment.q2 = p2
        return segment

    @staticmethod
    def bezier(p1, q1, q2, p2):
        segment = BezierSegment()
        segment.is_curved = True
        segment.q1, segment.q2 = q1, q2
        segment.p1, segment.p2 = p1, p2
        return segment

    def get_point(self, key):
        return getattr(self, key)

    def on_touch_down(self, touch):
        max_dist = dp(self.select_dist)
        
        if not self.parent.selected:
            return super().on_touch_down(touch)

        keys_to_test = ["p1", "q1", "q2", "p2"] if self.is_curved else ["p1", "p2"]
        for key in keys_to_test:
            if dist(touch.pos, getattr(self, key)) < max_dist:
                touch.ud['selected'] = key
                touch.grab(self)

                state.modified = True

                return True

    def on_touch_move(self, touch):
        if touch.grab_current is not self:
            return super().on_touch_move(touch)

        key = touch.ud['selected']
        setattr(self, key, touch.pos)

        if state.header.selected_tool == ToolMode.PEN:
            self.is_curved = True
            self.is_finished = False
            state.draw_mode = DrawMode.HOLDING_DOWN

            setattr(self, "p2", touch.pos)

        if key in ["p1", "p2"]:
            self.parent.move(self, key, touch.pos)

    def on_touch_up(self, touch):
        if touch.grab_current is not self:
            return super().on_touch_up(touch)

        touch.ungrab(self)

        if state.header.selected_tool == ToolMode.PEN:
            self.is_finished = True
            state.draw_mode = DrawMode.DRAW


class BezierPath(Widget):
    color = ListProperty([1, 1, 1])
    index = NumericProperty(0)
    selected = BooleanProperty(False)

    def __init__(self, segments: List[BezierSegment], color=None, index=None, selected=False, **kwargs):
        super().__init__(**kwargs)

        if color is not None:
            self.color = color

        if index is not None:
            self.index = index

        self.selected = selected

        for segment in segments:
            self.add_segment(segment)

    def clone(self):
        segments = [segment.clone() for segment in self.children]
        return BezierPath(segments, self.color, self.index, self.selected)

    def add_segment(self, segment: BezierSegment):
        self.add_widget(segment, index=len(self.children))

    def move(self, segment, key, pos):
        idx = self.children.index(segment)

        if not (idx == 0 and key == "p1") and not (idx == len(self.children) - 1 and key == "p2"):
            idx2, key2 = (idx-1, "p2") if key == "p1" else (idx+1, "p1")
            setattr(self.children[idx2], key2, pos)

    def add_widget(self, widget, index=0, canvas=None):
        super().add_widget(widget, index=index, canvas=canvas)

    def remove_widget(self, widget):
        super().remove_widget(widget)

    @staticmethod
    def from_svg_path(svg_path: SVGPath, *args, **kwargs):
        segments = []
        for command in svg_path.path_commands:
            if isinstance(command, SVGCommandBezier):
                segment = BezierSegment.bezier(flip_vertical(command.p1.tolist()), flip_vertical(command.q1.tolist()),
                                               flip_vertical(command.q2.tolist()), flip_vertical(command.p2.tolist()))
                segments.append(segment)
            elif isinstance(command, SVGCommandLine):
                segment = BezierSegment.line(flip_vertical(command.start_pos.tolist()),
                                             flip_vertical(command.end_pos.tolist()))
                segments.append(segment)

        path = BezierPath(segments, *args, **kwargs)
        return path

    def to_svg_path(self):
        path_commands = []
        for segment in self.children:
            if segment.is_curved:
                command = SVGCommandBezier(Point(*flip_vertical(segment.p1)), Point(*flip_vertical(segment.q1)),
                                           Point(*flip_vertical(segment.q2)), Point(*flip_vertical(segment.p2)))
            else:
                command = SVGCommandLine(Point(*flip_vertical(segment.p1)), Point(*flip_vertical(segment.p2)))
            path_commands.append(command)
        svg_path = SVGPath(path_commands)
        return svg_path


class Sketch(Widget):
    color = ListProperty([1, 1, 1])
    points = ListProperty([])

    def __init__(self, points, color=None, **kwargs):
        super().__init__(**kwargs)

        if color is not None:
            self.color = color

        self.points = points

    def on_touch_move(self, touch):
        if touch.grab_current is not self:
            return super().on_touch_move(touch)

        self.points.extend(touch.pos)

    def on_touch_up(self, touch):
        if touch.grab_current is not self:
            return super().on_touch_up(touch)

        touch.ungrab(self)

        self.parent.on_sketch_done(self)

    def to_svg_path(self):
        points = [Point(x, 255 - y) for x, y in zip(self.points[::2], self.points[1::2])]
        commands = [SVGCommandMove(points[0])] + [SVGCommandLine(p1, p2) for p1, p2 in zip(points[:-1], points[1:])]
        svg_path = SVGPath.from_commands(commands).path
        return svg_path


class EditorView(Scatter):
    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos) and touch.is_mouse_scrolling:
            if touch.button == 'scrolldown':
                if self.scale < 10:
                    self.scale = self.scale * 1.1
            elif touch.button == 'scrollup':
                if self.scale > 1:
                    self.scale = self.scale * 0.8
            return True

        return super().on_touch_down(touch)


class DrawViewbox(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        Window.bind(mouse_pos=self.on_mouse_pos)

    @property
    def nb_paths(self):
        return len(self.children)

    def _get_color(self, idx):
        color = color_dict[colors[idx % len(colors)]]
        return color

    def on_mouse_pos(self, _, abs_pos):
        pos = (Vector(abs_pos) - Vector(self.parent.pos)) / self.parent.scale

        if state.header.selected_tool == ToolMode.PEN and state.draw_mode == DrawMode.DRAW:
            segment = state.current_path.children[-1]
            segment.p2 = segment.q2 = pos

    def on_sketch_done(self, sketch: Sketch):
        # Digitalize points to Bézier path
        svg_path = preprocess_svg_path(sketch.to_svg_path(), force_smooth=True)

        path_idx = state.sidebar.nb_paths
        path = BezierPath.from_svg_path(svg_path, color=sketch.color, index=path_idx, selected=True)
        self.remove_widget(sketch)

        self.add_new_path(path, svg_path)

    def on_path_done(self, path: BezierPath):
        svg_path = preprocess_svg_path(path.to_svg_path())

        path_idx = state.sidebar.nb_paths
        new_path = BezierPath.from_svg_path(svg_path, color=path.color, index=path_idx, selected=True)
        self.remove_widget(path)

        self.add_new_path(new_path, svg_path)

    def paste(self, path: BezierPath):
        path = path.clone()

        path_idx = state.sidebar.nb_paths
        path.color = self._get_color(path_idx)
        path.selected = True

        svg_path = path.to_svg_path()

        self.add_new_path(path, svg_path)

    def unselect_all(self):
        for path in self.children:
            path.selected = False

    def get_path(self, path_idx):
        index = self.nb_paths - 1 - path_idx
        return self.children[index]

    def add_new_path(self, path: BezierSegment, svg_path: SVGPath):
        self.add_path(path, svg_path, force_rerender_miniature=True)

        state.modified = True
        state.timeline.make_keyframe(True)
        state.sidebar.select(path.index)

    def add_path(self, path: BezierPath, svg_path: SVGPath, force_rerender_miniature=False):
        path_idx = state.sidebar.nb_paths
        self.add_widget(path)

        miniature_path = os.path.join(state.project.cache_dir, f"{state.timeline.selected_frame}_{path_idx}.png")
        if not os.path.exists(miniature_path) or force_rerender_miniature:
            svg_path = normalized_path(svg_path)
            svg_path.draw(viewbox=svg_path.bbox().make_square(min_size=12),
                          file_path=os.path.join(state.project.cache_dir, f"{state.timeline.selected_frame}_{path_idx}.png"),
                          do_display=False)

        if not state.header.is_playing:
            state.sidebar._add_path()

    def on_touch_down(self, touch):
        if state.header.selected_tool == ToolMode.PLAY:
            return False

        if state.header.selected_tool == ToolMode.PEN and self.collide_point(*touch.pos):
            state.draw_mode = DrawMode.DRAW

            if state.current_path is None:
                path = BezierPath([], color=self._get_color(len(self.children)), selected=True)
                self.add_widget(path)
                state.current_path = path

            l = BezierSegment.line(touch.pos, touch.pos)

            touch.ud["selected"] = "q1"
            touch.grab(l)

            state.current_path.add_segment(l)

            state.modified = True

            return True

        if state.header.selected_tool == ToolMode.PENCIL and self.collide_point(*touch.pos):
            l = Sketch([*touch.pos], color=self._get_color(len(self.children)))
            self.add_widget(l)
            touch.grab(l)

            state.modified = True

            return True

        if super().on_touch_down(touch):
            return True

    def clear(self):
        state.draw_viewbox.clear_widgets()
        state.sidebar.erase()

    def add_widget(self, widget, index=0, canvas=None):
        super().add_widget(widget, index=index, canvas=canvas)

    def remove_widget(self, widget):
        super().remove_widget(widget)

    def to_svg(self):
        svg_path_groups = []
        for path in reversed(self.children):
            svg_path_groups.append(path.to_svg_path().to_group())

        svg = SVG(svg_path_groups, viewbox=Bbox(256))
        return svg

    def load_svg(self, svg: SVG, frame_idx):
        kivy_bezierpaths = []
        for idx, svg_path in enumerate(svg.paths):
            path = BezierPath.from_svg_path(svg_path, color=self._get_color(idx), index=idx, selected=False)
            kivy_bezierpaths.append(path)
            self.add_path(path, svg_path, force_rerender_miniature=True)

        state.project.frames[frame_idx].svg = svg
        state.project.frames[frame_idx].kivy_bezierpaths = kivy_bezierpaths

    def load_cached(self, svg: SVG, kivy_bezierpaths: List[BezierPath]):
        for path, svg_path in zip(kivy_bezierpaths, svg.paths):
            self.add_path(path, svg_path)

    def load_frame(self, frame_idx):
        svg = state.project.frames[frame_idx].svg
        kivy_bezierpaths = state.project.frames[frame_idx].kivy_bezierpaths

        if kivy_bezierpaths is None:
            self.load_svg(svg, frame_idx)
        else:
            self.load_cached(svg, kivy_bezierpaths)

        self.unselect_all()

    def save_frame(self):
        svg = self.to_svg()
        state.project.frames[state.current_frame].svg = svg
        state.project.frames[state.current_frame].kivy_bezierpaths = [child for child in reversed(self.children) if isinstance(child, BezierPath)]


class HeaderButton(Button):
    pass


class UpButton(Button):
    def on_press(self):
        self.parent.move_up()


class DownButton(Button):
    def on_press(self):
        self.parent.move_down()


class ReverseButton(Button):
    def on_press(self):
        self.parent.reverse()


class FrameView(Button):
    index = NumericProperty(0)
    keyframe = BooleanProperty(False)

    def __init__(self, index, keyframe=False, **kwargs):
        super().__init__(**kwargs)

        self.index = index
        self.keyframe = keyframe

    def on_press(self):
        state.timeline.selected_frame = self.index


class TimeLine(ScrollView):
    selected_frame = NumericProperty(-1)

    @property
    def timeline(self):
        return self.ids.timeline

    @property
    def nb_frames(self):
        return len(self.timeline.children)

    def on_selected_frame(self, *args):
        self._update_frame(self.selected_frame)

    def _update_frame(self, new_frame_idx):
        if state.current_frame >= 0 and state.modified:
            state.draw_viewbox.save_frame()

        state.current_frame = new_frame_idx
        state.draw_viewbox.clear()
        state.modified = False

        state.draw_viewbox.load_frame(new_frame_idx)

    def _add_frame(self, keyframe=False):
        idx = self.nb_frames
        new_frame = FrameView(idx, keyframe=keyframe)

        self.timeline.add_widget(new_frame)
        return idx

    def get_frame(self, frame_idx):
        index = self.nb_frames - 1 - frame_idx
        return self.timeline.children[index]

    def make_keyframe(self, is_keyframe=None):
        if is_keyframe is None:
            is_keyframe = not self.get_frame(state.timeline.selected_frame).keyframe

        self.get_frame(state.timeline.selected_frame).keyframe = is_keyframe
        state.project.frames[state.timeline.selected_frame].keyframe = is_keyframe


class TitleWidget(AlignedTextInput):
    pass


class Padding(Label):
    pass


class FileChoosePopup(Popup):
    load = ObjectProperty()
    path = StringProperty(".")


class DeepSVGWidget(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        state.main_widget = self
        state.header = self.ids.header
        state.sidebar = self.ids.sidebar_scroll
        state.draw_viewbox = self.ids.editor.ids.draw_viewbox
        state.timeline = self.ids.timeline_scroll

        self._load_project()

    def _load_project(self):
        for frame in state.project.frames:
            state.timeline._add_frame(keyframe=frame.keyframe)

        state.timeline.selected_frame = 0


class DeepSVGApp(App):
    def build(self):
        self.title = 'DeepSVG Editor'

        Window.bind(on_request_close=self.on_request_close)
        Window.bind(on_keyboard=self.on_keyboard)

        return DeepSVGWidget()

    def save(self):
        state.draw_viewbox.save_frame()

        state.save_state()
        state.project.save_project()

    def on_request_close(self, *args, **kwargs):
        self.save()

        self.stop()

    def on_keyboard(self, window, key, scancode, codepoint, modifier):
        CTRL_PRESSED = (modifier == ['ctrl'] or modifier == ['meta'])

        if codepoint == "h" and not CTRL_PRESSED:
            # Hand tool
            state.header.selected_tool = ToolMode.MOVE

        elif codepoint == "p" and not CTRL_PRESSED:
            # Pen tool
            state.header.selected_tool = ToolMode.PEN

        elif CTRL_PRESSED and codepoint == "p":
            # Pencil tool
            state.header.selected_tool = ToolMode.PENCIL

        elif codepoint == "k" and not CTRL_PRESSED:
            # Make keypoint
            state.timeline.make_keyframe()

        elif CTRL_PRESSED and codepoint == 'q':
            # Quit
            self.on_request_close()

        elif CTRL_PRESSED and codepoint == 'i':
            # Import
            self.file_chooser = FileChoosePopup(load=self.on_file_chosen)
            self.file_chooser.open()

        elif CTRL_PRESSED and codepoint == "e":
            # Export
            state.project.export_to_gif(loop_mode=state.loop_mode)

        elif CTRL_PRESSED and codepoint == 'c':
            # Copy
            if state.sidebar.selected_path_idx >= 0:
                state.clipboard = state.draw_viewbox.get_path(state.sidebar.selected_path_idx).clone()

        elif CTRL_PRESSED and codepoint == 'v':
            # Paste
            if isinstance(state.clipboard, BezierPath):
                state.draw_viewbox.paste(state.clipboard)

        elif CTRL_PRESSED and codepoint == 's':
            # Save
            self.save()

        elif key == Keys.SPACEBAR:
            # Play/Pause
            state.header.selected_tool = ToolMode.PLAY

            if state.header.is_playing:
                state.header.pause_animation()
            else:
                state.header.play_animation()

        elif key == Keys.LEFT:
            # Previous frame
            if state.current_frame > 0:
                state.timeline.selected_frame = state.current_frame - 1

        elif key == Keys.RIGHT:
            # Next frame
            if state.current_frame < state.timeline.nb_frames - 1:
                state.timeline.selected_frame = state.current_frame + 1

    def on_file_chosen(self, selection):
        file_path = str(selection[0])
        self.file_chooser.dismiss()

        if file_path:
            if not file_path.endswith(".svg"):
                return

            svg = SVG.load_svg(file_path)
            svg = SVGTensorDataset.simplify(svg)
            svg = SVGTensorDataset.preprocess(svg, mean=True)

            state.draw_viewbox.load_svg(svg, frame_idx=state.timeline.selected_frame)
            state.modified = True
            state.timeline.make_keyframe(True)


if __name__ == "__main__":
    DeepSVGApp().run()
