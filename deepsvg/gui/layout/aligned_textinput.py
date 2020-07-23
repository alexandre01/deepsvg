from kivy.uix.textinput import TextInput
from kivy.properties import StringProperty

DEFAULT_PADDING = 6


class AlignedTextInput(TextInput):
    halign = StringProperty('left')
    valign = StringProperty('top')

    def __init__(self, **kwargs):
        self.halign = kwargs.get("halign", "left")
        self.valign = kwargs.get("valign", "top")

        self.bind(on_text=self.on_text)

        super().__init__(**kwargs)

    def on_text(self, instance, value):
        self.redraw()

    def on_size(self, instance, value):
        self.redraw()

    def redraw(self):
        """
        Note: This methods depends on internal variables of its TextInput
        base class (_lines_rects and _refresh_text())
        """

        self._refresh_text(self.text)

        max_size = max(self._lines_rects, key=lambda r: r.size[0]).size
        num_lines = len(self._lines_rects)

        px = [DEFAULT_PADDING, DEFAULT_PADDING]
        py = [DEFAULT_PADDING, DEFAULT_PADDING]

        if self.halign == 'center':
            d = (self.width - max_size[0]) / 2.0 - DEFAULT_PADDING
            px = [d, d]
        elif self.halign == 'right':
            px[0] = self.width - max_size[0] - DEFAULT_PADDING

        if self.valign == 'middle':
            d = (self.height - max_size[1] * num_lines) / 2.0 - DEFAULT_PADDING
            py = [d + 5, d - 5]
        elif self.valign == 'bottom':
            py[0] = self.height - max_size[1] * num_lines - DEFAULT_PADDING

        self.padding_x = px
        self.padding_y = py
