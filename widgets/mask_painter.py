#
#    Author: David Futschik
#    Provided as part of the Chunkmogrify project, 2021.
#

import numpy as np

from qtutil import *

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

from time import perf_counter as time
from _C_canvas import canvas_to_masks


# This is inefficient, but it was the quickest solution to the problem.
# Ideally we would just paint along a path.
# Alternative: use QCursor and scaled Pixmap, but it has its own set of problems.
class EllipseBrush:
    def __init__(self, height, width, visual_color=QColor.fromRgb(25, 127, 255)):
        self.image = QImage(width, height, QImage.Format_RGBA8888)
        self.image.fill(Qt.transparent)
        self.visual_color = visual_color
        self.pen = QPen(self.visual_color)
        self.pen.setWidth(3)
        self.pen.setCapStyle(Qt.RoundCap)

        self.clear_pen = QPen(Qt.transparent)
        self.clear_pen.setWidth(6)
        self.clear_pen.setCapStyle(Qt.RoundCap)
        self.last_visual_arguments = None

    def clear_last_visual(self, into_painter=None):
        # very blunt method, could just draw a big transparent ellipse
        # self.image.fill(Qt.transparent)
        if self.last_visual_arguments is not None:
            if into_painter is None:
                painter = QPainter(self.image)
            else:
                painter = into_painter
            painter.setCompositionMode(QPainter.CompositionMode_Clear)
            painter.setRenderHints( QPainter.Antialiasing ) # AA for the brush layer
            painter.setPen(self.clear_pen)

            args = self.last_visual_arguments
            args[2] += 3; args[3] += 3
            painter.setBrush(Qt.transparent)
            painter.drawEllipse(*args)
            if into_painter is None:
                painter.end()

    def update_visual(self, mx, my, size):
        painter = QPainter(self.image)
        self.clear_last_visual(painter)
        painter.setCompositionMode(QPainter.CompositionMode_Source)

        painter.setRenderHints( QPainter.Antialiasing ) # AA for the brush layer
        painter.setPen(self.pen)
        half_radius = float(size) / 2
        self.last_visual_arguments = [mx * self.image.width() - half_radius, my * self.image.height() - half_radius, size, size]
        painter.drawEllipse(*self.last_visual_arguments)
        painter.end()

    def draw(self, qpainter, x, y, color, size):
        half_radius = float(size) / 2
        qpainter.setBrush(color)
        qpainter.setPen(color)
        qpainter.drawEllipse(x - half_radius, y - half_radius, size, size)

# Careful about choice of colors, because there is a clear bug in Qt where the alpha is NOT entirely ignored even with the correct 
# CompositionMode.
COLORS = [
    (0,   0,   0,   0),
    (128, 128, 128, 90), # 1
    (128, 255, 128, 90), # 2
    (255, 128, 128, 90), # 3
    (128, 128, 255, 90), # 4
    (150, 51 , 201, 90), # 5
    (201, 150, 51 , 90), # 6
    (60 , 119, 150, 90), # 7
    (77 , 255, 238, 90), # 8
    (255, 244, 20 , 90), # 9
    (6  , 68 , 17 , 90), # 10
    (196, 11 , 0  , 90), # 11
    (99 , 48 , 11 , 90), # 12
]

# Don't store the buf!
mask_buf = None
def canvas_to_numpy_mask(qt_image, colors):
    global mask_buf
    asnp = qim2np(qt_image, swapaxes=False)
    
    # This is the numpy way:
    # color_cmps = np_colors[None, None, :, :]
    # # (1024, 1024, 4, 1) == (1, 1, 4, 8)
    # cmps = np.equal(asnp[:, :, :, None], color_cmps, out=cmp_buf)
    # logical_masks = cmps.all(axis=2)[:, :, 1:]
    # masks = logical_masks

    # C++ optimized way:

    if mask_buf is None:
        masks = canvas_to_masks(asnp, colors)
        mask_buf = masks
    else:
        masks = canvas_to_masks(asnp, colors, output_buffer=mask_buf)
    return masks[:, :, 1:]

# Don't store the buffers anywhere!
canvas_buf = None
def numpy_mask_to_numpy_canvas(numpy_mask):
    global canvas_buf
    if canvas_buf is None: canvas_buf = np.zeros((numpy_mask.shape[0], numpy_mask.shape[1], 4), dtype=np.uint8)
    else: pass # canvas_buf.fill(0.)
    # Use COLORS once there are multiple masks

    # The mask needs to have the index of the resulting color & be flattened
    numpy_mask = numpy_mask[:, :, :] * (np.arange(1, numpy_mask.shape[2] + 1))[None, None, :]
    numpy_mask = numpy_mask.sum(axis=2)

    np.choose(
        numpy_mask.astype(np.uint8)[:, :, None],
        COLORS,
        mode='clip',
        out=canvas_buf
    )
    return canvas_buf

class PainterWidget:
    def __init__(self, height, width, max_segments, reporting_fn):
        self.show_brush = False
        self.brush_size = 15
        self.brush = EllipseBrush(height, width)
        self.canvas = QImage(width, height, QImage.Format_RGBA8888)
        self.canvas.fill(Qt.transparent)
        self.update_callbacks = []
        self.color_index = 1
        self.last_known_paint_pos = None
        self.reporting_fn = reporting_fn
        self.max_segments = max_segments
        self.np_colors = np.array(COLORS).astype(np.uint8)[:self.max_segments+1]
        if self.max_segments > len(COLORS) - 2:
            print(f'Requested {self.max_segments} masks, but only {len(COLORS) - 2} colors are defined. Setting max segments to {len(COLORS) - 2}')
            self.max_segments = len(COLORS) - 2
        
        self._hide = False
        self._enabled = True
        self._active_buttons = {}

    def get_volatile_masks(self):
        asnp = qim2np(self.canvas, swapaxes=False)
        masks = canvas_to_masks(asnp, self.np_colors)
        return asnp, masks

    def draw(self, qpainter, qevent):
        if self._hide: return
        # this actually only draws the overlay with brush
        qpainter.setOpacity(1)
        # This overlays current context with the actual painting content
        qpainter.drawImage(qevent.rect(), self.canvas)
        # This overlays the current context with the brush visual
        qpainter.drawImage(qevent.rect(), self.brush.image)
        
    def change_brush_size(self, amount):
        self.brush_size += amount
        self.brush_size = 1 if self.brush_size < 1 else self.brush_size
        self.update_visuals()

    def change_active_color(self, dst_idx):
        self.color_index = dst_idx
        # Active color is between 1 and N
        if self.color_index < 1: self.color_index = 1
        if self.color_index >= min(self.max_segments + 1, len(COLORS)): self.color_index = min(self.max_segments + 1, len(COLORS)) - 1
        self.reporting_fn(f"Current color index: {self.color_index} - {{{COLORS[self.color_index]}}}")

    def active_color(self):
        return QColor.fromRgb(*COLORS[self.color_index])

    def rewrite_with_numpy_mask(self, target, writeback=False, actor=None):
        # Changes was initiated by gui and this is just a notification of change
        if actor == 'gui': return

        old_canvas = qim2np(self.canvas)
        old_canvas[:] = numpy_mask_to_numpy_canvas(target)[:]
        self.canvas = np2qim(old_canvas, do_copy=False)
        if writeback:
            for c in self.update_callbacks:
                c(target, actor='gui')

    # dx, dy in [0,1]
    # called on each mouse event, even when button is not down
    def paint_at(self, button, dy, dx):
        self.last_known_paint_pos = (dx, dy)
        self.update_visuals()

        csx, csy = dx * self.canvas.width(), dy * self.canvas.height()
        if button == MouseButton.LEFT:
            self._paint_at_pos(csx, csy, self.active_color())
        if button == MouseButton.RIGHT:
            self._paint_at_pos(csx, csy, QColor.fromRgb(*COLORS[0]))
        if button == MouseButton.MIDDLE:
            self._pick_color(csx, csy)

    def update_mouse_position(self, dy, dx):
        self.last_known_paint_pos = (dx, dy)
        if self.enabled():
            self.update_visuals()

    def update_visuals(self):
        if self.last_known_paint_pos is None: return
        self.brush.update_visual(self.last_known_paint_pos[0], self.last_known_paint_pos[1], self.brush_size)

    def clear_visuals(self):
        self.brush.clear_last_visual()

    def _paint_at_pos(self, x, y, color):
        qpainter = QPainter(self.canvas)
        qpainter.setCompositionMode(QPainter.CompositionMode_Source) # composition mode: Source (to prevent alpha multiplication)
        self.brush.draw(qpainter, x, y, color, self.brush_size)
        qpainter.end()
        # duplicate changes into the writeback buffer
        # this makes a copy after each paint call !
        # Notify all interested objects.
        numpy_mask = canvas_to_numpy_mask(self.canvas, self.np_colors)
        for c in self.update_callbacks:
            c(numpy_mask, actor='gui')

    def _pick_color(self, x, y):
        c = self.canvas.pixelColor(x, y)
        idx = None
        for i, ref_c in enumerate(COLORS):
            if c.getRgb() == ref_c:
                idx = i
        if idx is None: raise RuntimeError(f"Color index not found (color has wrong representation of {c.getRgb()})")
        self.change_active_color(idx)

    # Enabled will be queried for event forwarding.
    def enabled(self):
        return self._enabled

    def toggle(self, to):
        self._enabled = to
        if (not to):
            # If enabled = False then draw will no longer be called and visual might stick around.
            self.clear_visuals()
        else:
            # if enabled = True then visual is not drawn and mouse position is not updated.
            self.update_visuals()

    def hide(self):
        self._hide = True

    def show(self):
        self._hide = False

    def mouse_enter(self):
        pass

    def mouse_leave(self):
        self.clear_visuals()

    def mouse_zoom(self, angle):
        modifiers = QApplication.keyboardModifiers()
        if modifiers == Qt.ShiftModifier:
            self.change_active_color(self.color_index + 1 if angle > 0 else self.color_index - 1)
        else:
            self.change_brush_size(angle)

    def mouse_down(self, y, x, btn):
        self._active_buttons[btn] = True
        self.paint_at(btn, y, x)

    def mouse_up(self, y, x, btn):
        self._active_buttons[btn] = None

    def mouse_moved(self, y, x):
        if self.enabled():
            for (btn, active) in self._active_buttons.items():
                if active: self.paint_at(btn, y, x)
        self.update_mouse_position(y, x)

    def key_down(self, key):
        if key == '+':
            self.change_active_color(self.color_index + 1)
        if key == '-':
            self.change_active_color(self.color_index - 1)
