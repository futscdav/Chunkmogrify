#
#    Author: David Futschik
#    Provided as part of the Chunkmogrify project, 2021.
#

import numpy as np

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

from qtutil import *

class KeepArWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ar = 1. # 16 / 9

    def set_widget(self, widget):
        self.setLayout(QBoxLayout(QBoxLayout.LeftToRight, self))
        self.layout().addItem(QSpacerItem(0, 0))
        self.layout().addWidget(widget)
        self.layout().addItem(QSpacerItem(0, 0))

    def set_ar(self, ar):
        self.ar = ar

    def resizeEvent(self, event):
        w, h = event.size().width(), event.size().height()
        
        if w / h > self.ar:
            self.layout().setDirection(QBoxLayout.LeftToRight)
            widget_stretch = h * self.ar
            outer_stretch = (w - widget_stretch) / 2 + 0.5
        else:
            self.layout().setDirection(QBoxLayout.TopToBottom)
            widget_stretch = w / self.ar
            outer_stretch = (h - widget_stretch) / 2 + 0.5
        
        self.layout().setStretch(0, outer_stretch)
        self.layout().setStretch(1, widget_stretch)
        self.layout().setStretch(2, outer_stretch)


class WorkspaceWidget(QWidget):
    def __init__(self, app, initial_image, forward_widgets=[], parent=None):
        super().__init__(parent)
        self.pixmap = None
        self.resize_hint = 2048
        self.aspect_ratio = 1. # 16. / 9.
        self.resize_to_aspect = True
        self.painting_enabled = True
        self.setLayout(QBoxLayout(QBoxLayout.LeftToRight, self))

        self.empty_pixmap_label = "Start by opening an image file. Go to <File/Image from File>"
        if initial_image:
            self.content = npy_loader(initial_image)
            self.has_content = True
        else:
            self.content = np.zeros((1024, 1024, 4))
            hbox = QHBoxLayout()
            hbox.addStretch()
            hbox.addWidget(QLabel(self.empty_pixmap_label))
            hbox.addStretch()
            self.layout().addLayout(hbox)
            self.placeholder_content = hbox
            self.has_content = False

        self.overlay = None
        self.overlay_enabled = True
        self.forward_widgets = forward_widgets

        self.inset_border_params = None
        self.scale_factor = 1

        policy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)

        self._update_image()

    def set_content(self, image: np.ndarray, update=True):
        assert image.dtype == np.uint8, "Workspace widget expects content to be a uint8 numpy array."
        self.content = image
        if not self.has_content:  # Did not previously have content and showed placeholder.
            self.layout().removeItem(self.placeholder_content)
            destroy_layout(self.placeholder_content)
            self.has_content = True
        if update: self._update_image()
        
    def set_overlay(self, overlay: np.ndarray, update=True):
        assert overlay.dtype == np.uint8, "Workspace widget expects overlay to be a uint8 numpy array."
        assert overlay.shape[2] == 4, "Workspace overlay must be a 4 dimensional image."
        self.overlay = overlay
        if update: self._update_image()

    def _update_image(self):
        if self.overlay is not None and self.overlay_enabled:
            alpha = self.overlay[:, :, 3:3]
            with_overlay = self.overlay * alpha + (1 - alpha) * self.content[:, :, 0:3]
            image = with_overlay
        else:
            image = self.content
        self.pixmap = QPixmap(np2qim(image))
        self.update()

    def toggle_overlay(self, val):
        self.overlay_enabled = val
        self._update_image()

    def current_image_as_numpy(self):
        return qim2np(self.pixmap.toImage()).copy()

    def get_current(self, with_overlay):
        npy = self.content[:, :, 0:3]
        if with_overlay and self.overlay is not None:
            alpha = self.overlay[:, :, 3:3]
            npy = self.overlay * alpha + (1 - alpha) * self.content[:, :, 0:3]
        return npy

    def inset_border(self, pxs, color):
        self.inset_border_params = {
            'px': pxs,
            'color': color
        }
        self.update()

    def set_scale_factor(self, factor):
        self.scale_factor = factor
        self.update()

    def paintEvent(self, event):
        pixmap = self.pixmap

        qpainter = QPainter(self)
        qpainter.drawPixmap(event.rect(), pixmap)
        if self.inset_border_params:
            rect = event.rect()
            rect.adjust(0, 0, -1, -1)
            qpainter.setPen(
                QPen(
                    QColor(*self.inset_border_params['color'], 255), 
                    self.inset_border_params['px'])
            )
            qpainter.drawRect(rect)

        for w in self.forward_widgets:
            w.draw(qpainter, event)
        
        if self.resize_to_aspect:
            self.resize_to_aspect = False
            w, h = self.resize_to()
        qpainter.end()

    def is_pos_outside_bounds(self, pos):
        x, y = pos.x(), pos.y()
        mx, my = self.width(), self.height()
        if x < 0 or y < 0: return True
        if x >= mx or y >= my: return True
        return False

    def resize_to(self):
        # figure out what the limiting dimension is
        w, h = self.width(), self.height()
        w = (int(1.0 / self.height_to_width_ratio() * self.height()))
        w = int(w * self.scale_factor)
        h = int(h * self.scale_factor)
        return (w, h)

    def height_to_width_ratio(self):
        if self.pixmap is None:
            return self.aspect_ratio
        return float(self.pixmap.height()) / self.pixmap.width()

    def sizeHint(self):
        base_width = int(self.resize_hint * self.scale_factor)
        return QSize(base_width, self.heightForWidth(base_width))

    def resizeEvent(self, event):
        if int(self.width() * self.height_to_width_ratio()) != self.height():
           self.resize_to_aspect = True
        self.resize_to_aspect = True

    def heightForWidth(self, width):
        return int(self.height_to_width_ratio() * width)

    def enterEvent(self, event):
        self.setMouseTracking(True)
        self.setFocus()
        for w in self.forward_widgets:
            if w.enabled():
                w.mouse_enter()
        self.update()

    def leaveEvent(self, event):
        self.setMouseTracking(False)
        for w in self.forward_widgets:
            if w.enabled():
                w.mouse_leave()
        self.update()

    def mouseMoveEvent(self, event):
        if self.is_pos_outside_bounds(event.localPos()):
            # call this if you dont want to continue drawing when you return into the widget
            # self.stop_painting()
            return
        sx = (event.pos().x() / self.width())
        sy = (event.pos().y() / self.height())

        for w in self.forward_widgets:
            # if w.enabled():
            # Right now, let's propagate mouse events even when disabled. The widget should decide.
            # This is a workaround so that if the forwarded widget gets enabled after being disabled,
            # it will still have a chance to know where the cursor is currently.
            w.mouse_moved(sy, sx)
        self.update()

    def wheelEvent(self, event):
        for w in self.forward_widgets:
            if w.enabled():
                w.mouse_zoom(event.angleDelta().y() * 1/8)
        self.update()

    def mousePressEvent(self, event):
        btn = MouseButton(event.button())

        sx = event.localPos().x() / self.width()
        sy = event.localPos().y() / self.height()

        for w in self.forward_widgets:
            if w.enabled():
                w.mouse_down(sy, sx, btn)
        self.update()

    def mouseReleaseEvent(self, event):
        sx = event.localPos().x() / self.width()
        sy = event.localPos().y() / self.height()
        btn = MouseButton(event.button())
        for w in self.forward_widgets:
            if w.enabled():
                w.mouse_up(sy, sx, btn)
        self.update()

    def keyPressEvent(self, event):
        # This doesn't always work, and some keys are not represented, but overall it's enough
        key = event.text()
        for w in self.forward_widgets:
            if w.enabled():
                w.key_down(key)
        self.update()
