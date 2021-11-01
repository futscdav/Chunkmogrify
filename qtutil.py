#
#    Author: David Futschik
#    Provided as part of the Chunkmogrify project, 2021.
#

import os
import re
import numpy as np
import PIL.Image as Image
from enum import Enum
from threading import Thread
from threading import Timer
from time import time, perf_counter

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

class dotdict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
    def has_set(self, attr):
        return attr in self and self[attr] is not None

class MouseButton(Enum):
    LEFT    = 1
    RIGHT   = 2
    MIDDLE  = 4
    SIDE_1  = 8
    SIDE_2  = 16

def QHLine():
    w = QFrame()
    w.setFrameShape(QFrame.HLine)
    w.setFrameShadow(QFrame.Sunken)
    return w

def QVLine():
    w = QFrame()
    w.setFrameShape(QFrame.VLine)
    w.setFrameShadow(QFrame.Raised)
    return w

def qpixmap_loader(path: str):
    if not os.path.exists(path):
        print(f"Warning: {path} does not exist, but queued for load as pixmap")
    else:
        print(f"Loading {path}")
    return QPixmap(path)

def npy_loader(path: str):
    if not os.path.exists(path):
        raise RuntimeError(f"{path} does not exist")
    return np.array(Image.open(path))

def make_dirs_if_not_exists(path):
    if not os.path.exists(path) or not os.path.isdir(path):
        os.makedirs(path)

def export_image(prefix, npy_image):
    if not os.path.exists(prefix):
        os.mkdir(prefix)
    ls = os.listdir(prefix)
    prev_imgs = [re.match(r'^\d+', x) for x in ls]
    prev_imgs = [int(x.group()) for x in prev_imgs if x is not None]
    cur_id = max(prev_imgs, default=-1) + 1
    fname = os.path.join(prefix, f'{cur_id:05d}.png')
    Image.fromarray(npy_image).save(fname)
    return fname

def image_save(npy_image, path):
    Image.fromarray(npy_image).save(path)

def qim2np(qimage, swapaxes=True):
    # assumes BGRA if swapaxes is True
    # assert(qimage.format() == QImage.Format_8888)
    bytes_per_pxl = 4
    if qimage.format() == QImage.Format_Grayscale8:
        bytes_per_pxl = 1
    qimage.__array_interface__ = {
        'shape': (qimage.height(), qimage.width(), bytes_per_pxl),
        'typestr': "|u1",
        'data': (int(qimage.bits()), False),
        'version': 3
    }
    npim = np.asarray(qimage)
    # npim is now BGRA, cast it into RGBA
    if bytes_per_pxl == 4 and swapaxes:
        npim = npim[...,(2,1,0,3)]
    return npim

def np2qim(nparr, fmt_select='auto', do_copy=True):
    h, w = nparr.shape[0:2]
    # the copy is required unless reference is desired!
    if do_copy:
        nparr = nparr.astype('uint8').copy()
    else:
        nparr = np.ascontiguousarray(nparr.astype('uint8'))
    if nparr.ndim == 2:
        # or Alpha8 when it has no 3rd dimension (masks)
        fmt = QImage.Format_Grayscale8 #Alpha8
    elif fmt_select != 'auto':
        fmt = fmt_select
    else:
        fmt = {3: QImage.Format_RGB888, 4: QImage.Format_RGBA8888}[nparr.shape[2]]
    qim = QImage(nparr, w, h, nparr.strides[0], fmt) # _Premultiplied 
    return qim

def destroy_layout(layout):
    if layout is not None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
            else:
                destroy_layout(item.layout())

class NotifyWait(QObject):
    acquire = pyqtSignal(str)
    release = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent=parent)
        self.msg = ""
        d = QProgressDialog(self.msg, "Cancel", 0, 0, parent=get_default_parent())
        d.setCancelButton(None)
        d.setWindowTitle("Working")
        d.setWindowFlags(d.windowFlags() & ~Qt.WindowCloseButtonHint)
        self.d = d
        self.d.cancel()

    def _show(self):
        self.d.setLabelText(self.msg)
        self.d.exec_()

    def _hide(self):
        self.d.done(0)


class RaiseError(QObject):
    raiseme = pyqtSignal(str, str, bool) # is_fatal

    def __init__(self, parent=None) -> None:
        super().__init__(parent=parent)

    def _show(self, where, what, is_fatal):
        msg = QMessageBox(self.parent())
        msg.setText(where)
        msg.setInformativeText(what)
        msg.setWindowTitle("Error")
        msg.setIcon(QMessageBox.Critical)
        msgresizer = QSpacerItem(500, 0, QSizePolicy.Minimum, QSizePolicy.Expanding)
        msg.layout().addItem(msgresizer, msg.layout().rowCount(), 0, 1, msg.layout().columnCount())
        msg.exec_()
        if is_fatal:
            exit(1)

_global_parent = None
_notify_wait = None
_global_error = None
def set_default_parent(parent):
    global _global_parent
    global _notify_wait
    global _global_error
    _global_parent = parent
    _notify_wait = NotifyWait(_global_parent)
    _global_error = RaiseError(_global_parent)

    def s(msg):
        _notify_wait.msg = msg
        _notify_wait._show()
    def e():
        _notify_wait._hide()
    _notify_wait.acquire.connect(s)
    _notify_wait.release.connect(e)

    def e(where, what, is_fatal):
        _global_error._show(where, what, is_fatal)
    _global_error.raiseme.connect(e)

def get_default_parent():
    global _global_parent
    return _global_parent
def get_notify_wait():
    global _notify_wait
    return _notify_wait
def get_global_error():
    global _global_error
    return _global_error

def notify_user_wait(message):
    d = QProgressDialog(message, "Cancel", 0, 0, parent=get_default_parent())
    d.setCancelButton(None)
    d.setWindowTitle("Working")
    d.setWindowFlags(d.windowFlags() & ~Qt.WindowCloseButtonHint)
    return d


def notify_user_error(where, what, parent=None):
    msg = QMessageBox(parent)
    msg.setText(where)
    msg.setInformativeText(what)
    msg.setWindowTitle("Error")
    msg.setIcon(QMessageBox.Critical)
    msgresizer = QSpacerItem(500, 0, QSizePolicy.Minimum, QSizePolicy.Expanding)
    msg.layout().addItem(msgresizer, msg.layout().rowCount(), 0, 1, msg.layout().columnCount())
    msg.exec_()

# Really only works when called from main thread.
def execute_with_wait(message, fn, *args, **kwargs):
    d = notify_user_wait(message)

    def fn_impl():
        fn(*args, **kwargs)
        d.done(0)
    t = Thread(target=fn_impl)
    t.start()
    d.exec_()

class NowOrDelayTimer:
    def __init__(self, interval):
        self.interval = interval
        self.last_ok_time = 0
        self.timer = None

    def update(self, do_run):
        if self.last_ok_time < time() - self.interval:
            self.last_ok_time = time()
            do_run()
            # cancel pending update
            if self.timer: self.timer.cancel()
        else:
            # cancel last pending update
            if self.timer: self.timer.cancel()
            def closure():
                self.update(do_run)
            self.timer = Timer(self.last_ok_time + self.interval - time(), closure)
            self.timer.start()

class MeasureTime:
    def __init__(self, name, disable=False):
        self.name = name
        self.start = None
        self.disable = disable
    
    def __enter__(self):
        if not self.disable:
            self.start = perf_counter()

    def __exit__(self, type, value, traceback):
        if not self.disable:
            end = perf_counter()
            print(f'{self.name}: {end-self.start:.3f}')