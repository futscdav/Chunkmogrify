#
#    Author: David Futschik
#    Provided as part of the Chunkmogrify project, 2021.
#

from typing import Any, Callable, List

import os
import re
import PIL.Image
import numpy as np

# MASK
# 1 = Optimize in this region
# 0 = Do not optimize in this region
class MaskState:
    def __init__(self, height, width, max_segments, output_fns: List[Callable[[np.ndarray], Any]]):
        self.output = output_fns
        self.h, self.w, self.c = height, width, max_segments
        self.np_buffer = np.zeros((height, width, max_segments), dtype=np.float)
        self.update_callbacks = []
        self.mask_version = 0

    def set_to(self, new_buffer: np.ndarray, **cb_kwargs):
        assert self.np_buffer.min() >= 0. and self.np_buffer.max() <= 1., \
            f"Mask has range [{self.np_buffer.min()}, {self.np_buffer.max()}]"
        buffer_c = new_buffer.shape[2]
        self.np_buffer[..., :min(self.c, buffer_c)] = new_buffer[..., :min(self.c, buffer_c)]
        self.mask_version += 1
        for c in self.update_callbacks:
            c(self.np_buffer, **cb_kwargs)

    def load_masks(self, source_dir):
        if not os.path.exists(source_dir):
            print(f"Could not load masks from {source_dir}")
            return
        ls = os.listdir(source_dir)
        found_idx = [re.match(r'^\d+', x) for x in ls]
        found_idx = [int(f.group()) for f in found_idx if f is not None]
        # 0 used to be the entire thing, in that case found_idx.remove(0)
        for idx in found_idx:
            # load the image
            m = np.array(PIL.Image.open(os.path.join(source_dir, f'{idx:02d}.png')).convert('L'))
            if idx >= self.np_buffer.shape[2]:
                print(f"Skipping mask {idx}")
                continue
            self.np_buffer[..., idx] = m / 255.
        self.mask_version += 1
        for c in self.update_callbacks:
            c(self.np_buffer)

    def save_masks(self, target_dir, painter, max_segments):
        # Get it from painter to include the RGB as it is in the app.
        rgb_masks, npy_masks = painter.get_volatile_masks()
        PIL.Image.fromarray(rgb_masks).save(os.path.join(target_dir, "rgb.png"))
        # max_segments + 1 because "empty" is the 0th mask.
        PIL.Image.fromarray((npy_masks[:, :, 0] * 255).astype(np.uint8)).save(os.path.join(target_dir, f"all.png"))
        for i in range(1, min(max_segments + 1, npy_masks.shape[2])):
            PIL.Image.fromarray((npy_masks[:, :, i] * 255).astype(np.uint8)).save(os.path.join(target_dir, f"{i - 1:02d}.png"))

    def get_mask_version(self):
        return self.mask_version

    def numpy_buffer(self):
        assert self.np_buffer.min() >= 0. and self.np_buffer.max() <= 1., \
            f"Mask has range [{self.np_buffer.min()}, {self.np_buffer.max()}]"
        return self.np_buffer
