#
#    Author: David Futschik
#    Provided as part of the Chunkmogrify project, 2021.
#

# This is the glue code for going from ui to result
import os
import cv2
import torch
import numpy as np

from PIL import Image

try:
    from align import align_face_npy_with_params
    do_align = True
except Exception as e:
    print(f"Warning: {e}")
    do_align = False
from typing import Any, Callable

from config import dotdict, global_config
from qtutil import get_notify_wait, NowOrDelayTimer, MeasureTime, get_global_error
from _C_heatmap import heatmap
from stylegan_tune import PivotalTuning
from stylegan_project import StyleganProvider, StyleganProjector, \
                             StyleganSSpaceProjector, StyleganWPlusProjector, \
                             styleclip_edit
from mask_refinement import mask_refine
from scripts.idempotent_blend import poisson_edit
from skimage.morphology import dilation, disk

projection_modes = [
    "w_projection",
    "s_projection",
    "wplus_projection"
]
edit_modes = [
    "none",
    "w_edit",
    "s_edit"
]
output_modes = [
    "target_only",
    "projection_only",
    "masked",
    "blended",
    # "idempotent_blend" # Poisson that only changes the mask contents, very slow for now.
]

def load_and_align_image(path, align):
    pil = Image.open(path).convert('RGB')
    global do_align
    if do_align and align:
        aligned, params = align_face_npy_with_params(np.array(pil))
    else:
        print("Skipping alignment and resizing to native resolution")
        aligned = np.array(pil)
        params = None
    aligned = np.array(Image.fromarray(aligned).resize(global_config().generator_native_resolution))
    return aligned, params

gan = None
def init_gan(force=False):
    global gan
    if gan is None or force:
        if global_config().ui_debug_run:
            print('WARNING: GAN loading disabled for ui debug.')
        else:
            def load_impl():
                global gan
                gan = StyleganProvider(global_config().generator_path, global_config().device, global_config().generator_native_resolution)
            get_notify_wait().acquire.emit("Loading resources..")
            exception = None
            try:
                load_impl()
                # Run dummy input to compile exts.
                gan.g.synthesis(torch.randn([1, gan.g.num_ws, gan.g.w_dim]).to(global_config().device), None)
            except Exception as e:
                exception = e
            get_notify_wait().release.emit()
            if exception:
                print(str(exception))
                get_global_error().raiseme.emit("Initializing", f"Error: {str(exception)}", True)

class SynthesisState:
    def __init__(self, output: Callable[[np.ndarray], Any], mask_push: Callable[[np.ndarray], Any], mask_pull: Callable[[], np.ndarray],
                    mask_version: Callable[[], int], projector_added: Callable, projector_removed: Callable):
        # Load model as when needed, init to None
        self.sg = None

        self.projector = None
        self.projection_mode = global_config().projection_mode
        assert self.projection_mode in projection_modes, f"Unknown projection mode {self.projection_mode}"
        print(f"Projection mode set to {self.projection_mode}")
        self.target_image, self.alignment_params = None, None
        self.mask_version = mask_version
        self.mask_pull = mask_pull
        self.mask_push = mask_push
        self.output_mode = "masked"
        self.output = output
        self.unaltered_output = None
        self.difference_output = None

        self.projector_added = projector_added
        self.projector_removed = projector_removed

        # Synthesize only after this time has passed since last
        self.minimum_projection_window = global_config().minimum_projection_update_window 
        self.minimum_difference_window = 0.5

        self.difference_timer = NowOrDelayTimer(self.minimum_difference_window)
        self.projection_timer = NowOrDelayTimer(self.minimum_projection_window)

        self.last_used_mask_version = None
        self.last_output = None
        self.last_params = []
        self.pti_optimizer = None

    def reset_projection(self):
        if self.pti_optimizer is not None:
            self.pti_optimizer = None
            # Model changed, need to reload original gan
            init_gan(force=True)
            global gan
            self.sg = gan
        self.projector = None
        self.last_output = None
        self._synthesize_with_last_params()
        self.projector_removed()
        self.show_difference(self.target_image, self.target_image)

    def set_unaltered_output(self, unaltered_output):
        self.unaltered_output = unaltered_output
        if self.unaltered_output and self.target_image:
            self.unaltered_output(self.target_image)

    def _update_unaltered_output(self):
        if self.unaltered_output is not None and self.target_image is not None:
            self.unaltered_output(self.target_image)

    def set_difference_output(self, difference_output):
        self.difference_output = difference_output

    def load_image_and_reset(self, path, align=True):
        image, params = load_and_align_image(path, align)
        self.alignment_params = params
        self.set_image_and_reset(image)

    def set_image_and_reset(self, np_image):
        self.target_image = np_image
        # Will draw the new image.
        self.reset_projection()

    def set_output_mode(self, mode):
        self.output_mode = mode
        if self.projector is not None:
            self._synthesize_with_last_params()

    def mask_refine(self, dA, dB, iters):
        # only works for segment 1 now (last_output[0])
        refine_input = self.projector.provider.numpy_uint8_to_torch(self.last_output)
        new_mask_torch = mask_refine(self.projector.current_mask, self.projector.target_image, refine_input, dA, dB, iters)
        new_mask_np = self.sg.torch_mask_to_numpy(new_mask_torch.permute(1, 2, 3, 0)[0, :, :, :]) # Assume all channels of the mask are the same
        self.mask_push(new_mask_np, actor="refine")

    def on_mask_changed(self, _new_mask, actor=None):
        # Synthesize is correct but makes the mask drawing feel choppy, so check if it's needed.
        if self.output_mode == "target_only" or self.output_mode == "projection_only":
            return
        if self.last_output is not None:
            output = self._apply_output_mode(self.last_output)
            self.output(output)
            # Computing difference also makes it pretty choppy, so only do it periodically
            self.show_difference(output, self.target_image, check_frequency=True)

    def show_difference(self, x, y, check_frequency=True):
        def update():
            difference = np.abs(x.astype(np.float) - y)
            heatmap_ = heatmap(np.linalg.norm(difference, axis=2).astype(np.float32), 0, 255)
            self.difference_output(heatmap_)

        if self.difference_output:
            if check_frequency:
                self.difference_timer.update(update)
            else:
                update()

    # Todo: move export & import somewhere else.
    def export_projections(self, prefix):
        if self.projector is None:
            return

        assert self.projector.mode == "w_projection", f"{self.projector.mode} mode is not supported for export"
        inp = self.projector.current_projected_w()
        mode = 'wplus'
        for idx in range(len(inp)):
            current_w = inp[idx:idx+1].repeat([1, self.sg.num_ws(), 1])
            res = self.sg.generate(current_w, mode)
            res_np = self.sg.torch_to_numpy_uint8(res)[0].astype('uint8')
            Image.fromarray(res_np).save(os.path.join(prefix, f'{idx:02d}.png'))

    def save_ws(self, target_dir):
        assert self.projector is not None
        w = self.get_w_torch()
        for i in range(len(w)):
            torch.save(w[i][None, ...], os.path.join(target_dir, f'{i:02d}_w.pt'))

    # Very tightly coupled to W projector..
    def load_ws(self, source_dir):
        assert self.projector is not None
        name_template = os.path.join(source_dir, '{idx:02d}_w.pt')
        num_expect = self.projector.current_input().shape[0]
        for seg in range(num_expect):
            where = name_template.format(idx=seg)
            if os.path.exists(where):
                w = torch.load(where)
                with torch.no_grad():
                    self.projector.current_projected_w_volatile()[seg].copy_(w[:, 0, :])
            else: print(f"Couldn't find input file for segment {seg}.")
        self._synthesize_with_last_params()

    def get_w_torch(self):
        if self.projector is None:
            return None
        assert self.projection_mode == "w_projection", f"{self.projection_mode} cannot export W"
        w = self.projector.current_projected_w().detach()
        if w.shape[1] == 1:
            w = w.repeat([1, self.sg.num_ws(), 1])
        return w.cpu()

    def pti_step(self):
        if self.projector is None: raise ValueError("Cannot run PTI without projection")
        if self.projection_mode != "w_projection": raise ValueError("Cannot run PTI without W")
        if self.pti_optimizer is None:
            if self.projector.target_image.shape[0] > 1:
                print("WARNING: Running Pivotal Tuning only on the first segment!")
            w_pivot = self.projector.current_projected_w()[:1, :, :].detach().clone()
            w_pivot.requires_grad_(False)
            if w_pivot.shape[1] == 1: w_pivot = w_pivot.repeat([1, self.sg.num_ws(), 1])
            self.pti_optimizer = PivotalTuning(self.sg.g.synthesis,
                self.sg.device,
                w_pivot, # Must be a W projector
                self.projector.target_image[:1, ...], 
                self.projector.current_mask[:1, ...],
                0, 1, 3e-4)
        self.pti_optimizer.step()
        self._synthesize_with_last_params()

    def forward_projection(self):
        if self.projector is None:
            self._init_projector()
        pass
        self._synthesize_with_last_params()

    def synthesize_with_project_step(self):
        # Desired projection mode changed or this is the 1st call.
        if self.projector is None or self.projector.mode != self.projection_mode:
            self._init_projector()

        # Update mask if necessary
        mask_ver = self.mask_version()
        if mask_ver != self.last_used_mask_version:
            self.projector.change_current_mask(self.mask_pull())
            self.last_used_mask_version = mask_ver
            
        # Run projector once, apply output mode
        # This discards edits, if that is not desired, simply replace with a .synthesize call.
        output = next(self.projector)
        self.last_output = output
        def update():
            nonlocal output
            output = self._apply_output_mode(output)
            self._update_unaltered_output()
            self.show_difference(output, self.target_image, check_frequency=True)
            self.output(output)
        self.projection_timer.update(update)

    def synthesize_with_params(self, edit_params):
        def impl():
            self.last_params = edit_params
            with torch.no_grad():
                return self._synthesize_with_params(edit_params)
        # self.synthesis_timer.update(impl)
        impl()

    def _synthesize_with_last_params(self):
        self.synthesize_with_params(self.last_params)

    def _synthesize_with_params_per_segment(self, edit_params):
        if self.projector is None:
            self._update_unaltered_output()
            self.output(self.target_image)
            return

        edits_by_type = dotdict()
        for edit in edit_params:
            if edit.type not in edits_by_type: edits_by_type[edit.type] = []
            edits_by_type[edit.type].append(edit)

        current_w = self.projector.current_input().detach().clone()
        if current_w.shape[1] == 1: # Broadcast to W+ for editing
            current_w = current_w.repeat([1, self.sg.num_ws(), 1])
        
        def make_segment_slice(selection):
            return slice(None) if selection == 'all' else \
                        slice(int(selection), int(selection) + 1)

        # Apply styleclip edits.
        for sc_edit in edits_by_type.get('styleclip_edit', []):
            segment_slice = make_segment_slice(sc_edit.segment)
            current_w[segment_slice, :, :] = styleclip_edit(sc_edit.parameters.model, 
                                                            current_w[segment_slice, :, :], 
                                                            sc_edit.parameters.strength)
        # Apply W edits.
        for w_edit in edits_by_type.get('w_edit', []):
            segment_slice = make_segment_slice(w_edit.segment)
            direction, lower_n, upper_n = self.sg.directions[w_edit.parameters.direction]
            total_multiplier = w_edit.parameters.value * w_edit.parameters.multiplier
            add_vector = total_multiplier * direction
            current_w[segment_slice, lower_n:upper_n, :] = (current_w + add_vector)[segment_slice, lower_n:upper_n, :]

        # Apply S edits.
        if len(edits_by_type.get('s_edit', [])) > 0:
            # Run until S space.
            current_ss = self.sg.w_to_s(current_w)
            current_ss = [s.detach().clone() for s in current_ss]
            for s_edit in edits_by_type.get('s_edit', []):
                segment_slice = make_segment_slice(s_edit.segment)
                add_value = s_edit.parameters.value * s_edit.parameters.multiplier
                current_ss[s_edit.parameters.layer][segment_slice, s_edit.parameters.channel] += add_value
            output_t = self.sg.generate(current_ss, 's')
        else:
            output_t = self.sg.generate(current_w, 'wplus')

        # Convert to images.
        output = self.sg.torch_to_numpy_uint8(output_t)
        self.last_output = output

        # Call the output functor with result
        output = self._apply_output_mode(output) # [0]
        self.show_difference(output, self.target_image, check_frequency=True)
        self.output(output)


    def _synthesize_with_params(self, edit_params):
        return self._synthesize_with_params_per_segment(edit_params)

    def _apply_output_mode(self, projection):
        # Compose the result together
        if self.output_mode == "projection_only":
            # Show first projection only for now :(
            output = projection[0]
        elif self.output_mode == "target_only":
            output = self.target_image
        elif self.output_mode == "masked":
            # Simple masking without any blend
            if self.last_used_mask_version != self.mask_version():
                mask = self.mask_pull()
                self.projector.change_current_mask(mask)
                self.last_used_mask_version = self.mask_version()
            output = self.projector.numpy_compose(projection)[0]
        elif self.output_mode == "blended":
            # Blended mode
            dst = self.target_image
            msk = (self.mask_pull() * 255).astype(np.uint8)

            if msk.shape[2] > 2: # Double blend to reduce leakage of original gradients
                for msk_idx in range(msk.shape[2]):
                    msk_slice = msk_slice = msk[:, :, msk_idx]
                    src = projection[msk_idx]
                    dilmsk = dilation(msk_slice, disk(5))
                    x, y, w, h = cv2.boundingRect(dilmsk)
                    center = ( int(x + w / 2 + 0.01), int(y + h / 2 + 0.01) )
                    dst[msk_slice == 255] = self.target_image[msk_slice == 255]
                    dst2 = cv2.seamlessClone(src, dst, dilmsk, center, cv2.NORMAL_CLONE)
                    dst = cv2.seamlessClone(dst2, dst2, msk_slice, center, cv2.NORMAL_CLONE)
            else: # Original version.
                for msk_idx in range(msk.shape[2]):
                    msk_slice = msk[:, :, msk_idx]
                    # the poisson editing allows for placing anywhere in the dst image, 
                    # so waste some time finding the bounding rect of the mask again :(
                    # which is consistent with how the implementation (see source) does it internally
                    x, y, w, h = cv2.boundingRect(msk_slice)
                    src = projection[msk_idx]
                    center = ( int(x + w / 2 + 0.01), int(y + h / 2 + 0.01) )
                    dst = cv2.seamlessClone(src, dst, msk_slice, center, cv2.NORMAL_CLONE)

            output = dst
        elif self.output_mode == "idempotent_blend":
            src = projection[0]
            dst = self.target_image
            msk = (self.mask_pull() * 255).astype(np.uint8)
            dst_cpy = dst.copy()
            dst = poisson_edit(src, dst_cpy, msk[: , :, 0], (0, 0))
            output = dst
        elif self.output_mode == "netspace":
            # blend using sg
            output = self.target_image
        else:
            raise ValueError(f"{self.output_mode} not defined")
        return output

    def _init_projector(self):
        init_gan()
        global gan
        if self.sg is None:
            self.sg = gan

        clasz = {
            'w_projection': StyleganProjector,
            'wplus_projection': StyleganWPlusProjector,
            's_projection': StyleganSSpaceProjector,
        }[self.projection_mode]
        print(f"Initializing {clasz.__name__}")

        if global_config().initial_w is not None:
            init = torch.load(global_config().initial_w)
            if init.shape != (1, 1, 512):
                init = init[:, 0:1, :]
                if init.shape != (1, 1, 512): raise ValueError(f"Wrong W shape {init.shape}.")
        else:
            init = None
        kwargs = global_config().projection_args
        
        projector = clasz(gan, self.target_image, self.mask_pull(), w_init=init, **kwargs) #  **kwargs
        projector.mode = self.projection_mode
        self.projector = projector
        self.projector_added()




