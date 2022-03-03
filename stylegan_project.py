#
#    Author: David Futschik
#    Provided as part of the Chunkmogrify project, 2021.
#

import torch
import stylegan_legacy
import dnnlib
import math
import types
import functools

import numpy as np

from torch import optim
from torch.nn import functional as F
from torch_utils import misc
from config import global_config
from w_directions import known_directions
from styleclip_presets import pretrained_models as styleclip_models
from mask_refinement import mask_refine
from styleclip_mapper import StyleCLIPMapper

class dotdict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
    def has_set(self, attr):
        return attr in self and self[attr] is not None

class PerceptualLoss:
    def __init__(self, device, use_torchvision=False):
        self.using_torchvision = use_torchvision
        # preload vgg
        if use_torchvision:
            import torchvision
            self.vgg = torchvision.models.vgg16(pretrained=True).to(device)
            self.vgg.requires_grad_(False)

            def to_vgg_space(x):
                x = x / 255
                x = x - (torch.Tensor([0.485, 0.456, 0.406]).to(x.device).view(1, -1, 1, 1))
                x = x / (torch.Tensor([0.229, 0.224, 0.225]).to(x.device).view(1, -1, 1, 1))
                return x

            def new_forward(self, x):
                feature_submod = self.features
                x = to_vgg_space(x)
                f = feature_submod(x)
                f = f.flatten(1)
                f = f / f.norm()
                print(f.shape)
                return f
            self.vgg.forward = types.MethodType(new_forward, self.vgg)
        else:
            url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
            with dnnlib.util.open_url(url) as f:
                self.vgg = torch.jit.load(f).eval().to(device)
        
    def __call__(self, x):
        if not self.using_torchvision:
            return self.vgg(x, resize_images=False, return_lpips=True)
        return self.vgg(x)

class StyleganProvider:
    def __init__(self, network_pkl, device, native_resolution):
        np.random.seed(0)
        torch.manual_seed(0)
        self.device = device

        if self.device.startswith('cuda'):
            if not torch.cuda.is_available():
                print(f"Device {device} requested, but cuda is not available. Using CPU.")
                self.device = 'cpu'

        if global_config().generator_load_raw:
            g_ema = torch.load(network_pkl).requires_grad_(False).to(device)
        else:
            with dnnlib.util.open_url(network_pkl) as fp:
                g_ema = stylegan_legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device)

        # Because G carries its code with it, need to reconstruct it for code changes
        G_kwargs = dnnlib.EasyDict(class_name='stylegan_networks.Generator', z_dim=512, w_dim=512, mapping_kwargs=dnnlib.EasyDict(), synthesis_kwargs=dnnlib.EasyDict())
        # G_kwargs.synthesis_kwargs.channel_base = int(1 * 32768)
        # G_kwargs.synthesis_kwargs.channel_max = 512
        G_kwargs.mapping_kwargs.num_layers = 8
        G_kwargs.synthesis_kwargs.num_fp16_res = 4 # enable mixed-precision training
        G_kwargs.synthesis_kwargs.conv_clamp = 256 # clamp activations to avoid float16 overflow

        assert native_resolution[0] == native_resolution[1], "Only square images supported."
        common_kwargs = dict(c_dim=0, img_resolution=native_resolution[0], img_channels=3)
        self.g = dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).train().requires_grad_(False).to(device)
        misc.copy_params_and_buffers(g_ema, self.g, require_all=True, rand_init_extra_channels=False)
        del g_ema

        # Precompute mean space
        w_avg_samples = 10000
        z_samples = np.random.RandomState(123).randn(w_avg_samples, self.g.z_dim)
        w_samples = self.g.mapping(torch.from_numpy(z_samples).to(self.device), None)  # [N, L, C]
        w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)       # [N, 1, C]
        w_avg = np.mean(w_samples, axis=0, keepdims=True)      # [1, 1, C]
        w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5
        self.mean_latent = w_avg
        self.std_latent = w_std

        self.vgg = PerceptualLoss(device=device, use_torchvision=False)

        # load directions
        self.directions = {}
        for name, args in known_directions().items():
            vec, lower_n, upper_n = args
            if name in ['synthetic_glasses']:
                z = torch.from_numpy(z_samples).to(device)
                w = self.g.mapping(z.to(device), None)
                zpd = z + vec.to(device)
                wpd = self.g.mapping(zpd, None)
                d = wpd - w
                d = -d.mean(dim=0)
                vec = d

            if type(vec) == str:
                self.directions[name] = (torch.tensor(np.load(vec, allow_pickle=True), device=self.device).unsqueeze(0), lower_n, upper_n)
            else:
                self.directions[name] = (torch.tensor(vec, device=self.device).unsqueeze(0), lower_n, upper_n)
            
            if self.directions[name][0].shape[1] == 1:
                self.directions[name] = (self.directions[name][0].repeat(1, self.num_ws(), 1), self.directions[name][1], self.directions[name][2])

            
        # Fix network for cpu generation
        if self.device == 'cpu':
            self.g.synthesis.forward = functools.partial(self.g.synthesis.forward, force_fp32=True)

    def generate(self, input, mode):
        with torch.no_grad():
            is_cpu = global_config().device == 'cpu'
            if mode == 'wplus' or mode == "wplus_projection":
                i = self.g.synthesis(input, noise_mode='const', force_fp32=is_cpu)
            elif mode == 'w' or mode == "w_projection":
                i = self.g.synthesis(input.repeat([1, self.num_ws(), 1]), noise_mode='const', force_fp32=is_cpu)
            elif mode == 's' or mode == 's_projection':
                i = self.g.synthesis.forward_s(input, noise_mode='const', force_fp32=is_cpu)
        return i

    def generate_numpy(self, input, mode):
        t = self.generate(input, mode)
        return self.torch_to_numpy_uint8(t)

    def w_to_s(self, w):
        with torch.no_grad():
            if w.shape[1] == 1:
                w = w.repeat([1, self.num_ws(), 1])
            _, ss = self.g.synthesis(w, return_ss=True)
        return ss

    def num_ws(self):
        return self.g.mapping.num_ws

    def numpy_uint8_to_torch(self, x):
        if type(x) == list:
            # Explicit np.array(x) is about 100 times faster than letting torch handle it.
            # This bug is documented and reported, but WONTFIX.
            tensor = torch.tensor(np.array(x), device=self.device, dtype=torch.float32, requires_grad=False)
            # Drop channels past 3.
            tensor = tensor.permute((0, 3, 1, 2))[:, :3, ...]
        else:
            assert x.dtype == np.uint8, f"Wrong type {x.dtype} (expected uint8)"
            if x.shape[2] == 4:
                x = x[..., :3]
            tensor = torch.tensor(x.transpose((2,0,1)), device=self.device, dtype=torch.float32, requires_grad=False).unsqueeze(0)
        # Equivalent of ToTensor.
        tensor.div_(255.)
        # Normalize to -1, 1.
        tensor.add_(-0.5).mul_(2.)
        return tensor

    def numpy_float_mask_to_torch(self, x):
        # assume already in [0, 1, (N)]
        assert x.dtype == np.float, "Wrong mask type"
        assert x.min() >= 0. and x.max() <= 1., f"Mask array has range [{x.min()},{x.max()}]"
        tensor = torch.tensor(x, dtype=torch.float32, device=self.device, requires_grad=False)
        if tensor.ndim == 3:
            tensor = tensor.permute((2, 0, 1))
        return tensor

    def torch_mask_to_numpy(self, x):
        assert x.min() >= 0. and x.max() <= 1., f"Mask array has range [{x.min()},{x.max()}]"
        return x.cpu().numpy()

    def torch_to_numpy_uint8(self, x):
        # assert x.min() >= -1.01 and x.max() <= 1.01, f"Image doesn't have the correct range ({x.min(), x.max()})"
        image = (x.clamp_(min=-1, max=1)
                  .add(1).div_(2).mul_(255).clip_(0, 255)
                  .type(torch.uint8)
                  .permute(0, 2, 3, 1)
                  .to("cpu").numpy())
        images = []
        for i in range(len(image)):
            im = image[i]
            images.append(im)
        return images

# Copied from Kornia to avoid the dependency just for this.
def dilation(
    tensor: torch.Tensor,
    kernel: torch.Tensor,
    structuring_element = None,
    origin = None,
    border_type: str = 'geodesic',
    border_value: float = 0.0,
    max_val: float = 1e4,
) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(tensor)}")

    if len(tensor.shape) != 4:
        raise ValueError(f"Input size must have 4 dimensions. Got {tensor.dim()}")

    if not isinstance(kernel, torch.Tensor):
        raise TypeError(f"Kernel type is not a torch.Tensor. Got {type(kernel)}")

    if len(kernel.shape) != 2:
        raise ValueError(f"Kernel size must have 2 dimensions. Got {kernel.dim()}")

    # origin
    se_h, se_w = kernel.shape
    if origin is None:
        origin = [se_h // 2, se_w // 2]

    # pad
    pad_e = [origin[1], se_w - origin[1] - 1, origin[0], se_h - origin[0] - 1]
    if border_type == 'geodesic':
        border_value = -max_val
        border_type = 'constant'
    output: torch.Tensor = F.pad(tensor, pad_e, mode=border_type, value=border_value)

    # computation
    if structuring_element is None:
        neighborhood = torch.zeros_like(kernel)
        neighborhood[kernel == 0] = -max_val
    else:
        neighborhood = structuring_element.clone()
        neighborhood[kernel == 0] = -max_val

    output = output.unfold(2, se_h, 1).unfold(3, se_w, 1)
    output, _ = torch.max(output + neighborhood.flip((0, 1)), 4)
    output, _ = torch.max(output, 4)

    return output

def is_w(x):
    assert x.ndim == 3
    if x.shape[1] == 1: return True
    return torch.all(x[:, 0:1, :] == x[:, :, :])

class StyleganProjector:
    def __init__(self, provider: StyleganProvider,
                    image: np.ndarray,
                    initial_mask: np.ndarray,
                    iters=500, w_init=None, lr_init=0.1, lr_rampup=0.05, 
                    lr_rampdown=0.25, noisy_latent=False, noise_regularize_weight=1e5, w_plus=False, resize_to=512,
                    l1_loss_weight=0., l2_loss_weight=0., percept_downsample=1, mean_latent_loss_weight=0.):
        self.provider = provider
        self.device = provider.device
        self.target_image = self.provider.numpy_uint8_to_torch(image)

        self.current_mask = self.provider.numpy_float_mask_to_torch(initial_mask)
        if self.current_mask.ndim == 2:
            # One mask for all channels & images
            self.current_mask = self.current_mask.unsqueeze(0).repeat([self.target_image.shape[1], 1, 1])
        if self.current_mask.ndim == 3:
            # Multiple masks:: Set Number of Target Images equal to number of masks!
            self.current_mask = self.current_mask.unsqueeze(1).repeat([1, self.target_image.shape[1], 1, 1])
            self.target_image = self.target_image.repeat([self.current_mask.shape[0], 1, 1, 1])
        assert self.current_mask.ndim == 4, "Wrong mask arguments"

        if self.current_mask.sum() <= 1.:
            print("Warning: Initial mask is empty!")

        self.current_iter = 0
        self.max_iters = iters
        self.lr_init = lr_init
        self.lr_rampup = lr_rampup
        self.lr_rampdown = lr_rampdown
        self.noisy_latent = noisy_latent
        self.noise_regularize_weight = noise_regularize_weight
        self.w_plus = w_plus
        self.resize_to = resize_to
        self.l1_loss_weight = l1_loss_weight
        self.l2_loss_weight = l2_loss_weight
        self.mean_latent_loss_weight = mean_latent_loss_weight
        self.percept_downsample = percept_downsample
        if w_init is None:
            self.w_init = torch.tensor(self.provider.mean_latent, dtype=torch.float32, device=self.provider.device, requires_grad=False)
            if self.w_plus:
                self.w_init = self.w_init.repeat(1, provider.g.mapping.num_ws, 1)
        else:
            self.w_init = torch.tensor(w_init, dtype=torch.float32, device=self.provider.device, requires_grad=False)

        self.noise = { name: buf for (name, buf) in self.provider.g.synthesis.named_buffers() if 'noise_const' in name }

        self.optimizer = None

    # Unused. Only used for the entangled stepper.
    def _init_optimizer(self):
        self.current_w = self.w_init.clone()
        # same number as target images / masks
        self.current_w = self.current_w.repeat((self.target_image.shape[0], 1, 1))
        self.current_w.requires_grad = True
        self.optimize = [self.current_w] + list(self.noise.values())
        self.optimizer = optim.Adam(self.optimize, lr=self.lr_init)

    def num_ws(self):
        return self.provider.g.mapping.num_ws

    def current_projected_w(self):
        if not hasattr(self, 'current_ws'):
            return torch.cat([self.w_init.clone() for _ in range(self.target_image.shape[0])], dim=0)
        return torch.cat(self.current_ws, dim=0)
        # WARNING: CHANGED FOR INDIVIDUAL FWD
        # return self.current_w.detach().clone()

    def current_projected_w_volatile(self):
        # If no step has been done, errors out.
        return self.current_ws

    def current_input(self):
        return self.current_projected_w()

    def current_mask_as_numpy(self):
        return self.current_mask.cpu().numpy()

    def generate(self, w):
        if w.shape[1] == 1:
            w = w.repeat([1, self.provider.g.mapping.num_ws, 1])
        timg = self.provider.generate(w, mode='wplus')
        return timg

    def generate_numpy(self, w):
        return self.provider.torch_to_numpy_uint8(self.generate(w))

    def change_current_mask(self, mask_np: np.ndarray):
        self.current_mask = self.provider.numpy_float_mask_to_torch(mask_np)
        if self.current_mask.ndim == 2:
            # One mask for all channels & images
            self.current_mask = self.current_mask.unsqueeze(0).repeat([self.target_image.shape[1], 1, 1])
        if self.current_mask.ndim == 3:
            # Multiple masks:: Set Number of Target Images equal to number of masks!
            self.current_mask = self.current_mask.unsqueeze(1).repeat([1, self.target_image.shape[1], 1, 1])
            assert self.target_image.shape[0] == self.current_mask.shape[0], "Target not updated to mask sizes (??)"
        assert self.current_mask.ndim == 4, "Wrong mask arguments"

        # Sanity check, no values are too far from 0 or 1
        # assert (torch.logical_and(self.current_mask < 0.9, self.current_mask > 0.1)).any() == False

    def get_lr(self, step):
        lr_ramp = min(1, (1 - step) / self.lr_rampdown)
        lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
        lr_ramp = lr_ramp * min(1, step / self.lr_rampup)
        return self.lr_init * lr_ramp

    def numpy_compose(self, res):
        # Expect a list / stack of images, 1 for each mask
        # composite = self.provider.numpy_uint8_to_torch(res) * self.current_mask + \
        #         self.target_image * ( 1 - self.current_mask )
        composite = (self.provider.numpy_uint8_to_torch(res) * self.current_mask).sum(dim=0, keepdim=True) + \
                        (self.target_image[0] * (1 - self.current_mask.sum(dim=0)))
        return self.provider.torch_to_numpy_uint8(composite)

    def mask_compose(self, res):
        return res * self.current_mask + self.target_image * (1 - self.current_mask)

    def current_as_torch(self):
        w = self.current_projected_w().repeat([1, self.provider.g.mapping.num_ws, 1])
        timg = self.provider.generate(w, 'wplus')
        return timg

    def current_as_numpy(self):
        timg = self.current_as_torch()
        res = self.provider.torch_to_numpy_uint8(timg)
        return res

    def _calc_loss_with_bbox(self, x, m):
        # Find the bbox of the mask
        # This assumes torch argmax returns the first occurence!
        def first_of(x, dim):
            v, m = torch.max(x, dim=dim)
            m[v == 0.] = x.shape[dim]
            return m.min()
        def bbox_get(x):
            assert x.ndim == 2
            top = first_of(x, dim=0)
            bottom = x.shape[0] - first_of(torch.flip(x, dims=(0,)), dim=0)
            
            left = first_of(x, dim=1)
            right = x.shape[1] - first_of(torch.flip(x, dims=(1,)), dim=1)
            return (top, bottom, left, right)
        with torch.no_grad():
            bbox = bbox_get(m[0, 0])
        
        loss = 0.
        target_images = (self.target_image[0:1, ...] + 1) * (255./2)
        target_images = target_images[:, :, bbox[0]:bbox[1]+1, bbox[2]:bbox[3]+1]
        target_images = F.interpolate(target_images, size=(256, 256), mode='area')
        target_features = self.provider.vgg(target_images, resize_images=False)
        synth_images = (x + 1) * (255./2)
        synth_images = synth_images[:, :, bbox[0]:bbox[1]+1, bbox[2]:bbox[3]+1]
        synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')
        synth_features = self.provider.vgg(synth_images)
        vgg_dist = self._pairwise_distances_cos(synth_features, target_features)# .square().sum()
        loss += vgg_dist
        return loss

    def _pairwise_distances_cos(self, x, y):
        x_norm = torch.sqrt((x**2).sum(1).view(-1, 1))
        y_t = torch.transpose(y, 0, 1)
        y_norm = torch.sqrt((y**2).sum(1).view(1, -1))
        dist = 1.-torch.mm(x, y_t)/x_norm/y_norm
        return dist

    def _calc_loss(self, x, w=None, wplus=None):
        size = (int(x.shape[2] * self.percept_downsample), int(x.shape[3] * self.percept_downsample))
        if not hasattr(self, 'precomputed_loss_data'):
            self.precomputed_loss_data = dotdict()
            if self.target_image.shape[2] > size[0] or self.target_image.shape[3] > size[1]:
                print("Loss Taking Features from 0th image!")
                target_images = (self.target_image[0:1, ...] + 1) * (255./2)
                target_images = F.interpolate(target_images, size=size, mode='area')
            else:
                target_images = (self.target_image[0:1, ...] + 1) * (255./2)
            self.precomputed_loss_data.target_features = self.provider.vgg(target_images)
        
        loss = 0.
        synth_images = (x + 1) * (255./2)
        synth_images = F.interpolate(synth_images, size=size, mode='area')
        synth_features = self.provider.vgg(synth_images)
        vgg_dist = (synth_features - self.precomputed_loss_data.target_features).square().sum()
        loss += vgg_dist

        if self.l1_loss_weight > 0.:
            loss += self.l1_loss_weight * F.l1_loss(x, self.target_image[0:1])

        if self.l2_loss_weight > 0.:
            loss += self.l2_loss_weight * F.mse_loss(x, self.target_image[0:1])

        if self.mean_latent_loss_weight > 0. and w is not None:
            loss += self.mean_latent_loss_weight * (
                ((w - torch.tensor(self.provider.mean_latent).to(self.device)) 
                    / torch.tensor(self.provider.std_latent).to(self.device)) ** 2
            ).mean()

        if self.mean_latent_loss_weight > 0. and wplus is not None:
            loss += self.mean_latent_loss_weight * (
                ((wplus - torch.tensor(self.provider.mean_latent).to(self.device)) 
                    / torch.tensor(self.provider.std_latent).to(self.device)) ** 2
            ).mean() / wplus.shape[1]

        # noise regularization
        if self.noise_regularize_weight > 0.:
            noise_loss = 0.0
            for v in self.noise.values():
                noise = v[None, None, :, :] # must be [1,1,H,W] for F.avg_pool2d()
                while True:
                    noise_loss += (noise * torch.roll(noise, shifts=1, dims=3)).mean() ** 2
                    noise_loss += (noise * torch.roll(noise, shifts=1, dims=2)).mean() ** 2
                    if noise.shape[2] <= 8:
                        break
                    noise = F.avg_pool2d(noise, kernel_size=2)
            loss += self.noise_regularize_weight * noise_loss
        return loss

    def _project_step_individual(self):
        if not hasattr(self, 'optimizers'): # self.optimizers is None, so init.

            self.current_ws = [self.w_init.clone() for _ in range(self.target_image.shape[0])]

            # same number as target images / masks
            for x in self.current_ws:
                x.requires_grad = True
            self.optimizers = [optim.Adam([x], lr=self.lr_init) for x in self.current_ws]
            print("Current loss: L2 distance in hypercolumns (no bbox)")

        t = self.current_iter / self.max_iters
        current_lr = self.get_lr(t)
        
        # run batches of 1 to conserve GPU mem
        all_res = []
        for xx in range(self.target_image.shape[0]):
            if torch.all(self.current_mask[xx:xx+1, ...] == 0.): 
                all_res.append(self.target_image[xx:xx+1])
                continue

            self.optimizers[xx].param_groups[0]['lr'] = current_lr

            w_use = self.current_ws[xx]
            
            if w_use.shape[1] == 1:
                w_use = w_use.repeat([1, self.provider.g.mapping.num_ws, 1])
            
            seg_r = self.provider.g.synthesis(w_use, noise_mode='const')
            all_res.append(seg_r.detach().clone())
            
            # Compose approach
            expanded_mask = dilation(self.current_mask[xx:xx+1], torch.ones(7, 7, device=self.current_mask.device))
            seg_r = seg_r * expanded_mask + self.target_image[xx:xx+1, ...] * (1 - expanded_mask)

            # Grad masking approach, works a bit worse, because averaging was done on full image
            # and the vgg loss is leaky in that regard.
            # seg_r.register_hook(lambda grad: grad * self.current_mask[xx:xx+1, ...])

            loss = 0.
            
            # If bbox approach is desired:
            # loss += self._calc_loss_with_bbox(seg_r, self.current_mask[xx:xx+1])
            loss += self._calc_loss(seg_r, w_use)
            self.optimizers[xx].zero_grad()
            loss.backward()
            self.optimizers[xx].step()

        # Normalize noise.
        with torch.no_grad():
            for buf in self.noise.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()

        res = torch.cat(all_res)
        return self.provider.torch_to_numpy_uint8(res.detach())
        
    # Unused. Entangles different projections
    def _project_step(self):
        if self.optimizer is None:
            self._init_optimizer()

        g = self.provider.g
        t = self.current_iter / self.max_iters
        current_lr = self.get_lr(t)
        # set the current lr for all params
        self.optimizer.param_groups[0]['lr'] = current_lr
        if self.noisy_latent:
            initial_noise_factor, noise_ramp_length = 0.05, 0.75
            w_noise_scale = self.provider.std_latent * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
            w_noise = torch.randn_like(self.current_w) * w_noise_scale
            w_use = self.current_w + w_noise
        else:
            w_use = self.current_w
        
        if w_use.shape[1] == 1:
            w_use = w_use.repeat([1, self.provider.g.mapping.num_ws, 1])
        
        # run batches of 1 to conserve GPU mem
        all_res = []
        for xx in range(len(w_use)):
            if torch.all(self.current_mask[xx:xx+1, ...] == 0.): 
                all_res.append(self.target_image[xx:xx+1])
                continue
            all_res.append(self.provider.g.synthesis(w_use[xx, ...][None, ...], noise_mode='const'))
        res = torch.cat(all_res)

        composite = res.detach().clone()

        # mask out for loss calc
        res = self.mask_compose(res)

        loss = self._calc_loss(res)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Normalize noise.
        with torch.no_grad():
            for buf in self.noise.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()
        return self.provider.torch_to_numpy_uint8(composite.detach())

    def _mask_refine_step(self, current_active_image=None, dt=None):
        if current_active_image is None:
            # run current w forward:
            current_active_image = self.current_as_torch()

        # ONLY WORKS FOR 1 SEGMENT
        new_mask = mask_refine(self.current_mask, self.target_image, current_active_image.detach(), iters=300)
        self.current_mask = new_mask.view(-1, 1, self.target_image.shape[2], self.target_image.shape[3]).repeat([1, self.target_image.shape[1], 1, 1])

    def __next__(self):
        nxt = self._project_step_individual() # self._project_step()
        self.current_iter += 1
        return nxt

class StyleganWPlusProjector(StyleganProjector):
    def __init__(self, provider: StyleganProvider,
                    image: np.ndarray,
                    initial_mask: np.ndarray,
                    iters=500, w_init=None, lr_init=0.1, lr_rampup=0.05, 
                    lr_rampdown=0.25, noisy_latent=False, noise_regularize_weight=1e3, w_plus=False, resize_to=512, **kwargs):
        super().__init__(provider, image, initial_mask, iters, w_init, lr_init, lr_rampup, lr_rampdown, 
                noisy_latent, noise_regularize_weight, w_plus=False, resize_to=resize_to, **kwargs)
        print(f"Ignoring {kwargs}")

    def generate(self, wplus):
        return self.provider.g.synthesis(wplus, noise_mode='const')

    def current_as_torch(self):
        return self.generate(self.current_wplus)

    def current_as_numpy(self):
        return self.generate_numpy(self.current_wplus)

    def current_input(self):
        return self.current_projected_wplus()

    def current_projected_wplus(self):
        return self.current_wplus.detach().clone()

    def generate_numpy(self, wplus):
        timg = self.generate(wplus)
        res = self.provider.torch_to_numpy_uint8(timg)
        return res

    def _init_optimizer(self):
        self.current_w = self.w_init.clone()
        # same number as target images / masks
        self.current_wplus = self.current_w.repeat((self.target_image.shape[0], self.provider.g.mapping.num_ws, 1))
        self.current_wplus.requires_grad = True
        self.optimize = [self.current_wplus] + list(self.noise.values())
        self.optimizer = optim.Adam(self.optimize, lr=self.lr_init)

    def _project_step(self):
        if self.optimizer is None:
            self._init_optimizer()

        g = self.provider.g
        t = self.current_iter / self.max_iters
        current_lr = self.get_lr(t)
        # set the current lr for all params
        self.optimizer.param_groups[0]['lr'] = current_lr
        assert not self.noisy_latent, "Noisy latent not supported for S space"
        
        # run batches of 1 to conserve GPU mem
        res = self.provider.g.synthesis(self.current_wplus, noise_mode='const')

        composite = res.detach().clone()

        # mask out for loss calc
        res = self.mask_compose(res)

        loss = self._calc_loss(res, wplus=self.current_wplus)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Normalize noise.
        with torch.no_grad():
            for buf in self.noise.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()
        return self.provider.torch_to_numpy_uint8(composite.detach())

    def __next__(self):
        nxt = self._project_step() # self._project_step()
        self.current_iter += 1
        return nxt


class StyleganSSpaceProjector(StyleganProjector):
    def __init__(self, provider: StyleganProvider,
                    image: np.ndarray,
                    initial_mask: np.ndarray,
                    iters=500, w_init=None, lr_init=0.1, lr_rampup=0.05, 
                    lr_rampdown=0.25, noisy_latent=False, noise_regularize_weight=1e3, w_plus=False, resize_to=512, **kwargs):
        super().__init__(provider, image, initial_mask, iters, w_init, lr_init, lr_rampup, 
            lr_rampdown, noisy_latent, noise_regularize_weight, w_plus=False, resize_to=resize_to, **kwargs)
        print(f"Ignoring {kwargs}")

    def generate(self, ss):
        return self.provider.g.synthesis.forward_s(ss, noise_mode='const')

    def current_as_torch(self):
        return self.generate(self.current_ss)

    def current_as_numpy(self):
        return self.generate_numpy(self.current_ss)

    def current_input(self):
        return self.current_projected_ss()

    def current_projected_ss(self):
        return self.current_ss

    def generate_numpy(self, ss):
        timg = self.generate(ss)
        res = self.provider.torch_to_numpy_uint8(timg)
        return res

    def _init_optimizer(self):
        with torch.no_grad():
            _, self.ss_init = self.provider.g.synthesis(self.w_init.repeat([1, self.provider.g.mapping.num_ws, 1]), return_ss=True)
        print("Size of ss:", len(self.ss_init))
        print("Shape: ", [s.shape for s in self.ss_init])
        self.current_ss = [s.clone().requires_grad_(True) for s in self.ss_init]
        self.optimize = self.current_ss + list(self.noise.values())
        self.optimizer = optim.Adam(self.optimize, lr=self.lr_init)

    def _project_step(self):
        if self.optimizer is None:
            self._init_optimizer()

        g = self.provider.g
        t = self.current_iter / self.max_iters
        current_lr = self.get_lr(t)
        # set the current lr for all params
        self.optimizer.param_groups[0]['lr'] = current_lr
        assert not self.noisy_latent, "Noisy latent not supported for S space"
        
        # run batches of 1 to conserve GPU mem
        res = self.provider.g.synthesis.forward_s(self.current_ss, noise_mode='const')

        composite = res.detach().clone()

        # mask out for loss calc
        res = self.mask_compose(res)

        loss = self._calc_loss(res)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Normalize noise.
        with torch.no_grad():
            for buf in self.noise.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()
        return self.provider.torch_to_numpy_uint8(composite.detach())

    def __next__(self):
        nxt = self._project_step()
        self.current_iter += 1
        return nxt


_styleclip_model_cache = dotdict()
def styleclip_edit(name, w, strength):
    global _styleclip_model_cache
    if name not in _styleclip_model_cache:
        path = styleclip_models()[name]
        checkpoint = torch.load(path)
        options = checkpoint['opts']
        options = dotdict(**options)
        options.checkpoint_path = path
        model = StyleCLIPMapper(options)
        model.eval()
        model.to(global_config().device)
        _styleclip_model_cache[name] = model
    w_new = w + strength * _styleclip_model_cache[name].mapper(w)
    return w_new
