#
#    Author: David Futschik
#    Provided as part of the Chunkmogrify project, 2021.
#

import math
import torch
import numpy as np
from torch.nn import functional as F

def torch_grad(x):
    a = torch.tensor([[-1, 0, 1]], dtype=torch.float32, device=x.device, requires_grad=False).view((1, 1, 1, 3))
    b = torch.tensor([[-1, 0, 1]], dtype=torch.float32, device=x.device, requires_grad=False).view((1, 1, 3, 1))
    G_x = F.conv2d(x, a) / 2
    G_y = F.conv2d(x, b) / 2
    G_x = F.pad(G_x, (1, 1, 0, 0), 'constant', 0.)
    G_y = F.pad(G_y, (0, 0, 1, 1), 'constant', 0.)
    return [G_x, G_y]

def torch_normgrad_curv(x):
    G_x, G_y = torch_grad(x)
    G = torch.sqrt(torch.pow(G_x,2) + torch.pow(G_y,2))
    # div 
    div = torch_grad(G_x / (G + 1e-8) )[0] + torch_grad(G_y / (G + 1e-8))[1]
    return G, div

def contrast_magnify(x, min=0, max=64, fromval=0., toval=255.):
    mul_by = (toval - min) / max
    x = ((x - min) * mul_by).clip(fromval, toval)
    return x

def mask_refine(mask, image1, image2, dt_A=0.001, dt_B=0.1, iters=300):

    S = (mask - 0.5)
    P = S[:, 0:1, :, :]
    
    Pg = torch_normgrad_curv(P)[0]

    I_1_mean = ((Pg * image1)).sum(dim=(2,3)) / Pg.sum()
    I_2_mean = ((Pg * image2)).sum(dim=(2,3)) / Pg.sum()

    assert I_1_mean.shape == (1, 3), "Mean wrong shape"
    I_1 = image1 - I_1_mean[:, :, None, None]
    I_2 = image2 - I_2_mean[:, :, None, None]
    Fn = I_1 - I_2
    Fn = Fn.norm(dim=1) / math.sqrt(12)

    P = P * 255
    Fn = Fn * 255

    Fn = contrast_magnify(Fn)

    with torch.no_grad():
        for _ in range(iters):
            ng, div = torch_normgrad_curv(P)
            P -= dt_A * Fn * ng - dt_B * ng * div

    P = torch.where(P < 0, 0., 1.)

    new_mask = P
    return new_mask

if __name__ == "__main__":
    import PIL.Image

    i1 = PIL.Image.open("_I1_.png")
    i2 = PIL.Image.open("_I2_.png")
    s = PIL.Image.open("_P_.png")

    i1 = (torch.tensor( np.array(i1).astype(np.float), dtype=torch.float32, device='cuda:0' ) / 127.5).permute((2,0,1)).unsqueeze(0)
    i2 = (torch.tensor( np.array(i2).astype(np.float), dtype=torch.float32, device='cuda:0' ) / 127.5).permute((2,0,1)).unsqueeze(0)
    s = (torch.tensor( np.array(s).astype(np.float), dtype=torch.float32, device='cuda:0' ) / 255.).unsqueeze(0).unsqueeze(0)

    m = mask_refine(s, i1 - 1, i2 - 1)

    d = m.cpu().numpy()[0].transpose((1,2,0)).__mul__(255.).astype(np.uint8)
    PIL.Image.fromarray(d[:, :, 0]).save("maskrefine.png")
