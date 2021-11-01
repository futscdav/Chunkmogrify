#
#    Author: David Futschik
#    Provided as part of the Chunkmogrify project, 2021.
#

import os
import torch
from synthesis import init_gan
from qtutil import make_dirs_if_not_exists

init_gan()
from synthesis import gan

keys = gan.g.synthesis._modules.keys()

export_to = 'components'
make_dirs_if_not_exists(export_to)
all_w = []
for key in keys:

    for subkey in gan.g.synthesis._modules[key]._modules.keys():
        w = gan.g.synthesis._modules[key]._modules[subkey].affine._parameters['weight']
        all_w.append(w)
        eigs = torch.svd(w).V.cpu()
        torch.save(eigs, os.path.join(export_to, f'{key}_{subkey}.pt'))

all_w = torch.cat(all_w, dim=0)
eigs = torch.svd(all_w).V.cpu()
torch.save(eigs, os.path.join(export_to, f'all.pt'))
