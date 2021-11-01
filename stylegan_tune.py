#
#    Author: David Futschik
#    Provided as part of the Chunkmogrify project, 2021.
#

# Straightforward Pivotal Tuning Implementation.
import torch
from lpips import LPIPS
from stylegan_project import dilation

# Changes the model itself. Make a clone if that is not desired.
# Note that mask does not change dynamically.
class PivotalTuning:
    def __init__(self, model, device, pivot_w, target_image, mask=None, alpha=1, 
            lambda_l2=1, lr=3e-4):
        self.model = model
        self.device = device
        self.w = pivot_w
        self.target = target_image
        self.mask = mask

        self.alpha = alpha
        self.lambda_l2 = lambda_l2
        self.initial_lr = lr
        self.optimizer = None
        self.current_iter = 0

    def step(self):
        if self.optimizer is None:
            self._init_opt()

        self.optimizer.zero_grad()
        current_image = self.model(self.w)
        loss = self._calc_loss(current_image)
        loss.backward()
        self.optimizer.step()

        self.current_iter += 1
        return self.model

    def iters_done(self):
        return self.current_iter

    def _calc_loss(self, x):
        if self.mask is not None:
            expanded_mask = dilation(self.mask, torch.ones(7, 7, device=self.device))
            x = x * expanded_mask + self.target * (1 - expanded_mask)

        l2_loss = torch.nn.MSELoss()(x, self.target)
        lp_loss = self.lpips_loss(x, self.target)
        sphere_loss = 0 # Seems to be disabled in PTI anyway.

        loss = l2_loss + lp_loss
        return loss

    def _init_opt(self):
        self.model.requires_grad_(True)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.initial_lr)
        self.lpips_loss = LPIPS().to(self.device)
