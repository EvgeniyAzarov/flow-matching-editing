import torch
import torch.nn.functional as F
import numpy as np
import torchode as to
import matplotlib.pyplot as plt

from models.unet import UNet
from torch import nn


class FlowMatching(nn.Module):
    def __init__(self, c_unet=16, sigma=1e-5):
        super().__init__()

        self.sigma = sigma
        self.model = UNet(
            in_channels=1,
            out_channels=1,
            base_dim=32,
            dim_mults=[2, 4, 8, 16],
            n_classes=10,
            cond_channels=1,
            time_embedding_dim=1,
        )
    
    def forward(self, batch):
        # latent: [B, C, H, W]
        latent, labels = batch

        noise = torch.randn_like(latent)
        time = torch.rand(latent.shape[0], device=latent.device)
        t = time.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        # eq 22 and 23 in the paper
        x_t = (1 - (1 - self.sigma) * t) * noise + t * latent
        target = latent - (1 - self.sigma) * noise

        v = self.model(x_t, labels, t)
        loss = F.mse_loss(v, target)

        return loss
    
    def generate(self, batch_size, h=32, w=32, n_steps=2, classes=None, device="cuda"):
        noise = torch.randn(batch_size, self.model.in_channels, h, w, device=device)
        t = torch.linspace(0, 1, n_steps, device=device)
        t = t.unsqueeze(0).repeat(batch_size, 1)

        if classes is None:
            classes = torch.randint(0, 10, (batch_size, ), device=device)
        else:
            classes = torch.tensor(classes).to(device)

        def fn(t, x):
            x = x.view(*noise.shape)
            t = t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            out = self.model(x, classes, t)
            return out.view(batch_size, -1)

        # ODE Solver
        y0 = noise.view(batch_size, -1)
        term = to.ODETerm(fn)
        step_method = to.Dopri5(term=term)
        step_size_controller = to.IntegralController(
            atol=1e-6, rtol=1e-3, term=term
        )
        solver = to.AutoDiffAdjoint(step_method, step_size_controller)
        output = solver.solve(to.InitialValueProblem(y0=y0, t_eval=t))
        output = output.ys[:, -1].view(*noise.shape)

        return output


def display_generations(model, filepath, device="cuda", n_samples=5):
    generated = model.generate(n_samples * n_samples, device=device)

    fig, axes = plt.subplots(n_samples, n_samples, figsize=(n_samples * 2, n_samples * 2))
    axs = axes.flatten()

    for i in range(n_samples * n_samples):
        axs[i].imshow(generated[i].detach().cpu().numpy()[0], cmap="grey")

    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
