import torch
import torch.nn.functional as F
import numpy as np
import torchode as to
import matplotlib.pyplot as plt

from models.unet_cond import UNetCond
from torch import nn
from transformers import AutoFeatureExtractor, Data2VecVisionModel
from torchvision.transforms import v2
from torchvision import models


class FlowMatchingForEditing(nn.Module):
    def __init__(self, sigma=1e-5):
        super().__init__()

        self.sigma = sigma
        self.cond_channels = 32 + 4
        self.model = UNetCond(
            in_channels=1,
            out_channels=1,
            base_dim=32,
            dim_mults=[2, 4, 8, 16],
            n_classes=10,
            class_channels=1,
            time_embedding_dim=1,
            cond_channels=self.cond_channels,
        )

        self.effnet_transform = v2.Compose([
            v2.Grayscale(3),   # Convert 1-channel to 3-channel
            v2.Resize((224, 224)),
            v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize as per ImageNet
        ])
        self.effnet_model = models.efficientnet_b0(pretrained=True)
        self.effnet_proj = nn.Linear(1000, 32)

        self.slope_angle_proj = nn.Linear(1, 4)
        
    def forward(self, batch):
        # latent: [B, C, H, W]
        latent, labels = batch["latent"], batch["label"]

        effnet_inputs = self.effnet_transform(latent)
        with torch.no_grad():
            effnet_outputs= self.effnet_model(effnet_inputs)
        effnet_features = self.effnet_proj(effnet_outputs) 

        slope_angles = batch["slope_angle"].unsqueeze(-1)
        slope_angles = self.slope_angle_proj(slope_angles)

        condition = torch.cat([effnet_features, slope_angles], dim=1)

        noise = torch.randn_like(latent)
        time = torch.rand(latent.shape[0], device=latent.device)
        t = time.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        # eq 22 and 23 in the paper
        x_t = (1 - (1 - self.sigma) * t) * noise + t * latent
        target = latent - (1 - self.sigma) * noise

        v = self.model(x_t, labels, t, condition=condition)
        loss = F.mse_loss(v, target)

        return loss

    def generate(self, images, classes, angles, h=32, w=32, device="cuda"):
        batch_size = images.shape[0]
        classes = torch.tensor(classes).to(device)

        effnet_inputs = self.effnet_transform(images)
        with torch.no_grad():
            effnet_outputs= self.effnet_model(effnet_inputs)
        effnet_features = self.effnet_proj(effnet_outputs) 

        slope_angles = torch.tensor(angles, device=device).unsqueeze(-1)
        slope_angles = self.slope_angle_proj(slope_angles)

        condition = torch.cat([effnet_features, slope_angles], dim=1)

        noise = torch.randn(batch_size, self.model.in_channels, h, w, device=device)
        t = torch.linspace(0, 1, 2, device=device)
        t = t.unsqueeze(0).repeat(batch_size, 1)

        def fn(t, x):
            x = x.view(*noise.shape)
            t = t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            out = self.model(x, classes, t, condition=condition)
            return out.view(batch_size, -1)

        # ODE Solver
        y0 = noise.view(batch_size, -1)
        term = to.ODETerm(fn)
        step_method = to.Dopri5(term=term)
        step_size_controller = to.IntegralController(atol=1e-6, rtol=1e-3, term=term)
        solver = to.AutoDiffAdjoint(step_method, step_size_controller)
        output = solver.solve(to.InitialValueProblem(y0=y0, t_eval=t))
        output = output.ys[:, -1].view(*noise.shape)

        return output

