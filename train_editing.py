import os
import torch
import torchvision
import torchvision.transforms.v2 as v2
import numpy as np
from torch.utils.data import DataLoader
from torch import nn
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
from datetime import datetime
from models.fmfe import FlowMatchingForEditing
from transformers import AutoFeatureExtractor 

num_epochs = 80
batch_size = 512
lr = 3e-4 # Karpathy constant
# lr = 1e-3


class MNISTIncline(torchvision.datasets.MNIST):
    def __init__(
        self,
        root,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
        threshold=0.1,
    ):
        super().__init__(root, train, transform, target_transform, download)

        self.threshold = threshold

    def _calculate_slope_angle(self, img):
        img_numpy = img.numpy()
        yx_coords = np.column_stack(np.where(img_numpy > self.threshold))
        yx_mean = yx_coords.mean(axis=0)
        yx_coords_centered = yx_coords - yx_mean

        U, S, V = torch.pca_lowrank(
            torch.tensor(yx_coords_centered, dtype=torch.float32), q=1
        )
        main_component = V[:, 0].numpy()[::-1]
        if main_component[1] > 0:
            main_component *= -1

        slope_angle = torch.pi / 2 + np.atan2(main_component[1], main_component[0])
        
        # convert to degrees
        slope_angle = slope_angle * 180 / torch.pi

        return slope_angle

    def __getitem__(self, index):
        img, label = super().__getitem__(index)

        slope_angle = self._calculate_slope_angle(img)

        return {
            "latent": img, 
            "label": label, 
            "slope_angle": slope_angle,
        }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((32, 32), interpolation=v2.InterpolationMode.NEAREST),
            v2.Normalize(mean=(0.5,), std=(0.5,)),
        ]
    )
    trainset = MNISTIncline(
        root="data", download=True, train=True, transform=transform
    )
    train_loader = DataLoader(
        dataset=trainset, batch_size=batch_size, shuffle=True, num_workers=8
    )

    fm = FlowMatchingForEditing().to(device)
    optim = torch.optim.Adam(fm.parameters(), lr=lr)
    # scheduler = OneCycleLR(
    #     optim,
    #     lr,
    #     total_steps=num_epochs * len(train_loader),
    #     pct_start=0.1,
    #     anneal_strategy="cos",
    # )

    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M")
    exp_name = "editingv3"
    base_exp_path = f"logging/{exp_name}/{timestamp}"
    vals_path = os.path.join(base_exp_path, "vals")
    ckpt_path = os.path.join(base_exp_path, "checkpoints")
    os.makedirs(base_exp_path, exist_ok=True)
    os.makedirs(vals_path, exist_ok=True)
    os.makedirs(ckpt_path, exist_ok=True)

    for epoch in range(num_epochs):
        fm.train()
        loop = tqdm(
            train_loader, total=len(train_loader), desc=f"Epoch {epoch}"
        )
        for batch in loop:
            for key in batch:
                batch[key] = batch[key].to(device)
            
            loss = fm(batch)

            optim.zero_grad()
            loss.backward()
            optim.step()
            # scheduler.step()

            # loop.set_postfix(loss=loss.item(), lr=f"{scheduler.get_last_lr()[0]:.6e}")
            loop.set_postfix(loss=loss.item())
        loop.close()

        # if epoch >= 10:
        #     with torch.inference_mode():
        #         fm.eval()
        #         display_generations(
        #             fm, filepath=f"{vals_path}/epoch_{epoch}.png", device=device
        #         )

        torch.save(fm.state_dict(), f"{ckpt_path}/last.ckpt")


if __name__ == "__main__":
    main()
