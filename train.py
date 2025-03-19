import os
import torch
import torchvision
import torchvision.transforms.v2 as v2
from torch.utils.data import DataLoader
from torch import nn
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
from datetime import datetime
from models.fm import FlowMatching, display_generations

num_epochs = 80
batch_size = 512 
# lr = 3e-4 # Karpathy constant
lr = 1e-3 


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((32, 32), interpolation=v2.InterpolationMode.NEAREST),
        v2.Normalize(mean=(0.5,), std=(0.5,))
    ])
    trainset = torchvision.datasets.MNIST(root="data", download=True, train=True, transform=transform)
    train_loader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, num_workers=8)

    fm = FlowMatching().to(device)
    optim = torch.optim.Adam(fm.parameters(), lr=lr)
    scheduler=OneCycleLR(optim,lr,total_steps=num_epochs*len(train_loader),pct_start=0.1,anneal_strategy='cos')

    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M")
    exp_name = "condgenv2"
    base_exp_path = f"logging/{exp_name}/{timestamp}"
    vals_path = os.path.join(base_exp_path, "vals")
    ckpt_path = os.path.join(base_exp_path, "checkpoints")
    os.makedirs(base_exp_path, exist_ok=True)
    os.makedirs(vals_path, exist_ok=True)
    os.makedirs(ckpt_path, exist_ok=True)

    for epoch in range(num_epochs):
        fm.train()
        loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}")
        for _, (x, y) in loop:
            x = x.to(device)
            y = y.to(device)
            loss = fm((x, y))

            optim.zero_grad()
            loss.backward()
            optim.step()
            scheduler.step()

            loop.set_postfix(
                loss=loss.item(),
                lr=f"{scheduler.get_last_lr()[0]:.6e}"
            )
        loop.close()

        if epoch >= 10:
            with torch.inference_mode():
                fm.eval()
                display_generations(
                    fm, 
                    filepath=f"{vals_path}/epoch_{epoch}.png",
                    device=device
                )

        torch.save(fm.state_dict(), f"{ckpt_path}/last.ckpt")
        
if __name__ == "__main__":
    main()