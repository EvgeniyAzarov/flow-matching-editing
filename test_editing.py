# %%
import torch
import matplotlib.pyplot as plt
import numpy as np
from models.fmfe import FlowMatchingForEditing
from models.fm import FlowMatching

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"

def save_images(images, filepath):
    n_samples = images.shape[0]
    fig, axes = plt.subplots(1, n_samples, figsize=(n_samples * 2, 2))
    axs = axes.flatten()

    for i in range(n_samples):
        axs[i].imshow(images[i].detach().cpu().numpy()[0], cmap="grey")

    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()


# %%
fmfe = FlowMatchingForEditing()
fm = FlowMatching()

# %%
state_dict = torch.load("logging/editingv2/2025-03-18-1632/checkpoints/last.ckpt")
fmfe.load_state_dict(state_dict)
fmfe.to(device)
fmfe.eval()

# %%
state_dict = torch.load("logging/condgenv2/2025-03-13-2345/checkpoints/last.ckpt")
fm.load_state_dict(state_dict)
fm.to(device)
fm.eval()

batch_size = 5

def generate_batch(model, device, classes=None):
    if classes is None:
        classes = np.random.randint(0, 10, batch_size)

    images = model.generate(
        batch_size, 
        classes=classes, 
        device=device
    )

    return images, classes


classes = np.random.randint(0, 10, batch_size)
images, classes = generate_batch(fm, device, classes)
save_images(images, "out/img.png")
while True:
    action = input("[r or slope_angle] ")

    if action == "r":
        images, classes = generate_batch(fm, device, classes)
        save_images(images, "out/img.png")
        continue

    slope_angles = [float(action)] * batch_size
    images = fmfe.generate(images, classes, slope_angles, device=device)
    save_images(images, "out/img.png")
        
