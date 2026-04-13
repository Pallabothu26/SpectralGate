import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from util.dataset import get_dataset
from util.options import args_parser
import os

# Check if running on cluster (no display)
on_cluster = not os.environ.get('DISPLAY')

# Parse arguments (or hardcode for testing)
args = args_parser()
args.dataset = 'cifar10'  # or 'cifar100'
args.add_noise = True  # Set to True to visualize with noise
args.noise_std = 0.05  # Adjust as needed

# Load dataset
dataset_train, dataset_test, dict_users = get_dataset(args)

# Function to visualize images
def visualize_dataset(dataset, num_images=5, title="Dataset Visualization", save_path=None):
    fig, axes = plt.subplots(1, num_images, figsize=(15, 3))
    fig.suptitle(title)

    for i in range(num_images):
        image, label = dataset[i]
        # Denormalize for display (CIFAR-10 normalization)
        mean = torch.tensor([0.4914, 0.4822, 0.4465])
        std = torch.tensor([0.2023, 0.1994, 0.2010])
        image = image * std[:, None, None] + mean[:, None, None]
        image = torch.clamp(image, 0, 1)  # Ensure [0,1] for display

        axes[i].imshow(image.permute(1, 2, 0))  # CHW to HWC
        axes[i].set_title(f"Label: {label}")
        axes[i].axis('off')

    if save_path:
        plt.savefig(save_path)
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()

# Visualize training set
if on_cluster:
    visualize_dataset(dataset_train, title="Training Set (with noise if enabled)", save_path="train_visualization.png")
else:
    visualize_dataset(dataset_train, title="Training Set (with noise if enabled)")

# Visualize test set (always clean)
args.add_noise = False  # Disable noise for test set
dataset_train_clean, dataset_test, _ = get_dataset(args)
if on_cluster:
    visualize_dataset(dataset_test, title="Test Set (Clean)", save_path="test_visualization.png")
else:
    visualize_dataset(dataset_test, title="Test Set (Clean)")