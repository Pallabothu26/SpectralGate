import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import copy


# =========================
# Low-Frequency Mask (NEW)
# =========================
def low_freq_mask(shape, ratio, device):
    """
    Softer low-frequency mask (keeps more useful signal)
    """
    mask = torch.zeros(shape, device=device)

    center = [s // 2 for s in shape]

    # 🔥 KEY FIX: larger radius scaling
    radius = [max(1, int(ratio * s)) for s in shape]

    slices = tuple(
        slice(max(0, c - r), min(s, c + r))
        for c, r, s in zip(center, radius, shape)
    )

    mask[slices] = 1
    return mask


# =========================
# Spectral Gate on DELTA (FIXED)
# =========================
def spectral_gate_delta(delta_dict, ratio, skip_bn=True):
    """
    Applies FedSpectralGate on model updates (delta).
    Uses true low-frequency masking instead of magnitude top-k.
    """
    filtered = {}

    for name, param in delta_dict.items():

        # Skip bias and BN layers
        if skip_bn and ("bias" in name or "bn" in name):
            filtered[name] = param
            continue

        # FFT
        w_fft = torch.fft.fftn(param)

        # Shift to center low frequencies
        w_fft_shifted = torch.fft.fftshift(w_fft)

        # Create mask
        mask = low_freq_mask(
            w_fft_shifted.shape,
            ratio,
            device=w_fft_shifted.device
        )

        # Apply mask
        w_fft_filtered = w_fft_shifted * mask

        # Shift back
        w_fft_unshifted = torch.fft.ifftshift(w_fft_filtered)

        # Inverse FFT
        w_filtered = torch.fft.ifftn(w_fft_unshifted).real

        filtered[name] = w_filtered

    return filtered


# =========================
# Energy Logging (IMPROVED)
# =========================
def get_spectral_energy_delta(delta_dict, low_cut=0.2):
    """
    Calculates low vs high frequency energy using centered spectrum.
    """
    low_e_sum = 0
    high_e_sum = 0
    count = 0

    for name, param in delta_dict.items():
        if "weight" in name and "bn" not in name:

            w_fft = torch.fft.fftn(param)
            w_fft_shifted = torch.fft.fftshift(w_fft)

            mag = torch.abs(w_fft_shifted)
            total_energy = torch.sum(mag ** 2)

            mask = low_freq_mask(
                mag.shape,
                low_cut,
                device=mag.device
            )

            low_energy = torch.sum((mag ** 2) * mask)
            high_energy = total_energy - low_energy + 1e-8

            low_e_sum += (low_energy / total_energy).item()
            high_e_sum += (high_energy / total_energy).item()
            count += 1

    if count == 0:
        return 0, 0

    return low_e_sum / count, high_e_sum / count


# =========================
# Dataset Split
# =========================
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


# =========================
# Local Training + Delta
# =========================
class LocalUpdate(object):
    def __init__(self, args, dataset, idxs):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train = DataLoader(
            DatasetSplit(dataset, idxs),
            batch_size=self.args.local_bs,
            shuffle=True
        )

    def update_weights(self, net, client_idx):
        """
        Performs local training and returns DELTA instead of full model.
        """
        global_model = copy.deepcopy(net)

        net.train()
        optimizer = torch.optim.SGD(
            net.parameters(),
            lr=self.args.lr,
            momentum=self.args.momentum
        )

        epoch_loss = []

        # =========================
        # Local Training
        # =========================
        for _ in range(self.args.local_ep):
            batch_loss = []

            for images, labels in self.ldr_train:
                images = images.to(self.args.device)
                labels = labels.to(self.args.device)

                net.zero_grad()
                outputs = net(images)
                loss = self.loss_func(outputs, labels)

                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        # =========================
        # Compute DELTA
        # =========================
        local_weights = net.state_dict()
        global_weights = global_model.state_dict()

        delta = {
            key: local_weights[key] - global_weights[key]
            for key in local_weights.keys()
        }

        # =========================
        # BEFORE Logging
        # =========================
        if self.args.spectral_gate:
            low_pre, high_pre = get_spectral_energy_delta(
                delta, self.args.low_cut
            )
            print(f"BEFORE [Client {client_idx}] lowE={low_pre:.3f}, highE={high_pre:.3f}")

        # =========================
        # Apply Spectral Gate
        # =========================
        if self.args.spectral_gate:
            delta = spectral_gate_delta(
                delta,
                ratio=self.args.ratio,
                skip_bn=self.args.skip_bn
            )

            # AFTER Logging
            low_post, high_post = get_spectral_energy_delta(
                delta, self.args.low_cut
            )
            print(f"AFTER  [Client {client_idx}] lowE={low_post:.3f}, highE={high_post:.3f}")

        return delta, sum(epoch_loss) / len(epoch_loss)


# =========================
# Global Testing
# =========================
def globaltest(net, test_dataset, args):
    net.eval()
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=100,
        shuffle=False
    )

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(args.device)
            labels = labels.to(args.device)

            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total