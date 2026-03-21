import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import copy


# =========================
# Spectral Gate on DELTA
# =========================
def spectral_gate_delta(delta_dict, ratio, skip_bn=True):
    """
    Applies FedSpectralGate on model updates (delta).

    Args:
        delta_dict: dict of model updates (local - global)
        ratio: fraction of frequencies to KEEP
        skip_bn: whether to skip bias and BN layers

    Returns:
        filtered delta_dict
    """
    filtered = {}

    for name, param in delta_dict.items():

        # Skip bias and BN layers
        if skip_bn and ("bias" in name or "bn" in name):
            filtered[name] = param
            continue

        # FFT (frequency domain)
        w_fft = torch.fft.fftn(param)

        # Magnitude (energy)
        mag = torch.abs(w_fft)
        flat_mag = mag.view(-1)

        # Top-k selection
        k = max(1, int(flat_mag.numel() * ratio))
        threshold = torch.topk(flat_mag, k).values[-1]

        mask = mag >= threshold

        # Apply mask
        w_fft_filtered = w_fft * mask

        # Inverse FFT
        w_filtered = torch.fft.ifftn(w_fft_filtered).real

        filtered[name] = w_filtered

    return filtered


# =========================
# Energy Logging (on DELTA)
# =========================
def get_spectral_energy_delta(delta_dict, low_cut=0.2):
    """
    Calculates low-frequency vs high-frequency energy for logging.
    """
    low_e_sum = 0
    high_e_sum = 0
    count = 0

    for name, param in delta_dict.items():
        if "weight" in name and "bn" not in name:

            w_fft = torch.fft.fftn(param)
            mag = torch.abs(w_fft).view(-1)

            sorted_mag, _ = torch.sort(mag, descending=True)
            split_idx = int(len(sorted_mag) * low_cut)

            low_e = torch.sum(sorted_mag[:split_idx])
            high_e = torch.sum(sorted_mag[split_idx:])
            total_e = low_e + high_e + 1e-8  # avoid divide by zero

            low_e_sum += (low_e / total_e).item()
            high_e_sum += (high_e / total_e).item()
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
        # 🔴 Save global model BEFORE training
        global_model = copy.deepcopy(net)

        net.train()
        optimizer = torch.optim.SGD(
            net.parameters(),
            lr=self.args.lr,
            momentum=self.args.momentum
        )

        epoch_loss = []

        # =========================
        # Local Training Loop
        # =========================
        for iter in range(self.args.local_ep):
            batch_loss = []

            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images = images.to(self.args.device)
                labels = labels.to(self.args.device)

                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)

                # Optional FedProx (not used now)
                if self.args.beta > 0:
                    pass

                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        # =========================
        # Compute DELTA (local - global)
        # =========================
        local_weights = net.state_dict()
        global_weights = global_model.state_dict()

        delta = {}
        for key in local_weights.keys():
            delta[key] = local_weights[key] - global_weights[key]

        # =========================
        # Log BEFORE spectral gating
        # =========================
        if self.args.spectral_gate:
            low_pre, high_pre = get_spectral_energy_delta(delta, self.args.low_cut)
            print(f"BEFORE [Client {client_idx}] lowE={low_pre:.3f}, highE={high_pre:.3f}")

        # =========================
        # Apply FedSpectralGate
        # =========================
        if self.args.spectral_gate:
            delta = spectral_gate_delta(
                delta,
                ratio=self.args.ratio,
                skip_bn=(self.args.skip_bn == 1)
            )

            # Log AFTER gating
            low_post, high_post = get_spectral_energy_delta(delta, self.args.low_cut)
            print(f"AFTER  [Client {client_idx}] lowE={low_post:.3f}, highE={high_post:.3f}")

        # =========================
        # Return DELTA (not model)
        # =========================
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