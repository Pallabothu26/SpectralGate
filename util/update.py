import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import copy


# =========================
# Spectral Gate on DELTA (FIXED: top-k by magnitude)
# =========================
def spectral_gate_delta(delta_dict, ratio, skip_bn=True):
    """
    Applies FedSpectralGate on model update deltas.
    Keeps the top `ratio` fraction of FFT coefficients by magnitude.
    This correctly suppresses heterogeneity-driven high-frequency noise.
    """
    filtered = {}

    for name, param in delta_dict.items():

        if skip_bn and ("bias" in name or "bn" in name):
            filtered[name] = param
            continue

        # Cast to float32 for FFT stability
        p = param.float()

        # FFT
        w_fft = torch.fft.fftn(p)
        mag = torch.abs(w_fft)

        # Keep top `ratio` fraction by energy magnitude
        k = max(1, int(ratio * mag.numel()))
        threshold = torch.topk(mag.flatten(), k).values.min()
        mask = (mag >= threshold).float()

        # Apply mask and invert
        w_filtered = torch.fft.ifftn(w_fft * mask).real

        # Cast back to original dtype (e.g. float32 or bfloat16)
        filtered[name] = w_filtered.to(param.dtype)

    return filtered


# =========================
# Energy Logging (FIXED: same top-k logic as gate)
# =========================
def get_spectral_energy_delta(delta_dict, ratio):
    """
    Measures what fraction of total spectral energy is in the
    top-`ratio` coefficients (by magnitude) vs the rest.
    Uses the exact same masking logic as spectral_gate_delta.
    """
    low_e_sum = 0.0
    high_e_sum = 0.0
    count = 0

    for name, param in delta_dict.items():
        if "weight" not in name or "bn" in name:
            continue

        p = param.float()
        w_fft = torch.fft.fftn(p)
        mag = torch.abs(w_fft)
        total_energy = torch.sum(mag ** 2)

        if total_energy < 1e-10:
            continue  # skip dead layers (e.g. at round 0)

        k = max(1, int(ratio * mag.numel()))
        threshold = torch.topk(mag.flatten(), k).values.min()
        mask = (mag >= threshold).float()

        low_energy = torch.sum((mag ** 2) * mask)
        high_energy = total_energy - low_energy

        low_e_sum  += (low_energy  / total_energy).item()
        high_e_sum += (high_energy / total_energy).item()
        count += 1

    if count == 0:
        return 0.0, 0.0

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
        global_model = copy.deepcopy(net)

        net.train()
        optimizer = torch.optim.SGD(
            net.parameters(),
            lr=self.args.lr,
            momentum=self.args.momentum
        )

        epoch_loss = []

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

        # Compute delta
        local_weights  = net.state_dict()
        global_weights = global_model.state_dict()
        delta = {
            key: local_weights[key].float() - global_weights[key].float()
            for key in local_weights.keys()
        }

        # BEFORE logging
        if self.args.spectral_gate:
            low_pre, high_pre = get_spectral_energy_delta(delta, self.args.ratio)
            print(f"BEFORE [Client {client_idx}] lowE={low_pre:.3f}, highE={high_pre:.3f}")

        # Apply spectral gate
        if self.args.spectral_gate:
            delta = spectral_gate_delta(delta, ratio=self.args.ratio, skip_bn=self.args.skip_bn)

            # AFTER logging
            low_post, high_post = get_spectral_energy_delta(delta, self.args.ratio)
            print(f"AFTER  [Client {client_idx}] lowE={low_post:.3f}, highE={high_post:.3f}")

        return delta, sum(epoch_loss) / len(epoch_loss)


# =========================
# Global Testing
# =========================
def globaltest(net, test_dataset, args):
    net.eval()
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=100, shuffle=False
    )

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images  = images.to(args.device)
            labels  = labels.to(args.device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total   += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total
