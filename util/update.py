import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np 
import copy

def apply_spectral_gate(model, ratio, skip_bn=True):
    """
    Applies the FedSpectralGate: FFT -> Mask (Top Ratio) -> IFFT.
    This 'kills' high-frequency noise in the weights.
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            # Skip Bias and Batch Normalization layers if skip_bn is True
            if skip_bn and ("bias" in name or "bn" in name or "weight" not in name):
                continue
            
            # 1. Transform weights to Frequency Domain (FFT)
            # Use rfftn for real-valued n-dimensional input
            w = param.data
            w_fft = torch.fft.rfftn(w)
            
            # 2. Calculate Magnitude/Energy
            mag = torch.abs(w_fft)
            
            # 3. Create the Gate (Top Ratio Mask)
            # Flatten to find the top k threshold
            flat_mag = mag.view(-1)
            k = int(flat_mag.numel() * ratio)
            if k < 1: k = 1
            
            # Find the value threshold that separates top 'ratio' from the rest
            threshold = torch.topk(flat_mag, k).values[-1]
            mask = mag >= threshold
            
            # 4. Apply the Gate (Zero out everything below threshold)
            w_fft_gated = w_fft * mask
            
            # 5. Transform back to Spatial Domain (IFFT)
            # Use irfftn to get real weights back
            param.data = torch.fft.irfftn(w_fft_gated, s=w.shape)
            
    return model

def get_spectral_energy(model, low_cut=0.2):
    """
    Helper to calculate Low-Energy vs High-Energy for logging.
    """
    low_e_sum = 0
    high_e_sum = 0
    count = 0
    
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "weight" in name and "bn" not in name:
                w_fft = torch.fft.rfftn(param.data)
                mag = torch.abs(w_fft).view(-1)
                
                # Sort by magnitude to define energy split
                sorted_mag, _ = torch.sort(mag, descending=True)
                split_idx = int(len(sorted_mag) * low_cut)
                
                low_e = torch.sum(sorted_mag[:split_idx])
                high_e = torch.sum(sorted_mag[split_idx:])
                total_e = low_e + high_e
                
                low_e_sum += (low_e / total_e).item()
                high_e_sum += (high_e / total_e).item()
                count += 1
                
    return low_e_sum/count, high_e_sum/count

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

class LocalUpdate(object):
    def __init__(self, args, dataset, idxs):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

    def update_weights(self, net, client_idx):
        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        # Log energy BEFORE gating
        low_pre, high_pre = get_spectral_energy(net, self.args.low_cut)
        print(f"BEFORE [Client {client_idx}] lowE={low_pre:.3f}, highE={high_pre:.3f}")

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                
                # FedProx logic (if beta > 0)
                if self.args.beta > 0:
                    # Proximal term logic would go here if needed
                    pass

                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        
        # --- Apply FedSpectralGate AFTER Local Training ---
        if self.args.spectral_gate:
            net = apply_spectral_gate(net, ratio=self.args.ratio, skip_bn=(self.args.skip_bn == 1))
            
            # Log energy AFTER gating
            low_post, high_post = get_spectral_energy(net, self.args.low_cut)
            print(f"AFTER  [Client {client_idx}] lowE={low_post:.3f}, highE={high_post:.3f}")

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

def globaltest(net, test_dataset, args):
    net.eval()
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=100, shuffle=False)
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(args.device), labels.to(args.device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total