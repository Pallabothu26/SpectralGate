import copy
import torch

def FedAvg_noniid(w, dict_len):
    """
    Standard Weighted Federated Averaging.
    In FedSpectralGate, the 'w' list already contains the 
    spectrally-filtered weights from each client.
    """
    # Create a deep copy of the first client's weights as the base for averaging
    w_avg = copy.deepcopy(w[0])
    
    # Calculate the total number of samples across all participating clients
    total_samples = sum(dict_len)
    
    # Iterate through each layer/parameter in the model
    for k in w_avg.keys():
        # Multiply the first client's parameter by its weight (sample count)
        w_avg[k] = w_avg[k] * dict_len[0]
        
        # Add the weighted parameters from the remaining clients
        for i in range(1, len(w)):
            w_avg[k] += w[i][k] * dict_len[i]
            
        # Divide by total samples to get the weighted average
        # This results in the new Global Model weights
        w_avg[k] = torch.div(w_avg[k], total_samples)
        
    return w_avg