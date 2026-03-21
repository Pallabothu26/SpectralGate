import copy
import torch


def FedAvg_noniid(global_weights, deltas, dict_len):
    """
    FedSpectralGate Aggregation:
    
    Instead of averaging model weights, we:
    1. Average client DELTAS (updates)
    2. Add averaged delta to global model
    
    Args:
        global_weights: current global model (state_dict)
        deltas: list of client updates (delta dictionaries)
        dict_len: number of samples per client (for weighting)
    
    Returns:
        updated global model weights
    """

    # Deep copy global weights (we will update this)
    w_global = copy.deepcopy(global_weights)

    # Total samples across clients
    total_samples = sum(dict_len)

    # Initialize average delta
    avg_delta = {}

    # Iterate through each layer
    for k in w_global.keys():

        # Start with zero tensor of same shape
        avg_delta[k] = torch.zeros_like(w_global[k])

        # Weighted sum of deltas
        for i in range(len(deltas)):
            avg_delta[k] += deltas[i][k] * dict_len[i]

        # Normalize by total samples
        avg_delta[k] = avg_delta[k] / total_samples

    #  Apply update: global = global + avg_delta
    for k in w_global.keys():
        w_global[k] += avg_delta[k]

    return w_global