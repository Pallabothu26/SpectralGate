import os
import copy
import numpy as np
import random
import torch

from util.options import args_parser
from util.update import LocalUpdate, globaltest
from util.fedavg import FedAvg_noniid
from util.dataset import get_dataset
from model.build_model import build_model

if __name__ == '__main__':
    # 1. Setup and Arguments
    args = args_parser()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Deterministic behavior for research reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # 2. Prepare Data and Model
    # This uses our updated dataset logic for CIFAR-10/100
    dataset_train, dataset_test, dict_users = get_dataset(args)
    netglob = build_model(args)
    netglob.to(args.device)
    
    # 3. Logging Setup
    if not os.path.exists("./results/"):
        os.makedirs("./results/")
    
    log_name = f"./results/{args.dataset}_{args.model}_ratio{args.ratio}_alpha{args.alpha_dirichlet}_gate{args.spectral_gate}.txt"
    f_acc = open(log_name, 'a')
    f_acc.write(f"Starting FedSpectralGate: Ratio={args.ratio}, Gate={args.spectral_gate}\n")

    # 4. Federated Training Loop
    m = max(int(args.frac * args.num_users), 1)
    
    print(f"--- Starting Training on {args.dataset} with {args.model} ---")
    if args.spectral_gate:
        print(f"--- FedSpectralGate ACTIVE (Ratio: {args.ratio}) ---")
    else:
        print("--- Baseline FedAvg (Spectral Gate OFF) ---")

    for rnd in range(args.rounds):
        w_locals, loss_locals = [], []
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        
        for idx in idxs_users:
            # Create a local trainer for this client
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            
            # Local training + Spectral Gating (inside update_weights)
            # We pass a clean copy of the global model to the client
            w_local, loss_local = local.update_weights(
                net=copy.deepcopy(netglob).to(args.device), 
                client_idx=idx
            )
            
            w_locals.append(copy.deepcopy(w_local))
            loss_locals.append(loss_local)
        
        # 5. Server-Side Aggregation
        dict_len = [len(dict_users[idx]) for idx in idxs_users]
        w_glob = FedAvg_noniid(w_locals, dict_len)
        
        # Update Global Model
        netglob.load_state_dict(w_glob)

        # 6. Evaluation
        acc_test = globaltest(netglob, dataset_test, args)
        avg_loss = sum(loss_locals) / len(loss_locals)
        
        print(f'Round {rnd:3d}, Avg Loss: {avg_loss:.3f}, Global Test Acc: {acc_test:.4f}')
        f_acc.write(f"Round {rnd}, Acc: {acc_test:.4f}, Loss: {avg_loss:.4f}\n")
        f_acc.flush()

    f_acc.close()
    torch.cuda.empty_cache()
    print("Training Complete. Results saved to results/ folder.")