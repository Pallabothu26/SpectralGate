import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    
    # Federated Learning Arguments
    parser.add_argument('--rounds', type=int, default=100, help="rounds of training")
    parser.add_argument('--local_ep', type=int, default=5, help="number of local epochs")
    parser.add_argument('--num_users', type=int, default=10, help="number of users: K")
    parser.add_argument('--frac', type=float, default=1.0, help="fraction of selected clients")
    parser.add_argument('--local_bs', type=int, default=64, help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.9, help="SGD momentum")
    parser.add_argument('--beta', type=float, default=0, help="coefficient for local proximal (FedProx)")

    # --- NEW: FedSpectralGate Arguments ---
    parser.add_argument('--spectral_gate', type=int, default=1, help="1 to enable Spectral Gate, 0 to disable")
    parser.add_argument('--ratio', type=float, default=0.2, help="Ratio of top-frequency components to keep (e.g., 0.2 = keep top 20%)")
    parser.add_argument('--low_cut', type=float, default=0.2, help="Threshold percentage to define 'low frequency' for energy logging")
    parser.add_argument('--skip_bn', type=int, default=1, help="1 to skip Batch Normalization layers during spectral filtering")
    # ---------------------------------------

    # Dataset and Model Arguments
    parser.add_argument('--model', type=str, default='resnet18', help="model name")
    parser.add_argument('--dataset', type=str, default='cifar10', help="name of dataset: cifar10 or cifar100")
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    
    # Data Heterogeneity (Non-IID) Arguments
    parser.add_argument('--iid', action='store_true', help="Default is Non-IID. Use --iid for IID setting")
    parser.add_argument('--non_iid_prob_class', type=float, default=0.9, help="Probability for label skew (higher = more skewed)")
    parser.add_argument('--alpha_dirichlet', type=float, default=0.5, help="Dirichlet parameter for non-iid sampling")

    # Original Selective-FD arguments (Keeping for compatibility, though we may disable their logic later)
    parser.add_argument('--iteration1', type=int, default=5)
    parser.add_argument('--rounds1', type=int, default=200)
    parser.add_argument('--rounds2', type=int, default=200)
    parser.add_argument('--frac1', type=float, default=0.01)
    parser.add_argument('--frac2', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=1, help="random seed")

    return parser.parse_args()