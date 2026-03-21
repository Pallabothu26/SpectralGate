import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    
    # ================= Federated Learning =================
    parser.add_argument('--rounds', type=int, default=100,
                        help="Total global training rounds")

    parser.add_argument('--local_ep', type=int, default=5,
                        help="Number of local epochs per client")

    parser.add_argument('--num_users', type=int, default=10,
                        help="Number of clients (K)")

    parser.add_argument('--frac', type=float, default=1.0,
                        help="Fraction of clients selected each round")

    parser.add_argument('--local_bs', type=int, default=64,
                        help="Local batch size")

    parser.add_argument('--lr', type=float, default=0.01,
                        help="Learning rate")

    parser.add_argument('--momentum', type=float, default=0.9,
                        help="SGD momentum")

    parser.add_argument('--beta', type=float, default=0,
                        help="FedProx coefficient (0 = FedAvg)")

    # ================= FedSpectralGate =================
    parser.add_argument('--spectral_gate', action='store_true',
                        help="Enable FedSpectralGate (default: OFF)")

    parser.add_argument('--ratio', type=float, default=0.2,
                        help="Top-k frequency ratio to KEEP (e.g., 0.2 = keep top 20%)")

    parser.add_argument('--low_cut', type=float, default=0.2,
                        help="Threshold to measure low-frequency energy")

    parser.add_argument('--skip_bn', action='store_true',
                        help="Skip BatchNorm and bias layers during spectral filtering")

    # ================= Model & Dataset =================
    parser.add_argument('--model', type=str, default='resnet18',
                        help="Model name")

    parser.add_argument('--dataset', type=str, default='cifar10',
                        help="Dataset: cifar10 or cifar100")

    parser.add_argument('--num_classes', type=int, default=10,
                        help="Number of classes")

    # ================= Non-IID Settings =================
    parser.add_argument('--iid', action='store_true',
                        help="Use IID data distribution")

    parser.add_argument('--non_iid_prob_class', type=float, default=0.9,
                        help="Label skew probability")

    parser.add_argument('--alpha_dirichlet', type=float, default=0.5,
                        help="Dirichlet distribution parameter")

    # ================= Legacy (Optional Cleanup Later) =================
    parser.add_argument('--iteration1', type=int, default=5)
    parser.add_argument('--rounds1', type=int, default=200)
    parser.add_argument('--rounds2', type=int, default=200)
    parser.add_argument('--frac1', type=float, default=0.01)
    parser.add_argument('--frac2', type=float, default=0.1)

    # ================= Reproducibility =================
    parser.add_argument('--seed', type=int, default=1,
                        help="Random seed")

    return parser.parse_args()