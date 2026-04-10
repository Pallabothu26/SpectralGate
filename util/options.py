import argparse
from html import parser

def args_parser():
    parser = argparse.ArgumentParser()

    # ================= Federated Learning =================
    parser.add_argument('--rounds', type=int, default=10,
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

    parser.add_argument('--ratio', type=float, default=0.8,
                        help="Top-k fraction of FFT coefficients to keep per layer (0.2 = keep top 20%% by magnitude)")

    parser.add_argument('--skip_bn', action='store_true',
                        help="Skip BatchNorm and bias layers during spectral filtering")

    # ================= Model & Dataset =================
    parser.add_argument('--model', type=str, default='resnet18',
                        help="Model name: resnet18 or resnet34")

    parser.add_argument('--dataset', type=str, default='cifar10',
                        help="Dataset: cifar10 or cifar100")

    parser.add_argument('--num_classes', type=int, default=10,
                        help="Number of output classes")

    # ================= Non-IID Settings =================
    parser.add_argument('--iid', action='store_true',
                        help="Use IID data distribution (default: Non-IID)")

    parser.add_argument('--non_iid_prob_class', type=float, default=0.9,
                        help="Label skew probability for non-IID sampling")

    parser.add_argument('--alpha_dirichlet', type=float, default=0.5,
                        help="Dirichlet concentration parameter (lower = more heterogeneous)")
    
    # ================= Noise Addition for Robustness Testing =================
    parser.add_argument('--add_noise', action='store_true',
                    help="Add Gaussian noise to client training data for robustness testing")

    parser.add_argument('--noise_std', type=float, default=0.05,
                    help="Standard deviation of Gaussian noise added to images (only if --add_noise is enabled)")


    # ================= Reproducibility =================
    parser.add_argument('--seed', type=int, default=1,
                        help="Random seed for reproducibility")

    return parser.parse_args()