from PIL import Image
import os
import numpy as np
import torch
from torchvision import datasets, transforms
from util.sampling import iid_sampling, non_iid_dirichlet_sampling
import torch.utils

def get_dataset(args):
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.dataset == 'cifar10':
        data_path = '../data/cifar10'
        args.num_classes = 10
        # Standard CIFAR-10 normalization to help the Spectral Gate focus on signal
        trans_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                 std=[0.2023, 0.1994, 0.2010])],
        )
        trans_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                 std=[0.2023, 0.1994, 0.2010])],
        )
        dataset_train = datasets.CIFAR10(data_path, train=True, download=True, transform=trans_train)
        dataset_test = datasets.CIFAR10(data_path, train=False, download=True, transform=trans_val)
        n_train = len(dataset_train)
        y_train = np.array(dataset_train.targets)

    elif args.dataset == 'cifar100':
        data_path = '../data/cifar100'
        args.num_classes = 100
        # CIFAR-100 requires slightly different normalization
        trans_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                 std=[0.2675, 0.2565, 0.2761])],
        )
        trans_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                 std=[0.2675, 0.2565, 0.2761])],
        )
        dataset_train = datasets.CIFAR100(data_path, train=True, download=True, transform=trans_train)
        dataset_test = datasets.CIFAR100(data_path, train=False, download=True, transform=trans_val)
        n_train = len(dataset_train)
        y_train = np.array(dataset_train.targets)

    else:
        exit('Error: unrecognized dataset. FedSpectralGate currently supports CIFAR-10/100.')

    # Logic to handle Data Heterogeneity (Injecting the noise source)
    if args.iid:
        print("Sampling Setting: IID")
        dict_users = iid_sampling(n_train, args.num_users, args.seed)
    else:
        print(f"Sampling Setting: Non-IID (Alpha: {args.alpha_dirichlet})")
        # Higher non_iid_prob_class and lower alpha_dirichlet = more spectral junk
        dict_users = non_iid_dirichlet_sampling(y_train, args.num_classes, args.non_iid_prob_class, 
                                               args.num_users, args.seed, args.alpha_dirichlet)

    return dataset_train, dataset_test, dict_users