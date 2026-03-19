import numpy as np

def iid_sampling(n_train, num_users, seed):
    """
    Uniform distribution of data across clients.
    Useful as a baseline to show that SpectralGate doesn't hurt 
    performance even when there is no noise.
    """
    np.random.seed(seed)
    num_items = int(n_train/num_users)
    dict_users, all_idxs = {}, [i for i in range(n_train)]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs)-dict_users[i])
    return dict_users

def non_iid_dirichlet_sampling(y_train, num_classes, p, num_users, seed, alpha_dirichlet=0.5):
    """
    The heart of heterogeneity for FedSpectralGate.
    Using a low alpha (e.g., 0.5) creates sharp label skew.
    """
    np.random.seed(seed)
    
    # 1. Determine which classes each client 'prefers' based on probability p
    Phi = np.random.binomial(1, p, size=(num_users, num_classes))
    n_classes_per_client = np.sum(Phi, axis=1)
    
    # Ensure no client is empty
    while np.min(n_classes_per_client) == 0:
        invalid_idx = np.where(n_classes_per_client==0)[0]
        Phi[invalid_idx] = np.random.binomial(1, p, size=(len(invalid_idx), num_classes))
        n_classes_per_client = np.sum(Phi, axis=1)
        
    # 2. Map classes to clients
    Psi = [list(np.where(Phi[:, j]==1)[0]) for j in range(num_classes)]
    num_clients_per_class = np.array([len(x) for x in Psi])
    
    dict_users = {i: set() for i in range(num_users)}
    
    # 3. Use Dirichlet distribution to distribute class samples among interested clients
    for class_i in range(num_classes):
        all_idxs = np.where(y_train == class_i)[0]
        np.random.shuffle(all_idxs)
        
        if num_clients_per_class[class_i] > 0:
            # Low alpha_dirichlet = higher imbalance
            p_dirichlet = np.random.dirichlet([alpha_dirichlet] * num_clients_per_class[class_i])
            
            # Split indices based on Dirichlet proportions
            proportions = (p_dirichlet * len(all_idxs)).astype(int)
            # Adjust for rounding errors to ensure all samples are used
            proportions[-1] += len(all_idxs) - proportions.sum()
            
            start = 0
            for i, client_k in enumerate(Psi[class_i]):
                end = start + proportions[i]
                dict_users[client_k].update(all_idxs[start:end])
                start = end
                
    return dict_users