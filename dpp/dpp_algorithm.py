import numpy as np
from diffprivlib.mechanisms import Exponential

def similarity(u, v):
    dot_product_uv = np.dot(u, v)
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    
    similarity_uv = dot_product_uv / (norm_u * norm_v)
    return similarity_uv

def kernel_matrix(data):
    n = len(data)
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K[i, j] = similarity(data[i], data[j])
    return K

# TODO: this is relatively final, so Jack needs to check that DP is legit and we are done
def select_eigenvectors(eigenvalues, eigenvectors, n, dp=True):
    selected_eigenvalues = []
    selected_eigenvectors = []
    alpha = 1
    sigma = 1e-6
    if dp:
        eps = np.abs(np.log(n) - np.log(n - 1))        
    for eigenvalue, eigenvector in zip(eigenvalues, eigenvectors.T):
        if dp:
            # bin_exp actually decides only on the coin flip, the probability itself is unaffected
            bin_mech = Exponential(epsilon=2*eps, sensitivity=eps, utility=[0, eigenvalue])
            # now we actually calculate the prob
            prob = np.exp(alpha * np.log(eigenvalue + sigma)) / (np.exp(alpha * np.log(eigenvalue + sigma)) + 1)
            print(prob)
            prob = prob * bin_mech.randomise()
            if np.random.rand() < prob:
                selected_eigenvalues.append(eigenvalue)
                selected_eigenvectors.append(eigenvector)
        else:
            prob = eigenvalue / (eigenvalue + 1)
            if np.random.rand() < prob:
                selected_eigenvalues.append(eigenvalue)
                selected_eigenvectors.append(eigenvector)
    
    return np.array(selected_eigenvalues), np.array(selected_eigenvectors).T


def select_items(eigenvectors, items, idxs=True):
    subset_Y = []

    for eigenvector in eigenvectors.T:
        # Calculate squared magnitude of components
        squared_magnitude = np.square(np.abs(eigenvector))

        # Normalize squared magnitudes to probabilities
        # probabilities = squared_magnitude / np.sum(squared_magnitude)

        # Select an item based on probabilities
        selected_index = np.random.choice(len(eigenvector), p=squared_magnitude)
        if idxs:
            subset_Y.append(selected_index)
        else:
            selected_item = items[selected_index]
            subset_Y.append(selected_item)
    
    return subset_Y

def probability_of_selection(K, Y):
    # Extract submatrix Ky
    Ky = K[Y][:, Y]
    
    # Compute determinants
    det_Ky = np.linalg.det(Ky)
    det_K_plus_I = np.linalg.det(K + np.eye(K.shape[0]))  # Adding identity matrix
    
    # Compute probability of selection
    probability = det_Ky / det_K_plus_I
    
    return probability
