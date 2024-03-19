import numpy as np

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

def select_eigenvectors(eigenvalues, eigenvectors):
    selected_eigenvalues = []
    selected_eigenvectors = []

    for eigenvalue, eigenvector in zip(eigenvalues, eigenvectors.T):
        prob = eigenvalue / (eigenvalue + 1)
        if np.random.rand() < prob:
            selected_eigenvalues.append(eigenvalue)
            selected_eigenvectors.append(eigenvector)
    
    return np.array(selected_eigenvalues), np.array(selected_eigenvectors).T


def select_items(eigenvectors, items):
    subset_Y = []

    for eigenvector in eigenvectors.T:
        # Calculate squared magnitude of components
        squared_magnitude = np.square(np.abs(eigenvector))

        # Normalize squared magnitudes to probabilities
        probabilities = squared_magnitude / np.sum(squared_magnitude)

        # Select an item based on probabilities
        selected_index = np.random.choice(len(eigenvector), p=probabilities)
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
