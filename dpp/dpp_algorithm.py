from typing import List

import numpy as np
from diffprivlib.mechanisms import Exponential


def similarity(u: np.array, v: np.array) -> float:
    """
    Cosine similarity between two vectors u and v

    Args:
    u: np.array (n,)
    v: np.array (n,)

    Returns:
    float: Cosine similarity between u and v
    """
    dot_product_uv = np.dot(u, v)
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)

    similarity_uv = dot_product_uv / (norm_u * norm_v)
    return similarity_uv


def kernel_matrix(data: np.array) -> np.array:
    """
    Computes the kernel matrix K for a given dataset,
    where K[i, j] = similarity(data[i], data[j])

    Args:
    data: np.array (n, d), where n is the number of data points
    and d is the dimension of each data point

    Returns:
    np.array (n, n): Kernel matrix K
    """
    n = len(data)
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if j < i:
                K[i, j] = K[j, i]
            else:
                K[i, j] = similarity(data[i], data[j])
            K[i, j] = similarity(data[i], data[j])
    return K


def select_eigenvectors(
    eigenvalues: np.array, eigenvectors: np.array, dp=True
) -> np.array:
    """
    Samples eigenvectors based on eigenvalues.
    If do is False, the probability of selecting an eigenvector is proportional to its
    eigenvalue.
    If dp is True, the selection is differentially private and the probability of selecting an eigenvector is
    proportional to the exponential of its eigenvalue.

    Args:
    eigenvalues: np.array (n,)
    eigenvectors: np.array (n, n)
    dp: bool, whether to use differential privacy

    Returns:
    np.array (n,): Selected eigenvalues
    """
    n = len(eigenvalues)

    selected_eigenvalues = []
    selected_eigenvectors = []
    alpha = 1
    sigma = 1e-6
    for eigenvalue, eigenvector in zip(eigenvalues, eigenvectors.T):
        if dp:
            eps = np.abs(np.log(n) - np.log(n - 1))
            # bin_exp actually decides only on the coin flip, the probability itself is unaffected
            bin_mech = Exponential(
                epsilon=2 * eps, sensitivity=eps, utility=[0, eigenvalue]
            )
            # now we actually calculate the prob
            prob = np.exp(alpha * np.log(eigenvalue + sigma)) / (
                np.exp(alpha * np.log(eigenvalue + sigma)) + 1
            )
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


def select_indexes(eigenvectors: np.array) -> List[int]:
    """
    Selects indexes of embeddings based on the eigenvectors of the similarity matrix.
    For each eigenvector, it samples one index with a probability proportional
    to the squared magnitude of the eigenvector components.

    Args:
    eigenvectors: np.array (n, n), eigenvectors of the similarity matrix

    Returns:
    List[int]: List of indexes of the selected embeddings
    """
    subset_Y = []

    for eigenvector in eigenvectors.T:
        # Calculate squared magnitude of components
        squared_magnitude = np.square(np.abs(eigenvector))

        # Normalize squared magnitudes to probabilities
        # probabilities = squared_magnitude / np.sum(squared_magnitude)

        # Select an item based on probabilities
        selected_index = np.random.choice(len(eigenvector), p=squared_magnitude)
        subset_Y.append(selected_index)

    return subset_Y


def sample_embeddings(embedding_matrix: np.array, eigenvectors: np.array) -> np.array:
    """
    Samples embeddings from the embedding matrix based on the subset of indexes.

    Args:
    embedding_matrix: np.array (n, d), where n is the number of embeddings
    and d is the dimension of each embedding

    Returns:
    np.array (m, d): Sampled embeddings
    """
    subset_Y = select_indexes(eigenvectors)
    return embedding_matrix[subset_Y]


def probability_of_selection(K: np.array, Y: List[int]) -> float:
    """
    Computes the probability of selecting a subset Y of embeddings based on the kernel matrix K.
    The probability is proportional to the determinant of the submatrix Ky and the determinant of K + I,
    where Ky is the submatrix of K corresponding to the selected embeddings.

    Args:
    K: np.array (n, n), kernel matrix
    Y: List[int], list of indexes of the selected embeddings

    Returns:
    float: Probability of selecting the subset Y
    """
    # Extract submatrix Ky
    Ky = K[Y][:, Y]

    # Compute determinants
    det_Ky = np.linalg.det(Ky)
    det_K_plus_I = np.linalg.det(K + np.eye(K.shape[0]))  # Adding identity matrix

    # Compute probability of selection
    probability = det_Ky / det_K_plus_I

    return probability
