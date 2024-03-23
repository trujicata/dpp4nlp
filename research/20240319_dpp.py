# %%
import start

import numpy as np
from dpp.dpp_algorithm import *

# %%
# Load the embeddings
embs = np.load("data/roberta_embeddings_50.npy")
embs.shape
# %%
K = kernel_matrix(embs)
eigenvalues, eigenvectors = np.linalg.eig(K)
selected_val, selected_vec = select_eigenvectors(eigenvalues, eigenvectors)
final_vecs = select_items(selected_vec, embs)

# %%
len(final_vecs)

# %%
