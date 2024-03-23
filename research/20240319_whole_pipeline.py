# %%
import start

import pandas as pd
import numpy as np
from dpp.dpp_algorithm import *


from text_encoder.models import RoBERTaModel
from torch.utils.tensorboard import SummaryWriter

# %%
ds = pd.read_csv("data/News_Category_Dataset_v3.csv")
ds.head()

# %%
# Now, create a new dataset with less rows
ds_toy = ds.head(1000)
ds_toy.head()
# %%
roberta_model = RoBERTaModel()

# %%
# Encode the short description column of the dataset and save it in a new column
embedding_list = []

for i, row in ds_toy.iterrows():
    embedding = roberta_model.encode(row["short_description"])
    # To numpy
    embedding = embedding.detach().numpy()
    embedding_list.append(embedding)

# %%
# Save the embeddings as a torch tensor
emb_matrix = np.array(embedding_list)
emb_matrix = emb_matrix.squeeze()
emb_matrix.shape
# %%
# save the embeddings
np.save("data/roberta_embeddings_toy.npy", emb_matrix)
# %%
# Visualize the embeddings with tensorboard
writer = SummaryWriter("runs/roberta_embeddings_toy")
writer.add_embedding(emb_matrix, metadata=ds_toy["category"])
# %%
K = kernel_matrix(emb_matrix)
eigenvalues, eigenvectors = np.linalg.eig(K)
selected_val, selected_vec = select_eigenvectors(eigenvalues, eigenvectors)
final_vecs = select_items(selected_vec, emb_matrix)

# %%
len(final_vecs)

# %%
