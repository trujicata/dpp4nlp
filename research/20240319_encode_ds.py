# %%
import start
import torch

import pandas as pd

from text_encoder.models import RoBERTaModel
from torch.utils.tensorboard import SummaryWriter

# %%
ds = pd.read_csv("data/News_Category_Dataset_v3_50.csv")
ds.head()
# %%
roberta_model = RoBERTaModel()

# %%
from tqdm import tqdm

# Encode the short description column of the dataset and save it in a new column
embedding_list = []

for i, row in tqdm(ds.iterrows()):
    embedding = roberta_model.encode(row["short_description"])
    embedding = embedding.detach().numpy()
    embedding_list.append(embedding)

# %%
import numpy as np

embeddings_matrix = np.array(embedding_list)
embeddings_matrix = embeddings_matrix.squeeze(1)
print(embeddings_matrix.shape)
np.save(
    "/Users/dima/Documents/dpp4nlp/data/embeddings_tensor_50.npy", embeddings_matrix
)
# %%
# Visualize the embeddings with tensorboard
writer = SummaryWriter("runs/roberta_embeddings_50")
writer.add_embedding(embeddings_matrix, metadata=ds["category"])

# %%
