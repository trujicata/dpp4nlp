# %%
import start
import torch

import pandas as pd

from text_encoder.roberta.model import RoBERTaModel
from torch.utils.tensorboard import SummaryWriter

# %%
ds = pd.read_csv("data/News_Category_Dataset_v3_50.csv")
ds.head()
# %%
roberta_model = RoBERTaModel()

# %%
# Encode the short description column of the dataset and save it in a new column
embedding_list = []

for i, row in ds.iterrows():
    embedding = roberta_model.encode(row["short_description"])
    embedding_list.append(embedding)

# %%
# Save the embeddings as a torch tensor
embeddings_tensor = torch.cat(embedding_list)
torch.save(embeddings_tensor, "data/embeddings_tensor_50.pt")

# %%
# Visualize the embeddings with tensorboard
writer = SummaryWriter("runs/roberta_embeddings_50")
writer.add_embedding(embeddings_tensor, metadata=ds["category"])

# %%
