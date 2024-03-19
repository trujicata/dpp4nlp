# %%
import start
import json
import pandas as pd

# %%
# Load the dataset
json_path = "data/News_Category_Dataset_v3.json"
with open(json_path, "r") as f:
    data = [json.loads(line) for line in f]
data
# %%
df = pd.DataFrame(data)
# Reduce the dataset to the first 1000 rows
df = df.head(10)
df.head()
# %%
df.info()
# %%
from transformers import RobertaConfig, RobertaModel

# Initializing a RoBERTa configuration
configuration = RobertaConfig()

# Initializing a model (with random weights) from the configuration
model = RobertaModel(configuration)

# Accessing the model configuration
configuration = model.config
# %%
# Run the model through the news description
from transformers import RobertaTokenizer

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
df["short_description"] = df["short_description"].fillna("")
inputs = tokenizer(df["short_description"].tolist(), return_tensors="pt", padding=True, truncation=True)
outputs = model(**inputs)
outputs

#######################################################
# %%
import start
import json
import pandas as pd

# %%
# Load the dataset
json_path = "data/News_Category_Dataset_v3.json"
with open(json_path, "r") as f:
    data = [json.loads(line) for line in f]
data
# %%
df = pd.DataFrame(data)
# Reduce the dataset to the first 1000 rows
df = df.head(100)
df.head()
# %%
from text_encoder.roberta.model import RoBERTaModel

roberta = RoBERTaModel()
ids = df.index
descp = df["short_description"].tolist()
data = {}
for i, d in zip(ids, descp):
    data[i] = roberta.encode(d)  

# %%
import numpy as np
import hdbscan
import matplotlib.pyplot as plt

# Extract embeddings from dictionary
embeddings = np.array([data[i].detach().numpy() for i in ids])
embeddings = np.squeeze(embeddings, axis=1)

# Perform clustering using HDBSCAN
clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=3)
cluster_labels = clusterer.fit_predict(embeddings)

# Plot the clusters
plt.figure(figsize=(8, 6))
for cluster_label in set(cluster_labels):
    cluster_mask = (cluster_labels == cluster_label)
    plt.scatter(embeddings[cluster_mask, 0], embeddings[cluster_mask, 1], label=f'Cluster {cluster_label}')

plt.title('HDBSCAN Clustering')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.legend()
plt.show()




# %%
