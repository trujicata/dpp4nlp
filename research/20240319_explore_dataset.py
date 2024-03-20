# %%
import start  # noqa

import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from text_encoder.roberta.model import RoBERTaModel, SFREmbedding, MXBai

# %%
dataset_path = "../data/News_Category_Dataset_v3.json"

# Open json into a pandas dataframe
df = pd.read_json(dataset_path, lines=True)
df
# %%
category_counts = df.value_counts("category")
print(f"Number of categories: {len(category_counts)}")
print(f"5 most common categories:\n{category_counts.head(5)}")
print(f"5 least common categories:\n{category_counts.tail(5)}")
print(f"Total number of articles: {df.shape[0]}")
print(
    f"Average number of articles per category: {df.shape[0] / len(category_counts):.2f}"
)
# %%
roberta = RoBERTaModel()
roberta
# %%
sfr_model = SFREmbedding()
sfr_model.model
# %%
mxbai = MXBai()
mxbai
# %%
# %%
# Sample 1000 articles from the dataset
sample = df.sample(3000)
sample


# %%
def transform_query(query: str) -> str:
    return f"Identify the topic or theme of the given text: {query}"


# For each article, encode the headline and short_description
# using the RoBERTa model
encoded_texts = []
for i, row in tqdm(sample.iterrows(), total=sample.shape[0]):
    headline = row["headline"]
    short_description = row["short_description"]
    article_text = headline + ". " + short_description
    article_embedding = (
        mxbai.encode(transform_query(article_text)).detach().squeeze().cpu().numpy()
    )
    encoded_texts.append(
        {
            "category": row["category"],
            "headline": headline,
            "short_description": short_description,
            "article_embedding": article_embedding,
        }
    )

encoded_texts_df = pd.DataFrame(encoded_texts)
encoded_texts_df
# %%
# Save the encoded texts to a parquet file
encoded_texts_df.to_parquet("../data/encoded_texts.parquet")
# %%
# Load the encoded texts from the parquet file
encoded_texts_df = pd.read_parquet("../data/encoded_texts.parquet")
encoded_texts_df
# %%
article_embeddings = encoded_texts_df["article_embedding"].to_list()
article_embeddings = np.array(article_embeddings)
article_embeddings.shape
# %%
article_embeddings[0]


# %%
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    # return np.dot(a, b)


# Create similarity matrix
similarity_matrix = np.zeros((article_embeddings.shape[0], article_embeddings.shape[0]))
for i in tqdm(range(article_embeddings.shape[0])):
    for j in range(article_embeddings.shape[0]):
        # stop after the diagonal
        if j < i:
            similarity_matrix[i, j] = similarity_matrix[j, i]
            continue
        similarity_matrix[i, j] = cosine_similarity(
            article_embeddings[i], article_embeddings[j]
        )

similarity_matrix
# %%
# Find the most similar articles to the first article
most_similar_articles = np.argsort(similarity_matrix[10])[::-1]
# most_similar_articles = np.argsort(similarity_matrix[1])
encoded_texts_df.iloc[most_similar_articles[:5]]
# %%
similarity_matrix
# %%
# Save the similarity matrix to a npy file
np.save("../data/similarity_matrix.npy", similarity_matrix)
# %%
# %%
article_text
# %%
eigenvalues, eigenvectors = np.linalg.eig(similarity_matrix)

# %%
# Plot the eigenvalues
fig, ax = plt.subplots()
ax.plot(eigenvalues)
ax.set_yscale("log")
ax.set_ylabel("Eigenvalue")
ax.set_xlabel("Index")
plt.show()


# %%
eigenvalues
# %%

# For each eigenvalue, independently keep each eigenvector v_i with probability
# p_1 = eigenvalue_i / (eigenvalue_i + 1)

# Sample the eigenvectors
sampled_eigenvectors = []
sampled_eigenvalues = []
for i in range(eigenvectors.shape[0]):
    p = eigenvalues[i] / (eigenvalues[i] + 1)
    if np.random.rand() < p:
        sampled_eigenvectors.append(eigenvectors[i])
        sampled_eigenvalues.append(eigenvalues[i])

sampled_eigenvectors = np.array(sampled_eigenvectors)
sampled_eigenvalues = np.array(sampled_eigenvalues)

print(
    "Number of sampled eigenvectors: "
    f"{sampled_eigenvectors.shape[0]} / {eigenvectors.shape[0]}"
)
print(f"Sampled eigenvalues: {sampled_eigenvalues}")
# %%
# %%
# %%
# %%
Y = []
for eigenvector in sampled_eigenvectors:
    p = np.square(eigenvector)
    p = p / p.sum()
    # for j in range(eigenvector.shape[0]):
    #     if np.random.rand() <= p[j]:
    #         Y_i.append(j)
    ind = np.random.choice(eigenvector.shape[0], p=p)
    # Sample the largest index
    # ind = np.argmax(p)
    Y.append(ind)

# print(f"Number of items in Y: {sum(len(y) for y in Y)}")
# print(f"Number of unique items in Y: {len(set(y for y_i in Y for y in y_i))}")
Y
# %%
print(f"Number of categories in Y: {len(sample.iloc[Y].value_counts('category') )}")
sample.iloc[Y].value_counts("category")

# %%
# %%
# np.linalg.norm(eigenvectors, axis=1)
sampled_eigenvectors.shape

# %%
sampled_eigenvectors[0] ** 2
# %%


# %%
# %%
Y = []
for eigenvector in sampled_eigenvectors:
    # w = np.square(eigenvector)
    w = eigenvector / eigenvector.sum()
    mean_embedding = np.dot(article_embeddings.T, w)
    mean_embedding = mean_embedding / np.linalg.norm(mean_embedding)
    Y.append(mean_embedding)

Y = np.array(Y)
Y = Y + 1e-2 * np.random.randn(*Y.shape)
Y.shape
# %%
np.linalg.norm(Y, axis=1)
# %%
Y_self_similarity_matrix = np.zeros((Y.shape[0], Y.shape[0]))
for i in tqdm(range(Y.shape[0])):
    for j in range(Y.shape[0]):
        Y_self_similarity_matrix[i, j] = cosine_similarity(Y[i], Y[j])
# %%
(Y_self_similarity_matrix < 0.99).sum()
# %%

# Do similarity between the mean embedding and the article embeddings
# Find the most similar articles to the mean embedding
Y_similarity_matrix = np.zeros((Y.shape[0], article_embeddings.shape[0]))
for i in tqdm(range(Y.shape[0])):
    for j in range(article_embeddings.shape[0]):
        Y_similarity_matrix[i, j] = cosine_similarity(Y[i], article_embeddings[j])

Y_similarity_matrix.shape
# %%
indx = 0
num_similar = 5
most_similar_articles = np.argsort(Y_similarity_matrix[indx])[::-1]
print(f"Most similar articles to Y[{indx}]:")
print(f"----------")
print(f"With scores: {Y_similarity_matrix[indx, most_similar_articles[:num_similar]]}")
sample.iloc[most_similar_articles[:num_similar]]
# %%
# %%
print(
    f"Number of categories in Y_flat: {len(encoded_texts_df.iloc[Y].value_counts('category') )}"
)
encoded_texts_df.iloc[Y].value_counts("category")
# encoded_texts_df.iloc[Y[0]]
# %%
len(sample["category"].unique())
sample["category"].value_counts()
# %%
sampled_eigenvectors.shape
sampled_eigenvalues
# %%
eigenvector**2
# %%
# Plot p as a histogram
fig, ax = plt.subplots()
ax.hist(eigenvector**2, bins=50)
ax.set_xlabel("Probability")
ax.set_ylabel("Count")
plt.show()
# %%
# Plot p as bar plot
fig, ax = plt.subplots()
ax.bar(range(eigenvector.shape[0]), eigenvector**2)
# ax.bar(
#     range(sampled_eigenvector.shape[0]),
#     (sampled_eigenvector**2 / (sampled_eigenvector**2).sum()),
# )
ax.set_xlabel("Index")
ax.set_ylabel("Probability")
plt.show()
# %%
# Get index of 5 largest values of eigenvectors
largest_indices = np.argsort(eigenvector**2)[::-1][:5]
largest_indices
# %%
np.argsort(eigenvector**2)[::-1]
(eigenvector**2)[1705]
# %%
article_embeddings[2].mean()

# %%
(article_embeddings[2] ** 2).sum()
# %%
