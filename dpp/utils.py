import numpy as np
import pandas as pd

from matplotlib import pyplot as plt


def get_metrics(
    encoded_texts_df: pd.DataFrame, sampled_embeddings: pd.DataFrame
) -> None:
    """
    Get the metrics for the sampled embeddings

    Args:
    encoded_texts_df: pd.DataFrame, DataFrame containing the encoded texts
    sampled_embeddings: pd.DataFrame, DataFrame containing the sampled embeddings

    Returns:
    None
    """
    all_categories = encoded_texts_df["class"].unique()
    sampled_categories = sampled_embeddings["class"].unique()

    missing_categories = set(all_categories) - set(sampled_categories)
    if missing_categories:
        print("-----------------------------------------")
        print(f"{len(missing_categories)} Missing categories: {missing_categories}")
        print(
            f"Propotion of missing categories: {len(missing_categories) / len(all_categories)}"
        )
        print("-----------------------------------------")

    encoded_texts_df["class"].hist(bins=encoded_texts_df["class"].nunique())
    plt.xticks(rotation=90)
    plt.show()
    sampled_embeddings["class"].hist(bins=sampled_embeddings["class"].nunique())
    plt.xticks(rotation=90)
    plt.show()


def show_eigenvalues(eigenvalues: np.array) -> None:
    """
    Show the eigenvalues of the similarity matrix

    Args:
    eigenvalues: np.array, eigenvalues of the similarity matrix

    Returns:
    None
    """
    eigenvalues = eigenvalues[::-1]
    plt.plot(eigenvalues)
    plt.yscale("log")
    plt.title("Eigenvalues of the Similarity Matrix")
    plt.show()
