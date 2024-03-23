from typing import List, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm
from dpp.dpp_algorithm import (
    kernel_matrix,
    select_eigenvectors,
    select_indexes,
)
from dpp.utils import get_metrics, show_eigenvalues
from text_encoder.models import TextEncoder


class DPP4NLP:
    def __init__(
        self,
        model: TextEncoder,
        dp: bool = True,
    ):
        self.model = model
        self.dp = dp

    def encode_texts(
        self, data: pd.DataFrame, sample: int = 1000, y: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Encode the texts in the dataset using the model
        """
        data = data.iloc[:sample]
        encoded_texts = []
        for i, row in tqdm(data.iterrows(), total=data.shape[0]):
            text = " ".join(row)

            embedding = self.model.encode(text).detach().squeeze().numpy()
            encoded_texts.append(
                {
                    "class": y[i] if y else None,
                    "text": text,
                    "embedding": embedding,
                }
            )
        return pd.DataFrame(encoded_texts)

    def get_representative_embeddings(
        self, data: pd.DataFrame, sample: int = 1000, y: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Get diverse and representative embeddings from the dataset using DPP

        Args:
        data: pd.DataFrame, dataset to sample from and encode using the model. Each
            row in the DataFrame should contain the text to encode.
        sample: int, number of samples to take from the dataset
        y: Optional[List[str]], list of labels for the samples

        Returns:
        pd.DataFrame: DataFrame containing the sampled embeddings and corresponding
            texts and labels
        """
        encoded_texts_df = self.encode_texts(data, sample, y)
        similarity_matrix = kernel_matrix(encoded_texts_df["embedding"].to_list())
        eigenvalues, eigenvectors = np.linalg.eigh(similarity_matrix)

        show_eigenvalues(eigenvalues)

        _, sampled_eigenvectors = select_eigenvectors(
            eigenvalues, eigenvectors, dp=self.dp
        )
        sampled_indexes = select_indexes(sampled_eigenvectors)
        sampled_embeddings = encoded_texts_df.iloc[sampled_indexes]

        get_metrics(encoded_texts_df, sampled_embeddings)
        return sampled_embeddings
