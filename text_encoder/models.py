from typing import Sequence, Dict


import torch
import torch.nn.functional as F
import numpy as np
from torch import Tensor
from transformers import AutoTokenizer, AutoModel


class TextEncoder:
    def __init__(self, model_name: str, max_tokens: int = 512):
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_tokens = max_tokens
        self.model.eval()

    def encode(self, texts: Sequence[str], pooling: str = "cls"):
        raise NotImplementedError


class RoBERTaModel(TextEncoder):
    def __init__(self):
        super().__init__("roberta-base")

    def encode(self, texts: Sequence[str], pooling: str = "cls") -> Tensor:
        tokenized_text = self.tokenizer(
            texts, return_tensors="pt", padding=False, truncation=True
        )
        embeddings = self.model(**tokenized_text).last_hidden_state
        if pooling == "cls":
            pooled_output = embeddings[:, 0, :]
        elif pooling == "mean":
            pooled_output = embeddings.mean(dim=1)
        return pooled_output


class SFREmbedding(TextEncoder):
    def __init__(self):
        super().__init__("Salesforce/SFR-Embedding-Mistral", 4096)

    def last_token_pool(
        self, last_hidden_states: Tensor, attention_mask: Tensor
    ) -> Tensor:
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[
                torch.arange(batch_size, device=last_hidden_states.device),
                sequence_lengths,
            ]

    def encode(self, texts: Sequence[str], l2_normalize: bool = True):
        batch_dict = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        outputs = self.model(**batch_dict)
        embeddings = self.last_token_pool(
            outputs.last_hidden_state, batch_dict["attention_mask"]
        )

        if l2_normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)

        else:
            return embeddings


class MXBai(TextEncoder):
    def __init__(self):
        super().__init__("mixedbread-ai/mxbai-embed-large-v1", 4096)

    # The model works really well with cls pooling (default) but also with mean poolin.
    def pooling(
        self, outputs: torch.Tensor, inputs: Dict, strategy: str = "cls"
    ) -> np.ndarray:
        if strategy == "cls":
            outputs = outputs[:, 0]
        elif strategy == "mean":
            outputs = torch.sum(
                outputs * inputs["attention_mask"][:, :, None], dim=1
            ) / torch.sum(inputs["attention_mask"])
        else:
            raise NotImplementedError
        return outputs.detach().cpu()

    def encode(self, texts: Sequence[str], pooling: str = "cls") -> Tensor:
        texts = f"Identify the topic or theme of the given text: {texts}"

        inputs = self.tokenizer(texts, padding=True, return_tensors="pt")
        outputs = self.model(inputs["input_ids"]).last_hidden_state
        embeddings = self.pooling(outputs, inputs, pooling)

        return F.normalize(embeddings, p=2, dim=1)
