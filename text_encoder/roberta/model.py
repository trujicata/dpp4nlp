import os
from typing import Sequence, Dict


import torch
import torch.nn.functional as F
import numpy as np
from torch import Tensor
from transformers import AutoTokenizer, AutoModel, RobertaModel, RobertaTokenizer


class RoBERTaModel:
    def __init__(self, model_name: str = "roberta-base"):

        if not os.path.exists(
            f"text_encoder/roberta/weights/{model_name}.pt"
        ) or not os.path.exists(f"text_encoder/roberta/weights/{model_name}_tokenizer"):
            self.model = RobertaModel.from_pretrained(model_name)
            self.tokenizer = RobertaTokenizer.from_pretrained(model_name)

            os.makedirs("text_encoder/roberta/weights", exist_ok=True)

            torch.save(self.model, f"text_encoder/roberta/weights/{model_name}.pt")
            self.tokenizer.save_pretrained(
                f"text_encoder/roberta/weights/{model_name}_tokenizer"
            )
        else:
            self.model = torch.load(f"text_encoder/roberta/weights/{model_name}.pt")
            self.model.eval()
            self.tokenizer = RobertaTokenizer.from_pretrained(
                f"text_encoder/roberta/weights/{model_name}_tokenizer"
            )

    def encode(self, text: str, pooling: str = "cls"):
        tokenized_text = self.tokenizer(
            text, return_tensors="pt", padding=False, truncation=True
        )
        embeddings = self.model(**tokenized_text).last_hidden_state
        if pooling == "cls":
            pooled_output = embeddings[:, 0, :]
        elif pooling == "mean":
            pooled_output = embeddings.mean(dim=1)
        return pooled_output


class SFREmbedding:
    def __init__(self):
        self.model = AutoModel.from_pretrained("Salesforce/SFR-Embedding-Mistral")
        self.tokenizer = AutoTokenizer.from_pretrained(
            "Salesforce/SFR-Embedding-Mistral"
        )

        self.max_length = 4096

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


class MXBai:
    def __init__(self):
        self.model = AutoModel.from_pretrained("mixedbread-ai/mxbai-embed-large-v1")
        self.tokenizer = AutoTokenizer.from_pretrained(
            "mixedbread-ai/mxbai-embed-large-v1"
        )

        self.max_length = 4096

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

    def encode(self, texts: Sequence[str]):
        inputs = self.tokenizer(texts, padding=True, return_tensors="pt")
        outputs = self.model(**inputs).last_hidden_state
        embeddings = self.pooling(outputs, inputs, "cls")

        return F.normalize(embeddings, p=2, dim=1)
